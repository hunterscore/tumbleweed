"""Evaluation loop: runs a model over a labeled split and reports the SPEC
pose error metric (mean translation error, mean rotation error, mean score).
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from tumbleweed.data.camera import load_camera
from tumbleweed.data.speedplus import SpeedPlusDataset
from tumbleweed.data.transforms import build_keypoint_transforms, build_transforms
from tumbleweed.metrics import distribution_stats, flip_rate, speed_score
from tumbleweed.models.keypoint_resnet import KeypointPoseEstimator, KeypointResNet18
from tumbleweed.models.pose_resnet import PoseResNet18

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_PER_SAMPLE_HEADER = (
    ["filename", "translation_error", "rotation_error_deg", "speed_score", "gt_range_m"]
    + [f"t_gt_{c}" for c in "xyz"]
    + [f"t_pred_{c}" for c in "xyz"]
    + [f"q_gt_{c}" for c in "wxyz"]
    + [f"q_pred_{c}" for c in "wxyz"]
)


def _write_error_csv(
    path: str,
    metrics: dict,
    per_sample: bool,
    filenames: list,
    t_err: torch.Tensor,
    q_err: torch.Tensor,
    score: torch.Tensor,
    gt_range: Optional[torch.Tensor],
    t_gt: Optional[torch.Tensor] = None,
    t_pred: Optional[torch.Tensor] = None,
    q_gt: Optional[torch.Tensor] = None,
    q_pred: Optional[torch.Tensor] = None,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        if per_sample:
            writer.writerow(_PER_SAMPLE_HEADER)
            for fname, te, qe, s, r, tg, tp, qg, qp in zip(
                filenames,
                t_err.tolist(),
                q_err.tolist(),
                score.tolist(),
                gt_range.tolist(),
                t_gt.tolist(),
                t_pred.tolist(),
                q_gt.tolist(),
                q_pred.tolist(),
            ):
                writer.writerow([fname, te, qe, s, r, *tg, *tp, *qg, *qp])
        else:
            writer.writerow(list(metrics.keys()))
            writer.writerow(list(metrics.values()))

    logger.info("wrote %s error CSV to %s", "per-sample" if per_sample else "summary", out_path)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    error_csv_path: Optional[str] = None,
    per_sample: bool = False,
) -> dict:
    """Run `model` over `dataloader` and return aggregate SPEC metrics.

    Requires the dataloader's dataset to carry pose labels
    (``SpeedPlusDataset.has_pose``). If ``error_csv_path`` is given, also
    writes either a one-row summary or (if ``per_sample``) a per-sample
    breakdown (filename, translation/rotation error, GT range) to that path
    - the per-sample form is the raw material for range-dependent error
    analysis and for calibrating the closed-loop noise model later.
    """
    model.eval()

    all_t_err, all_q_err, all_score = [], [], []
    all_range, all_filenames = [], []
    all_t_gt, all_t_pred, all_q_gt, all_q_pred = [], [], [], []
    for batch in dataloader:
        images = batch["image"].to(device)
        t_gt = batch["translation"].to(device)
        q_gt = batch["quaternion"].to(device)

        t_pred, q_pred = model(images)
        result = speed_score(t_pred, q_pred, t_gt, q_gt)

        all_t_err.append(result["translation_error"].cpu())
        all_q_err.append(result["rotation_error_deg"].cpu())
        all_score.append(result["score"].cpu())
        if per_sample:
            all_range.append(t_gt.norm(dim=-1).cpu())
            all_filenames.extend(batch["filename"])
            all_t_gt.append(t_gt.cpu())
            all_t_pred.append(t_pred.cpu())
            all_q_gt.append(q_gt.cpu())
            all_q_pred.append(q_pred.cpu())

    t_err = torch.cat(all_t_err)
    q_err = torch.cat(all_q_err)
    score = torch.cat(all_score)
    gt_range = torch.cat(all_range) if per_sample else None

    metrics = {
        "num_samples": len(score),
        **distribution_stats("translation_error", t_err),
        **distribution_stats("rotation_error_deg", q_err),
        "flip_rate_90deg": flip_rate(q_err, threshold_deg=90.0),
        "mean_speed_score": score.mean().item(),
    }

    if error_csv_path is not None:
        _write_error_csv(
            error_csv_path,
            metrics,
            per_sample,
            all_filenames,
            t_err,
            q_err,
            score,
            gt_range,
            t_gt=torch.cat(all_t_gt) if per_sample else None,
            t_pred=torch.cat(all_t_pred) if per_sample else None,
            q_gt=torch.cat(all_q_gt) if per_sample else None,
            q_pred=torch.cat(all_q_pred) if per_sample else None,
        )

    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a tumbleweed pose checkpoint")
    parser.add_argument("--data-root", type=str, default="data/speedplusv2")
    parser.add_argument("--domain", type=str, default="synthetic")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="regression",
        choices=["regression", "keypoint"],
        help="architecture the checkpoint belongs to",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--keypoint-image-size",
        type=int,
        nargs=2,
        default=[384, 640],
        help="input size for --model keypoint; must match training",
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--error-csv",
        type=str,
        default=None,
        help="path to save error metrics; defaults to results/<checkpoint-stem>_<domain>-<split>.csv",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="dump per-sample errors (filename, translation/rotation error, GT range) instead of just the summary",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prior-checkpoint", type=str, default=None, help="regression checkpoint as a pose prior for --model keypoint")
    return parser


def build_model(args: argparse.Namespace):
    """Load the checkpoint into whichever architecture produced it.

    Both return ``(translation, quaternion)`` from ``forward``, so everything
    downstream of here is shared.
    """
    state = torch.load(args.checkpoint, map_location=args.device)
    weights = state["model"] if "model" in state else state

    prior_model = None
    
    if args.model == "regression":
        model = PoseResNet18(pretrained=False)
        model.load_state_dict(weights)
        return model.to(args.device), build_transforms(image_size=args.image_size, train=False)

    # load trained regression prior
    if args.prior_checkpoint is not None:
        prior_state = torch.load(args.prior_checkpoint, map_location=args.device)
        prior_weights = prior_state["model"] if "model" in prior_state else prior_state
        prior_model = PoseResNet18(pretrained=False)
        prior_model.load_state_dict(prior_weights)


    camera = load_camera(Path(args.data_root) / "camera.json")
    network = KeypointResNet18(pretrained=False)
    network.load_state_dict(weights)
    image_size = tuple(args.keypoint_image_size)
    estimator = KeypointPoseEstimator(
        network, camera, image_size=image_size, stride=args.stride, prior_model=prior_model
    ).to(args.device)
    transform = build_keypoint_transforms(camera, image_size=image_size, stride=args.stride)
    return estimator, transform


def main():
    args = build_arg_parser().parse_args()

    model, transform = build_model(args)

    dataset = SpeedPlusDataset(
        root=args.data_root,
        domain=args.domain,
        split=args.split,
        transform=transform,
    )
    if not dataset.has_pose:
        raise ValueError(
            f"{args.domain}/{args.split} has no pose labels; SPEC scoring "
            "requires a labeled split (e.g. synthetic/validation)."
        )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    error_csv_path = args.error_csv
    if error_csv_path is None:
        checkpoint_stem = Path(args.checkpoint).stem
        error_csv_path = f"results/{checkpoint_stem}_{args.domain}-{args.split}.csv"

    metrics = evaluate(
        model, dataloader, args.device, error_csv_path=error_csv_path, per_sample=args.advanced
    )
    if isinstance(model, KeypointPoseEstimator):
        logger.info("pnp_failure_rate=%.4f", model.failure_rate)
        if model.prior_model is not None:
            logger.info("prior_rejection_rate=%.4f", model.rejection_rate)
    logger.info(
        "eval: n=%d  mean_t_err=%.4f  mean_q_err_deg=%.4f  median_q_err_deg=%.4f  "
        "flip_rate_90deg=%.4f  mean_speed_score=%.4f",
        metrics["num_samples"],
        metrics["mean_translation_error"],
        metrics["mean_rotation_error_deg"],
        metrics["median_rotation_error_deg"],
        metrics["flip_rate_90deg"],
        metrics["mean_speed_score"],
    )


if __name__ == "__main__":
    main()
