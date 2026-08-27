"""Evaluation loop: runs a model over a labeled split and reports the SPEC
pose error metric (mean translation error, mean rotation error, mean score).
"""
from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from tumbleweed.data.speedplus import SpeedPlusDataset
from tumbleweed.data.transforms import build_transforms
from tumbleweed.metrics import speed_score
from tumbleweed.models.pose_resnet import PoseResNet18

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: str) -> dict:
    """Run `model` over `dataloader` and return aggregate SPEC metrics.

    Requires the dataloader's dataset to carry pose labels
    (``SpeedPlusDataset.has_pose``).
    """
    model.eval()

    all_t_err, all_q_err, all_score = [], [], []
    for batch in dataloader:
        images = batch["image"].to(device)
        t_gt = batch["translation"].to(device)
        q_gt = batch["quaternion"].to(device)

        t_pred, q_pred = model(images)
        result = speed_score(t_pred, q_pred, t_gt, q_gt)

        all_t_err.append(result["translation_error"].cpu())
        all_q_err.append(result["rotation_error_deg"].cpu())
        all_score.append(result["score"].cpu())

    t_err = torch.cat(all_t_err)
    q_err = torch.cat(all_q_err)
    score = torch.cat(all_score)

    return {
        "mean_translation_error": t_err.mean().item(),
        "mean_rotation_error_deg": q_err.mean().item(),
        "mean_speed_score": score.mean().item(),
        "num_samples": len(score),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a tumbleweed pose checkpoint")
    parser.add_argument("--data-root", type=str, default="data/speedplusv2")
    parser.add_argument("--domain", type=str, default="synthetic")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = build_arg_parser().parse_args()

    dataset = SpeedPlusDataset(
        root=args.data_root,
        domain=args.domain,
        split=args.split,
        transform=build_transforms(image_size=args.image_size, train=False),
    )
    if not dataset.has_pose:
        raise ValueError(
            f"{args.domain}/{args.split} has no pose labels; SPEC scoring "
            "requires a labeled split (e.g. synthetic/validation)."
        )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = PoseResNet18(pretrained=False).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"] if "model" in state else state)

    metrics = evaluate(model, dataloader, args.device)
    logger.info(
        "eval: n=%d  mean_t_err=%.4f  mean_q_err_deg=%.4f  mean_speed_score=%.4f",
        metrics["num_samples"],
        metrics["mean_translation_error"],
        metrics["mean_rotation_error_deg"],
        metrics["mean_speed_score"],
    )


if __name__ == "__main__":
    main()
