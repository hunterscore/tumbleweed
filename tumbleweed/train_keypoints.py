"""Training loop for the keypoint-heatmap + PnP pose pipeline."""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from tumbleweed.config import KeypointTrainConfig
from tumbleweed.data.camera import load_camera
from tumbleweed.data.speedplus import SpeedPlusDataset
from tumbleweed.data.transforms import build_keypoint_transforms
from tumbleweed.eval import evaluate
from tumbleweed.geometry.heatmaps import decode_heatmaps
from tumbleweed.losses import KeypointLoss
from tumbleweed.models.keypoint_resnet import KeypointPoseEstimator, KeypointResNet18

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = KeypointTrainConfig()
    parser = argparse.ArgumentParser(description="Train the keypoint heatmap pose model")
    parser.add_argument("--data-root", type=str, default=defaults.data_root)
    parser.add_argument("--image-size", type=int, nargs=2, default=list(defaults.image_size))
    parser.add_argument("--stride", type=int, default=defaults.stride)
    parser.add_argument("--sigma", type=float, default=defaults.sigma)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--peak-weight", type=float, default=defaults.peak_weight)
    parser.add_argument("--min-confidence", type=float, default=defaults.min_confidence)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default=defaults.checkpoint_dir)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="truncate both splits; for overfit smoke tests",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def keypoint_pixel_error(
    coords: torch.Tensor, target_px: torch.Tensor, visible: torch.Tensor, to_full_res
) -> float:
    """Mean full-resolution pixel distance between decoded and GT keypoints."""
    scale = torch.as_tensor(to_full_res, dtype=coords.dtype, device=coords.device)
    error = torch.norm(coords * scale - target_px.to(coords.device), dim=-1)
    mask = visible.to(coords.device)
    return ((error * mask).sum() / mask.sum().clamp(min=1.0)).item()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    image_size = tuple(args.image_size)

    camera = load_camera(Path(args.data_root) / "camera.json")
    transform = build_keypoint_transforms(
        camera, image_size=image_size, stride=args.stride, sigma=args.sigma
    )

    train_set = SpeedPlusDataset(args.data_root, "synthetic", "train", transform=transform)
    val_set = SpeedPlusDataset(args.data_root, "synthetic", "validation", transform=transform)
    if args.max_samples is not None:
        train_set = Subset(train_set, range(min(args.max_samples, len(train_set))))
        val_set = Subset(val_set, range(min(args.max_samples, len(val_set))))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    network = KeypointResNet18(pretrained=not args.no_pretrained).to(args.device)
    estimator = KeypointPoseEstimator(
        network,
        camera,
        image_size=image_size,
        stride=args.stride,
        min_confidence=args.min_confidence,
    ).to(args.device)
    criterion = KeypointLoss(peak_weight=args.peak_weight)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")

    metrics_csv_path = checkpoint_dir / "metrics.csv"
    metrics_csv_file = open(metrics_csv_path, "w", newline="")
    metrics_csv_writer = None

    for epoch in range(1, args.epochs + 1):
        network.train()
        running_loss = 0.0
        running_px_error = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            images = batch["image"].to(args.device)
            target = batch["heatmaps"].to(args.device)
            visible = batch["keypoint_visible"].to(args.device)

            optimizer.zero_grad()
            pred = network(images)
            loss, logs = criterion(pred, target, visible)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                coords, _ = decode_heatmaps(pred)
                px_error = keypoint_pixel_error(
                    coords, batch["keypoints_px"], visible, estimator.to_full_res
                )

            running_loss += loss.item()
            running_px_error += px_error
            pbar.set_postfix(loss=logs["loss/total"], px_err=px_error)

        train_loss = running_loss / len(train_loader)
        train_px_error = running_px_error / len(train_loader)

        estimator.reset_failure_stats()
        metrics = evaluate(estimator, val_loader, args.device)
        metrics["pnp_failure_rate"] = estimator.failure_rate

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_keypoint_px_error": train_px_error,
            **metrics,
        }
        if metrics_csv_writer is None:
            metrics_csv_writer = csv.DictWriter(metrics_csv_file, fieldnames=list(row.keys()))
            metrics_csv_writer.writeheader()
        metrics_csv_writer.writerow(row)
        metrics_csv_file.flush()

        logger.info(
            "epoch %d: loss=%.5f  train_px_err=%.2f  val_q_err_deg=%.4f  "
            "flip_rate_90deg=%.4f  pnp_fail=%.4f  val_speed_score=%.4f",
            epoch,
            train_loss,
            train_px_error,
            metrics["mean_rotation_error_deg"],
            metrics["flip_rate_90deg"],
            metrics["pnp_failure_rate"],
            metrics["mean_speed_score"],
        )

        torch.save({"model": network.state_dict(), "epoch": epoch}, checkpoint_dir / "last.pt")
        if metrics["mean_speed_score"] < best_score:
            best_score = metrics["mean_speed_score"]
            torch.save({"model": network.state_dict(), "epoch": epoch}, checkpoint_dir / "best.pt")
            logger.info("epoch %d: new best speed_score=%.4f, saved checkpoint", epoch, best_score)

    metrics_csv_file.close()
    logger.info("wrote per-epoch metrics to %s", metrics_csv_path)


def main():
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
