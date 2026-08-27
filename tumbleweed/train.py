"""Training loop for the ResNet18 direct pose regression baseline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tumbleweed.config import TrainConfig
from tumbleweed.data.speedplus import SpeedPlusDataset
from tumbleweed.data.transforms import build_transforms
from tumbleweed.eval import evaluate
from tumbleweed.losses import PoseLoss
from tumbleweed.models.pose_resnet import PoseResNet18

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train tumbleweed's ResNet18 pose baseline")
    parser.add_argument("--data-root", type=str, default=defaults.data_root)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--rotation-weight", type=float, default=defaults.rotation_weight)
    parser.add_argument("--translation-weight", type=float, default=defaults.translation_weight)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default=defaults.checkpoint_dir)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    train_set = SpeedPlusDataset(
        root=args.data_root,
        domain="synthetic",
        split="train",
        transform=build_transforms(image_size=args.image_size, train=True),
    )
    val_set = SpeedPlusDataset(
        root=args.data_root,
        domain="synthetic",
        split="validation",
        transform=build_transforms(image_size=args.image_size, train=False),
    )
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

    model = PoseResNet18(pretrained=not args.no_pretrained).to(args.device)
    criterion = PoseLoss(
        rotation_weight=args.rotation_weight, translation_weight=args.translation_weight
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            images = batch["image"].to(args.device)
            t_gt = batch["translation"].to(args.device)
            q_gt = batch["quaternion"].to(args.device)

            optimizer.zero_grad()
            t_pred, q_pred = model(images)
            loss, logs = criterion(t_pred, q_pred, t_gt, q_gt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=logs["loss/total"])

        train_loss = running_loss / len(train_loader)
        metrics = evaluate(model, val_loader, args.device)
        logger.info(
            "epoch %d: train_loss=%.4f  val_t_err=%.4f  val_q_err_deg=%.4f  val_speed_score=%.4f",
            epoch,
            train_loss,
            metrics["mean_translation_error"],
            metrics["mean_rotation_error_deg"],
            metrics["mean_speed_score"],
        )

        torch.save({"model": model.state_dict(), "epoch": epoch}, checkpoint_dir / "last.pt")
        if metrics["mean_speed_score"] < best_score:
            best_score = metrics["mean_speed_score"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, checkpoint_dir / "best.pt")
            logger.info("epoch %d: new best speed_score=%.4f, saved checkpoint", epoch, best_score)


def main():
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
