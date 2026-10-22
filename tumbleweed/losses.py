"""Losses for pose estimation: direct-regression pose losses and heatmap MSE."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tumbleweed.geometry.quaternion import geodesic_angle


def geodesic_rotation_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """Mean geodesic distance (radians) between predicted/GT quaternions."""
    return geodesic_angle(q_pred, q_gt).mean()


def translation_mse_loss(t_pred: torch.Tensor, t_gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(t_pred, t_gt)


class PoseLoss(nn.Module):
    """Weighted sum of translation MSE and geodesic rotation loss."""

    def __init__(self, rotation_weight: float = 1.0, translation_weight: float = 1.0):
        super().__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

    def forward(
        self,
        t_pred: torch.Tensor,
        q_pred: torch.Tensor,
        t_gt: torch.Tensor,
        q_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        rot_loss = geodesic_rotation_loss(q_pred, q_gt)
        trans_loss = translation_mse_loss(t_pred, t_gt)
        total = self.translation_weight * trans_loss + self.rotation_weight * rot_loss

        logs = {
            "loss/total": total.item(),
            "loss/translation_mse": trans_loss.item(),
            "loss/rotation_geodesic_rad": rot_loss.item(),
        }
        return total, logs


class KeypointLoss(nn.Module):
    """Visibility-masked MSE between predicted and target keypoint heatmaps.

    Targets are almost entirely zeros, so plain MSE is minimized fairly well by
    predicting nothing anywhere; ``peak_weight`` upweights the pixels covered by
    a target Gaussian to keep that collapse from being the easy optimum.
    """

    def __init__(self, peak_weight: float = 20.0):
        super().__init__()
        self.peak_weight = peak_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        visible: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        weight = 1.0 + self.peak_weight * target
        per_keypoint = (weight * (pred - target) ** 2).mean(dim=(-2, -1))

        mask = visible
        loss = (per_keypoint * mask).sum() / mask.sum().clamp(min=1.0)

        logs = {
            "loss/total": loss.item(),
            "loss/heatmap_mse": loss.item(),
            "loss/visible_frac": mask.mean().item(),
        }
        return loss, logs
