"""Losses for direct pose regression: geodesic rotation loss + translation MSE."""
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
