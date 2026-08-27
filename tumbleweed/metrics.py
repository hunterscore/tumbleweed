"""SPEC pose error metric (per-sample rotation + normalized translation error).

The Kelvins SPEC/SPEC2021 competitions built on SPEED/SPEED+ score a pose
estimate as ``e = e_t + e_q``, where:

    e_t = ||t_pred - t_gt|| / ||t_gt||                    (unitless)
    e_q = 2 * arccos(|<q_pred_hat, q_gt_hat>|)             (radians)

both computed per-sample and then averaged over the evaluation set. This
module implements that combined score. The official rules also zero out
near-negligible errors below fixed thresholds before averaging; the exact
threshold constants aren't reproduced here since we can't verify them
against the current competition rules, so thresholding is off by default —
pass explicit values via ``rotation_threshold_deg``/``translation_threshold``
if you need to match a specific leaderboard exactly.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from tumbleweed.geometry.quaternion import geodesic_angle


def rotation_error_deg(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
    """Per-sample rotation error in degrees, shape (N,)."""
    return torch.rad2deg(geodesic_angle(q_pred, q_gt))


def translation_error_norm(
    t_pred: torch.Tensor, t_gt: torch.Tensor, eps: float = 1e-9
) -> torch.Tensor:
    """Per-sample translation error normalized by GT range, shape (N,)."""
    return torch.norm(t_pred - t_gt, dim=-1) / torch.norm(t_gt, dim=-1).clamp(min=eps)


def speed_score(
    t_pred: torch.Tensor,
    q_pred: torch.Tensor,
    t_gt: torch.Tensor,
    q_gt: torch.Tensor,
    rotation_threshold_deg: Optional[float] = None,
    translation_threshold: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Per-sample SPEC pose score components, each of shape (N,).

    If both thresholds are given, samples whose errors fall under both are
    scored 0 for that component (near-negligible pose error rule).
    """
    e_t = translation_error_norm(t_pred, t_gt)
    e_q_deg = rotation_error_deg(q_pred, q_gt)
    e_q_rad = torch.deg2rad(e_q_deg)

    if rotation_threshold_deg is not None and translation_threshold is not None:
        negligible = (e_t < translation_threshold) & (e_q_deg < rotation_threshold_deg)
        e_t = torch.where(negligible, torch.zeros_like(e_t), e_t)
        e_q_rad = torch.where(negligible, torch.zeros_like(e_q_rad), e_q_rad)

    score = e_t + e_q_rad
    return {
        "translation_error": e_t,
        "rotation_error_deg": e_q_deg,
        "score": score,
    }
