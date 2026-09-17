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


def distribution_stats(name: str, values: torch.Tensor) -> Dict[str, float]:
    """mean/median/std/p90/p95 of a per-sample error tensor, prefixed by name.

    A single mean hides heavy-tailed error distributions - e.g. this
    baseline's rotation error has a median around 11 deg but a p99 over
    130 deg, because a small fraction of samples are near-total orientation
    flips rather than everything being uniformly imprecise. Report the full
    spread, not just the mean, whenever these errors matter.
    """
    return {
        f"mean_{name}": values.mean().item(),
        f"median_{name}": values.median().item(),
        f"std_{name}": values.std().item(),
        f"p90_{name}": torch.quantile(values, 0.90).item(),
        f"p95_{name}": torch.quantile(values, 0.95).item(),
    }


def flip_rate(rotation_error_deg: torch.Tensor, threshold_deg: float = 90.0) -> float:
    """Fraction of samples whose rotation error exceeds `threshold_deg`.

    A direct-regression pose head outputs one committed quaternion even when
    an image is genuinely ambiguous between two orientations (e.g. a
    near-symmetric spacecraft body) - when it guesses wrong, the result
    tends to be a near-total flip (~180 deg) rather than a small miss. This
    tracks how often that happens, independent of the mean/median, which
    barely move when a small fraction of predictions are catastrophically
    wrong. Default threshold (90 deg) flags "wrong orientation family", not
    just "imprecise".
    """
    return (rotation_error_deg > threshold_deg).float().mean().item()


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
