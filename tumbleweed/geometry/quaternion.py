"""Batched, differentiable quaternion utilities.

Convention: quaternions are **scalar-first**, ``q = (w, x, y, z)``, matching
the ``q_vbs2tango_true`` field in SPEED+ labels. All functions accept
tensors of shape ``(..., 4)`` and broadcast over arbitrary leading batch
dimensions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize quaternions along the last dimension."""
    return F.normalize(q, p=2, dim=-1, eps=eps)


def to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions (..., 4) to rotation matrices (..., 3, 3).

    Does NOT re-normalize internally; callers should normalize first if the
    input isn't already a unit quaternion (e.g. raw network output).
    """
    w, x, y, z = q.unbind(-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)

    rows = [
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


def multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ⊗ q2 of scalar-first quaternions (..., 4).

    Composes rotations: if q1 rotates frame A->B and q2 rotates frame B->C,
    ``multiply(q2, q1)`` rotates A->C (same convention as rotation matrices,
    where composed rotation is ``R2 @ R1``).
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate (w, -x, -y, -z): inverts the rotation direction."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def inverse(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Multiplicative inverse of q (== conjugate for a unit quaternion)."""
    norm_sq = (q * q).sum(dim=-1, keepdim=True).clamp(min=eps)
    return conjugate(q) / norm_sq


def geodesic_angle(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Angle (radians) of the shortest rotation between two quaternions.

    Both inputs are normalized internally. Handles the double-cover of SO(3)
    by a quaternion (q and -q represent the same rotation) by taking the
    absolute value of the dot product, and clamps to avoid NaN gradients
    from acos at +-1.
    """
    q1 = normalize(q1)
    q2 = normalize(q2)
    dot = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0 - eps)
    return 2.0 * torch.acos(dot)
