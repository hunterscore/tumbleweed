"""Decode predicted keypoint heatmaps into sub-pixel 2D coordinates."""
from __future__ import annotations

from typing import Tuple

import torch


def decode_heatmaps(
    heatmaps: torch.Tensor, window: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Locate each heatmap's peak to sub-pixel accuracy.

    A plain argmax quantizes to the heatmap grid, which at stride 4 is a 4-pixel
    error in the input image before PnP ever runs. Refining with an
    intensity-weighted centroid over a small window around the peak recovers the
    fraction of a cell, while staying local enough that a second, weaker peak
    elsewhere (the symmetric-corner failure mode) cannot drag the estimate.

    Args:
        heatmaps: (N, K, H, W) raw network output.
        window: half-width of the refinement window, in heatmap cells.

    Returns:
        coords: (N, K, 2) peak locations in heatmap coordinates (u, v).
        confidence: (N, K) peak value of each heatmap.
    """
    n, k, height, width = heatmaps.shape
    flat = heatmaps.flatten(2)
    confidence, flat_idx = flat.max(dim=2)
    peak_v = torch.div(flat_idx, width, rounding_mode="floor")
    peak_u = flat_idx % width

    du = torch.arange(-window, window + 1, device=heatmaps.device)
    offsets_u = (peak_u[..., None] + du).clamp(0, width - 1)
    offsets_v = (peak_v[..., None] + du).clamp(0, height - 1)

    # Gather the (2*window+1)^2 patch around each peak.
    batch_idx = torch.arange(n, device=heatmaps.device)[:, None, None, None]
    kp_idx = torch.arange(k, device=heatmaps.device)[None, :, None, None]
    patch = heatmaps[
        batch_idx, kp_idx, offsets_v[..., :, None], offsets_u[..., None, :]
    ]

    # Weight by activation above the patch floor so a negative background
    # doesn't produce negative weights.
    weights = (patch - patch.amin(dim=(-2, -1), keepdim=True)).clamp(min=0)
    total = weights.sum(dim=(-2, -1)).clamp(min=1e-6)
    u = (weights.sum(dim=-2) * offsets_u).sum(dim=-1) / total
    v = (weights.sum(dim=-1) * offsets_v).sum(dim=-1) / total

    return torch.stack([u, v], dim=-1), confidence
