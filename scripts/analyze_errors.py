#!/usr/bin/env python3
"""Analyze a per-sample error CSV produced by `eval.py --advanced`.

Reports the full error distribution (not just the mean), the flip rate
(fraction of near-total ~180 deg orientation errors - the dominant failure
mode found in the baseline model), how error varies with GT range, and the
worst individual predictions - so a single eval run can be turned into an
actual diagnosis instead of one aggregate number.

Usage:
    python scripts/analyze_errors.py --csv results/best_synthetic-validation.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tumbleweed.metrics import distribution_stats, flip_rate


def load_per_sample_csv(path: str):
    rows = list(csv.DictReader(open(path)))
    if not rows or "gt_range_m" not in rows[0]:
        raise ValueError(
            f"{path} doesn't look like a per-sample error CSV - "
            "run `eval.py --advanced` to produce one."
        )
    return rows


def print_distribution(name: str, values: torch.Tensor) -> None:
    stats = distribution_stats(name, values)
    print(f"\n{name}:")
    for key in ["mean", "median", "std", "p90", "p95"]:
        print(f"  {key:8s}: {stats[f'{key}_{name}']:.3f}")


def print_range_bins(rotation_error: np.ndarray, gt_range: np.ndarray, num_bins: int) -> None:
    order = np.argsort(gt_range)
    rng_sorted = gt_range[order]
    rot_sorted = rotation_error[order]
    n = len(gt_range)
    bin_size = n // num_bins

    print(f"\nrotation error by GT range ({num_bins} bins):")
    for i in range(num_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < num_bins - 1 else n
        chunk_rng = rng_sorted[start:end]
        chunk_rot = rot_sorted[start:end]
        print(
            f"  {chunk_rng.min():5.1f}-{chunk_rng.max():5.1f}m: "
            f"mean={chunk_rot.mean():6.2f} deg  median={np.median(chunk_rot):6.2f} deg  n={end - start}"
        )


def print_worst_cases(rows: list, n: int) -> None:
    worst = sorted(rows, key=lambda r: -float(r["rotation_error_deg"]))[:n]
    print(f"\nworst {n} predictions by rotation error:")
    for r in worst:
        print(
            f"  {r['filename']:20s} rot_err={float(r['rotation_error_deg']):7.2f} deg  "
            f"trans_err={float(r['translation_error']):.4f}  range={float(r['gt_range_m']):.2f}m"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze a per-sample pose error CSV")
    parser.add_argument("--csv", type=str, required=True, help="per-sample CSV from eval.py --advanced")
    parser.add_argument("--num-bins", type=int, default=5, help="number of GT-range bins")
    parser.add_argument("--worst-n", type=int, default=10, help="number of worst cases to list")
    parser.add_argument(
        "--flip-threshold", type=float, default=90.0, help="rotation error (deg) counted as a flip"
    )
    args = parser.parse_args()

    rows = load_per_sample_csv(args.csv)
    rot = torch.tensor([float(r["rotation_error_deg"]) for r in rows])
    trans = torch.tensor([float(r["translation_error"]) for r in rows])
    gt_range = np.array([float(r["gt_range_m"]) for r in rows])

    print(f"n = {len(rows)}")
    print_distribution("rotation_error_deg", rot)
    print_distribution("translation_error", trans)

    rate = flip_rate(rot, threshold_deg=args.flip_threshold)
    print(f"\nflip_rate (> {args.flip_threshold:.0f} deg): {rate:.4f}")

    corr = np.corrcoef(rot.numpy(), gt_range)[0, 1]
    print(f"corr(rotation_error_deg, gt_range_m): {corr:.4f}")

    print_range_bins(rot.numpy(), gt_range, args.num_bins)
    print_worst_cases(rows, args.worst_n)


if __name__ == "__main__":
    main()
