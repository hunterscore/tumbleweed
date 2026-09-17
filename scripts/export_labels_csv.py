#!/usr/bin/env python3
"""Flatten SPEED+ label JSON files into CSV for quick visual inspection.

Usage:
    python scripts/export_labels_csv.py --data-root data/speedplusv2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

SPLITS = [
    ("synthetic", "train"),
    ("synthetic", "validation"),
    ("lightbox", "test"),
    ("sunlamp", "test"),
]


def export(root: Path, domain: str, split: str) -> None:
    label_path = root / domain / f"{split}.json"
    if not label_path.exists():
        print(f"skip {domain}/{split}: {label_path} not found")
        return

    with open(label_path) as f:
        entries = json.load(f)

    out_path = root / domain / f"{split}.csv"
    has_pose = len(entries) > 0 and "q_vbs2tango_true" in entries[0]

    fieldnames = ["filename"]
    if has_pose:
        fieldnames += ["qw", "qx", "qy", "qz", "tx", "ty", "tz", "range_m"]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for e in entries:
            row = [e["filename"]]
            if has_pose:
                qw, qx, qy, qz = e["q_vbs2tango_true"]
                tx, ty, tz = e["r_Vo2To_vbs_true"]
                range_m = math.sqrt(tx * tx + ty * ty + tz * tz)
                row += [qw, qx, qy, qz, tx, ty, tz, f"{range_m:.4f}"]
            writer.writerow(row)

    print(f"wrote {out_path} ({len(entries)} rows, pose={has_pose})")


def main():
    parser = argparse.ArgumentParser(description="Export SPEED+ label JSON as CSV")
    parser.add_argument("--data-root", type=str, default="data/speedplusv2")
    args = parser.parse_args()

    root = Path(args.data_root)
    for domain, split in SPLITS:
        export(root, domain, split)


if __name__ == "__main__":
    main()
