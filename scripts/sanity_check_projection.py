#!/usr/bin/env python3
"""Sanity check: project a SPEED+ ground-truth pose onto its image.

Draws a body-frame axis triad (X=red, Y=green, Z=blue) at the Tango origin
using the label's quaternion/translation and the dataset's camera
intrinsics. Run this BEFORE trusting any training results — if the axes
don't sit on the spacecraft body in the image, the label parsing, quaternion
convention, or camera intrinsics are being misread somewhere upstream of
the model.

Usage:
    python scripts/sanity_check_projection.py \
        --data-root data/speedplus --domain synthetic --split train \
        --index 0 --output /tmp/sanity_check.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tumbleweed.data.camera import load_camera
from tumbleweed.geometry.projection import axis_points, project_points


def draw_pose(image: np.ndarray, points_2d: np.ndarray, colors: np.ndarray) -> np.ndarray:
    origin = tuple(points_2d[0].astype(int))
    out = image.copy()
    for i in range(3):
        tip = tuple(points_2d[i + 1].astype(int))
        cv2.line(out, origin, tip, colors[i].tolist(), thickness=4, lineType=cv2.LINE_AA)
    cv2.circle(out, origin, 6, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Project a SPEED+ GT pose onto its image")
    parser.add_argument("--data-root", type=str, default="data/speedplus")
    parser.add_argument("--domain", type=str, default="synthetic")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--index", type=int, default=0, help="index into the label list")
    parser.add_argument("--axis-length", type=float, default=0.5, help="meters")
    parser.add_argument("--output", type=str, default="sanity_check_projection.png")
    args = parser.parse_args()

    root = Path(args.data_root)
    camera = load_camera(root / "camera.json")

    label_path = root / args.domain / f"{args.split}.json"
    with open(label_path) as f:
        entries = json.load(f)
    entry = entries[args.index]

    if "q_vbs2tango_true" not in entry:
        raise ValueError(
            f"{args.domain}/{args.split} entry {args.index} has no pose label "
            "(lightbox/sunlamp test splits are unlabeled)."
        )

    q = np.asarray(entry["q_vbs2tango_true"], dtype=np.float64)
    t = np.asarray(entry["r_Vo2To_vbs_true"], dtype=np.float64)

    image_path = root / args.domain / "images" / entry["filename"]
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    points_body, colors = axis_points(length=args.axis_length)
    points_2d = project_points(points_body, q, t, camera)

    out = draw_pose(image, points_2d, colors)
    cv2.imwrite(args.output, out)
    print(f"filename: {entry['filename']}")
    print(f"q_vbs2tango_true: {q.tolist()}")
    print(f"r_Vo2To_vbs_true: {t.tolist()}")
    print(f"projected origin (px): {points_2d[0].tolist()}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
