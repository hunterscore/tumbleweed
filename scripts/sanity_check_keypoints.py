#!/usr/bin/env python3
"""Sanity check: project the 11 Tango keypoints through a GT pose onto its image.

Run this BEFORE training a keypoint model. If the numbered points don't sit
on the spacecraft's real corners / antenna tips across varied poses, the
keypoint model's frame convention doesn't match the SPEED+ labels.

Usage:
    python scripts/sanity_check_keypoints.py --data-root data/speedplusv2 \
        --indices 0 500 1000 --output-dir /tmp/kp_check
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
from tumbleweed.geometry.projection import project_points
from tumbleweed.geometry.tango_keypoints import BODY_EDGES, TANGO_KEYPOINTS_BODY


def main():
    parser = argparse.ArgumentParser(description="Project Tango keypoints onto SPEED+ images")
    parser.add_argument("--data-root", type=str, default="data/speedplusv2")
    parser.add_argument("--domain", type=str, default="synthetic")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--indices", type=int, nargs="+", default=[0])
    parser.add_argument("--output-dir", type=str, default="keypoint_checks")
    args = parser.parse_args()

    root = Path(args.data_root)
    camera = load_camera(root / "camera.json")
    with open(root / args.domain / f"{args.split}.json") as f:
        entries = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in args.indices:
        entry = entries[idx]
        q = np.asarray(entry["q_vbs2tango_true"], dtype=np.float64)
        t = np.asarray(entry["r_Vo2To_vbs_true"], dtype=np.float64)
        image = cv2.imread(str(root / args.domain / "images" / entry["filename"]))
        if image is None:
            raise FileNotFoundError(entry["filename"])

        pts = project_points(TANGO_KEYPOINTS_BODY, q, t, camera)
        for a, b in BODY_EDGES:
            cv2.line(
                image, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                (255, 160, 0), 2, cv2.LINE_AA,
            )
        for i, (u, v) in enumerate(pts):
            center = (int(round(u)), int(round(v)))
            cv2.circle(image, center, 8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                image, str(i), (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA,
            )

        out_path = out_dir / f"{Path(entry['filename']).stem}_keypoints.png"
        cv2.imwrite(str(out_path), image)
        print(f"[{idx}] {entry['filename']}  q={q.tolist()}  t={t.tolist()}  -> {out_path}")


if __name__ == "__main__":
    main()
