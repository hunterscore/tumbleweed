"""Loader for SPEED+'s camera.json intrinsics file."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics for the SPEED+ synthetic/hardware camera."""

    camera_matrix: np.ndarray  # (3, 3) float64
    dist_coeffs: np.ndarray  # (5,) float64, OpenCV order [k1, k2, p1, p2, k3]
    width: int
    height: int

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])


def load_camera(path: Union[str, Path]) -> CameraIntrinsics:
    """Parse SPEED+'s ``camera.json`` into a :class:`CameraIntrinsics`.

    SPEED+ ships a single ``camera.json`` at the dataset root shared by every
    domain (synthetic / lightbox / sunlamp). The file is expected to contain
    a ``cameraMatrix`` (3x3), ``distCoeffs`` and image dimensions under
    ``Nu``/``Nv`` (or the lowercase ``nu``/``nv`` used by some releases).
    """
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)

    if "cameraMatrix" in raw:
        camera_matrix = np.array(raw["cameraMatrix"], dtype=np.float64)
    else:
        # Fall back to reconstructing K from individual fx/fy/ccx/ccy fields
        # in case a release ships those instead of a pre-built matrix.
        fx = raw.get("fx") or raw.get("fpx")
        fy = raw.get("fy") or raw.get("fpy")
        cx = raw["ccx"]
        cy = raw["ccy"]
        camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    dist_coeffs = np.array(
        raw.get("distCoeffs", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float64
    )

    width = int(raw.get("Nu", raw.get("nu", raw.get("width", 1920))))
    height = int(raw.get("Nv", raw.get("nv", raw.get("height", 1200))))

    return CameraIntrinsics(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        width=width,
        height=height,
    )
