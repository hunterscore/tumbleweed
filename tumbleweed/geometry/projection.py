"""Project 3D points into the image plane given a SPEED+ pose + camera.

Used by ``scripts/sanity_check_projection.py`` to visually verify that the
JSON pose labels, quaternion convention, and camera intrinsics are all being
interpreted consistently, independent of the model.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch

from tumbleweed.data.camera import CameraIntrinsics
from tumbleweed.geometry.quaternion import to_rotation_matrix


def project_points(
    points_body: np.ndarray,
    q_vbs2tango: np.ndarray,
    t_vbs2tango: np.ndarray,
    camera: CameraIntrinsics,
) -> np.ndarray:
    """Project 3D points defined in the Tango body frame into pixel coords.

    ``q_vbs2tango`` rotates the camera (VBS) frame into the Tango body
    frame, and ``t_vbs2tango`` is the Tango origin expressed in the camera
    frame. To bring a body-frame point into the camera frame we therefore
    apply the inverse rotation (the transpose, since it's orthonormal) and
    then add the translation: ``p_cam = R^T @ p_body + t``.

    Args:
        points_body: (N, 3) points in the Tango body frame, meters.
        q_vbs2tango: (4,) scalar-first unit quaternion.
        t_vbs2tango: (3,) translation, meters.
        camera: parsed camera.json intrinsics.

    Returns:
        (N, 2) pixel coordinates.
    """
    q = torch.as_tensor(q_vbs2tango, dtype=torch.float64)
    r_vbs2tango = to_rotation_matrix(q[None, :])[0].numpy()
    r_tango2vbs = r_vbs2tango.T

    rvec, _ = cv2.Rodrigues(r_tango2vbs)
    tvec = np.asarray(t_vbs2tango, dtype=np.float64).reshape(3, 1)

    points_2d, _ = cv2.projectPoints(
        np.asarray(points_body, dtype=np.float64),
        rvec,
        tvec,
        camera.camera_matrix,
        camera.dist_coeffs,
    )
    return points_2d.reshape(-1, 2)


def axis_points(length: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Body-frame origin + unit-axis endpoints for drawing a pose triad.

    Returns (origin_and_axes, colors) where origin_and_axes is (4, 3):
    [origin, +X tip, +Y tip, +Z tip], and colors is the BGR color to draw
    each of the three axis segments (X=red, Y=green, Z=blue).
    """
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [length, 0.0, 0.0],
            [0.0, length, 0.0],
            [0.0, 0.0, length],
        ]
    )
    colors = np.array(
        [
            [0, 0, 255],  # X: red (BGR)
            [0, 255, 0],  # Y: green
            [255, 0, 0],  # Z: blue
        ]
    )
    return pts, colors
