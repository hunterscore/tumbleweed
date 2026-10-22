"""Project 3D points into the image plane given a SPEED+ pose + camera.

Used by ``scripts/sanity_check_projection.py`` to visually verify that the
JSON pose labels, quaternion convention, and camera intrinsics are all being
interpreted consistently, independent of the model.
"""
from __future__ import annotations

from typing import Optional, Tuple

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

    ``q_vbs2tango`` names the vbs->tango DCM in the aerospace (passive)
    convention, but :func:`to_rotation_matrix` returns the Hamilton (active)
    rotation matrix, which is that DCM's transpose. So ``to_rotation_matrix``
    applied to ``q_vbs2tango`` already yields tango->vbs, and a body-frame
    point reaches the camera frame as ``p_cam = R @ p_body + t`` with no
    further transpose. Verified against the dataset by reprojecting the Tango
    keypoint model onto labeled images (``scripts/sanity_check_keypoints.py``).

    Args:
        points_body: (N, 3) points in the Tango body frame, meters.
        q_vbs2tango: (4,) scalar-first unit quaternion.
        t_vbs2tango: (3,) translation, meters.
        camera: parsed camera.json intrinsics.

    Returns:
        (N, 2) pixel coordinates.
    """
    q = torch.as_tensor(q_vbs2tango, dtype=torch.float64)
    r_tango2vbs = to_rotation_matrix(q[None, :])[0].numpy()

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


def _dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color,
    thickness: int,
    dash_length: float = 10.0,
    gap_length: float = 6.0,
) -> None:
    """Draw a dashed line segment from `pt1` to `pt2` (in place)."""
    start = np.asarray(pt1, dtype=np.float64)
    end = np.asarray(pt2, dtype=np.float64)
    length = float(np.linalg.norm(end - start))
    if length < 1e-6:
        return
    direction = (end - start) / length

    step = dash_length + gap_length
    offset = 0.0
    while offset < length:
        dash_end = min(offset + dash_length, length)
        p1 = start + direction * offset
        p2 = start + direction * dash_end
        cv2.line(image, tuple(p1.astype(int)), tuple(p2.astype(int)), color, thickness, cv2.LINE_AA)
        offset += step


def draw_pose(
    image: np.ndarray,
    points_2d: np.ndarray,
    colors: np.ndarray,
    thickness: int = 4,
    dashed: bool = False,
    origin_color: Optional[Tuple[int, int, int]] = (0, 255, 255),
) -> np.ndarray:
    """Draw a projected axis triad onto a copy of `image`.

    Args:
        points_2d: (4, 2) pixel coords, as returned by ``project_points`` on
            ``axis_points()``'s output: [origin, +X tip, +Y tip, +Z tip].
        colors: (3, 3) BGR color per axis, as returned by ``axis_points``.
        dashed: draw dashed rather than solid lines - useful for overlaying a
            second pose (e.g. predicted) on the same image as a first (e.g.
            ground truth) without the two triads being confusable.
        origin_color: BGR fill color for the origin marker dot, or None to
            skip it (e.g. for the second pose in an overlay, to avoid two
            overlapping dots at slightly different pixel locations).
    """
    origin = tuple(points_2d[0].astype(int))
    out = image.copy()
    for i in range(3):
        tip = tuple(points_2d[i + 1].astype(int))
        color = colors[i].tolist()
        if dashed:
            _dashed_line(out, origin, tip, color, thickness)
        else:
            cv2.line(out, origin, tip, color, thickness=thickness, lineType=cv2.LINE_AA)
    if origin_color is not None:
        cv2.circle(out, origin, 6, origin_color, thickness=-1, lineType=cv2.LINE_AA)
    return out
