"""Recover a SPEED+ pose from 2D<->3D keypoint correspondences via PnP."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from tumbleweed.data.camera import CameraIntrinsics
from tumbleweed.geometry.projection import project_points
from tumbleweed.geometry.quaternion import from_rotation_matrix
from tumbleweed.geometry.tango_keypoints import TANGO_KEYPOINTS_BODY

MIN_CORRESPONDENCES = 4


@dataclass
class PnPResult:
    """Pose in SPEED+ label convention, plus whether the solve actually worked."""

    quaternion: np.ndarray  # (4,) scalar-first q_vbs2tango
    translation: np.ndarray  # (3,) meters, Tango origin in camera frame
    success: bool
    num_inliers: int


def solve_pose(
    keypoints_px: np.ndarray,
    camera: CameraIntrinsics,
    confidence: Optional[np.ndarray] = None,
    keypoints_body: np.ndarray = TANGO_KEYPOINTS_BODY,
    min_confidence: float = 0.1,
    reprojection_error: float = 8.0,
) -> PnPResult:
    """Solve for the pose that projects ``keypoints_body`` onto ``keypoints_px``.

    Coordinates must be in full-resolution pixels, matching ``camera``.

    RANSAC does the work of ignoring keypoints the detector placed badly —
    self-occluded points have no visibility label to filter on, so they arrive
    as confident-looking but wrong detections.
    """
    points_2d = np.asarray(keypoints_px, dtype=np.float64).reshape(-1, 2)
    points_3d = np.asarray(keypoints_body, dtype=np.float64).reshape(-1, 3)

    if confidence is not None:
        keep = np.asarray(confidence, dtype=np.float64).reshape(-1) >= min_confidence
        points_2d, points_3d = points_2d[keep], points_3d[keep]

    if len(points_2d) < MIN_CORRESPONDENCES:
        return _failure()

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d,
        points_2d,
        camera.camera_matrix,
        camera.dist_coeffs,
        reprojectionError=reprojection_error,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < MIN_CORRESPONDENCES:
        return _failure()

    idx = inliers.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(
        points_3d[idx], points_2d[idx], camera.camera_matrix, camera.dist_coeffs, rvec, tvec
    )

    # project_points uses p_cam = R @ p_body + t, so the matrix cv2 returns is
    # exactly the one to_rotation_matrix produces from q_vbs2tango.
    r_tango2vbs, _ = cv2.Rodrigues(rvec)
    q = from_rotation_matrix(torch.as_tensor(r_tango2vbs, dtype=torch.float64)[None])[0]

    return PnPResult(
        quaternion=q.numpy(),
        translation=tvec.reshape(3),
        success=True,
        num_inliers=len(idx),
    )


def select_by_prior(
    candidates_px: np.ndarray,
    values: np.ndarray,
    q_prior: np.ndarray,
    t_prior: np.ndarray,
    camera: CameraIntrinsics,
    keypoints_body: np.ndarray = TANGO_KEYPOINTS_BODY,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick, for each keypoint, the heatmap candidate nearest a coarse pose prior.

    The detector often puts a blob on both members of a mirror pair. The prior
    (e.g. the regression model's pose) is too coarse for PnP on its own but
    easily close enough to say which blob is which keypoint.

    Args:
        candidates_px: (K, k, 2) candidate locations from ``find_peaks``, in
            full-resolution pixels (u, v), for one image.
        values: (K, k) peak value of each candidate; non-finite means "no peak".
        q_prior: (4,) scalar-first q_vbs2tango of the prior pose.
        t_prior: (3,) prior translation, meters.
        camera: intrinsics the prior is projected with.
        keypoints_body: (K, 3) keypoint model the prior is projected from.

    Returns:
        keypoints_px: (K, 2) chosen location per keypoint, ready for ``solve_pose``.
        confidence: (K,) peak value of each chosen candidate, so ``solve_pose``'s
            ``min_confidence`` filter still applies.
    """
    expected_px = project_points(keypoints_body, q_prior, t_prior, camera)  # (K, 2)

    diff = candidates_px - expected_px[:, None, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist = np.where(np.isfinite(values), dist, np.inf)

    choice = dist.argmin(axis=-1)

    rows = np.arange(len(candidates_px))
    keypoints_px = candidates_px[rows, choice]
    confidence = values[rows, choice]

    return keypoints_px, confidence


def _failure() -> PnPResult:
    return PnPResult(
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        translation=np.array([0.0, 0.0, 0.0]),
        success=False,
        num_inliers=0,
    )
