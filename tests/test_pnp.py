import numpy as np
import torch

from tumbleweed.geometry.heatmaps import decode_heatmaps
from tumbleweed.geometry.pnp import solve_pose
from tumbleweed.geometry.projection import project_points
from tumbleweed.geometry.quaternion import geodesic_angle, normalize
from tumbleweed.geometry.tango_keypoints import TANGO_KEYPOINTS_BODY
from tests.test_tango_keypoints import CAMERA, Q_GT, T_GT


def _rotation_error_deg(q_a, q_b):
    angle = geodesic_angle(
        torch.as_tensor(q_a, dtype=torch.float64)[None],
        torch.as_tensor(q_b, dtype=torch.float64)[None],
    )
    return float(np.rad2deg(angle.item()))


def test_pnp_recovers_the_pose_that_generated_the_projection():
    keypoints_px = project_points(TANGO_KEYPOINTS_BODY, Q_GT, T_GT, CAMERA)
    result = solve_pose(keypoints_px, CAMERA)

    assert result.success
    assert result.num_inliers == len(TANGO_KEYPOINTS_BODY)
    # 0.051 deg is geodesic_angle's own eps-clamp floor, not solver error.
    assert _rotation_error_deg(result.quaternion, Q_GT) < 0.06
    assert np.allclose(result.translation, T_GT, atol=1e-4)


def test_pnp_is_robust_to_moderate_keypoint_noise():
    rng = np.random.default_rng(0)
    keypoints_px = project_points(TANGO_KEYPOINTS_BODY, Q_GT, T_GT, CAMERA)

    errors = []
    for _ in range(20):
        noisy = keypoints_px + rng.normal(0, 2.0, keypoints_px.shape)
        result = solve_pose(noisy, CAMERA)
        assert result.success
        errors.append(_rotation_error_deg(result.quaternion, Q_GT))

    assert max(errors) < 5.0


def test_pnp_rejects_a_grossly_misplaced_keypoint():
    keypoints_px = project_points(TANGO_KEYPOINTS_BODY, Q_GT, T_GT, CAMERA)
    corrupted = keypoints_px.copy()
    corrupted[3] = [10.0, 10.0]

    result = solve_pose(corrupted, CAMERA)

    assert result.success
    assert result.num_inliers == len(TANGO_KEYPOINTS_BODY) - 1
    assert _rotation_error_deg(result.quaternion, Q_GT) < 1.0


def test_pnp_reports_failure_when_too_few_keypoints_are_confident():
    keypoints_px = project_points(TANGO_KEYPOINTS_BODY, Q_GT, T_GT, CAMERA)
    confidence = np.zeros(len(keypoints_px))
    confidence[:3] = 1.0

    result = solve_pose(keypoints_px, CAMERA, confidence)

    assert not result.success
    assert result.num_inliers == 0
    assert np.allclose(normalize(torch.as_tensor(result.quaternion)[None]).numpy().sum(), 1.0)


def test_decode_heatmaps_recovers_subpixel_peaks():
    truth = torch.tensor([[[40.3, 20.7], [100.9, 60.2]]])
    heatmaps = torch.zeros(1, 2, 96, 160)
    vs = torch.arange(96).float()[:, None]
    us = torch.arange(160).float()[None, :]
    for i, (u, v) in enumerate(truth[0]):
        heatmaps[0, i] = torch.exp(-((us - u) ** 2 + (vs - v) ** 2) / (2 * 2.0**2))

    coords, confidence = decode_heatmaps(heatmaps)

    assert torch.allclose(coords, truth, atol=0.15)
    assert torch.all(confidence > 0.9)
