"""Tests for picking heatmap candidates with a coarse pose prior (``select_by_prior``)."""
import numpy as np
import torch

from tumbleweed.geometry.pnp import select_by_prior, solve_pose
from tumbleweed.geometry.projection import project_points
from tumbleweed.geometry.tango_keypoints import TANGO_KEYPOINTS_BODY
from tests.test_pnp import _rotation_error_deg
from tests.test_tango_keypoints import CAMERA, Q_GT, T_GT
from tumbleweed.geometry.quaternion import multiply, normalize


def _make_candidates(true_px, decoy_offset_px, correct_slot=0, k=4):
    """Build (K, k, 2) candidates and (K, k) values around the true keypoints.

    Each keypoint gets the true location in ``correct_slot`` and decoys,
    ``decoy_offset_px`` away, in the other slots.
    """
    offsets = decoy_offset_px*np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    decoys = true_px[:, None, :] + offsets
    candidates = decoys.copy()
    candidates[:, correct_slot] = true_px

    values = np.tile(np.array([1.0, 0.8, 0.6, 0.4]), (len(true_px), 1))

    return candidates, values


def _perturb_quaternion(q, angle_deg, axis):
    """Rotate scalar-first ``q`` by ``angle_deg`` about ``axis``, so the prior is off by about that much."""
    angle_rad = np.deg2rad(angle_deg)
    axis = axis / np.linalg.norm(axis)

    delta_q = np.array([np.cos(angle_rad/2), axis[0]*np.sin(angle_rad/2),
                         axis[1]*np.sin(angle_rad/2), axis[2]*np.sin(angle_rad/2)])

    q_prior = normalize(multiply(torch.as_tensor(delta_q), torch.as_tensor(q))).numpy()

    return q_prior


# A regression-sized prior error: ~median rotation, and ~20 cm of translation
# at ~6.7 m range. Projected, this lands up to ~107 px from the true keypoints,
# while each mirror twin stays at least ~136 px farther away than that.
PRIOR_ERROR_DEG = 10.0
PRIOR_AXIS = [1, 1, 0]
PRIOR_T_OFFSET = np.array([0.1, -0.1, 0.15])

# Generic decoys sit this far from the truth, well beyond the prior's error.
FAR_DECOY_PX = 300.0


def _true_px():
    return project_points(TANGO_KEYPOINTS_BODY, Q_GT, T_GT, CAMERA)


def _mirror_px():
    """Where each keypoint's left/right twin (body x -> -x) appears: the detector's usual swap."""
    return project_points(TANGO_KEYPOINTS_BODY * [-1, 1, 1], Q_GT, T_GT, CAMERA)


def _mirror_candidates():
    """True point in slot 1, its brighter mirror twin in slot 0, far decoys after."""
    candidates, values = _make_candidates(_true_px(), FAR_DECOY_PX, correct_slot=1)
    candidates[:, 0] = _mirror_px()
    return candidates, values


def _offset_prior():
    q = _perturb_quaternion(Q_GT, PRIOR_ERROR_DEG, PRIOR_AXIS)
    return q, T_GT + PRIOR_T_OFFSET


def test_picks_the_right_candidate_when_it_is_not_the_strongest():
    """The true location sits in slot 1 under a brighter decoy; an exact prior still picks it."""
    true_px = _true_px()
    candidates, values = _make_candidates(true_px, 50.0, correct_slot=1)

    chosen_px, confidence = select_by_prior(candidates, values, Q_GT, T_GT, CAMERA)

    np.testing.assert_allclose(chosen_px, true_px)
    np.testing.assert_allclose(confidence, values[:, 1])


def test_tolerates_a_prior_about_ten_degrees_off():
    """A regression-sized error (~10 deg) still lands nearer the right blob than its mirror twin."""
    candidates, values = _mirror_candidates()
    q_prior, t_prior = _offset_prior()

    chosen_px, confidence = select_by_prior(candidates, values, q_prior, t_prior, CAMERA)

    np.testing.assert_allclose(chosen_px, _true_px())
    np.testing.assert_allclose(confidence, values[:, 1])


def test_skips_slots_without_a_peak():
    """A -inf slot is never chosen, even if its location matches the prior exactly."""
    true_px = _true_px()
    candidates, values = _make_candidates(true_px, FAR_DECOY_PX, correct_slot=1)
    # Keypoint 0: an empty slot exactly on the prior, the real peak 3 px away.
    candidates[0, 0] = true_px[0]
    values[0, 0] = -np.inf
    candidates[0, 1] = true_px[0] + [3.0, 0.0]

    chosen_px, confidence = select_by_prior(candidates, values, Q_GT, T_GT, CAMERA)

    np.testing.assert_allclose(chosen_px[0], true_px[0] + [3.0, 0.0])
    assert np.isfinite(confidence).all()


def test_selected_points_recover_the_pose_through_solve_pose():
    """End to end: perturbed prior -> select_by_prior -> solve_pose gets the GT pose back."""
    candidates, values = _mirror_candidates()
    q_prior, t_prior = _offset_prior()

    chosen_px, confidence = select_by_prior(candidates, values, q_prior, t_prior, CAMERA)
    result = solve_pose(chosen_px, CAMERA, confidence, reprojection_error=20.0)

    assert result.success
    assert result.num_inliers == len(TANGO_KEYPOINTS_BODY)
    # The prior was 10 deg off; PnP on the right points removes that error.
    assert _rotation_error_deg(result.quaternion, Q_GT) < 0.1
    np.testing.assert_allclose(result.translation, T_GT, atol=1e-3)
