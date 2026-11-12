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
    # TODO: build the small-rotation quaternion from axis and angle, compose it
    #       with q, normalize.
    angle_rad = np.deg2rad(angle_deg)
    axis = axis / np.linalg.norm(axis)

    delta_q = np.array([np.cos(angle_rad/2), axis[0]*np.sin(angle_rad/2),
                         axis[1]*np.sin(angle_rad/2), axis[2]*np.sin(angle_rad/2)])

    q_prior = normalize(multiply(torch.as_tensor(delta_q), torch.as_tensor(q))).numpy()

    return q_prior


def test_picks_the_right_candidate_when_it_is_not_the_strongest():
    """The true location sits in slot 1 under a brighter decoy; an exact prior still picks it."""
    # TODO: project the GT pose to get true_px.
    # TODO: candidates with the true point in slot 1 and decoys far enough
    #       away to be unambiguous.
    # TODO: select_by_prior with the GT pose as prior.
    # TODO: assert the chosen points equal true_px and the confidences are the
    #       slot-1 values.
    raise NotImplementedError


def test_tolerates_a_prior_about_ten_degrees_off():
    """A regression-sized error (~10 deg) still lands nearer the right blob than its mirror twin."""
    # TODO: prior = GT rotated ~10 deg (and optionally t nudged a little).
    # TODO: decoys at the mirror-pair locations: project TANGO_KEYPOINTS_BODY
    #       with x negated, the actual failure mode.
    # TODO: assert every chosen point is the true one.
    # TODO: optionally print or assert the projected-prior error in px, to see
    #       how much margin ~10 deg gives at this range.
    raise NotImplementedError


def test_skips_slots_without_a_peak():
    """A -inf slot is never chosen, even if its location matches the prior exactly."""
    # TODO: put a candidate exactly at true_px for one keypoint but set its
    #       value to -inf; put the real (finite) one a few px away.
    # TODO: assert the finite one is chosen and its confidence is finite.
    raise NotImplementedError


def test_selected_points_recover_the_pose_through_solve_pose():
    """End to end: perturbed prior -> select_by_prior -> solve_pose gets the GT pose back."""
    # TODO: candidates with mirror decoys, prior ~10 deg off, as above.
    # TODO: select_by_prior, then solve_pose with the returned confidence and
    #       reprojection_error=20 (the estimator's setting).
    # TODO: assert success, rotation error well under the prior's error,
    #       translation close to T_GT.
    raise NotImplementedError
