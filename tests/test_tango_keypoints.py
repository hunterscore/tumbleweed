import numpy as np

from tumbleweed.data.camera import CameraIntrinsics
from tumbleweed.geometry.projection import project_points
from tumbleweed.geometry.tango_keypoints import (
    BODY_EDGES,
    NUM_KEYPOINTS,
    TANGO_KEYPOINTS_BODY,
)

# SPEED+ camera.json, shared by every domain.
CAMERA = CameraIntrinsics(
    camera_matrix=np.array(
        [[2988.5795163815555, 0.0, 960.0], [0.0, 2988.3401159176124, 600.0], [0.0, 0.0, 1.0]]
    ),
    dist_coeffs=np.array(
        [
            -0.22383016606510672,
            0.5140979708910638,
            -0.0006649961199834066,
            -0.00021404771667484594,
            -0.13124227429077406,
        ]
    ),
    width=1920,
    height=1200,
)

# synthetic/train.json entry for img014999.jpg.
Q_GT = np.array([0.25508, -0.130871, -0.449804, -0.845863])
T_GT = np.array([-0.028673, 0.33117, 6.723207])

# Verified visually: this wireframe wraps the spacecraft body and points 8/10
# land on the antenna boom tips. Regenerating these requires re-running
# scripts/sanity_check_keypoints.py and re-confirming by eye.
EXPECTED_PX = np.array(
    [
        [990.550, 1010.448],
        [1168.290, 827.368],
        [907.491, 722.954],
        [718.399, 888.976],
        [1023.409, 866.767],
        [1157.311, 734.839],
        [888.095, 630.983],
        [745.205, 750.276],
        [1255.594, 807.284],
        [873.244, 659.510],
        [687.869, 921.379],
    ]
)


def test_keypoint_model_shape():
    assert TANGO_KEYPOINTS_BODY.shape == (NUM_KEYPOINTS, 3)
    assert TANGO_KEYPOINTS_BODY.dtype == np.float64


def test_body_edges_index_the_corner_points():
    assert max(max(e) for e in BODY_EDGES) < 8


def test_keypoints_span_plausible_spacecraft_extent():
    extent = TANGO_KEYPOINTS_BODY.max(axis=0) - TANGO_KEYPOINTS_BODY.min(axis=0)
    assert np.all(extent > 0.3) and np.all(extent < 1.5)


def test_reprojection_matches_verified_pixels():
    projected = project_points(TANGO_KEYPOINTS_BODY, Q_GT, T_GT, CAMERA)
    assert np.allclose(projected, EXPECTED_PX, atol=1e-2)


def test_reprojection_lands_inside_the_image():
    projected = project_points(TANGO_KEYPOINTS_BODY, Q_GT, T_GT, CAMERA)
    assert np.all(projected[:, 0] > 0) and np.all(projected[:, 0] < CAMERA.width)
    assert np.all(projected[:, 1] > 0) and np.all(projected[:, 1] < CAMERA.height)
