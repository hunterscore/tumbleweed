"""3D keypoint model of the Tango spacecraft (11 points, Tango body frame, meters).

Source: ``src/utils/tangoPoints.mat`` in tpark94/speedplusbaseline (MIT
license), the SPEED+ author's baseline repo; the 11-point scheme follows the
KRN paper (Park, Sharma & D'Amico, arXiv:1909.00392).

Index groups: 0-3 and 4-7 are body/panel corners, 8-10 are antenna tips.
The frame is the Tango body frame used by ``q_vbs2tango`` / ``project_points``.
"""
from __future__ import annotations

import numpy as np

NUM_KEYPOINTS = 11

TANGO_KEYPOINTS_BODY = np.array(
    [
        [-0.37, -0.385, 0.3215],
        [-0.37, 0.385, 0.3215],
        [0.37, 0.385, 0.3215],
        [0.37, -0.385, 0.3215],
        [-0.37, -0.264, 0.0],
        [-0.37, 0.304, 0.0],
        [0.37, 0.304, 0.0],
        [0.37, -0.264, 0.0],
        [-0.5427, 0.4877, 0.2535],
        [0.5427, 0.4877, 0.2591],
        [0.305, -0.579, 0.2515],
    ],
    dtype=np.float64,
)

BODY_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
