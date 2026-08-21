from __future__ import annotations

import numpy as np

from unilab.envs.manipulation.simtoolreal.keypoints import (
    compute_keypoints,
    keypoint_max_dist,
    observation_keypoint_size,
)


def test_keypoints_use_metre_scale_without_phi_25x_trap() -> None:
    pos = np.zeros((1, 3), dtype=np.float32)
    quat = np.array([[1, 0, 0, 0]], dtype=np.float32)
    size = observation_keypoint_size(np.array([[1, 2, 3]], dtype=np.float32), 0.04)
    points = compute_keypoints(pos, quat, size)
    assert points.shape == (1, 4, 3)
    assert float(np.max(np.abs(points))) < 0.1


def test_keypoint_distance_is_max_over_fixed_corner_order() -> None:
    obj = np.zeros((1, 4, 3), dtype=np.float32)
    goal = obj.copy()
    goal[0, 2, 1] = 0.25
    np.testing.assert_allclose(keypoint_max_dist(obj, goal), [0.25])
