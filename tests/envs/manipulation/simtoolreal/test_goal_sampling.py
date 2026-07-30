"""Unit tests for T4 goal sampling functions."""

import numpy as np
import pytest

from unilab.envs.manipulation.simtoolreal.goal_sampling import (
    sample_absolute_goal,
    sample_delta_goal,
)


def test_sample_absolute_goal_shape():
    """Absolute sampling returns correct shapes."""
    mins = (-0.35, -0.2, 0.6)
    maxs = (0.35, 0.2, 0.95)
    scale = 1.0
    n = 10

    pos, quat = sample_absolute_goal(mins, maxs, scale, n)

    assert pos.shape == (n, 3)
    assert quat.shape == (n, 4)
    assert pos.dtype == np.float32
    assert quat.dtype == np.float32


def test_sample_absolute_goal_in_workspace():
    """Absolute sampling respects workspace bounds."""
    mins = np.array([-0.35, -0.2, 0.6], dtype=np.float32)
    maxs = np.array([0.35, 0.2, 0.95], dtype=np.float32)
    scale = 1.0
    n = 100

    pos, quat = sample_absolute_goal(mins, maxs, scale, n)

    # Position in workspace.
    assert np.all(pos >= mins)
    assert np.all(pos <= maxs)

    # Quaternion is unit.
    norms = np.linalg.norm(quat, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_sample_absolute_goal_scale():
    """Scale parameter shrinks the workspace."""
    mins = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
    maxs = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    scale = 0.5
    n = 100

    pos, _ = sample_absolute_goal(mins, maxs, scale, n)

    # With scale=0.5, the effective bounds are half the original.
    # Center is (0, 0, 0.5), half-extent is (0.5, 0.5, 0.25).
    assert np.all(pos >= [-0.5, -0.5, 0.25])
    assert np.all(pos <= [0.5, 0.5, 0.75])


def test_sample_delta_goal_shape():
    """Delta sampling returns correct shapes."""
    n = 10
    prev_pos = np.random.randn(n, 3).astype(np.float32)
    prev_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))

    delta_distance = 0.1
    delta_rotation_degrees = 90.0
    mins = (-0.35, -0.2, 0.6)
    maxs = (0.35, 0.2, 0.95)
    scale = 1.0

    pos, quat = sample_delta_goal(
        prev_pos, prev_quat, delta_distance, delta_rotation_degrees, mins, maxs, scale
    )

    assert pos.shape == (n, 3)
    assert quat.shape == (n, 4)
    assert pos.dtype == np.float32
    assert quat.dtype == np.float32


def test_sample_delta_goal_clamped():
    """Delta sampling clamps to workspace."""
    n = 10
    # Start at upper corner.
    prev_pos = np.tile(np.array([0.35, 0.2, 0.95], dtype=np.float32), (n, 1))
    prev_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))

    delta_distance = 0.1
    delta_rotation_degrees = 90.0
    mins = (-0.35, -0.2, 0.6)
    maxs = (0.35, 0.2, 0.95)
    scale = 1.0

    pos, quat = sample_delta_goal(
        prev_pos, prev_quat, delta_distance, delta_rotation_degrees, mins, maxs, scale
    )

    # Result clamped to workspace.
    assert np.all(pos >= np.array(mins, dtype=np.float32))
    assert np.all(pos <= np.array(maxs, dtype=np.float32))

    # Quaternion is unit.
    norms = np.linalg.norm(quat, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_sample_delta_goal_perturbation_bounded():
    """Delta sampling perturbs within delta_distance."""
    n = 50
    prev_pos = np.zeros((n, 3), dtype=np.float32)
    prev_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))

    delta_distance = 0.1
    delta_rotation_degrees = 90.0
    mins = (-1.0, -1.0, -1.0)
    maxs = (1.0, 1.0, 1.0)
    scale = 1.0

    pos, _ = sample_delta_goal(
        prev_pos, prev_quat, delta_distance, delta_rotation_degrees, mins, maxs, scale
    )

    # Distance from origin should be within delta_distance (with some margin for
    # uniform noise extremes).
    distances = np.linalg.norm(pos, axis=-1)
    assert np.all(distances <= delta_distance * np.sqrt(3) + 1e-5)
