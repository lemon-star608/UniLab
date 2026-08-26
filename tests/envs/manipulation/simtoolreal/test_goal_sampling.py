from __future__ import annotations

import numpy as np

from unilab.envs.manipulation.simtoolreal.goal_sampling import (
    sample_absolute_goal,
    sample_delta_goal,
)


def test_absolute_goal_respects_scaled_workspace_and_wxyz(monkeypatch) -> None:
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.full(shape, 0.5, dtype=np.float32))
    pos, quat = sample_absolute_goal((-1, -2, 0), (1, 2, 2), 0.5, 1)
    np.testing.assert_allclose(pos, [[0.0, 0.0, 1.0]])
    assert quat.shape == (1, 4)
    np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1.0, rtol=1e-6)


def test_delta_goal_clamps_position_and_composes_orientation(monkeypatch) -> None:
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.ones(shape, dtype=np.float32))
    monkeypatch.setattr(
        np.random,
        "randn",
        lambda *shape: np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (shape[0], 1)),
    )
    pos, quat = sample_delta_goal(
        np.array([[0.9, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        1.0,
        90.0,
        (-1, -1, -1),
        (1, 1, 1),
        1.0,
    )
    np.testing.assert_allclose(pos, [[1.0, 1.0, 1.0]])
    np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1.0, rtol=1e-6)
