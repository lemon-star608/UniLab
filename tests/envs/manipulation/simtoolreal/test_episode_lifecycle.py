"""Unit tests for SimToolReal episode lifecycle functions."""

import numpy as np
import pytest

from unilab.envs.manipulation.simtoolreal.config import GoalCfg, RewardCfg, TerminationCfg
from unilab.envs.manipulation.simtoolreal.episode_lifecycle import (
    advance_goal_on_success,
    compute_success,
    compute_terminations,
    update_tolerance_curriculum,
)


class MockEnv:
    """Minimal env mock for lifecycle tests."""

    def __init__(self, num_envs=4):
        self._num_envs = num_envs
        self._cfg = type(
            "Cfg",
            (),
            {
                "reward": RewardCfg(),
                "goal": GoalCfg(),
                "termination": TerminationCfg(),
            },
        )()
        self._cfg.reset = type(
            "ResetCfg",
            (),
            {
                "fixed_goal_pose": None,
                "fixed_trajectory_file": "",
                "goal_sampling_type": self._cfg.goal.goal_sampling_type,
            },
        )()
        self._current_success_tolerance = 0.075
        self._frame_counter = 0
        self._last_curriculum_update = 0
        self._autoreset_envs = np.zeros(num_envs, dtype=bool)

        # Minimal state.info.
        self._state = type(
            "State",
            (),
            {
                "info": {
                    "successes": np.zeros(num_envs, dtype=np.int32),
                    "goal_pos": np.zeros((num_envs, 3), dtype=np.float32),
                    "goal_quat": np.tile(
                        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_envs, 1)
                    ),
                    "closest_keypoint_max_dist": np.full(num_envs, -1.0, dtype=np.float32),
                    "closest_fingertip_dist": np.full((num_envs, 5), -1.0, dtype=np.float32),
                    "near_goal_steps": np.zeros(num_envs, dtype=np.int32),
                    "steps": np.zeros(num_envs, dtype=np.int32),
                    "prev_episode_successes": np.zeros(num_envs, dtype=np.int32),
                },
                "terminated": np.zeros(num_envs, dtype=bool),
            },
        )()

    def get_object_pos(self):
        return np.ones((self._num_envs, 3), dtype=np.float32) * 0.5


def test_compute_success_threshold_with_keypoint_scale():
    """Success threshold is tolerance * keypoint_scale (★ easy-to-miss item)."""
    env = MockEnv(num_envs=4)
    env._cfg.goal.success_steps = 1
    env._cfg.goal.keypoint_scale = 1.5
    env._current_success_tolerance = 0.075

    # Effective threshold is 0.075 * 1.5 = 0.1125 m.
    # Distance 0.112 should NOT succeed, 0.113 should NOT, 0.1125 is boundary.
    keypoints_max_dist = np.array([0.110, 0.1125, 0.113, 0.120], dtype=np.float32)

    is_success = compute_success(env, keypoints_max_dist)

    # 0.110 and 0.1125 succeed (≤ threshold), 0.113 and 0.120 fail.
    assert np.array_equal(is_success, [True, True, False, False])


def test_compute_success_cumulative_steps():
    """Cumulative mode counts up when near, holds when far."""
    env = MockEnv(num_envs=2)
    env._cfg.goal.success_steps = 3
    env._cfg.termination.force_consecutive_near_goal_steps = False
    env._state.info["near_goal_steps"][:] = [1, 2]

    # env 0: near → accumulates to 2.
    # env 1: far → holds at 2.
    keypoints_max_dist = np.array([0.05, 0.20], dtype=np.float32)
    is_success = compute_success(env, keypoints_max_dist)

    assert np.array_equal(env._state.info["near_goal_steps"], [2, 2])
    assert np.array_equal(is_success, [False, False])

    # Next step, both near.
    keypoints_max_dist = np.array([0.05, 0.05], dtype=np.float32)
    is_success = compute_success(env, keypoints_max_dist)

    assert np.array_equal(env._state.info["near_goal_steps"], [3, 3])
    assert np.array_equal(is_success, [True, True])


def test_compute_success_consecutive_steps():
    """Consecutive mode resets counter when far."""
    env = MockEnv(num_envs=2)
    env._cfg.goal.success_steps = 3
    env._cfg.termination.force_consecutive_near_goal_steps = True
    env._state.info["near_goal_steps"][:] = [2, 1]

    # env 0: near → accumulates to 3.
    # env 1: far → resets to 0.
    keypoints_max_dist = np.array([0.05, 0.20], dtype=np.float32)
    is_success = compute_success(env, keypoints_max_dist)

    assert np.array_equal(env._state.info["near_goal_steps"], [3, 0])
    assert np.array_equal(is_success, [True, False])


def test_advance_goal_on_success_resets_trackers():
    """Successful envs reset d*, near_goal_steps, and steps."""
    env = MockEnv(num_envs=4)
    env._state.info["successes"][:] = [0, 1, 2, 3]
    env._state.info["near_goal_steps"][:] = 10
    env._state.info["steps"][:] = 100
    env._state.info["closest_keypoint_max_dist"][:] = 0.05

    is_success = np.array([False, True, False, True], dtype=bool)

    advance_goal_on_success(env, is_success)

    # Successes incremented for env 1 and 3.
    assert np.array_equal(env._state.info["successes"], [0, 2, 2, 4])

    # d* and near_goal_steps reset for successful envs.
    expected_dist = np.array([0.05, -1.0, 0.05, -1.0], dtype=np.float32)
    assert np.allclose(env._state.info["closest_keypoint_max_dist"], expected_dist)
    assert np.array_equal(env._state.info["near_goal_steps"], [10, 0, 10, 0])

    # ★ steps zeroed for successful envs (D2 mechanism).
    assert np.array_equal(env._state.info["steps"], [100, 0, 100, 0])

    # Goal positions changed (can't check exact values, just that they differ).
    assert not np.allclose(env._state.info["goal_pos"][1], 0.0)
    assert not np.allclose(env._state.info["goal_pos"][3], 0.0)


def test_advance_goal_on_success_honors_absolute_sampling(monkeypatch: pytest.MonkeyPatch):
    """Success advance dispatches to absolute sampling when configured."""
    env = MockEnv(num_envs=1)
    env._cfg.goal.goal_sampling_type = "absolute"
    env._cfg.reset.goal_sampling_type = "absolute"

    expected_pos = np.array([[0.2, 0.1, 0.8]], dtype=np.float32)
    expected_quat = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    def sample_absolute_goal(*, mins, maxs, scale, n):
        assert n == 1
        return expected_pos.copy(), expected_quat.copy()

    monkeypatch.setattr(
        "unilab.envs.manipulation.simtoolreal.episode_lifecycle.sample_absolute_goal",
        sample_absolute_goal,
        raising=False,
    )

    advance_goal_on_success(env, np.array([True], dtype=bool))

    np.testing.assert_array_equal(env._state.info["goal_pos"], expected_pos)
    np.testing.assert_array_equal(env._state.info["goal_quat"], expected_quat)


def test_advance_goal_on_success_fixed_pose_has_priority(monkeypatch: pytest.MonkeyPatch):
    """A fixed goal pose overrides both trajectory and sampling modes."""
    env = MockEnv(num_envs=1)
    env._cfg.goal.goal_sampling_type = "absolute"
    env._cfg.reset.goal_sampling_type = "absolute"
    env._cfg.reset.fixed_goal_pose = (0.3, -0.1, 0.7, 1.0, 0.0, 0.0, 0.0)
    env._cfg.reset.fixed_trajectory_file = "ignored.json"

    monkeypatch.setattr(
        "unilab.envs.manipulation.simtoolreal.episode_lifecycle.sample_absolute_goal",
        lambda **_: (_ for _ in ()).throw(AssertionError("sampler must not run")),
        raising=False,
    )

    advance_goal_on_success(env, np.array([True], dtype=bool))

    np.testing.assert_allclose(env._state.info["goal_pos"], [[0.3, -0.1, 0.7]])
    np.testing.assert_allclose(env._state.info["goal_quat"], [[1.0, 0.0, 0.0, 0.0]])


def test_advance_goal_on_success_advances_fixed_trajectory() -> None:
    """Fixed trajectories advance one waypoint without a physical reset."""
    env = MockEnv(num_envs=1)
    env._cfg.reset.fixed_trajectory_file = "trajectory.json"
    env._fixed_traj_pos = np.array([[[0.1, 0.2, 0.7], [0.2, 0.2, 0.7]]], dtype=np.float32)
    env._fixed_traj_quat = np.array(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], dtype=np.float32
    )
    env._traj_id = np.array([0], dtype=np.int64)
    env._traj_step = np.array([0], dtype=np.int64)
    env._state.info["near_goal_steps"][:] = 7
    env._state.info["steps"][:] = 42

    advance_goal_on_success(env, np.array([True], dtype=bool))

    np.testing.assert_allclose(env._state.info["goal_pos"], [[0.2, 0.2, 0.7]])
    np.testing.assert_allclose(env._state.info["goal_quat"], [[0.0, 1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(env._traj_step, [1])
    np.testing.assert_array_equal(env._state.info["near_goal_steps"], [0])
    np.testing.assert_array_equal(env._state.info["steps"], [0])


def test_compute_terminations_drop():
    """Object z < 0.1 triggers termination."""
    env = MockEnv(num_envs=4)
    # Mock get_object_pos to return specific z values.
    env.get_object_pos = lambda: np.array(
        [[0.0, 0.0, 0.2], [0.0, 0.0, 0.05], [0.0, 0.0, 0.1], [0.0, 0.0, 0.09]],
        dtype=np.float32,
    )
    env._curr_fingertip_distances = np.zeros((4, 5), dtype=np.float32)

    is_success = np.zeros(4, dtype=bool)
    terminated, _ = compute_terminations(env, is_success)

    # Only env 1 and 3 drop (z < 0.1).
    assert np.array_equal(terminated, [False, True, False, True])


def test_compute_terminations_hand_far():
    """Any fingertip distance > 1.5 m triggers termination."""
    env = MockEnv(num_envs=3)
    env._curr_fingertip_distances = np.array(
        [[0.5, 0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.6, 1.0, 1.0], [1.49, 1.49, 1.49, 1.49, 1.49]],
        dtype=np.float32,
    )

    is_success = np.zeros(3, dtype=bool)
    terminated, _ = compute_terminations(env, is_success)

    # Only env 1 has a fingertip > 1.5.
    assert np.array_equal(terminated, [False, True, False])


def test_compute_terminations_requires_fingertip_distance_cache():
    """Missing live geometry cache must fail instead of disabling hand_far."""
    env = MockEnv(num_envs=1)

    with pytest.raises(AttributeError, match="_curr_fingertip_distances"):
        compute_terminations(env, np.zeros(1, dtype=bool))


def test_compute_terminations_max_successes():
    """successes >= max_consecutive_successes triggers termination."""
    env = MockEnv(num_envs=4)
    env._cfg.termination.max_consecutive_successes = 3
    env._state.info["successes"][:] = [0, 2, 3, 5]
    env._curr_fingertip_distances = np.zeros((4, 5), dtype=np.float32)

    is_success = np.zeros(4, dtype=bool)
    terminated, _ = compute_terminations(env, is_success)

    # env 2 and 3 reached max_consecutive_successes.
    assert np.array_equal(terminated, [False, False, True, True])


def test_update_tolerance_curriculum_requires_initialized_counters():
    """Lifecycle counters are construction-time owner state, not lazy fallbacks."""
    env = MockEnv(num_envs=1)
    del env._frame_counter

    with pytest.raises(AttributeError, match="_frame_counter"):
        update_tolerance_curriculum(env)


def test_update_tolerance_curriculum_shrink():
    """Tolerance shrinks when mean prev_episode_successes >= threshold."""
    env = MockEnv(num_envs=4)
    env._cfg.termination.tolerance_curriculum_interval = 100
    env._cfg.termination.tolerance_curriculum_increment = 0.9
    env._cfg.termination.tolerance_curriculum_success_threshold = 3.0
    env._cfg.goal.success_tolerance = 0.075
    env._cfg.goal.target_success_tolerance = 0.01
    env._current_success_tolerance = 0.075
    env._frame_counter = 0
    env._last_curriculum_update = 0

    # Set prev_episode_successes high enough.
    env._state.info["prev_episode_successes"][:] = [3, 3, 4, 2]  # mean = 3.0

    # Advance to just before interval.
    for _ in range(99):
        update_tolerance_curriculum(env)

    assert env._current_success_tolerance == 0.075

    # Next step triggers curriculum.
    update_tolerance_curriculum(env)

    assert env._current_success_tolerance == pytest.approx(0.075 * 0.9)
    assert env._last_curriculum_update == 100


def test_update_tolerance_curriculum_floor():
    """Tolerance doesn't drop below target_success_tolerance."""
    env = MockEnv(num_envs=4)
    env._cfg.termination.tolerance_curriculum_interval = 1
    env._cfg.termination.tolerance_curriculum_increment = 0.5
    env._cfg.termination.tolerance_curriculum_success_threshold = 0.0
    env._cfg.goal.success_tolerance = 0.075
    env._cfg.goal.target_success_tolerance = 0.02
    env._current_success_tolerance = 0.03
    env._state.info["prev_episode_successes"][:] = 5

    update_tolerance_curriculum(env)

    # 0.03 * 0.5 = 0.015, but floor is 0.02.
    assert env._current_success_tolerance == 0.02


def test_update_tolerance_curriculum_no_shrink():
    """Tolerance doesn't shrink when mean < threshold."""
    env = MockEnv(num_envs=4)
    env._cfg.termination.tolerance_curriculum_interval = 1
    env._cfg.termination.tolerance_curriculum_success_threshold = 3.0
    env._current_success_tolerance = 0.075
    env._state.info["prev_episode_successes"][:] = [1, 1, 2, 2]  # mean = 1.5 < 3.0

    for _ in range(10):
        update_tolerance_curriculum(env)

    # Tolerance unchanged.
    assert env._current_success_tolerance == 0.075
