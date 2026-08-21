"""Unit tests for SimToolReal reward computation (T3).

Validates the seven reward terms, d* delta-progress, lifting latch, and the
no-global-scaling contract (audit fix P0-1). These are pure-function tests;
they do not require a running env or Isaac Sim.

Acceptance criteria from MIGRATION_02 T3:
  - No global scaling (direct sum returned)
  - Action penalties = joint velocity L1, negative sign
  - lifted latch (prev_lifted=True → stays True)
  - Once lifted, lift_rew==0 and keypoint_rew activates
  - d* delta-progress (only reward improvements, d*=min update)
  - Reach bonus amortized to 100/step by default
  - env._reward_terms populated with 8 keys
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from unilab.envs.manipulation.simtoolreal.rewards import (
    action_penalty,
    compute_rewards,
    distance_delta_reward,
    keypoint_reward,
    lifting_reward,
    reach_goal_bonus,
    update_near_goal_steps,
)

if TYPE_CHECKING:
    pass


class TestLiftingReward:
    """Test lifting_reward (progress + bonus + latch)."""

    def test_z_lift_offset(self) -> None:
        """Verify the +0.05 offset in z_lift (source :17)."""
        object_z = np.array([0.5, 0.55], dtype=np.float32)
        object_init_z = np.array([0.4, 0.4], dtype=np.float32)
        prev_lifted = np.array([False, False], dtype=bool)

        lift_rew, _, lifted = lifting_reward(
            object_z=object_z,
            object_init_z=object_init_z,
            prev_lifted=prev_lifted,
            lifting_bonus_threshold=0.15,
            lifting_bonus=300.0,
            lifting_rew_scale=20.0,
        )

        # z_lift = 0.05 + (0.5 - 0.4) = 0.15, z_lift = 0.05 + (0.55 - 0.4) = 0.20
        # Env 0: z_lift=0.15 not > 0.15 → lifted=False → lift_rew = 0.15 * 20 = 3.0
        # Env 1: z_lift=0.20 > 0.15 → lifted=True → lift_rew = 0 (zeroed by :22)
        # This tests the offset AND the immediate zeroing once threshold is crossed
        expected = np.array([3.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(lift_rew, expected, rtol=1e-6)
        assert lifted[0] is np.False_
        assert lifted[1] is np.True_

    def test_lifting_latch(self) -> None:
        """Once lifted, the latch stays True (source :19, boolean OR)."""
        object_z = np.array([0.5, 0.3], dtype=np.float32)
        object_init_z = np.array([0.4, 0.4], dtype=np.float32)
        prev_lifted = np.array([True, False], dtype=bool)

        _, _, lifted = lifting_reward(
            object_z=object_z,
            object_init_z=object_init_z,
            prev_lifted=prev_lifted,
            lifting_bonus_threshold=0.15,
            lifting_bonus=300.0,
            lifting_rew_scale=20.0,
        )

        # Env 0: prev_lifted=True → stays True even if z_lift drops
        # Env 1: z_lift = 0.05 + (0.3-0.4) = -0.05 < 0.15 → still False
        assert lifted[0] is np.True_
        assert lifted[1] is np.False_

    def test_one_shot_bonus(self) -> None:
        """Bonus fires only on the crossing frame (source :20)."""
        object_z = np.array([0.6, 0.6], dtype=np.float32)
        object_init_z = np.array([0.4, 0.4], dtype=np.float32)

        # First call: cross threshold
        _, bonus1, lifted1 = lifting_reward(
            object_z=object_z,
            object_init_z=object_init_z,
            prev_lifted=np.array([False, False], dtype=bool),
            lifting_bonus_threshold=0.15,
            lifting_bonus=300.0,
            lifting_rew_scale=20.0,
        )
        assert bonus1[0] == 300.0
        assert lifted1[0] is np.True_

        # Second call: already lifted
        _, bonus2, lifted2 = lifting_reward(
            object_z=object_z,
            object_init_z=object_init_z,
            prev_lifted=lifted1,
            lifting_bonus_threshold=0.15,
            lifting_bonus=300.0,
            lifting_rew_scale=20.0,
        )
        assert bonus2[0] == 0.0
        assert lifted2[0] is np.True_

    def test_lift_rew_zeroes_after_lifted(self) -> None:
        """Once lifted, lift_rew becomes zero (source :22, *(~lifted))."""
        # Use object_z below threshold for "before lift" case
        object_z_before = np.array(
            [0.5], dtype=np.float32
        )  # z_lift = 0.05 + 0.1 = 0.15 (not > 0.15)
        object_z_after = np.array([0.6], dtype=np.float32)  # z_lift = 0.05 + 0.2 = 0.25 (> 0.15)
        object_init_z = np.array([0.4], dtype=np.float32)

        # Before lift (z_lift = 0.15, exactly at threshold but not > threshold)
        lift_rew1, _, lifted1 = lifting_reward(
            object_z=object_z_before,
            object_init_z=object_init_z,
            prev_lifted=np.array([False], dtype=bool),
            lifting_bonus_threshold=0.15,
            lifting_bonus=300.0,
            lifting_rew_scale=20.0,
        )
        assert lift_rew1[0] > 0.0  # Should get reward
        assert lifted1[0] is np.False_  # Not lifted yet (0.15 is not > 0.15)

        # After lift (z_lift = 0.25 > 0.15, crosses threshold)
        lift_rew2, _, lifted2 = lifting_reward(
            object_z=object_z_after,
            object_init_z=object_init_z,
            prev_lifted=np.array([False], dtype=bool),
            lifting_bonus_threshold=0.15,
            lifting_bonus=300.0,
            lifting_rew_scale=20.0,
        )
        assert lift_rew2[0] == 0.0  # Immediately zeroed when lifted
        assert lifted2[0] is np.True_  # Now lifted

        # Also test with prev_lifted=True to confirm latch
        lift_rew3, _, lifted3 = lifting_reward(
            object_z=object_z_after,
            object_init_z=object_init_z,
            prev_lifted=np.array([True], dtype=bool),
            lifting_bonus_threshold=0.15,
            lifting_bonus=300.0,
            lifting_rew_scale=20.0,
        )
        assert lift_rew3[0] == 0.0
        assert lifted3[0] is np.True_


class TestDistanceDeltaReward:
    """Test fingertip d* delta-progress."""

    def test_delta_progress_only_rewards_improvement(self) -> None:
        """Only reward when current < d* (source :33)."""
        curr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        closest = np.array([[1.5, 2.0, 4.0]], dtype=np.float32)
        lifted = np.array([False], dtype=bool)

        rew, new_closest = distance_delta_reward(
            curr_fingertip_dist=curr,
            closest_fingertip_dist=closest,
            lifted=lifted,
            rew_scale=50.0,
        )

        # Deltas: (1.5-1.0, 2.0-2.0, 4.0-3.0) = (0.5, 0, 1.0)
        # Sum = 1.5, scaled by 50.0
        assert rew[0] == pytest.approx(1.5 * 50.0)
        np.testing.assert_allclose(new_closest, [[1.0, 2.0, 3.0]], rtol=1e-6)

    def test_delta_clamped_to_zero(self) -> None:
        """Negative deltas (worse than d*) clamp to zero (source :35)."""
        curr = np.array([[2.0, 3.0]], dtype=np.float32)
        closest = np.array([[1.0, 2.0]], dtype=np.float32)
        lifted = np.array([False], dtype=bool)

        rew, new_closest = distance_delta_reward(
            curr_fingertip_dist=curr,
            closest_fingertip_dist=closest,
            lifted=lifted,
            rew_scale=50.0,
        )

        # Deltas: (1.0-2.0, 2.0-3.0) = (-1.0, -1.0), clamped to 0
        assert rew[0] == 0.0
        # d* stays at the best ever seen
        np.testing.assert_allclose(new_closest, [[1.0, 2.0]], rtol=1e-6)

    def test_gated_on_not_lifted(self) -> None:
        """Reward is zero when lifted (source :36, *(~lifted))."""
        curr = np.array([[1.0]], dtype=np.float32)
        closest = np.array([[2.0]], dtype=np.float32)
        lifted = np.array([True], dtype=bool)

        rew, _ = distance_delta_reward(
            curr_fingertip_dist=curr,
            closest_fingertip_dist=closest,
            lifted=lifted,
            rew_scale=50.0,
        )

        assert rew[0] == 0.0


class TestKeypointReward:
    """Test keypoint d* delta-progress."""

    def test_delta_progress_keypoint(self) -> None:
        """Only reward when current < d* (source :47)."""
        curr = np.array([0.1, 0.2], dtype=np.float32)
        closest = np.array([0.15, 0.2], dtype=np.float32)
        lifted = np.array([True, True], dtype=bool)

        rew, new_closest = keypoint_reward(
            keypoints_max_dist=curr,
            closest_keypoint_max_dist=closest,
            lifted=lifted,
            rew_scale=200.0,
        )

        # Deltas: (0.15-0.1, 0.2-0.2) = (0.05, 0), scaled by 200.0
        expected_rew = np.array([0.05 * 200.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(rew, expected_rew, rtol=1e-6)
        np.testing.assert_allclose(new_closest, [0.1, 0.2], rtol=1e-6)

    def test_gated_on_lifted(self) -> None:
        """Reward is zero when not lifted (source :50, *lifted)."""
        curr = np.array([0.1], dtype=np.float32)
        closest = np.array([0.2], dtype=np.float32)
        lifted = np.array([False], dtype=bool)

        rew, _ = keypoint_reward(
            keypoints_max_dist=curr,
            closest_keypoint_max_dist=closest,
            lifted=lifted,
            rew_scale=200.0,
        )

        assert rew[0] == 0.0


class TestActionPenalty:
    """Test joint velocity L1 penalties."""

    def test_penalty_is_joint_velocity_l1(self) -> None:
        """Penalty = L1 norm of joint velocity, not action (source :62-63)."""
        joint_vel = np.array(
            [
                # Env 0: arm [1, -2, 3, -4, 5, -6, 7], hand [0.1]*22
                [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0] + [0.1] * 22,
            ],
            dtype=np.float32,
        )
        arm_slice = slice(0, 7)
        hand_slice = slice(7, 29)

        kuka_pen, hand_pen = action_penalty(
            joint_vel=joint_vel,
            arm_slice=arm_slice,
            hand_slice=hand_slice,
            kuka_scale=0.03,  # Positive, negated inside function
            hand_scale=0.003,  # Positive, negated inside function
        )

        # Arm L1 = sum(abs([1, -2, 3, -4, 5, -6, 7])) = 28
        # Hand L1 = sum(abs([0.1]*22)) = 2.2
        # Source :62-63 negates inside: -scale * L1
        expected_kuka = -0.03 * 28.0
        expected_hand = -0.003 * 2.2
        assert kuka_pen[0] == pytest.approx(expected_kuka)
        assert hand_pen[0] == pytest.approx(expected_hand)

    def test_penalty_is_negative(self) -> None:
        """Penalties are negative (function negates the positive config scale)."""
        joint_vel = np.array([[1.0] * 29], dtype=np.float32)
        arm_slice = slice(0, 7)
        hand_slice = slice(7, 29)

        kuka_pen, hand_pen = action_penalty(
            joint_vel=joint_vel,
            arm_slice=arm_slice,
            hand_slice=hand_slice,
            kuka_scale=0.03,  # Positive config value
            hand_scale=0.003,  # Positive config value
        )

        assert kuka_pen[0] < 0.0
        assert hand_pen[0] < 0.0


class TestReachGoalBonus:
    """Test reach-goal bonus (amortized vs lump-sum)."""

    def test_amortized_by_default(self) -> None:
        """Default: amortize as near_goal * (1000 / 10) = 100/step (source :89)."""
        near_goal = np.array([True, False], dtype=bool)
        is_success = np.array([False, False], dtype=bool)

        bonus = reach_goal_bonus(
            near_goal=near_goal,
            is_success=is_success,
            reach_goal_bonus_value=1000.0,
            success_steps=10,
            force_consecutive=False,
        )

        # Env 0: near_goal=True → 1000/10 = 100
        # Env 1: near_goal=False → 0
        expected = np.array([100.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(bonus, expected, rtol=1e-6)

    def test_lump_sum_on_success(self) -> None:
        """When force_consecutive=True, give lump sum on success (source :88)."""
        near_goal = np.array([True, True], dtype=bool)
        is_success = np.array([True, False], dtype=bool)

        bonus = reach_goal_bonus(
            near_goal=near_goal,
            is_success=is_success,
            reach_goal_bonus_value=1000.0,
            success_steps=10,
            force_consecutive=True,
        )

        # Env 0: is_success=True → 1000
        # Env 1: is_success=False → 0
        expected = np.array([1000.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(bonus, expected, rtol=1e-6)


class TestUpdateNearGoalSteps:
    """Test near-goal step counter."""

    def test_cumulative_by_default(self) -> None:
        """Default: cumulative counter (source :76)."""
        near_goal = np.array([True, False, True], dtype=bool)
        near_goal_steps = np.array([5, 3, 0], dtype=np.int32)

        updated = update_near_goal_steps(
            near_goal=near_goal,
            near_goal_steps=near_goal_steps,
            force_consecutive=False,
        )

        # Cumulative: [5+1, 3+0, 0+1] = [6, 3, 1]
        expected = np.array([6, 3, 1], dtype=np.int32)
        np.testing.assert_array_equal(updated, expected)

    def test_consecutive_resets_on_miss(self) -> None:
        """When force_consecutive=True, reset to zero if not near (source :75)."""
        near_goal = np.array([True, False, True], dtype=bool)
        near_goal_steps = np.array([5, 3, 0], dtype=np.int32)

        updated = update_near_goal_steps(
            near_goal=near_goal,
            near_goal_steps=near_goal_steps,
            force_consecutive=True,
        )

        # Consecutive: [(5+1)*1, (3+0)*0, (0+1)*1] = [6, 0, 1]
        expected = np.array([6, 0, 1], dtype=np.int32)
        np.testing.assert_array_equal(updated, expected)


class TestComputeRewards:
    """Test the full compute_rewards function (integration)."""

    def test_no_global_scaling(self) -> None:
        """Reward is the direct sum, no ctrl_dt or 0.01 multiplier (P0-1 fix)."""
        env = self._make_mock_env(num_envs=1)
        info = self._make_mock_info(num_envs=1)

        # Set up non-zero contributions from each term
        env._object_pos = np.array([[0.0, 0.0, 0.6]], dtype=np.float32)
        info["object_init_z"] = np.array([0.4], dtype=np.float32)
        info["lifted_object"] = np.array([False], dtype=bool)
        env._curr_fingertip_distances = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        info["closest_fingertip_dist"] = np.array([[1.5, 1.5, 1.5, 1.5, 1.5]], dtype=np.float32)
        env._keypoints_max_dist = np.array([0.1], dtype=np.float32)
        info["closest_keypoint_max_dist"] = np.array([0.2], dtype=np.float32)
        env._joint_vel = np.ones((1, 29), dtype=np.float32)
        env._near_goal = np.array([True], dtype=bool)
        env._is_success = np.array([False], dtype=bool)

        reward = compute_rewards(env, info)

        # Verify it's the sum of the terms in env._reward_terms
        assert hasattr(env, "_reward_terms")
        terms = env._reward_terms
        manual_sum = (
            terms["lifting_rew"][0]
            + terms["lift_bonus_rew"][0]
            + terms["fingertip_delta_rew"][0]
            + terms["keypoint_rew"][0]
            + terms["kuka_actions_penalty"][0]
            + terms["hand_actions_penalty"][0]
            + terms["bonus_rew"][0]
        )
        assert reward[0] == pytest.approx(manual_sum)
        assert terms["total_reward"][0] == pytest.approx(manual_sum)

    def test_reward_terms_populated(self) -> None:
        """env._reward_terms must have 8 keys (source :142-151)."""
        env = self._make_mock_env(num_envs=2)
        info = self._make_mock_info(num_envs=2)

        compute_rewards(env, info)

        assert hasattr(env, "_reward_terms")
        expected_keys = {
            "fingertip_delta_rew",
            "lifting_rew",
            "lift_bonus_rew",
            "keypoint_rew",
            "kuka_actions_penalty",
            "hand_actions_penalty",
            "bonus_rew",
            "total_reward",
        }
        assert set(env._reward_terms.keys()) == expected_keys
        for key, val in env._reward_terms.items():
            assert val.shape == (2,), f"Key {key} has wrong shape"

    def test_phase_gating(self) -> None:
        """Before lift: fingertip+lift reward. After lift: keypoint reward."""
        # Before lift
        env = self._make_mock_env(num_envs=1)
        info = self._make_mock_info(num_envs=1)
        info["lifted_object"] = np.array([False], dtype=bool)
        env._curr_fingertip_distances = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        info["closest_fingertip_dist"] = np.array([[2.0, 2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
        env._keypoints_max_dist = np.array([0.1], dtype=np.float32)
        info["closest_keypoint_max_dist"] = np.array([0.2], dtype=np.float32)

        compute_rewards(env, info)
        assert env._reward_terms["fingertip_delta_rew"][0] > 0.0
        assert env._reward_terms["lifting_rew"][0] > 0.0
        assert env._reward_terms["keypoint_rew"][0] == 0.0

        # After lift
        env2 = self._make_mock_env(num_envs=1)
        info2 = self._make_mock_info(num_envs=1)
        info2["lifted_object"] = np.array([True], dtype=bool)
        env2._curr_fingertip_distances = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        info2["closest_fingertip_dist"] = np.array([[2.0, 2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
        env2._keypoints_max_dist = np.array([0.1], dtype=np.float32)
        info2["closest_keypoint_max_dist"] = np.array([0.2], dtype=np.float32)

        compute_rewards(env2, info2)
        assert env2._reward_terms["fingertip_delta_rew"][0] == 0.0
        assert env2._reward_terms["lifting_rew"][0] == 0.0
        assert env2._reward_terms["keypoint_rew"][0] > 0.0

    def test_d_star_updates_in_place(self) -> None:
        """d* trackers in info are updated by the reward function."""
        env = self._make_mock_env(num_envs=1)
        info = self._make_mock_info(num_envs=1)
        info["closest_fingertip_dist"] = np.array([[2.0, 2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
        info["closest_keypoint_max_dist"] = np.array([0.5], dtype=np.float32)
        env._curr_fingertip_distances = np.array([[1.0, 1.5, 2.5, 1.0, 1.0]], dtype=np.float32)
        env._keypoints_max_dist = np.array([0.3], dtype=np.float32)

        compute_rewards(env, info)

        # d* should be updated to min(old, current)
        expected_ft = np.array([[1.0, 1.5, 2.0, 1.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(info["closest_fingertip_dist"], expected_ft, rtol=1e-6)
        assert info["closest_keypoint_max_dist"][0] == pytest.approx(0.3)

    @staticmethod
    def _make_mock_env(num_envs: int):
        """Create a minimal mock env with the required attributes."""
        env = MagicMock()
        env.cfg.reward_config.lifting_bonus_threshold = 0.15
        env.cfg.reward_config.lifting_bonus = 300.0
        env.cfg.reward_config.lifting_rew_scale = 20.0
        env.cfg.reward_config.distance_delta_rew_scale = 50.0
        env.cfg.reward_config.keypoint_rew_scale = 200.0
        env.cfg.reward_config.kuka_actions_penalty_scale = 0.03  # Positive, negated in function
        env.cfg.reward_config.hand_actions_penalty_scale = 0.003  # Positive, negated in function
        env.cfg.reward_config.reach_goal_bonus = 1000.0
        # success_steps is on GoalCfg, not TerminationCfg (contract §5.0). The
        # source keeps it on TerminationCfg (cfg:437), but this port regroups
        # goal-side fields. Pinning it on termination let compute_rewards read a
        # bare MagicMock, which numpy broadcasts as shape (0,) — the bug T7 hit.
        env.cfg.goal.success_steps = 10
        env.cfg.termination.force_consecutive_near_goal_steps = False

        env._object_pos = np.zeros((num_envs, 3), dtype=np.float32)
        env._curr_fingertip_distances = np.zeros((num_envs, 5), dtype=np.float32)
        env._keypoints_max_dist = np.zeros((num_envs,), dtype=np.float32)
        env._joint_vel = np.zeros((num_envs, 29), dtype=np.float32)
        env._arm_slice = slice(0, 7)
        env._hand_slice = slice(7, 29)
        env._near_goal = np.zeros((num_envs,), dtype=bool)
        env._is_success = np.zeros((num_envs,), dtype=bool)
        return env

    @staticmethod
    def _make_mock_info(num_envs: int) -> dict[str, np.ndarray]:
        """Create a minimal mock info dict with d* trackers."""
        return {
            "object_init_z": np.zeros((num_envs,), dtype=np.float32),
            "lifted_object": np.zeros((num_envs,), dtype=bool),
            "closest_fingertip_dist": np.zeros((num_envs, 5), dtype=np.float32),
            "closest_keypoint_max_dist": np.zeros((num_envs,), dtype=np.float32),
        }
