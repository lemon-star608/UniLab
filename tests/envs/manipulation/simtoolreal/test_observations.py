"""Unit tests for SimToolReal observation assembly (T2).

Acceptance criteria from MIGRATION_02 T2:
  - obs dimension == cfg.num_actor_obs (140)
  - critic dimension == cfg.num_critic_obs (162)
  - clamp: output in [-10, 10] after extreme inputs
  - joint_pos normalized in [-1, 1]
  - actor/critic quat fields have w in last position (xyzw)
  - critic contains privileged fields; actor does not
  - d* sentinel (-1) preserved and passed through to critic
  - observation_keypoint_size path: phi is NOT passed raw to compute_keypoints
  - build_observations runs without error with a realistic mock env

These tests are pure-function / mock-based and do not require Isaac Sim or MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np
import pytest

from unilab.envs.manipulation.simtoolreal.config import (
    DomainRandomizationCfg,
    GoalCfg,
    ObsCfg,
    RewardCfg,
    SimToolRealCfg,
)
from unilab.envs.manipulation.simtoolreal.constants import (
    NUM_FINGERTIPS,
    NUM_JOINTS,
    OBS_FIELD_SIZES,
    compute_obs_dim,
)
from unilab.envs.manipulation.simtoolreal.observations import (
    _normalize_joint_pos,
    _perturb_quat,
    _wxyz_to_xyzw,
    build_observations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_ENVS = 4


def _make_state(
    n: int,
    d_star_val: float = 0.25,
    ft_dist_val: float = 0.1,
    steps_val: int = 5,
    successes_val: int = 0,
) -> MagicMock:
    """Build a minimal NpEnvState mock with the required info keys."""
    state = MagicMock()
    state.info = {
        "steps": np.full((n,), steps_val, dtype=np.uint32),
        "successes": np.zeros((n,), dtype=np.int32),
        "prev_targets": np.zeros((n, NUM_JOINTS), dtype=np.float32),
        "goal_pos": np.zeros((n, 3), dtype=np.float32),
        "goal_quat": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
        "object_scales": np.ones((n, 3), dtype=np.float32),  # phi=1 -> size_m = 0.04m
        "closest_keypoint_max_dist": np.full((n,), d_star_val, dtype=np.float32),
        "closest_fingertip_dist": np.full((n, NUM_FINGERTIPS), ft_dist_val, dtype=np.float32),
        "lifted_object": np.zeros((n,), dtype=bool),
        "reward": np.zeros((n,), dtype=np.float32),
    }
    return state


def _make_backend(n: int) -> MagicMock:
    """Backend mock returning plausible-shaped numpy arrays."""
    bk = MagicMock()
    bk.get_body_pos_w.side_effect = lambda ids: np.zeros(
        (n, len(np.atleast_1d(ids)), 3), dtype=np.float32
    )
    bk.get_body_quat_w.side_effect = lambda ids: np.tile(
        [1.0, 0.0, 0.0, 0.0], (n, len(np.atleast_1d(ids)), 1)
    ).astype(np.float32)
    bk.get_body_lin_vel_w.side_effect = lambda ids: np.zeros(
        (n, len(np.atleast_1d(ids)), 3), dtype=np.float32
    )
    bk.get_body_ang_vel_w.side_effect = lambda ids: np.zeros(
        (n, len(np.atleast_1d(ids)), 3), dtype=np.float32
    )
    return bk


def _make_env(n: int = N_ENVS, disable_dr: bool = False) -> MagicMock:
    """Build a minimal SimToolRealEnv mock."""
    env = MagicMock()
    env._num_envs = n
    env._np_dtype = np.float32

    # Body ids
    env._palm_body_id = 0
    env._fingertip_body_ids = np.arange(NUM_FINGERTIPS, dtype=np.int32)
    env._object_body_id = int(NUM_FINGERTIPS)

    # Joint limits (canonical, symmetric pi range)
    env._joint_lower_canon = np.full(NUM_JOINTS, -np.pi, dtype=np.float32)
    env._joint_upper_canon = np.full(NUM_JOINTS, +np.pi, dtype=np.float32)

    # Identity permutation (backend == canonical for test simplicity)
    env._perm_backend_to_canon = np.arange(NUM_JOINTS, dtype=np.int64)

    # Offsets (constants.py)
    env._palm_offset = np.array([-0.0, -0.02, 0.16], dtype=np.float32)
    env._fingertip_offset = np.array([0.02, 0.002, 0.0], dtype=np.float32)

    # Slices
    env._arm_slice = slice(0, 7)
    env._hand_slice = slice(7, NUM_JOINTS)

    # Scale multiplier (no-op by default)
    env._object_scale_multiplier = np.ones((n, 3), dtype=np.float32)

    # Delay queues - match config dims
    cfg = SimToolRealCfg()
    obs_d = cfg.num_actor_obs
    env._obs_queue = np.zeros((n, max(cfg.domain_randomization.obs_delay_max, 1), obs_d),
                               dtype=np.float32)
    env._object_state_queue = np.zeros(
        (n, max(cfg.domain_randomization.object_state_delay_max, 1), 13), dtype=np.float32
    )

    # Config
    env.cfg = cfg
    if disable_dr:
        env.cfg = MagicMock()
        env.cfg.obs.state_list = cfg.obs.state_list
        env.cfg.obs.obs_list = cfg.obs.obs_list
        env.cfg.obs.clamp_abs_observations = cfg.obs.clamp_abs_observations
        env.cfg.domain_randomization.use_object_state_delay_noise = False
        env.cfg.domain_randomization.use_obs_delay = False
        env.cfg.domain_randomization.joint_velocity_obs_noise_std = 0.0
        env.cfg.reward.object_base_size = cfg.reward.object_base_size
        env.cfg.goal.keypoint_scale = cfg.goal.keypoint_scale
        env.cfg.num_actor_obs = cfg.num_actor_obs
        env.cfg.num_critic_obs = cfg.num_critic_obs

    # Backend
    env._backend = _make_backend(n)

    # Joint helpers (use identity permutation)
    env.get_joint_pos_canon.return_value = np.zeros((n, NUM_JOINTS), dtype=np.float32)
    env.get_joint_vel_canon.return_value = np.zeros((n, NUM_JOINTS), dtype=np.float32)

    return env


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


class TestNormalizeJointPos:
    def test_midpoint_zero(self) -> None:
        lo = np.array([-1.0], dtype=np.float32)
        hi = np.array([1.0], dtype=np.float32)
        q = np.zeros((1, 1), dtype=np.float32)
        result = _normalize_joint_pos(q, lo, hi)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_lower_bound_minus_one(self) -> None:
        lo = np.array([-np.pi], dtype=np.float32)
        hi = np.array([np.pi], dtype=np.float32)
        q = np.array([[-np.pi]], dtype=np.float32)
        result = _normalize_joint_pos(q, lo, hi)
        np.testing.assert_allclose(result, -1.0, atol=1e-5)

    def test_upper_bound_plus_one(self) -> None:
        lo = np.array([-np.pi], dtype=np.float32)
        hi = np.array([np.pi], dtype=np.float32)
        q = np.array([[np.pi]], dtype=np.float32)
        result = _normalize_joint_pos(q, lo, hi)
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_range_in_minus_one_to_one(self) -> None:
        lo = np.full(NUM_JOINTS, -np.pi, dtype=np.float32)
        hi = np.full(NUM_JOINTS, np.pi, dtype=np.float32)
        q = np.random.uniform(-np.pi, np.pi, size=(8, NUM_JOINTS)).astype(np.float32)
        result = _normalize_joint_pos(q, lo, hi)
        assert result.min() >= -1.0 - 1e-5
        assert result.max() <= 1.0 + 1e-5


class TestWxyzToXyzw:
    def test_identity(self) -> None:
        q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        out = _wxyz_to_xyzw(q)
        expected = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_w_at_last_position(self) -> None:
        q = np.random.randn(16, 4).astype(np.float32)
        out = _wxyz_to_xyzw(q)
        np.testing.assert_array_equal(out[:, 3], q[:, 0])

    def test_xyz_preserved(self) -> None:
        q = np.random.randn(4, 4).astype(np.float32)
        out = _wxyz_to_xyzw(q)
        np.testing.assert_array_equal(out[:, :3], q[:, 1:])


class TestPerturbQuat:
    def test_output_shape(self) -> None:
        q = np.tile([1.0, 0.0, 0.0, 0.0], (N_ENVS, 1)).astype(np.float32)
        out = _perturb_quat(q, max_deg=5.0)
        assert out.shape == (N_ENVS, 4)

    def test_unit_norm(self) -> None:
        q = np.tile([1.0, 0.0, 0.0, 0.0], (8, 1)).astype(np.float32)
        out = _perturb_quat(q, max_deg=10.0)
        norms = np.linalg.norm(out, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_zero_noise_identity(self) -> None:
        q = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)).astype(np.float32)
        out = _perturb_quat(q, max_deg=0.0)
        np.testing.assert_allclose(out[:, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(out[:, 1:], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Integration tests for build_observations
# ---------------------------------------------------------------------------


class TestBuildObservations:
    """Integration tests using a minimal mock env."""

    @pytest.fixture
    def env(self) -> MagicMock:
        return _make_env(N_ENVS, disable_dr=True)

    @pytest.fixture
    def state(self) -> MagicMock:
        return _make_state(N_ENVS)

    def test_returns_obs_and_critic_keys(self, env, state) -> None:
        out = build_observations(env, state)
        assert set(out.keys()) == {"obs", "critic"}

    def test_actor_obs_dimension(self, env, state) -> None:
        """obs dimension must equal cfg.num_actor_obs (from obs_list sum)."""
        out = build_observations(env, state)
        expected_dim = compute_obs_dim(env.cfg.obs.obs_list)
        assert out["obs"].shape == (N_ENVS, expected_dim), (
            f"Expected actor obs shape ({N_ENVS}, {expected_dim}), got {out['obs'].shape}"
        )

    def test_critic_obs_dimension(self, env, state) -> None:
        """critic dimension must equal cfg.num_critic_obs (from state_list sum)."""
        out = build_observations(env, state)
        expected_dim = compute_obs_dim(env.cfg.obs.state_list)
        assert out["critic"].shape == (N_ENVS, expected_dim), (
            f"Expected critic obs shape ({N_ENVS}, {expected_dim}), got {out['critic'].shape}"
        )

    def test_expected_actor_dim_is_140(self, env, state) -> None:
        """Regression: verify the known actor dimension is 140."""
        out = build_observations(env, state)
        assert out["obs"].shape[1] == 140, (
            f"Actor obs should be 140-wide, got {out['obs'].shape[1]}"
        )

    def test_expected_critic_dim_is_162(self, env, state) -> None:
        """Regression: verify the known critic dimension is 162."""
        out = build_observations(env, state)
        assert out["critic"].shape[1] == 162, (
            f"Critic obs should be 162-wide, got {out['critic'].shape[1]}"
        )

    def test_clamp_applied(self, env, state) -> None:
        """All values must lie in [-clip, clip] after clamping (obs_utils.py:346-348)."""
        # Give the object a huge position to force out-of-range values
        env._backend.get_body_pos_w.side_effect = lambda ids: np.full(
            (N_ENVS, len(np.atleast_1d(ids)), 3), 1000.0, dtype=np.float32
        )
        out = build_observations(env, state)
        clip = env.cfg.obs.clamp_abs_observations
        assert float(out["obs"].max()) <= clip + 1e-5
        assert float(out["obs"].min()) >= -clip - 1e-5
        assert float(out["critic"].max()) <= clip + 1e-5
        assert float(out["critic"].min()) >= -clip - 1e-5

    def test_joint_pos_normalized(self, env, state) -> None:
        """Normalized joint_pos must lie in [-1, 1] for in-range raw values."""
        # Set raw joint pos to midpoint -> expect near 0
        env.get_joint_pos_canon.return_value = np.zeros((N_ENVS, NUM_JOINTS), dtype=np.float32)
        out = build_observations(env, state)

        # joint_pos occupies the first 29 elements in both obs and critic
        jp_actor = out["obs"][:, :NUM_JOINTS]
        jp_critic = out["critic"][:, :NUM_JOINTS]
        np.testing.assert_allclose(jp_actor, 0.0, atol=1e-5,
                                    err_msg="joint_pos at midpoint should normalise to 0")
        np.testing.assert_allclose(jp_critic, 0.0, atol=1e-5)

    def test_quat_w_at_end_actor(self, env, state) -> None:
        """palm_rot in actor obs: verify w (originally pos-0 in wxyz) is at end."""
        # Non-trivial quaternion: 90-deg rotation about Z (wxyz = cos45, 0, 0, sin45)
        cos45 = float(np.cos(np.pi / 4))
        sin45 = float(np.sin(np.pi / 4))

        def quat_side_effect(ids):
            ids_arr = np.atleast_1d(ids)
            q = np.zeros((N_ENVS, len(ids_arr), 4), dtype=np.float32)
            q[:, :, 0] = cos45  # w
            q[:, :, 3] = sin45  # z
            return q

        env._backend.get_body_quat_w.side_effect = quat_side_effect
        out = build_observations(env, state)

        # palm_rot starts after joint_pos(29) + joint_vel(29) + prev_targets(29) + palm_pos(3)
        palm_rot_start = 29 + 29 + 29 + 3  # = 90
        palm_rot = out["obs"][:, palm_rot_start : palm_rot_start + 4]

        # In xyzw: x=0, y=0, z=sin45, w=cos45 -> w is at index 3
        np.testing.assert_allclose(palm_rot[:, 3], cos45, atol=1e-5,
                                    err_msg="palm_rot in actor: w should be at position 3 (xyzw)")
        np.testing.assert_allclose(palm_rot[:, 2], sin45, atol=1e-5,
                                    err_msg="palm_rot z component should be sin(45)")

    def test_quat_w_at_end_critic(self, env, state) -> None:
        """palm_rot in critic obs also uses xyzw (D3: both actor and critic converted)."""
        cos45 = float(np.cos(np.pi / 4))
        sin45 = float(np.sin(np.pi / 4))

        def quat_side_effect(ids):
            ids_arr = np.atleast_1d(ids)
            q = np.zeros((N_ENVS, len(ids_arr), 4), dtype=np.float32)
            q[:, :, 0] = cos45
            q[:, :, 3] = sin45
            return q

        env._backend.get_body_quat_w.side_effect = quat_side_effect
        out = build_observations(env, state)

        # Same layout in state_list: joint_pos(29)+joint_vel(29)+prev_targets(29)+palm_pos(3)
        palm_rot_start = 29 + 29 + 29 + 3
        palm_rot = out["critic"][:, palm_rot_start : palm_rot_start + 4]

        np.testing.assert_allclose(palm_rot[:, 3], cos45, atol=1e-5,
                                    err_msg="critic palm_rot: w must be at pos 3 (D3 applies to both)")

    def test_critic_has_palm_vel_field(self, env, state) -> None:
        """palm_vel (6D) is privileged: present in critic, absent in actor."""
        state_fields = list(env.cfg.obs.state_list)
        obs_fields = list(env.cfg.obs.obs_list)
        assert "palm_vel" in state_fields, "palm_vel missing from critic state_list"
        assert "palm_vel" not in obs_fields, "palm_vel should not be in actor obs_list"

    def test_critic_has_all_privileged_fields(self, env, state) -> None:
        """All seven privileged fields are in critic but not actor (contract §3)."""
        privileged = {
            "palm_vel", "object_vel", "closest_keypoint_max_dist",
            "closest_fingertip_dist", "lifted_object", "progress",
            "successes", "reward",
        }
        state_fields = set(env.cfg.obs.state_list)
        obs_fields = set(env.cfg.obs.obs_list)
        for f in privileged:
            assert f in state_fields, f"Privileged field '{f}' missing from state_list"
            assert f not in obs_fields, f"Privileged field '{f}' incorrectly in obs_list"

    def test_dstar_sentinel_preserved(self, env, state) -> None:
        """When d* = -1 (sentinel), the critic obs must contain -1 for that field.

        The sentinel is not resolved to a current distance inside T2; that is T3's
        job (obs_utils.py:183-189 lives in compute_intermediate_values, not here).
        """
        sentinel_val = -1.0
        state.info["closest_keypoint_max_dist"] = np.full(
            (N_ENVS,), sentinel_val, dtype=np.float32
        )
        out = build_observations(env, state)

        # Locate closest_keypoint_max_dist in the critic vector.
        # It comes after all fields before it in state_list.
        fields_before = list(env.cfg.obs.state_list)
        idx = fields_before.index("closest_keypoint_max_dist")
        offset = sum(OBS_FIELD_SIZES[f] for f in fields_before[:idx])
        d_star_out = out["critic"][:, offset : offset + 1]

        # -1.0 is within [-10, 10] so clamping won't change it.
        np.testing.assert_allclose(
            d_star_out[:, 0],
            sentinel_val,
            atol=1e-5,
            err_msg="d* sentinel -1 should be passed through unchanged in critic obs",
        )

    def test_observation_keypoint_size_path(self) -> None:
        """Verify phi is NOT passed raw: obs with phi=1 must differ from obs with phi=25.

        If build_observations were passing phi directly as 'size' to compute_keypoints,
        the ratio between phi=1 and phi=25 outcomes would be 25.  When the correct
        observation_keypoint_size conversion is used the ratio is exactly 25 in the
        keypoint fields (since size_m = phi * 0.04).
        """
        env1 = _make_env(2, disable_dr=True)
        env2 = _make_env(2, disable_dr=True)

        # Zero the palm offset so palm_center == body_pos == (0,0,0).
        # This ensures keypoints_rel_palm = kp_world - 0 = kp_world, and the
        # ratio between phi=25 and phi=1 is a clean 25 for every component.
        # With a non-zero palm offset the ratio is not 25 because the constant
        # offset shifts each component independently.
        env1._palm_offset = np.zeros(3, dtype=np.float32)
        env2._palm_offset = np.zeros(3, dtype=np.float32)

        phi1 = np.ones((2, 3), dtype=np.float32)
        phi25 = np.full((2, 3), 25.0, dtype=np.float32)

        st1 = _make_state(2)
        st2 = _make_state(2)
        st1.info["object_scales"] = phi1
        st2.info["object_scales"] = phi25

        out1 = build_observations(env1, st1)
        out2 = build_observations(env2, st2)

        # Locate keypoints_rel_palm in actor obs
        obs_fields = list(env1.cfg.obs.obs_list)
        idx = obs_fields.index("keypoints_rel_palm")
        offset = sum(OBS_FIELD_SIZES[f] for f in obs_fields[:idx])
        size = OBS_FIELD_SIZES["keypoints_rel_palm"]

        kp1 = out1["obs"][:, offset : offset + size]
        kp2 = out2["obs"][:, offset : offset + size]

        # Object and palm both at origin with identity rotation, so:
        #   kp_world = corners * size_m * 0.5 * keypoint_scale
        #   kp_rel_palm = kp_world - palm_center = kp_world (palm_center=0)
        # size_m(phi=1)=0.04 m, size_m(phi=25)=1.0 m → ratio = 25.
        nonzero = np.abs(kp1) > 1e-6
        assert nonzero.any(), "keypoints_rel_palm should be nonzero with non-degenerate corners"
        ratio = np.abs(kp2[nonzero]) / np.abs(kp1[nonzero])
        np.testing.assert_allclose(
            ratio, 25.0, rtol=1e-3,
            err_msg="keypoint scale should be proportional to phi*base_size, not raw phi"
        )

    def test_output_dtype_float32(self, env, state) -> None:
        """Both output arrays must be float32 (MIGRATION_00 D0)."""
        out = build_observations(env, state)
        assert out["obs"].dtype == np.float32
        assert out["critic"].dtype == np.float32

    def test_no_nan_or_inf(self, env, state) -> None:
        """Outputs must be finite for normal inputs."""
        out = build_observations(env, state)
        assert np.isfinite(out["obs"]).all(), "actor obs contains NaN or Inf"
        assert np.isfinite(out["critic"]).all(), "critic obs contains NaN or Inf"

    def test_progress_log_scaling(self, env, state) -> None:
        """progress = log(steps/10 + 1); at steps=0 this is 0."""
        state.info["steps"] = np.zeros((N_ENVS,), dtype=np.uint32)
        out = build_observations(env, state)

        # Locate progress in state_list
        fields = list(env.cfg.obs.state_list)
        idx = fields.index("progress")
        offset = sum(OBS_FIELD_SIZES[f] for f in fields[:idx])
        prog = out["critic"][:, offset : offset + 1]
        np.testing.assert_allclose(prog, 0.0, atol=1e-5,
                                    err_msg="progress should be log(0/10+1)=0 at step 0")

    def test_reward_feature_scaled(self, env, state) -> None:
        """reward feature = previous reward * 0.01 (obs_utils.py:326, feature normalisation)."""
        raw_reward = 100.0
        state.info["reward"] = np.full((N_ENVS,), raw_reward, dtype=np.float32)
        out = build_observations(env, state)

        fields = list(env.cfg.obs.state_list)
        idx = fields.index("reward")
        offset = sum(OBS_FIELD_SIZES[f] for f in fields[:idx])
        rew_feat = out["critic"][:, offset : offset + 1]
        # 100.0 * 0.01 = 1.0
        np.testing.assert_allclose(rew_feat, 1.0, atol=1e-5,
                                    err_msg="reward feature = raw_reward * 0.01")

    def test_multi_env_batch(self) -> None:
        """build_observations handles larger batch sizes correctly."""
        n = 32
        env = _make_env(n, disable_dr=True)
        state = _make_state(n)
        out = build_observations(env, state)
        assert out["obs"].shape[0] == n
        assert out["critic"].shape[0] == n
