"""Unit tests for the SimToolReal action pipeline (T1).

Pure-function tests against a mocked env; no MuJoCo, no Isaac Sim. They pin the
four load-bearing structural facts of action_utils.py:18-75 plus the numbers
from migration guide §4:

  - permute canonical -> backend happens first (:34)
  - action delay is applied before the control law (:36-48)
  - arm is clamped twice: after the accumulation (:53) and after the EMA (:58)
  - hand is clamped once, after the EMA (:69)
  - dt = ctrl_dt = 1/60, not sim_dt = 1/120
  - EMA = 0.1 means 0.1*new + 0.9*old
  - prev_targets = cur_targets.copy(), never an alias
  - flushed delay queue (fresh episode) returns the current action
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from unilab.envs.manipulation.simtoolreal.action_pipeline import apply_action_pipeline
from unilab.envs.manipulation.simtoolreal.config import SimToolRealCfg
from unilab.envs.manipulation.simtoolreal.constants import (
    NUM_ARM_JOINTS,
    NUM_HAND_JOINTS,
    NUM_JOINTS,
)

N_ENVS = 4
DT = 1.0 / 60.0
DOF_SPEED_SCALE = 1.5
EMA = 0.1


def _make_env(
    n: int = N_ENVS,
    *,
    steps: int = 5,
    successes: int = 0,
    prev: np.ndarray | None = None,
    arm_limit: float = 1.0,
    hand_limit: float = 0.5,
    perm: np.ndarray | None = None,
    use_action_delay: bool = False,
) -> MagicMock:
    """Build a minimal SimToolRealEnv mock with a seeded ``_state.info`` bus.

    Limits are symmetric and uniform so expected values stay hand-checkable.
    ``steps=5`` by default, i.e. mid-episode, so the delay queue does not flush
    unless a test asks for it.
    """
    env = MagicMock()
    env._num_envs = n
    env._np_dtype = np.float32
    env._replay_target_backend_order = None

    env._arm_slice = slice(0, NUM_ARM_JOINTS)
    env._hand_slice = slice(NUM_ARM_JOINTS, NUM_JOINTS)

    env._arm_lower = np.full(NUM_ARM_JOINTS, -arm_limit, dtype=np.float32)
    env._arm_upper = np.full(NUM_ARM_JOINTS, +arm_limit, dtype=np.float32)
    env._hand_lower = np.full(NUM_HAND_JOINTS, -hand_limit, dtype=np.float32)
    env._hand_upper = np.full(NUM_HAND_JOINTS, +hand_limit, dtype=np.float32)

    env._perm_canon_to_backend = (
        np.arange(NUM_JOINTS, dtype=np.int64) if perm is None else np.asarray(perm, dtype=np.int64)
    )

    cfg = SimToolRealCfg()
    cfg.domain_randomization.use_action_delay = use_action_delay
    env.cfg = cfg

    env._action_queue = np.zeros(
        (n, max(int(cfg.domain_randomization.action_delay_max), 1), NUM_JOINTS), dtype=np.float32
    )

    prev_targets = (
        np.zeros((n, NUM_JOINTS), dtype=np.float32)
        if prev is None
        else np.asarray(prev, dtype=np.float32).copy()
    )

    state = MagicMock()
    state.info = {
        "steps": np.full((n,), steps, dtype=np.uint32),
        "successes": np.full((n,), successes, dtype=np.int32),
        "prev_targets": prev_targets,
        "cur_targets": prev_targets.copy(),
        "last_actions": np.zeros((n, NUM_JOINTS), dtype=np.float32),
        "current_actions": np.zeros((n, NUM_JOINTS), dtype=np.float32),
    }
    env._state = state
    return env


def _actions(n: int = N_ENVS, arm: float = 0.0, hand: float = 0.0) -> np.ndarray:
    """Canonical-order action with a constant arm value and hand value."""
    a = np.zeros((n, NUM_JOINTS), dtype=np.float32)
    a[:, :NUM_ARM_JOINTS] = arm
    a[:, NUM_ARM_JOINTS:] = hand
    return a


def _arm(env: MagicMock) -> np.ndarray:
    """Arm slice of the produced targets."""
    return env._state.info["cur_targets"][:, :NUM_ARM_JOINTS]


def _hand(env: MagicMock) -> np.ndarray:
    """Hand slice of the produced targets."""
    return env._state.info["cur_targets"][:, NUM_ARM_JOINTS:]


class TestArmVelocityDelta:
    """Arm law: prev + 1.5*dt*action, clamp, EMA(0.1), clamp again (:50-58)."""

    def test_velocity_delta_and_ema_values(self) -> None:
        """One step from zero: raw = 1.5 * (1/60) * 1 = 0.025, EMA -> 0.0025."""
        env = _make_env()
        apply_action_pipeline(env, _actions(arm=1.0))

        expected = EMA * (DOF_SPEED_SCALE * DT * 1.0)
        assert np.allclose(_arm(env), expected, atol=1e-7)
        assert np.isclose(expected, 0.0025)

    def test_dt_is_ctrl_dt_not_sim_dt(self) -> None:
        """dt must be 1/60 (guide §4 easy-to-miss list), not sim_dt = 1/120."""
        env = _make_env()
        apply_action_pipeline(env, _actions(arm=1.0))

        with_ctrl_dt = EMA * DOF_SPEED_SCALE * (1.0 / 60.0)
        with_sim_dt = EMA * DOF_SPEED_SCALE * (1.0 / 120.0)
        assert np.allclose(_arm(env), with_ctrl_dt, atol=1e-7)
        assert not np.allclose(_arm(env), with_sim_dt, atol=1e-7)

    def test_ema_is_ten_percent_new(self) -> None:
        """EMA = 0.1*new + 0.9*old, i.e. heavy smoothing toward the old target."""
        prev = np.full((N_ENVS, NUM_JOINTS), 0.5, dtype=np.float32)
        env = _make_env(prev=prev)
        apply_action_pipeline(env, _actions(arm=1.0))

        arm_raw = 0.5 + DOF_SPEED_SCALE * DT * 1.0
        assert np.allclose(_arm(env), EMA * arm_raw + (1.0 - EMA) * 0.5, atol=1e-7)
        # The reversed weighting (0.9*new + 0.1*old) would land elsewhere.
        assert not np.allclose(_arm(env), 0.9 * arm_raw + 0.1 * 0.5, atol=1e-7)


class TestArmDoubleClamp:
    """Both arm clamps are real and observable (:53 and :58)."""

    def test_first_clamp_binds_before_ema(self) -> None:
        """prev=0.99, action=+1: raw 1.015 is clamped to 1.0 *before* the EMA."""
        prev = np.full((N_ENVS, NUM_JOINTS), 0.99, dtype=np.float32)
        env = _make_env(prev=prev, arm_limit=1.0, hand_limit=2.0)
        apply_action_pipeline(env, _actions(arm=1.0))

        with_clamp = EMA * 1.0 + (1.0 - EMA) * 0.99  # 0.991
        without_clamp = EMA * 1.015 + (1.0 - EMA) * 0.99  # 0.9925
        assert np.allclose(_arm(env), with_clamp, atol=1e-6)
        assert not np.allclose(_arm(env), without_clamp, atol=1e-6)

    def test_second_clamp_binds_after_ema(self) -> None:
        """An out-of-range prev makes the post-EMA clamp the only thing saving us.

        prev=2.0 (outside +/-1), action=0: raw clamps to 1.0, then
        EMA = 0.1*1.0 + 0.9*2.0 = 1.9, which only clamp #2 pulls back to 1.0.
        """
        prev = np.full((N_ENVS, NUM_JOINTS), 2.0, dtype=np.float32)
        env = _make_env(prev=prev, arm_limit=1.0)
        apply_action_pipeline(env, _actions(arm=0.0))

        assert np.allclose(_arm(env), 1.0, atol=1e-6)
        assert not np.allclose(_arm(env), 1.9, atol=1e-6)

    def test_arm_stays_in_limits_over_many_steps(self) -> None:
        """Random saturating actions never push the accumulator out of range."""
        rng = np.random.default_rng(0)
        env = _make_env(arm_limit=1.0)

        for _ in range(400):
            actions = rng.uniform(-5.0, 5.0, size=(N_ENVS, NUM_JOINTS)).astype(np.float32)
            apply_action_pipeline(env, actions)
            arm = _arm(env)
            assert np.all(arm >= env._arm_lower - 1e-6)
            assert np.all(arm <= env._arm_upper + 1e-6)


class TestHandAbsolute:
    """Hand law: lower + 0.5*(a+1)*(upper-lower), EMA(0.1), one clamp (:60-69)."""

    def test_plus_one_maps_to_upper_minus_one_to_lower(self) -> None:
        """action=+1 -> upper and action=-1 -> lower, both seen through the EMA."""
        env_up = _make_env(hand_limit=0.5)
        apply_action_pipeline(env_up, _actions(hand=1.0))
        # hand_raw = upper = 0.5; prev = 0 -> EMA = 0.1*0.5 = 0.05
        assert np.allclose(_hand(env_up), EMA * 0.5, atol=1e-7)

        env_dn = _make_env(hand_limit=0.5)
        apply_action_pipeline(env_dn, _actions(hand=-1.0))
        # hand_raw = lower = -0.5 -> EMA = 0.1*(-0.5) = -0.05
        assert np.allclose(_hand(env_dn), EMA * -0.5, atol=1e-7)

    def test_saturated_action_converges_to_limit(self) -> None:
        """Held action=+1 walks the EMA onto the upper limit; -1 onto the lower."""
        env = _make_env(hand_limit=0.5)
        for _ in range(300):
            apply_action_pipeline(env, _actions(hand=1.0))
        assert np.allclose(_hand(env), env._hand_upper, atol=1e-5)

        for _ in range(300):
            apply_action_pipeline(env, _actions(hand=-1.0))
        assert np.allclose(_hand(env), env._hand_lower, atol=1e-5)

    def test_midpoint_action_maps_to_midrange(self) -> None:
        """action=0 maps to the range midpoint (0 for symmetric limits)."""
        prev = np.zeros((N_ENVS, NUM_JOINTS), dtype=np.float32)
        env = _make_env(prev=prev, hand_limit=0.5)
        apply_action_pipeline(env, _actions(hand=0.0))
        assert np.allclose(_hand(env), 0.0, atol=1e-7)

    def test_single_clamp_is_after_the_ema(self) -> None:
        """An unclamped action overshoots, and only the post-EMA clamp catches it.

        There is no input clamp (guide §4), so action=+3 gives hand_raw = 1.5.
        prev=0.45 -> EMA = 0.1*1.5 + 0.9*0.45 = 0.555 > 0.5, clamped to 0.5.
        A hypothetical pre-EMA clamp would instead give 0.1*0.5 + 0.9*0.45 = 0.455.
        """
        prev = np.full((N_ENVS, NUM_JOINTS), 0.45, dtype=np.float32)
        env = _make_env(prev=prev, hand_limit=0.5)
        apply_action_pipeline(env, _actions(hand=3.0))

        assert np.allclose(_hand(env), 0.5, atol=1e-6)
        assert not np.allclose(_hand(env), 0.455, atol=1e-6)

    def test_hand_stays_in_limits_over_many_steps(self) -> None:
        """Random out-of-range actions never leave the joint range."""
        rng = np.random.default_rng(1)
        env = _make_env(hand_limit=0.5)

        for _ in range(200):
            actions = rng.uniform(-4.0, 4.0, size=(N_ENVS, NUM_JOINTS)).astype(np.float32)
            apply_action_pipeline(env, actions)
            hand = _hand(env)
            assert np.all(hand >= env._hand_lower - 1e-6)
            assert np.all(hand <= env._hand_upper + 1e-6)


class TestPermuteFirst:
    """canonical -> backend permute happens before anything else (:34)."""

    def test_arm_law_reads_permuted_slots(self) -> None:
        """With a reversed permutation, backend slot j gets canonical index 28-j."""
        perm = np.arange(NUM_JOINTS, dtype=np.int64)[::-1].copy()
        env = _make_env(perm=perm, arm_limit=2.0, hand_limit=2.0)

        actions = np.tile(
            np.arange(NUM_JOINTS, dtype=np.float32) / 100.0, (N_ENVS, 1)
        )  # canonical action k = k/100
        apply_action_pipeline(env, actions)

        arm_action_backend = actions[:, perm][:, :NUM_ARM_JOINTS]
        expected = EMA * np.clip(
            DOF_SPEED_SCALE * DT * arm_action_backend, env._arm_lower, env._arm_upper
        )
        assert np.allclose(_arm(env), expected, atol=1e-7)

    def test_permutation_actually_changes_the_result(self) -> None:
        """Identity vs reversed permutation must not agree, or the permute is dead."""
        actions = np.tile(np.arange(NUM_JOINTS, dtype=np.float32) / 100.0, (N_ENVS, 1))

        env_id = _make_env(arm_limit=2.0, hand_limit=2.0)
        apply_action_pipeline(env_id, actions)

        perm = np.arange(NUM_JOINTS, dtype=np.int64)[::-1].copy()
        env_rev = _make_env(perm=perm, arm_limit=2.0, hand_limit=2.0)
        apply_action_pipeline(env_rev, actions)

        assert not np.allclose(_arm(env_id), _arm(env_rev), atol=1e-7)


class TestActionDelay:
    """Delay queue is applied before the control law (:36-48)."""

    @staticmethod
    def _fixed_index(idx: int):
        """Return a ``np.random.randint`` stand-in that always picks ``idx``."""

        def _randint(low, high=None, size=None, dtype=int):  # noqa: ANN001, ANN202
            n = size[0] if isinstance(size, tuple) else size
            return np.full((n,), idx, dtype=np.int64)

        return _randint

    def test_flush_on_fresh_episode_returns_current_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """steps==0 & successes==0 fills every slot, so any index gives the new action."""
        monkeypatch.setattr(np.random, "randint", self._fixed_index(2))

        env = _make_env(steps=0, successes=0, use_action_delay=True)
        env._action_queue[:] = 7.0  # stale garbage from the previous episode
        apply_action_pipeline(env, _actions(arm=1.0, hand=1.0))

        ref = _make_env(steps=0, successes=0, use_action_delay=False)
        apply_action_pipeline(ref, _actions(arm=1.0, hand=1.0))

        assert np.allclose(env._state.info["cur_targets"], ref._state.info["cur_targets"])
        # Every slot now holds the current action, so the garbage is gone.
        assert np.allclose(env._action_queue[:, :, :NUM_ARM_JOINTS], 1.0)

    def test_delayed_action_drives_the_control_law(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mid-episode, a sampled older slot is what the arm/hand laws consume."""
        monkeypatch.setattr(np.random, "randint", self._fixed_index(1))

        env = _make_env(steps=5, use_action_delay=True)
        env._action_queue[:, 0, :] = 0.4  # becomes slot 1 after the roll
        env._action_queue[:, 1, :] = 0.8
        env._action_queue[:, 2, :] = 0.2
        apply_action_pipeline(env, _actions(arm=1.0, hand=1.0))

        ref = _make_env(steps=5, use_action_delay=False)
        apply_action_pipeline(ref, _actions(arm=0.4, hand=0.4))

        assert np.allclose(env._state.info["cur_targets"], ref._state.info["cur_targets"])

    def test_queue_roll_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Newest frame lands in slot 0; older frames shift toward the tail."""
        monkeypatch.setattr(np.random, "randint", self._fixed_index(0))

        env = _make_env(steps=5, use_action_delay=True)
        env._action_queue[:, 0, :] = 0.4
        env._action_queue[:, 1, :] = 0.8
        env._action_queue[:, 2, :] = 0.2
        apply_action_pipeline(env, _actions(arm=1.0, hand=1.0))

        assert np.allclose(env._action_queue[:, 0, :], 1.0)
        assert np.allclose(env._action_queue[:, 1, :], 0.4)
        assert np.allclose(env._action_queue[:, 2, :], 0.8)

    def test_goal_advance_does_not_flush(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """successes>0 with steps==0 is an intra-episode advance (D2), not a reset."""
        monkeypatch.setattr(np.random, "randint", self._fixed_index(1))

        env = _make_env(steps=0, successes=1, use_action_delay=True)
        env._action_queue[:] = 0.4
        apply_action_pipeline(env, _actions(arm=1.0, hand=1.0))

        ref = _make_env(steps=0, successes=1, use_action_delay=False)
        apply_action_pipeline(ref, _actions(arm=0.4, hand=0.4))

        assert np.allclose(env._state.info["cur_targets"], ref._state.info["cur_targets"])

    def test_disabled_delay_leaves_queue_untouched(self) -> None:
        """use_action_delay=False bypasses the queue entirely."""
        env = _make_env(use_action_delay=False)
        apply_action_pipeline(env, _actions(arm=1.0, hand=1.0))
        assert np.allclose(env._action_queue, 0.0)

    def test_per_env_indices_are_independent(self) -> None:
        """Each env draws its own delay index, so envs can disagree mid-episode."""
        env = _make_env(n=64, steps=5, use_action_delay=True)
        # Slot 0 rolls to slot 1; distinct slot values make the draw observable.
        env._action_queue[:, 0, :] = 0.0
        env._action_queue[:, 1, :] = 1.0
        env._action_queue[:, 2, :] = 1.0
        np.random.seed(0)
        apply_action_pipeline(env, _actions(n=64, arm=0.0, hand=0.0))

        arm = env._state.info["cur_targets"][:, :NUM_ARM_JOINTS]
        assert len(np.unique(np.round(arm[:, 0], 6))) > 1


class TestPrevTargetsAliasing:
    """prev_targets = cur_targets.copy(), never a shared view (:74)."""

    def test_prev_and_cur_are_distinct_buffers(self) -> None:
        """Mutating cur_targets after the call must not disturb prev_targets."""
        env = _make_env()
        apply_action_pipeline(env, _actions(arm=1.0, hand=1.0))

        info = env._state.info
        assert info["prev_targets"] is not info["cur_targets"]
        assert not np.shares_memory(info["prev_targets"], info["cur_targets"])

        snapshot = info["prev_targets"].copy()
        info["cur_targets"][:] = 999.0
        assert np.array_equal(info["prev_targets"], snapshot)

    def test_prev_equals_cur_after_the_call(self) -> None:
        """The snapshot is taken at the end, so the two hold equal values."""
        env = _make_env()
        apply_action_pipeline(env, _actions(arm=0.7, hand=-0.3))
        info = env._state.info
        assert np.array_equal(info["prev_targets"], info["cur_targets"])

    def test_prev_targets_is_the_arm_reference(self) -> None:
        """Arm is relative: action=0 holds the (non-zero) seeded prev target."""
        prev = np.full((N_ENVS, NUM_JOINTS), 0.3, dtype=np.float32)
        env = _make_env(prev=prev, hand_limit=2.0)
        apply_action_pipeline(env, _actions(arm=0.0, hand=0.0))
        assert np.allclose(_arm(env), 0.3, atol=1e-7)

    def test_buffers_are_updated_in_place(self) -> None:
        """The info arrays keep their identity so the base-class reset scatter works."""
        env = _make_env()
        info = env._state.info
        cur_before, prev_before = info["cur_targets"], info["prev_targets"]
        apply_action_pipeline(env, _actions(arm=1.0))
        assert info["cur_targets"] is cur_before
        assert info["prev_targets"] is prev_before


class TestContract:
    """Contract §4.1 / §2.1 surface: return value, dtype, info keys, errors."""

    def test_returns_none(self) -> None:
        """Frozen signature returns nothing; the targets are the side effect."""
        env = _make_env()
        assert apply_action_pipeline(env, _actions(arm=1.0)) is None

    def test_targets_stay_float32(self) -> None:
        """D0: float32 everywhere, even when handed float64 actions."""
        env = _make_env()
        apply_action_pipeline(env, _actions(arm=1.0).astype(np.float64))
        info = env._state.info
        assert info["cur_targets"].dtype == np.float32
        assert info["prev_targets"].dtype == np.float32

    def test_raw_action_bookkeeping_rotates(self) -> None:
        """current_actions holds this step's raw canonical action; last_actions the previous."""
        env = _make_env()
        first = _actions(arm=0.5, hand=-0.5)
        second = _actions(arm=-0.25, hand=0.75)

        apply_action_pipeline(env, first)
        info = env._state.info
        assert np.allclose(info["current_actions"], first)
        assert np.allclose(info["last_actions"], 0.0)

        apply_action_pipeline(env, second)
        assert np.allclose(info["current_actions"], second)
        assert np.allclose(info["last_actions"], first)

    def test_raw_actions_are_canonical_and_undelayed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bookkeeping records the policy output, before permute and before delay."""
        monkeypatch.setattr(np.random, "randint", TestActionDelay._fixed_index(2))
        perm = np.arange(NUM_JOINTS, dtype=np.int64)[::-1].copy()
        env = _make_env(perm=perm, steps=5, use_action_delay=True)
        env._action_queue[:] = 0.9

        actions = np.tile(np.arange(NUM_JOINTS, dtype=np.float32) / 100.0, (N_ENVS, 1))
        apply_action_pipeline(env, actions)
        assert np.allclose(env._state.info["current_actions"], actions)

    def test_wrong_action_shape_raises(self) -> None:
        """A mis-shaped action is a caller bug, not something to broadcast through."""
        env = _make_env()
        with pytest.raises(ValueError, match="actions must have shape"):
            apply_action_pipeline(env, np.zeros((N_ENVS, NUM_JOINTS - 1), dtype=np.float32))

    def test_replay_path_writes_backend_targets_directly(self) -> None:
        """Debug replay bypasses the whole pipeline (:20-25)."""
        env = _make_env()
        replay = np.full((N_ENVS, NUM_JOINTS), 0.123, dtype=np.float32)
        env._replay_target_backend_order = replay

        apply_action_pipeline(env, _actions(arm=1.0, hand=1.0))
        info = env._state.info
        assert np.allclose(info["cur_targets"], 0.123)
        assert np.allclose(info["prev_targets"], 0.123)
        assert not np.shares_memory(info["prev_targets"], info["cur_targets"])
