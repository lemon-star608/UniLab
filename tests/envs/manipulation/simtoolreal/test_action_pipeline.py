from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unilab.envs.manipulation.simtoolreal.action_pipeline import apply_action_pipeline


def _env() -> SimpleNamespace:
    n = 2
    info = {
        "steps": np.zeros(n, dtype=np.uint32),
        "successes": np.zeros(n, dtype=np.int32),
        "cur_targets": np.zeros((n, 4), dtype=np.float32),
        "prev_targets": np.zeros((n, 4), dtype=np.float32),
        "last_actions": np.zeros((n, 4), dtype=np.float32),
        "current_actions": np.zeros((n, 4), dtype=np.float32),
    }
    return SimpleNamespace(
        _state=SimpleNamespace(info=info),
        _np_dtype=np.dtype(np.float32),
        _num_envs=n,
        _perm_canon_to_backend=np.array([2, 0, 3, 1]),
        _arm_slice=slice(0, 2),
        _hand_slice=slice(2, 4),
        _arm_lower=np.full(2, -10.0, dtype=np.float32),
        _arm_upper=np.full(2, 10.0, dtype=np.float32),
        _hand_lower=np.full(2, -1.0, dtype=np.float32),
        _hand_upper=np.full(2, 1.0, dtype=np.float32),
        _action_queue=np.zeros((n, 2, 4), dtype=np.float32),
        cfg=SimpleNamespace(
            ctrl_dt=1.0 / 60.0,
            action=SimpleNamespace(
                clip_actions=1.0,
                dof_speed_scale=1.5,
                arm_moving_average=0.1,
                hand_moving_average=0.1,
            ),
            domain_randomization=SimpleNamespace(use_action_delay=False, action_delay_max=0),
        ),
    )


def test_action_pipeline_applies_canonical_permutation_and_control_order() -> None:
    env = _env()
    actions = np.array([[2.0, -2.0, 0.5, -0.5], [0.25, 0.5, -0.25, -0.5]], dtype=np.float32)
    apply_action_pipeline(env, actions)
    np.testing.assert_array_equal(env._state.info["current_actions"], np.clip(actions, -1, 1))
    assert np.all(env._state.info["cur_targets"] <= 10.0)
    assert not np.shares_memory(env._state.info["cur_targets"], env._state.info["prev_targets"])


def test_goal_advance_does_not_flush_delay_queue(monkeypatch) -> None:
    env = _env()
    env._state.info["steps"][:] = 0
    env._state.info["successes"][:] = 1
    monkeypatch.setattr(np.random, "randint", lambda low, high, size: np.zeros(size, dtype=np.intp))
    env.cfg.domain_randomization.use_action_delay = True
    env.cfg.domain_randomization.action_delay_max = 2
    apply_action_pipeline(env, np.ones((2, 4), dtype=np.float32))
    assert not np.all(env._action_queue == 0.0)
