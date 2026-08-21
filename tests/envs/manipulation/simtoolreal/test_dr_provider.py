from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unilab.dr import DomainRandomizationCapabilities
from unilab.envs.manipulation.simtoolreal.config import SimToolRealCfg
from unilab.envs.manipulation.simtoolreal.dr_provider import SimToolRealDRProvider


def _env() -> SimpleNamespace:
    n = 3
    cfg = SimToolRealCfg()
    cfg.reset.fixed_start_pose = (0.1, 0.2, 0.63, 1.0, 0.0, 0.0, 0.0)
    cfg.reset.fixed_goal_pose = (0.2, 0.1, 0.8, 1.0, 0.0, 0.0, 0.0)
    return SimpleNamespace(
        _num_envs=n,
        nq=36,
        nv=35,
        cfg=cfg,
        _default_joint_pos_canon=np.zeros(29, dtype=np.float32),
        _joint_lower_canon=np.full(29, -1.0, dtype=np.float32),
        _joint_upper_canon=np.full(29, 1.0, dtype=np.float32),
        _dof_pos_idx_canon=np.arange(29),
        _dof_vel_idx_canon=np.arange(29),
        _obj_pos_slice=slice(29, 32),
        _obj_quat_slice=slice(32, 36),
        _perm_canon_to_backend=np.arange(29),
        _object_forces=np.ones((n, 3), dtype=np.float32),
        _object_torques=np.ones((n, 3), dtype=np.float32),
        _action_queue=np.ones((n, 2, 29), dtype=np.float32),
        _obs_queue=np.ones((n, 2, 140), dtype=np.float32),
        _object_state_queue=np.ones((n, 2, 13), dtype=np.float32),
        _random_force_prob=np.ones(n, dtype=np.float32),
        _random_torque_prob=np.ones(n, dtype=np.float32),
        _object_scale_multiplier=np.ones((n, 3), dtype=np.float32),
        _state_cache_lifted_object=np.ones(n, dtype=bool),
        resolve_object_scale=lambda: np.ones((n, 3), dtype=np.float32),
        _state=SimpleNamespace(info={"successes": np.array([2, 1, 0], dtype=np.int32)}),
    )


def test_validate_uses_public_dr_contract_without_backend_probe() -> None:
    env = _env()
    SimToolRealDRProvider().validate(env, DomainRandomizationCapabilities())


def test_source_random_reset_plan_maps_fixed_table_z_and_only_selected_rows(monkeypatch) -> None:
    env = _env()
    env.cfg.reset.fixed_start_pose = None
    env.cfg.reset.fixed_goal_pose = None
    env_ids = np.array([1, 2], dtype=np.intp)
    monkeypatch.setattr(np.random, "rand", lambda *shape: np.full(shape, 0.5, dtype=np.float32))
    monkeypatch.setattr(
        np.random,
        "uniform",
        lambda *args, **kwargs: np.zeros(kwargs.get("size", args[-1]), dtype=np.float32),
    )
    monkeypatch.setattr(np.random, "randint", lambda low, high, size: np.zeros(size, dtype=np.intp))
    provider = SimToolRealDRProvider()
    plan = provider.build_reset_plan(env, env_ids)
    assert plan.qpos.shape == (2, 36)
    np.testing.assert_allclose(plan.qpos[:, 31], 0.63)
    np.testing.assert_array_equal(env._object_forces[0], [1, 1, 1])
    np.testing.assert_array_equal(env._object_forces[1:], 0.0)
    np.testing.assert_array_equal(env._state_cache_lifted_object, [True, False, False])
    np.testing.assert_array_equal(plan.info_updates["successes"], [0, 0])
