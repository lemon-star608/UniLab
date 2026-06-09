"""Contract tests for the Go2ArmPosForce environment."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

_POS_FORCE_MODULE = "unilab.envs.locomotion.go2_arm.pos_force"
_REGISTRY_MODULE = "unilab.base.registry"


def _skip_if_no_mujoco():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco.batch_env import BatchEnvPool  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env not available")


def _registry_module():
    return importlib.import_module(_REGISTRY_MODULE)


def _ensure_registered() -> None:
    registry = _registry_module()
    registry.ensure_registries()
    if not registry.contains("Go2ArmPosForce"):
        importlib.import_module(_POS_FORCE_MODULE)


def _default_reward_cfg(**overrides):
    from unilab.envs.locomotion.go2_arm.pos_force import RewardConfig

    cfg = dict(
        scales={
            "tracking_lin_vel_force_world": 2.0,
            "tracking_ee_force_world": 2.0,
            "tracking_ang_vel": 1.0,
            "ref_dof_leg": 1.0,
            "alive": 1.5,
            "base_height": -2.0,
            "torques": -5.0e-6,
        },
        tracking_sigma=0.25,
        base_height_target=0.40,
    )
    cfg.update(overrides)
    return RewardConfig(**cfg)


def _make_env(num_envs: int = 2, env_cfg_override: dict | None = None):
    _ensure_registered()
    registry = _registry_module()
    override = {"reward_config": _default_reward_cfg()}
    if env_cfg_override:
        override.update(env_cfg_override)
    return registry.make(
        "Go2ArmPosForce",
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override=override,
    )


def test_pos_force_cfg_registered():
    _ensure_registered()
    registry = _registry_module()
    assert registry.contains("Go2ArmPosForce")


def test_pos_force_registers_mujoco_backend():
    _ensure_registered()
    registry = _registry_module()
    meta = registry._envs["Go2ArmPosForce"]
    assert meta.support_sim_backend("mujoco")


@pytest.mark.slow
def test_pos_force_obs_groups_spec():
    _skip_if_no_mujoco()
    env = _make_env(num_envs=2)
    spec = env.obs_groups_spec
    assert set(spec) == {"obs", "critic"}
    # actor single-step = 76, default history = 32; critic single-step = 130, 1-step.
    assert spec["obs"] == 32 * 76
    assert spec["critic"] == 130


@pytest.mark.slow
def test_pos_force_reset_step_contract():
    _skip_if_no_mujoco()
    env = _make_env(num_envs=2)
    state = env.init_state()
    assert state.obs["obs"].shape == (2, 32 * 76)
    assert state.obs["critic"].shape == (2, 130)

    actions = np.zeros((2, 18), dtype=np.float64)
    state = env.step(actions)
    assert state.reward.shape == (2,)
    assert np.isfinite(state.reward).all()
    assert state.obs["obs"].shape == (2, 32 * 76)
    assert np.isfinite(state.obs["obs"]).all()
    assert np.isfinite(state.obs["critic"]).all()


@pytest.mark.slow
def test_pos_force_torque_within_limits():
    _skip_if_no_mujoco()
    env = _make_env(num_envs=2)
    env.init_state()
    for _ in range(10):
        env.step(np.zeros((2, 18), dtype=np.float64))
    # Python PD torque must respect the per-joint limits (legs/j1-3 24, wrist 8).
    assert np.all(np.abs(env._last_torque) <= env._torque_limits + 1e-6)


@pytest.mark.slow
def test_pos_force_external_forces_apply_and_observe():
    _skip_if_no_mujoco()
    env = _make_env(
        num_envs=4,
        env_cfg_override={
            "commands": {
                "force_start_step": 0,
                "push_gripper_interval_s": [0.1, 0.2],
                "push_base_interval_s": [0.1, 0.2],
                "gripper_forced_prob_ext": 1.0,
                "base_forced_prob_ext": 1.0,
            }
        },
    )
    env.init_state()
    max_ee = 0.0
    max_base = 0.0
    for _ in range(120):
        env.step(np.zeros((4, 18), dtype=np.float64))
        max_ee = max(max_ee, float(np.abs(env._force_ee_world).max()))
        max_base = max(max_base, float(np.abs(env._force_base_world).max()))
    # External forces fired and stayed within configured ranges.
    assert max_ee > 0.0
    assert max_base > 0.0
    assert max_ee <= abs(env._cfg.commands.max_push_force_xyz_gripper_ext[1]) + 1e-3
    assert max_base <= abs(env._cfg.commands.max_push_force_xyz_base_ext[1]) + 1e-3


@pytest.mark.slow
def test_pos_force_no_forces_before_curriculum():
    _skip_if_no_mujoco()
    env = _make_env(num_envs=2, env_cfg_override={"commands": {"force_start_step": 10_000}})
    env.init_state()
    for _ in range(50):
        env.step(np.zeros((2, 18), dtype=np.float64))
    # No external force before the curriculum start step.
    assert np.all(env._force_ee_world == 0.0)
    assert np.all(env._force_base_world == 0.0)
