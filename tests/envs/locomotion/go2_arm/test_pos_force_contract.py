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
            # Exercise the re-added feet/contact/dof terms (and their sensors).
            "feet_contact_number": 2.0,
            "feet_air_time": 1.0,
            "feet_height": 1.0,
            "feet_height_high": -15.0,
            "feet_pos_xy": -0.5,
            "feet_drag": -8.0e-4,
            "feet_contact_forces": -1.0e-3,
            "collision": -5.0,
            "dof_pos_limits": -10.0,
            "stand_still": 0.5,
            "dof_acc": -2.5e-7,
            "dof_acc_arm": -4.5e-7,
            "dof_vel_arm": -2.0e-4,
            "action_rate_arm": -0.045,
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


def test_ref_dof_leg_is_l1_exponential():
    """Regression for the migration bug: UniFP uses exp(-L1_err * 0.1), not L2/sigma_force."""
    from types import SimpleNamespace

    from unilab.envs.locomotion.go2_arm.pos_force import (
        NUM_LEG,
        Go2ArmPosForceEnv,
        RewardConfig,
    )

    rc = RewardConfig(scales={}, ref_dof_scale=0.1)
    ref = np.zeros((2, NUM_LEG), dtype=np.float64)
    stub = SimpleNamespace(_ref_dof_pos=ref, _reward_cfg=rc)
    dof_pos = np.concatenate(
        [np.full((2, NUM_LEG), 0.5), np.zeros((2, 6))], axis=1
    )  # 18 dofs; arm ignored
    ctx = SimpleNamespace(dof_pos=dof_pos)
    out = Go2ArmPosForceEnv._reward_ref_dof_leg(stub, ctx)
    expected = np.exp(-(0.5 * NUM_LEG) * 0.1)  # L1 error = 0.5*12, temp 0.1
    assert np.allclose(out, expected)


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
    # Derive both dims (history x single-step) rather than hardcode so obs-layout
    # and history-length changes stay in sync.
    h_a = env._cfg.history.num_actor_history
    h_c = env._cfg.history.num_critic_history
    assert spec["obs"] == h_a * env._actor_single_obs_dim()
    assert spec["critic"] == h_c * env._critic_single_obs_dim()


@pytest.mark.slow
def test_pos_force_reset_step_contract():
    _skip_if_no_mujoco()
    env = _make_env(num_envs=2)
    critic_dim = env._cfg.history.num_critic_history * env._critic_single_obs_dim()
    state = env.init_state()
    assert state.obs["obs"].shape == (2, 32 * 76)
    assert state.obs["critic"].shape == (2, critic_dim)

    actions = np.zeros((2, 18), dtype=np.float64)
    state = env.step(actions)
    assert state.reward.shape == (2,)
    assert np.isfinite(state.reward).all()
    assert state.obs["obs"].shape == (2, 32 * 76)
    assert np.isfinite(state.obs["obs"]).all()
    assert np.isfinite(state.obs["critic"]).all()
    # Re-added feet/contact reward terms must stay finite and well-signed.
    ctx = env._last_reward_ctx
    assert np.all(env._reward_collision(ctx) >= 0.0)
    assert np.all(env._reward_dof_pos_limits(ctx) >= 0.0)
    assert np.all(np.isfinite(env._reward_feet_drag(ctx)))


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
    # UniFP zeroes the commanded base-force z-component and attenuates the
    # external base-force z to force_z_base_ext_scale (0.05).
    assert np.all(env._force_base_cmd[:, 2] == 0.0)
    z_cap = 0.05 * abs(env._cfg.commands.max_push_force_xyz_base_ext[1]) + 1e-3
    assert float(np.abs(env._force_base_world[:, 2]).max()) <= z_cap


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
