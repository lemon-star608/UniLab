"""Contract tests for the A2ArmPosForce environment (Unitree A2 + Airbot Play).

The A2 MJCF mirrors the Go2 sensor/geom/leg-ordering contract (legs FL,FR,RL,RR;
named foot/base/leg geoms; Go2-named IMU/foot/thigh sensors), so the env reuses
``Go2ArmPosForceEnv`` unchanged. These tests prove the A2 model + config + env
chain constructs and steps in MuJoCo with A2-specific physics: 18 DOF and
per-joint leg torque limits (hip/thigh 120 Nm, calf 180 Nm)."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def _skip_if_no_mujoco():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco.batch_env import BatchEnvPool  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env not available")


def _ensure_registered() -> None:
    from unilab.base import registry

    registry.ensure_registries()
    if not registry.contains("A2ArmPosForce"):
        importlib.import_module("unilab.envs.locomotion.go2_arm.pos_force")


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
        },
        tracking_sigma=0.25,
        base_height_target=0.45,
        max_contact_force=400.0,
        feet_height_target=0.12,
        feet_height_high_target=0.24,
    )
    cfg.update(overrides)
    return RewardConfig(**cfg)


def _make_a2_env(num_envs: int = 2):
    from unilab.base import registry

    _ensure_registered()
    return registry.make(
        "A2ArmPosForce",
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override={"reward_config": _default_reward_cfg()},
    )


def test_a2_pos_force_registered():
    """Registers without MuJoCo (decorators run on module import)."""
    from unilab.base import registry

    _ensure_registered()
    assert registry.contains("A2ArmPosForce")


def test_a2_obs_layout_matches_go2_contract():
    """A2 is isomorphic (18 DOF, same arm), so the obs/critic layout equals Go2's."""
    _skip_if_no_mujoco()
    env = _make_a2_env(num_envs=2)
    spec = env.obs_groups_spec
    assert set(spec) == {"obs", "critic"}
    assert spec["obs"] == env._cfg.history.num_actor_history * env._actor_single_obs_dim()
    assert spec["critic"] == env._cfg.history.num_critic_history * env._critic_single_obs_dim()
    assert spec["obs"] == 32 * 76


@pytest.mark.slow
def test_a2_constructs_with_18_dof_and_per_joint_torque():
    _skip_if_no_mujoco()
    env = _make_a2_env()
    assert env._num_action == 18
    # A2 leg limits hip/thigh 120, calf 180 (tiled x4); arm 24/24/24/8/8/8.
    assert np.allclose(env._torque_limits[:12], [120, 120, 180] * 4)
    assert np.allclose(env._torque_limits[12:], [24, 24, 24, 8, 8, 8])


@pytest.mark.slow
def test_a2_init_step_runs_and_torque_within_limits():
    """End-to-end: init + steps must run (all A2 sensors/geoms resolve) with finite
    obs/reward, and the Python PD torque stays within the A2 per-joint limits."""
    _skip_if_no_mujoco()
    env = _make_a2_env(num_envs=2)
    critic_dim = env._cfg.history.num_critic_history * env._critic_single_obs_dim()
    state = env.init_state()
    assert state.obs["obs"].shape == (2, 32 * 76)
    assert state.obs["critic"].shape == (2, critic_dim)
    for _ in range(10):
        state = env.step(np.zeros((2, 18), dtype=np.float64))
    assert np.isfinite(state.reward).all()
    assert np.isfinite(state.obs["obs"]).all()
    assert np.isfinite(state.obs["critic"]).all()
    assert np.all(np.abs(env._last_torque) <= env._torque_limits + 1e-6)
