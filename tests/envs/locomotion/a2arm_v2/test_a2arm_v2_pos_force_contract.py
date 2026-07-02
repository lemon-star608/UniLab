"""Contract tests for the A2ArmV2PosForce environment (A2 + P7v3 + UMI gripper).

The A2ArmV2 MJCF reuses the A2 legs/base/sensor suite VERBATIM (so it mirrors
the Go2/A2 pos-force sensor/geom/leg-ordering contract) and splices in the P7v3
arm (joint1-6 actuated, joint7 welded) + UMI gripper (welded). The env reuses
``Go2ArmPosForceEnv`` unchanged. These tests prove the model + config + env chain
constructs and steps in MuJoCo with A2ArmV2-specific physics: 18 DOF, A2 leg
torque limits (hip/thigh 120, calf 180) + P7v3 arm limits (j1-4 30, j5-6 10),
the shoulder-aligned sphere centre, and the FK-verified home arm pose."""

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
    if not registry.contains("A2ArmV2PosForce"):
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


def _make_v2_env(num_envs: int = 2):
    from unilab.base import registry

    _ensure_registered()
    return registry.make(
        "A2ArmV2PosForce",
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override={"reward_config": _default_reward_cfg()},
    )


def test_a2arm_v2_pos_force_registered():
    """Registers without MuJoCo (decorators run on module import)."""
    from unilab.base import registry

    _ensure_registered()
    assert registry.contains("A2ArmV2PosForce")


def test_a2arm_v2_exported_from_package():
    """Cfg/Env are exported from the go2_arm package (public API, __all__)."""
    import unilab.envs.locomotion.go2_arm as pkg

    assert "A2ArmV2PosForceCfg" in pkg.__all__
    assert "A2ArmV2PosForceEnv" in pkg.__all__
    assert hasattr(pkg, "A2ArmV2PosForceCfg")
    assert hasattr(pkg, "A2ArmV2PosForceEnv")


def test_a2arm_v2_config_geometry():
    """Sphere centre is shoulder-aligned; home pose is [0.5 m, 45 deg, 0]."""
    from unilab.envs.locomotion.go2_arm.pos_force import A2ArmV2PosForceCfg

    cfg = A2ArmV2PosForceCfg()
    # Shoulder height = base 0.465 (per-pair stand) + arm_base 0.1209 + 0.16452 ~= 0.750.
    assert cfg.goal_ee.sphere_center.x_offset == pytest.approx(0.0625)
    assert cfg.goal_ee.sphere_center.z_invariant_offset == pytest.approx(0.750, abs=0.01)
    # Home target [radius, pitch, yaw] (shoulder-relative, base-height invariant).
    assert cfg.goal_ee.init_pos_start[0] == pytest.approx(0.5)
    assert cfg.goal_ee.init_pos_start[1] == pytest.approx(np.pi / 4.0)
    assert cfg.goal_ee.init_pos_start[2] == pytest.approx(0.0)


def test_a2arm_v2_obs_layout_matches_go2_contract():
    """A2ArmV2 is isomorphic (18 DOF, 6-DOF arm), so obs/critic layout == Go2's."""
    _skip_if_no_mujoco()
    env = _make_v2_env(num_envs=2)
    spec = env.obs_groups_spec
    assert set(spec) == {"obs", "critic"}
    assert spec["obs"] == env._cfg.history.num_actor_history * env._actor_single_obs_dim()
    assert spec["critic"] == env._cfg.history.num_critic_history * env._critic_single_obs_dim()
    assert spec["obs"] == 32 * 76


@pytest.mark.slow
def test_a2arm_v2_constructs_with_18_dof_and_per_joint_torque():
    _skip_if_no_mujoco()
    env = _make_v2_env()
    assert env._num_action == 18
    # A2 leg limits hip/thigh 120, calf 180 (tiled x4); P7v3 arm 30/30/30/30/10/10.
    assert np.allclose(env._torque_limits[:12], [120, 120, 180] * 4)
    assert np.allclose(env._torque_limits[12:], [30, 30, 30, 30, 10, 10])


@pytest.mark.slow
def test_a2arm_v2_home_pose_fk():
    """Home keyframe FK: end_link sits at [0.5 m, 45 deg, 0] from the shoulder."""
    _skip_if_no_mujoco()
    import mujoco

    from unilab.assets import ASSETS_ROOT_PATH

    model_path = str(
        ASSETS_ROOT_PATH / "robots" / "a2arm_v2" / "scene_pos_force.xml"
    )
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    kf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    d.qpos[:] = m.key_qpos[kf]
    mujoco.mj_forward(m, d)

    armbase_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_base_link")
    ee_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "end_link")
    shoulder = d.xpos[armbase_id] + np.array([0.0, 0.0, 0.16452])
    rel = d.xpos[ee_id] - shoulder
    radius = float(np.linalg.norm(rel))
    pitch = float(np.arcsin(np.clip(rel[2] / radius, -1.0, 1.0)))
    yaw = float(np.arctan2(rel[1], rel[0]))

    assert radius == pytest.approx(0.5, abs=0.02)
    assert np.degrees(pitch) == pytest.approx(45.0, abs=3.0)
    assert np.degrees(yaw) == pytest.approx(0.0, abs=3.0)


@pytest.mark.slow
def test_a2arm_v2_init_step_runs_and_torque_within_limits():
    """End-to-end: init + steps must run (all A2ArmV2 sensors/geoms resolve) with
    finite obs/reward, and the Python PD torque stays within the per-joint limits."""
    _skip_if_no_mujoco()
    env = _make_v2_env(num_envs=2)
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
