"""Contract tests for the A2JoystickFlat environment (leg-only Unitree A2).

The A2 leg-only MJCF mirrors the Go2 joystick sensor/geom/leg-ordering
contract (legs FL,FR,RL,RR; foot geoms+sites FL/FR/RL/RR; Go2-named IMU/foot
sensors) and uses <position> actuators, so the env reuses Go2WalkTask
unchanged. These tests prove the A2 model + scene + config + env chain
constructs and steps in MuJoCo as a 12-DOF joystick task."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from unilab.assets import ASSETS_ROOT_PATH


def _skip_if_no_mujoco():
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco.batch_env import BatchEnvPool  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env not available")


def test_a2_robot_xml_compiles_with_12_position_actuators():
    """a2.xml loads standalone and exposes exactly 12 position-style leg
    actuators in the FL,FR,RL,RR x hip,thigh,calf order."""
    mujoco = pytest.importorskip("mujoco")
    xml = ASSETS_ROOT_PATH / "robots" / "a2" / "a2.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    assert model.nu == 12
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert names == [
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
    ]
    # Position actuators carry an affine bias (kp in gainprm[0]); motor actuators do not.
    affine = int(mujoco.mjtBias.mjBIAS_AFFINE)
    assert all(int(model.actuator_biastype[i]) == affine for i in range(model.nu))


def test_a2_scene_loads_with_foot_contacts_and_home_keyframe():
    """scene_flat.xml includes a2.xml + floor, exposes the four foot-contact
    sensors and the joystick foot-pos/IMU sensors, and a home keyframe whose
    qpos is base(7)+12 leg = 19."""
    mujoco = pytest.importorskip("mujoco")
    xml = ASSETS_ROOT_PATH / "robots" / "a2" / "scene_flat.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))

    sensor_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)
    }
    for required in [
        "gyro",
        "local_linvel",
        "upvector",
        "FL_pos",
        "FR_pos",
        "RL_pos",
        "RR_pos",
        "FL_foot_contact",
        "FR_foot_contact",
        "RL_foot_contact",
        "RR_foot_contact",
    ]:
        assert required in sensor_names, f"missing sensor {required}"

    # home keyframe present, qpos length = 7 (free base) + 12 (legs).
    assert model.nkey >= 1
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key_id >= 0
    assert model.nq == 19
    # foot geoms used by the contact sensors exist.
    for g in ["FL", "FR", "RL", "RR", "floor"]:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) >= 0


def _ensure_registered() -> None:
    from unilab.base import registry

    registry.ensure_registries()
    if not registry.contains("A2JoystickFlat"):
        importlib.import_module("unilab.envs.locomotion.a2.joystick")


def test_a2_joystick_registered():
    """Registers without MuJoCo (decorators run on module import)."""
    from unilab.base import registry

    _ensure_registered()
    assert registry.contains("A2JoystickFlat")


def test_a2_joystick_yaml_composes_and_targets_a2():
    """The owner YAML composes under Hydra and selects the A2JoystickFlat task
    with a reward block that injects into the env's reward_config."""
    from hydra import compose, initialize

    with initialize(config_path="../../../../conf/ppo", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=a2_joystick_flat/mujoco"])
    assert cfg.training.task_name == "A2JoystickFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert "tracking_lin_vel" in cfg.reward.scales


def _default_reward_cfg():
    from unilab.envs.locomotion.go2.joystick import RewardConfig

    return RewardConfig(
        scales={
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -5.0,
            "ang_vel_xy": -0.1,
            "base_height": -100.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "contact": 0.24,
            "swing_feet_z": 4.0,
        },
        tracking_sigma=0.25,
        base_height_target=0.45,
    )


def _make_a2_env(num_envs: int = 2):
    from unilab.base import registry

    _ensure_registered()
    return registry.make(
        "A2JoystickFlat",
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override={"reward_config": _default_reward_cfg()},
    )


@pytest.mark.slow
def test_a2_joystick_obs_layout_and_12_dof():
    _skip_if_no_mujoco()
    env = _make_a2_env(num_envs=2)
    assert env._num_action == 12
    assert env.default_angles.shape == (12,)
    assert env.obs_groups_spec == {"obs": 49, "critic": 52}


@pytest.mark.slow
def test_a2_joystick_init_step_runs_finite():
    """End-to-end: init + steps must run (all A2 sensors/geoms resolve) with
    finite obs/reward, proving the leg-only A2 asset satisfies the joystick
    sensor contract on the hot path."""
    _skip_if_no_mujoco()

    env = _make_a2_env(num_envs=2)
    state = env.init_state()
    assert state.obs["obs"].shape == (2, 49)
    assert state.obs["critic"].shape == (2, 52)
    for _ in range(10):
        state = env.step(np.zeros((2, 12), dtype=np.float64))
    assert np.isfinite(state.reward).all()
    assert np.isfinite(state.obs["obs"]).all()
    assert np.isfinite(state.obs["critic"]).all()


@pytest.mark.slow
def test_a2_joystick_dr_on_constructs_and_steps_finite():
    """With DR on (incl. base_link interval push), the env constructs and steps
    with finite obs/reward — proving push_body_name resolves to a real body and
    the mass/COM/kp-kd randomization path is sound.

    registry.make() uses dataclass defaults (not the owner YAML), so DR-on
    fields that default False are forced on explicitly via env_cfg_override.
    randomize_body_mass / randomize_ground_friction / randomize_dof_armature
    are left off: Go2JoystickDomainRandomizationProvider (reused by A2) does
    not cache the required baseline tables, so enabling them raises ValueError
    at validate() time. The YAML-surface is covered by
    test_a2_joystick_domain_rand_fully_configured."""
    _skip_if_no_mujoco()
    from unilab.envs.locomotion.a2.joystick import A2JoystickDomainRandConfig

    dr_on = A2JoystickDomainRandConfig(
        randomize_base_mass=True,
        added_mass_range=[0.0, 8.0],
        randomize_body_mass=False,  # provider does not cache base_body_mass
        random_com=True,
        com_offset_x=[-0.08, 0.08],
        com_offset_y=[-0.08, 0.08],
        com_offset_z=[-0.08, 0.08],
        randomize_ground_friction=False,  # provider does not cache geom friction
        randomize_dof_armature=False,  # provider does not cache dof armature
        randomize_kp=True,
        randomize_kd=True,
        push_robots=True,
        push_interval=400,
        push_body_name="base_link",
    )
    env = _make_a2_env_with_dr(dr_on)

    # DR fields are active on the constructed config.
    assert env._cfg.domain_rand.push_robots is True
    assert env._cfg.domain_rand.push_body_name == "base_link"
    assert env._cfg.domain_rand.randomize_base_mass is True
    assert list(env._cfg.domain_rand.com_offset_z) == [-0.08, 0.08]

    state = env.init_state()
    assert state.obs["obs"].shape == (4, 49)
    # 10 steps exercises reset-time DR (mass/friction/COM/armature/kp-kd) + stepping.
    for _ in range(10):
        state = env.step(np.zeros((4, 12), dtype=np.float64))
    assert np.isfinite(state.reward).all()
    assert np.isfinite(state.obs["obs"]).all()
    assert np.isfinite(state.obs["critic"]).all()


def _make_a2_env_with_dr(dr_cfg):
    """Like _make_a2_env but injects a fully DR-on config."""
    from unilab.base import registry

    _ensure_registered()
    return registry.make(
        "A2JoystickFlat",
        sim_backend="mujoco",
        num_envs=4,
        env_cfg_override={"reward_config": _default_reward_cfg(), "domain_rand": dr_cfg},
    )


def test_a2_joystick_domain_rand_fully_configured():
    """Owner YAML enables DR switches supported by Go2JoystickDomainRandomizationProvider,
    with A2-scale ranges, 3-axis COM, base_link push target, and the 500-iteration budget.

    randomize_body_mass / randomize_ground_friction / randomize_dof_armature are OFF because
    Go2JoystickDomainRandomizationProvider (reused by A2) does not cache the required
    baseline tables (base_body_mass, geom_friction, dof_armature).  The remaining 5 on /
    3 off / gravity off pattern covers the switches actually exercised at runtime."""
    from hydra import compose, initialize

    with initialize(config_path="../../../../conf/ppo", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=a2_joystick_flat/mujoco"])

    dr = cfg.env.domain_rand
    # 5 on / gravity + body_mass + friction + armature off
    assert dr.randomize_base_mass is True
    assert dr.randomize_body_mass is False   # provider does not cache base_body_mass
    assert dr.random_com is True
    assert dr.randomize_gravity is False
    assert dr.randomize_ground_friction is False   # provider does not cache geom friction
    assert dr.randomize_dof_armature is False   # provider does not cache dof armature
    assert dr.randomize_kp is True
    assert dr.randomize_kd is True
    assert dr.push_robots is True
    # 3-axis COM present + A2-scale value
    assert list(dr.com_offset_x) == [-0.08, 0.08]
    assert list(dr.com_offset_y) == [-0.08, 0.08]
    assert list(dr.com_offset_z) == [-0.08, 0.08]
    # A2-calibrated ranges + push target
    assert list(dr.added_mass_range) == [0.0, 8.0]
    assert list(dr.ground_friction_multiplier_range) == [0.5, 1.8]
    assert dr.push_interval == 400
    assert dr.push_body_name == "base_link"
    # bumped budget
    assert cfg.algo.max_iterations == 500
