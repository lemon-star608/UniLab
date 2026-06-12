"""Contract tests for the Go2ArmPosForce environment."""

from __future__ import annotations

import importlib
from types import MethodType

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
        base_height_target=0.45,
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
def test_clip_actions_and_obs_bound_runaway():
    """UniFP ±100 clamps (clip_actions / clip_observations): a blown-up policy
    (huge actions) must not poison the obs (last_actions block) or the action_rate
    reward — the safety valve that prevents the feedback runaway that drove early
    A2 training to value-loss=inf and the std>=0 crash."""
    _skip_if_no_mujoco()
    env = _make_env(num_envs=2)
    assert env._clip_actions == 100.0 and env._clip_obs == 100.0
    env.init_state()
    state = env.step(np.full((2, 18), 1.0e6, dtype=np.float64))
    # raw action clamped BEFORE it enters obs / action_rate reward
    assert np.all(np.abs(state.info["current_actions"]) <= 100.0 + 1e-6)
    # actor + critic obs bounded (no 1e6 leaking via the last_actions block)
    assert np.all(np.abs(state.obs["obs"]) <= 100.0 + 1e-6)
    assert np.all(np.abs(state.obs["critic"]) <= 100.0 + 1e-6)
    assert np.isfinite(state.reward).all()


@pytest.mark.slow
def test_pos_force_external_forces_apply_and_observe():
    _skip_if_no_mujoco()
    env = _make_env(
        num_envs=4,
        env_cfg_override={
            "commands": {
                "force_start_step": 0,
                "push_gripper_interval_s_cmd": [0.1, 0.2],
                "push_gripper_interval_s_ext": [0.1, 0.2],
                "push_base_interval_s_cmd": [0.1, 0.2],
                "push_base_interval_s_ext": [0.1, 0.2],
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


def test_obs_noise_matches_unifp_effective_magnitude():
    """Observation noise must reproduce UniFP's effective magnitude.

    UniFP adds noise to the ALREADY-SCALED observation, so the effective
    perturbation equals ``noise_scale * obs_scale`` (with noise_level=1):
        dof_pos:  noise_scales.dof_pos(0.01) * obs_scales.dof_pos(1.0)  = 0.01
        dof_vel:  noise_scales.dof_vel(1.5)  * obs_scales.dof_vel(0.05) = 0.075
    The migrated config over-noised these (0.03 and 0.5 -> 3x / 6.7x too large),
    which corrupts exactly the proprioception the CSE estimator consumes.
    """
    from unilab.envs.locomotion.go2_arm.pos_force import ObsScales, PosForceNoiseConfig

    n = PosForceNoiseConfig()
    s = ObsScales()
    assert n.level == pytest.approx(1.0)
    assert n.scale_joint_angle == pytest.approx(0.01 * s.dof_pos)
    assert n.scale_joint_vel == pytest.approx(1.5 * s.dof_vel)
    # Orientation / angular-velocity noise already matched UniFP and must stay put.
    assert n.scale_orn == pytest.approx(0.05)
    assert n.scale_ang_vel == pytest.approx(0.2)


def test_termination_uses_unifp_roll_pitch_thresholds():
    """UniFP terminates on |pitch| > 1.0 rad or |roll| > 0.8 rad.

    The migration used ``gravity_z <= 0`` (~90 deg tip on either axis), which lets
    the robot fall far past UniFP's per-axis Euler limits before terminating and
    changes the survivable-state set / ``alive`` reward integral.
    """
    from types import SimpleNamespace

    from unilab.envs.locomotion.go2_arm.pos_force import Go2ArmPosForceEnv

    def quat_roll(r: float) -> np.ndarray:
        return np.array([np.cos(r / 2.0), np.sin(r / 2.0), 0.0, 0.0])

    def quat_pitch(p: float) -> np.ndarray:
        return np.array([np.cos(p / 2.0), 0.0, np.sin(p / 2.0), 0.0])

    quats = np.stack(
        [
            np.array([1.0, 0.0, 0.0, 0.0]),  # upright -> alive
            quat_roll(0.9),  # |roll| 0.9 > 0.8 -> terminate
            quat_roll(0.7),  # |roll| 0.7 < 0.8 -> alive
            quat_roll(-0.9),  # roll asymmetry: -0.9 also terminates
            quat_pitch(1.1),  # |pitch| 1.1 > 1.0 -> terminate
            quat_pitch(0.9),  # |pitch| 0.9 < 1.0 -> alive (gravity_z<=0 would keep alive too)
        ]
    )
    stub = SimpleNamespace()
    term = Go2ArmPosForceEnv._compute_terminated(stub, quats)
    assert term.dtype == bool
    assert term.tolist() == [False, True, False, True, True, False]


def test_force_schedule_hold_is_fixed_settling_independent_of_duration():
    """UniFP ramps force up over push_duration, HOLDS at peak for a fixed
    settling_time (gripper 0.5 s, base 1.0 s), then ramps down. The migration
    made hold ~ push_duration/3, so the peak-hold scaled with the sampled
    duration and shrank the steady-state force-holding the policy practises.

    Contract: the peak-hold length is governed by ``settling`` and is
    independent of the sampled ``push_duration``.
    """
    from unilab.envs.locomotion.go2_arm.pos_force import _ForceSchedule

    def peak_hold(duration: int, settling: int) -> int:
        sched = _ForceSchedule(
            num_envs=1,
            mag_range=(10.0, 10.0),
            interval_range=(1, 1),
            duration_range=(duration, duration),
            prob=1.0,
            dtype=np.float64,
            settling=settling,
        )
        peak = 0
        for _ in range(2 * duration + settling + 5):
            f = float(sched.step(enabled=True)[0, 0])
            if np.isclose(f, 10.0):
                peak += 1
        return peak

    # Same settling, very different push_duration -> identical peak-hold length.
    assert peak_hold(duration=5, settling=8) == peak_hold(duration=15, settling=8)
    # The peak hold lasts at least the configured settling (a sustained hold).
    assert peak_hold(duration=5, settling=8) >= 8
    # Longer settling -> strictly longer peak hold (hold is settling-driven).
    assert peak_hold(duration=5, settling=24) > peak_hold(duration=5, settling=8)


def test_force_commands_have_separate_cmd_ext_intervals():
    """UniFP uses distinct cmd vs ext force-episode intervals (gripper cmd
    [5,10] / ext [6,12]; base cmd [5,10] / ext [8,14]); the migration collapsed
    them into one shared interval. Restore the separate fields + settling times.
    """
    from unilab.envs.locomotion.go2_arm.pos_force import PosForceCommandsConfig

    c = PosForceCommandsConfig()
    assert c.push_gripper_interval_s_cmd == [5.0, 10.0]
    assert c.push_gripper_interval_s_ext == [6.0, 12.0]
    assert c.push_base_interval_s_cmd == [5.0, 10.0]
    assert c.push_base_interval_s_ext == [8.0, 14.0]
    assert c.settling_time_force_gripper_s == pytest.approx(0.5)
    assert c.settling_time_force_base_s == pytest.approx(1.0)


def test_push_is_velocity_impulse_not_force():
    """UniFP ``_push_robots`` is a base-velocity impulse (max_push_vel_xy=0.3 m/s),
    not the 0.3 N ``xfrc`` no-op the migration shipped (0.3 N on a ~15 kg base is
    imperceptible). The shared force-based interval push must be disabled.
    """
    from unilab.envs.locomotion.go2_arm.pos_force import PosForceDomainRandConfig

    dr = PosForceDomainRandConfig()
    assert dr.velocity_push is True
    assert dr.max_push_vel_xy == pytest.approx(0.3)
    # build_interval_push_plan / validate_interval_push_support both gate on
    # push_robots; keeping it False disables the force-based no-op.
    assert dr.push_robots is False
    assert not hasattr(dr, "max_force")


def _velocity_push_stub(dr):
    from types import SimpleNamespace

    from unilab.envs.locomotion.go2_arm.pos_force import (
        Go2ArmPosForceEnv,
        PosForceCommandsConfig,
    )

    n = 2000
    base_vel = np.zeros((n, 3), dtype=np.float64)
    stub = SimpleNamespace(
        _num_envs=n,
        _np_dtype=np.float64,
        step_counter=dr.push_interval,  # divisible -> push fires
        _backend=SimpleNamespace(get_base_lin_vel=lambda: base_vel),
        _cfg=SimpleNamespace(domain_rand=dr, commands=PosForceCommandsConfig()),
    )
    stub._command_is_moving = MethodType(Go2ArmPosForceEnv._command_is_moving, stub)
    commands = np.zeros((n, 15), dtype=np.float64)
    commands[n // 2 :, 0] = 1.0  # second half moving in x; first half standing
    Go2ArmPosForceEnv._maybe_apply_velocity_push(stub, commands)
    moving_max = float(np.abs(base_vel[n // 2 :, 0:2]).max())
    standing_max = float(np.abs(base_vel[: n // 2, 0:2]).max())
    return base_vel, moving_max, standing_max


def test_velocity_push_standing_scale_defaults_to_unifp_1x():
    """The push overwrites base x/y velocity; the standing amplification defaults
    to 1.0 -- UniFP's _push_robots 2.5x branch is dead code (effectively 1x), so
    standing envs are NOT amplified by default."""
    from unilab.envs.locomotion.go2_arm.pos_force import PosForceDomainRandConfig

    np.random.seed(0)
    dr = PosForceDomainRandConfig()
    assert dr.velocity_push_standing_scale == pytest.approx(1.0)
    base_vel, moving_max, standing_max = _velocity_push_stub(dr)
    assert np.any(base_vel[:, 0:2] != 0.0)  # a real velocity impulse, not a no-op
    assert np.all(base_vel[:, 2] == 0.0)  # vertical velocity untouched
    # 1x: standing is NOT amplified -> both bounded by vmax
    assert moving_max <= dr.max_push_vel_xy + 1e-9
    assert standing_max <= dr.max_push_vel_xy + 1e-9


def test_velocity_push_standing_scale_amplifies_when_configured():
    """Setting velocity_push_standing_scale > 1 amplifies standing-env pushes."""
    from unilab.envs.locomotion.go2_arm.pos_force import PosForceDomainRandConfig

    np.random.seed(0)
    dr = PosForceDomainRandConfig()
    dr.velocity_push_standing_scale = 2.5
    _, moving_max, standing_max = _velocity_push_stub(dr)
    assert moving_max <= dr.max_push_vel_xy + 1e-9
    assert standing_max > dr.max_push_vel_xy  # amplified beyond the base bound
    assert standing_max <= 2.5 * dr.max_push_vel_xy + 1e-9


def test_velocity_push_only_fires_on_interval():
    """No push on steps that are not multiples of push_interval."""
    from types import SimpleNamespace

    from unilab.envs.locomotion.go2_arm.pos_force import (
        Go2ArmPosForceEnv,
        PosForceCommandsConfig,
        PosForceDomainRandConfig,
    )

    n = 4
    base_vel = np.zeros((n, 3), dtype=np.float64)
    dr = PosForceDomainRandConfig()
    stub = SimpleNamespace(
        _num_envs=n,
        _np_dtype=np.float64,
        step_counter=dr.push_interval + 1,  # NOT divisible -> no push
        _backend=SimpleNamespace(get_base_lin_vel=lambda: base_vel),
        _cfg=SimpleNamespace(domain_rand=dr, commands=PosForceCommandsConfig()),
    )
    stub._command_is_moving = MethodType(Go2ArmPosForceEnv._command_is_moving, stub)
    Go2ArmPosForceEnv._maybe_apply_velocity_push(stub, np.zeros((n, 15)))
    assert np.all(base_vel == 0.0)


def test_soft_dof_pos_limits_shrinks_to_middle_fraction():
    """UniFP penalizes dof_pos against SOFT limits = the middle ``soft`` fraction
    of each joint range (m +/- 0.5*r*soft). The migration used the hard limits
    (soft_dof_pos_limit was dead), so the -10 penalty triggered too late.
    """
    from unilab.envs.locomotion.go2_arm.pos_force import _soft_dof_pos_limits

    hard = np.array([[-1.0, 1.0], [0.0, 2.0], [-3.0, 1.0]])  # ranges 2, 2, 4
    soft = _soft_dof_pos_limits(hard, 0.8)
    # middle 80% about each midpoint (mid 0/1/-1, half-range 1/1/2)
    assert np.allclose(soft, [[-0.8, 0.8], [0.2, 1.8], [-2.6, 0.6]])
    # soft=1.0 -> unchanged (hard); narrower for soft<1
    assert np.allclose(_soft_dof_pos_limits(hard, 1.0), hard)
    assert (soft[:, 1] - soft[:, 0] < hard[:, 1] - hard[:, 0]).all()


def test_leg_torque_limit_accepts_per_joint_list():
    """A2 has non-uniform leg torque limits (hip/thigh 120, calf 180). The env must
    accept a length-3 per-leg spec (tiled x4) and a full length-12 list, not only a scalar."""
    from unilab.envs.locomotion.go2_arm.pos_force import _expand_leg_torque_limit

    assert np.allclose(_expand_leg_torque_limit(24.0), np.full(12, 24.0))
    assert np.allclose(_expand_leg_torque_limit([120.0, 120.0, 180.0]), [120, 120, 180] * 4)
    full = list(range(12))
    assert np.allclose(_expand_leg_torque_limit(full), full)


def test_expand_gain_tiles_per_leg_spec():
    """A2 leg PD gains are per-joint (hip/thigh/calf). _expand_gain must tile a length-3
    spec across the 12 leg joints, while leaving scalars and exact-size lists unchanged."""
    from unilab.envs.locomotion.go2_arm.base import _expand_gain

    assert np.allclose(_expand_gain("leg_kp", 60.0, 35.0, 12), np.full(12, 60.0))
    assert np.allclose(_expand_gain("leg_kp", [100.0, 100.0, 150.0], 35.0, 12), [100, 100, 150] * 4)
    arm = [95.0, 115.0, 100.0, 52.0, 54.0, 55.0]
    assert np.allclose(_expand_gain("arm_kp", arm, 35.0, 6), arm)


def test_foot_friction_dr_randomizes_feet():
    """The foot geom has priority=1, so the ground-friction DR never reaches the
    contact (it scales the floor, which the foot overrides). The FOOT friction
    must itself be randomized per-env in [0.5, 1.8] (matching UniFP's per-env
    foot-shape friction): all 4 feet share one per-env value, non-foot geoms
    untouched.
    """
    from types import SimpleNamespace

    from unilab.envs.locomotion.go2_arm.pos_force import (
        Go2ArmPosForceEnv,
        PosForceDomainRandConfig,
    )

    np.random.seed(0)
    n, ngeom = 32, 40
    foot_ids = np.array([5, 9, 13, 17])
    base = np.tile(np.array([0.8, 0.02, 0.01]), (ngeom, 1))
    payload = SimpleNamespace(geom_friction=np.broadcast_to(base, (n, ngeom, 3)).copy())
    dr = PosForceDomainRandConfig()
    stub = SimpleNamespace(
        _num_envs=n,
        _np_dtype=np.float64,
        _foot_geom_ids=foot_ids,
        _base_geom_friction_full=base.copy(),
        _cfg=SimpleNamespace(domain_rand=dr),
    )
    Go2ArmPosForceEnv._apply_foot_friction_dr(stub, payload, np.arange(n, dtype=np.int32))

    assert dr.randomize_foot_friction is True
    assert dr.foot_friction_range == [0.5, 1.8]
    ff = payload.geom_friction[:, foot_ids, 0]
    assert ff.std() > 0.05  # varies across envs (not the fixed 0.8 no-op)
    assert ff.min() >= 0.5 - 1e-9 and ff.max() <= 1.8 + 1e-9
    assert np.allclose(ff, ff[:, :1])  # all 4 feet share one per-env bucket
    other = np.setdiff1d(np.arange(ngeom), foot_ids)
    assert np.allclose(payload.geom_friction[:, other, 0], 0.8)  # non-foot untouched


@pytest.mark.slow
def test_dof_pos_limits_are_soft_not_hard():
    """End-to-end: the env's dof_pos_limits are the soft (middle 0.8) limits,
    strictly inside the model's hard joint range."""
    _skip_if_no_mujoco()
    from unilab.envs.locomotion.go2_arm.pos_force import _soft_dof_pos_limits

    env = _make_env(num_envs=1)
    soft = env._dof_pos_limits
    hard = np.asarray(env._backend.get_joint_range(), dtype=soft.dtype)[:18]
    assert np.allclose(soft, _soft_dof_pos_limits(hard, 0.8))
    assert (soft[:, 1] - soft[:, 0] < hard[:, 1] - hard[:, 0] - 1e-9).all()


@pytest.mark.slow
def test_foot_friction_varies_across_envs_after_reset():
    """End-to-end: after reset the per-env foot friction (privileged readback)
    actually varies in [0.5, 1.8] — the DR now reaches the feet."""
    _skip_if_no_mujoco()
    env = _make_env(num_envs=8)
    env.init_state()
    fr = env._dr_friction[:, 0]
    assert fr.min() >= 0.5 - 1e-6
    assert fr.max() <= 1.8 + 1e-6
    assert fr.std() > 0.05  # genuinely varies across envs (not fixed 0.8)


@pytest.mark.slow
def test_velocity_push_perturbs_base_in_sim():
    """End-to-end: the in-env velocity write reaches the physics state and moves
    the base (the old 0.3 N xfrc force was a near-no-op)."""
    _skip_if_no_mujoco()
    env = _make_env(
        num_envs=4,
        env_cfg_override={
            "domain_rand": {
                "velocity_push": True,
                "push_interval": 1,  # push every control step
                "max_push_vel_xy": 2.0,  # large, unambiguous impulse
            }
        },
    )
    env.init_state()
    max_speed = 0.0
    for _ in range(8):
        env.step(np.zeros((4, 18), dtype=np.float64))
        max_speed = max(max_speed, float(np.abs(env._backend.get_base_lin_vel()[:, 0:2]).max()))
    assert max_speed > 0.5
