"""Go2 + AirBot-Play unified position-force whole-body control task.

Faithful port of the UniFP (CoRL 2025) ``go2arm_pos_force`` task into UniLab's
NpEnv contract. Unlike ``Go2ArmManipLoco`` (IK-driven arm), here all 18 joints
(12 legs + 6 arm) are controlled by the RL policy through a Python PD law fed to
``<motor>`` actuators, and the policy jointly tracks end-effector position and
commanded forces. Locomotion, the full observation/command layout, the
external-force push controller, the UniFP reward set, and the CSE estimator
targets are all implemented here.

Known gaps (deferred): rough-flat terrain randomization — this task runs on the
flat MJCF floor only.

Control law (UniFP ``_compute_torques``), recomputed every physics substep:

    target  = default_dof_pos + action * motor_strength * action_scale
    torque  = kp * (target - q) - kd * qd
    torque  = clip(torque, -torque_limit, torque_limit)

The end-effector goal is expressed in UniFP's yaw-aligned, gravity-referenced
sphere frame (terrain-relative z), not the arm-base-local frame used by
``manip_loco``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr.types import ResetPlan
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import (
    np_quat_apply,
    np_quat_apply_inverse,
    np_yaw_quat,
)
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext, run_reward_dispatch
from unilab.envs.locomotion.go2_arm.base import (
    Go2ArmBaseCfg,
    Go2ArmBaseEnv,
    Go2ArmSensor,
    build_go2_arm_position_gains,
)

# Command vector layout (15 dims), matching UniFP INDEX_* constants.
CMD_VEL = slice(0, 3)  # base lin_vel_x, lin_vel_y, ang_vel_yaw
CMD_EE_POS = slice(3, 6)  # EE goal in sphere coords [radius, pitch, yaw]
CMD_EE_ORN = slice(6, 9)  # EE goal orientation [roll, pitch, yaw]
CMD_EE_FORCE = slice(9, 12)  # EE force command [fx, fy, fz]
CMD_BASE_FORCE = slice(12, 15)  # base force command [fx, fy, fz]
NUM_COMMANDS = 15

NUM_ACTIONS = 18
NUM_LEG = 12
NUM_ARM = 6


def _default_model_file() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "go2_arm" / "scene_pos_force.xml")


def _default_scene() -> SceneCfg:
    return SceneCfg(model_file=_default_model_file())


def sphere2cart(sphere: np.ndarray) -> np.ndarray:
    """UniFP convention: sphere [l, pitch, yaw] -> cart [x, y, z]."""
    l = sphere[..., 0]
    pitch = sphere[..., 1]
    yaw = sphere[..., 2]
    x = l * np.cos(pitch) * np.cos(yaw)
    y = l * np.cos(pitch) * np.sin(yaw)
    z = l * np.sin(pitch)
    return np.stack([x, y, z], axis=-1)


def cart2sphere(cart: np.ndarray) -> np.ndarray:
    """UniFP convention: cart [x, y, z] -> sphere [l, pitch, yaw]."""
    cart = np.asarray(cart)
    xy_len = np.linalg.norm(cart[..., :2], axis=-1)
    l = np.linalg.norm(cart, axis=-1)
    pitch = np.arctan2(cart[..., 2], xy_len)
    yaw = np.arctan2(cart[..., 1], cart[..., 0])
    return np.stack([l, pitch, yaw], axis=-1)


def _roll_pitch_from_quat(quat: np.ndarray) -> np.ndarray:
    """Extract (roll, pitch) from a (w, x, y, z) quaternion batch."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.stack([roll, pitch], axis=1)


class _ForceSchedule:
    """Per-env trapezoidal force episodes (ramp up -> hold -> ramp down).

    Faithful in spirit to UniFP's ``_push_gripper`` / ``_push_robot_base``
    state machines: each env idles for a random interval, then runs one force
    episode toward a random target with a symmetric ramp and a hold, then idles
    again. ``prob`` is the probability that any given episode actually fires;
    otherwise the force stays zero for that interval (UniFP's ``freed_envs``).
    The returned force is per-env ``(N, 3)`` in newtons.
    """

    def __init__(
        self,
        num_envs: int,
        mag_range: Sequence[float],
        interval_range: Sequence[int],
        duration_range: Sequence[int],
        prob: float,
        dtype: Any,
        z_scale: float = 1.0,
    ):
        self._n = num_envs
        self._mag = mag_range
        self._interval = interval_range
        self._duration = duration_range
        self._prob = float(prob)
        self._dtype = dtype
        # UniFP attenuates / zeros the z-component for base forces:
        # base-ext z is scaled by force_z_base_ext_scale (0.05); base-cmd z is 0.
        self._z_scale = float(z_scale)
        self.current = np.zeros((num_envs, 3), dtype=dtype)
        self._target = np.zeros((num_envs, 3), dtype=dtype)
        self._active = np.zeros((num_envs,), dtype=bool)
        self._elapsed = np.zeros((num_envs,), dtype=np.int32)
        self._ramp = np.ones((num_envs,), dtype=np.int32)
        self._hold = np.zeros((num_envs,), dtype=np.int32)
        self._timer = np.zeros((num_envs,), dtype=np.int32)
        self.reset(np.arange(num_envs, dtype=np.int32))

    def reset(self, env_ids: np.ndarray) -> None:
        n = len(env_ids)
        if n == 0:
            return
        self.current[env_ids] = 0.0
        self._target[env_ids] = 0.0
        self._active[env_ids] = False
        self._elapsed[env_ids] = 0
        self._timer[env_ids] = np.random.randint(self._interval[0], self._interval[1] + 1, size=n)

    def _start_episodes(self, env_ids: np.ndarray) -> None:
        n = len(env_ids)
        fire = np.random.uniform(size=n) < self._prob
        fire_ids = env_ids[fire]
        if len(fire_ids) > 0:
            mags = np.random.uniform(self._mag[0], self._mag[1], size=(len(fire_ids), 3))
            mags[:, 2] *= self._z_scale
            self._target[fire_ids] = mags.astype(self._dtype)
            dur = np.random.randint(self._duration[0], self._duration[1] + 1, size=len(fire_ids))
            self._ramp[fire_ids] = np.maximum(1, dur // 3)
            self._hold[fire_ids] = dur - 2 * np.maximum(1, dur // 3)
            self._elapsed[fire_ids] = 0
            self._active[fire_ids] = True
        # Non-firing envs simply re-arm their idle timer.
        idle_ids = env_ids[~fire]
        if len(idle_ids) > 0:
            self._timer[idle_ids] = np.random.randint(
                self._interval[0], self._interval[1] + 1, size=len(idle_ids)
            )

    def step(self, enabled: bool) -> np.ndarray:
        """Advance one control step and return the current force ``(N, 3)``."""
        if not enabled:
            self.current[:] = 0.0
            return self.current
        # Idle envs: count down; start an episode when the timer elapses.
        idle = ~self._active
        self._timer[idle] -= 1
        due = idle & (self._timer <= 0)
        if np.any(due):
            self._start_episodes(np.flatnonzero(due).astype(np.int32))
        # Active envs: evaluate the trapezoidal profile.
        active_ids = np.flatnonzero(self._active).astype(np.int32)
        if len(active_ids) > 0:
            e = self._elapsed[active_ids].astype(self._dtype)
            ramp = self._ramp[active_ids].astype(self._dtype)
            hold = self._hold[active_ids].astype(self._dtype)
            total = 2.0 * ramp + hold
            frac = np.ones_like(e)
            up = e < ramp
            frac[up] = e[up] / ramp[up]
            down = e >= (ramp + hold)
            frac[down] = np.clip((total[down] - e[down]) / ramp[down], 0.0, 1.0)
            self.current[active_ids] = self._target[active_ids] * frac[:, None]
            self._elapsed[active_ids] += 1
            done = self._elapsed[active_ids] >= total.astype(np.int32)
            if np.any(done):
                done_ids = active_ids[done]
                self._active[done_ids] = False
                self.current[done_ids] = 0.0
                self._target[done_ids] = 0.0
                self._timer[done_ids] = np.random.randint(
                    self._interval[0], self._interval[1] + 1, size=len(done_ids)
                )
        return self.current


# ── Configuration dataclasses ────────────────────────────────────────────────


@dataclass
class ObsScales:
    lin_vel: float = 2.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    ee_force: float = 0.01
    base_force: float = 0.01
    ee_sphe_radius: float = 0.5
    ee_sphe_pitch: float = 1.0
    ee_sphe_yaw: float = 1.3
    ee_orn: float = 1.0


@dataclass
class PosForceNoiseConfig:
    level: float = 1.0  # UniFP add_noise=True, noise_level=1.0 (was 0.0 -> perfect obs)
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 0.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_ang_vel: float = 0.2
    scale_orn: float = 0.05


@dataclass
class SphereCenter:
    x_offset: float = 0.2  # forward of base, in yaw frame
    y_offset: float = 0.0
    z_invariant_offset: float = 0.6  # terrain-relative height of the sphere center


@dataclass
class GoalEEConfig:
    command_mode: str = "sphere"
    arm_induced_pitch: float = 0.38
    sphere_center: SphereCenter = field(default_factory=SphereCenter)
    # Sphere sampling ranges [radius, pitch, yaw] (UniFP pos_l / pos_p / pos_y).
    pos_l: list[float] = field(default_factory=lambda: [0.25, 0.60])
    pos_p: list[float] = field(default_factory=lambda: [-2.0 * np.pi / 5.0, 2.0 * np.pi / 5.0])
    pos_y: list[float] = field(default_factory=lambda: [-3.0 * np.pi / 5.0, 3.0 * np.pi / 5.0])
    delta_orn_r: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    delta_orn_p: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    delta_orn_y: list[float] = field(default_factory=lambda: [-0.5, 0.5])
    init_pos_start: list[float] = field(default_factory=lambda: [0.5, np.pi / 4.0, 0.0])
    init_pos_end: list[float] = field(default_factory=lambda: [0.5, 0.0, 0.0])
    traj_time: list[float] = field(default_factory=lambda: [1.0, 3.0])
    hold_time: list[float] = field(default_factory=lambda: [0.5, 2.0])
    collision_upper_limits: list[float] = field(default_factory=lambda: [0.25, 0.2, -0.15])
    collision_lower_limits: list[float] = field(default_factory=lambda: [-0.7, -0.2, -0.8])
    underground_limit: float = -0.7
    num_collision_check_samples: int = 10
    num_resample_attempts: int = 10


@dataclass
class PosForceCommandsConfig:
    lin_vel_x: list[float] = field(default_factory=lambda: [-0.6, 0.6])
    lin_vel_y: list[float] = field(default_factory=lambda: [-0.4, 0.4])
    ang_vel_yaw: list[float] = field(default_factory=lambda: [-0.6, 0.6])
    lin_vel_x_clip: float = 0.1
    lin_vel_y_clip: float = 0.1
    ang_vel_yaw_clip: float = 0.2
    zero_vel_cmd_prob: float = 0.3
    # Once external forces are active, UniFP raises the zero-velocity command
    # probability so the policy practises standing/holding under force.
    zero_vel_cmd_prob_after_force: float = 0.8
    resample_time_s: float = 5.0
    # Force command magnitudes (N). cmd = commanded force the policy must exert
    # (drives the EE/base setpoint offset); ext = external disturbance force
    # applied to the body (becomes the realized force in the observation).
    max_push_force_xyz_gripper_cmd: list[float] = field(default_factory=lambda: [-18.0, 18.0])
    max_push_force_xyz_gripper_ext: list[float] = field(default_factory=lambda: [-15.0, 15.0])
    max_push_force_xyz_base_cmd: list[float] = field(default_factory=lambda: [-25.0, 25.0])
    max_push_force_xyz_base_ext: list[float] = field(default_factory=lambda: [-20.0, 20.0])
    # Force-episode timing in seconds [min, max]: idle interval and active duration.
    push_gripper_interval_s: list[float] = field(default_factory=lambda: [5.0, 10.0])
    push_gripper_duration_s: list[float] = field(default_factory=lambda: [0.5, 1.5])
    push_base_interval_s: list[float] = field(default_factory=lambda: [6.0, 12.0])
    push_base_duration_s: list[float] = field(default_factory=lambda: [0.5, 1.5])
    gripper_forced_prob_cmd: float = 0.4
    gripper_forced_prob_ext: float = 0.3
    base_forced_prob_cmd: float = 0.4
    base_forced_prob_ext: float = 0.3
    # Force/compliance stiffness used to offset the EE position / base velocity
    # setpoint: ee_goal += F/gripper_force_kp, base_vel_cmd += F/base_force_kd.
    gripper_force_kp: float = 200.0
    base_force_kd: float = 200.0
    # UniFP attenuates the external base force's vertical component to 5%.
    force_z_base_ext_scale: float = 0.05
    # Force curriculum: external forces start after this many control steps.
    force_start_step: int = 6000


@dataclass
class PosForceControlConfig:
    action_scale: float = 0.25
    arm_action_scale: float = 0.12
    simulate_action_latency: bool = False
    # UniFP applies a 3-control-step action delay (sim2real latency model).
    action_delay_steps: int = 3
    # PD gains (UniFP go2arm); reused via build_go2_arm_position_gains.
    Kp: float = 35.0
    Kd: float = 0.5
    leg_kp: float = 60.0
    leg_kd: float = 2.0
    arm_kp: list[float] = field(default_factory=lambda: [95.0, 115.0, 100.0, 52.0, 54.0, 55.0])
    arm_kd: list[float] = field(default_factory=lambda: [3.5, 3.8, 2.5, 1.5, 1.5, 1.5])
    # Per-joint torque limits (legs + joint1-3: 24 Nm, wrist joint4-6: 8 Nm).
    leg_torque_limit: float = 24.0
    arm_torque_limit: list[float] = field(default_factory=lambda: [24.0, 24.0, 24.0, 8.0, 8.0, 8.0])


@dataclass
class PosForceDomainRandConfig:
    # Field names MUST match the shared DR applier contract in unilab.dr.dr_utils
    # (it reads them via getattr; UniFP-style names silently no-op).
    randomize_ground_friction: bool = True
    # Multiplier on the model's default ground friction (default ~1.0 => ~[0.5, 1.8]).
    ground_friction_multiplier_range: list[float] = field(default_factory=lambda: [0.5, 1.8])
    randomize_base_mass: bool = True
    added_mass_range: list[float] = field(default_factory=lambda: [0.0, 4.0])
    random_com: bool = True
    com_offset_x: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    com_offset_y: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    randomize_motor_strength: bool = True
    leg_motor_strength_range: list[float] = field(default_factory=lambda: [0.85, 1.15])
    arm_motor_strength_range: list[float] = field(default_factory=lambda: [0.85, 1.15])
    # Gripper payload mass (UniFP adds 0-0.12 kg to the wrist/gripper link).
    randomize_gripper_mass: bool = True
    gripper_added_mass_range: list[float] = field(default_factory=lambda: [0.0, 0.12])
    # Root-velocity push DR (handled by the shared interval push plan).
    push_robots: bool = True
    push_interval: int = 400
    max_force: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.0])
    push_body_name: str = "base"


@dataclass
class GaitConfig:
    cycle_time: float = 0.64
    target_joint_pos_scale: float = 0.17
    target_joint_pos_thd: float = 0.5


@dataclass
class RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float = 0.25
    tracking_ee_sigma: float = 1.0
    base_height_target: float = 0.40
    soft_dof_pos_limit: float = 0.8
    soft_torque_limit: float = 0.9
    max_contact_force: float = 200.0
    # Leg gait-reference tracking: exp(-L1_err * ref_dof_scale) (UniFP uses 0.1).
    ref_dof_scale: float = 0.1
    # stand_still: exp(-L1_err * stand_still_scale) at zero command (UniFP 0.05).
    stand_still_scale: float = 0.05
    feet_air_time_threshold: float = 0.5
    feet_height_target: float = 0.10
    feet_height_high_target: float = 0.20


@dataclass
class HistoryConfig:
    # Actor sees a long proprioceptive history (UniFP frame_stack=32); the CSE
    # estimator compresses it. The critic sees the current privileged step, so
    # the estimator target is critic_obs[:, :12] (estimator.target_start=0).
    num_actor_history: int = 32
    num_critic_history: int = 1


@dataclass
class ResetConfig:
    leg_pos_scale_range: list[float] = field(default_factory=lambda: [0.5, 1.5])
    arm_init_dof_pos_range: float = 0.3
    yaw_range: float = np.pi / 2.0


@registry.envcfg("Go2ArmPosForce")
@dataclass
class Go2ArmPosForceCfg(Go2ArmBaseCfg):
    scene: SceneCfg = field(default_factory=_default_scene)
    model_file: str = field(default_factory=_default_model_file)
    max_episode_seconds: float = 20.0
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
    obs_scales: ObsScales = field(default_factory=ObsScales)
    noise_config: PosForceNoiseConfig = field(default_factory=PosForceNoiseConfig)  # type: ignore[assignment]
    control_config: PosForceControlConfig = field(default_factory=PosForceControlConfig)  # type: ignore[assignment]
    commands: PosForceCommandsConfig = field(default_factory=PosForceCommandsConfig)
    goal_ee: GoalEEConfig = field(default_factory=GoalEEConfig)
    domain_rand: PosForceDomainRandConfig = field(default_factory=PosForceDomainRandConfig)
    gait: GaitConfig = field(default_factory=GaitConfig)
    reward_config: RewardConfig | None = None
    history: HistoryConfig = field(default_factory=HistoryConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    sensor: Go2ArmSensor = field(default_factory=Go2ArmSensor)  # type: ignore[assignment]


# ── Domain randomization provider ────────────────────────────────────────────


class Go2ArmPosForceDRProvider(LocomotionDRProvider):
    """Reset/interval DR for the pos-force task.

    Velocity commands are sampled into the 15-dim command vector; motor-strength
    is owned by the env (sampled here and cached) so it can scale the action in
    the Python PD law and feed the privileged observation. EE goals and history
    buffers are reset through env hooks.
    """

    def __init__(
        self,
        *,
        base_body_mass: np.ndarray | None = None,
        base_geom_friction: np.ndarray | None = None,
        ground_geom_id: int | None = None,
    ):
        self._base_body_mass = base_body_mass
        self._base_geom_friction = base_geom_friction
        self._ground_geom_id = ground_geom_id

    def _get_reset_randomization_baselines(
        self, env: Any
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None, np.ndarray | None]:
        return (self._base_body_mass, self._base_geom_friction, self._ground_geom_id, None)

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        return env._sample_velocity_commands(num_reset)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        plan = super().build_reset_plan(env, env_ids)
        env._apply_reset_dofs(plan.qpos, env_ids)
        env._sample_motor_strength(env_ids)
        env._apply_gripper_mass_dr(plan.randomization, env_ids)
        env._cache_reset_dr(env_ids, plan.randomization)
        env.reset_ee_goals(env_ids, plan.info_updates["commands"])
        env._reset_force_schedules(env_ids)
        env._history_obs_buf[env_ids] = 0.0
        env._history_critic_buf[env_ids] = 0.0
        env._gait_indices[env_ids] = 0.0
        env._feet_air_time[env_ids] = 0.0
        env._last_contacts[env_ids] = False
        env._first_contact[env_ids] = False
        env._last_dof_vel[env_ids] = 0.0
        env._action_history[env_ids] = 0.0
        return plan

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: np.ndarray,
        info_updates: dict[str, Any],
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        sliced_info: dict[str, Any] = {}
        for k, v in info_updates.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == env._num_envs:
                sliced_info[k] = v[env_ids]
            else:
                sliced_info[k] = v
        actor_raw, critic_raw = env._compute_raw_obs(sliced_info, env_ids=env_ids)
        return cast(
            "dict[str, np.ndarray]",
            env._update_history(actor_raw, critic_raw, env_ids=env_ids),
        )


# ── Environment ──────────────────────────────────────────────────────────────


@registry.env("Go2ArmPosForce", sim_backend="mujoco")
class Go2ArmPosForceEnv(Go2ArmBaseEnv):
    _cfg: Go2ArmPosForceCfg

    def __init__(self, cfg: Go2ArmPosForceCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        if backend_type != "mujoco":
            raise ValueError(f"Go2ArmPosForce currently supports only mujoco, got {backend_type!r}")

        scene = cfg.scene if cfg.scene is not None else SceneCfg(model_file=cfg.model_file)
        backend = create_backend(
            backend_type,
            scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        if self._num_action != NUM_ACTIONS:
            raise ValueError(f"Go2ArmPosForce expects 18 actuators, got {self._num_action}")

        dtype = get_global_dtype()
        self._np_dtype = dtype
        self._gravity_sensor = "upvector"
        self._reward_cfg = cfg.reward_config
        self._enable_reward_log = True

        # PD law tables (18,).
        gains = build_go2_arm_position_gains(cfg.control_config)  # type: ignore[arg-type]
        self._kp = gains["kp"].astype(np.float64)
        self._kd = gains["kd"].astype(np.float64)
        self._torque_limits = np.concatenate(
            [
                np.full((NUM_LEG,), cfg.control_config.leg_torque_limit, dtype=np.float64),
                np.asarray(cfg.control_config.arm_torque_limit, dtype=np.float64),
            ]
        )
        self._action_scale = np.concatenate(
            [
                np.full((NUM_LEG,), cfg.control_config.action_scale, dtype=dtype),
                np.full((NUM_ARM,), cfg.control_config.arm_action_scale, dtype=dtype),
            ]
        )
        self._motor_strength = np.ones((num_envs, NUM_ACTIONS), dtype=dtype)
        # Action-delay ring buffer (UniFP action_delay control steps).
        self._action_delay_steps = max(0, int(cfg.control_config.action_delay_steps))
        self._action_history = np.zeros(
            (num_envs, self._action_delay_steps + 1, NUM_ACTIONS), dtype=dtype
        )
        self._last_torque = np.zeros((num_envs, NUM_ACTIONS), dtype=dtype)
        self._motor_targets = np.tile(self.default_angles.astype(dtype), (num_envs, 1))
        self._backend.set_pre_step_control(self._pre_step_pd_control)

        # Privileged DR readback caches.
        self._dr_friction = np.zeros((num_envs, 1), dtype=dtype)
        self._dr_base_mass = np.zeros((num_envs, 1), dtype=dtype)
        self._dr_base_com = np.zeros((num_envs, 3), dtype=dtype)
        self._dr_gripper_mass = np.zeros((num_envs, 1), dtype=dtype)
        # (DR readback caches above are populated at every reset.)
        # Gripper-mass DR: resolve the wrist/gripper body and cache the full
        # per-body mass table so we can inject a per-env mass delta at reset.
        self._gripper_body_id = int(self._backend.get_body_ids([cfg.asset.ee_body_name])[0])
        self._full_body_mass = self._backend.get_body_mass().astype(np.float64)

        # EE goal buffers.
        self._init_ee_goal_buffers(num_envs)
        self._arm_base_offset = np.asarray([0.0045, 0.0, 0.0945], dtype=dtype)
        self._ee_center_offset = np.asarray(
            [
                cfg.goal_ee.sphere_center.x_offset,
                cfg.goal_ee.sphere_center.y_offset,
                cfg.goal_ee.sphere_center.z_invariant_offset,
            ],
            dtype=dtype,
        )

        # Gait buffers.
        self._gait_indices = np.zeros((num_envs,), dtype=dtype)
        self._ref_dof_pos = np.tile(self.default_angles[:NUM_LEG].astype(dtype), (num_envs, 1))
        self._stance_mask = np.zeros((num_envs, 4), dtype=dtype)

        # Feet / dof-acceleration reward state.
        num_feet = len(cfg.sensor.feet_force)
        self._foot_contact = np.zeros((num_envs, num_feet), dtype=bool)
        self._last_contacts = np.zeros((num_envs, num_feet), dtype=bool)
        self._first_contact = np.zeros((num_envs, num_feet), dtype=bool)
        self._feet_air_time = np.zeros((num_envs, num_feet), dtype=dtype)
        self._air_time_snapshot = np.zeros((num_envs, num_feet), dtype=dtype)
        self._last_dof_vel = np.zeros((num_envs, NUM_ACTIONS), dtype=dtype)
        # Hard joint position limits (18 actuated joints), for dof_pos_limits.
        joint_range = self._backend.get_joint_range()
        if joint_range is None:
            raise ValueError("backend.get_joint_range() returned None; required for dof_pos_limits")
        self._dof_pos_limits = np.asarray(joint_range, dtype=dtype)[:NUM_ACTIONS]

        # Force application bodies (resolved once, cold path).
        self._ee_body_id = int(self._backend.get_body_ids([cfg.asset.ee_body_name])[0])
        self._base_body_id = int(self._backend.get_body_ids([cfg.asset.base_name])[0])
        self._force_body_ids = np.asarray([self._ee_body_id, self._base_body_id], dtype=np.int32)
        self._gripper_force_kp = float(cfg.commands.gripper_force_kp)
        self._base_force_kd = float(cfg.commands.base_force_kd)
        self._force_start_step = int(cfg.commands.force_start_step)

        # Realized external force buffers (world frame) and commanded forces.
        self._force_ee_world = np.zeros((num_envs, 3), dtype=dtype)
        self._force_base_world = np.zeros((num_envs, 3), dtype=dtype)
        self._force_ee_cmd = np.zeros((num_envs, 3), dtype=dtype)  # yaw-local frame
        self._force_base_cmd = np.zeros((num_envs, 3), dtype=dtype)  # yaw-local frame

        ctrl_dt = cfg.ctrl_dt
        cmds = cfg.commands

        def _steps(seconds: list[float]) -> tuple[int, int]:
            return (max(1, int(seconds[0] / ctrl_dt)), max(1, int(seconds[1] / ctrl_dt)))

        g_int = _steps(cmds.push_gripper_interval_s)
        g_dur = _steps(cmds.push_gripper_duration_s)
        b_int = _steps(cmds.push_base_interval_s)
        b_dur = _steps(cmds.push_base_duration_s)
        self._sched_gripper_cmd = _ForceSchedule(
            num_envs,
            tuple(cmds.max_push_force_xyz_gripper_cmd),
            g_int,
            g_dur,
            cmds.gripper_forced_prob_cmd,
            dtype,
        )
        self._sched_gripper_ext = _ForceSchedule(
            num_envs,
            tuple(cmds.max_push_force_xyz_gripper_ext),
            g_int,
            g_dur,
            cmds.gripper_forced_prob_ext,
            dtype,
        )
        self._sched_base_cmd = _ForceSchedule(
            num_envs,
            tuple(cmds.max_push_force_xyz_base_cmd),
            b_int,
            b_dur,
            cmds.base_forced_prob_cmd,
            dtype,
            z_scale=0.0,  # UniFP zeroes the commanded base force z-component.
        )
        self._sched_base_ext = _ForceSchedule(
            num_envs,
            tuple(cmds.max_push_force_xyz_base_ext),
            b_int,
            b_dur,
            cmds.base_forced_prob_ext,
            dtype,
            z_scale=float(cmds.force_z_base_ext_scale),
        )

        # Command resample timer.
        self._cmd_resample_steps = max(1, int(cfg.commands.resample_time_s / cfg.ctrl_dt))
        self._cmd_timer = np.random.randint(
            0, self._cmd_resample_steps, size=(num_envs,), dtype=np.int32
        )

        self._commands_scale = self._build_commands_scale()
        self._init_reward_functions()

        # Single-step obs dims.
        self._actor_one_step_dim = self._actor_single_obs_dim()
        self._critic_one_step_dim = self._critic_single_obs_dim()
        H_a = cfg.history.num_actor_history
        H_c = cfg.history.num_critic_history
        self._history_obs_buf = np.zeros((num_envs, H_a * self._actor_one_step_dim), dtype=dtype)
        self._history_critic_buf = np.zeros(
            (num_envs, H_c * self._critic_one_step_dim), dtype=dtype
        )

        base_body_mass = backend.get_body_mass() if cfg.domain_rand.randomize_base_mass else None
        base_geom_friction = None
        ground_geom_id = None
        if cfg.domain_rand.randomize_ground_friction:
            base_geom_friction = backend.get_geom_friction()
            ground_geom_id = backend.get_geom_id(cfg.asset.ground)
        # Remember the ground geom so the friction privileged readback reads the
        # geom that is actually randomized (not geom 0, a robot geom).
        self._ground_geom_id = ground_geom_id
        self._init_domain_randomization(
            Go2ArmPosForceDRProvider(
                base_body_mass=base_body_mass,
                base_geom_friction=base_geom_friction,
                ground_geom_id=ground_geom_id,
            )
        )

    # ── obs spec ──────────────────────────────────────────────────────────

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        H_a = self._cfg.history.num_actor_history
        H_c = self._cfg.history.num_critic_history
        return {
            "obs": H_a * self._actor_one_step_dim,
            "critic": H_c * self._critic_one_step_dim,
        }

    def _actor_single_obs_dim(self) -> int:
        # body_orient(2) + ang_vel(3) + dof_diff(18) + dof_vel(18)
        # + last_actions(18) + sin/cos(2) + commands(15)
        return 2 + 3 + NUM_ACTIONS + NUM_ACTIONS + NUM_ACTIONS + 2 + NUM_COMMANDS

    def _critic_single_obs_dim(self) -> int:
        # CSE targets front block (12): base_lin_vel(3) + ee_pos_sphere(3)
        #   + force_ee(3) + force_base(3)
        # then: leg_ref_diff(12) + dr_block(6) + motor_strength(18) + stance(4)
        #   + contact_mask(4) + proj_gravity(3) + ee_goal_offset_sphere(3)
        #   + actor_core(...)
        # The original UniFP block also carried 17 legacy leg/base mass slots
        # (inert zeros); they are intentionally omitted here.
        cse = 12
        privileged = NUM_LEG + 6 + NUM_ACTIONS + 4 + 4 + 3 + 3
        return cse + privileged + self._actor_single_obs_dim()

    # ── EE goal sampling (UniFP yaw + gravity frame) ──────────────────────

    def _init_ee_goal_buffers(self, num_envs: int) -> None:
        dtype = get_global_dtype()
        self._curr_ee_goal_sphere = np.zeros((num_envs, 3), dtype=dtype)
        self._curr_ee_goal_cart = np.zeros((num_envs, 3), dtype=dtype)
        self._curr_ee_goal_world = np.zeros((num_envs, 3), dtype=dtype)
        self._ee_start_sphere = np.zeros((num_envs, 3), dtype=dtype)
        self._ee_goal_sphere = np.zeros((num_envs, 3), dtype=dtype)
        self._ee_goal_orn_euler = np.zeros((num_envs, 3), dtype=dtype)
        self._arm_goal_timer = np.zeros((num_envs,), dtype=np.int32)
        self._traj_steps = np.ones((num_envs,), dtype=np.int32)
        self._traj_total_steps = np.ones((num_envs,), dtype=np.int32)

    def get_ee_goal_spherical_center(self) -> np.ndarray:
        """Sphere center: base XY (yaw frame) + offset, terrain-relative z."""
        base_pos = self._backend.get_base_pos()
        base_yaw_quat = np_yaw_quat(self._backend.get_base_quat())
        center = np.zeros((self._num_envs, 3), dtype=self._np_dtype)
        center[:, :2] = base_pos[:, :2]
        center = center + np_quat_apply(
            base_yaw_quat, np.tile(self._ee_center_offset, (self._num_envs, 1))
        )
        return center

    def _collision_check(self, starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
        cfg = self._cfg.goal_ee
        n = max(2, cfg.num_collision_check_samples)
        t = np.linspace(0.0, 1.0, n, dtype=self._np_dtype)
        path_sphere = starts[:, None, :] + (goals - starts)[:, None, :] * t[None, :, None]
        path_cart = sphere2cart(path_sphere.reshape(-1, 3)).reshape(len(starts), n, 3)
        upper = np.asarray(cfg.collision_upper_limits, dtype=self._np_dtype)
        lower = np.asarray(cfg.collision_lower_limits, dtype=self._np_dtype)
        inside = np.all(path_cart < upper, axis=2) & np.all(path_cart > lower, axis=2)
        underground = np.any(path_cart[..., 2] < float(cfg.underground_limit), axis=1)
        return np.any(inside, axis=1) | underground

    def _resample_ee_goal_sphere(self, env_ids: np.ndarray) -> None:
        cfg = self._cfg.goal_ee
        n = len(env_ids)
        start = self._ee_start_sphere[env_ids]
        candidates = np.tile(np.asarray(cfg.init_pos_end, dtype=self._np_dtype), (n, 1))
        remaining = np.arange(n, dtype=np.int32)
        for _ in range(max(1, cfg.num_resample_attempts)):
            m = len(remaining)
            l = np.random.uniform(*cfg.pos_l, size=m).astype(self._np_dtype)
            p = np.random.uniform(*cfg.pos_p, size=m).astype(self._np_dtype)
            y = np.random.uniform(*cfg.pos_y, size=m).astype(self._np_dtype)
            new_goals = np.stack([l, p, y], axis=1)
            candidates[remaining] = new_goals
            unsafe = self._collision_check(start[remaining], new_goals)
            remaining = remaining[unsafe]
            if len(remaining) == 0:
                break
        self._ee_goal_sphere[env_ids] = candidates

    def reset_ee_goals(self, env_ids: np.ndarray, commands: np.ndarray) -> None:
        """Reset EE goal trajectory: start at init_pos_start, sample an end goal."""
        env_ids = np.asarray(env_ids, dtype=np.int32).reshape(-1)
        if len(env_ids) == 0:
            return
        cfg = self._cfg.goal_ee
        self._ee_start_sphere[env_ids] = np.asarray(cfg.init_pos_start, dtype=self._np_dtype)
        self._curr_ee_goal_sphere[env_ids] = self._ee_start_sphere[env_ids]
        self._resample_ee_goal_sphere(env_ids)
        dt = self._cfg.ctrl_dt
        traj_t = np.random.uniform(*cfg.traj_time, size=len(env_ids))
        hold_t = np.random.uniform(*cfg.hold_time, size=len(env_ids))
        traj_s = np.maximum(1, np.round(traj_t / dt).astype(np.int32))
        hold_s = np.maximum(0, np.round(hold_t / dt).astype(np.int32))
        self._traj_steps[env_ids] = traj_s
        self._traj_total_steps[env_ids] = traj_s + hold_s
        self._arm_goal_timer[env_ids] = 0
        # Write the starting sphere command into the (sliced) command vector.
        commands[:, CMD_EE_POS] = self._curr_ee_goal_sphere[env_ids]

    def _update_ee_goal_trajectory(self, commands: np.ndarray) -> None:
        """Advance the EE goal along start->goal, then resample after the hold."""
        timer = self._arm_goal_timer
        traj = self._traj_steps.astype(self._np_dtype)
        t = np.clip(timer.astype(self._np_dtype) / np.maximum(traj, 1.0), 0.0, 1.0)
        self._curr_ee_goal_sphere = (
            self._ee_start_sphere + (self._ee_goal_sphere - self._ee_start_sphere) * t[:, None]
        )
        done = timer >= self._traj_total_steps
        if np.any(done):
            done_ids = np.flatnonzero(done).astype(np.int32)
            self._ee_start_sphere[done_ids] = self._ee_goal_sphere[done_ids]
            self._resample_ee_goal_sphere(done_ids)
            self._arm_goal_timer[done_ids] = 0
            dt = self._cfg.ctrl_dt
            cfg = self._cfg.goal_ee
            traj_t = np.random.uniform(*cfg.traj_time, size=len(done_ids))
            hold_t = np.random.uniform(*cfg.hold_time, size=len(done_ids))
            traj_s = np.maximum(1, np.round(traj_t / dt).astype(np.int32))
            hold_s = np.maximum(0, np.round(hold_t / dt).astype(np.int32))
            self._traj_steps[done_ids] = traj_s
            self._traj_total_steps[done_ids] = traj_s + hold_s
        self._arm_goal_timer += 1
        self._curr_ee_goal_cart = sphere2cart(self._curr_ee_goal_sphere)
        base_yaw_quat = np_yaw_quat(self._backend.get_base_quat())
        self._curr_ee_goal_world = self.get_ee_goal_spherical_center() + np_quat_apply(
            base_yaw_quat, self._curr_ee_goal_cart
        )
        commands[:, CMD_EE_POS] = self._curr_ee_goal_sphere

    def _ee_pos_sphere_measured(self) -> np.ndarray:
        """Measured EE position in the yaw+gravity sphere frame (CSE target)."""
        ee_world = self._ee_world_pos()
        base_yaw_quat = np_yaw_quat(self._backend.get_base_quat())
        ee_local = np_quat_apply_inverse(
            base_yaw_quat, ee_world - self.get_ee_goal_spherical_center()
        )
        return cart2sphere(ee_local)

    def _ee_world_pos(self) -> np.ndarray:
        """End-effector world position from the armbasepoint-relative sensors."""
        base = self._backend.get_sensor_data("armbasepoint_world_pos")
        quat = self._backend.get_sensor_data("armbasepoint_world_quat")
        rel = self._backend.get_sensor_data(self._cfg.sensor.ee_local_pos)
        return base + np_quat_apply(quat, rel)

    # ── commands ──────────────────────────────────────────────────────────

    def _build_commands_scale(self) -> np.ndarray:
        s = self._cfg.obs_scales
        return np.asarray(
            [
                s.lin_vel,
                s.lin_vel,
                s.ang_vel,
                s.ee_sphe_radius,
                s.ee_sphe_pitch,
                s.ee_sphe_yaw,
                s.ee_orn,
                s.ee_orn,
                s.ee_orn,
                s.ee_force,
                s.ee_force,
                s.ee_force,
                s.base_force,
                s.base_force,
                s.base_force,
            ],
            dtype=self._np_dtype,
        )

    def _sample_velocity_commands(self, num_reset: int) -> np.ndarray:
        cfg = self._cfg.commands
        commands = np.zeros((num_reset, NUM_COMMANDS), dtype=self._np_dtype)
        commands[:, 0] = np.random.uniform(*cfg.lin_vel_x, size=num_reset)
        commands[:, 1] = np.random.uniform(*cfg.lin_vel_y, size=num_reset)
        commands[:, 2] = np.random.uniform(*cfg.ang_vel_yaw, size=num_reset)
        # Raise the zero-velocity probability once forces are active (UniFP).
        zero_prob = (
            cfg.zero_vel_cmd_prob_after_force
            if self.step_counter >= self._force_start_step
            else cfg.zero_vel_cmd_prob
        )
        zero_mask = np.random.uniform(size=num_reset) < zero_prob
        commands[zero_mask, 0:3] = 0.0
        # Deadband small commands to zero for stable standing.
        small = (
            (np.abs(commands[:, 0]) < cfg.lin_vel_x_clip)
            & (np.abs(commands[:, 1]) < cfg.lin_vel_y_clip)
            & (np.abs(commands[:, 2]) < cfg.ang_vel_yaw_clip)
        )
        commands[small, 0:3] = 0.0
        return commands

    def _command_is_moving(self, commands: np.ndarray) -> np.ndarray:
        cfg = self._cfg.commands
        return (
            (np.abs(commands[:, 0]) > cfg.lin_vel_x_clip)
            | (np.abs(commands[:, 1]) > cfg.lin_vel_y_clip)
            | (np.abs(commands[:, 2]) > cfg.ang_vel_yaw_clip)
        )

    def _resample_commands_if_due(self, info: dict) -> None:
        self._cmd_timer += 1
        due = self._cmd_timer >= self._cmd_resample_steps
        if np.any(due):
            ids = np.flatnonzero(due).astype(np.int32)
            new_cmds = self._sample_velocity_commands(len(ids))
            info["commands"][ids, 0:3] = new_cmds[:, 0:3]
            self._cmd_timer[ids] = 0

    # ── reset DOFs / motor strength ───────────────────────────────────────

    def _apply_reset_dofs(self, qpos: np.ndarray, env_ids: np.ndarray) -> None:
        """UniFP-style reset: scale leg dofs, jitter arm dofs around default."""
        cfg = self._cfg.reset
        n = len(env_ids)
        base = 7  # floating base (pos3 + quat4) precedes the joint dofs
        leg_scale = np.random.uniform(*cfg.leg_pos_scale_range, size=(n, NUM_LEG)).astype(
            qpos.dtype
        )
        qpos[:, base : base + NUM_LEG] = self.default_angles[:NUM_LEG] * leg_scale
        arm_jitter = np.random.uniform(
            -cfg.arm_init_dof_pos_range, cfg.arm_init_dof_pos_range, size=(n, NUM_ARM)
        ).astype(qpos.dtype)
        qpos[:, base + NUM_LEG : base + NUM_ACTIONS] = self.default_angles[NUM_LEG:] + arm_jitter

    def _sample_motor_strength(self, env_ids: np.ndarray) -> None:
        cfg = self._cfg.domain_rand
        if not cfg.randomize_motor_strength:
            return
        n = len(env_ids)
        leg = np.random.uniform(*cfg.leg_motor_strength_range, size=(n, NUM_LEG))
        arm = np.random.uniform(*cfg.arm_motor_strength_range, size=(n, NUM_ARM))
        self._motor_strength[env_ids] = np.concatenate([leg, arm], axis=1).astype(self._np_dtype)

    def _apply_gripper_mass_dr(self, randomization: Any, env_ids: np.ndarray) -> None:
        """Inject a per-env gripper-mass delta into the reset body-mass table.

        The backend composes ``body_mass`` (a full per-body table) with
        ``base_mass_delta`` (added at the base body), so setting a gripper-
        augmented table here coexists with the base-mass randomization.
        """
        cfg = self._cfg.domain_rand
        if randomization is None or not cfg.randomize_gripper_mass:
            return
        n = len(env_ids)
        delta = np.random.uniform(*cfg.gripper_added_mass_range, size=n)
        self._dr_gripper_mass[env_ids] = delta.astype(self._np_dtype).reshape(-1, 1)
        nbody = self._full_body_mass.shape[0]
        table = getattr(randomization, "body_mass", None)
        if table is None:
            table = np.broadcast_to(self._full_body_mass, (n, nbody)).copy()
        else:
            table = np.asarray(table, dtype=np.float64).copy()
        table[:, self._gripper_body_id] += delta
        randomization.body_mass = table

    def _cache_reset_dr(self, env_ids: np.ndarray, randomization: Any) -> None:
        """Cache realized reset DR values for the privileged observation."""
        if randomization is None:
            return
        mass = getattr(randomization, "base_mass_delta", None)
        if mass is not None:
            self._dr_base_mass[env_ids] = np.asarray(mass, dtype=self._np_dtype).reshape(-1, 1)
        com = getattr(randomization, "base_com_offset", None)
        if com is not None:
            self._dr_base_com[env_ids] = np.asarray(com, dtype=self._np_dtype).reshape(-1, 3)
        friction = getattr(randomization, "geom_friction", None)
        if friction is not None:
            fr = np.asarray(friction, dtype=self._np_dtype)
            # Tangential friction of the ground geom (the one we randomize).
            gid = self._ground_geom_id if self._ground_geom_id is not None else 0
            self._dr_friction[env_ids] = fr[:, gid, 0:1]

    # ── control: Python PD via per-substep pre-step hook ──────────────────

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        actions = np.asarray(actions, dtype=self._np_dtype)
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions
        # The applied (executed) action is delayed; the policy's raw current/last
        # actions above still drive observations and the action_rate rewards.
        if self._action_delay_steps > 0:
            self._action_history[:, :-1] = self._action_history[:, 1:]
            self._action_history[:, -1] = actions
            exec_actions = self._action_history[:, 0]
        elif self._cfg.control_config.simulate_action_latency:
            exec_actions = state.info["last_actions"]
        else:
            exec_actions = actions
        self._update_forces(state)
        # Position targets = default + action * motor_strength * action_scale.
        self._motor_targets = (
            self.default_angles + exec_actions * self._motor_strength * self._action_scale
        ).astype(self._np_dtype)
        return self._motor_targets

    def _update_forces(self, state: NpEnvState) -> None:
        """Advance the force episodes and stage external forces for this step.

        cmd forces are commanded (written into the command vector, yaw-local
        frame); ext forces are external disturbances applied to the EE and base
        bodies via the backend (world frame) and become the realized force in
        the observation.
        """
        enabled = self.step_counter >= self._force_start_step
        self._force_ee_cmd = self._sched_gripper_cmd.step(enabled)
        self._force_ee_world = self._sched_gripper_ext.step(enabled)
        self._force_base_cmd = self._sched_base_cmd.step(enabled)
        self._force_base_world = self._sched_base_ext.step(enabled)

        commands = state.info.get("commands")
        if commands is not None and commands.shape[0] == self._num_envs:
            commands[:, CMD_EE_FORCE] = self._force_ee_cmd
            commands[:, CMD_BASE_FORCE] = self._force_base_cmd

        applied = np.stack([self._force_ee_world, self._force_base_world], axis=1)
        if enabled and np.any(applied):
            self._backend.apply_body_force(self._force_body_ids, applied.astype(np.float64))

    def _reset_force_schedules(self, env_ids: np.ndarray) -> None:
        for sched in (
            self._sched_gripper_cmd,
            self._sched_gripper_ext,
            self._sched_base_cmd,
            self._sched_base_ext,
        ):
            sched.reset(env_ids)
        self._force_ee_world[env_ids] = 0.0
        self._force_base_world[env_ids] = 0.0
        self._force_ee_cmd[env_ids] = 0.0
        self._force_base_cmd[env_ids] = 0.0

    def _pre_step_pd_control(self, backend: Any, targets: np.ndarray) -> np.ndarray:
        q = backend.get_dof_pos()
        qd = backend.get_dof_vel()
        torque = self._kp * (targets - q) - self._kd * qd
        torque = np.clip(torque, -self._torque_limits, self._torque_limits)
        self._last_torque = torque.astype(self._np_dtype)
        return torque

    # ── gait (UniFP humanoid-gym trot reference) ──────────────────────────

    def _update_gait(self, info: dict) -> None:
        cfg = self._cfg.gait
        moving = self._command_is_moving(info["commands"])
        self._gait_indices = np.remainder(
            self._gait_indices + self._cfg.ctrl_dt / cfg.cycle_time, 1.0
        )
        self._gait_indices[~moving] = 0.0
        sin_pos = np.sin(2.0 * np.pi * self._gait_indices)
        thd = cfg.target_joint_pos_thd
        sin_l = sin_pos + thd
        sin_r = sin_pos - thd
        stance = np.zeros((self._num_envs, 4), dtype=self._np_dtype)
        stance[:, 0] = sin_l >= 0
        stance[:, 3] = sin_l >= 0
        stance[:, 1] = sin_r < 0
        stance[:, 2] = sin_r < 0
        self._stance_mask = stance

        ref = np.tile(self.default_angles[:NUM_LEG], (self._num_envs, 1)).astype(self._np_dtype)
        scale_1 = cfg.target_joint_pos_scale / (1.0 - thd)
        scale_2 = scale_1 * 2.0
        sl = np.where(sin_l > 0, 0.0, sin_l)
        sr = np.where(sin_r < 0, 0.0, sin_r)
        ref[:, 1] -= sl * scale_1
        ref[:, 2] += sl * scale_2
        ref[:, 10] -= sl * scale_1
        ref[:, 11] += sl * scale_2
        ref[:, 4] += sr * scale_1
        ref[:, 5] -= sr * scale_2
        ref[:, 7] += sr * scale_1
        ref[:, 8] -= sr * scale_2
        self._ref_dof_pos = ref

    # ── feet / contact sensor readers ─────────────────────────────────────

    def _get_foot_force_vec(self) -> np.ndarray:
        """Per-foot contact force vector (num_envs, 4, 3)."""
        return np.stack(
            [self._backend.get_sensor_data(n) for n in self._cfg.sensor.feet_force_vec], axis=1
        )

    def _get_foot_vel(self) -> np.ndarray:
        """Per-foot world linear velocity (num_envs, 4, 3)."""
        return np.stack(
            [self._backend.get_sensor_data(n) for n in self._cfg.sensor.feet_vel], axis=1
        )

    def _get_thigh_pos(self) -> np.ndarray:
        """Per-thigh world position (num_envs, 4, 3)."""
        return np.stack(
            [self._backend.get_sensor_data(n) for n in self._cfg.sensor.thigh_pos], axis=1
        )

    def _get_undesired_contacts(self) -> np.ndarray:
        """Binary contact flags on penalised bodies (num_envs, K)."""
        return np.stack(
            [self._backend.get_sensor_data(n)[:, 0] for n in self._cfg.sensor.undesired_contact],
            axis=1,
        )

    def _update_contact_timers(self, contact: np.ndarray) -> None:
        """UniFP feet_air_time bookkeeping (PhysX-style contact filtering)."""
        contact_filt = contact | self._last_contacts
        self._last_contacts = contact
        self._first_contact = (self._feet_air_time > 0.0) & contact_filt
        self._feet_air_time = self._feet_air_time + self._cfg.ctrl_dt
        # Snapshot the air time the reward consumes, then reset feet in contact.
        self._air_time_snapshot = self._feet_air_time.copy()
        self._feet_air_time = self._feet_air_time * (~contact_filt)

    # ── step ──────────────────────────────────────────────────────────────

    def update_state(self, state: NpEnvState) -> NpEnvState:
        info = state.info
        self._resample_commands_if_due(info)
        self._update_gait(info)
        self._update_ee_goal_trajectory(info["commands"])
        info["torques"] = self._last_torque

        self._foot_contact = self.get_foot_contact() > 0.5
        self._update_contact_timers(self._foot_contact)

        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._gravity_sensor)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        env_ids = np.arange(self._num_envs, dtype=np.int32)
        actor_raw, critic_raw = self._compute_raw_obs(info, env_ids=env_ids)
        obs = self._update_history(actor_raw, critic_raw, env_ids=env_ids)
        reward = self._compute_reward(info, linvel, gyro, gravity, dof_pos, dof_vel)
        terminated = self._compute_terminated(gravity)
        # dof acceleration uses the previous step's velocity; update after rewards.
        self._last_dof_vel = dof_vel.copy()
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        # Fall when the projected up-axis tips past horizontal.
        return gravity[:, 2] <= 0.0

    # ── observations ──────────────────────────────────────────────────────

    def _compute_raw_obs(self, info: dict, env_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (actor_raw, critic_raw) single-step observations for env_ids."""
        dtype = self._np_dtype
        s = self._cfg.obs_scales

        base_quat = self._backend.get_base_quat()[env_ids]
        base_ang_vel = self._backend.get_base_ang_vel()[env_ids]
        gravity = self._backend.get_sensor_data(self._gravity_sensor)[env_ids]
        dof_pos = self.get_dof_pos()[env_ids]
        dof_vel = self.get_dof_vel()[env_ids]
        base_lin_vel = self.get_local_linvel()[env_ids]
        commands = info["commands"]
        commands = commands if commands.shape[0] == len(env_ids) else commands[env_ids]
        last_actions = info.get(
            "current_actions", np.zeros((len(env_ids), NUM_ACTIONS), dtype=dtype)
        )
        last_actions = (
            last_actions if last_actions.shape[0] == len(env_ids) else last_actions[env_ids]
        )

        body_rp = _roll_pitch_from_quat(base_quat)
        diff = (dof_pos - self.default_angles) * s.dof_pos
        dvel = dof_vel * s.dof_vel
        sin_phase = np.sin(2.0 * np.pi * self._gait_indices[env_ids])[:, None]
        cos_phase = np.cos(2.0 * np.pi * self._gait_indices[env_ids])[:, None]
        cmd_scaled = commands * self._commands_scale

        # Noise (actor only).
        nz = self._cfg.noise_config
        body_rp_n = self._obs_noise(body_rp, nz.scale_orn)
        ang_vel_n = self._obs_noise(base_ang_vel, nz.scale_ang_vel)
        diff_n = self._obs_noise(diff, nz.scale_joint_angle)
        dvel_n = self._obs_noise(dvel, nz.scale_joint_vel)

        actor_raw = np.concatenate(
            [
                body_rp_n,
                ang_vel_n * s.ang_vel,
                diff_n,
                dvel_n,
                last_actions,
                sin_phase,
                cos_phase,
                cmd_scaled,
            ],
            axis=1,
            dtype=dtype,
        )

        # CSE estimator targets (front 12 of critic obs).
        ee_pos_sphere = self._ee_pos_sphere_measured()[env_ids]
        force_ee_local = np_quat_apply_inverse(
            np_yaw_quat(base_quat), self._force_ee_world[env_ids]
        )
        force_base_local = np_quat_apply_inverse(
            np_yaw_quat(base_quat), self._force_base_world[env_ids]
        )
        ee_sphere_scaled = ee_pos_sphere * np.asarray(
            [s.ee_sphe_radius, s.ee_sphe_pitch, s.ee_sphe_yaw], dtype=dtype
        )
        cse_targets = np.concatenate(
            [
                base_lin_vel * s.lin_vel,
                ee_sphere_scaled,
                force_ee_local * s.ee_force,
                force_base_local * s.base_force,
            ],
            axis=1,
            dtype=dtype,
        )

        leg_ref_diff = dof_pos[:, :NUM_LEG] - self._ref_dof_pos[env_ids]
        dr_block = np.concatenate(
            [
                self._dr_friction[env_ids],
                self._dr_base_mass[env_ids],
                self._dr_base_com[env_ids],
                self._dr_gripper_mass[env_ids],
            ],
            axis=1,
        )
        motor_strength = self._motor_strength[env_ids] - 1.0
        contact_mask = self.get_foot_contact()[env_ids].astype(dtype)

        # Force-offset EE goal expressed in the sphere frame (UniFP privileged).
        base_yaw_quat = np_yaw_quat(base_quat)
        force_cmd_world = np_quat_apply(base_yaw_quat, self._force_ee_cmd[env_ids])
        forces_offset = self._force_ee_world[env_ids] + force_cmd_world
        goal_offset_world = forces_offset / self._gripper_force_kp + self._curr_ee_goal_world[env_ids]
        center = self.get_ee_goal_spherical_center()[env_ids]
        goal_offset_local = np_quat_apply_inverse(base_yaw_quat, goal_offset_world - center)
        ee_goal_offset_sphere = cart2sphere(goal_offset_local) * np.asarray(
            [s.ee_sphe_radius, s.ee_sphe_pitch, s.ee_sphe_yaw], dtype=dtype
        )

        critic_core = np.concatenate(
            [
                body_rp,
                base_ang_vel * s.ang_vel,
                diff,
                dvel,
                last_actions,
                sin_phase,
                cos_phase,
                cmd_scaled,
            ],
            axis=1,
            dtype=dtype,
        )
        critic_raw = np.concatenate(
            [
                cse_targets,
                leg_ref_diff,
                dr_block,
                motor_strength,
                self._stance_mask[env_ids],
                contact_mask,
                gravity,
                ee_goal_offset_sphere,
                critic_core,
            ],
            axis=1,
            dtype=dtype,
        )
        return actor_raw, critic_raw

    def _update_history(
        self, actor_raw: np.ndarray, critic_raw: np.ndarray, env_ids: np.ndarray
    ) -> dict[str, np.ndarray]:
        H_a = self._cfg.history.num_actor_history
        H_c = self._cfg.history.num_critic_history
        a_dim = self._actor_one_step_dim
        c_dim = self._critic_one_step_dim
        buf_a = self._history_obs_buf[env_ids]
        buf_c = self._history_critic_buf[env_ids]
        if H_a > 1:
            buf_a = np.concatenate([buf_a[:, a_dim:], actor_raw], axis=1)
        else:
            buf_a = actor_raw
        if H_c > 1:
            # Critic history is newest-frame-FIRST so the current-frame CSE target
            # block stays at offset 0 (estimator.target_start=0) regardless of H_c.
            buf_c = np.concatenate([critic_raw, buf_c[:, :-c_dim]], axis=1)
        else:
            buf_c = critic_raw
        self._history_obs_buf[env_ids] = buf_a
        self._history_critic_buf[env_ids] = buf_c
        return {"obs": buf_a.copy(), "critic": buf_c.copy()}

    # ── rewards ───────────────────────────────────────────────────────────

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_lin_vel_force_world": self._reward_tracking_lin_vel_force_world,
            "tracking_ee_force_world": self._reward_tracking_ee_force_world,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "alive": self._reward_alive,
            "ref_dof_leg": self._reward_ref_dof_leg,
            "action_rate": self._reward_action_rate,
            "action_rate_arm": self._reward_action_rate_arm,
            "torques": self._reward_torques,
            "dof_vel": self._reward_dof_vel,
            "dof_vel_arm": self._reward_dof_vel_arm,
            "dof_acc": self._reward_dof_acc,
            "dof_acc_arm": self._reward_dof_acc_arm,
            "base_height": self._reward_base_height,
            "hip_pos": self._reward_hip_pos,
            "torque_limits": self._reward_torque_limits,
            "dof_pos_limits": self._reward_dof_pos_limits,
            "stand_still": self._reward_stand_still,
            "collision": self._reward_collision,
            "feet_contact_number": self._reward_feet_contact_number,
            "feet_air_time": self._reward_feet_air_time,
            "feet_height": self._reward_feet_height,
            "feet_height_high": self._reward_feet_height_high,
            "feet_pos_xy": self._reward_feet_pos_xy,
            "feet_drag": self._reward_feet_drag,
            "feet_contact_forces": self._reward_feet_contact_forces,
        }

    def _compute_reward(
        self,
        info: dict,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> np.ndarray:
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            gravity=gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._backend.get_base_pos()[:, 2],
        )
        self._last_reward_ctx = ctx
        # Shared reduction: also emits per-term means into info["log"]["reward/*"]
        # so TensorBoard shows each reward component's contribution.
        return run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
        )

    def _reward_tracking_lin_vel(self, ctx: RewardContext) -> np.ndarray:
        err = np.sum(np.square(ctx.info["commands"][:, 0:2] - ctx.linvel[:, 0:2]), axis=1)
        return np.exp(-err / ctx.tracking_sigma)

    def _reward_tracking_ang_vel(self, ctx: RewardContext) -> np.ndarray:
        err = np.square(ctx.info["commands"][:, 2] - ctx.gyro[:, 2])
        return np.exp(-err / ctx.tracking_sigma)

    def _reward_tracking_ee_force_world(self, ctx: RewardContext) -> np.ndarray:
        """EE force tracking: reward the EE at the force-offset goal.

        Under EE stiffness ``gripper_force_kp`` the end-effector settles at
        ``goal + (ext + cmd) / kp``; rewarding that offset position teaches the
        policy to comply with the external force and exert the commanded force.
        """
        base_yaw_quat = np_yaw_quat(self._backend.get_base_quat())
        force_cmd_world = np_quat_apply(base_yaw_quat, self._force_ee_cmd)
        forces_offset = self._force_ee_world + force_cmd_world
        goal_offset = forces_offset / self._gripper_force_kp + self._curr_ee_goal_world
        ee_world = self._ee_world_pos()
        ee_pos_error = np.sum(np.abs(ee_world - goal_offset), axis=1)
        return np.exp(-ee_pos_error / self._reward_cfg.tracking_ee_sigma * 2.0)

    def _reward_tracking_lin_vel_force_world(self, ctx: RewardContext) -> np.ndarray:
        """Base velocity tracking with a force-induced setpoint offset.

        Under base damping ``base_force_kd`` the base settles at
        ``cmd_vel + (ext + cmd) / kd``; tracking that offset velocity teaches
        compliance with / production of base force.
        """
        base_yaw_quat = np_yaw_quat(self._backend.get_base_quat())
        force_base_local = np_quat_apply_inverse(base_yaw_quat, self._force_base_world)
        forces_offset = force_base_local + self._force_base_cmd
        vel_cmd = ctx.info["commands"][:, 0:2]
        base_vel_offset = forces_offset[:, 0:2] / self._base_force_kd + vel_cmd
        cfg = self._cfg.commands
        moving = (
            (np.abs(base_vel_offset[:, 0]) > cfg.lin_vel_x_clip)
            | (np.abs(base_vel_offset[:, 1]) > cfg.lin_vel_y_clip)
            | (np.abs(ctx.info["commands"][:, 2]) > cfg.ang_vel_yaw_clip)
        )
        base_vel_offset = base_vel_offset * moving[:, None]
        lin_err = np.sum(np.square(base_vel_offset - ctx.linvel[:, 0:2]), axis=1)
        return np.exp(-lin_err / ctx.tracking_sigma)

    def _reward_lin_vel_z(self, ctx: RewardContext) -> np.ndarray:
        return np.square(ctx.linvel[:, 2])

    def _reward_ang_vel_xy(self, ctx: RewardContext) -> np.ndarray:
        return np.sum(np.square(ctx.gyro[:, 0:2]), axis=1)

    def _reward_alive(self, ctx: RewardContext) -> np.ndarray:
        return np.ones((ctx.num_envs,), dtype=self._np_dtype)

    def _reward_ref_dof_leg(self, ctx: RewardContext) -> np.ndarray:
        # UniFP _reward_ref_dof_leg: L1 joint error, exp(-err * 0.1) (not L2-squared).
        err = np.sum(np.abs(ctx.dof_pos[:, :NUM_LEG] - self._ref_dof_pos), axis=1)
        return np.exp(-err * self._reward_cfg.ref_dof_scale)

    def _reward_action_rate(self, ctx: RewardContext) -> np.ndarray:
        # Legs only (UniFP); arm has its own action_rate_arm term.
        cur = ctx.info.get("current_actions")
        last = ctx.info.get("last_actions")
        if cur is None or last is None:
            return np.zeros((ctx.num_envs,), dtype=self._np_dtype)
        return np.sum(np.square(cur[:, :NUM_LEG] - last[:, :NUM_LEG]), axis=1)

    def _reward_action_rate_arm(self, ctx: RewardContext) -> np.ndarray:
        cur = ctx.info.get("current_actions")
        last = ctx.info.get("last_actions")
        if cur is None or last is None:
            return np.zeros((ctx.num_envs,), dtype=self._np_dtype)
        return np.sum(np.square(cur[:, NUM_LEG:] - last[:, NUM_LEG:]), axis=1)

    def _reward_torques(self, ctx: RewardContext) -> np.ndarray:
        # Legs only (UniFP _reward_torques slices [:, :12]).
        return np.sum(np.square(self._last_torque[:, :NUM_LEG]), axis=1)

    def _reward_dof_vel(self, ctx: RewardContext) -> np.ndarray:
        # Legs only (UniFP _reward_dof_vel slices [:, :12]).
        dof_vel = np.asarray(ctx.dof_vel)
        return np.sum(np.square(dof_vel[:, :NUM_LEG]), axis=1)

    def _reward_dof_vel_arm(self, ctx: RewardContext) -> np.ndarray:
        dof_vel = np.asarray(ctx.dof_vel)
        return np.sum(np.square(dof_vel[:, NUM_LEG:NUM_ACTIONS]), axis=1)

    def _reward_dof_acc(self, ctx: RewardContext) -> np.ndarray:
        dof_vel = np.asarray(ctx.dof_vel)
        acc = (self._last_dof_vel[:, :NUM_LEG] - dof_vel[:, :NUM_LEG]) / self._cfg.ctrl_dt
        return np.sum(np.square(acc), axis=1)

    def _reward_dof_acc_arm(self, ctx: RewardContext) -> np.ndarray:
        dof_vel = np.asarray(ctx.dof_vel)
        acc = (
            self._last_dof_vel[:, NUM_LEG:NUM_ACTIONS] - dof_vel[:, NUM_LEG:NUM_ACTIONS]
        ) / self._cfg.ctrl_dt
        return np.sum(np.square(acc), axis=1)

    def _reward_base_height(self, ctx: RewardContext) -> np.ndarray:
        return np.square(ctx.base_height - ctx.base_height_target)

    def _reward_hip_pos(self, ctx: RewardContext) -> np.ndarray:
        hip_idx = np.asarray([0, 3, 6, 9])
        return np.sum(np.square(ctx.dof_pos[:, hip_idx] - self.default_angles[hip_idx]), axis=1)

    def _reward_torque_limits(self, ctx: RewardContext) -> np.ndarray:
        margin = self._reward_cfg.soft_torque_limit * self._torque_limits
        over = np.abs(self._last_torque) - margin
        return np.sum(np.clip(over, 0.0, None), axis=1)

    def _reward_dof_pos_limits(self, ctx: RewardContext) -> np.ndarray:
        # Penalize positions outside the hard joint range (18 actuated joints).
        lower = self._dof_pos_limits[:, 0]
        upper = self._dof_pos_limits[:, 1]
        out = np.clip(lower - ctx.dof_pos[:, :NUM_ACTIONS], 0.0, None)
        out += np.clip(ctx.dof_pos[:, :NUM_ACTIONS] - upper, 0.0, None)
        return np.sum(out, axis=1)

    def _reward_stand_still(self, ctx: RewardContext) -> np.ndarray:
        # Reward holding the default leg pose at zero command; off when walking.
        err = np.sum(np.abs(ctx.dof_pos[:, :NUM_LEG] - self.default_angles[:NUM_LEG]), axis=1)
        rew = np.exp(-err * self._reward_cfg.stand_still_scale)
        rew[self._command_is_moving(ctx.info["commands"])] = 0.0
        return rew

    def _reward_collision(self, ctx: RewardContext) -> np.ndarray:
        contacts = self._get_undesired_contacts()
        return np.sum((contacts > 0.5).astype(self._np_dtype), axis=1)

    def _reward_feet_contact_number(self, ctx: RewardContext) -> np.ndarray:
        # Reward foot contact matching the gait stance phase (UniFP: +1 / -0.3).
        contact = self._foot_contact
        stance = self._stance_mask > 0.5
        reward = np.where(contact == stance, 1.0, -0.3)
        return np.mean(reward, axis=1).astype(self._np_dtype)

    def _reward_feet_air_time(self, ctx: RewardContext) -> np.ndarray:
        thd = self._reward_cfg.feet_air_time_threshold
        rew = np.sum((self._air_time_snapshot - thd) * self._first_contact, axis=1)
        rew = rew * self._command_is_moving(ctx.info["commands"])
        return rew.astype(self._np_dtype)

    def _reward_feet_height(self, ctx: RewardContext) -> np.ndarray:
        # Front feet only: encourage clearance up to the target (UniFP clamp max=0).
        feet_z = self.get_foot_pos()[:, :2, 2]
        rew = np.clip(np.max(feet_z, axis=1) - self._reward_cfg.feet_height_target, None, 0.0)
        rew[~self._command_is_moving(ctx.info["commands"])] = 0.0
        return rew.astype(self._np_dtype)

    def _reward_feet_height_high(self, ctx: RewardContext) -> np.ndarray:
        # Penalize lifting any foot above the high threshold (UniFP clamp min=0).
        feet_z = self.get_foot_pos()[:, :, 2]
        rew = np.clip(np.max(feet_z, axis=1) - self._reward_cfg.feet_height_high_target, 0.0, None)
        rew[~self._command_is_moving(ctx.info["commands"])] = 0.0
        return rew.astype(self._np_dtype)

    def _reward_feet_pos_xy(self, ctx: RewardContext) -> np.ndarray:
        feet_xy = self.get_foot_pos()[:, :, :2]
        thigh_xy = self._get_thigh_pos()[:, :, :2]
        diff = np.linalg.norm(feet_xy - thigh_xy, axis=2)
        return np.mean(diff, axis=1).astype(self._np_dtype)

    def _reward_feet_drag(self, ctx: RewardContext) -> np.ndarray:
        feet_vel = np.sum(np.abs(self._get_foot_vel()), axis=2)
        foot_forces = np.linalg.norm(self._get_foot_force_vec(), axis=2)
        return np.sum(foot_forces * feet_vel, axis=1).astype(self._np_dtype)

    def _reward_feet_contact_forces(self, ctx: RewardContext) -> np.ndarray:
        forces = np.linalg.norm(self._get_foot_force_vec(), axis=2)
        over = np.clip(forces - self._reward_cfg.max_contact_force, 0.0, None)
        return np.sum(over, axis=1).astype(self._np_dtype)
