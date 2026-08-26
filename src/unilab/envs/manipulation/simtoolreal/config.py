"""Config schema for the SimToolReal goal-pose-reaching task.

Every default is copied from the SimToolReal source configclasses in
``isaacsimenvs/tasks/simtoolreal/simtoolreal_env_cfg.py``. Source line numbers
are cited per field group.

Two deliberate regroupings relative to the source (interface contract §5.0):

* The source has **no** ``GoalCfg``. Goal-side fields live in the source's
  ``ResetCfg`` (goal sampling), ``TerminationCfg`` (success thresholds), and
  ``RewardCfg`` (``keypoint_scale``). The contract groups them under
  :class:`GoalCfg`; each field cites its true source location.
* The source's ``StudentObsCfg`` (cfg:154) is a distillation-only path and is
  **not** ported.

``max_episode_seconds`` is UniLab's ``EnvCfg`` field name for what the source
calls ``episode_length_s`` (cfg:548). UniLab derives
``max_episode_steps = max_episode_seconds / ctrl_dt``, which must equal the
source's ``TerminationCfg.episode_length`` (cfg:431); :meth:`SimToolRealCfg.validate`
enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.base import EnvCfg
from unilab.base.scene import SceneCfg

from .constants import NUM_JOINTS, compute_obs_dim

SIMTOOLREAL_SCENE_FILE: str = str(ASSETS_ROOT_PATH / "robots" / "kuka_sharpa" / "scene.xml")


@dataclass
class AssetsCfg:
    """Asset paths, procedural-pool knobs, and static per-material frictions.

    Source: ``AssetsCfg`` (cfg:56-101). Deployment-dead Source URDF paths are
    omitted; provenance lives with the shipped MJCF. Frictions are baked into
    that MJCF rather than randomized per reset.
    """

    # Per-env table mesh scaling at scene-build time (cfg:65-70).
    table_scale_range_x: tuple[float, float] = (1.0, 1.0)
    table_scale_range_y: tuple[float, float] = (1.0, 1.0)
    table_scale_num_variants: int = 1

    # Object pool selection (cfg:72-94). The reduced pool has 50 samples per matching
    # distribution (12 distributions × 50 = 600 compiled scenes).
    object_name: str = "handle_head_primitives"
    object_urdf: str = ""
    object_scale: tuple[float, float, float] | None = None
    handle_head_types: tuple[str, ...] = (
        "hammer",
        "screwdriver",
        "marker",
        "spatula",
        "eraser",
        "brush",
    )
    num_assets_per_type: int = 50
    shuffle_assets: bool = True
    object_pool_enabled: bool = True
    object_pool_seed: int = 42

    # Static per-material friction records (cfg:97-101). These values are baked
    # into the shipped MJCF and cannot be changed by runtime config overrides.
    # Fingertip friction applies only to FINGERTIP_LINK_NAMES (scene_utils.py:1584).
    modify_asset_frictions: bool = True
    robot_friction: float = 0.5
    finger_tip_friction: float = 1.5
    object_friction: float = 0.5
    table_friction: float = 0.5


@dataclass
class ObsCfg:
    """Asymmetric actor-critic observation layout and clamping (cfg:108-145)."""

    # Critic sees the full state list; the actor sees the obs_list subset.
    state_list: tuple[str, ...] = (
        "joint_pos",
        "joint_vel",
        "prev_action_targets",
        "palm_pos",
        "palm_rot",
        "palm_vel",
        "object_rot",
        "object_vel",
        "fingertip_pos_rel_palm",
        "keypoints_rel_palm",
        "keypoints_rel_goal",
        "object_scales",
        "closest_keypoint_max_dist",
        "closest_fingertip_dist",
        "lifted_object",
        "progress",
        "successes",
        "reward",
    )
    obs_list: tuple[str, ...] = (
        "joint_pos",
        "joint_vel",
        "prev_action_targets",
        "palm_pos",
        "palm_rot",
        "object_rot",
        "fingertip_pos_rel_palm",
        "keypoints_rel_palm",
        "keypoints_rel_goal",
        "object_scales",
    )

    clamp_abs_observations: float = 10.0


@dataclass
class ActionCfg:
    """Joint-position-target control with moving-average smoothing (cfg:317-322)."""

    arm_moving_average: float = 0.1
    hand_moving_average: float = 0.1
    dof_speed_scale: float = 1.5
    clip_actions: float = 1.0


@dataclass
class RewardCfg:
    """Reward term scales (cfg:331-350).

    ``keypoint_scale`` lives on :class:`GoalCfg` in this port even though the
    source keeps it here (cfg:337) — see the module docstring. Read it as
    ``cfg.goal.keypoint_scale``.
    """

    keypoint_rew_scale: float = 200.0
    object_base_size: float = 0.04
    # Fixed keypoint extent used by the reward/success path (cfg:339).
    fixed_size: tuple[float, float, float] = (0.141, 0.03025, 0.0271)
    fixed_size_keypoint_reward: bool = True

    lifting_rew_scale: float = 20.0
    lifting_bonus: float = 300.0
    lifting_bonus_threshold: float = 0.15

    distance_delta_rew_scale: float = 50.0
    reach_goal_bonus: float = 1000.0

    kuka_actions_penalty_scale: float = 0.03
    hand_actions_penalty_scale: float = 0.003


@dataclass
class GoalCfg:
    """Goal sampling plus success criterion.

    Regrouped from three source configclasses (interface contract §5.0):

    ==========================  ===================================  ======
    field                       source location                      line
    ==========================  ===================================  ======
    ``goal_sampling_type``      ``ResetCfg.goal_sampling_type``       389
    ``delta_goal_distance``     ``ResetCfg.delta_goal_distance``      390
    ``delta_rotation_degrees``  ``ResetCfg.delta_rotation_degrees``   391
    ``mins``                    ``ResetCfg.target_volume_mins``       392
    ``maxs``                    ``ResetCfg.target_volume_maxs``       393
    ``target_volume_region_scale``  ``ResetCfg....region_scale``      394
    ``success_tolerance``       ``TerminationCfg.success_tolerance``  433
    ``target_success_tolerance``  ``TerminationCfg....``              434
    ``success_steps``           ``TerminationCfg.success_steps``      437
    ``keypoint_scale``          ``RewardCfg.keypoint_scale``          337
    ==========================  ===================================  ======

    The success threshold is ``current_success_tolerance * keypoint_scale``, not
    ``success_tolerance`` alone (obs_utils.py:195), so the 0.075 default means an
    effective 0.1125 m gate.
    """

    goal_sampling_type: str = "delta"  # "delta" | "absolute"
    delta_goal_distance: float = 0.1
    delta_rotation_degrees: float = 90.0
    mins: tuple[float, float, float] = (-0.35, -0.2, 0.6)
    maxs: tuple[float, float, float] = (0.35, 0.2, 0.95)
    target_volume_region_scale: float = 1.0

    success_tolerance: float = 0.075  # curriculum start
    target_success_tolerance: float = 0.01  # curriculum floor
    eval_success_tolerance: float | None = None
    success_steps: int = 10

    keypoint_scale: float = 1.5


@dataclass
class ResetCfg:
    """Initial-state distribution (source ``ResetCfg``, cfg:359-386).

    Goal-sampling fields from the source class live in :class:`GoalCfg`. This
    owner keeps robot/object pose sampling fields and exposes two explicit pose
    modes: ``source_random`` for upstream full-SO(3) behavior and
    ``horizontal_near_table`` for the shipped MuJoCo task.
    """

    # ``source_random`` preserves the upstream full-SO(3), table-relative
    # sampler. The MuJoCo owner selects ``horizontal_near_table`` to keep the
    # authored +x tool axis horizontal and write an absolute object z without
    # stacking the source-compatible table reference/offset/noise terms.
    object_pose_mode: str = "source_random"
    horizontal_near_table_z: float = 0.575

    # Initial object pose noise (cfg:363-366).
    reset_position_noise_x: float = 0.1
    reset_position_noise_y: float = 0.1
    reset_position_noise_z: float = 0.02
    fixed_start_pose: tuple[float, float, float, float, float, float, float] | None = None

    # Joint state noise on reset (cfg:369-371).
    reset_dof_pos_random_interval_arm: float = 0.1
    reset_dof_pos_random_interval_fingers: float = 0.1
    reset_dof_vel_random_interval: float = 0.5

    # DexToolBench eval pose offset (cfg:375).
    start_arm_higher: bool = False

    # The source samples a table-height variable. In this MuJoCo port it is an
    # object-spawn reference only; the table remains fixed in the MJCF.
    table_reset_z: float = 0.38
    object_spawn_z_reference_range: float = 0.0
    table_object_z_offset: float = 0.25
    table_reset_xy_range_m: tuple[float, float] = (0.0, 0.0)
    table_reset_yaw_range_deg: float = 0.0

    # Debug-only fixed goal pose (cfg:399): (x, y, z, qw, qx, qy, qz).
    fixed_goal_pose: tuple[float, float, float, float, float, float, float] | None = None

    # Fixed-trajectory ablation (cfg:411-412).
    fixed_trajectory_file: str = ""
    fixed_trajectory_count: int = 0


@dataclass
class TerminationCfg:
    """Episode-end conditions and the tolerance curriculum (cfg:421-444).

    ``drop_z=0.1`` and ``hand_far=1.5`` are hardcoded literals in the source
    (termination_utils.py:55,62), not config fields, so they are absent here.
    Success thresholds are on :class:`GoalCfg` (see §5.0 regrouping).
    """

    # Policy steps; 600 * decimation * sim_dt = 10 s (cfg:431).
    episode_length: int = 600

    max_consecutive_successes: int = 50
    force_consecutive_near_goal_steps: bool = False

    # Tolerance curriculum — the only curriculum in v1 (cfg:442-444). The episode-lifecycle owner applies it.
    tolerance_curriculum_increment: float = 0.9
    tolerance_curriculum_interval: int = 3000
    tolerance_curriculum_success_threshold: float = 3.0


@dataclass
class DomainRandomizationCfg:
    """Sim2real DR set (cfg:452-507).

    Scoped to per-episode / per-step perturbations. Physics-param DR (gravity,
    DOF damping/stiffness/effort/friction/armature, rigid-body mass, rigid-shape
    friction/restitution) is **not** ported in v1 — the source docstring
    (cfg:453-458) says so explicitly, and ``reset_env_state``
    (reset_utils.py:384-400) only touches poses and trackers. Hence
    ``ResetPlan.randomization`` stays ``None`` for this task.
    """

    # Obs / action latency (cfg:461-464).
    use_obs_delay: bool = True
    obs_delay_max: int = 3
    use_action_delay: bool = True
    action_delay_max: int = 3

    # Object state delay + observed-pose noise (cfg:467-473).
    use_object_state_delay_noise: bool = True
    object_state_delay_max: int = 10
    object_state_xyz_noise_std: float = 0.01
    object_state_rotation_noise_degrees: float = 5.0
    object_scale_noise_multiplier_range: tuple[float, float] = (1.0, 1.0)

    # Per-step Gaussian noise on joint-velocity obs (cfg:476).
    joint_velocity_obs_noise_std: float = 0.1

    # Random force/torque impulses on the object body (cfg:479-489).
    force_scale: float = 20.0
    force_prob_range: tuple[float, float] = (0.001, 0.1)
    force_decay: float = 0.0
    force_decay_interval: float = 0.08
    force_only_when_lifted: bool = True

    torque_scale: float = 2.0
    torque_prob_range: tuple[float, float] = (0.001, 0.1)
    torque_decay: float = 0.0
    torque_decay_interval: float = 0.08
    torque_only_when_lifted: bool = True

    # Per-env friction randomization, sampled ONCE at scene init (cfg:505-507).
    # Default (1.0, 1.0) is a no-op. Bucketing caps distinct material count.
    object_friction_scale_range: tuple[float, float] = (1.0, 1.0)
    fingertip_friction_scale_range: tuple[float, float] = (1.0, 1.0)
    friction_n_buckets: int = 16


@registry.envcfg("SimToolReal")
@dataclass
class SimToolRealCfg(EnvCfg):
    """Top-level config for the SimToolReal goal-pose-reaching task.

    Frequencies follow the task contract: ``sim_dt=1/120`` and
    ``ctrl_dt=1/60`` give ``sim_substeps=2``, matching the source's
    ``decimation=2`` + ``sim.dt=1/120`` (cfg:518,547).
    """

    scene: SceneCfg = field(default_factory=lambda: SceneCfg(model_file=SIMTOOLREAL_SCENE_FILE))

    sim_dt: float = 1.0 / 120.0
    ctrl_dt: float = 1.0 / 60.0
    # UniLab's name for the source's episode_length_s (cfg:548).
    max_episode_seconds: float = 10.0

    action_space: int = NUM_JOINTS

    assets: AssetsCfg = field(default_factory=AssetsCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    action: ActionCfg = field(default_factory=ActionCfg)
    reward_config: RewardCfg = field(default_factory=RewardCfg)
    goal: GoalCfg = field(default_factory=GoalCfg)
    reset: ResetCfg = field(default_factory=ResetCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    domain_randomization: DomainRandomizationCfg = field(default_factory=DomainRandomizationCfg)

    @property
    def num_actor_obs(self) -> int:
        """Actor observation width, summed from ``obs.obs_list``."""
        return compute_obs_dim(self.obs.obs_list)

    @property
    def num_critic_obs(self) -> int:
        """Critic observation width, summed from ``obs.state_list``."""
        return compute_obs_dim(self.obs.state_list)

    def validate(self) -> None:
        """Validate asset, frequency, and episode-length consistency.

        Raises:
            ValueError: If an asset-owned friction record differs from the
                shipped MJCF, if ``sim_dt > ctrl_dt`` (base class), if the
                derived substep count is not the source's ``decimation=2``, or
                if the derived step budget disagrees with
                ``termination.episode_length``.
        """
        super().validate()
        baked_friction_cfg: tuple[tuple[str, bool | float], ...] = (
            ("modify_asset_frictions", True),
            ("robot_friction", 0.5),
            ("finger_tip_friction", 1.5),
            ("object_friction", 0.5),
            ("table_friction", 0.5),
        )
        for field_name, baked_value in baked_friction_cfg:
            configured_value = getattr(self.assets, field_name)
            if configured_value != baked_value:
                raise ValueError(
                    f"assets.{field_name} is asset-owned and baked into the shipped MJCF; "
                    f"expected {baked_value!r}, got {configured_value!r}"
                )
        count = self.assets.num_assets_per_type
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(
                f"assets.num_assets_per_type must be a positive integer, got {count!r}"
            )

        reset_mode = self.reset.object_pose_mode
        supported_reset_modes = {"source_random", "horizontal_near_table"}
        if reset_mode not in supported_reset_modes:
            raise ValueError(
                "reset.object_pose_mode must be one of "
                f"{sorted(supported_reset_modes)!r}, got {reset_mode!r}"
            )
        if self.sim_substeps != 2:
            raise ValueError(
                "SimToolReal expects sim_substeps == 2 (source decimation=2, "
                f"cfg:547); got {self.sim_substeps} from sim_dt={self.sim_dt} "
                f"ctrl_dt={self.ctrl_dt}"
            )
        derived_steps = self.max_episode_steps
        if derived_steps != self.termination.episode_length:
            raise ValueError(
                "max_episode_seconds / ctrl_dt must equal "
                "termination.episode_length (source cfg:431); got "
                f"{derived_steps} vs {self.termination.episode_length}"
            )
        if self.action_space != NUM_JOINTS:
            raise ValueError(f"action_space must be {NUM_JOINTS}, got {self.action_space}")
        if (self.num_actor_obs, self.num_critic_obs) != (140, 162):
            raise ValueError(
                "SimToolReal observation layout must be actor=140 and critic=162, "
                f"got {self.num_actor_obs} and {self.num_critic_obs}"
            )
        if self.obs.clamp_abs_observations <= 0.0:
            raise ValueError("obs.clamp_abs_observations must be positive")
        if not 0.0 <= self.action.arm_moving_average <= 1.0:
            raise ValueError("action.arm_moving_average must be in [0, 1]")
        if not 0.0 <= self.action.hand_moving_average <= 1.0:
            raise ValueError("action.hand_moving_average must be in [0, 1]")
        if self.action.dof_speed_scale <= 0.0 or self.action.clip_actions <= 0.0:
            raise ValueError("action speed scale and clip must be positive")

        mins = tuple(self.goal.mins)
        maxs = tuple(self.goal.maxs)
        if len(mins) != 3 or len(maxs) != 3 or any(lo >= hi for lo, hi in zip(mins, maxs)):
            raise ValueError("goal mins/maxs must be ordered three-vectors")
        if self.goal.goal_sampling_type not in {"absolute", "delta"}:
            raise ValueError("goal.goal_sampling_type must be 'absolute' or 'delta'")
        if self.goal.success_steps <= 0 or self.goal.target_volume_region_scale <= 0.0:
            raise ValueError("goal success steps and workspace scale must be positive")
        if not 0.0 < self.goal.target_success_tolerance <= self.goal.success_tolerance:
            raise ValueError("goal tolerance floor must be positive and no larger than its start")

        if self.reset.object_spawn_z_reference_range != 0.0:
            raise ValueError(
                "reset.object_spawn_z_reference_range is fixed at 0.0 for the shipped table"
            )
        for name in (
            "reset_position_noise_x",
            "reset_position_noise_y",
            "reset_position_noise_z",
            "reset_dof_pos_random_interval_arm",
            "reset_dof_pos_random_interval_fingers",
            "reset_dof_vel_random_interval",
        ):
            if float(getattr(self.reset, name)) < 0.0:
                raise ValueError(f"reset.{name} must be non-negative")
        for name in ("fixed_start_pose", "fixed_goal_pose"):
            pose = getattr(self.reset, name)
            if pose is not None and len(pose) != 7:
                raise ValueError(f"reset.{name} must contain xyz+wxyz")

        dr = self.domain_randomization
        for name in ("obs_delay_max", "action_delay_max", "object_state_delay_max"):
            value = getattr(dr, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"domain_randomization.{name} must be a non-negative integer")
        for name in (
            "object_state_xyz_noise_std",
            "object_state_rotation_noise_degrees",
            "joint_velocity_obs_noise_std",
            "force_scale",
            "torque_scale",
        ):
            if float(getattr(dr, name)) < 0.0:
                raise ValueError(f"domain_randomization.{name} must be non-negative")
        for name in (
            "force_prob_range",
            "torque_prob_range",
            "object_scale_noise_multiplier_range",
            "object_friction_scale_range",
            "fingertip_friction_scale_range",
        ):
            lo, hi = getattr(dr, name)
            lower_bound = 0.0 if name.endswith("prob_range") else 1e-12
            upper_bound = 1.0 if name.endswith("prob_range") else float("inf")
            if not lower_bound < lo <= hi <= upper_bound:
                raise ValueError(f"domain_randomization.{name} has invalid range {(lo, hi)}")
