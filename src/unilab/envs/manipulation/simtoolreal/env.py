"""SimToolReal env: the two ``NpEnv`` hooks and the embodiment wiring.

T0 built the class skeleton, the ``state.info`` bus, the canonical/backend joint
permutations, the MJCF scene wiring, and the backend hookup. T7 filled the two
hooks in, calling the T1-T6 modules in the source's order:

``apply_action``  (source ``_pre_physics_step`` + ``_apply_action``,
simtoolreal_env.py:58-64)
    1. :func:`~.action_pipeline.apply_action_pipeline` (T1)
    2. :func:`~.dr_wrench.apply_wrench_dr` (T6)
    3. return ``info["cur_targets"]`` as the PD target

``update_state``  (source ``_get_dones`` -> ``_get_rewards`` ->
``_get_observations``, simtoolreal_env.py:66-77)
    1. :func:`~.episode_lifecycle.update_tolerance_curriculum` (T4)
    2. :meth:`SimToolRealEnv._compute_intermediate_values` (T5 keypoints,
       fingertip distances, d\\* sentinel resolution), then
       :func:`~.episode_lifecycle.compute_success` and
       :func:`~.episode_lifecycle.advance_goal_on_success` (T4)
    3. :func:`~.rewards.compute_rewards` (T3)
    4. :func:`~.observations.build_observations` (T2)
    5. :func:`~.episode_lifecycle.compute_terminations` (T4)

All arrays are numpy ``float32`` on CPU (MIGRATION_00 decision D0). Quaternions
are ``wxyz`` internally; the ``xyzw`` conversion happens only when packing
observations (decision D3).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.dr import DomainRandomizationProvider
from unilab.dtype_config import get_global_dtype

from .action_pipeline import apply_action_pipeline
from .config import SimToolRealCfg
from .constants import (
    DEFAULT_JOINT_POS,
    FINGERTIP_LINK_NAMES,
    FINGERTIP_OFFSET,
    JOINT_NAMES_CANONICAL,
    KEYPOINT_CORNERS,
    NUM_ARM_JOINTS,
    NUM_FINGERTIPS,
    NUM_JOINTS,
    OBJECT_BODY_NAME,
    PALM_BODY_NAME,
    PALM_CENTER_OFFSET,
    ROBOT_ROOT_BODY_NAME,
)
from .dr_provider import SimToolRealDRProvider
from .dr_wrench import apply_wrench_dr, sample_log_uniform
from .episode_lifecycle import (
    advance_goal_on_success,
    compute_success,
    compute_terminations,
    update_tolerance_curriculum,
)
from .keypoints import compute_keypoints_from_offsets, keypoint_max_dist
from .observations import build_observations
from .rewards import compute_rewards

# Object free joint contributes 7 qpos / 6 qvel entries on top of the 29 robot
# hinges. The scene authored by ``unilab.tools.build_simtoolreal_assets``
# includes the robot before the object, so the object occupies the tail.
OBJECT_QPOS_DIM = 7
OBJECT_QVEL_DIM = 6
OBJECT_HANDLE_GEOM_NAME = "object_handle"


@registry.env("SimToolReal", sim_backend="mujoco")
class SimToolRealEnv(NpEnv):
    """KUKA iiwa14 + Sharpa hand goal-pose-reaching env (T0 skeleton)."""

    _cfg: SimToolRealCfg

    def __init__(
        self,
        cfg: SimToolRealCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
        dr_provider: DomainRandomizationProvider | None = None,
    ) -> None:
        """Build the backend, cache embodiment indices, and install the DR provider.

        Args:
            cfg: Task config. Validated before the backend is created.
            num_envs: Number of vectorized environments.
            backend_type: Backend key passed to ``create_backend``.
            dr_provider: Optional DR provider override. Defaults to
                :class:`~.dr_provider.SimToolRealDRProvider`.
        """
        cfg.validate()

        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=ROBOT_ROOT_BODY_NAME,
            add_body_sensors=True,
            **env_backend_kwargs(cfg),
        )
        super().__init__(cfg, backend, num_envs)

        self._np_dtype = get_global_dtype()
        self._num_action = NUM_JOINTS

        self._build_joint_permutations()
        self._build_joint_limits()
        self._build_default_pose()
        self._build_body_ids()
        self._build_state_layout()
        self._build_geometry_constants()
        self._build_delay_queues()
        self._build_wrench_dr_state()
        self._build_step_caches()

        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._num_action,), dtype=np.float32
        )

        provider = dr_provider if dr_provider is not None else SimToolRealDRProvider()
        self._init_domain_randomization(provider)

    # ------------------------------------------------------------------ #
    # Construction helpers                                               #
    # ------------------------------------------------------------------ #

    def _build_joint_permutations(self) -> None:
        """Cache canonical<->backend joint permutations (decision D9).

        Backend order is defined by ascending ``qpos`` address, matching how the
        backend lays out ``get_dof_pos()``. Both arrays are *gather* indices, the
        same convention as the source's ``_perm_canon_to_lab`` /
        ``_perm_lab_to_canon`` (reset_utils.py:43-49):
        ``x_backend = x_canon[_perm_canon_to_backend]``.
        """
        qpos_idx = np.asarray(
            self._backend.get_joint_dof_pos_indices(list(JOINT_NAMES_CANONICAL)),
            dtype=np.int64,
        )
        qvel_idx = np.asarray(
            self._backend.get_joint_dof_vel_indices(list(JOINT_NAMES_CANONICAL)),
            dtype=np.int64,
        )
        if qpos_idx.shape != (NUM_JOINTS,):
            raise ValueError(
                f"Expected {NUM_JOINTS} canonical joints in the model, got {qpos_idx.shape}"
            )

        # Rank of each canonical joint among the 29 by qpos address == its
        # backend slot. Gathering canonical data with this yields backend order.
        backend_slot_of_canon = np.argsort(np.argsort(qpos_idx)).astype(np.int64)
        self._perm_canon_to_backend = np.argsort(backend_slot_of_canon).astype(np.int64)
        self._perm_backend_to_canon = backend_slot_of_canon

        # Canonical-order addresses into the backend's dof_pos / dof_vel views.
        self._dof_pos_idx_canon = qpos_idx
        self._dof_vel_idx_canon = qvel_idx

        self._arm_slice = slice(0, NUM_ARM_JOINTS)
        self._hand_slice = slice(NUM_ARM_JOINTS, NUM_JOINTS)

    def _build_joint_limits(self) -> None:
        """Cache joint position limits in both canonical and backend order.

        The scene authors one position actuator per canonical joint with
        ``inheritrange="1"``, so the actuator ctrl range equals the joint range.
        Mirrors the source split (reset_utils.py:53-62): ``_joint_*_canon`` are
        canonical-order (observation normalization), while ``_arm_*``/``_hand_*``
        are backend-order (action target clamping).
        """
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=self._np_dtype)
        if ctrl_range.shape[0] != NUM_JOINTS:
            raise ValueError(
                f"Scene must expose exactly {NUM_JOINTS} actuators, got {ctrl_range.shape[0]}"
            )

        lower_backend = ctrl_range[:, 0]
        upper_backend = ctrl_range[:, 1]

        self._joint_lower_canon = np.ascontiguousarray(
            lower_backend[self._perm_backend_to_canon], dtype=self._np_dtype
        )
        self._joint_upper_canon = np.ascontiguousarray(
            upper_backend[self._perm_backend_to_canon], dtype=self._np_dtype
        )

        self._arm_lower = np.ascontiguousarray(lower_backend[self._arm_slice], dtype=self._np_dtype)
        self._arm_upper = np.ascontiguousarray(upper_backend[self._arm_slice], dtype=self._np_dtype)
        self._hand_lower = np.ascontiguousarray(
            lower_backend[self._hand_slice], dtype=self._np_dtype
        )
        self._hand_upper = np.ascontiguousarray(
            upper_backend[self._hand_slice], dtype=self._np_dtype
        )

    def _build_default_pose(self) -> None:
        """Cache the default joint pose in canonical and backend order.

        Values come from ``DEFAULT_JOINT_POS`` (arm pose scene_utils.py:127, hand
        joints 0.0 per scene_utils.py:167), clipped into the joint limits so a
        restricted URDF can never seed an out-of-range target.
        """
        canon = np.asarray(
            [DEFAULT_JOINT_POS[name] for name in JOINT_NAMES_CANONICAL], dtype=self._np_dtype
        )
        canon = np.clip(canon, self._joint_lower_canon, self._joint_upper_canon)
        self._default_joint_pos_canon = np.ascontiguousarray(canon, dtype=self._np_dtype)
        self._default_joint_pos_backend = np.ascontiguousarray(
            canon[self._perm_canon_to_backend], dtype=self._np_dtype
        )

    def _build_body_ids(self) -> None:
        """Resolve palm, fingertip, and object body ids from the backend."""
        self._palm_body_id = int(self._backend.get_body_ids([PALM_BODY_NAME])[0])
        self._fingertip_body_ids = np.asarray(
            self._backend.get_body_ids(list(FINGERTIP_LINK_NAMES)), dtype=np.int32
        )
        if self._fingertip_body_ids.shape[0] != NUM_FINGERTIPS:
            raise ValueError(
                f"Expected {NUM_FINGERTIPS} fingertip bodies, got "
                f"{self._fingertip_body_ids.shape[0]}"
            )
        self._object_body_id = int(self._backend.get_body_ids([OBJECT_BODY_NAME])[0])

    def _build_state_layout(self) -> None:
        """Cache qpos/qvel sizes and the object free-joint slices.

        The object free joint is the scene's last joint, so it occupies the tail
        of ``qpos``/``qvel``. Asserted here so a scene edit that reorders the
        includes fails loudly instead of silently corrupting resets.
        """
        self.nq = int(np.asarray(self._backend.get_default_qpos()).shape[0])
        self.nv = int(np.asarray(self._backend.get_init_qvel()).shape[0])

        expected_nq = NUM_JOINTS + OBJECT_QPOS_DIM
        expected_nv = NUM_JOINTS + OBJECT_QVEL_DIM
        if (self.nq, self.nv) != (expected_nq, expected_nv):
            raise ValueError(
                "SimToolReal scene must contain 29 robot hinges plus one object "
                f"free joint (nq={expected_nq}, nv={expected_nv}); got "
                f"nq={self.nq}, nv={self.nv}"
            )

        obj_qpos_start = self.nq - OBJECT_QPOS_DIM
        self._obj_pos_slice = slice(obj_qpos_start, obj_qpos_start + 3)
        self._obj_quat_slice = slice(obj_qpos_start + 3, obj_qpos_start + 7)
        self._obj_qvel_slice = slice(self.nv - OBJECT_QVEL_DIM, self.nv)

    def _build_geometry_constants(self) -> None:
        """Cache palm/fingertip offsets and the fixed reward keypoint offsets.

        ``_keypoint_offsets_fixed`` follows reset_utils.py:80-83:
        ``corners * 0.5 * keypoint_scale * reward.fixed_size``. The per-object
        (``phi``-scaled) observation keypoints are T5's job.
        """
        self._palm_offset = np.asarray(PALM_CENTER_OFFSET, dtype=self._np_dtype)
        self._fingertip_offset = np.asarray(FINGERTIP_OFFSET, dtype=self._np_dtype)
        self._kp_corners = np.asarray(KEYPOINT_CORNERS, dtype=self._np_dtype)

        fixed_size = np.asarray(self._cfg.reward.fixed_size, dtype=self._np_dtype)
        keypoint_scale = float(self._cfg.goal.keypoint_scale)
        self._keypoint_offsets_fixed = np.ascontiguousarray(
            self._kp_corners * (0.5 * keypoint_scale * fixed_size)[None, :],
            dtype=self._np_dtype,
        )

    def _build_delay_queues(self) -> None:
        """Allocate the delay queues and wrench-DR buffers (contract §3).

        These live as instance attributes rather than in ``state.info`` because
        they carry per-step temporal state (decision D5). T6 owns the push/sample
        logic; T0 only allocates them.
        """
        dr = self._cfg.domain_randomization
        n = self._num_envs

        self._action_queue = np.zeros(
            (n, max(int(dr.action_delay_max), 1), NUM_JOINTS), dtype=self._np_dtype
        )
        self._obs_queue = np.zeros(
            (n, max(int(dr.obs_delay_max), 1), self._cfg.num_actor_obs), dtype=self._np_dtype
        )
        # Object state frame: pos(3) + quat(4) + lin/ang vel(6).
        self._object_state_queue = np.zeros(
            (n, max(int(dr.object_state_delay_max), 1), 13), dtype=self._np_dtype
        )

        self._object_forces = np.zeros((n, 3), dtype=self._np_dtype)
        self._object_torques = np.zeros((n, 3), dtype=self._np_dtype)

    def _build_wrench_dr_state(self) -> None:
        """Cache the object mass and per-env wrench trigger probabilities (T6/§4.8).

        ``backend.get_body_mass()`` takes **no arguments** and returns the whole
        ``(nbody,)`` model table (backend/base.py:191, mujoco/backend.py:731-732),
        so the object entry is indexed out and broadcast to ``(N,)`` — the shape
        ``dr_wrench.apply_wrench_dr`` expects.

        The trigger probabilities are drawn from log-uniform
        ``[force_prob_range]`` (source action_utils.py:10-15). The source redraws
        them on every reset (reset_utils.py:409-414); that resample lives in the
        DR provider's ``build_reset_plan``, and this is the init-time seed.
        """
        dr = self._cfg.domain_randomization
        n = self._num_envs

        mass_table = np.asarray(self._backend.get_body_mass(), dtype=self._np_dtype)
        self._object_mass = np.full(
            (n,), float(mass_table[self._object_body_id]), dtype=self._np_dtype
        )

        self._random_force_prob = sample_log_uniform(
            dr.force_prob_range[0], dr.force_prob_range[1], n
        ).astype(self._np_dtype)
        self._random_torque_prob = sample_log_uniform(
            dr.torque_prob_range[0], dr.torque_prob_range[1], n
        ).astype(self._np_dtype)

        # Per-env object-extent noise on the observation keypoints
        # (obs_utils.py:275). Default range (1.0, 1.0) is a no-op; the provider
        # resamples it per reset (reset_utils.py:415-418).
        self._object_scale_multiplier = np.ones((n, 3), dtype=self._np_dtype)

    def _build_step_caches(self) -> None:
        """Allocate the per-step intermediate buffers shared by T2/T3/T4/T6.

        ``compute_intermediate_values`` (obs_utils.py:153-202) recomputes all of
        these every control step; they are allocated here so the reward and
        observation modules can rely on them existing from step zero.

        ``_state_cache_lifted_object`` is the one exception: ``apply_action``
        runs *before* ``update_state``, so the wrench DR gate must read the
        **previous** step's latch, matching the source comment
        "``_lifted_object`` is from the previous step because rewards update
        later" (action_utils.py:109).
        """
        n = self._num_envs

        self._curr_fingertip_distances = np.zeros((n, NUM_FINGERTIPS), dtype=self._np_dtype)
        self._keypoints_max_dist = np.zeros((n,), dtype=self._np_dtype)
        self._near_goal = np.zeros((n,), dtype=bool)
        self._is_success = np.zeros((n,), dtype=bool)
        self._joint_vel = np.zeros((n, NUM_JOINTS), dtype=self._np_dtype)
        self._object_pos = np.zeros((n, 3), dtype=self._np_dtype)
        self._object_quat = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=self._np_dtype), (n, 1))
        self._state_cache_lifted_object = np.zeros((n,), dtype=bool)
        self._reward_terms: dict[str, np.ndarray] = {}

        # Tolerance curriculum trackers (termination_utils.py:12-14). T4
        # initializes these lazily; seeding them here means
        # ``_current_success_tolerance`` is readable before the first step.
        self._frame_counter = 0
        self._last_curriculum_update = 0
        self._current_success_tolerance = float(self._cfg.goal.success_tolerance)

    # ------------------------------------------------------------------ #
    # Env contract                                                       #
    # ------------------------------------------------------------------ #

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        """Asymmetric actor/critic observation widths, summed from ObsCfg lists."""
        return {"obs": self._cfg.num_actor_obs, "critic": self._cfg.num_critic_obs}

    @property
    def action_space(self) -> gym.spaces.Box:
        """Normalized joint-target action space, shape ``(29,)`` in canonical order."""
        return self._action_space

    @property
    def cfg(self) -> SimToolRealCfg:
        """Return the task config, narrowed to :class:`SimToolRealCfg`.

        ``NpEnv.cfg`` is annotated as returning the base ``EnvCfg``, which has no
        ``obs`` / ``action`` / ``reward`` / ``goal`` / ``domain_randomization``
        fields. Every T1-T3 module reads those through ``env.cfg``, so without
        this narrowing the type checker rejects all of them. Runtime behaviour is
        unchanged — it is the same object, just correctly typed.
        """
        return self._cfg

    @property
    def backend(self):
        """Return the simulation backend.

        ``NpEnv`` only exposes ``_backend``, but the T6 wrench module calls
        ``env.backend.apply_body_wrench`` (dr_wrench.py:135). The public alias
        lives here, in the subclass, so the base class stays untouched (D1).
        """
        return self._backend

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """Turn policy actions into PD joint targets and stage the wrench DR.

        Reproduces the pre-physics half of the source control step
        (simtoolreal_env.py:58-64), in order:

        1. :func:`~.action_pipeline.apply_action_pipeline` — canonical->backend
           permute, action delay, arm velocity-delta (double clamp), hand
           absolute mapping, EMA (T1, action_utils.py:18-75).
        2. :func:`~.dr_wrench.apply_wrench_dr` — per-env random force/torque
           impulse on the object, gated on the **previous** step's lifted latch
           (T6, action_utils.py:77-130).
        3. Return ``info["cur_targets"]``. The base class forwards it to
           ``backend.step`` as the position target for both sim substeps, which
           is what the source's ``_apply_action`` (:62-64) does once per
           decimation tick.

        Args:
            actions: Policy actions, shape ``(num_envs, 29)``, canonical joint
                order, nominally ``[-1, 1]`` but deliberately not clamped
                (guide §4).
            state: Current env state; ``info`` is read and written in place.

        Returns:
            Joint position targets, shape ``(num_envs, 29)``, backend actuator
            order, float32.
        """
        apply_action_pipeline(self, actions)
        apply_wrench_dr(self)
        return np.asarray(state.info["cur_targets"], dtype=self._np_dtype)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """Run the post-physics half of the control step.

        Reproduces the source's ``_get_dones`` -> ``_get_rewards`` ->
        ``_get_observations`` order (simtoolreal_env.py:66-77). One deliberate
        relocation: the source advances the goal *inside*
        ``compute_terminations`` (termination_utils.py:47-51), which runs before
        ``_get_rewards``. Here the advance is called right after the success gate
        — same position in the sequence, so the reward still sees the reset d*
        trackers and the incremented ``successes``.

        Args:
            state: Current env state; ``obs``, ``reward``, ``terminated``,
                ``truncated``, and ``info`` are all written in place.

        Returns:
            The same state instance, updated.
        """
        info = state.info

        # 1. Tolerance curriculum, before success detection (T4).
        update_tolerance_curriculum(self)

        # 2. Shared geometry: keypoints, fingertip distances, d* sentinels (T5).
        self._compute_intermediate_values(info)

        # 2e/2f. Success gate, then intra-episode goal advance (T4, D2). The
        #        advance zeroes info["steps"], so the base class's steps += 1
        #        (np_env.py:205) yields 1 and truncation cannot fire on success.
        is_success = compute_success(self, self._keypoints_max_dist)
        advance_goal_on_success(self, is_success)

        # 3. Reward (T3), no global scaling (reward_utils.py:141).
        reward = compute_rewards(self, info)
        state.reward[:] = reward
        # Critic reward feature reads this step's reward (obs_utils.py:326).
        info["reward"][:] = reward

        # 4. Observations (T2), after the reward so the critic's reward feature
        #    and lifted latch are current.
        for group, values in build_observations(self, state).items():
            state.obs[group][:] = values

        # 5. Terminations (T4). Success is deliberately excluded (D2).
        terminated, truncated = compute_terminations(self, is_success)
        state.terminated[:] = terminated
        state.truncated[:] = truncated

        # 6. Snapshot the latch for the next step's wrench gate (action_utils.py:109).
        self._state_cache_lifted_object = np.asarray(info["lifted_object"], dtype=bool).copy()

        self._publish_step_log(info, is_success)
        return state

    def _compute_intermediate_values(self, info: dict) -> None:
        """Refresh the geometry shared by reward, success, and termination.

        numpy port of ``obs_utils.compute_intermediate_values``
        (obs_utils.py:153-202), minus the success gate itself (T4 owns that).

        Fills ``_joint_vel``, ``_object_pos``, ``_object_quat``,
        ``_curr_fingertip_distances``, and ``_keypoints_max_dist``, then resolves
        the ``d*`` sentinels.

        The reward/success keypoints use the **fixed** offsets
        (``_keypoint_offsets_fixed``, cfg ``fixed_size_keypoint_reward=True``),
        not the per-object ``phi``-scaled ones the observations use
        (obs_utils.py:172-175, contract §4.3).

        Fingertip distances use the **raw** fingertip body origins, with no
        ``FINGERTIP_OFFSET`` applied (obs_utils.py:166-170) — the offset is an
        observation-only shift (obs_utils.py:252-257).

        Args:
            info: ``state.info``; the ``closest_*`` d* trackers are updated in
                place.
        """
        self._joint_vel = self.get_joint_vel_canon()

        obj_ids = np.asarray([self._object_body_id], dtype=np.int32)
        self._object_pos = np.asarray(
            self._backend.get_body_pos_w(obj_ids)[:, 0, :], dtype=self._np_dtype
        )
        self._object_quat = np.asarray(
            self._backend.get_body_quat_w(obj_ids)[:, 0, :], dtype=self._np_dtype
        )

        # Fingertip -> object distances, raw body origins (obs_utils.py:166-170).
        ft_pos = np.asarray(
            self._backend.get_body_pos_w(self._fingertip_body_ids), dtype=self._np_dtype
        )
        self._curr_fingertip_distances = np.linalg.norm(
            ft_pos - self._object_pos[:, None, :], axis=-1
        ).astype(self._np_dtype)

        # Reward/success keypoints: fixed offsets (obs_utils.py:172-180).
        obj_kp = compute_keypoints_from_offsets(
            self._object_pos, self._object_quat, self._keypoint_offsets_fixed
        )
        goal_kp = compute_keypoints_from_offsets(
            np.asarray(info["goal_pos"], dtype=self._np_dtype),
            np.asarray(info["goal_quat"], dtype=self._np_dtype),
            self._keypoint_offsets_fixed,
        )
        self._keypoints_max_dist = keypoint_max_dist(obj_kp, goal_kp)

        # d* sentinel resolution (obs_utils.py:182-190). The -1 seeded at reset
        # and at every goal advance means "no history yet"; the first observed
        # distance becomes closest-so-far. Skipping this would leave d* at -1, so
        # every delta would be negative, clip to 0, and both progress rewards
        # would stay dead for the whole run.
        kp_star = info["closest_keypoint_max_dist"]
        np.copyto(kp_star, self._keypoints_max_dist, where=kp_star < 0.0)
        ft_star = info["closest_fingertip_dist"]
        np.copyto(ft_star, self._curr_fingertip_distances, where=ft_star < 0.0)

    def _publish_step_log(self, info: dict, is_success: np.ndarray) -> None:
        """Write per-term reward means and episode counters into ``info["log"]``.

        Mirrors ``logging_utils.log_step_metrics`` (logging_utils.py:8-31), which
        publishes ``_reward_terms`` plus the success counters. ``log`` is a plain
        dict, so the base class copies the reference verbatim rather than
        scattering it per env (np_env.py:288-298, contract §2.4).

        Args:
            info: ``state.info``.
            is_success: This step's success mask, shape ``(num_envs,)``.
        """
        log: dict[str, float] = {
            f"reward/{name}": float(np.mean(term)) for name, term in self._reward_terms.items()
        }
        log["success/step_rate"] = float(np.mean(is_success))
        log["success/episode_mean"] = float(np.mean(info["successes"]))
        log["success/prev_episode_mean"] = float(np.mean(info["prev_episode_successes"]))
        log["success/current_tolerance"] = float(self._current_success_tolerance)
        log["object/lifted_rate"] = float(np.mean(info["lifted_object"]))
        log["object/keypoint_max_dist"] = float(np.mean(self._keypoints_max_dist))
        info["log"] = log

    # ------------------------------------------------------------------ #
    # Read helpers shared by later tasks                                 #
    # ------------------------------------------------------------------ #

    def get_joint_pos_canon(self) -> np.ndarray:
        """Return joint positions in canonical order, shape ``(num_envs, 29)``."""
        return np.asarray(
            self._backend.get_dof_pos()[:, self._dof_pos_idx_canon], dtype=self._np_dtype
        )

    def get_joint_vel_canon(self) -> np.ndarray:
        """Return joint velocities in canonical order, shape ``(num_envs, 29)``."""
        return np.asarray(
            self._backend.get_dof_vel()[:, self._dof_vel_idx_canon], dtype=self._np_dtype
        )

    def get_object_pos(self) -> np.ndarray:
        """Return the object body world position, shape ``(num_envs, 3)``."""
        return np.asarray(
            self._backend.get_body_pos_w(np.asarray([self._object_body_id], dtype=np.int32))[
                :, 0, :
            ],
            dtype=self._np_dtype,
        )

    def get_object_quat(self) -> np.ndarray:
        """Return the object body world orientation (wxyz), shape ``(num_envs, 4)``."""
        return np.asarray(
            self._backend.get_body_quat_w(np.asarray([self._object_body_id], dtype=np.int32))[
                :, 0, :
            ],
            dtype=self._np_dtype,
        )

    @property
    def default_joint_pos_backend(self) -> np.ndarray:
        """Default joint pose in backend order, shape ``(29,)``."""
        return self._default_joint_pos_backend

    def resolve_object_scale(self) -> np.ndarray:
        """Return the handle bbox scale ``phi``, normalized by ``object_base_size``.

        Only the handle contributes, per migration guide §2 — the head is
        excluded. MuJoCo box ``geom_size`` stores half-extents, hence the ``2 *``.

        Returns:
            Array of shape ``(3,)``.
        """
        half_extent = np.asarray(
            self._backend.get_geom_size(OBJECT_HANDLE_GEOM_NAME), dtype=self._np_dtype
        )[:3]
        base = float(self._cfg.reward.object_base_size)
        return np.asarray(2.0 * half_extent / base, dtype=self._np_dtype)


__all__ = ["SimToolRealEnv"]
