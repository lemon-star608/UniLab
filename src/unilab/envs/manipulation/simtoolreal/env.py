"""SimToolReal env skeleton (migration task T0).

Scope: the class skeleton, the ``state.info`` bus, canonical/backend joint
permutations, MJCF scene wiring, and backend hookup. The algorithm bodies are
owned by later tasks:

* ``apply_action`` returns zeros — the real action pipeline (delay queue, arm
  velocity-delta, hand absolute mapping, EMA, wrench DR) is T1.
* ``update_state`` returns the state unchanged — observations are T2, reward and
  the ``d*`` progress terms are T3, goal advance / episode lifecycle is T4,
  keypoints are T5, and the delay queues are T6.

Those two stubs are the intended T0 deliverable, not defects.

All arrays are numpy ``float32`` on CPU (MIGRATION_00 decision D0). Quaternions
are ``wxyz`` internally; the ``xyzw`` conversion happens only when packing
observations, which is T2's job (decision D3).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.dr import DomainRandomizationProvider
from unilab.dtype_config import get_global_dtype

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

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """Return backend control for this step. **T0 stub: always zeros.**

        The real pipeline (canonical->backend permute, action delay, arm
        velocity-delta with double clamp, hand absolute mapping, EMA smoothing,
        wrench DR) is T1, ported from action_utils.py:18-75.

        Args:
            actions: Policy actions, shape ``(num_envs, 29)``, canonical order.
            state: Current env state. Untouched by this stub.

        Returns:
            Zero control of shape ``(num_envs, 29)`` in backend actuator order.
        """
        del actions, state
        return np.zeros((self._num_envs, self._num_action), dtype=self._np_dtype)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """Return the state unchanged. **T0 stub.**

        Observations (T2), reward and ``d*`` progress (T3), success/goal advance
        and termination (T4) all land here later.

        Args:
            state: Current env state.

        Returns:
            The same state instance, unmodified.
        """
        return state

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
