"""Domain-randomization provider for SimToolReal (minimal T0 version).

The base ``step`` loop routes every reset through the provider
(``np_env.py:247,388-391``), and ``init_state`` marks all envs terminated so the
very first call already needs a working provider. T0 therefore ships the minimal
runnable version:

* :meth:`SimToolRealDRProvider.validate` declares **no** reset DR terms.
* :meth:`SimToolRealDRProvider.build_reset_plan` writes the default robot pose
  plus the nominal object pose and seeds every ``state.info`` key from interface
  contract §2, with ``ResetPlan.randomization=None``.
* :meth:`SimToolRealDRProvider.build_reset_observation` returns zero observations.

The real reset distribution (object / table / robot pose sampling, first absolute
goal sample, tolerance curriculum — reset_utils.py:200-302) is T4.

No per-reset physics-parameter DR is ported: the source config docstring says so
explicitly (simtoolreal_env_cfg.py:453-458) and ``reset_env_state``
(reset_utils.py:384-400) only touches poses and trackers. UniLab's
``ResetRandomizationPayload`` supports mass/friction/kp/kd/gravity/armature, but
capability is not licence (MIGRATION_00 decision D5).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.dr import (
    DomainRandomizationCapabilities,
    DomainRandomizationProvider,
    IntervalRandomizationPlan,
    ResetPlan,
)
from unilab.dtype_config import get_global_dtype

from .constants import NUM_FINGERTIPS, NUM_JOINTS
from .dr_wrench import sample_log_uniform
from .goal_sampling import np_random_orientation, sample_absolute_goal

# Sentinel for the d* progress trackers: -1 means "no history yet"
# (migration guide §5).
DSTAR_SENTINEL: float = -1.0


class SimToolRealDRProvider(DomainRandomizationProvider):
    """Reset provider for SimToolReal. T0 ships the minimal runnable version."""

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        """Check that the backend can service resets.

        Declares no reset randomization terms on purpose (decision D5): this port
        has no per-reset physics-parameter DR, so there is nothing to negotiate
        against ``capabilities``.

        Args:
            env: Owning env instance.
            capabilities: Backend-declared DR capabilities. Unused — see above.

        Raises:
            RuntimeError: If the backend does not expose ``set_state``, which the
                reset path requires.
        """
        del capabilities
        if not callable(getattr(env._backend, "set_state", None)):
            raise RuntimeError(
                f"{type(env._backend).__name__} does not implement set_state(), "
                "which SimToolReal resets require"
            )

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        """Build the reset state with full pose randomization (T4 completion).

        Samples object/robot poses, table height, and first absolute goal. Tolerance
        curriculum update is handled externally before reset (termination_utils.py:10-36).

        Args:
            env: Owning :class:`~.env.SimToolRealEnv`.
            env_ids: Indices of environments being reset.

        Returns:
            A :class:`~unilab.dr.types.ResetPlan` whose ``randomization`` is
            ``None`` by design (decision D5: no per-reset physics-param DR).
        """
        num_reset = int(len(env_ids))
        if num_reset == 0:
            return ResetPlan(
                env_ids=env_ids,
                qpos=np.zeros((0, env.nq), dtype=np.float64),
                qvel=np.zeros((0, env.nv), dtype=np.float64),
                info_updates={},
                randomization=None,
            )

        reset_cfg = env.cfg.reset
        goal_cfg = env.cfg.goal

        qpos = np.zeros((num_reset, env.nq), dtype=np.float64)
        qvel = np.zeros((num_reset, env.nv), dtype=np.float64)

        # ──────────────────────────────────────────────────────────────────────
        # Robot DOF state: default pose + noise (reset_utils.py:200-221)
        # ──────────────────────────────────────────────────────────────────────
        default_pos = env._default_joint_pos_canon.astype(np.float32)
        lower_canon = env._joint_lower_canon
        upper_canon = env._joint_upper_canon

        reset_scale = np.zeros(NUM_JOINTS, dtype=np.float32)
        reset_scale[:7] = float(reset_cfg.reset_dof_pos_random_interval_arm)
        reset_scale[7:] = float(reset_cfg.reset_dof_pos_random_interval_fingers)

        sampled_pos = lower_canon + (upper_canon - lower_canon) * np.random.rand(
            num_reset, NUM_JOINTS
        ).astype(np.float32)
        joint_pos_canon = (
            default_pos[None, :] * (1.0 - reset_scale[None, :]) + sampled_pos * reset_scale[None, :]
        )
        joint_pos_canon = np.clip(joint_pos_canon, lower_canon[None, :], upper_canon[None, :])

        joint_vel_canon = np.random.uniform(
            -float(reset_cfg.reset_dof_vel_random_interval),
            float(reset_cfg.reset_dof_vel_random_interval),
            (num_reset, NUM_JOINTS),
        ).astype(np.float32)

        qpos[:, env._dof_pos_idx_canon] = joint_pos_canon.astype(np.float64)
        qvel[:, env._dof_vel_idx_canon] = joint_vel_canon.astype(np.float64)

        # ──────────────────────────────────────────────────────────────────────
        # Table z height: randomized (reset_utils.py:223-270)
        # ──────────────────────────────────────────────────────────────────────
        table_z = float(reset_cfg.table_reset_z) + np.random.uniform(
            -float(reset_cfg.table_reset_z_range),
            float(reset_cfg.table_reset_z_range),
            num_reset,
        ).astype(np.float32)

        # ──────────────────────────────────────────────────────────────────────
        # Object pose: random xy + table z offset + random orientation
        # (reset_utils.py:271-302)
        # ──────────────────────────────────────────────────────────────────────

        if reset_cfg.fixed_start_pose is not None:
            fixed = np.asarray(reset_cfg.fixed_start_pose, dtype=np.float32)
            obj_pos = np.tile(fixed[:3], (num_reset, 1))
            obj_quat = np.tile(fixed[3:], (num_reset, 1))
        else:
            noise = np.random.uniform(-1.0, 1.0, (num_reset, 3)).astype(np.float32)
            obj_pos = np.stack(
                [
                    noise[:, 0] * float(reset_cfg.reset_position_noise_x),
                    noise[:, 1] * float(reset_cfg.reset_position_noise_y),
                    table_z
                    + float(reset_cfg.table_object_z_offset)
                    + noise[:, 2] * float(reset_cfg.reset_position_noise_z),
                ],
                axis=-1,
            )
            obj_quat = np_random_orientation(num_reset)

        qpos[:, env._obj_pos_slice] = obj_pos.astype(np.float64)
        qpos[:, env._obj_quat_slice] = obj_quat.astype(np.float64)

        # Object init z for lifted-reward reference (reset_utils.py:301).
        object_init_z = obj_pos[:, 2].copy()

        # ──────────────────────────────────────────────────────────────────────
        # First goal: absolute mode (reset_utils.py:391)
        # ──────────────────────────────────────────────────────────────────────
        goal_pos, goal_quat = sample_absolute_goal(
            mins=goal_cfg.mins,
            maxs=goal_cfg.maxs,
            scale=float(goal_cfg.target_volume_region_scale),
            n=num_reset,
        )

        # ──────────────────────────────────────────────────────────────────────
        # Build info_updates with all tracker keys (contract §2)
        # ──────────────────────────────────────────────────────────────────────
        info_updates = self._build_info_updates_full(
            env=env,
            num_reset=num_reset,
            joint_pos_backend=joint_pos_canon[:, env._perm_canon_to_backend],
            object_init_z=object_init_z,
            object_pos=obj_pos,
            object_quat=obj_quat,
            goal_pos=goal_pos,
            goal_quat=goal_quat,
            env_ids=env_ids,
        )

        # ──────────────────────────────────────────────────────────────────────
        # Clear wrench cache + flush delay queues (reset_utils.py:397-401)
        # ──────────────────────────────────────────────────────────────────────
        env._object_forces[env_ids] = 0.0
        env._object_torques[env_ids] = 0.0
        env._action_queue[env_ids] = 0.0
        env._obs_queue[env_ids] = 0.0
        env._object_state_queue[env_ids] = 0.0

        # ──────────────────────────────────────────────────────────────────────
        # Per-reset DR resample (reset_utils.py:408-418)
        # ──────────────────────────────────────────────────────────────────────
        # The source redraws the wrench trigger probabilities and the object-extent
        # noise on **every** reset, not once at init. T7's __init__ only seeds them.
        dr = env.cfg.domain_randomization
        env._random_force_prob[env_ids] = sample_log_uniform(
            dr.force_prob_range[0], dr.force_prob_range[1], num_reset
        )
        env._random_torque_prob[env_ids] = sample_log_uniform(
            dr.torque_prob_range[0], dr.torque_prob_range[1], num_reset
        )
        scale_lo, scale_hi = dr.object_scale_noise_multiplier_range
        env._object_scale_multiplier[env_ids] = np.random.uniform(
            scale_lo, scale_hi, (num_reset, 3)
        ).astype(get_global_dtype())

        # The lifted latch also drives the next step's wrench gate, and
        # apply_action reads the cache before update_state refreshes it. Clearing
        # it here keeps a freshly reset env from inheriting the old episode's
        # latch (source equivalent: reset_utils.py:396 clears _lifted_object).
        env._state_cache_lifted_object[env_ids] = False

        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=None,
        )

    def _build_info_updates(self, env: Any, num_reset: int, *, object_z: float) -> dict[str, Any]:
        """Seed every ``state.info`` key from interface contract §2 (T0 minimal version).

        Args:
            env: Owning env instance.
            num_reset: Number of environments being reset.
            object_z: Nominal object spawn height, used for ``object_init_z``.

        Returns:
            Mapping of info key to freshly allocated per-env values.
        """
        dtype = get_global_dtype()
        n = num_reset

        # prev_targets seeds from the default pose, not zeros (contract §2.1).
        default_backend = np.broadcast_to(env._default_joint_pos_backend, (n, NUM_JOINTS)).astype(
            dtype
        )

        object_scales = np.broadcast_to(env.resolve_object_scale(), (n, 3)).astype(dtype)

        return {
            # §2.1 action / control
            "prev_targets": default_backend.copy(),
            "cur_targets": default_backend.copy(),
            "last_actions": np.zeros((n, NUM_JOINTS), dtype=dtype),
            "current_actions": np.zeros((n, NUM_JOINTS), dtype=dtype),
            # §2.2 goal / episode
            "goal_pos": np.zeros((n, 3), dtype=dtype),
            "goal_quat": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=dtype), (n, 1)),
            "successes": np.zeros((n,), dtype=np.int32),
            "near_goal_steps": np.zeros((n,), dtype=np.int32),
            "object_init_z": np.full((n,), object_z, dtype=dtype),
            "lifted_object": np.zeros((n,), dtype=bool),
            "prev_episode_successes": np.zeros((n,), dtype=np.int32),
            # §2.3 d* progress, sentinel = -1
            "closest_keypoint_max_dist": np.full((n,), DSTAR_SENTINEL, dtype=dtype),
            "closest_fingertip_dist": np.full((n, NUM_FINGERTIPS), DSTAR_SENTINEL, dtype=dtype),
            # §2.4 observation / physics caches
            "prev_object_pos": np.tile(np.asarray([0.0, 0.0, object_z], dtype=dtype), (n, 1)),
            "prev_object_quat": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=dtype), (n, 1)),
            "object_scales": object_scales,
            "reward": np.zeros((n,), dtype=dtype),
            # Reward-term logging dict. Not per-env, so the base class copies the
            # reference verbatim (np_env.py:294-295) and T3 overwrites it wholesale.
            "log": {},
        }

    def _build_info_updates_full(
        self,
        env: Any,
        num_reset: int,
        joint_pos_backend: np.ndarray,
        object_init_z: np.ndarray,
        object_pos: np.ndarray,
        object_quat: np.ndarray,
        goal_pos: np.ndarray,
        goal_quat: np.ndarray,
        env_ids: np.ndarray,
    ) -> dict[str, Any]:
        """Seed all ``state.info`` keys with full reset randomization (T4).

        Args:
            env: Owning env instance.
            num_reset: Number of environments being reset.
            joint_pos_backend: Sampled robot joint positions, backend order.
            object_init_z: Object z coordinates for lifted-reward reference.
            object_pos: Sampled object positions.
            object_quat: Sampled object orientations (wxyz).
            goal_pos: Sampled goal positions.
            goal_quat: Sampled goal orientations (wxyz).
            env_ids: Indices being reset, for prev_episode_successes transfer.

        Returns:
            Mapping of info key to freshly allocated per-env values.
        """
        dtype = get_global_dtype()
        n = num_reset

        object_scales = np.broadcast_to(env.resolve_object_scale(), (n, 3)).astype(dtype)

        # Transfer prev_episode_successes from current successes (reset_utils.py:393).
        # Guard: check both state existence AND key existence (first reset has no "successes" yet).
        prev_episode_successes = np.zeros((n,), dtype=np.int32)
        state = getattr(env, "_state", None)
        if state is not None and "successes" in state.info:
            prev_episode_successes[:] = state.info["successes"][env_ids]

        return {
            # §2.1 action / control — seed from noisy joint pose (reset_utils.py:219-220)
            "prev_targets": joint_pos_backend.astype(dtype),
            "cur_targets": joint_pos_backend.astype(dtype).copy(),
            "last_actions": np.zeros((n, NUM_JOINTS), dtype=dtype),
            "current_actions": np.zeros((n, NUM_JOINTS), dtype=dtype),
            # §2.2 goal / episode
            "goal_pos": goal_pos.astype(dtype),
            "goal_quat": goal_quat.astype(dtype),
            "successes": np.zeros((n,), dtype=np.int32),
            "near_goal_steps": np.zeros((n,), dtype=np.int32),
            "object_init_z": object_init_z.astype(dtype),
            "lifted_object": np.zeros((n,), dtype=bool),
            "prev_episode_successes": prev_episode_successes,
            # §2.3 d* progress, sentinel = -1 (reset_utils.py:373-375,395)
            "closest_keypoint_max_dist": np.full((n,), DSTAR_SENTINEL, dtype=dtype),
            "closest_fingertip_dist": np.full((n, NUM_FINGERTIPS), DSTAR_SENTINEL, dtype=dtype),
            # §2.4 observation / physics caches
            "prev_object_pos": object_pos.astype(dtype),
            "prev_object_quat": object_quat.astype(dtype),
            "object_scales": object_scales,
            "reward": np.zeros((n,), dtype=dtype),
            "log": {},
        }

    def build_reset_observation(
        self, env: Any, env_ids: np.ndarray, info_updates: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        """Return post-reset observations. **T0 stub: zeros.**

        Real reset observations need the T2 observation builder; until then this
        returns correctly shaped zeros so the base class can scatter them.

        Args:
            env: Owning env instance.
            env_ids: Indices being reset, used only for the batch size.
            info_updates: Values from :meth:`build_reset_plan`. Unused here.

        Returns:
            Mapping of observation group name to a zero array of shape
            ``(len(env_ids), width)``.
        """
        del info_updates
        dtype = get_global_dtype()
        n = int(len(env_ids))
        return {
            name: np.zeros((n, width), dtype=dtype) for name, width in env.obs_groups_spec.items()
        }

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        """Return ``None``: wrench DR does not use the interval plan.

        ``IntervalRandomizationPlan`` carries only ``body_force``, with no torque
        channel, so the force+torque impulses go through a direct backend call
        inside ``apply_action`` instead (decisions D5/D6). That is T1/T6 work.

        Args:
            env: Owning env instance.
            step_counter: Global step counter.

        Returns:
            ``None``.
        """
        del env, step_counter
        return None


__all__ = ["SimToolRealDRProvider", "DSTAR_SENTINEL"]
