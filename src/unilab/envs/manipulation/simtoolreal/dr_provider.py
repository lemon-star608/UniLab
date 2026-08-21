"""Task-owned reset and domain-randomization provider for SimToolReal.

The provider owns cold-path compiled-model assignment and hot-path state reset.
It samples robot/object poses, the first absolute goal, and reset trackers, then
builds the real post-reset observation for the affected rows. Reset plans do not
carry physics-parameter randomization: the source reset mutates poses and
trackers only, while interval wrench/noise behavior remains in its dedicated
owners.
"""

from __future__ import annotations

import json
from pathlib import Path
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


def _sample_yaw_quaternions(n: int) -> np.ndarray:
    """Sample full-range yaw rotations as MuJoCo-order ``wxyz`` quaternions."""
    yaw = np.random.uniform(-np.pi, np.pi, n).astype(np.float32)
    half_yaw = 0.5 * yaw
    quaternions = np.zeros((n, 4), dtype=np.float32)
    quaternions[:, 0] = np.cos(half_yaw)
    quaternions[:, 3] = np.sin(half_yaw)
    return quaternions


class SimToolRealDRProvider(DomainRandomizationProvider):
    """Reset and compiled-model assignment owner for SimToolReal."""

    @staticmethod
    def _ensure_fixed_trajectory_cache(env: Any) -> None:
        """Load the optional trajectory pool once on the cold/cache path."""
        if not env.cfg.reset.fixed_trajectory_file or hasattr(env, "_fixed_traj_pos"):
            return
        path = Path(env.cfg.reset.fixed_trajectory_file)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        all_pos = np.asarray(payload["pos"], dtype=np.float32)
        all_quat = np.asarray(payload["quat_wxyz"], dtype=np.float32)
        if all_pos.ndim != 3 or all_pos.shape[-1] != 3:
            raise ValueError(f"invalid trajectory position shape in {path}: {all_pos.shape}")
        if all_quat.shape != (*all_pos.shape[:2], 4):
            raise ValueError(f"invalid trajectory quaternion shape in {path}: {all_quat.shape}")
        n_take = int(env.cfg.reset.fixed_trajectory_count) or all_pos.shape[0]
        if n_take > all_pos.shape[0]:
            raise ValueError(
                f"fixed_trajectory_count={n_take} exceeds pool size {all_pos.shape[0]} in {path}"
            )
        env._fixed_traj_pos = all_pos[:n_take].copy()
        env._fixed_traj_quat = all_quat[:n_take].copy()
        env._traj_id = np.zeros((env._num_envs,), dtype=np.int64)
        env._traj_step = np.zeros((env._num_envs,), dtype=np.int64)

    @staticmethod
    def _require_fixed_trajectory_cache(env: Any) -> None:
        required = ("_fixed_traj_pos", "_fixed_traj_quat", "_traj_id", "_traj_step")
        if not all(name in vars(env) for name in required):
            raise RuntimeError(
                "fixed trajectory cache must be initialized before reset; "
                "call the init randomization hook first"
            )

    def build_init_randomization_plan(self, env: Any):
        """Build the cold-path compiled source-model assignment plan.

        Delegates to ``env.build_init_randomization_plan()``. Disabled pools
        select one real compiled source tool; no placeholder path exists.

        Args:
            env: Owning :class:`~.env.SimToolRealEnv`.

        Returns:
            :class:`~unilab.dr.types.InitRandomizationPlan`.
        """
        self._ensure_fixed_trajectory_cache(env)
        return env.build_init_randomization_plan()

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        """Check that the backend can service resets.

        Declares no reset randomization terms on purpose (decision D5): this port
        has no per-reset physics-parameter DR, so there is nothing to negotiate
        against ``capabilities``.

        Args:
            env: Owning env instance.
            capabilities: Backend-declared DR capabilities. Unused — see above.

        The reset manager validates and invokes the public ``SimBackend.set_state``
        contract. This task provider does not inspect backend capabilities.
        """
        del env, capabilities

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        """Build reset state and trackers for the configured owner pose mode.

        Samples robot joint state, object pose, and the first absolute goal.
        Tolerance curriculum updates remain in the episode-lifecycle owner.

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
        # Object pose: random xy + table z offset + random orientation
        # (reset_utils.py:271-302)
        # ──────────────────────────────────────────────────────────────────────

        if reset_cfg.fixed_start_pose is not None:
            fixed = np.asarray(reset_cfg.fixed_start_pose, dtype=np.float32)
            obj_pos = np.tile(fixed[:3], (num_reset, 1))
            obj_quat = np.tile(fixed[3:], (num_reset, 1))
        elif reset_cfg.object_pose_mode == "horizontal_near_table":
            xy = np.random.uniform(-1.0, 1.0, (num_reset, 2)).astype(np.float32)
            obj_pos = np.empty((num_reset, 3), dtype=np.float32)
            obj_pos[:, 0] = xy[:, 0] * float(reset_cfg.reset_position_noise_x)
            obj_pos[:, 1] = xy[:, 1] * float(reset_cfg.reset_position_noise_y)
            obj_pos[:, 2] = float(reset_cfg.horizontal_near_table_z)
            obj_quat = _sample_yaw_quaternions(num_reset)
        else:
            # Source-compatible object sampler. The sampled z reference affects
            # only object spawn; the MuJoCo table itself remains static.
            object_spawn_z_reference = float(reset_cfg.table_reset_z) + np.random.uniform(
                -float(reset_cfg.object_spawn_z_reference_range),
                float(reset_cfg.object_spawn_z_reference_range),
                num_reset,
            ).astype(np.float32)
            noise = np.random.uniform(-1.0, 1.0, (num_reset, 3)).astype(np.float32)
            obj_pos = np.stack(
                [
                    noise[:, 0] * float(reset_cfg.reset_position_noise_x),
                    noise[:, 1] * float(reset_cfg.reset_position_noise_y),
                    object_spawn_z_reference
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

        # Source priority: fixed pose > fixed trajectory > absolute first goal.
        if reset_cfg.fixed_goal_pose is not None:
            fixed = np.asarray(reset_cfg.fixed_goal_pose, dtype=np.float32)
            goal_pos = np.tile(fixed[:3], (num_reset, 1))
            goal_quat = np.tile(fixed[3:], (num_reset, 1))
        elif reset_cfg.fixed_trajectory_file:
            self._require_fixed_trajectory_cache(env)
            n_traj = env._fixed_traj_pos.shape[0]
            env._traj_id[env_ids] = np.random.randint(0, n_traj, size=num_reset)
            env._traj_step[env_ids] = 0
            traj_id = env._traj_id[env_ids]
            goal_pos = env._fixed_traj_pos[traj_id, 0]
            goal_quat = env._fixed_traj_quat[traj_id, 0]
        else:
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
        # noise on **every** reset, not once at init. construction only seeds them.
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
        """Seed all ``state.info`` keys with the sampled reset state.

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

        object_scales_raw = env.resolve_object_scale()
        # Pool mode stores per-env scales; single-tool mode stores one scale.
        if object_scales_raw.ndim == 2:
            # Per-env: index by env_ids
            object_scales = object_scales_raw[env_ids].astype(dtype)
        else:
            object_scales = np.broadcast_to(object_scales_raw, (n, 3)).astype(dtype)

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
        """Return the real post-reset observations for ``env_ids``.

        Delegates to :func:`~.observations.build_reset_observations`, the
        row-restricted sibling of the ordinary observation builder. The base class scatters the
        result into ``state.obs`` for exactly these rows (np_env.py:282-284), so
        it is the first observation of the new episode that the policy sees —
        which is why it must be the real thing and not zeros.

        Two ordering facts make this correct at this point in the sequence:

        * ``backend.set_state`` has already run (dr/manager.py:52 before :70), so
          the backend is in the post-reset state.
        * ``state.info`` has **not** been updated yet (np_env.py:287-297 runs
          after this call), so the fresh trackers are read out of
          ``info_updates`` instead. Reading ``state.info`` here would produce
          stale-but-plausible observations, which are harder to detect than zeros.

        ``steps`` is the one field the reset plan does not carry: a fresh episode
        is at step 0 by definition, so it is supplied as zeros. That also makes
        the critic's ``progress`` feature ``log(0/10+1) = 0``.

        Args:
            env: Owning :class:`~.env.SimToolRealEnv`.
            env_ids: Indices being reset, shape ``(k,)``.
            info_updates: Values from :meth:`build_reset_plan`, already sized
                ``(k, ...)`` for these rows.

        Returns:
            Mapping of observation group name to an array of shape
            ``(len(env_ids), width)``.
        """
        from .observations import build_reset_observations

        ids = np.asarray(env_ids, dtype=np.intp)
        n = int(ids.shape[0])
        if n == 0:
            dtype = get_global_dtype()
            return {
                name: np.zeros((0, width), dtype=dtype)
                for name, width in env.obs_groups_spec.items()
            }

        reset_info = dict(info_updates)
        # Fresh episode: step 0 (np_env.py:260 zeroes it for these rows too).
        reset_info["steps"] = np.zeros((n,), dtype=np.uint32)
        return build_reset_observations(env, ids, reset_info)

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        """Return ``None``: wrench DR does not use the interval plan.

        ``IntervalRandomizationPlan`` carries only ``body_force``, with no torque
        channel, so the force+torque impulses go through a direct backend call
        inside ``apply_action`` instead (decisions D5/D6).

        Args:
            env: Owning env instance.
            step_counter: Global step counter.

        Returns:
            ``None``.
        """
        del env, step_counter
        return None


__all__ = ["SimToolRealDRProvider", "DSTAR_SENTINEL"]
