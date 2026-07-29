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
        """Build the reset state and seed the ``state.info`` bus.

        Writes the default robot pose (noise-free in T0; T4 adds the sampled
        distribution) and the nominal object pose above the table.

        Args:
            env: Owning :class:`~.env.SimToolRealEnv`.
            env_ids: Indices of environments being reset.

        Returns:
            A :class:`~unilab.dr.types.ResetPlan` whose ``randomization`` is
            ``None`` by design.
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
        object_z = float(reset_cfg.table_reset_z) + float(reset_cfg.table_object_z_offset)

        qpos = np.zeros((num_reset, env.nq), dtype=np.float64)
        qpos[:, env._dof_pos_idx_canon] = env._default_joint_pos_canon.astype(np.float64)
        qpos[:, env._obj_pos_slice] = np.asarray([0.0, 0.0, object_z], dtype=np.float64)
        qpos[:, env._obj_quat_slice] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        qvel = np.zeros((num_reset, env.nv), dtype=np.float64)

        info_updates = self._build_info_updates(env, num_reset, object_z=object_z)

        # Match the source by clearing cached external object wrenches on reset
        # (reset_utils.py:400) and flushing the delay queues (:397-399).
        env._object_forces[env_ids] = 0.0
        env._object_torques[env_ids] = 0.0
        env._action_queue[env_ids] = 0.0
        env._obs_queue[env_ids] = 0.0
        env._object_state_queue[env_ids] = 0.0

        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=None,
        )

    def _build_info_updates(self, env: Any, num_reset: int, *, object_z: float) -> dict[str, Any]:
        """Seed every ``state.info`` key from interface contract §2.

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
