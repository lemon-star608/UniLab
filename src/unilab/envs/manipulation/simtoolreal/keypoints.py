"""Keypoint geometry for SimToolReal.

Keypoints turn a 6D pose into four 3D points so a single distance metric encodes
both translation and rotation error. The distance used by the reward and the
success gate is the paper's Eq. 2::

    d(o, g) = max_i || obj_kp_i - goal_kp_i ||

The SimToolReal source has no standalone ``compute_keypoints``: the offsets are
precomputed once per env in ``reset_utils.allocate_state_buffers`` and rotated by
the inline ``obs_utils._keypoints_world`` helper. This module packages both
halves. Source locations, relative to the SimToolReal repo root
(``isaacsimenvs/tasks/simtoolreal/``):

    KEYPOINT_CORNERS            utils/obs_utils.py:28-33
    _keypoints_world            utils/obs_utils.py:103-112
    observation offsets (phi)   utils/reset_utils.py:74-77
    reward offsets (fixed)      utils/reset_utils.py:80-83
    per-object scale DR         utils/obs_utils.py:275
    keypoint_scale = 1.5        simtoolreal_env_cfg.py:337 (RewardCfg)
    object_base_size = 0.04     simtoolreal_env_cfg.py:338 (RewardCfg)
    fixed_size                  simtoolreal_env_cfg.py:339 (RewardCfg)
    d(o, g) = max_i ||.||       utils/obs_utils.py:180

**Two keypoint sizes, one formula.** Both paths are
``corners * size * 0.5 * keypoint_scale``; only ``size`` (in **metres**) differs:

* observation path (per object, carries the scale DR)::

      size = phi * cfg.reward_config.object_base_size * object_scale_multiplier

  ``phi`` is *dimensionless* — the handle bbox divided by ``object_base_size``
  (0.04 m), see migration guide §2 and ``env.resolve_object_scale``. Use
  :func:`observation_keypoint_size` rather than passing ``phi`` straight in;
  a raw ``phi`` would inflate every offset by 1 / 0.04 = 25x.

* reward / success path (fixed across the whole object pool)::

      size = cfg.reward_config.fixed_size

  The env caches that into ``env._keypoint_offsets_fixed`` (a ``(4, 3)``
  array), so the reward side should call
  :func:`compute_keypoints_from_offsets` with it instead of recomputing.

All arrays are numpy ``float32`` (MIGRATION_00 decision D0). Quaternions are
``wxyz`` throughout; the ``xyzw`` conversion belongs to the observation packing
step (decision D3), not here.
"""

from __future__ import annotations

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.utils.rotation import np_quat_apply

from .constants import KEYPOINT_CORNERS, NUM_KEYPOINTS

# cfg.goal.keypoint_scale. The contract regroups it onto GoalCfg; the source
# keeps it on RewardCfg (cfg:337). Callers should pass cfg.goal.keypoint_scale
# explicitly — this default exists so the frozen three-argument signature in
# interface contract §4.3 stays callable as-is.
DEFAULT_KEYPOINT_SCALE: float = 1.5

# Unit-cube corners, object frame, before scaling (obs_utils.py:28-33). Order is
# load-bearing: keypoint_max_dist pairs obj_kp[i] with goal_kp[i] elementwise.
_CORNERS: np.ndarray = np.asarray(KEYPOINT_CORNERS, dtype=np.float32)


def keypoint_offsets(
    size: np.ndarray,
    *,
    keypoint_scale: float = DEFAULT_KEYPOINT_SCALE,
) -> np.ndarray:
    """Return object-frame keypoint offsets for one or many object sizes.

    Ports the offset arithmetic shared by reset_utils.py:74-77 (per-object) and
    reset_utils.py:80-83 (fixed): ``corners * size * 0.5 * keypoint_scale``.

    Args:
        size: Object extent in **metres**, shape ``(3,)`` or ``(N, 3)``. This is
            a full extent, not a half-extent; the ``0.5`` in the formula turns it
            into one.
        keypoint_scale: ``cfg.goal.keypoint_scale`` (source cfg:337, 1.5).

    Returns:
        Offsets of shape ``(4, 3)`` for a ``(3,)`` size, or ``(N, 4, 3)`` for an
        ``(N, 3)`` size.

    Raises:
        ValueError: If ``size`` is not ``(3,)`` or ``(N, 3)``.
    """
    dtype = get_global_dtype()
    size_arr = np.asarray(size, dtype=dtype)
    corners = _CORNERS.astype(dtype, copy=False)

    scale = np.asarray(0.5 * keypoint_scale, dtype=dtype)

    if size_arr.shape == (3,):
        return np.ascontiguousarray(corners * (size_arr * scale), dtype=dtype)
    if size_arr.ndim == 2 and size_arr.shape[1] == 3:
        # (1, 4, 3) * (N, 1, 3) -> (N, 4, 3)
        return np.ascontiguousarray(
            corners[None, :, :] * (size_arr * scale)[:, None, :], dtype=dtype
        )
    raise ValueError(f"size must have shape (3,) or (N, 3), got {size_arr.shape}")


def compute_keypoints_from_offsets(
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Rotate and translate precomputed object-frame offsets into world frame.

    numpy port of ``obs_utils._keypoints_world`` (obs_utils.py:103-112). Use this
    when the offsets are already cached — notably ``env._keypoint_offsets_fixed``
    on the reward/success path, cached from reset_utils.py:80-83.

    Args:
        pos: Body positions, shape ``(N, 3)``.
        quat_wxyz: Body orientations, shape ``(N, 4)``, **wxyz** (decision D3).
        offsets: Object-frame offsets, shape ``(4, 3)`` (shared across envs) or
            ``(N, 4, 3)`` (per env).

    Returns:
        World-frame keypoints of shape ``(N, 4, 3)``.

    Raises:
        ValueError: If any input shape is inconsistent.
    """
    dtype = get_global_dtype()
    pos_arr = np.asarray(pos, dtype=dtype)
    quat_arr = np.asarray(quat_wxyz, dtype=dtype)
    offsets_arr = np.asarray(offsets, dtype=dtype)

    if pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
        raise ValueError(f"pos must have shape (N, 3), got {pos_arr.shape}")
    if quat_arr.shape != (pos_arr.shape[0], 4):
        raise ValueError(f"quat_wxyz must have shape {(pos_arr.shape[0], 4)}, got {quat_arr.shape}")

    n_envs = pos_arr.shape[0]
    if offsets_arr.ndim == 2:
        offsets_arr = np.broadcast_to(offsets_arr, (n_envs, *offsets_arr.shape))
    if offsets_arr.ndim != 3 or offsets_arr.shape[0] != n_envs or offsets_arr.shape[2] != 3:
        raise ValueError(
            f"offsets must have shape (K, 3) or ({n_envs}, K, 3), got {offsets_arr.shape}"
        )

    n_keypoints = offsets_arr.shape[1]

    # Flatten to (N*K, ...) so the batched quaternion helper applies one rotation
    # per offset, matching the source's expand+reshape (obs_utils.py:110-112).
    quat_flat = np.repeat(quat_arr, n_keypoints, axis=0)
    offsets_flat = np.ascontiguousarray(offsets_arr).reshape(-1, 3)

    rotated = np_quat_apply(quat_flat, offsets_flat).reshape(n_envs, n_keypoints, 3)
    keypoints = pos_arr[:, None, :] + rotated
    return np.ascontiguousarray(keypoints, dtype=dtype)


def compute_keypoints(
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
    size: np.ndarray,
    *,
    keypoint_scale: float = DEFAULT_KEYPOINT_SCALE,
) -> np.ndarray:
    """Return world-frame keypoints for a batch of 6D poses.

    Frozen signature, interface contract §4.3::

        kp_i = np_quat_apply(quat_wxyz, corners_i * size * 0.5 * keypoint_scale) + pos

    Works for both keypoint sizes — pass a per-object ``(N, 3)`` size on the
    observation path (see :func:`observation_keypoint_size`) or a shared ``(3,)``
    size such as ``cfg.reward_config.fixed_size`` on the reward path.

    Args:
        pos: Body positions, shape ``(N, 3)``. Same frame as the goal positions
            it will be compared against (env-origin-relative, per contract §2.2).
        quat_wxyz: Body orientations, shape ``(N, 4)``, **wxyz**.
        size: Object extent in **metres**, shape ``(3,)`` or ``(N, 3)``. Not
            ``phi`` — ``phi`` is dimensionless, see the module docstring.
        keypoint_scale: ``cfg.goal.keypoint_scale`` (source cfg:337, 1.5).

    Returns:
        World-frame keypoints of shape ``(N, 4, 3)``, float32.

    Raises:
        ValueError: If any input shape is inconsistent.
    """
    offsets = keypoint_offsets(size, keypoint_scale=keypoint_scale)
    return compute_keypoints_from_offsets(pos, quat_wxyz, offsets)


def keypoint_max_dist(obj_kp: np.ndarray, goal_kp: np.ndarray) -> np.ndarray:
    """Return the max per-keypoint Euclidean distance between two keypoint sets.

    The paper's Eq. 2 distance, ported from obs_utils.py:180
    (``torch.norm(obj_kp - goal_kp, dim=-1).max(dim=-1).values``). Feeds the
    keypoint reward, the ``d*`` progress tracker, and the success gate — the last
    of which compares against ``current_success_tolerance * keypoint_scale``
    (obs_utils.py:195), not the raw tolerance. The episode-lifecycle owner applies that gate.

    Args:
        obj_kp: Object keypoints, shape ``(N, 4, 3)``.
        goal_kp: Goal keypoints, shape ``(N, 4, 3)``, index-aligned with
            ``obj_kp`` — the corner order in :data:`KEYPOINT_CORNERS` is what
            makes the pairing meaningful.

    Returns:
        Distances of shape ``(N,)``, float32, non-negative.

    Raises:
        ValueError: If the two arrays disagree in shape or are not ``(N, K, 3)``.
    """
    dtype = get_global_dtype()
    obj_arr = np.asarray(obj_kp, dtype=dtype)
    goal_arr = np.asarray(goal_kp, dtype=dtype)

    if obj_arr.ndim != 3 or obj_arr.shape[2] != 3:
        raise ValueError(f"obj_kp must have shape (N, K, 3), got {obj_arr.shape}")
    if obj_arr.shape != goal_arr.shape:
        raise ValueError(
            f"obj_kp and goal_kp must have the same shape, got {obj_arr.shape} and {goal_arr.shape}"
        )

    per_keypoint = np.linalg.norm(obj_arr - goal_arr, axis=-1)
    return np.asarray(per_keypoint.max(axis=-1), dtype=dtype)


def observation_keypoint_size(
    phi: np.ndarray,
    object_base_size: float,
    scale_multiplier: np.ndarray | None = None,
) -> np.ndarray:
    """Convert dimensionless ``phi`` into the observation-path size in metres.

    The source never materializes this quantity: reset_utils.py:74-77 bakes
    ``phi * object_base_size * keypoint_scale * 0.5`` into ``_keypoint_offsets``,
    and obs_utils.py:275 then multiplies by the per-env ``object_scale_multiplier``
    DR factor. Splitting the metres-valued size out keeps
    :func:`compute_keypoints` size-agnostic and stops ``phi`` being mistaken for
    a length (it is 25x too large, since ``object_base_size`` is 0.04 m).

    Args:
        phi: Handle bbox normalized by ``object_base_size``, shape ``(3,)`` or
            ``(N, 3)``. This is ``env.resolve_object_scale()`` / the
            ``object_scales`` entry of ``state.info``.
        object_base_size: ``cfg.reward_config.object_base_size`` (source cfg:338, 0.04).
        scale_multiplier: Optional per-env ``object_scale_noise_multiplier_range``
            sample, shape ``(3,)`` or ``(N, 3)`` (obs_utils.py:275). The default
            DR range is ``(1.0, 1.0)``, i.e. a no-op.

    Returns:
        Object extent in metres, broadcast of the inputs' shapes.
    """
    dtype = get_global_dtype()
    size = np.asarray(phi, dtype=dtype) * np.asarray(object_base_size, dtype=dtype)
    if scale_multiplier is not None:
        size = size * np.asarray(scale_multiplier, dtype=dtype)
    return np.ascontiguousarray(size, dtype=dtype)


__all__ = [
    "DEFAULT_KEYPOINT_SCALE",
    "NUM_KEYPOINTS",
    "compute_keypoints",
    "compute_keypoints_from_offsets",
    "keypoint_max_dist",
    "keypoint_offsets",
    "observation_keypoint_size",
]
