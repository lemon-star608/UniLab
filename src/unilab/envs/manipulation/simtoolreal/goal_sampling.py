"""Goal-pose samplers for SimToolReal (T4).

Ported from SimToolReal/utils/goal_sampling.py. Two sampling modes:

* **absolute**: uniformly sample position and orientation inside the workspace.
* **delta**: perturb the previous goal by a bounded random walk.

The contract signature (MIGRATION_01 §4.6) freezes the parameter order and
names to enable parallel work. All arrays are numpy, dtype float32 on CPU.
"""

from __future__ import annotations

import numpy as np

from unilab.utils.rotation import np_quat_mul


def np_random_orientation(n: int) -> np.ndarray:
    """Sample n uniformly random unit quaternions (wxyz).

    Uses subgroup algorithm from Shoemake, K., "Uniform random rotations",
    Graphics Gems III, 1992.
    """
    u = np.random.rand(n, 3).astype(np.float32)
    q = np.empty((n, 4), dtype=np.float32)

    sqrt1_u1 = np.sqrt(1.0 - u[:, 0])
    sqrtu1 = np.sqrt(u[:, 0])
    two_pi_u2 = 2.0 * np.pi * u[:, 1]
    two_pi_u3 = 2.0 * np.pi * u[:, 2]

    q[:, 0] = sqrtu1 * np.cos(two_pi_u3)  # w
    q[:, 1] = sqrt1_u1 * np.sin(two_pi_u2)  # x
    q[:, 2] = sqrt1_u1 * np.cos(two_pi_u2)  # y
    q[:, 3] = sqrtu1 * np.sin(two_pi_u3)  # z

    return q


def np_quat_from_angle_axis(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Convert angle-axis representation to unit quaternion (wxyz).

    Args:
        angle: Rotation angles in radians, shape (n,).
        axis: Rotation axes (assumed normalized), shape (n, 3).

    Returns:
        Unit quaternions (wxyz), shape (n, 4).
    """
    half_angle = angle * 0.5
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    q = np.empty((angle.shape[0], 4), dtype=np.float32)
    q[:, 0] = cos_half  # w
    q[:, 1] = axis[:, 0] * sin_half  # x
    q[:, 2] = axis[:, 1] * sin_half  # y
    q[:, 3] = axis[:, 2] * sin_half  # z

    return q


def _scale_workspace_bounds(
    mins: np.ndarray, maxs: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scale workspace bounds about their center."""
    center = 0.5 * (mins + maxs)
    half = 0.5 * (maxs - mins) * scale
    return center - half, center + half


def sample_absolute_goal(
    mins: tuple[float, float, float] | np.ndarray,
    maxs: tuple[float, float, float] | np.ndarray,
    scale: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample position and orientation inside the workspace.

    Args:
        mins: Workspace lower bounds (x, y, z).
        maxs: Workspace upper bounds (x, y, z).
        scale: Scale factor applied about the workspace center.
        n: Number of goals to sample.

    Returns:
        pos: Sampled positions, shape ``(n, 3)``.
        quat: Sampled orientations (wxyz), shape ``(n, 4)``.
    """
    mins_arr = np.asarray(mins, dtype=np.float32)
    maxs_arr = np.asarray(maxs, dtype=np.float32)
    lo, hi = _scale_workspace_bounds(mins_arr, maxs_arr, scale)
    pos = lo + (hi - lo) * np.random.rand(n, 3).astype(np.float32)
    quat = np_random_orientation(n)
    return pos, quat


def sample_delta_goal(
    prev_pos: np.ndarray,
    prev_quat: np.ndarray,
    delta_distance: float,
    delta_rotation_degrees: float,
    mins: tuple[float, float, float] | np.ndarray,
    maxs: tuple[float, float, float] | np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb the previous goal by a bounded random walk.

    Args:
        prev_pos: Previous goal position, shape ``(n, 3)``.
        prev_quat: Previous goal orientation (wxyz), shape ``(n, 4)``.
        delta_distance: Maximum position perturbation (meters).
        delta_rotation_degrees: Maximum rotation perturbation (degrees).
        mins: Workspace lower bounds (x, y, z).
        maxs: Workspace upper bounds (x, y, z).
        scale: Scale factor applied about the workspace center.

    Returns:
        pos: New goal positions clamped to workspace, shape ``(n, 3)``.
        quat: New goal orientations (wxyz), shape ``(n, 4)``.
    """
    n = prev_pos.shape[0]
    mins_arr = np.asarray(mins, dtype=np.float32)
    maxs_arr = np.asarray(maxs, dtype=np.float32)
    lo, hi = _scale_workspace_bounds(mins_arr, maxs_arr, scale)

    # Position: uniform noise in [-delta_distance, +delta_distance] per axis.
    pos_noise = (np.random.rand(n, 3).astype(np.float32) * 2.0 - 1.0) * delta_distance
    new_pos = np.clip(prev_pos + pos_noise, lo, hi)

    # Orientation: angle-axis rotation. Axis uniformly on sphere, angle uniformly
    # in [-delta_rotation_degrees, +delta_rotation_degrees].
    axis = np.random.randn(n, 3).astype(np.float32)
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    angle = (
        (np.random.rand(n).astype(np.float32) * 2.0 - 1.0)
        * delta_rotation_degrees
        * (np.pi / 180.0)
    )
    dq = np_quat_from_angle_axis(angle, axis)
    new_quat = np_quat_mul(dq, prev_quat)
    return new_pos, new_quat


__all__ = ["sample_absolute_goal", "sample_delta_goal"]
