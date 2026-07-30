"""Episode lifecycle: success detection, goal advance, termination (T4).

Ported from SimToolReal termination_utils.py and the relevant sections of
obs_utils.py. Implements the "success-doesn't-end-episode" mechanism (D2):
successful envs sample a new goal, reset their d* trackers and episode length,
but do not enter the ``terminated`` state.

The tolerance curriculum (termination_utils.py:10-36) is also here, called
before success detection in ``update_state``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .goal_sampling import sample_delta_goal

if TYPE_CHECKING:
    from unilab.base.np_env import NpEnvState

    from .env import SimToolRealEnv


def _require_state(env: SimToolRealEnv) -> NpEnvState:
    """Return ``env._state``, asserting it has been initialized.

    ``NpEnv._state`` is ``Optional``, so every read needs narrowing. The base
    class uses the same ``assert`` idiom (np_env.py:178,248). ``update_state``
    only ever runs after ``init_state``, so this cannot fire in the step loop.

    Args:
        env: Owning env instance.

    Returns:
        The live :class:`~unilab.base.np_env.NpEnvState`.
    """
    state = env._state
    assert state is not None, "episode lifecycle called before init_state()"
    return state


def update_tolerance_curriculum(env: SimToolRealEnv) -> None:
    """Shrink success tolerance when completed episodes average enough goals.

    Called from ``update_state`` before computing success. Tracks a global frame
    counter (not episode count) and checks every ``tolerance_curriculum_interval``
    frames whether the mean of ``prev_episode_successes`` (successes from completed
    episodes only, transferred at reset) meets the threshold. If so, multiplies
    the current tolerance by ``tolerance_curriculum_increment`` (0.9), floored at
    ``target_success_tolerance`` (0.01).

    Source: termination_utils.py:10-36.

    Args:
        env: Owning env instance. Mutates ``env._current_success_tolerance``,
            ``env._frame_counter``, and ``env._last_curriculum_update``.
    """
    if not hasattr(env, "_frame_counter"):
        env._frame_counter = 0
        env._last_curriculum_update = 0
        env._current_success_tolerance = float(env._cfg.goal.success_tolerance)

    env._frame_counter += 1
    term_cfg = env._cfg.termination

    if env._frame_counter - env._last_curriculum_update >= term_cfg.tolerance_curriculum_interval:
        prev_successes = _require_state(env).info["prev_episode_successes"].astype(np.float32)
        threshold = float(term_cfg.tolerance_curriculum_success_threshold)

        if prev_successes.size > 0 and prev_successes.mean() >= threshold:
            new_tol = env._current_success_tolerance * float(
                term_cfg.tolerance_curriculum_increment
            )
            new_tol = max(
                min(new_tol, float(env._cfg.goal.success_tolerance)),
                float(env._cfg.goal.target_success_tolerance),
            )
            env._current_success_tolerance = new_tol
            env._last_curriculum_update = env._frame_counter

    # Eval mode pins the tolerance (termination_utils.py:34-36).
    # eval_success_tolerance is on GoalCfg (contract §5.0 regrouping).
    if env._cfg.goal.eval_success_tolerance is not None:
        env._current_success_tolerance = float(env._cfg.goal.eval_success_tolerance)


def compute_success(env: SimToolRealEnv, keypoints_max_dist: np.ndarray) -> np.ndarray:
    """Determine which envs have reached the goal.

    Success = keypoint distance ≤ threshold for ``success_steps`` consecutive (or
    cumulative) steps. The threshold is ``current_success_tolerance * keypoint_scale``
    (obs_utils.py:195), **not** the config tolerance alone — so 0.075 → 0.1125 m.

    Source: obs_utils.py:192-202.

    Args:
        env: Owning env. Reads and mutates ``state.info["near_goal_steps"]``.
        keypoints_max_dist: Max keypoint distance per env, shape ``(N,)``.

    **Side effect**: publishes ``env._near_goal`` and ``env._is_success``, which
    the reward's ``reach_goal_bonus`` reads (rewards.py:320-321). The source sets
    both as env attributes at the same point (obs_utils.py:196,202); without the
    publish the reach bonus would read the stale zeros allocated at init and stay
    permanently 0.

    Returns:
        Boolean mask of shape ``(N,)`` indicating success.
    """
    term_cfg = env._cfg.termination
    goal_cfg = env._cfg.goal
    info = _require_state(env).info

    # ★ Multiply by keypoint_scale (easy-to-miss item, guide §6).
    # keypoint_scale is on GoalCfg, not RewardCfg (contract §5.0 regrouping).
    tol = env._current_success_tolerance * float(goal_cfg.keypoint_scale)
    near_goal = keypoints_max_dist <= tol

    # Read/write from state.info, not env._near_goal_steps instance attribute.
    ng_steps = info["near_goal_steps"]

    # Update near_goal_steps: cumulative or consecutive.
    if term_cfg.force_consecutive_near_goal_steps:
        # Consecutive: any non-near step resets counter to 0.
        ng_steps[:] = (ng_steps + near_goal.astype(np.int32)) * near_goal.astype(np.int32)
    else:
        # Cumulative (default): count up when near, hold when far.
        ng_steps += near_goal.astype(np.int32)

    is_success = ng_steps >= goal_cfg.success_steps

    # Publish for the reward's reach bonus (obs_utils.py:196,202).
    env._near_goal = near_goal
    env._is_success = is_success
    return is_success


def advance_goal_on_success(env: SimToolRealEnv, is_success: np.ndarray) -> None:
    """Sample next goal for successful envs and reset their d*/length trackers.

    Implements the D2 "success doesn't end episode" mechanism (termination_utils.py:46-51):
    successful envs increment ``successes``, sample a delta goal, reset d* to sentinel,
    clear ``near_goal_steps``, and **zero ``state.info["steps"]``** so the base class's
    ``steps += 1`` (np_env.py:205) yields 1, preventing timeout truncation on the same step.

    Source: termination_utils.py:46-51, reset_utils.py:372-381.

    Args:
        env: Owning env. Mutates ``state.info`` for successful envs.
        is_success: Boolean mask of shape ``(N,)``.
    """
    success_ids = np.flatnonzero(is_success)
    if success_ids.size == 0:
        return

    info = _require_state(env).info
    goal_cfg = env._cfg.goal

    # Increment successes (termination_utils.py:46).
    info["successes"][success_ids] += 1

    # Sample next goal in delta mode (reset_utils.py:378-381).
    prev_pos = info["goal_pos"][success_ids]
    prev_quat = info["goal_quat"][success_ids]

    new_pos, new_quat = sample_delta_goal(
        prev_pos=prev_pos,
        prev_quat=prev_quat,
        delta_distance=float(goal_cfg.delta_goal_distance),
        delta_rotation_degrees=float(goal_cfg.delta_rotation_degrees),
        mins=goal_cfg.mins,
        maxs=goal_cfg.maxs,
        scale=float(goal_cfg.target_volume_region_scale),
    )

    info["goal_pos"][success_ids] = new_pos
    info["goal_quat"][success_ids] = new_quat

    # Reset d* trackers to sentinel (reset_utils.py:373-375).
    from .dr_provider import DSTAR_SENTINEL

    info["closest_keypoint_max_dist"][success_ids] = DSTAR_SENTINEL
    # closest_fingertip_dist is (N, 5), fill all fingertips.
    info["closest_fingertip_dist"][success_ids, :] = DSTAR_SENTINEL

    # Clear near_goal_steps (reset_utils.py:375).
    info["near_goal_steps"][success_ids] = 0

    # ★ Zero steps so base class's steps+=1 yields 1, not 600 (termination_utils.py:51).
    # This prevents timeout truncation from firing on the success step.
    info["steps"][success_ids] = 0


def compute_terminations(
    env: SimToolRealEnv, is_success: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute terminated and truncated masks (termination_utils.py:39-72).

    Termination causes (none include success per D2):
    - object z < 0.1 (drop)
    - any fingertip distance > 1.5 m (hand_far)
    - successes >= max_consecutive_successes (if > 0)

    Truncation is handled by the base class (steps >= max_episode_length).

    Args:
        env: Owning env instance.
        is_success: Success mask for this step (used to check max_consecutive_successes).

    Returns:
        terminated: Boolean mask of shape ``(N,)``.
        truncated: Boolean mask of shape ``(N,)`` (currently all False; base handles it).
    """
    term_cfg = env._cfg.termination
    n = env._num_envs

    # Object z < 0.1 (termination_utils.py:54-55).
    object_z = env.get_object_pos()[:, 2]
    fall = object_z < 0.1

    # Any fingertip distance > 1.5 m (termination_utils.py:62-63).
    # _curr_fingertip_distances is computed in update_state (T3/T5 responsibility).
    # For T4 we'll check it exists; if not, skip this termination (graceful degradation).
    if hasattr(env, "_curr_fingertip_distances"):
        hand_far = env._curr_fingertip_distances.max(axis=-1) > 1.5
    else:
        hand_far = np.zeros(n, dtype=bool)

    # Max consecutive successes (termination_utils.py:57-60).
    if term_cfg.max_consecutive_successes > 0:
        successes = _require_state(env).info["successes"]
        max_successes_reached = successes >= term_cfg.max_consecutive_successes
    else:
        max_successes_reached = np.zeros(n, dtype=bool)

    terminated = fall | hand_far | max_successes_reached

    # Truncation is handled by base class (np_env.py:207).
    truncated = np.zeros(n, dtype=bool)

    return terminated, truncated


__all__ = [
    "update_tolerance_curriculum",
    "compute_success",
    "advance_goal_on_success",
    "compute_terminations",
]
