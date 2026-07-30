"""SimToolReal action pipeline (migration task T1).

numpy port of ``isaacsimenvs/tasks/simtoolreal/utils/action_utils.py:18-75``
(``apply_action_pipeline``). torch -> numpy per MIGRATION_00 decision D0: all
arrays are ``float32`` on CPU, ``.clone()`` becomes ``.copy()``,
``torch.clamp`` becomes ``np.clip``, ``torch.roll`` becomes ``np.roll``.

Step order is load-bearing and matches the source line for line:

1. **canonical -> backend permute first** (:34). Everything downstream — limits,
   slices, queue frames, ``cur_targets`` — lives in backend joint order.
2. **action delay before the control law** (:36-48), not after. The delayed
   action is what the arm/hand laws see.
3. **arm: velocity-delta accumulator with two clamps** — once after the
   accumulation (:53), once after the EMA (:58).
4. **hand: absolute [-1,1] -> [lower,upper] mapping with one clamp** after the
   EMA (:69).
5. ``prev_targets = cur_targets.copy()`` (:74) so the two never alias.

Constants (contract §4.1, guide §4): ``dof_speed_scale=1.5``,
``arm_moving_average=0.1``, ``hand_moving_average=0.1``, ``dt=ctrl_dt=1/60``.
``EMA=0.1`` means ``0.1 * new + 0.9 * old`` — heavy smoothing, and the guide
flags it as sim-to-real critical.

The policy action is **not** clamped on input (guide §4: "无输入 clamp"), so the
hand mapping can leave the joint range and genuinely needs its clamp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from unilab.base.np_env import NpEnvState

    from .env import SimToolRealEnv


def _episode_start(state: NpEnvState) -> np.ndarray:
    """Return the per-env "brand new episode" mask used to flush delay queues.

    ``(steps == 0) & (successes == 0)`` — contract §4.2, ported from the
    source's ``(episode_length_buf == 0) & (_successes == 0)``
    (action_utils.py:38). The ``successes`` half is what keeps an intra-episode
    goal advance (decision D2) from being mistaken for a fresh episode.

    Args:
        state: Current env state; reads ``info["steps"]`` and
            ``info["successes"]``.

    Returns:
        Boolean array of shape ``(num_envs,)``.
    """
    steps = np.asarray(state.info["steps"], dtype=np.int64)
    successes = np.asarray(state.info["successes"], dtype=np.int64)
    return (steps == 0) & (successes == 0)


def _stub_push_and_sample(
    queue: np.ndarray,
    new_values: np.ndarray,
    env: SimToolRealEnv,
    flush: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stand-in for T6's ``delay_buffer.push_and_sample_delay`` (contract §4.2).

    Faithful port of the source's queue handling (action_utils.py:39-48), so the
    delay semantics are already correct before T6 lands: flush fills every slot,
    then the queue rolls and slot 0 takes the newest frame, then each env draws
    an independent index in ``[0, L)``.

    Args:
        queue: Rolling buffer of shape ``(N, L, D)``. Updated out of place.
        new_values: Newest frame, shape ``(N, D)``.
        env: Owning env, used only for ``num_envs``.
        flush: Optional ``(N,)`` bool mask; those envs get every slot filled
            with their new frame, so any sampled index returns the current
            value.

    Returns:
        ``(updated_queue, delayed)`` where ``delayed`` has shape ``(N, D)``.
    """
    del env  # signature parity with T6; N comes from the queue itself
    num_envs, length = queue.shape[0], queue.shape[1]

    if flush is not None and flush.any():
        queue[flush] = new_values[flush, None, :]

    queue = np.roll(queue, shift=1, axis=1)
    queue[:, 0, :] = new_values

    delay_idx = np.random.randint(0, length, size=(num_envs,))
    delayed = queue[np.arange(num_envs), delay_idx]
    return queue, delayed


def _push_and_sample(
    queue: np.ndarray,
    new_values: np.ndarray,
    env: SimToolRealEnv,
    flush: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to T6's delay buffer when it exists, else the local stub.

    Same handoff pattern T2 uses in ``observations.py:177-188``.
    """
    try:
        from .delay_buffer import push_and_sample_delay  # T6 product
    except ImportError:
        return _stub_push_and_sample(queue, new_values, env, flush=flush)
    return push_and_sample_delay(queue, new_values, env, flush=flush)


def apply_action_pipeline(env: SimToolRealEnv, actions: np.ndarray) -> None:
    """Turn policy actions into joint position targets. No return value.

    Side effects on ``env._state.info`` (contract §2.1), all backend joint
    order: writes ``cur_targets`` and ``prev_targets`` (and the bookkeeping
    ``last_actions`` / ``current_actions``, which stay in canonical order —
    they are the raw policy output). The caller (``env.apply_action``) reads
    ``cur_targets`` back out as the backend control.

    Args:
        env: Owning env. Reads ``_perm_canon_to_backend``, ``_arm_slice`` /
            ``_hand_slice``, ``_arm_lower`` / ``_arm_upper`` /
            ``_hand_lower`` / ``_hand_upper`` (all backend order),
            ``_action_queue``, ``cfg.action``, ``cfg.ctrl_dt``, and
            ``cfg.domain_randomization``.
        actions: Policy actions of shape ``(num_envs, 29)`` in canonical joint
            order, nominally in ``[-1, 1]`` but **not** clamped here.

    Raises:
        ValueError: If ``actions`` does not have shape ``(num_envs, 29)``.
    """
    state = env._state
    if state is None:
        raise ValueError("apply_action_pipeline requires an initialized env state")

    info = state.info
    dtype = env._np_dtype

    # Debug replay path: caller supplies a ready backend-order target
    # (action_utils.py:20-25).
    replay_target = getattr(env, "_replay_target_backend_order", None)
    if replay_target is not None:
        info["cur_targets"][:] = np.asarray(replay_target, dtype=dtype)
        info["prev_targets"][:] = info["cur_targets"]
        return

    dr = env.cfg.domain_randomization
    act_cfg = env.cfg.action
    dt = np.float32(env.cfg.ctrl_dt)  # policy step = 1/60 s, NOT sim_dt=1/120

    actions_canon = np.asarray(actions, dtype=dtype)
    expected = (env._num_envs, info["cur_targets"].shape[1])
    if actions_canon.shape != expected:
        raise ValueError(f"actions must have shape {expected}, got {actions_canon.shape}")

    # Raw-action bookkeeping, canonical order, before delay/permute (contract §2.1).
    info["last_actions"][:] = info["current_actions"]
    info["current_actions"][:] = actions_canon

    # 1. Canonical policy order -> backend order (action_utils.py:34). Must come
    #    first: the limit tables and target buffers below are all backend order.
    actions_backend = actions_canon[:, env._perm_canon_to_backend]

    # 2. Action delay, applied *before* the control law (action_utils.py:36-48).
    if dr.use_action_delay and int(dr.action_delay_max) > 0:
        env._action_queue, actions_backend = _push_and_sample(
            env._action_queue,
            actions_backend,
            env,
            flush=_episode_start(state),
        )

    prev_targets = info["prev_targets"]
    prev_arm = prev_targets[:, env._arm_slice]
    prev_hand = prev_targets[:, env._hand_slice]

    # 3. Arm: velocity-delta accumulator, clamped twice (action_utils.py:50-58).
    arm_action = actions_backend[:, env._arm_slice]
    arm_raw = prev_arm + np.float32(act_cfg.dof_speed_scale) * dt * arm_action
    arm_raw = np.clip(arm_raw, env._arm_lower, env._arm_upper)  # clamp #1
    arm_ma = np.float32(act_cfg.arm_moving_average)
    arm_smoothed = arm_ma * arm_raw + (np.float32(1.0) - arm_ma) * prev_arm
    arm_smoothed = np.clip(arm_smoothed, env._arm_lower, env._arm_upper)  # clamp #2

    # 4. Hand: absolute [-1,1] -> [lower,upper], clamped once (action_utils.py:60-69).
    hand_action = actions_backend[:, env._hand_slice]
    hand_span = env._hand_upper - env._hand_lower
    hand_raw = env._hand_lower + np.float32(0.5) * (hand_action + np.float32(1.0)) * hand_span
    hand_ma = np.float32(act_cfg.hand_moving_average)
    hand_smoothed = hand_ma * hand_raw + (np.float32(1.0) - hand_ma) * prev_hand
    hand_smoothed = np.clip(hand_smoothed, env._hand_lower, env._hand_upper)  # clamp #1 (only)

    # 5. Publish backend-order targets, then snapshot for the next step. The
    #    copy is what stops prev/cur from aliasing (action_utils.py:71-74).
    cur_targets = info["cur_targets"]
    cur_targets[:, env._arm_slice] = arm_smoothed.astype(dtype, copy=False)
    cur_targets[:, env._hand_slice] = hand_smoothed.astype(dtype, copy=False)
    prev_targets[:] = cur_targets


__all__ = ["apply_action_pipeline"]
