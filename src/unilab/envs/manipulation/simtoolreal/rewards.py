"""Reward computation for SimToolReal.

Ports the seven-term reward from ``reward_utils.py``, plus the delta-progress
(d*) mechanism and the lifting latch. Source locations, relative to the
SimToolReal repo root (``isaacsimenvs/tasks/simtoolreal/``):

    compute_rewards             utils/reward_utils.py:92-153
    lifting_reward              utils/reward_utils.py:8-23
    distance_delta_reward       utils/reward_utils.py:26-37
    keypoint_reward             utils/reward_utils.py:40-51
    action_penalty              utils/reward_utils.py:54-64
    reach_goal_bonus            utils/reward_utils.py:79-89
    update_near_goal_steps      utils/reward_utils.py:67-76

**No global scaling.** Line :141 returns the direct sum of the seven terms,
with no ``ctrl_dt`` or ``0.01`` multiplier. The ``×0.01`` that appears in
``obs_utils.py:326`` (the ``reward`` privileged feature) is a *feature
normalization* for the critic, not a reward-signal scale. The training-side
``scale_value: 0.01`` (cfg/train/SimToolRealSAPG.yaml:77) is applied by the Hydra owner config, not this module.

**Two action penalties, both on joint velocity.** Despite the name
"action_penalty," both terms penalize the L1 norm of *joint velocity*, not the
magnitude of the policy's action output (:62-63).

**Lifting latch.** ``lifted_object`` is set ``True`` once ``z_lift`` crosses
the threshold and stays ``True`` for the rest of the episode (:19, boolean OR).
Once latched, the lifting progress reward zeroes out (:22, ``*(~lifted)``) and
the keypoint reward activates (:50, ``*lifted``).

**z_lift offset.** The lifting progress is ``0.05 + object_z - object_init_z``
(:17), not a bare difference.

**Reach bonus amortization.** When ``force_consecutive_near_goal_steps=False``
(the default), the 1000-point reach bonus is amortized as
``near_goal * (1000 / success_steps)`` = 100 points per step near the goal
(:86-89), accumulating to 1000 over ``success_steps=10``. It is not a
sparse one-shot bonus.

**Side effect: env._reward_terms.** The function must write an 8-key dict to
``env._reward_terms`` for logging (:142-151). Keys are
``fingertip_delta_rew``, ``lifting_rew``, ``lift_bonus_rew``, ``keypoint_rew``,
``kuka_actions_penalty``, ``hand_actions_penalty``, ``bonus_rew``,
``total_reward``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.dtype_config import get_global_dtype


def lifting_reward(
    object_z: np.ndarray,
    object_init_z: np.ndarray,
    prev_lifted: np.ndarray,
    lifting_bonus_threshold: float,
    lifting_bonus: float,
    lifting_rew_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute lifting progress, one-shot bonus, and latched lifted state.

    Ported from reward_utils.py:8-23.

    Args:
        object_z: Current object z-coordinate, shape ``(N,)``.
        object_init_z: Object z at episode start, shape ``(N,)``.
        prev_lifted: Previous lifted latch, shape ``(N,)``, bool.
        lifting_bonus_threshold: Threshold z-lift for the bonus (cfg 0.15).
        lifting_bonus: One-shot bonus value (cfg 300.0).
        lifting_rew_scale: Scale for the progress reward (cfg 20.0).

    Returns:
        A 3-tuple ``(lift_rew, lift_bonus_rew, lifted)`` where:
          - ``lift_rew``: Lifting progress reward, shape ``(N,)``.
          - ``lift_bonus_rew``: One-shot bonus, shape ``(N,)``.
          - ``lifted``: Updated latch, shape ``(N,)``, bool.
    """
    dtype = get_global_dtype()
    # Source :17 — note the +0.05 offset.
    z_lift = 0.05 + object_z - object_init_z
    # Progress clamped to [0, 0.5], source :18.
    lift_rew = np.clip(z_lift, 0.0, 0.5)
    # Latch: once True, stays True for the episode (boolean OR, :19).
    lifted = (z_lift > lifting_bonus_threshold) | prev_lifted
    # One-shot bonus on the frame that crosses the threshold (:20).
    just_crossed = lifted & ~prev_lifted
    lift_bonus = lifting_bonus * just_crossed.astype(dtype)
    # Zero out lift_rew once lifted (:22).
    lift_rew = lift_rew * (~lifted).astype(dtype)
    return lift_rew * lifting_rew_scale, lift_bonus, lifted


def distance_delta_reward(
    curr_fingertip_dist: np.ndarray,
    closest_fingertip_dist: np.ndarray,
    lifted: np.ndarray,
    rew_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reward fingertip progress before the object is lifted (d* per fingertip).

    Ported from reward_utils.py:26-37.

    Args:
        curr_fingertip_dist: Current distances, shape ``(N, num_fingertips)``.
        closest_fingertip_dist: Historical best (d*), shape ``(N, num_fingertips)``.
        lifted: Lifted latch, shape ``(N,)``, bool.
        rew_scale: Scale for this term (cfg 50.0).

    Returns:
        A 2-tuple ``(rew, new_closest)`` where:
          - ``rew``: Delta-progress reward, shape ``(N,)``.
          - ``new_closest``: Updated d*, shape ``(N, num_fingertips)``.
    """
    dtype = get_global_dtype()
    # Delta = d* - current; only reward if positive (closer than ever before), :33.
    deltas = closest_fingertip_dist - curr_fingertip_dist
    # Update d* to the new minimum (:34).
    new_closest = np.minimum(closest_fingertip_dist, curr_fingertip_dist)
    # Clamp delta to [0, 10] (:35).
    deltas = np.clip(deltas, 0.0, 10.0)
    # Sum across fingertips, gate on ~lifted (:36).
    rew = deltas.sum(axis=-1) * (~lifted).astype(dtype)
    return rew * rew_scale, new_closest


def keypoint_reward(
    keypoints_max_dist: np.ndarray,
    closest_keypoint_max_dist: np.ndarray,
    lifted: np.ndarray,
    rew_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reward keypoint progress after the object is lifted (d* on keypoint dist).

    Ported from reward_utils.py:40-51.

    Args:
        keypoints_max_dist: Current distance, shape ``(N,)``.
        closest_keypoint_max_dist: Historical best (d*), shape ``(N,)``.
        lifted: Lifted latch, shape ``(N,)``, bool.
        rew_scale: Scale for this term (cfg 200.0).

    Returns:
        A 2-tuple ``(rew, new_closest)`` where:
          - ``rew``: Delta-progress reward, shape ``(N,)``.
          - ``new_closest``: Updated d*, shape ``(N,)``.
    """
    dtype = get_global_dtype()
    # Delta = d* - current (:47).
    delta = closest_keypoint_max_dist - keypoints_max_dist
    # Update d* (:48).
    new_closest = np.minimum(closest_keypoint_max_dist, keypoints_max_dist)
    # Clamp to [0, 100] (:49).
    delta = np.clip(delta, 0.0, 100.0)
    # Gate on lifted (:50).
    rew = delta * lifted.astype(dtype)
    return rew * rew_scale, new_closest


def action_penalty(
    joint_vel: np.ndarray,
    arm_slice: slice,
    hand_slice: slice,
    kuka_scale: float,
    hand_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """L1 joint-velocity penalty (two terms: arm and hand).

    Ported from reward_utils.py:54-64. Despite the "action_penalty" name, this
    penalizes the L1 norm of *joint velocity*, not the action magnitude (:62-63).

    Args:
        joint_vel: Joint velocities, shape ``(N, num_dofs)``, in canonical order.
        arm_slice: Slice for the arm joints (typically ``slice(0, 7)``).
        hand_slice: Slice for the hand joints (typically ``slice(7, 29)``).
        kuka_scale: Arm penalty scale (cfg 0.03, **positive**; negated inside).
        hand_scale: Hand penalty scale (cfg 0.003, **positive**; negated inside).

    Returns:
        A 2-tuple ``(kuka_pen, hand_pen)``, both shape ``(N,)``, negative.
    """
    # Source :62-63 — negation inside the function, matching torch source.
    kuka = -kuka_scale * np.abs(joint_vel[:, arm_slice]).sum(axis=-1)
    hand = -hand_scale * np.abs(joint_vel[:, hand_slice]).sum(axis=-1)
    return kuka, hand


def update_near_goal_steps(
    near_goal: np.ndarray,
    near_goal_steps: np.ndarray,
    force_consecutive: bool,
) -> np.ndarray:
    """Update the near-goal counter, optionally requiring consecutive steps.

    Ported from reward_utils.py:67-76. Used by the episode-lifecycle success logic, not directly
    by the reward.

    Args:
        near_goal: Boolean mask, shape ``(N,)``.
        near_goal_steps: Current counter, shape ``(N,)``, int.
        force_consecutive: If True, reset to zero on any non-near step.

    Returns:
        Updated counter, shape ``(N,)``, int.
    """
    ng = near_goal.astype(np.int32)
    if force_consecutive:
        # Reset to zero if not near; increment if near (:75).
        return (near_goal_steps + ng) * ng
    # Default: cumulative (:76).
    return near_goal_steps + ng


def reach_goal_bonus(
    near_goal: np.ndarray,
    is_success: np.ndarray,
    reach_goal_bonus_value: float,
    success_steps: int,
    force_consecutive: bool,
) -> np.ndarray:
    """Return the reach-goal bonus, either lump-sum or amortized.

    Ported from reward_utils.py:79-89.

    Args:
        near_goal: Boolean mask, shape ``(N,)``.
        is_success: Boolean mask, shape ``(N,)``.
        reach_goal_bonus_value: Total bonus (cfg 1000.0).
        success_steps: Steps required for success (cfg 10).
        force_consecutive: If True, give the lump sum on success; otherwise
            amortize over the near-goal steps.

    Returns:
        Bonus reward, shape ``(N,)``.
    """
    dtype = get_global_dtype()
    if force_consecutive:
        # Lump sum on success (:88).
        return is_success.astype(dtype) * reach_goal_bonus_value
    # Default: amortize = near_goal * (1000 / 10) = 100 per step (:89).
    return near_goal.astype(dtype) * (reach_goal_bonus_value / success_steps)


def compute_rewards(env: Any, info: dict[str, np.ndarray]) -> np.ndarray:
    """Sum the seven reward terms and update d* trackers.

    Frozen signature from interface contract §4.5. The torch source is
    ``compute_rewards(env) -> torch.Tensor`` (reward_utils.py:92); the numpy
    version takes ``info`` as a parameter because cross-step state lives in
    ``state.info`` rather than as env attributes (decision D0, contract §2).

    **Side effect**: writes ``env._reward_terms``, an 8-key dict for logging
    (source :142-151). Keys are ``fingertip_delta_rew``, ``lifting_rew``,
    ``lift_bonus_rew``, ``keypoint_rew``, ``kuka_actions_penalty``,
    ``hand_actions_penalty``, ``bonus_rew``, ``total_reward``.

    **No global scaling.** Line :141 returns the direct sum; do not multiply by
    ``ctrl_dt`` or ``0.01``. The contract audit (P0-1) caught that error.

    Args:
        env: The environment instance. Read-only access to config, backend,
            and intermediate values (``_curr_fingertip_distances``,
            ``_keypoints_max_dist``, ``_near_goal``, ``_is_success``).
        info: The ``state.info`` dict. Read-write access to d* trackers
            (``closest_fingertip_dist``, ``closest_keypoint_max_dist``,
            ``lifted_object``).

    Returns:
        Total reward, shape ``(N,)``, float32.
    """
    dtype = get_global_dtype()
    rew_cfg = env.cfg.reward_config
    term_cfg = env.cfg.termination
    # success_steps lives on GoalCfg, not TerminationCfg — the source keeps it on
    # TerminationCfg (cfg:437) but the contract regroups goal-side fields onto
    # GoalCfg (§5.0). Reading it off term_cfg raises AttributeError on the real
    # config; it only survived unit test because the mock cfg is a MagicMock.
    goal_cfg = env.cfg.goal

    # 1. Lifting: progress + one-shot bonus + latch (source :99-107).
    lift_rew, lift_bonus_rew, new_lifted = lifting_reward(
        object_z=env._object_pos[:, 2],
        object_init_z=info["object_init_z"],
        prev_lifted=info["lifted_object"],
        lifting_bonus_threshold=rew_cfg.lifting_bonus_threshold,
        lifting_bonus=rew_cfg.lifting_bonus,
        lifting_rew_scale=rew_cfg.lifting_rew_scale,
    )
    info["lifted_object"] = new_lifted

    # 2. Fingertip delta-progress (d* per fingertip), gated on ~lifted (:109-115).
    ft_rew, new_closest_ft = distance_delta_reward(
        curr_fingertip_dist=env._curr_fingertip_distances,
        closest_fingertip_dist=info["closest_fingertip_dist"],
        lifted=info["lifted_object"],
        rew_scale=rew_cfg.distance_delta_rew_scale,
    )
    info["closest_fingertip_dist"] = new_closest_ft

    # 3. Keypoint delta-progress (d* on keypoint max dist), gated on lifted (:117-123).
    kp_rew, new_closest_kp = keypoint_reward(
        keypoints_max_dist=env._keypoints_max_dist,
        closest_keypoint_max_dist=info["closest_keypoint_max_dist"],
        lifted=info["lifted_object"],
        rew_scale=rew_cfg.keypoint_rew_scale,
    )
    info["closest_keypoint_max_dist"] = new_closest_kp

    # 4. Action penalties (joint velocity L1, two terms) (:125-131).
    kuka_pen, hand_pen = action_penalty(
        joint_vel=env._joint_vel,
        arm_slice=env._arm_slice,
        hand_slice=env._hand_slice,
        kuka_scale=rew_cfg.kuka_actions_penalty_scale,
        hand_scale=rew_cfg.hand_actions_penalty_scale,
    )

    # 5. Reach-goal bonus (amortized by default) (:133-139).
    bonus = reach_goal_bonus(
        near_goal=env._near_goal,
        is_success=env._is_success,
        reach_goal_bonus_value=rew_cfg.reach_goal_bonus,
        success_steps=goal_cfg.success_steps,
        force_consecutive=term_cfg.force_consecutive_near_goal_steps,
    )

    # 6. Sum all terms, no global scaling (:141).
    reward = lift_rew + lift_bonus_rew + ft_rew + kp_rew + kuka_pen + hand_pen + bonus

    # 7. Side effect: write _reward_terms for logging (:142-151).
    env._reward_terms = {
        "fingertip_delta_rew": ft_rew.astype(dtype, copy=False),
        "lifting_rew": lift_rew.astype(dtype, copy=False),
        "lift_bonus_rew": lift_bonus_rew.astype(dtype, copy=False),
        "keypoint_rew": kp_rew.astype(dtype, copy=False),
        "kuka_actions_penalty": kuka_pen.astype(dtype, copy=False),
        "hand_actions_penalty": hand_pen.astype(dtype, copy=False),
        "bonus_rew": bonus.astype(dtype, copy=False),
        "total_reward": reward.astype(dtype, copy=False),
    }

    return reward.astype(dtype, copy=False)


__all__ = [
    "action_penalty",
    "compute_rewards",
    "distance_delta_reward",
    "keypoint_reward",
    "lifting_reward",
    "reach_goal_bonus",
    "update_near_goal_steps",
]
