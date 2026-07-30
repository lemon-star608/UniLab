"""Observation assembly for SimToolReal (migration task T2).

Ports ``obs_utils.build_observations`` (obs_utils.py:231-349) to NumPy.

Returns two flat arrays per step:

* ``"obs"``    — actor observation (noisy, optionally delayed, xyzw quaternions)
* ``"critic"`` — critic state     (clean, no delay, same xyzw convention + privileged fields)

Both are clamped to ``[-cfg.obs.clamp_abs_observations, +cfg.obs.clamp_abs_observations]``.

Frozen signature (interface contract §4.4)::

    def build_observations(env, state) -> dict[str, np.ndarray]

All arrays are numpy float32 on CPU (MIGRATION_00 decision D0).
Quaternions are wxyz internally; xyzw conversion is applied to **both** actor
and critic before stacking (decision D3, obs_utils.py:303-306).

⚠️  25× trap: ``state.info["object_scales"]`` (phi) is dimensionless.
Pass it through :func:`observation_keypoint_size` before calling
:func:`compute_keypoints` (contract §4.3, T5 acceptance broadcast).

Delay-queue mechanics (T6 contract §4.2)
-----------------------------------------
T6 owns ``delay_buffer.push_and_sample_delay``.  Until T6 ships this module
imports that function at call time; if the module does not yet exist the stub
``_stub_push_and_sample`` is used instead (returns the current frame with no
delay, same interface).  Swap-in is a single import change — no other edits
needed.

Source file references (relative to the SimToolReal repo root,
``isaacsimenvs/tasks/simtoolreal/utils/``):

    build_observations      obs_utils.py:231
    _sample_delay           obs_utils.py:119
    d* sentinel             obs_utils.py:183-189
    _apply_object_state_dr  obs_utils.py:210-220
    _apply_obs_delay        obs_utils.py:223-228
    kp_offsets DR noise     obs_utils.py:275
    object_scales_obs       obs_utils.py:301
    wxyz→xyzw (both obs)    obs_utils.py:303-306
    reward feature ×0.01    obs_utils.py:326
    clamp [-10,10]          obs_utils.py:346-348
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.utils.rotation import np_quat_apply

from .constants import NUM_FINGERTIPS, NUM_KEYPOINTS
from .keypoints import compute_keypoints, observation_keypoint_size

if TYPE_CHECKING:
    from unilab.base.np_env import NpEnvState

    from .env import SimToolRealEnv


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert (N, 4) wxyz → xyzw.  obs_utils.py:304 pattern."""
    return np.concatenate([q[..., 1:], q[..., :1]], axis=-1)


def _normalize_joint_pos(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Normalize joint positions to [-1, 1].

    Formula: ``2*(q - lower) / (upper - lower) - 1``.
    Source: obs_utils.py:141-144 (``_canonical_joint_obs``).
    """
    return 2.0 * (q - lower) / (upper - lower) - 1.0


def _quat_from_angle_axis(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Batched angle-axis → wxyz quaternion.

    Args:
        angle: Shape ``(N,)``, radians.
        axis:  Shape ``(N, 3)``, must be unit vectors.

    Returns:
        ``(N, 4)`` wxyz quaternions.
    """
    half = (angle * 0.5).astype(get_global_dtype())
    w = np.cos(half)[:, None]
    xyz = np.sin(half)[:, None] * axis
    return np.concatenate([w, xyz], axis=-1)


def _perturb_quat(q_wxyz: np.ndarray, max_deg: float) -> np.ndarray:
    """Apply random-axis rotation noise to a batch of wxyz quaternions.

    numpy port of ``obs_utils._perturb_quat`` (obs_utils.py:75-85).

    Args:
        q_wxyz:  ``(N, 4)`` wxyz quaternions.
        max_deg: Maximum rotation magnitude in degrees.

    Returns:
        ``(N, 4)`` perturbed wxyz quaternions.
    """
    n = q_wxyz.shape[0]
    dtype = get_global_dtype()
    # Random unit axis
    raw = np.random.randn(n, 3).astype(dtype)
    norm = np.linalg.norm(raw, axis=-1, keepdims=True).clip(min=1e-8)
    axis = raw / norm

    max_rad = float(max_deg) * (np.pi / 180.0)
    angle = np.random.uniform(-max_rad, max_rad, size=(n,)).astype(dtype)

    dq = _quat_from_angle_axis(angle, axis)   # (N, 4) wxyz

    # q_new = dq * q_wxyz  (Hamilton product, batched)  obs_utils.py:85
    w1, x1, y1, z1 = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]
    w2, x2, y2, z2 = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1).astype(dtype)


def _episode_start(state: NpEnvState) -> np.ndarray:
    """Boolean mask: True on the very first step of a fresh episode.

    Mirrors ``obs_utils._episode_start``: ``(steps == 0) & (successes == 0)``.
    Used to flush delay queues (contract §4.2 / obs_utils.py:215).
    """
    steps = np.asarray(state.info["steps"], dtype=np.int32)
    successes = np.asarray(state.info["successes"], dtype=np.int32)
    return (steps == 0) & (successes == 0)


# ---------------------------------------------------------------------------
# Delay-queue helpers  (T6 stub until delay_buffer.py ships)
# ---------------------------------------------------------------------------


def _stub_push_and_sample(
    queue: np.ndarray,
    new_values: np.ndarray,
    env: SimToolRealEnv,
    flush: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """No-delay stub for the T6 ``push_and_sample_delay`` contract §4.2.

    Rolls the queue and returns the freshest frame so callers are exercising
    the real code path.  Swap out for ``delay_buffer.push_and_sample_delay``
    when T6 is complete.

    Mirrors obs_utils._sample_delay (obs_utils.py:119-133) without the random
    index: idx is always 0 (latest).
    """
    if flush is not None and flush.any():
        queue[flush] = new_values[flush, None, :]  # broadcast to all L slots
    queue = np.roll(queue, shift=1, axis=1)
    queue[:, 0, :] = new_values
    # Stub: always return the most-recent slot (no actual delay).
    return queue, new_values


def _push_and_sample(
    queue: np.ndarray,
    new_values: np.ndarray,
    env: SimToolRealEnv,
    flush: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to T6 implementation when available, else stub."""
    try:
        from .delay_buffer import push_and_sample_delay  # T6 product
        return push_and_sample_delay(queue, new_values, env, flush=flush)
    except ImportError:
        return _stub_push_and_sample(queue, new_values, env, flush=flush)


# ---------------------------------------------------------------------------
# Object-state delay + noise  (obs_utils.py:210-220)
# ---------------------------------------------------------------------------


def _apply_object_state_dr(
    env: SimToolRealEnv,
    obj_pos: np.ndarray,       # (N, 3)
    obj_rot_wxyz: np.ndarray,  # (N, 4) wxyz
    obj_linvel: np.ndarray,    # (N, 3)
    obj_angvel: np.ndarray,    # (N, 3)
    flush: np.ndarray,         # (N,) bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply object-state delay then pose noise.

    numpy port of ``_apply_object_state_dr`` (obs_utils.py:210-220).

    Returns:
        ``(noisy_pos, noisy_rot_wxyz, noisy_vel)`` each ``(N, 3/4/6)``.
    """
    dr = env.cfg.domain_randomization
    dtype = env._np_dtype

    # Pack state frame: pos(3) + quat(4) + linvel(3) + angvel(3) = 13
    state_frame = np.concatenate(
        [obj_pos, obj_rot_wxyz, obj_linvel, obj_angvel], axis=-1
    ).astype(dtype)

    # Push into rolling queue and sample delayed frame
    env._object_state_queue, delayed = _push_and_sample(
        env._object_state_queue, state_frame, env, flush=flush
    )

    # Unpack delayed frame
    noisy_pos = delayed[:, 0:3] + (
        np.random.randn(env._num_envs, 3).astype(dtype)
        * np.float32(dr.object_state_xyz_noise_std)
    )
    noisy_rot = _perturb_quat(delayed[:, 3:7], dr.object_state_rotation_noise_degrees)
    noisy_vel = delayed[:, 7:13]   # no velocity noise (obs_utils.py:219)

    return noisy_pos, noisy_rot, noisy_vel


# ---------------------------------------------------------------------------
# Policy-observation delay  (obs_utils.py:223-228)
# ---------------------------------------------------------------------------


def _apply_obs_delay(
    env: SimToolRealEnv,
    state: NpEnvState,
    policy_tensor: np.ndarray,  # (N, D_actor)
) -> np.ndarray:
    """Push actor obs into queue and sample per-env delayed frame.

    numpy port of ``_apply_obs_delay`` (obs_utils.py:223-228).
    """
    flush = _episode_start(state)
    env._obs_queue, delayed = _push_and_sample(
        env._obs_queue, policy_tensor, env, flush=flush
    )
    return delayed


# ---------------------------------------------------------------------------
# Scale-multiplier helper  (obs_utils.py:275, 301)
# ---------------------------------------------------------------------------


def _get_scale_multiplier(env: SimToolRealEnv) -> np.ndarray:
    """Return per-env object scale-noise multiplier, shape ``(N, 3)``.

    Lazily initialised to 1.0 (no-op) the first time it is accessed, matching
    the default DR range of ``(1.0, 1.0)``.  T4 / the DR provider will resample
    on every reset once it is set up.

    Source: ``env._object_scale_multiplier`` (obs_utils.py:275, reset_utils.py:87).
    """
    mult = getattr(env, "_object_scale_multiplier", None)
    if mult is None:
        mult = np.ones((env._num_envs, 3), dtype=env._np_dtype)
        env._object_scale_multiplier = mult
    return np.asarray(mult, dtype=env._np_dtype)


# ---------------------------------------------------------------------------
# Observation stacking helper
# ---------------------------------------------------------------------------


def _stack_obs(obs_dict: dict[str, np.ndarray], field_list: tuple[str, ...]) -> np.ndarray:
    """Concatenate named observation tensors in config order.

    Each tensor is reshaped to ``(N, -1)`` before concatenation so that 2-D
    fields (fingertip/keypoints) are automatically flattened.
    """
    parts = [obs_dict[f].reshape(obs_dict[f].shape[0], -1) for f in field_list]
    return np.concatenate(parts, axis=-1)


# ---------------------------------------------------------------------------
# Public API  (contract §4.4)
# ---------------------------------------------------------------------------


def build_observations(
    env: SimToolRealEnv,
    state: NpEnvState,
) -> dict[str, np.ndarray]:
    """Assemble actor and critic observations with obs-side domain randomisation.

    Frozen signature (interface contract §4.4).  All data is read from
    ``env.backend`` and ``state.info`` — no separate data arguments.

    Returns:
        ``{"obs": actor_obs, "critic": critic_obs}`` where:

        * ``actor_obs``  shape ``(N, num_actor_obs)``  — noisy, optionally
          delayed, xyzw quaternions (obs_utils.py:340-343).
        * ``critic_obs`` shape ``(N, num_critic_obs)`` — clean, no delay, same
          xyzw convention, plus privileged fields (obs_utils.py:339).

        Both arrays are float32 and clamped to
        ``±cfg.obs.clamp_abs_observations`` (obs_utils.py:346-348).
    """
    dtype = env._np_dtype
    n = env._num_envs
    info = state.info
    dr = env.cfg.domain_randomization
    rew_cfg = env.cfg.reward
    goal_cfg = env.cfg.goal

    # ------------------------------------------------------------------ #
    # 1. Joint state (canonical order)  obs_utils.py:136-145             #
    # ------------------------------------------------------------------ #
    joint_pos_raw = env.get_joint_pos_canon()    # (N, 29)
    joint_vel = env.get_joint_vel_canon()         # (N, 29)

    # Normalise joint positions: 2*(q-lower)/(upper-lower)-1  obs_utils.py:141
    joint_pos = _normalize_joint_pos(
        joint_pos_raw,
        env._joint_lower_canon,   # (29,)  obs_utils.py:obs_utils reset_utils.py:55
        env._joint_upper_canon,   # (29,)
    )

    # prev_action_targets: stored backend-order → convert to canonical  obs_utils.py:145
    prev_targets_backend = np.asarray(info["prev_targets"], dtype=dtype)   # (N, 29)
    prev_targets_canon = prev_targets_backend[:, env._perm_backend_to_canon]  # (N, 29)

    # ------------------------------------------------------------------ #
    # 2. Palm  obs_utils.py:238-246                                       #
    # ------------------------------------------------------------------ #
    palm_ids = np.asarray([env._palm_body_id], dtype=np.int32)
    palm_pos_w = env._backend.get_body_pos_w(palm_ids)[:, 0, :]       # (N, 3)
    palm_rot_wxyz = env._backend.get_body_quat_w(palm_ids)[:, 0, :]    # (N, 4) wxyz
    palm_linvel = env._backend.get_body_lin_vel_w(palm_ids)[:, 0, :]   # (N, 3)
    palm_angvel = env._backend.get_body_ang_vel_w(palm_ids)[:, 0, :]   # (N, 3)
    palm_vel = np.concatenate([palm_linvel, palm_angvel], axis=-1)      # (N, 6)

    # Apply palm-centre local offset: palm_pos = w_origin + R @ offset
    # obs_utils.py:243-246; offset = PALM_CENTER_OFFSET = (-0.0, -0.02, 0.16)
    palm_offset_b = np.broadcast_to(env._palm_offset, (n, 3)).copy()
    palm_center_pos = palm_pos_w + np_quat_apply(palm_rot_wxyz, palm_offset_b)  # (N, 3)

    # ------------------------------------------------------------------ #
    # 3. Fingertips  obs_utils.py:248-299                                 #
    # ------------------------------------------------------------------ #
    ft_ids = env._fingertip_body_ids                              # (5,) int32
    ft_pos_w = env._backend.get_body_pos_w(ft_ids)               # (N, 5, 3)
    ft_rot_w = env._backend.get_body_quat_w(ft_ids)              # (N, 5, 4) wxyz

    # Apply fingertip local offset per finger  obs_utils.py:252-257
    ft_off_b = np.broadcast_to(env._fingertip_offset, (n * NUM_FINGERTIPS, 3)).copy()
    ft_rot_flat = ft_rot_w.reshape(-1, 4)                         # (N*5, 4)
    ft_center_flat = ft_pos_w.reshape(-1, 3) + np_quat_apply(ft_rot_flat, ft_off_b)
    ft_center = ft_center_flat.reshape(n, NUM_FINGERTIPS, 3)      # (N, 5, 3)

    # Fingertip positions relative to palm centre  obs_utils.py:297-299
    fingertip_pos_rel_palm = ft_center - palm_center_pos[:, None, :]   # (N, 5, 3)

    # ------------------------------------------------------------------ #
    # 4. Object  obs_utils.py:259-263                                     #
    # ------------------------------------------------------------------ #
    obj_ids = np.asarray([env._object_body_id], dtype=np.int32)
    obj_pos = env._backend.get_body_pos_w(obj_ids)[:, 0, :]        # (N, 3)
    obj_rot_wxyz = env._backend.get_body_quat_w(obj_ids)[:, 0, :]   # (N, 4) wxyz
    obj_linvel = env._backend.get_body_lin_vel_w(obj_ids)[:, 0, :]  # (N, 3)
    obj_angvel = env._backend.get_body_ang_vel_w(obj_ids)[:, 0, :]  # (N, 3)
    obj_vel = np.concatenate([obj_linvel, obj_angvel], axis=-1)      # (N, 6)

    # ------------------------------------------------------------------ #
    # 5. Goal  (D7: no physics goal_viz, read from state.info)            #
    # ------------------------------------------------------------------ #
    goal_pos = np.asarray(info["goal_pos"], dtype=dtype)       # (N, 3)
    goal_rot_wxyz = np.asarray(info["goal_quat"], dtype=dtype)  # (N, 4) wxyz

    # ------------------------------------------------------------------ #
    # 6. Object-state delay + noise  obs_utils.py:268-273                 #
    # ------------------------------------------------------------------ #
    flush = _episode_start(state)
    if dr.use_object_state_delay_noise:
        noisy_obj_pos, noisy_obj_rot_wxyz, noisy_obj_vel = _apply_object_state_dr(
            env, obj_pos, obj_rot_wxyz, obj_linvel, obj_angvel, flush
        )
    else:
        noisy_obj_pos = obj_pos
        noisy_obj_rot_wxyz = obj_rot_wxyz
        noisy_obj_vel = obj_vel

    # ------------------------------------------------------------------ #
    # 7. Observation-path keypoints  obs_utils.py:275-295                 #
    # ------------------------------------------------------------------ #
    # scale_multiplier: per-env DR noise on object extents  obs_utils.py:275
    scale_mult = _get_scale_multiplier(env)    # (N, 3) ≈ 1.0 by default

    # phi = state.info["object_scales"] is dimensionless (handle bbox / 0.04).
    # Must convert to metres via observation_keypoint_size before compute_keypoints.
    # Passing phi directly → 25× inflated offsets (contract §4.3 ⚠ 25× trap).
    phi = np.asarray(info["object_scales"], dtype=dtype)   # (N, 3) dimensionless
    size_m = observation_keypoint_size(
        phi,
        rew_cfg.object_base_size,   # 0.04 m  obs_utils.py:275 / reset_utils.py:76
        scale_mult,
    )   # (N, 3) metres

    ks = float(goal_cfg.keypoint_scale)    # 1.5  obs_utils.py via simtoolreal_env_cfg.py:337

    # Clean keypoints (for critic / no noise)
    obj_kp = compute_keypoints(obj_pos, obj_rot_wxyz, size_m, keypoint_scale=ks)    # (N,4,3)
    goal_kp = compute_keypoints(goal_pos, goal_rot_wxyz, size_m, keypoint_scale=ks) # (N,4,3)

    # Noisy keypoints (for actor)  obs_utils.py:278
    noisy_obj_kp = compute_keypoints(
        noisy_obj_pos, noisy_obj_rot_wxyz, size_m, keypoint_scale=ks
    )  # (N,4,3)

    # Keypoint positions relative to palm and goal  obs_utils.py:292-295
    keypoints_rel_palm_clean = obj_kp - palm_center_pos[:, None, :]     # (N,4,3)
    keypoints_rel_palm_noisy = noisy_obj_kp - palm_center_pos[:, None, :]
    keypoints_rel_goal_clean = obj_kp - goal_kp
    keypoints_rel_goal_noisy = noisy_obj_kp - goal_kp   # goal not perturbed (no yaw noise)

    # ------------------------------------------------------------------ #
    # 8. object_scales obs field  obs_utils.py:301                        #
    # ------------------------------------------------------------------ #
    # phi × multiplier: dimensionless, lets policy see per-object scale DR
    object_scales_obs = phi * scale_mult    # (N, 3) still dimensionless

    # ------------------------------------------------------------------ #
    # 9. wxyz → xyzw  (D3, obs_utils.py:303-306)                         #
    # BOTH actor and critic quaternion fields converted — not just actor. #
    # ------------------------------------------------------------------ #
    palm_rot_xyzw = _wxyz_to_xyzw(palm_rot_wxyz)
    obj_rot_xyzw = _wxyz_to_xyzw(obj_rot_wxyz)
    noisy_obj_rot_xyzw = _wxyz_to_xyzw(noisy_obj_rot_wxyz)

    # ------------------------------------------------------------------ #
    # 10. Privileged critic fields  obs_utils.py:321-326                  #
    # ------------------------------------------------------------------ #
    # d* sentinel: -1 on first step; compute_intermediate_values updates to
    # current distance on first real call.  Passed through directly here.
    # obs_utils.py:183-189  (MIGRATION_01 §2.3)
    d_star = np.asarray(info["closest_keypoint_max_dist"], dtype=dtype)[:, None]  # (N,1)
    d_star_ft = np.asarray(info["closest_fingertip_dist"], dtype=dtype)           # (N,5)
    lifted = np.asarray(info["lifted_object"], dtype=dtype)[:, None]              # (N,1)

    # progress: log(episode_step / 10 + 1)  obs_utils.py:324
    steps = np.asarray(info["steps"], dtype=np.float32)    # (N,) uint32 → f32
    progress = np.log(steps / 10.0 + 1.0).astype(dtype)[:, None]               # (N,1)

    # successes: log(successes + 1)  obs_utils.py:325
    succs = np.asarray(info["successes"], dtype=np.float32)
    successes = np.log(succs + 1.0).astype(dtype)[:, None]                      # (N,1)

    # reward feature: previous step's reward × 0.01 (feature normalisation,
    # NOT reward scaling)  obs_utils.py:326
    reward_feat = np.asarray(info["reward"], dtype=dtype)[:, None] * np.float32(0.01)  # (N,1)

    # ------------------------------------------------------------------ #
    # 11. Assemble clean (critic) and noisy (actor) obs dicts              #
    #     obs_utils.py:308-337                                             #
    # ------------------------------------------------------------------ #
    obs_clean: dict[str, np.ndarray] = {
        "joint_pos":              joint_pos,
        "joint_vel":              joint_vel,
        "prev_action_targets":    prev_targets_canon,
        "palm_pos":               palm_center_pos,
        "palm_rot":               palm_rot_xyzw,            # xyzw
        "palm_vel":               palm_vel,
        "object_rot":             obj_rot_xyzw,             # xyzw clean
        "object_vel":             obj_vel,
        "fingertip_pos_rel_palm": fingertip_pos_rel_palm,
        "keypoints_rel_palm":     keypoints_rel_palm_clean,
        "keypoints_rel_goal":     keypoints_rel_goal_clean,
        "object_scales":          object_scales_obs,
        "closest_keypoint_max_dist": d_star,
        "closest_fingertip_dist": d_star_ft,
        "lifted_object":          lifted,
        "progress":               progress,
        "successes":              successes,
        "reward":                 reward_feat,
    }

    # obs_noisy inherits all clean fields then overrides the perturbed ones
    # obs_utils.py:329-337
    obs_noisy = dict(obs_clean)
    obs_noisy["object_rot"] = noisy_obj_rot_xyzw
    obs_noisy["object_vel"] = noisy_obj_vel
    obs_noisy["keypoints_rel_palm"] = keypoints_rel_palm_noisy
    obs_noisy["keypoints_rel_goal"] = keypoints_rel_goal_noisy
    if dr.joint_velocity_obs_noise_std > 0:
        noise = np.random.randn(n, joint_vel.shape[1]).astype(dtype)
        obs_noisy["joint_vel"] = (
            joint_vel + noise * np.float32(dr.joint_velocity_obs_noise_std)
        )

    # ------------------------------------------------------------------ #
    # 12. Stack → delay (actor only) → clamp  obs_utils.py:339-348        #
    # ------------------------------------------------------------------ #
    state_tensor = _stack_obs(obs_clean, env.cfg.obs.state_list)   # (N, N_CRITIC)
    policy_tensor = _stack_obs(obs_noisy, env.cfg.obs.obs_list)    # (N, N_ACTOR)

    if dr.use_obs_delay:
        policy_tensor = _apply_obs_delay(env, state, policy_tensor)

    clip = np.float32(env.cfg.obs.clamp_abs_observations)   # 10.0
    policy_tensor = np.clip(policy_tensor, -clip, clip)
    state_tensor = np.clip(state_tensor, -clip, clip)

    return {"obs": policy_tensor.astype(dtype), "critic": state_tensor.astype(dtype)}


__all__ = ["build_observations"]
