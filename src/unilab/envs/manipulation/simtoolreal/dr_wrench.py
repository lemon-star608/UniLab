"""Per-step random wrench (force + torque) domain randomisation.

Faithful numpy translation of SimToolReal ``apply_wrench_dr``
(``action_utils.py:77-130``).  The contract signature is frozen in
``MIGRATION_01_INTERFACE_CONTRACT.md §4.8``.

Key design decisions (verified against source):
* force = randn * object_mass * force_scale   (scale=20.0)
* torque = randn * object_mass * torque_scale  (scale=2.0) — torque *also*
  scales by mass, matching the source exactly.
* decay=0  →  impulse: ``_object_forces``/``_object_torques`` zeroed each step,
  then maybe a new impulse is sampled.
* Per-env Bernoulli trigger with probability drawn once at env init from
  log-uniform[0.001, 0.1] (stored as ``env._random_force_prob`` /
  ``env._random_torque_prob``).
* Lift gate: separate ``force_only_when_lifted`` / ``torque_only_when_lifted``
  config flags (both default True).
* Backend call: ``backend.apply_body_wrench`` (D6 — the only approved backend
  change in this migration).
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Init-time helper — call once from SimToolRealEnv.__init__
# ---------------------------------------------------------------------------

def sample_log_uniform(lo: float, hi: float, n: int) -> np.ndarray:
    """Sample *n* values i.i.d. from log-uniform[lo, hi].

    Numpy translation of ``action_utils.sample_log_uniform``
    (``action_utils.py:10-15``).

    Returns:
        ``float32`` array of shape ``(n,)``.
    """
    log_lo = math.log(lo + 1e-12)
    log_hi = math.log(hi + 1e-12)
    return np.exp(
        np.random.uniform(log_lo, log_hi, size=(n,))
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-step wrench DR — call from apply_action (before backend.step)
# ---------------------------------------------------------------------------

def apply_wrench_dr(env) -> None:
    """Apply per-step random force/torque impulses to the manipulated object.

    Translates ``SimToolReal action_utils.apply_wrench_dr`` (torch→numpy).

    Reads from ``env``:
        * ``env.cfg.domain_randomization``  (force_scale, torque_scale,
          force_decay, torque_decay, force_only_when_lifted,
          torque_only_when_lifted)
        * ``env._object_forces``     (N,3) f32 — mutable, zeroed each step
        * ``env._object_torques``    (N,3) f32 — mutable, zeroed each step
        * ``env._random_force_prob`` (N,)  f32 — per-env Bernoulli prob
        * ``env._random_torque_prob``(N,)  f32 — per-env Bernoulli prob
        * ``env._object_mass``       (N,)  f32 — cached at init
        * ``state.info["lifted_object"]``  (N,) bool  — from previous step
        * ``env._object_body_id``    int   — body id of the manipulated object
        * ``env.backend``            SimBackend

    Mutates:
        ``env._object_forces``, ``env._object_torques`` in place; also calls
        ``env.backend.apply_body_wrench`` to stage the xfrc for the next
        physics step.

    Note on lifted state:
        We read ``state.info["lifted_object"]`` (the value from the *previous*
        step's ``update_state``), exactly mirroring the Isaac Lab source comment
        "``_lifted_object`` is from the previous step because rewards update
        later" (``action_utils.py:109``).
    """
    dr = env.cfg.domain_randomization
    n = env.num_envs

    # ------------------------------------------------------------------
    # 1. Decay previous wrench (decay=0 → impulse: zero every step)
    # ------------------------------------------------------------------
    if dr.force_decay > 0.0:
        # Exponential decay: f *= decay^(dt / decay_interval)
        dt = env.cfg.ctrl_dt
        env._object_forces *= dr.force_decay ** (dt / dr.force_decay_interval)
    else:
        env._object_forces[:] = 0.0

    if dr.torque_decay > 0.0:
        dt = env.cfg.ctrl_dt
        env._object_torques *= dr.torque_decay ** (dt / dr.torque_decay_interval)
    else:
        env._object_torques[:] = 0.0

    # ------------------------------------------------------------------
    # 2. Per-env Bernoulli: sample new impulse with probability p
    # ------------------------------------------------------------------
    # force_fire / torque_fire: (N,) bool
    force_fire = np.random.random(n) < env._random_force_prob   # (N,)
    torque_fire = np.random.random(n) < env._random_torque_prob  # (N,)

    # mass: (N,1) for broadcast against (N,3)
    mass = env._object_mass[:, np.newaxis]  # (N,1)

    new_force = np.random.randn(n, 3).astype(np.float32) * mass * dr.force_scale   # (N,3)
    new_torque = np.random.randn(n, 3).astype(np.float32) * mass * dr.torque_scale  # (N,3)

    # Where fire, replace; else keep (already zeroed above for impulse mode)
    env._object_forces = np.where(force_fire[:, np.newaxis], new_force, env._object_forces)
    env._object_torques = np.where(torque_fire[:, np.newaxis], new_torque, env._object_torques)

    # ------------------------------------------------------------------
    # 3. Lift gate — zero out force/torque when object not yet lifted
    # ------------------------------------------------------------------
    # info["lifted_object"] is set by update_state in the *previous* step.
    lifted = env._state_cache_lifted_object  # (N,) bool, see note below
    if dr.force_only_when_lifted:
        env._object_forces *= lifted[:, np.newaxis].astype(np.float32)
    if dr.torque_only_when_lifted:
        env._object_torques *= lifted[:, np.newaxis].astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Apply to backend (D6 — stages in _pending_xfrc_applied)
    # ------------------------------------------------------------------
    # Backend expects shape (num_envs, num_bodies, 3); we have a single body.
    forces_3d = env._object_forces[:, np.newaxis, :]   # (N,1,3)
    torques_3d = env._object_torques[:, np.newaxis, :]  # (N,1,3)
    body_ids = np.array([env._object_body_id], dtype=np.int32)
    env.backend.apply_body_wrench(body_ids, forces_3d, torques_3d)
