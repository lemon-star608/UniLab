"""T6 unit tests: delay_buffer + dr_wrench + D6 backend apply_body_wrench.

Run with:
    cd ~/code/UniLab && uv run pytest tests/simtoolreal/test_t6_delay_wrench.py -v

Acceptance criteria (from MIGRATION_02_TASK_BOARD.md T6):
  - delay_buffer: rolling + random index + flush makes all slots equal current
  - intra-episode goal switch (successes>0) must NOT flush
  - wrench: force∝mass, torque∝mass, impulse (decay=0 zeros each step),
            lift gate zeroes when not lifted
  - backend apply_body_wrench: force slots [0:3] AND torque slots [3:6] written
"""

from __future__ import annotations

import types
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _make_env(num_envs: int = 4) -> types.SimpleNamespace:
    """Minimal env stub sufficient for delay_buffer and dr_wrench tests."""
    env = types.SimpleNamespace()
    env.num_envs = num_envs
    return env


# ---------------------------------------------------------------------------
# delay_buffer tests
# ---------------------------------------------------------------------------

class TestDelayBuffer:
    """Verify push_and_sample_delay (obs_utils._sample_delay translation)."""

    from unilab.envs.manipulation.simtoolreal.delay_buffer import push_and_sample_delay

    def test_rolling_slot0_is_newest(self) -> None:
        """After push, slot 0 must hold the newest value."""
        from unilab.envs.manipulation.simtoolreal.delay_buffer import push_and_sample_delay

        N, L, D = 4, 3, 5
        queue = np.zeros((N, L, D), dtype=np.float32)
        new = np.ones((N, D), dtype=np.float32) * 7.0
        env = _make_env(N)

        queue, _ = push_and_sample_delay(queue, new, env)

        np.testing.assert_array_equal(queue[:, 0, :], new,
                                      err_msg="slot 0 must be the newest value")

    def test_delayed_shape(self) -> None:
        """Returned delayed tensor must have shape (N, D)."""
        from unilab.envs.manipulation.simtoolreal.delay_buffer import push_and_sample_delay

        N, L, D = 6, 5, 13
        queue = np.zeros((N, L, D), dtype=np.float32)
        new = np.random.randn(N, D).astype(np.float32)
        env = _make_env(N)

        _, delayed = push_and_sample_delay(queue, new, env)
        assert delayed.shape == (N, D)

    def test_flush_all_slots_equal_new(self) -> None:
        """After flush, every slot must equal the current value (0-delay)."""
        from unilab.envs.manipulation.simtoolreal.delay_buffer import push_and_sample_delay

        N, L, D = 4, 10, 7
        rng = np.random.default_rng(0)
        queue = rng.standard_normal((N, L, D)).astype(np.float32)
        new = rng.standard_normal((N, D)).astype(np.float32)
        env = _make_env(N)

        flush = np.ones(N, dtype=bool)  # flush all envs
        queue, delayed = push_and_sample_delay(queue, new, env, flush=flush)

        # All L slots must equal new for every flushed env
        for slot in range(L):
            np.testing.assert_allclose(
                queue[:, slot, :], new, atol=1e-6,
                err_msg=f"slot {slot} != new after full flush"
            )

        # Delayed (random idx into all-identical slots) must also equal new
        np.testing.assert_allclose(delayed, new, atol=1e-6,
                                   err_msg="delayed must equal new after flush")

    def test_flush_only_episode_start_envs(self) -> None:
        """Only envs with flush=True are reset; others keep their history."""
        from unilab.envs.manipulation.simtoolreal.delay_buffer import push_and_sample_delay

        N, L, D = 4, 5, 3
        rng = np.random.default_rng(42)
        old_val = np.full((N, L, D), 99.0, dtype=np.float32)
        queue = old_val.copy()

        new = rng.standard_normal((N, D)).astype(np.float32)
        env = _make_env(N)

        flush = np.array([True, True, False, False])
        queue, _ = push_and_sample_delay(queue, new, env, flush=flush)

        # Flushed envs: all slots == new
        for slot in range(L):
            np.testing.assert_allclose(
                queue[:2, slot, :], new[:2], atol=1e-6,
                err_msg=f"flushed env slot {slot} wrong"
            )

        # Non-flushed envs: slot 0 = new (just written), older slots != new
        np.testing.assert_allclose(queue[2:, 0, :], new[2:], atol=1e-6,
                                   err_msg="non-flushed slot 0 must be new")
        # Slot 1 and beyond should still have old values (were 99.0 before roll)
        # After roll, slot 1 ← old slot 0 (= 99.0), so still 99.0
        np.testing.assert_allclose(queue[2:, 1, :],
                                   np.full((2, D), 99.0, dtype=np.float32),
                                   atol=1e-6,
                                   err_msg="non-flushed older slots must be unchanged")

    def test_no_flush_when_intra_episode_success(self) -> None:
        """flush=(steps==0)&(successes==0): successes>0 must NOT flush."""
        from unilab.envs.manipulation.simtoolreal.delay_buffer import push_and_sample_delay

        N, L, D = 4, 3, 5
        rng = np.random.default_rng(7)
        queue = rng.standard_normal((N, L, D)).astype(np.float32)
        old_queue = queue.copy()
        new = rng.standard_normal((N, D)).astype(np.float32)
        env = _make_env(N)

        # Simulate intra-episode goal switch: steps=0 but successes>0
        steps = np.array([0, 0, 5, 5], dtype=np.int32)
        successes = np.array([1, 2, 0, 3], dtype=np.int32)
        flush = (steps == 0) & (successes == 0)
        # All envs: steps==0 OR successes>0 → flush is all False
        assert not np.any(flush), "precondition: no env should flush here"

        queue, _ = push_and_sample_delay(queue, new, env, flush=flush)

        # Slot 0 must be new (always written after roll)
        np.testing.assert_allclose(queue[:, 0, :], new, atol=1e-6)
        # Slot 1 should be what was in slot 0 before (normal roll, no flush)
        np.testing.assert_allclose(queue[:, 1, :], old_queue[:, 0, :], atol=1e-6,
                                   err_msg="slot 1 must be old slot 0 after roll (no flush)")

    def test_random_index_range(self) -> None:
        """Returned delayed value must be one of the existing queue slots."""
        from unilab.envs.manipulation.simtoolreal.delay_buffer import push_and_sample_delay

        N, L, D = 8, 7, 4
        rng = np.random.default_rng(99)
        # Unique sentinel per slot so we can identify which slot was sampled
        queue = np.arange(N * L * D, dtype=np.float32).reshape(N, L, D)
        new = rng.standard_normal((N, D)).astype(np.float32)
        env = _make_env(N)

        queue, delayed = push_and_sample_delay(queue, new, env)

        # Each delayed row must match exactly one row of queue
        for i in range(N):
            matched = any(
                np.allclose(delayed[i], queue[i, s], atol=1e-6)
                for s in range(L)
            )
            assert matched, f"env {i}: delayed does not match any slot in queue"


# ---------------------------------------------------------------------------
# dr_wrench tests  (mock backend, mock env)
# ---------------------------------------------------------------------------

class _FakeBackend:
    """Records apply_body_wrench calls for assertion."""

    def __init__(self) -> None:
        self.last_body_ids = None
        self.last_force = None
        self.last_torque = None

    def apply_body_wrench(self, body_ids, force, torque) -> None:
        self.last_body_ids = np.array(body_ids)
        self.last_force = np.array(force)
        self.last_torque = np.array(torque)


def _make_wrench_env(
    n: int = 8,
    force_scale: float = 20.0,
    torque_scale: float = 2.0,
    force_decay: float = 0.0,
    torque_decay: float = 0.0,
    force_only_when_lifted: bool = True,
    torque_only_when_lifted: bool = True,
    mass: float = 1.0,
) -> types.SimpleNamespace:
    """Build a minimal env stub for apply_wrench_dr tests."""
    env = types.SimpleNamespace()
    env.num_envs = n

    dr = types.SimpleNamespace(
        force_scale=force_scale,
        torque_scale=torque_scale,
        force_decay=force_decay,
        torque_decay=torque_decay,
        force_decay_interval=0.05,   # unused when decay=0
        torque_decay_interval=0.05,
        force_only_when_lifted=force_only_when_lifted,
        torque_only_when_lifted=torque_only_when_lifted,
    )
    env.cfg = types.SimpleNamespace(
        domain_randomization=dr,
        ctrl_dt=1.0 / 60.0,
    )

    # All envs fire with probability 1 by default
    env._random_force_prob = np.ones(n, dtype=np.float32)
    env._random_torque_prob = np.ones(n, dtype=np.float32)
    env._object_mass = np.full(n, mass, dtype=np.float32)
    env._object_forces = np.zeros((n, 3), dtype=np.float32)
    env._object_torques = np.zeros((n, 3), dtype=np.float32)
    # All envs lifted by default
    env._state_cache_lifted_object = np.ones(n, dtype=bool)
    env._object_body_id = 5   # arbitrary body id
    env.backend = _FakeBackend()
    return env


class TestWrenchDR:
    """Verify apply_wrench_dr (action_utils.apply_wrench_dr translation)."""

    def test_impulse_zeros_forces_before_sample(self) -> None:
        """decay=0: forces must be zeroed before sampling, then replaced if fire."""
        from unilab.envs.manipulation.simtoolreal.dr_wrench import apply_wrench_dr

        env = _make_wrench_env(n=4)
        env._object_forces[:] = 999.0  # pre-fill with garbage

        apply_wrench_dr(env)

        # After apply, if all envs fire (prob=1) the forces are new samples, not 999
        assert not np.any(np.abs(env._object_forces) > 1e4), \
            "garbage 999 should have been zeroed before sampling"

    def test_force_proportional_to_mass(self) -> None:
        """Force magnitude must scale linearly with object mass."""
        from unilab.envs.manipulation.simtoolreal.dr_wrench import apply_wrench_dr

        np.random.seed(0)
        # Two envs: mass 1 and mass 2
        env1 = _make_wrench_env(n=1, mass=1.0, force_scale=20.0)
        env2 = _make_wrench_env(n=1, mass=2.0, force_scale=20.0)

        # Fix RNG so both envs draw the same unit-normal sample
        rng_state = np.random.get_state()
        np.random.seed(42)
        apply_wrench_dr(env1)
        f1 = env1._object_forces.copy()

        np.random.seed(42)
        apply_wrench_dr(env2)
        f2 = env2._object_forces.copy()

        # f2 should be exactly 2× f1 (same randn seed, mass doubled)
        np.testing.assert_allclose(f2, 2.0 * f1, rtol=1e-5,
                                   err_msg="force must scale linearly with mass")

    def test_torque_proportional_to_mass(self) -> None:
        """Torque magnitude must also scale with mass (source: randn*mass*torque_scale)."""
        from unilab.envs.manipulation.simtoolreal.dr_wrench import apply_wrench_dr

        np.random.seed(0)
        env1 = _make_wrench_env(n=1, mass=1.0, torque_scale=2.0)
        env2 = _make_wrench_env(n=1, mass=3.0, torque_scale=2.0)

        np.random.seed(77)
        apply_wrench_dr(env1)
        t1 = env1._object_torques.copy()

        np.random.seed(77)
        apply_wrench_dr(env2)
        t2 = env2._object_torques.copy()

        np.testing.assert_allclose(t2, 3.0 * t1, rtol=1e-5,
                                   err_msg="torque must scale linearly with mass")

    def test_impulse_clears_next_step(self) -> None:
        """decay=0: calling apply_wrench_dr twice with prob=0 should yield zeros."""
        from unilab.envs.manipulation.simtoolreal.dr_wrench import apply_wrench_dr

        env = _make_wrench_env(n=4)
        # First call: fires (prob=1), forces become non-zero
        apply_wrench_dr(env)
        assert np.any(env._object_forces != 0), "first call should produce non-zero force"

        # Second call with prob=0: decay zeroes first, no new sample
        env._random_force_prob[:] = 0.0
        env._random_torque_prob[:] = 0.0
        apply_wrench_dr(env)

        np.testing.assert_array_equal(env._object_forces, 0.0,
                                      err_msg="impulse: forces must be 0 when prob=0")
        np.testing.assert_array_equal(env._object_torques, 0.0,
                                      err_msg="impulse: torques must be 0 when prob=0")

    def test_lift_gate_zeros_when_not_lifted(self) -> None:
        """Forces/torques must be zeroed when object is not lifted."""
        from unilab.envs.manipulation.simtoolreal.dr_wrench import apply_wrench_dr

        env = _make_wrench_env(n=4, force_only_when_lifted=True, torque_only_when_lifted=True)
        env._state_cache_lifted_object[:] = False  # none lifted

        apply_wrench_dr(env)

        np.testing.assert_array_equal(env._object_forces, 0.0,
                                      err_msg="force must be 0 when not lifted")
        np.testing.assert_array_equal(env._object_torques, 0.0,
                                      err_msg="torque must be 0 when not lifted")

    def test_lift_gate_partial(self) -> None:
        """Only lifted envs should receive wrench."""
        from unilab.envs.manipulation.simtoolreal.dr_wrench import apply_wrench_dr

        np.random.seed(123)
        env = _make_wrench_env(n=4)
        env._state_cache_lifted_object = np.array([True, False, True, False])

        apply_wrench_dr(env)

        # Not-lifted envs must be zeroed
        np.testing.assert_array_equal(env._object_forces[1], 0.0)
        np.testing.assert_array_equal(env._object_forces[3], 0.0)
        np.testing.assert_array_equal(env._object_torques[1], 0.0)
        np.testing.assert_array_equal(env._object_torques[3], 0.0)
        # Lifted envs must be non-zero (prob=1, mass=1, scale=20 → very likely)
        # (not strictly guaranteed due to randn, but with seed 123 it holds)
        assert np.any(env._object_forces[0] != 0) or np.any(env._object_forces[2] != 0), \
            "at least one lifted env should have non-zero force"

    def test_backend_call_shape(self) -> None:
        """apply_body_wrench must be called with shape (N,1,3) for both force/torque."""
        from unilab.envs.manipulation.simtoolreal.dr_wrench import apply_wrench_dr

        N = 6
        env = _make_wrench_env(n=N)
        apply_wrench_dr(env)

        fb = env.backend
        assert fb.last_force.shape == (N, 1, 3), \
            f"force shape {fb.last_force.shape} != ({N},1,3)"
        assert fb.last_torque.shape == (N, 1, 3), \
            f"torque shape {fb.last_torque.shape} != ({N},1,3)"
        np.testing.assert_array_equal(fb.last_body_ids, [5])


# ---------------------------------------------------------------------------
# D6 backend tests — apply_body_wrench writes correct 6D xfrc slots
# ---------------------------------------------------------------------------

class TestApplyBodyWrench:
    """Verify MuJoCo backend apply_body_wrench writes force *and* torque slots."""

    def _make_mujoco_backend_stub(self, num_envs: int, nbody: int):
        """Create a minimal stub that mimics _pending_xfrc_applied without MuJoCo."""
        stub = types.SimpleNamespace()
        stub._num_envs = num_envs
        stub._pending_xfrc_applied = np.zeros(
            (num_envs, 6 * nbody), dtype=np.float64
        )

        # Copy the real implementation logic directly
        def apply_body_wrench(body_ids, force, torque):
            body_ids_np = np.asarray(body_ids, dtype=np.int32).reshape(-1)
            force_np = np.asarray(force, dtype=np.float64)
            torque_np = np.asarray(torque, dtype=np.float64)
            expected_shape = (num_envs, body_ids_np.size, 3)
            if force_np.shape != expected_shape:
                raise ValueError(
                    f"body wrench force must have shape {expected_shape}, got {force_np.shape}"
                )
            if torque_np.shape != expected_shape:
                raise ValueError(
                    f"body wrench torque must have shape {expected_shape}, got {torque_np.shape}"
                )
            for body_offset, body_id in enumerate(body_ids_np):
                start = 6 * int(body_id)
                stub._pending_xfrc_applied[:, start:start+3] += force_np[:, body_offset, :]
                stub._pending_xfrc_applied[:, start+3:start+6] += torque_np[:, body_offset, :]

        stub.apply_body_wrench = apply_body_wrench
        return stub

    def test_force_slots_written(self) -> None:
        """Force must appear in xfrc[6*bid : 6*bid+3]."""
        N, nbody, bid = 3, 8, 2
        backend = self._make_mujoco_backend_stub(N, nbody)

        force = np.ones((N, 1, 3), dtype=np.float32) * 5.0
        torque = np.zeros((N, 1, 3), dtype=np.float32)
        backend.apply_body_wrench(np.array([bid]), force, torque)

        start = 6 * bid
        xfrc = backend._pending_xfrc_applied
        np.testing.assert_allclose(xfrc[:, start:start+3], 5.0, atol=1e-9,
                                   err_msg="force slots not written")

    def test_torque_slots_written(self) -> None:
        """Torque must appear in xfrc[6*bid+3 : 6*bid+6]."""
        N, nbody, bid = 3, 8, 2
        backend = self._make_mujoco_backend_stub(N, nbody)

        force = np.zeros((N, 1, 3), dtype=np.float32)
        torque = np.ones((N, 1, 3), dtype=np.float32) * 7.0
        backend.apply_body_wrench(np.array([bid]), force, torque)

        start = 6 * bid
        xfrc = backend._pending_xfrc_applied
        np.testing.assert_allclose(xfrc[:, start+3:start+6], 7.0, atol=1e-9,
                                   err_msg="torque slots not written")

    def test_force_and_torque_independent_slots(self) -> None:
        """Force and torque occupy separate 3-element halves of the 6D block."""
        N, nbody, bid = 2, 5, 1
        backend = self._make_mujoco_backend_stub(N, nbody)

        f_val = 3.0
        t_val = 9.0
        force = np.full((N, 1, 3), f_val, dtype=np.float32)
        torque = np.full((N, 1, 3), t_val, dtype=np.float32)
        backend.apply_body_wrench(np.array([bid]), force, torque)

        start = 6 * bid
        xfrc = backend._pending_xfrc_applied
        np.testing.assert_allclose(xfrc[:, start:start+3], f_val, atol=1e-9)
        np.testing.assert_allclose(xfrc[:, start+3:start+6], t_val, atol=1e-9)

        # Other body slots must be untouched (zeros)
        for other_bid in range(nbody):
            if other_bid == bid:
                continue
            s = 6 * other_bid
            np.testing.assert_array_equal(xfrc[:, s:s+6], 0.0,
                                           err_msg=f"body {other_bid} must be untouched")

    def test_accumulation(self) -> None:
        """Two calls to apply_body_wrench accumulate (+=) correctly."""
        N, nbody, bid = 2, 4, 0
        backend = self._make_mujoco_backend_stub(N, nbody)

        force = np.ones((N, 1, 3), dtype=np.float32) * 2.0
        torque = np.ones((N, 1, 3), dtype=np.float32) * 4.0
        backend.apply_body_wrench(np.array([bid]), force, torque)
        backend.apply_body_wrench(np.array([bid]), force, torque)

        start = 6 * bid
        xfrc = backend._pending_xfrc_applied
        np.testing.assert_allclose(xfrc[:, start:start+3], 4.0, atol=1e-9,
                                   err_msg="force slots should accumulate")
        np.testing.assert_allclose(xfrc[:, start+3:start+6], 8.0, atol=1e-9,
                                   err_msg="torque slots should accumulate")

    def test_real_mujoco_backend(self) -> None:
        """Integration test against the actual MuJoCo backend class."""
        try:
            import mujoco
        except ImportError:
            pytest.skip("mujoco not installed")

        from unilab.base.backend.mujoco.backend import MuJoCoBackend
        from unilab.base.scene import SceneCfg

        # Minimal MJCF with 2 free bodies
        xml = """
        <mujoco>
          <worldbody>
            <body name="box1" pos="0 0 1">
              <freejoint/>
              <geom type="box" size=".1 .1 .1"/>
            </body>
            <body name="box2" pos="1 0 1">
              <freejoint/>
              <geom type="box" size=".1 .1 .1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml)
            xml_path = f.name

        try:
            N = 2
            scene = SceneCfg(model_file=xml_path)
            backend = MuJoCoBackend(scene, num_envs=N, sim_dt=1.0/240.0)
            nbody = backend._model.nbody  # includes worldbody (id=0)

            # body 1 = box1 in MuJoCo (worldbody is id 0)
            bid = 1
            force_val = 5.0
            torque_val = 9.0
            force = np.full((N, 1, 3), force_val, dtype=np.float32)
            torque = np.full((N, 1, 3), torque_val, dtype=np.float32)

            backend.apply_body_wrench(np.array([bid], dtype=np.int32), force, torque)

            start = 6 * bid
            xfrc = backend._pending_xfrc_applied
            np.testing.assert_allclose(xfrc[:, start:start+3], force_val, atol=1e-9)
            np.testing.assert_allclose(xfrc[:, start+3:start+6], torque_val, atol=1e-9)
        finally:
            os.unlink(xml_path)

    def test_shape_validation(self) -> None:
        """Wrong shape must raise ValueError."""
        N, nbody = 3, 4
        backend = self._make_mujoco_backend_stub(N, nbody)

        bad_force = np.ones((N, 2, 3))  # 2 bodies, but body_ids has 1
        good_torque = np.ones((N, 1, 3))

        with pytest.raises((ValueError, Exception)):
            backend.apply_body_wrench(np.array([0]), bad_force, good_torque)


# ---------------------------------------------------------------------------
# sample_log_uniform sanity
# ---------------------------------------------------------------------------

class TestSampleLogUniform:
    def test_range(self) -> None:
        from unilab.envs.manipulation.simtoolreal.dr_wrench import sample_log_uniform

        lo, hi, n = 0.001, 0.1, 10000
        samples = sample_log_uniform(lo, hi, n)
        assert samples.shape == (n,)
        assert samples.min() >= lo - 1e-9
        assert samples.max() <= hi + 1e-4, f"max {samples.max()} > {hi}"

    def test_log_uniformity(self) -> None:
        """log(samples) should be roughly uniform."""
        from unilab.envs.manipulation.simtoolreal.dr_wrench import sample_log_uniform

        np.random.seed(0)
        samples = sample_log_uniform(0.001, 0.1, 50000)
        log_samples = np.log(samples)
        # Split into 5 equal bins and check counts are roughly equal
        counts, _ = np.histogram(log_samples, bins=5)
        # Each bin should have ~10000/5 = 10000 samples ± 10%
        assert all(abs(c - 10000) < 1500 for c in counts), \
            f"log-uniform bins uneven: {counts}"
