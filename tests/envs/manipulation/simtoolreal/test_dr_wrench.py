from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from unilab.envs.manipulation.simtoolreal.dr_wrench import apply_wrench_dr, sample_log_uniform


class _Backend:
    def __init__(self) -> None:
        self.calls = []

    def apply_body_wrench(self, body_ids, force, torque) -> None:
        self.calls.append((body_ids.copy(), force.copy(), torque.copy()))


def _env() -> SimpleNamespace:
    n = 2
    return SimpleNamespace(
        num_envs=n,
        cfg=SimpleNamespace(
            ctrl_dt=1.0 / 60.0,
            domain_randomization=SimpleNamespace(
                force_decay=0.0,
                torque_decay=0.0,
                force_decay_interval=0.08,
                torque_decay_interval=0.08,
                force_scale=20.0,
                torque_scale=2.0,
                force_only_when_lifted=True,
                torque_only_when_lifted=True,
            ),
        ),
        _object_forces=np.zeros((n, 3), dtype=np.float32),
        _object_torques=np.zeros((n, 3), dtype=np.float32),
        _random_force_prob=np.ones(n, dtype=np.float32),
        _random_torque_prob=np.ones(n, dtype=np.float32),
        _object_mass=np.array([2.0, 3.0], dtype=np.float32),
        _state_cache_lifted_object=np.array([True, False]),
        _object_body_id=7,
        backend=_Backend(),
    )


def test_log_uniform_range_is_finite_and_bounded() -> None:
    values = sample_log_uniform(0.001, 0.1, 100)
    assert values.dtype == np.float32
    assert np.all((values >= 0.001) & (values <= 0.1))


def test_wrench_uses_mass_scale_lift_gate_and_public_backend_call(monkeypatch) -> None:
    env = _env()
    monkeypatch.setattr(np.random, "random", lambda n: np.zeros(n, dtype=np.float32))
    monkeypatch.setattr(np.random, "randn", lambda *shape: np.ones(shape, dtype=np.float32))
    apply_wrench_dr(env)
    assert len(env.backend.calls) == 1
    body_ids, force, torque = env.backend.calls[0]
    np.testing.assert_array_equal(body_ids, [7])
    assert force.shape == torque.shape == (2, 1, 3)
    np.testing.assert_allclose(force[:, 0], [[40, 40, 40], [0, 0, 0]])
    np.testing.assert_allclose(torque[:, 0], [[4, 4, 4], [0, 0, 0]])
