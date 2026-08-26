from __future__ import annotations

import numpy as np

from unilab.envs.manipulation.simtoolreal.delay_buffer import (
    push_and_sample_delay,
    push_and_sample_delay_rows,
)


def test_delay_queue_roll_and_per_row_sampling(monkeypatch) -> None:
    queue = np.zeros((2, 3, 1), dtype=np.float32)
    queue[0, :, 0] = (1, 2, 3)
    queue[1, :, 0] = (4, 5, 6)
    monkeypatch.setattr(np.random, "randint", lambda low, high, size: np.array([2, 0]))
    updated, delayed = push_and_sample_delay(
        queue, np.array([[9], [8]], dtype=np.float32), object(), flush=np.array([True, False])
    )
    np.testing.assert_array_equal(updated[:, :, 0], [[9, 9, 9], [8, 4, 5]])
    np.testing.assert_array_equal(delayed[:, 0], [9, 8])

    monkeypatch.setattr(np.random, "randint", lambda low, high, size: np.array([1]))
    _, selected = push_and_sample_delay_rows(
        updated, np.array([[7]], dtype=np.float32), np.array([1]), flush=np.array([False])
    )
    np.testing.assert_array_equal(updated[0, :, 0], [9, 9, 9])
    np.testing.assert_array_equal(selected[:, 0], [8])


def test_delay_buffer_rejects_aliasing_by_returning_new_queue() -> None:
    queue = np.zeros((1, 2, 1), dtype=np.float32)
    updated, _ = push_and_sample_delay(queue, np.ones((1, 1), dtype=np.float32), object())
    assert updated is not queue
