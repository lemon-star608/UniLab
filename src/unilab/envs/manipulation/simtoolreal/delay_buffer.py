"""Delay queue for action / observation / object-state DR.

Faithful numpy translation of SimToolReal ``_sample_delay``
(``obs_utils.py:119-133``).  The contract signature is frozen in
``MIGRATION_01_INTERFACE_CONTRACT.md §4.2``.

:func:`push_and_sample_delay_rows` is the row-restricted sibling used by the
reset observation path; the frozen full-batch signature is untouched.
"""

from __future__ import annotations

import numpy as np


def push_and_sample_delay(
    queue: np.ndarray,
    new: np.ndarray,
    env,
    flush: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Push *new* into a rolling delay queue and return a randomly-sampled
    delayed value.

    Translates ``SimToolReal obs_utils._sample_delay`` (torch→numpy).

    Args:
        queue:  ``(N, L, D)`` rolling buffer; modified and returned.
        new:    ``(N, D)`` current frame to enqueue.
        env:    Environment instance (used for ``env.num_envs``).
        flush:  ``(N,)`` bool array.  Envs where ``flush`` is True have ALL L
                slots overwritten with *new* before the roll.  Use only for
                fully new episodes — ``flush = (steps == 0) & (successes == 0)``
                — **not** for intra-episode goal switches.

    Returns:
        ``(updated_queue, delayed)`` where ``delayed`` has shape ``(N, D)``.
        The caller must assign the returned queue back to the stored attribute::

            env._obs_queue, delayed = push_and_sample_delay(
                env._obs_queue, frame, env,
                flush=(info["steps"] == 0) & (info["successes"] == 0),
            )

    Notes:
        * Slot 0 always holds the *newest* value after the call.
        * Random index ``idx ∈ [0, L)``; delay range is ``[0, L-1]`` steps,
          i.e., the maximum achievable delay is ``L-1`` (delay_max exclusive).
        * Flush is applied **before** the roll so that even after filling, slot 0
          gets the current value and the queue is consistent.
    """
    # Derive N from the queue itself so the function works correctly with both
    # real env objects (env.num_envs property) and test mock objects that only
    # set env._num_envs.  The contract keeps `env` in the signature for parity
    # with the public contract; batch size comes from the queue itself.
    n_envs = queue.shape[0]
    L = queue.shape[1]

    # -- flush: overwrite all L slots of reset envs with the current value ----
    # Mirrors: queue[flush] = values[flush].unsqueeze(1).expand(-1, L, -1)
    if flush is not None and np.any(flush):
        # new[flush] shape (k, D) → (k, 1, D) → broadcast to (k, L, D)
        queue[flush] = new[flush, np.newaxis, :]  # broadcasts over L

    # -- roll: shift slots forward (slot i → slot i+1, slot L-1 wraps to 0) --
    # np.roll(axis=1, shift=1): queue[:, 1] ← queue[:, 0], ...,
    #                            queue[:, 0] ← queue[:, L-1]
    queue = np.roll(queue, shift=1, axis=1)

    # -- write newest value into slot 0 ----------------------------------------
    queue[:, 0, :] = new

    # -- sample a per-env random delay index in [0, L) -------------------------
    idx = np.random.randint(0, L, size=(n_envs,))
    delayed = queue[np.arange(n_envs), idx]  # (N, D)

    return queue, delayed


def push_and_sample_delay_rows(
    queue: np.ndarray,
    new: np.ndarray,
    rows: np.ndarray,
    flush: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-restricted :func:`push_and_sample_delay`.

    Same semantics as :func:`push_and_sample_delay`, applied to ``rows`` only:
    rows outside the selection keep their queue contents and get no sample. Used
    by the reset observation path, which must build observations for the envs
    being reset without disturbing the delay history of the envs that keep
    running (their frame for this step was already pushed by ``update_state``).

    Args:
        queue: ``(N, L, D)`` rolling buffer. The selected rows are updated in
            place; the same array is returned for call-site symmetry with
            :func:`push_and_sample_delay`.
        new:   ``(k, D)`` current frame for the selected rows, in ``rows`` order.
        rows:  ``(k,)`` env indices to update.
        flush: Optional ``(k,)`` bool mask, aligned to ``rows``. Rows where it is
            True have all ``L`` slots overwritten with ``new`` before the roll,
            so their sampled value is ``new`` regardless of the drawn delay.

    Returns:
        ``(queue, delayed)`` where ``delayed`` has shape ``(k, D)``.

    Notes:
        Vectorized: cost is ``O(k * L * D)`` with no per-env Python loop, so the
        reset path stays flat in ``num_envs``.
    """
    row_idx = np.asarray(rows, dtype=np.intp)
    k = int(row_idx.shape[0])
    L = queue.shape[1]

    # Fancy indexing copies, so mutate the copy and scatter it back.
    sub = queue[row_idx]  # (k, L, D)

    if flush is not None and np.any(flush):
        sub[flush] = new[flush, np.newaxis, :]

    sub = np.roll(sub, shift=1, axis=1)
    sub[:, 0, :] = new
    queue[row_idx] = sub

    idx = np.random.randint(0, L, size=(k,))
    delayed = sub[np.arange(k), idx]  # (k, D)

    return queue, delayed
