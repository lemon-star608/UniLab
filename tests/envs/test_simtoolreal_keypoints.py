"""T5 acceptance tests for the SimToolReal keypoint geometry.

All pure numpy — no backend, no MuJoCo, no generated assets. Covers the task-card
criteria:

* output shape ``(N, 4, 3)`` and float32 dtype,
* identity quat + zero pos reproduces ``corners * size * 0.5 * 1.5``,
* ``keypoint_max_dist`` is symmetric and non-negative,
* both keypoint sizes are correct (per-object ``phi`` vs the fixed reward size),
* ``size`` accepts ``(3,)`` and ``(N, 3)``.

Reference numbers cite the SimToolReal source; the full source index lives in the
docstring of ``unilab.envs.manipulation.simtoolreal.keypoints``.
"""

from __future__ import annotations

import numpy as np
import pytest

from unilab.envs.manipulation.simtoolreal.config import SimToolRealCfg
from unilab.envs.manipulation.simtoolreal.constants import KEYPOINT_CORNERS, NUM_KEYPOINTS
from unilab.envs.manipulation.simtoolreal.keypoints import (
    DEFAULT_KEYPOINT_SCALE,
    compute_keypoints,
    compute_keypoints_from_offsets,
    keypoint_max_dist,
    keypoint_offsets,
    observation_keypoint_size,
)
from unilab.utils.rotation import np_matrix_from_quat

CORNERS = np.asarray(KEYPOINT_CORNERS, dtype=np.float32)
IDENTITY_QUAT = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _random_quats(n: int, seed: int = 0) -> np.ndarray:
    """Return ``n`` unit quaternions (wxyz), float32."""
    rng = np.random.default_rng(seed)
    quat = rng.standard_normal((n, 4)).astype(np.float32)
    return np.asarray(quat / np.linalg.norm(quat, axis=-1, keepdims=True), dtype=np.float32)


def test_corner_order_and_scale_match_source() -> None:
    """Corner order (obs_utils.py:28-33) and keypoint_scale (cfg:337) are verbatim."""
    assert KEYPOINT_CORNERS == ((1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1))
    assert NUM_KEYPOINTS == 4
    assert DEFAULT_KEYPOINT_SCALE == pytest.approx(SimToolRealCfg().goal.keypoint_scale)
    assert DEFAULT_KEYPOINT_SCALE == pytest.approx(1.5)


def test_output_shape_and_dtype() -> None:
    """compute_keypoints returns (N, 4, 3) float32 (decision D0)."""
    n_envs = 6
    pos = np.zeros((n_envs, 3), dtype=np.float32)
    quat = _random_quats(n_envs)
    size = np.full((n_envs, 3), 0.1, dtype=np.float32)

    keypoints = compute_keypoints(pos, quat, size)

    assert keypoints.shape == (n_envs, NUM_KEYPOINTS, 3)
    assert keypoints.dtype == np.float32

    # float64 inputs must not leak a float64 result (decision D0).
    keypoints64 = compute_keypoints(
        pos.astype(np.float64), quat.astype(np.float64), size.astype(np.float64)
    )
    assert keypoints64.dtype == np.float32
    assert keypoint_max_dist(keypoints64, keypoints).dtype == np.float32


def test_identity_quat_zero_pos_equals_scaled_corners() -> None:
    """kp == corners * size * 0.5 * 1.5 at identity pose (reset_utils.py:74-83)."""
    n_envs = 3
    pos = np.zeros((n_envs, 3), dtype=np.float32)
    quat = np.broadcast_to(IDENTITY_QUAT, (n_envs, 4)).copy()
    size = np.asarray([0.141, 0.03025, 0.0271], dtype=np.float32)

    keypoints = compute_keypoints(pos, quat, size)

    expected = CORNERS * (size * 0.5 * 1.5)
    for env_index in range(n_envs):
        np.testing.assert_allclose(keypoints[env_index], expected, rtol=1e-6, atol=1e-7)


def test_size_accepts_shared_and_per_env_shapes() -> None:
    """A (3,) size behaves exactly like the same size tiled to (N, 3)."""
    n_envs = 4
    pos = np.asarray(np.arange(n_envs * 3, dtype=np.float32).reshape(n_envs, 3))
    quat = _random_quats(n_envs, seed=1)
    size_shared = np.asarray([0.2, 0.05, 0.03], dtype=np.float32)
    size_per_env = np.broadcast_to(size_shared, (n_envs, 3)).copy()

    np.testing.assert_allclose(
        compute_keypoints(pos, quat, size_shared),
        compute_keypoints(pos, quat, size_per_env),
        rtol=1e-6,
        atol=1e-7,
    )

    assert keypoint_offsets(size_shared).shape == (NUM_KEYPOINTS, 3)
    assert keypoint_offsets(size_per_env).shape == (n_envs, NUM_KEYPOINTS, 3)


def test_per_env_sizes_are_applied_independently() -> None:
    """Each env's keypoints scale with that env's own size, not a shared one."""
    pos = np.zeros((2, 3), dtype=np.float32)
    quat = np.broadcast_to(IDENTITY_QUAT, (2, 4)).copy()
    size = np.asarray([[0.1, 0.1, 0.1], [0.4, 0.4, 0.4]], dtype=np.float32)

    keypoints = compute_keypoints(pos, quat, size)

    # Env 1's size is 4x env 0's, so every keypoint is 4x further out.
    np.testing.assert_allclose(keypoints[1], 4.0 * keypoints[0], rtol=1e-6, atol=1e-7)


def test_rotation_matches_rotation_matrix_reference() -> None:
    """kp == R(quat) @ offset + pos, cross-checked against np_matrix_from_quat."""
    n_envs = 8
    rng = np.random.default_rng(7)
    pos = rng.standard_normal((n_envs, 3)).astype(np.float32)
    quat = _random_quats(n_envs, seed=3)
    size = np.abs(rng.standard_normal((n_envs, 3)).astype(np.float32)) + 0.01

    keypoints = compute_keypoints(pos, quat, size)

    offsets = keypoint_offsets(size)
    matrices = np_matrix_from_quat(quat)  # (N, 3, 3)
    expected = pos[:, None, :] + np.einsum("nij,nkj->nki", matrices, offsets)

    np.testing.assert_allclose(keypoints, expected, rtol=1e-5, atol=1e-6)


def test_translation_is_a_pure_offset() -> None:
    """Translating the pose translates every keypoint by the same vector."""
    n_envs = 3
    quat = _random_quats(n_envs, seed=5)
    size = np.asarray([0.15, 0.04, 0.03], dtype=np.float32)
    shift = np.asarray([0.3, -0.2, 1.1], dtype=np.float32)

    base = compute_keypoints(np.zeros((n_envs, 3), dtype=np.float32), quat, size)
    shifted = compute_keypoints(np.broadcast_to(shift, (n_envs, 3)).copy(), quat, size)

    np.testing.assert_allclose(
        shifted - base,
        np.broadcast_to(shift, (n_envs, NUM_KEYPOINTS, 3)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_rotation_preserves_keypoint_radii() -> None:
    """A rotation moves keypoints on a sphere: |kp - pos| is rotation-invariant."""
    n_envs = 5
    size = np.asarray([0.141, 0.03025, 0.0271], dtype=np.float32)
    pos = np.zeros((n_envs, 3), dtype=np.float32)

    identity = compute_keypoints(pos, np.broadcast_to(IDENTITY_QUAT, (n_envs, 4)).copy(), size)
    rotated = compute_keypoints(pos, _random_quats(n_envs, seed=11), size)

    np.testing.assert_allclose(
        np.linalg.norm(rotated, axis=-1),
        np.linalg.norm(identity, axis=-1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_from_offsets_matches_compute_keypoints() -> None:
    """The cached-offset entry point agrees with the size-taking one."""
    n_envs = 4
    rng = np.random.default_rng(13)
    pos = rng.standard_normal((n_envs, 3)).astype(np.float32)
    quat = _random_quats(n_envs, seed=13)
    size = np.asarray([0.141, 0.03025, 0.0271], dtype=np.float32)

    offsets_shared = keypoint_offsets(size)  # (4, 3), as T0 caches it
    np.testing.assert_allclose(
        compute_keypoints_from_offsets(pos, quat, offsets_shared),
        compute_keypoints(pos, quat, size),
        rtol=1e-6,
        atol=1e-7,
    )

    # A (N, 4, 3) offset array is accepted too (the source shape,
    # reset_utils.py:83 expands the fixed offsets per env).
    offsets_per_env = np.broadcast_to(offsets_shared, (n_envs, NUM_KEYPOINTS, 3)).copy()
    np.testing.assert_allclose(
        compute_keypoints_from_offsets(pos, quat, offsets_per_env),
        compute_keypoints(pos, quat, size),
        rtol=1e-6,
        atol=1e-7,
    )


def test_max_dist_shape_symmetry_and_sign() -> None:
    """d(o, g) is (N,), symmetric, non-negative, and zero iff the poses agree."""
    n_envs = 6
    rng = np.random.default_rng(17)
    size = np.asarray([0.141, 0.03025, 0.0271], dtype=np.float32)

    obj_kp = compute_keypoints(
        rng.standard_normal((n_envs, 3)).astype(np.float32), _random_quats(n_envs, seed=17), size
    )
    goal_kp = compute_keypoints(
        rng.standard_normal((n_envs, 3)).astype(np.float32), _random_quats(n_envs, seed=18), size
    )

    dist = keypoint_max_dist(obj_kp, goal_kp)

    assert dist.shape == (n_envs,)
    assert dist.dtype == np.float32
    assert np.all(dist >= 0.0)
    # Swapping the arguments is the same metric.
    np.testing.assert_allclose(dist, keypoint_max_dist(goal_kp, obj_kp), rtol=0, atol=0)
    # Identical keypoint sets give exactly zero.
    np.testing.assert_array_equal(
        keypoint_max_dist(obj_kp, obj_kp), np.zeros(n_envs, dtype=np.float32)
    )


def test_max_dist_takes_the_worst_corner_not_the_mean() -> None:
    """Eq. 2 is a max over the four corners (obs_utils.py:180)."""
    obj_kp = np.zeros((1, NUM_KEYPOINTS, 3), dtype=np.float32)
    goal_kp = np.zeros((1, NUM_KEYPOINTS, 3), dtype=np.float32)
    # Corner 2 is off by 0.5 m, the rest coincide.
    goal_kp[0, 2, 0] = 0.5

    dist = keypoint_max_dist(obj_kp, goal_kp)

    np.testing.assert_allclose(dist, [0.5], rtol=0, atol=1e-7)
    # A mean would have given 0.125 — guard against that regression.
    assert dist[0] > 0.125


def test_pure_translation_distance_equals_translation_norm() -> None:
    """With no relative rotation, every corner is off by the position delta."""
    n_envs = 2
    quat = _random_quats(n_envs, seed=19)
    size = np.asarray([0.141, 0.03025, 0.0271], dtype=np.float32)
    delta = np.asarray([0.03, -0.04, 0.12], dtype=np.float32)

    obj_kp = compute_keypoints(np.zeros((n_envs, 3), dtype=np.float32), quat, size)
    goal_kp = compute_keypoints(np.broadcast_to(delta, (n_envs, 3)).copy(), quat, size)

    np.testing.assert_allclose(
        keypoint_max_dist(obj_kp, goal_kp),
        np.full(n_envs, float(np.linalg.norm(delta)), dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )


def test_reward_side_size_matches_t0_cached_offsets() -> None:
    """The fixed reward size reproduces T0's ``_keypoint_offsets_fixed`` formula.

    T0 builds it as ``corners * 0.5 * keypoint_scale * cfg.reward.fixed_size``
    (env.py ``_build_geometry_constants``, ported from reset_utils.py:80-83), so
    the reward path must get the same offsets out of this module.
    """
    cfg = SimToolRealCfg()
    fixed_size = np.asarray(cfg.reward.fixed_size, dtype=np.float32)

    offsets = keypoint_offsets(fixed_size, keypoint_scale=cfg.goal.keypoint_scale)
    expected = CORNERS * (0.5 * cfg.goal.keypoint_scale * fixed_size)

    np.testing.assert_allclose(offsets, expected, rtol=1e-6, atol=1e-8)
    # Half-extents: 0.141 m of extent becomes 0.141 * 0.5 * 1.5 = 0.105750 m.
    np.testing.assert_allclose(
        np.abs(offsets[:, 0]), np.full(NUM_KEYPOINTS, 0.10575, dtype=np.float32), rtol=1e-6
    )


def test_observation_side_size_converts_phi_to_metres() -> None:
    """phi is dimensionless; the observation size is phi * 0.04 * multiplier."""
    cfg = SimToolRealCfg()
    base = cfg.reward.object_base_size
    # A handle bbox of (0.14, 0.02, 0.02) m -> phi = bbox / 0.04.
    bbox_m = np.asarray([0.14, 0.02, 0.02], dtype=np.float32)
    phi = bbox_m / base

    size = observation_keypoint_size(phi, base)

    assert size.dtype == np.float32
    np.testing.assert_allclose(size, bbox_m, rtol=1e-6, atol=1e-8)

    # The DR multiplier scales it further (obs_utils.py:275); default range is a no-op.
    multiplier = np.asarray([1.2, 0.8, 1.0], dtype=np.float32)
    np.testing.assert_allclose(
        observation_keypoint_size(phi, base, multiplier), bbox_m * multiplier, rtol=1e-6, atol=1e-8
    )

    # Feeding raw phi into compute_keypoints instead would inflate offsets by 1/0.04.
    pos = np.zeros((1, 3), dtype=np.float32)
    quat = IDENTITY_QUAT[None, :].copy()
    correct = compute_keypoints(pos, quat, size)
    wrong = compute_keypoints(pos, quat, np.asarray(phi, dtype=np.float32))
    np.testing.assert_allclose(wrong, correct / base, rtol=1e-5, atol=1e-6)


def test_two_keypoint_sizes_differ_and_both_flow_through_max_dist() -> None:
    """Observation (per-object) and reward (fixed) sizes give different metrics."""
    cfg = SimToolRealCfg()
    base = cfg.reward.object_base_size
    scale = cfg.goal.keypoint_scale
    n_envs = 2

    pos = np.zeros((n_envs, 3), dtype=np.float32)
    goal_pos = np.zeros((n_envs, 3), dtype=np.float32)
    quat = np.broadcast_to(IDENTITY_QUAT, (n_envs, 4)).copy()
    # 180 deg about z: keypoint mismatch is driven purely by object extent.
    goal_quat = np.broadcast_to(
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n_envs, 4)
    ).copy()

    phi = np.asarray([[3.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float32)
    obs_size = observation_keypoint_size(phi, base)
    obs_dist = keypoint_max_dist(
        compute_keypoints(pos, quat, obs_size, keypoint_scale=scale),
        compute_keypoints(goal_pos, goal_quat, obs_size, keypoint_scale=scale),
    )

    fixed_size = np.asarray(cfg.reward.fixed_size, dtype=np.float32)
    fixed_dist = keypoint_max_dist(
        compute_keypoints(pos, quat, fixed_size, keypoint_scale=scale),
        compute_keypoints(goal_pos, goal_quat, fixed_size, keypoint_scale=scale),
    )

    # The fixed size is shared, so both envs report the same reward-side distance;
    # the per-object size does not (that is the whole point of the split).
    assert fixed_dist[0] == pytest.approx(fixed_dist[1], rel=1e-6)
    assert obs_dist[0] != pytest.approx(obs_dist[1], rel=1e-3)

    # 180 deg about z maps (x, y, z) to (-x, -y, z), so corner 0 -> (-x, -y, z)
    # and the worst-corner distance is 2 * sqrt(x^2 + y^2) of the half-extent.
    half = obs_size * 0.5 * scale
    expected = 2.0 * np.linalg.norm(half[:, :2], axis=-1)
    np.testing.assert_allclose(obs_dist, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("pos_shape", "quat_shape", "size_shape"),
    [
        ((4, 2), (4, 4), (3,)),  # pos not (N, 3)
        ((4, 3), (4, 3), (3,)),  # quat not (N, 4)
        ((4, 3), (3, 4), (3,)),  # quat batch mismatch
        ((4, 3), (4, 4), (4,)),  # size not (3,) / (N, 3)
        ((4, 3), (4, 4), (3, 3)),  # per-env size with the wrong batch
    ],
)
def test_bad_shapes_raise(
    pos_shape: tuple[int, ...], quat_shape: tuple[int, ...], size_shape: tuple[int, ...]
) -> None:
    """Shape mistakes fail loudly instead of silently broadcasting."""
    with pytest.raises(ValueError):
        compute_keypoints(
            np.zeros(pos_shape, dtype=np.float32),
            np.zeros(quat_shape, dtype=np.float32),
            np.ones(size_shape, dtype=np.float32),
        )


def test_max_dist_rejects_mismatched_inputs() -> None:
    """keypoint_max_dist needs index-aligned (N, K, 3) inputs."""
    obj_kp = np.zeros((2, NUM_KEYPOINTS, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        keypoint_max_dist(obj_kp, np.zeros((3, NUM_KEYPOINTS, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        keypoint_max_dist(obj_kp, np.zeros((2, NUM_KEYPOINTS, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        keypoint_max_dist(np.zeros((2, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.float32))
