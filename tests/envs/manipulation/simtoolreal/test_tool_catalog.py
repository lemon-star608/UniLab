from __future__ import annotations

from collections import Counter

from unilab.envs.manipulation.simtoolreal.tool_catalog import ALL_TYPES, build_tool_catalog


def test_catalog_is_deterministic_and_has_exact_topology_census() -> None:
    first = build_tool_catalog(ALL_TYPES, num_per_type=50, seed=42, shuffle=True)
    second = build_tool_catalog(ALL_TYPES, num_per_type=50, seed=42, shuffle=True)
    assert first == second
    assert len(first) == 600
    assert Counter(spec.topology for spec in first) == {
        "box_box": 250,
        "capsule_box": 300,
        "box_only": 50,
    }


def test_catalog_does_not_mutate_global_numpy_rng() -> None:
    import numpy as np

    np.random.seed(123)
    before = np.random.random()
    build_tool_catalog(ALL_TYPES, num_per_type=1, seed=42, shuffle=True)
    after = np.random.random()
    np.random.seed(123)
    assert before == np.random.random()
    assert after == np.random.random()
