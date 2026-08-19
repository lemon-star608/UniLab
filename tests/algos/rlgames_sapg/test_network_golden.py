from __future__ import annotations

# The required provenance gate intentionally runs before the harness import.
# ruff: noqa: I001

import numpy as np

from tests.algos.rlgames_sapg._runtime_requirement import require_simtoolreal_rl_games

require_simtoolreal_rl_games()

from tests.algos.rlgames_sapg.source_network_harness import (
    MAPPED_TENSORS,
    load_network_fixture,
    replay_network_fixture,
)


def test_network_matches_canonical_source_fixture() -> None:
    fixture = load_network_fixture()
    actual = replay_network_fixture(fixture)

    assert set(actual.tensors) == set(MAPPED_TENSORS)
    assert actual.native_initialization_hashes_exact
    assert actual.weight_hashes_exact
    assert actual.input_hashes_exact
    for name in MAPPED_TENSORS:
        np.testing.assert_allclose(
            actual.tensors[name],
            fixture.tensors[name],
            atol=1e-6,
            rtol=1e-5,
            err_msg=name,
        )
    if actual.is_canonical_platform:
        assert actual.computed_hashes_exact
    assert actual.actor_parameter_signatures.matches(fixture.actor_parameter_signatures)
    assert actual.actor_gradient_signatures.matches(fixture.actor_gradient_signatures)
    assert actual.central_parameter_signatures.matches(fixture.central_parameter_signatures)
    assert actual.central_gradient_signatures.matches(fixture.central_gradient_signatures)


def test_fixture_keeps_actor_and_central_values_and_embeddings_distinct() -> None:
    fixture = load_network_fixture()

    assert fixture.tensors["actor_embedding"].shape == (6, 32)
    assert fixture.tensors["central_embedding"].shape == (6, 32)
    assert not np.array_equal(
        fixture.tensors["actor_embedding"], fixture.tensors["central_embedding"]
    )
    assert fixture.tensors["actor_shared_value"].shape == (12, 1)
    assert fixture.tensors["central_value"].shape == (12, 1)
    assert not np.array_equal(
        fixture.tensors["actor_shared_value"], fixture.tensors["central_value"]
    )
