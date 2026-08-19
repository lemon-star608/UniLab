from __future__ import annotations

# The required provenance gate intentionally runs before the harness import.
# ruff: noqa: I001

from tests.algos.rlgames_sapg._runtime_requirement import require_simtoolreal_rl_games

require_simtoolreal_rl_games()

import json
from unittest.mock import Mock

import numpy as np
import pytest

from scripts import generate_simtoolreal_sapg_rollout_fixture as rollout_generator
from tests.algos.rlgames_sapg import source_rollout_harness as rollout_harness
from tests.algos.rlgames_sapg.source_network_harness import array_metadata
from tests.algos.rlgames_sapg.source_rollout_harness import (
    SOURCE_HEAD,
    SOURCE_RL_GAMES_TREE,
    load_rollout_fixture,
    replay_rollout_fixture,
)

EXPECTED_NPZ_SHA256 = "3573cc3d0a3700b1c5985b63d7b16175d5ccee3dc8f4b7ef1a7bd7b6676819e8"
EXPECTED_MANIFEST_PAYLOAD_SHA256 = (
    "7d88cb01dce4607391a39d1fb31b21d8366d2bdadae2e0dce6eb02323c06901d"
)
RAW_EXPERIENCE_METADATA = json.loads(
    '{"actions":[[4,12,29],"float32"],"mus":[[4,12,29],"float32"],"sigmas":[[4,12,29],"float32"],"neglogpacs":[[4,12],"float32"],"values":[[4,12,1],"float32"],"obses":[[4,12,141],"float32"],"states":[[4,12,163],"float32"],"dones":[[4,12],"uint8"],"rewards":[[4,12,1],"float32"],"intr_rewards":[[4,12,1],"float32"]}'
)
RETURNED_FIELDS = "actions neglogpacs values mus sigmas obses states dones".split()


def test_rollout_fixture_has_fixed_provenance_and_inventory() -> None:
    fixture = load_rollout_fixture()
    manifest = fixture.manifest

    assert manifest["provenance"]["source_head"] == SOURCE_HEAD
    assert manifest["provenance"]["source_rl_games_tree"] == SOURCE_RL_GAMES_TREE
    assert manifest["fixture_files"]["npz"]["sha256"] == EXPECTED_NPZ_SHA256
    assert manifest["manifest_payload_sha256"] == EXPECTED_MANIFEST_PAYLOAD_SHA256
    assert manifest["generation"] == json.loads(
        '{"mode":"source-only","ordinary_pytest_regenerates":false,"python":"3.11"}'
    )
    contract = manifest["synthetic_contract"]
    assert contract["base_rows"] == 48
    assert contract["follower_rows"] == 8
    assert contract["augmented_rows"] == 56
    assert contract["actor_dataset_batches"] == [12, 12, 12, 20]
    assert contract["central_dataset_batches"] == [12, 12, 12, 20]


def test_fixture_freezes_complete_raw_experience_owner_layout() -> None:
    fixture = load_rollout_fixture()
    prefix = "buffer_raw_experience__"
    semantics = fixture.manifest["semantics"]
    raw = {
        name.removeprefix(prefix): value
        for name, value in fixture.arrays.items()
        if name.startswith(prefix)
    }

    assert set(raw) == set(RAW_EXPERIENCE_METADATA)
    for name, (shape, dtype) in RAW_EXPERIENCE_METADATA.items():
        assert array_metadata(raw[name]) == fixture.manifest["npz_arrays"][prefix + name]
        assert list(raw[name].shape) == shape
        assert str(raw[name].dtype) == dtype
    expected = json.loads(
        '{"raw_experience_axes":["time","env"],"raw_intr_rewards_present":true,"extras_mb_intr_rewards_is_none":true,"get_values_phases":["play","augment","augment"],"counterfactual_current_and_tail_calls":2,"counterfactual_privileged_states_relabelled_before_value":false,"augmented_states_relabelled_for_training":true,"counterfactual_intrinsic_reward_used":false}'
    )
    assert {key: semantics[key] for key in expected} == expected
    assert semantics["raw_obs_row_ids"] == [list(range(offset, 92, 8)) for offset in range(4)]
    assert semantics["raw_to_returned_base_native_exact"] == RETURNED_FIELDS
    assert semantics["returned_base_fields"] == RETURNED_FIELDS
    assert {"rewards", "intr_rewards"}.isdisjoint(semantics["returned_base_fields"])
    required_calls = "ExperienceBuffer.tensor_dict raw snapshot|rl_games.common.custom_utils.swap_and_flatten01 raw-to-flatten transform|rl_games.common.custom_utils.filter_leader via augment_batch_for_mixed_expl"
    assert set(required_calls.split("|")) <= set(fixture.manifest["native_calls"])


def test_target_array_metadata_rejects_drift_and_accepts_exact() -> None:
    expected = np.arange(8, dtype=np.float32)
    metadata = array_metadata(expected)
    changed = expected.copy()
    changed[0] = -1
    for actual, drift in (
        (expected.reshape(1, 8), "shape"),
        (expected.astype(np.float64), "dtype"),
        (changed, "content hash"),
    ):
        with pytest.raises(RuntimeError, match=rf"sample.*{drift}"):
            rollout_harness._validate_target_array_metadata("sample", actual, metadata)
    rollout_harness._validate_target_array_metadata("sample", expected.copy(), metadata)


class SubtractionProbe(np.ndarray):
    def __new__(cls, values, events):
        result = np.asarray(values).view(cls)
        result.events = events
        return result

    def __array_finalize__(self, source):
        self.events = getattr(source, "events", None)

    def __sub__(self, other):
        self.events.append("subtract")
        return super().__sub__(other)


def test_target_metadata_is_whole_inventory_gate_before_numeric(monkeypatch) -> None:
    expected = {name: np.arange(4, dtype=np.float32) for name in ("a", "b")}
    events = []
    actual = {
        "a": SubtractionProbe(expected["a"], events),
        "b": expected["b"].reshape(1, 4),
    }
    metadata = {name: array_metadata(value) for name, value in expected.items()}
    original_validate = rollout_harness._validate_target_array_metadata

    def validate(name, value, expected_metadata):
        events.append(f"validate:{name}")
        return original_validate(name, value, expected_metadata)

    monkeypatch.setattr(rollout_harness, "_validate_target_array_metadata", validate)
    with pytest.raises(RuntimeError, match=r"b.*shape"):
        rollout_harness._target_array_diagnostics(actual, expected, metadata)
    assert events == ["validate:a", "validate:b"]


def test_target_metadata_whole_inventory_gate_reports_sorted_missing_extra(monkeypatch) -> None:
    expected = {name: np.arange(4, dtype=np.float32) for name in ("a", "b")}
    actual = {"a": SubtractionProbe(expected["a"], []), "c": expected["a"].copy()}
    metadata = {name: array_metadata(value) for name, value in expected.items()}
    validator = Mock()
    monkeypatch.setattr(rollout_harness, "_validate_target_array_metadata", validator)
    with pytest.raises(RuntimeError, match=r"missing=\['b'\], extra=\['c'\]"):
        rollout_harness._target_array_diagnostics(actual, expected, metadata)
    assert actual["a"].events == [] and not validator.called


def _fixture_payload() -> tuple[dict, dict[str, np.ndarray]]:
    return {"schema_version": 1}, {"sample": np.arange(4, dtype=np.float32)}


def test_fixture_writer_accepts_real_output_directory(tmp_path) -> None:
    output = tmp_path / "real"
    rollout_generator._write(output, *_fixture_payload())
    assert all((output / name).is_file() for name in rollout_generator.FIXTURE_NAMES)


def test_fixture_writer_rejects_symlinked_ancestor_without_writing(tmp_path) -> None:
    outside = tmp_path / "outside/nested"
    outside.mkdir(parents=True)
    sentinels = (b"npz sentinel", b"manifest sentinel")
    leaves = [outside / name for name in rollout_generator.FIXTURE_NAMES]
    for path, content in zip(leaves, sentinels, strict=True):
        path.write_bytes(content)
    alias = tmp_path / "alias"
    alias.symlink_to(outside.parent, target_is_directory=True)
    rejected = False
    try:
        rollout_generator._write(alias / "nested", *_fixture_payload())
    except RuntimeError as error:
        rejected = "symlink" in str(error)
    observed = tuple(path.read_bytes() for path in leaves)
    assert (rejected, observed) == (True, sentinels)
    with pytest.raises(RuntimeError, match=r"component.*symlink"):
        rollout_generator._write(alias, *_fixture_payload())

    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(RuntimeError, match=r"component.*symlink"):
        rollout_generator._write(broken / "nested", *_fixture_payload())
    assert not (tmp_path / "missing").exists()
    file_output = tmp_path / "file-output"
    file_output.write_bytes(b"outside")
    with pytest.raises(RuntimeError, match=r"output.*real directory"):
        rollout_generator._write(file_output, *_fixture_payload())
    assert file_output.read_bytes() == b"outside"


def test_generate_passes_original_unresolved_output_to_writer(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("UNILAB_SAPG_ORACLE_MODE", "source")
    (tmp_path / "sub").mkdir()
    output = tmp_path / "sub" / ".." / "out"
    semantics = load_rollout_fixture().manifest["semantics"]
    capture = rollout_harness.RolloutCapture({}, semantics, {}, rollout_harness.CANONICAL_PLATFORM)
    git_results = iter((SOURCE_HEAD, SOURCE_RL_GAMES_TREE))
    monkeypatch.setattr(rollout_generator, "_run_git", lambda *_args: next(git_results))
    monkeypatch.setattr(rollout_generator, "_runner_params", lambda _source: ({}, {}, {}))
    monkeypatch.setattr(rollout_generator, "capture_rollout", lambda *_args: capture)
    monkeypatch.setattr(rollout_generator, "_verify_source_modules", lambda *_args: [])
    writes = []
    monkeypatch.setattr(rollout_generator, "_write", lambda raw, *_args: writes.append(raw))

    rollout_generator.generate(tmp_path / "source", output)

    assert writes == [output]
    assert writes[0] != output.resolve()


@pytest.mark.parametrize("leaf", rollout_generator.FIXTURE_NAMES)
@pytest.mark.parametrize("kind", ("normal", "broken", "directory"))
def test_fixture_writer_rejects_invalid_leaves_without_writing(tmp_path, leaf, kind) -> None:
    names = rollout_generator.FIXTURE_NAMES
    output = tmp_path / "output"
    output.mkdir()
    target = tmp_path / "outside-leaf"
    if kind == "directory":
        (output / leaf).mkdir()
    else:
        if kind == "normal":
            target.write_bytes(b"outside")
        (output / leaf).symlink_to(target)
    other = output / names[leaf == names[0]]
    other.write_bytes(b"other")
    with pytest.raises(RuntimeError, match=rf"{leaf}.*regular file"):
        rollout_generator._write(output, *_fixture_payload())
    assert other.read_bytes() == b"other"
    if kind == "normal":
        assert target.read_bytes() == b"outside"


def test_rnn_delegate_records_frozen_inputs_and_exact_owner_result() -> None:
    input_value = np.arange(3, dtype=np.float32)
    states = [np.arange(2, dtype=np.float32)]
    dones = np.arange(3, dtype=np.uint8)
    expected = tuple(value.copy() for value in (input_value, states[0], dones))
    sentinel = object()

    def mutate(input_arg, states_arg, dones_arg, _bptt_len):
        input_arg[:] = states_arg[0][:] = dones_arg[:] = 99
        return sentinel

    owner, records = Mock(side_effect=mutate), []
    result = rollout_harness._delegate_rnn_and_record(
        owner, records, (input_value, states, dones, 4)
    )

    assert (owner.call_count, len(records)) == (1, 1)
    assert result is sentinel
    for actual, frozen in zip(
        (records[0].input, records[0].states[0], records[0].dones), expected, strict=True
    ):
        np.testing.assert_array_equal(actual, frozen)


def test_rnn_probe_is_delegate_only_with_isolated_unmasked_diagnostic() -> None:
    fixture = load_rollout_fixture()
    semantics = fixture.manifest["semantics"]
    assert semantics["rnn_delegate_original_calls"] == 4
    assert semantics["rnn_isolated_unmasked_calls"] == 4
    assert not semantics["rng_consumption"]["isolated_unmasked_rnn_diagnostic"]
    states = fixture.manifest["rng_states"]
    assert states["before_unmasked_rnn_diagnostic"] == states["after_unmasked_rnn_diagnostic"]
    assert any(name.startswith("diagnostic_rnn_masked_probe__") for name in fixture.arrays)
    assert any(name.startswith("diagnostic_rnn_isolated_unmasked__") for name in fixture.arrays)
    assert all(semantics["rnn_mask_changes_returned_state"])


def test_augmented_state_carriers_require_exact_native_observation_relation() -> None:
    fixture = load_rollout_fixture()
    arrays = fixture.arrays
    semantics = fixture.manifest["semantics"]
    follower_indices = [
        semantics["base_row_ids"].index(row) for row in semantics["follower_row_ids"]
    ]
    assert rollout_harness._validate_augmented_state_carriers(arrays, follower_indices)
    for replacement, drift in ((123.0, "observation"), (np.nan, "finite")):
        mutated = dict(arrays)
        mutated["buffer_pre_shuffle__states"] = arrays["buffer_pre_shuffle__states"].copy()
        mutated["buffer_pre_shuffle__states"][-8:, -1] = replacement
        with pytest.raises(RuntimeError, match=drift):
            rollout_harness._validate_augmented_state_carriers(mutated, follower_indices)


def test_target_replays_canonical_source_rollout_exactly() -> None:
    fixture = load_rollout_fixture()
    replay = replay_rollout_fixture(fixture)

    assert replay.is_canonical_platform
    assert all(error == 0.0 for error in replay.max_abs_errors.values())
    assert all(error == 0.0 for error in replay.max_rel_errors.values())
