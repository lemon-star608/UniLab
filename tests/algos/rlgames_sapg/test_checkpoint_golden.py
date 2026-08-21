from __future__ import annotations

import copy

import pytest
import torch
from tests.algos.rlgames_sapg.source_rollout_harness import load_rollout_fixture

EXPECTED_CHECKPOINT_STATE_KEYS = [
    "assymetric_vf_nets",
    "central_val_stats",
    "current_lengths",
    "current_rewards",
    "current_shaped_rewards",
    "dones",
    "env_state",
    "epoch",
    "frame",
    "last_mean_rewards",
    "model",
    "obs",
    "optimizer",
    "rnn_states",
    "scaler",
    "trackers",
]


def test_checkpoint_state_descriptor_covers_every_tensor_and_boundary() -> None:
    from tests.algos.rlgames_sapg import source_checkpoint_harness as harness

    state = {
        0: {
            "model": {"weight": torch.tensor([[1.0, -2.0]], dtype=torch.float32)},
            "optimizer": {"state": {}, "param_groups": []},
            "env_state": None,
            "epoch": 7,
        }
    }

    descriptor = harness.describe_state(state)

    outer = descriptor["items"][0]
    assert outer["key"] == {"kind": "int", "value": 0}
    inner = {item["key"]["value"]: item["value"] for item in outer["value"]["items"]}
    assert set(inner) == {"model", "optimizer", "env_state", "epoch"}
    assert inner["env_state"] == {"kind": "none"}
    tensor = inner["model"]["items"][0]["value"]
    assert tensor["kind"] == "tensor"
    assert tensor["shape"] == [1, 2]
    assert tensor["dtype"] == "float32"
    assert len(tensor["sha256"]) == 64


def test_checkpoint_state_descriptor_rejects_unsupported_objects() -> None:
    from tests.algos.rlgames_sapg import source_checkpoint_harness as harness

    with pytest.raises(RuntimeError, match="unsupported checkpoint state"):
        harness.describe_state({"bad": object()})


def test_observable_comparison_allows_only_approved_numeric_tolerance() -> None:
    from tests.algos.rlgames_sapg import source_checkpoint_harness as harness

    source = harness.observe_tree(
        {"action": torch.tensor([1.0, -2.0]), "done": torch.tensor([0, 1])}
    )
    close = harness.observe_tree(
        {"action": torch.tensor([1.0 + 5e-7, -2.0]), "done": torch.tensor([0, 1])}
    )
    far = harness.observe_tree({"action": torch.tensor([1.01, -2.0]), "done": torch.tensor([0, 1])})

    assert harness.compare_observable(source, close, atol=1e-6, rtol=1e-5) == 2
    with pytest.raises(AssertionError, match="numeric observable mismatch"):
        harness.compare_observable(source, far, atol=1e-6, rtol=1e-5)


def test_observable_comparison_rejects_discrete_and_nonfinite_drift() -> None:
    from tests.algos.rlgames_sapg import source_checkpoint_harness as harness

    source = harness.observe_tree(
        {"value": torch.tensor([float("nan"), float("inf")]), "done": torch.tensor([0, 1])}
    )
    matched = harness.observe_tree(
        {"value": torch.tensor([float("nan"), float("inf")]), "done": torch.tensor([0, 1])}
    )
    wrong_done = harness.observe_tree(
        {"value": torch.tensor([float("nan"), float("inf")]), "done": torch.tensor([1, 1])}
    )
    wrong_inf = harness.observe_tree(
        {"value": torch.tensor([float("nan"), -float("inf")]), "done": torch.tensor([0, 1])}
    )

    assert harness.compare_observable(source, matched, atol=1e-6, rtol=1e-5) == 2
    with pytest.raises(AssertionError, match="exact observable mismatch"):
        harness.compare_observable(source, wrong_done, atol=1e-6, rtol=1e-5)
    with pytest.raises(AssertionError, match="non-finite observable mismatch"):
        harness.compare_observable(source, wrong_inf, atol=1e-6, rtol=1e-5)


def test_code5_runner_params_use_small_recorded_native_resource_boundary() -> None:
    from tests.algos.rlgames_sapg import source_checkpoint_harness as harness

    original = load_rollout_fixture().manifest["runner_params"]
    params = harness.code5_runner_params(original)
    config = params["config"]

    assert original["config"]["num_actors"] == 12
    assert config["num_actors"] == 6
    assert config["expl_coef_block_size"] == 1
    assert config["horizon_length"] == 4
    assert config["seq_length"] == 4
    assert config["minibatch_size"] == 12
    assert config["mini_epochs"] == 1
    assert config["mixed_precision"] is True
    assert config["use_others_experience"] == "none"
    assert params["network"]["mlp"]["units"] == [32, 32, 16, 16]
    assert params["network"]["rnn"]["units"] == 16
    assert config["central_value_config"]["minibatch_size"] == 12
    assert config["central_value_config"]["network"]["mlp"]["units"] == [32, 32, 16, 16]


def test_player_routing_contract_locks_six_envs_and_native_fallback() -> None:
    from tests.algos.rlgames_sapg import source_checkpoint_harness as harness

    cases = [
        {
            "env_count": 6,
            "network_ids": [50.0, 40.0, 30.0, 20.0, 10.0, 0.0],
            "embedding_ids": [50.0, 40.0, 30.0, 20.0, 10.0, 0.0],
            "selected_rows": [0, 1, 2, 3, 4, 5],
        },
        {
            "env_count": 5,
            "network_ids": [50.0, 40.0, 30.0, 20.0, 10.0, 0.0],
            "embedding_ids": [50.0, 37.5, 25.0, 12.5, 0.0],
            "selected_rows": [0, 0, 0, 0, 5],
        },
        {
            "env_count": 7,
            "network_ids": [50.0, 40.0, 30.0, 20.0, 10.0, 0.0],
            "embedding_ids": [
                50.0,
                41.66666793823242,
                33.333335876464844,
                25.0,
                16.66666603088379,
                8.333333015441895,
                0.0,
            ],
            "selected_rows": [0, 0, 0, 0, 0, 0, 5],
        },
    ]

    harness.validate_player_routing(cases)
    cases[1]["selected_rows"][2] = 2
    with pytest.raises(RuntimeError, match="player selected-row routing drift"):
        harness.validate_player_routing(cases)


def test_native_checkpoint_payload_records_source_save_boundary() -> None:
    from tests.algos.rlgames_sapg import _runtime_requirement
    from tests.algos.rlgames_sapg import source_checkpoint_harness as harness

    _runtime_requirement.require_simtoolreal_rl_games()
    params = harness.code5_runner_params(load_rollout_fixture().manifest["runner_params"])
    checkpoint = harness.create_native_checkpoint(params, _runtime_requirement.VENDOR_PACKAGE_ROOT)

    assert 0 < len(checkpoint.payload) < 1024 * 1024
    assert checkpoint.metadata["file_name"] == "source_checkpoint.pth"
    assert checkpoint.metadata["bytes"] == len(checkpoint.payload)
    assert len(checkpoint.metadata["sha256"]) == 64
    assert checkpoint.metadata["outer_keys"] == [{"kind": "int", "value": 0}]
    assert checkpoint.metadata["state_keys"] == EXPECTED_CHECKPOINT_STATE_KEYS
    assert checkpoint.metadata["env_state_is_none"] is True
    assert checkpoint.metadata["rng_saved"] is False
    assert checkpoint.metadata["central_optimizer_saved"] is False
    assert checkpoint.metadata["actor_optimizer_state_entries"] > 0
    assert checkpoint.metadata["state"]["kind"] == "dict"
    assert "rl_games.torch_runner" in checkpoint.loaded_modules


def test_native_resume_and_player_capture_uses_checkpoint_owners() -> None:
    from tests.algos.rlgames_sapg import _runtime_requirement
    from tests.algos.rlgames_sapg import source_checkpoint_harness as harness

    _runtime_requirement.require_simtoolreal_rl_games()
    params = harness.code5_runner_params(load_rollout_fixture().manifest["runner_params"])
    checkpoint = harness.create_native_checkpoint(params, _runtime_requirement.VENDOR_PACKAGE_ROOT)
    runtime = harness.capture_runtime(
        checkpoint.payload,
        params,
        _runtime_requirement.VENDOR_PACKAGE_ROOT,
    )

    resume = runtime["resume"]
    assert resume["loaded_state"] == checkpoint.metadata["state"]
    assert resume["runner_before_update"]["epoch_num"] == 7
    assert resume["runner_before_update"]["frame"] == 24
    assert resume["runner_before_update"]["central_optimizer_state_entries"] == 0
    assert resume["env_set_state_calls"] == [None]
    assert resume["external_rng_before"] != resume["external_rng_after"]
    assert resume["first_action_input"]["kind"] == "dict"
    assert resume["first_action_output"]["kind"] == "dict"
    assert resume["first_value_output"]["kind"] == "array"
    assert resume["native_return"]["kind"] == "dict"
    assert resume["final_state"]["kind"] == "dict"

    player = runtime["player"]
    assert player["owner"] == "rl_games.algos_torch.players.PpoPlayerContinuous"
    harness.validate_player_routing(player["cases"])
    for case in player["cases"]:
        assert case["env_set_state_calls"] == []
        assert case["deterministic"]["selected_action"]["kind"] == "array"
        assert case["stochastic"]["selected_action"]["kind"] == "array"
        assert case["deterministic"]["model_output"]["kind"] == "dict"
        assert case["stochastic"]["rnn_after"]["kind"] == "tuple"

    replay = harness.compare_runtime(runtime, copy.deepcopy(runtime))
    assert replay.player_counts == (6, 5, 7)
    assert replay.observable_leaves > 0
    assert replay.exact_state_sections >= 4

    wrong_runner = copy.deepcopy(runtime)
    wrong_runner["resume"]["runner_before_update"]["frame"] += 1
    with pytest.raises(AssertionError, match="resume runner state mismatch"):
        harness.compare_runtime(runtime, wrong_runner)

    wrong_action = copy.deepcopy(runtime)
    wrong_action["player"]["cases"][0]["deterministic"]["selected_action"]["data"][0] += 0.1
    with pytest.raises(AssertionError, match="numeric observable mismatch"):
        harness.compare_runtime(runtime, wrong_action)

    manifest = harness.build_fixture_manifest(
        checkpoint,
        runtime,
        params,
        generation_command="canonical-source-command",
    )
    harness.validate_fixture(manifest, checkpoint.payload)
    assert manifest["schema_version"] == 1
    assert manifest["generation_mode"] == "source-only"
    assert manifest["ordinary_pytest_regenerates"] is False
    assert manifest["payload"]["rng_saved"] is False
    assert manifest["payload"]["env_state_is_none"] is True
    assert manifest["provenance"]["loaded_rl_games_modules"] == checkpoint.loaded_modules
    assert len(manifest["canonical_payload_sha256"]) == 64

    wrong_boundary = copy.deepcopy(manifest)
    wrong_boundary["payload"]["rng_saved"] = True
    with pytest.raises(RuntimeError, match="checkpoint RNG boundary drift"):
        harness.validate_fixture(wrong_boundary, checkpoint.payload)
    with pytest.raises(RuntimeError, match="checkpoint payload hash drift"):
        harness.validate_fixture(manifest, checkpoint.payload + b"corrupt")


def test_frozen_checkpoint_fixture_has_external_and_canonical_anchors() -> None:
    from tests.algos.rlgames_sapg import source_checkpoint_harness as harness

    fixture = harness.load_fixture()
    assert fixture.manifest["schema_version"] == harness.SCHEMA_VERSION
    assert fixture.manifest["fixture_files"] == [
        harness.CHECKPOINT_FILE_NAME,
        harness.MANIFEST_FILE_NAME,
    ]
    assert harness.EXPECTED_CHECKPOINT_SHA256
    assert harness.EXPECTED_MANIFEST_SHA256
    assert harness.EXPECTED_PAYLOAD_SHA256 == fixture.manifest["canonical_payload_sha256"]
    assert fixture.manifest["payload"]["sha256"] == harness.EXPECTED_CHECKPOINT_SHA256


def test_frozen_checkpoint_fixture_replays_native_target() -> None:
    from tests.algos.rlgames_sapg import source_checkpoint_harness as harness

    result = harness.replay_fixture(harness.load_fixture())
    assert result.player_counts == (6, 5, 7)
    assert result.observable_leaves > 0
    assert result.exact_state_sections >= 4
    assert result.native_owner_paths


def test_checkpoint_manifest_rejects_namespace_and_routing_inventory_drift() -> None:
    from tests.algos.rlgames_sapg import source_checkpoint_harness as harness

    fixture = harness.load_fixture()
    namespace_drift = copy.deepcopy(fixture.manifest)
    module_name = next(name for name in namespace_drift["provenance"]["loaded_rl_games_modules"])
    namespace_drift["provenance"]["loaded_rl_games_modules"][module_name]["path"] = (
        "/outside/rl_games.py"
    )
    namespace_drift["canonical_payload_sha256"] = harness.canonical_sha256(namespace_drift)
    with pytest.raises(RuntimeError, match="namespace root/path drift"):
        harness.validate_fixture(namespace_drift, fixture.payload)

    routing_drift = copy.deepcopy(fixture.manifest)
    routing_drift["player"]["cases"].pop()
    routing_drift["canonical_payload_sha256"] = harness.canonical_sha256(routing_drift)
    with pytest.raises(RuntimeError, match="player case inventory drift"):
        harness.validate_fixture(routing_drift, fixture.payload)
