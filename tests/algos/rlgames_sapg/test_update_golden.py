from tests.algos.rlgames_sapg._runtime_requirement import require_simtoolreal_rl_games

require_simtoolreal_rl_games()

# The required runtime gate intentionally runs before the harness import.
# ruff: noqa: I001

import ast
import copy
import hashlib
import importlib.machinery
import io
import json
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.algos.rlgames_sapg import source_update_harness as harness
from tests.algos.rlgames_sapg.source_update_harness import (
    UpdateFixture,
    load_update_fixture,
    replay_update_fixture,
)


CASE_NAMES = ("normal_fp32", "normal_amp", "overflow_amp")
MANIFEST_KEYS = set(
    "schema_version generation_mode ordinary_pytest_regenerates provenance platform "
    "canonical_platform code3_anchors runner_params capture_contract cases npz_arrays "
    "exact_comparison_inventory numeric_comparison_inventory tolerances fixture_files "
    "canonical_payload_sha256 generation_command".split()
)
EXPECTED_UPDATE_NPZ_SHA256 = "df58bb09d67edd24a19f2a164a4851fa24b9f2d305e9826c10433635cee78463"
EXPECTED_UPDATE_MANIFEST_SHA256 = "748be517553df7689ee4a06991241e37fc205336f6a5638f2bdd168735d57e45"
EXPECTED_UPDATE_PAYLOAD_SHA256 = "686331c200b809b66b0978b855c75d359e8fffb51f918ca9f5ee2312dd44f397"
EXPECTED_SOURCE_HEAD = "2a9917533bfea70419ed2667a511d7238e5b3abc"
EXPECTED_SOURCE_RL_GAMES_TREE = "7a6a0bb090998d00565aaefa6ab9f2b3d356ace2"
EXPECTED_CODE3_ANCHORS = {
    "source_rollout_fp32.npz": ("3573cc3d0a3700b1c5985b63d7b16175d5ccee3dc8f4b7ef1a7bd7b6676819e8"),
    "source_rollout_manifest.json": (
        "785443d10e2037e0ca4e4b044dd1dc8207b438ea69555726eac9501ad8207d3f"
    ),
}
EXPECTED_SOURCE_OWNERS = {
    "train": {
        "path": "isaacsimenvs/cfg/train/SimToolRealSAPG.yaml",
        "blob": "f363d05d4a24b190b7837703b93270d8f3fe9a9c",
        "sha256": "04f30820094b062412541764b3feeb1492097e75afe5ad0df3fd0e2853496d34",
    },
    "task": {
        "path": "isaacsimenvs/cfg/task/SimToolReal.yaml",
        "blob": "6469d46867081b70edaa589dcb31c7090b64d45e",
        "sha256": "9d2bf514f75cc8c72b20da1e8ec971163bbd4cbdf6fc74812aa4a509340acb5e",
    },
}
OWNERS = {
    "runner": "rl_games.torch_runner.Runner",
    "agent": "rl_games.algos_torch.a2c_continuous.A2CAgent",
    "actor_dataset": "rl_games.common.datasets.PPODataset",
    "central_value": "rl_games.algos_torch.central_value.CentralValueTrain",
}
INPUT_KEYS = ("batch", "model", "optimizer", "scaler", "rms", "lr", "rng")
OUTPUT_KEYS = ("prepared", "model", "optimizer", "scaler", "rms", "lr", "rng")
RMS_ROLES = (
    "actor_input",
    "central_input",
    "actor_model_value",
    "active_central_value",
)
BATCH_FIELDS = (
    "actions",
    "dones",
    "mus",
    "neglogpacs",
    "obses",
    "off_policy_mask",
    "returns",
    "rnn_states",
    "sigmas",
    "states",
    "values",
)
PS_EXTRA_FIELDS = ("mb_intr_rewards", "rewards")
PREPARED_FIELDS = {
    "actor": (
        "actions",
        "advantages",
        "dones",
        "mu",
        "obs",
        "off_policy_mask",
        "old_logp_actions",
        "old_values",
        "returns",
        "rnn_masks",
        "rnn_states",
        "sigma",
    ),
    "central": (
        "actions",
        "advantages",
        "dones",
        "obs",
        "old_values",
        "returns",
        "rnn_masks",
    ),
}


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_record(name: str, value: np.ndarray, comparison: str) -> dict[str, object]:
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": _sha256(value),
        "comparison": comparison,
    }


def _array_name(case_name: str, phase: str, *path: str) -> str:
    return "__".join((case_name, phase, *path))


def _prepared_name(case_name: str, owner: str, field: str = "obs") -> str:
    return _array_name(case_name, "output", "prepared", owner, field)


def _stored_array(store, reference: dict[str, object]) -> np.ndarray:
    assert set(reference) == {"kind", "name"}
    assert reference["kind"] == "array"
    return store.arrays[reference["name"]]


def _array_reference_names(value: object) -> list[str]:
    if isinstance(value, dict):
        if value.get("kind") == "array":
            assert set(value) == {"kind", "name"}
            name = value["name"]
            assert isinstance(name, str)
            return [name]
        return [name for child in value.values() for name in _array_reference_names(child)]
    if isinstance(value, list):
        return [name for child in value for name in _array_reference_names(child)]
    return []


def _minimal_case(name: str) -> dict[str, object]:
    mixed_precision = name != "normal_fp32"
    overflow = name == "overflow_amp"
    empty_tree = {"kind": "dict", "items": {}}
    input_state = {key: {} for key in INPUT_KEYS}
    input_state["batch"] = empty_tree
    output_state = {key: {} for key in OUTPUT_KEYS}
    return {
        "name": name,
        "config": dict(
            mixed_precision=mixed_precision,
            mini_epochs=2,
            use_others_experience="none",
        ),
        "owners": dict(OWNERS),
        "input": input_state,
        "execution": {
            "identity_shuffle_calls": 0 if overflow else 1,
            "owner_call_order": ["prepare", "actor"]
            if overflow
            else ["prepare", "central", "actor"],
            "actor_update_attempts": 1 if overflow else 8,
            "actor_optimizer_steps": 0 if overflow else 8,
            "actor_scaler_skips": 1 if overflow else 0,
            "central_optimizer_steps": 0 if overflow else 8,
            "native_return": empty_tree,
            "overflow_mutation": "advantages[0]=+inf" if overflow else None,
        },
        "output": output_state,
        "restore": {"patches": True, "hooks": True},
    }


def _update_fixture(manifest: dict[str, object], arrays: dict[str, np.ndarray]) -> UpdateFixture:
    return UpdateFixture(manifest=manifest, arrays=arrays)


def in_memory_update_fixture() -> UpdateFixture:
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    name = _prepared_name("normal_fp32", "actor")
    metadata = {name: _array_record(name, array, "numeric")}
    manifest = {
        "schema_version": 2,
        "generation_mode": "source-only",
        "ordinary_pytest_regenerates": False,
        "provenance": {
            "source_head": EXPECTED_SOURCE_HEAD,
            "source_rl_games_tree": EXPECTED_SOURCE_RL_GAMES_TREE,
            "owners": copy.deepcopy(EXPECTED_SOURCE_OWNERS),
            "native_owner_paths": list(OWNERS.values()),
        },
        "platform": {},
        "canonical_platform": {},
        "code3_anchors": dict(EXPECTED_CODE3_ANCHORS),
        "runner_params": {},
        "capture_contract": {"case_names": list(CASE_NAMES)},
        "cases": [_minimal_case(name) for name in CASE_NAMES],
        "npz_arrays": metadata,
        "exact_comparison_inventory": [],
        "numeric_comparison_inventory": [name],
        "tolerances": {"atol": 1e-6, "rtol": 1e-5},
        "fixture_files": ["source_update_fp32.npz", "source_update_manifest.json"],
        "canonical_payload_sha256": "0" * 64,
        "generation_command": "test-only",
    }

    return _update_fixture(manifest, {name: array})


def mutated_arrays(fixture: UpdateFixture, drift: str) -> dict[str, np.ndarray]:
    arrays = {name: value.copy() for name, value in fixture.arrays.items()}
    name = sorted(arrays)[0]
    if drift == "missing":
        arrays.pop(name)
    elif drift == "extra":
        arrays["unexpected"] = np.zeros(1, dtype=np.uint8)
    elif drift == "shape":
        shaped = next(key for key, value in arrays.items() if value.ndim != 1)
        arrays[shaped] = arrays[shaped].reshape(-1)
    elif drift == "dtype":
        arrays[name] = arrays[name].astype(
            np.float32 if arrays[name].dtype == np.float64 else np.float64
        )
    elif drift == "content":
        arrays[name].view(np.uint8).reshape(-1)[0] ^= 1
    else:
        raise AssertionError(drift)
    return arrays


def _state_hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _snapshot_hash(value: object) -> str:
    return hashlib.sha256(harness.canonical_payload(value)).hexdigest()


def _tensor_state(label: str, shape: list[int], *, value: int | None = None) -> dict:
    state = {"shape": shape, "dtype": "float32", "sha256": _state_hash(label)}
    if value is not None:
        state["value"] = value
    return state


def _parameter_state(label: str) -> dict[str, object]:
    parameters = {"weight": _tensor_state(f"{label}:weight", [2, 3])}
    ordered_records = [{"name": name, **record} for name, record in sorted(parameters.items())]
    return {
        "parameters": parameters,
        "aggregate_sha256": _snapshot_hash(ordered_records),
    }


def _optimizer_state(label: str, step: int) -> dict[str, object]:
    state = (
        {
            "weight": {
                "step": _tensor_state(f"{label}:step:{step}", [], value=step),
                "exp_avg": _tensor_state(f"{label}:exp_avg:{step}", [2, 3]),
                "exp_avg_sq": _tensor_state(f"{label}:exp_avg_sq:{step}", [2, 3]),
            }
        }
        if step
        else {}
    )
    aggregate = {
        "param_groups": [{"params": ["weight"], "lr": 1e-4}],
        "state": state,
        "uninitialized": [] if step else ["weight"],
    }
    return {**aggregate, "aggregate_sha256": _snapshot_hash(aggregate)}


def _scaler_state(enabled: bool, scale: float | None = None) -> dict[str, object]:
    state_dict = (
        {
            "scale": scale,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "_growth_tracker": 0,
        }
        if enabled
        else {}
    )
    return {"enabled": enabled, "state_dict": state_dict}


def _rms_state(
    references: dict[str, tuple[dict[str, object], dict[str, object]]], count: float
) -> dict[str, object]:
    return {
        role: {
            "mean": copy.deepcopy(references[role][0]),
            "var": copy.deepcopy(references[role][1]),
            "count": count,
            "training": True,
        }
        for role in RMS_ROLES
    }


def _rng_state(
    keys_reference: dict[str, object],
    cpu_reference: dict[str, object],
    cuda_reference: dict[str, object],
) -> dict[str, object]:
    return {
        "numpy": {
            "algorithm": "MT19937",
            "keys": copy.deepcopy(keys_reference),
            "position": 0,
            "has_gauss": 0,
            "cached_gaussian": 0.0,
        },
        "torch_cpu": copy.deepcopy(cpu_reference),
        "torch_cuda": [copy.deepcopy(cuda_reference)],
    }


def _lr_state() -> dict[str, object]:
    return {
        "actor": {
            "last_lr": 1e-4,
            "optimizer_group_lrs": [1e-4],
            "scheduler_class": "rl_games.common.schedulers.IdentityScheduler",
            "scheduler_state": {"last_lr": 1e-4},
        },
        "central": {
            "lr": 1e-4,
            "optimizer_group_lrs": [1e-4],
            "scheduler_class": "rl_games.common.schedulers.IdentityScheduler",
            "scheduler_state": {"last_lr": 1e-4},
        },
    }


def complete_synthetic_capture() -> dict[str, object]:
    fixture = in_memory_update_fixture()
    manifest = copy.deepcopy(fixture.manifest)
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, object]] = {}
    inventories = {"exact": [], "numeric": []}

    def add_array(name: str, value: np.ndarray, comparison: str) -> dict[str, object]:
        assert name not in arrays
        arrays[name] = value.copy()
        metadata[name] = _array_record(name, arrays[name], comparison)
        inventories[comparison].append(name)
        return {"kind": "array", "name": name}

    manifest["capture_contract"] = {
        "case_names": list(CASE_NAMES),
        "rows": 56,
        "sequences": 14,
        "actor_dataset_batch_sizes": [12, 12, 12, 20],
        "central_dataset_batch_sizes": [12, 12, 12, 20],
        "rms_roles": list(RMS_ROLES),
        "rms_alias": {
            "four_roles_distinct": True,
            "active_is_central_value": True,
        },
    }
    cases = []
    for case_index, name in enumerate(CASE_NAMES):
        overflow = name == "overflow_amp"
        amp = name != "normal_fp32"
        actor_steps = 0 if overflow else 7 if amp else 8
        actor_skips = 1 if amp else 0
        base_float = np.arange(56, dtype=np.float32).reshape(56, 1) + np.float32(case_index * 100)

        def row_array(field: str, field_index: int) -> np.ndarray:
            if field == "dones":
                return ((np.arange(56) + case_index) % 2).astype(np.uint8)
            if field == "off_policy_mask":
                return ((np.arange(56) + field_index) % 2).astype(np.bool_)
            value = base_float + np.float32(field_index)
            if field in {"advantages", "neglogpacs", "old_logp_actions"}:
                return value[:, 0]
            return value

        def rnn_state_tree(*path: str, comparison: str) -> dict[str, object]:
            return {
                "kind": "list",
                "items": [
                    add_array(
                        _array_name(name, *path, "rnn_states", str(index)),
                        np.full(
                            (1, 14, 1),
                            case_index * 10 + index,
                            dtype=np.float32,
                        ),
                        comparison,
                    )
                    for index in range(2)
                ],
            }

        batch_dict_items: dict[str, object] = {}
        for field_index, field in enumerate(BATCH_FIELDS):
            if field == "rnn_states":
                batch_dict_items[field] = rnn_state_tree(
                    "input", "batch", "batch_dict", comparison="exact"
                )
            else:
                batch_dict_items[field] = add_array(
                    _array_name(name, "input", "batch", "batch_dict", field),
                    row_array(field, field_index),
                    "exact",
                )
        batch_refs = {
            "batch_dict": {
                "kind": "dict",
                "items": batch_dict_items,
            },
            "ps_extras": {
                "kind": "dict",
                "items": {
                    "mb_intr_rewards": {"kind": "none"},
                    "rewards": add_array(
                        _array_name(name, "input", "batch", "ps_extras", "rewards"),
                        base_float,
                        "exact",
                    ),
                },
            },
        }
        rms_refs: dict[str, dict[str, tuple[dict[str, object], dict[str, object]]]] = {
            "input": {},
            "output": {},
        }
        for phase, comparison in (("input", "exact"), ("output", "numeric")):
            for role_index, role in enumerate(RMS_ROLES):
                rms_value = np.array([case_index, role_index, phase == "output"], dtype=np.float32)
                rms_refs[phase][role] = (
                    add_array(
                        _array_name(name, phase, "rms", role, "mean"),
                        rms_value,
                        comparison,
                    ),
                    add_array(
                        _array_name(name, phase, "rms", role, "var"),
                        rms_value + np.float32(1),
                        comparison,
                    ),
                )
        prepared_refs: dict[str, dict[str, object]] = {}
        for owner, fields in PREPARED_FIELDS.items():
            prepared_items: dict[str, object] = {}
            for field_index, field in enumerate(fields):
                if field == "rnn_masks":
                    prepared_items[field] = {"kind": "none"}
                elif field == "rnn_states":
                    prepared_items[field] = rnn_state_tree(
                        "output", "prepared", owner, comparison="numeric"
                    )
                else:
                    value = row_array(field, field_index)
                    comparison = "numeric" if np.issubdtype(value.dtype, np.floating) else "exact"
                    prepared_items[field] = add_array(
                        _prepared_name(name, owner, field),
                        value,
                        comparison,
                    )
            prepared_refs[owner] = {"kind": "dict", "items": prepared_items}
        rng_refs = {
            phase: (
                add_array(
                    _array_name(name, phase, "rng", "numpy", "keys"),
                    np.arange(8, dtype=np.uint32),
                    "exact",
                ),
                add_array(
                    _array_name(name, phase, "rng", "torch_cpu"),
                    np.arange(16, dtype=np.uint8),
                    "exact",
                ),
                add_array(
                    _array_name(name, phase, "rng", "torch_cuda", "0"),
                    np.arange(16, dtype=np.uint8),
                    "exact",
                ),
            )
            for phase in ("input", "output")
        }
        input_models = {
            owner: _parameter_state(f"{name}:input:{owner}") for owner in ("actor", "central")
        }
        input_optimizers = {
            owner: _optimizer_state(f"{name}:input:{owner}", 0) for owner in ("actor", "central")
        }
        output_models = copy.deepcopy(input_models)
        output_optimizers = copy.deepcopy(input_optimizers)
        if not overflow:
            output_models = {
                owner: _parameter_state(f"{name}:output:{owner}") for owner in ("actor", "central")
            }
            output_optimizers = {
                "actor": _optimizer_state(f"{name}:output:actor", actor_steps),
                "central": _optimizer_state(f"{name}:output:central", 8),
            }
        scale_before = 65536.0 if amp else None
        scale_after = scale_before / 2 if amp else None
        native_lengths = {
            "a_losses": 1 if overflow else 8,
            "c_losses": 1 if overflow else 8,
            "b_losses": 1 if overflow else 8,
            "entropies": 1 if overflow else 8,
            "kls": 1 if overflow else 2,
        }
        native_return = {
            "kind": "dict",
            "items": {
                **{
                    field: {
                        "kind": "list",
                        "items": [
                            add_array(
                                _array_name(
                                    name,
                                    "execution",
                                    "native_return",
                                    field,
                                    str(index),
                                ),
                                np.asarray(case_index + index / 100, dtype=np.float32),
                                "numeric",
                            )
                            for index in range(length)
                        ],
                    }
                    for field, length in native_lengths.items()
                },
                "last_lr": {"kind": "scalar", "value": 1e-4},
                "lr_mul": {"kind": "scalar", "value": 1.0},
                "excluded_wall_clock_fields": {
                    "kind": "list",
                    "items": [
                        {"kind": "scalar", "value": field}
                        for field in ("play_time", "update_time", "total_time")
                    ],
                },
            },
        }
        case = _minimal_case(name)
        case["input"] = {
            "batch": {
                "kind": "dict",
                "items": {
                    "batch_dict": copy.deepcopy(batch_refs["batch_dict"]),
                    "played_frames": {"kind": "scalar", "value": 48},
                    "step_time": {"kind": "scalar", "value": 0.0},
                    "ps_extras": copy.deepcopy(batch_refs["ps_extras"]),
                },
            },
            "model": input_models,
            "optimizer": input_optimizers,
            "scaler": _scaler_state(amp, scale_before),
            "rms": _rms_state(rms_refs["input"], 0.0001),
            "lr": _lr_state(),
            "rng": _rng_state(*rng_refs["input"]),
        }
        case["execution"] = {
            "identity_shuffle_calls": 0 if overflow else 1,
            "owner_call_order": ["prepare", "actor"]
            if overflow
            else ["prepare", "central", "actor"],
            "actor_update_attempts": 1 if overflow else 8,
            "actor_optimizer_steps": actor_steps,
            "actor_scaler_skips": actor_skips,
            "central_optimizer_steps": 0 if overflow else 8,
            "native_return": native_return,
            "overflow_mutation": "advantages[0]=+inf" if overflow else None,
            "set_train_info_calls": [] if overflow else [{"frame": 48, "owner_is_agent": True}],
            "autocast": {"enabled": amp, "dtype": "torch.float16" if amp else None},
        }
        case["output"] = {
            "prepared": {
                owner: copy.deepcopy(reference) for owner, reference in prepared_refs.items()
            },
            "model": output_models,
            "optimizer": output_optimizers,
            "scaler": _scaler_state(amp, scale_after),
            "rms": _rms_state(rms_refs["output"], 0.0001 if overflow else 56.0001),
            "lr": _lr_state(),
            "rng": _rng_state(*rng_refs["output"]),
        }
        cases.append(case)
    manifest["cases"] = cases
    manifest["npz_arrays"] = metadata
    manifest["exact_comparison_inventory"] = inventories["exact"]
    manifest["numeric_comparison_inventory"] = inventories["numeric"]
    return {
        "manifest": manifest,
        "arrays": arrays,
    }


def comparison_pair() -> tuple[UpdateFixture, dict[str, object]]:
    source_capture = complete_synthetic_capture()
    source_fixture = _update_fixture(source_capture["manifest"], source_capture["arrays"])
    target_capture = copy.deepcopy(source_capture)
    return source_fixture, target_capture


def _set_array_payload(
    manifest: dict[str, object],
    arrays: dict[str, np.ndarray],
    name: str,
    value: np.ndarray,
) -> None:
    comparison = manifest["npz_arrays"][name]["comparison"]
    arrays[name] = np.ascontiguousarray(value)
    manifest["npz_arrays"][name] = _array_record(name, arrays[name], comparison)


def _replace_capture_array(
    capture: dict[str, object], reference: dict[str, object], value: np.ndarray
) -> None:
    assert reference["kind"] == "array"
    name = reference["name"]
    manifest = capture["manifest"]
    comparison = manifest["npz_arrays"][name]["comparison"]
    _set_array_payload(manifest, capture["arrays"], name, value)
    assert name in manifest[f"{comparison}_comparison_inventory"]


def _drop_capture_array(capture: dict[str, object], reference: dict[str, object]) -> None:
    assert reference["kind"] == "array"
    name = reference["name"]
    manifest = capture["manifest"]
    record = manifest["npz_arrays"].pop(name)
    capture["arrays"].pop(name)
    manifest[f"{record['comparison']}_comparison_inventory"].remove(name)


def _add_capture_array(
    capture: dict[str, object], name: str, value: np.ndarray, comparison: str
) -> dict[str, object]:
    manifest = capture["manifest"]
    assert name not in capture["arrays"]
    capture["arrays"][name] = np.ascontiguousarray(value)
    manifest["npz_arrays"][name] = _array_record(name, capture["arrays"][name], comparison)
    manifest[f"{comparison}_comparison_inventory"].append(name)
    return {"kind": "array", "name": name}


def test_in_memory_fixture_has_complete_ordered_state_contract() -> None:
    fixture = in_memory_update_fixture()
    assert set(fixture.manifest) == MANIFEST_KEYS
    assert len(fixture.arrays) == 1
    array_name, array = next(iter(fixture.arrays.items()))
    assert array.ndim == 2 and np.issubdtype(array.dtype, np.floating)
    assert set(fixture.manifest["npz_arrays"][array_name]) == set(
        "name shape dtype sha256 comparison".split()
    )
    assert [case["name"] for case in fixture.manifest["cases"]] == list(CASE_NAMES)
    for case in fixture.manifest["cases"]:
        assert set(case) == set("name config owners input execution output restore".split())
        assert case["owners"] == OWNERS
        assert set(case["input"]) == set(INPUT_KEYS)
        assert set(case["output"]) == set(OUTPUT_KEYS)


@pytest.mark.parametrize("drift", ["missing", "extra", "shape", "dtype", "content"])
def test_fixture_array_drift_is_rejected(drift: str) -> None:
    fixture = in_memory_update_fixture()
    with pytest.raises(RuntimeError, match=drift):
        harness._validate_array_inventory(
            mutated_arrays(fixture, drift), fixture.manifest["npz_arrays"]
        )


def test_update_fixture_has_only_manifest_and_arrays_fields() -> None:
    assert tuple(UpdateFixture.__dataclass_fields__) == ("manifest", "arrays")


def test_snapshot_store_rejects_duplicate_array_names() -> None:
    store = harness.SnapshotStore()
    store.tree("same", np.ones((2, 2), dtype=np.float32), comparison="numeric")
    with pytest.raises(RuntimeError, match="duplicate snapshot array: same"):
        store.tree("same", np.zeros((2, 2), dtype=np.float32), comparison="numeric")


def test_snapshot_store_tags_nonfinite_scalars() -> None:
    store = harness.SnapshotStore()
    assert store.tree(
        "values", [float("nan"), float("inf"), -float("inf")], comparison="exact"
    ) == {
        "kind": "list",
        "items": [
            {"kind": "scalar", "value": {"nonfinite": "nan"}},
            {"kind": "scalar", "value": {"nonfinite": "+inf"}},
            {"kind": "scalar", "value": {"nonfinite": "-inf"}},
        ],
    }
    with pytest.raises(ValueError):
        harness.canonical_payload({"raw": float("nan")})


def test_parameter_snapshot_covers_every_named_parameter() -> None:
    model = torch.nn.Linear(3, 2)
    with torch.no_grad():
        model.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        model.bias.copy_(torch.tensor([7.0, 8.0]))

    snapshot = harness.snapshot_parameters(model)

    assert set(snapshot) == {"parameters", "aggregate_sha256"}
    assert list(snapshot["parameters"]) == sorted(dict(model.named_parameters()))
    for name, parameter in model.named_parameters():
        array = parameter.detach().cpu().numpy()
        assert snapshot["parameters"][name] == {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _sha256(array),
        }
    assert len(snapshot["aggregate_sha256"]) == 64
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert harness.snapshot_parameters(model)["aggregate_sha256"] != snapshot["aggregate_sha256"]


def test_optimizer_snapshot_covers_groups_initialized_state_and_missing_state() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()

    named_parameters = dict(model.named_parameters())
    snapshot = harness.snapshot_optimizer(optimizer, named_parameters)

    assert set(snapshot) == {
        "param_groups",
        "state",
        "uninitialized",
        "aggregate_sha256",
    }
    assert len(snapshot["param_groups"]) == 1
    group = snapshot["param_groups"][0]
    assert group["params"] == ["weight", "bias"]
    assert set(group) == (set(optimizer.param_groups[0]) - {"params"}) | {"params"}
    assert group["lr"] == pytest.approx(0.01)
    assert set(snapshot["state"]) == {"weight"}
    assert snapshot["uninitialized"] == ["bias"]
    assert set(snapshot["state"]["weight"]) == set(optimizer.state[model.weight])
    for state_name, value in optimizer.state[model.weight].items():
        array = value.detach().cpu().numpy()
        expected = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _sha256(array),
        }
        if state_name == "step":
            expected["value"] = int(value.item())
        assert snapshot["state"]["weight"][state_name] == expected
    assert len(snapshot["aggregate_sha256"]) == 64
    optimizer.step()
    changed = harness.snapshot_optimizer(optimizer, named_parameters)
    assert changed["aggregate_sha256"] != snapshot["aggregate_sha256"]


@pytest.mark.parametrize(
    ("prefix", "comparison"),
    [("input__rms", "exact"), ("output__rms", "numeric")],
)
def test_rms_snapshot_covers_all_four_roles_and_full_state(prefix: str, comparison: str) -> None:
    snapshot_rms = harness.snapshot_rms
    store = harness.SnapshotStore()

    class TinyRms(torch.nn.Module):
        def __init__(self, offset: float) -> None:
            super().__init__()
            self.register_buffer("running_mean", torch.tensor([offset, offset + 1]))
            self.register_buffer("running_var", torch.tensor([offset + 2, offset + 3]))
            self.register_buffer("count", torch.tensor(offset + 4, dtype=torch.float64))

    roles = {
        "actor_input": TinyRms(0),
        "central_input": TinyRms(10),
        "actor_model_value": TinyRms(20),
        "active_central_value": TinyRms(30),
    }
    roles["actor_model_value"].eval()

    snapshot = snapshot_rms(store, roles, prefix)

    assert list(snapshot) == list(roles)
    for role, rms in roles.items():
        state = snapshot[role]
        assert set(state) == {"mean", "var", "count", "training"}
        np.testing.assert_array_equal(
            _stored_array(store, state["mean"]), rms.running_mean.detach().cpu().numpy()
        )
        np.testing.assert_array_equal(
            _stored_array(store, state["var"]), rms.running_var.detach().cpu().numpy()
        )
        assert store.metadata[state["mean"]["name"]]["comparison"] == comparison
        assert store.metadata[state["var"]["name"]]["comparison"] == comparison
        assert state["count"] == float(rms.count.item())
        assert state["training"] is rms.training


def test_rng_snapshot_covers_numpy_torch_cpu_and_every_cuda_device() -> None:
    snapshot_rng = harness.snapshot_rng
    store = harness.SnapshotStore()
    numpy_state = np.random.get_state()
    torch_cpu_state = torch.get_rng_state().cpu().numpy().copy()
    torch_cuda_states = [state.cpu().numpy().copy() for state in torch.cuda.get_rng_state_all()]

    snapshot = snapshot_rng(store, "input__rng")

    assert set(snapshot) == {"numpy", "torch_cpu", "torch_cuda"}
    assert set(snapshot["numpy"]) == {
        "algorithm",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    assert snapshot["numpy"]["algorithm"] == numpy_state[0]
    np.testing.assert_array_equal(_stored_array(store, snapshot["numpy"]["keys"]), numpy_state[1])
    assert snapshot["numpy"]["position"] == numpy_state[2]
    assert snapshot["numpy"]["has_gauss"] == numpy_state[3]
    assert snapshot["numpy"]["cached_gaussian"] == numpy_state[4]
    np.testing.assert_array_equal(_stored_array(store, snapshot["torch_cpu"]), torch_cpu_state)
    assert len(snapshot["torch_cuda"]) == torch.cuda.device_count()
    for reference, expected in zip(snapshot["torch_cuda"], torch_cuda_states, strict=True):
        np.testing.assert_array_equal(_stored_array(store, reference), expected)
    for metadata in store.metadata.values():
        assert metadata["comparison"] == "exact"


def test_lr_snapshot_covers_actor_and_central_native_owners() -> None:
    class TinyScheduler:
        def __init__(self, epoch: int, factor: float) -> None:
            self.epoch = epoch
            self.factor = factor
            self.enabled = True

    actor_left = torch.nn.Parameter(torch.zeros(1))
    actor_right = torch.nn.Parameter(torch.ones(1))
    actor_optimizer = torch.optim.SGD(
        [
            {"params": [actor_left], "lr": 0.001},
            {"params": [actor_right], "lr": 0.002},
        ]
    )
    central_parameter = torch.nn.Parameter(torch.full((1,), 2.0))
    central_optimizer = torch.optim.SGD([central_parameter], lr=0.003)
    actor_scheduler = TinyScheduler(epoch=4, factor=0.5)
    central_scheduler = TinyScheduler(epoch=7, factor=0.25)
    agent = SimpleNamespace(
        last_lr=0.001,
        optimizer=actor_optimizer,
        scheduler=actor_scheduler,
        central_value_net=SimpleNamespace(
            lr=0.003,
            optimizer=central_optimizer,
            scheduler=central_scheduler,
        ),
    )

    snapshot = harness.snapshot_lr(agent)

    assert set(snapshot) == {"actor", "central"}
    assert snapshot["actor"]["last_lr"] == pytest.approx(0.001)
    assert snapshot["actor"]["optimizer_group_lrs"] == pytest.approx([0.001, 0.002])
    assert snapshot["actor"]["scheduler_class"].endswith("TinyScheduler")
    assert snapshot["actor"]["scheduler_state"] == vars(actor_scheduler)
    assert snapshot["central"]["lr"] == pytest.approx(0.003)
    assert snapshot["central"]["optimizer_group_lrs"] == pytest.approx([0.003])
    assert snapshot["central"]["scheduler_class"].endswith("TinyScheduler")
    assert snapshot["central"]["scheduler_state"] == vars(central_scheduler)


def test_validate_capture_accepts_complete_synthetic_state_contract() -> None:
    harness.validate_capture(complete_synthetic_capture())


def test_synthetic_capture_matches_native_shape_contract_relationships() -> None:
    capture = complete_synthetic_capture()
    manifest = capture["manifest"]
    arrays = capture["arrays"]
    contract = manifest["capture_contract"]
    assert contract["rows"] == 56
    assert contract["sequences"] == 14
    assert contract["actor_dataset_batch_sizes"] == [12, 12, 12, 20]
    assert contract["central_dataset_batch_sizes"] == [12, 12, 12, 20]
    assert sum(contract["actor_dataset_batch_sizes"]) == contract["rows"]
    assert sum(contract["central_dataset_batch_sizes"]) == contract["rows"]
    assert contract["rows"] / contract["sequences"] == 4
    assert len(manifest["exact_comparison_inventory"]) == 90
    assert len(manifest["numeric_comparison_inventory"]) == 142

    for case in manifest["cases"]:
        batch_items = case["input"]["batch"]["items"]
        batch_dict = batch_items["batch_dict"]["items"]
        for field in set(BATCH_FIELDS) - {"rnn_states"}:
            reference = batch_dict[field]
            assert reference["kind"] == "array"
            assert arrays[reference["name"]].shape[0] == contract["rows"]
        input_rnn_states = batch_dict["rnn_states"]
        assert input_rnn_states["kind"] == "list"
        assert len(input_rnn_states["items"]) == 2
        assert all(
            reference["kind"] == "array"
            and arrays[reference["name"]].ndim == 3
            and arrays[reference["name"]].shape[1] == contract["sequences"]
            for reference in input_rnn_states["items"]
        )
        rewards = batch_items["ps_extras"]["items"]["rewards"]
        assert rewards["kind"] == "array"
        assert arrays[rewards["name"]].shape[0] == contract["rows"]

        prepared = case["output"]["prepared"]
        for owner in ("actor", "central"):
            items = prepared[owner]["items"]
            assert items["rnn_masks"] == {"kind": "none"}
            ordinary_fields = set(PREPARED_FIELDS[owner]) - {"rnn_masks", "rnn_states"}
            for field in ordinary_fields:
                reference = items[field]
                assert reference["kind"] == "array"
                assert arrays[reference["name"]].shape[0] == contract["rows"]
        actor_rnn_states = prepared["actor"]["items"]["rnn_states"]
        assert actor_rnn_states["kind"] == "list"
        assert len(actor_rnn_states["items"]) == 2
        assert all(
            reference["kind"] == "array"
            and arrays[reference["name"]].ndim == 3
            and arrays[reference["name"]].shape[1] == contract["sequences"]
            for reference in actor_rnn_states["items"]
        )


@pytest.mark.parametrize(
    ("drift", "expected_error"),
    [
        (
            "input-row",
            r"normal_fp32\.batch\.batch_dict\.actions row count mismatch",
        ),
        (
            "rewards-row",
            r"normal_fp32\.batch\.ps_extras\.rewards row count mismatch",
        ),
        (
            "input-rnn-sequences",
            r"normal_fp32\.batch\.batch_dict\.rnn_states\[0\] sequence count mismatch",
        ),
        (
            "actor-row",
            r"normal_fp32\.output\.prepared\.actor\.actions row count mismatch",
        ),
        (
            "central-row",
            r"normal_fp32\.output\.prepared\.central\.actions row count mismatch",
        ),
        (
            "actor-rnn-sequences",
            r"normal_fp32\.output\.prepared\.actor\.rnn_states\[0\] sequence count mismatch",
        ),
    ],
)
def test_validate_capture_rejects_native_shape_contract_drift(
    drift: str, expected_error: str
) -> None:
    capture = complete_synthetic_capture()
    normal = capture["manifest"]["cases"][0]
    batch_items = normal["input"]["batch"]["items"]
    batch_dict = batch_items["batch_dict"]["items"]
    prepared = normal["output"]["prepared"]
    if drift == "input-row":
        reference = batch_dict["actions"]
        replacement = capture["arrays"][reference["name"]][:55]
    elif drift == "rewards-row":
        reference = batch_items["ps_extras"]["items"]["rewards"]
        replacement = capture["arrays"][reference["name"]][:55]
    elif drift == "input-rnn-sequences":
        reference = batch_dict["rnn_states"]["items"][0]
        replacement = capture["arrays"][reference["name"]][:, :13, :]
    elif drift == "actor-row":
        reference = prepared["actor"]["items"]["actions"]
        replacement = capture["arrays"][reference["name"]][:55]
    elif drift == "central-row":
        reference = prepared["central"]["items"]["actions"]
        replacement = capture["arrays"][reference["name"]][:55]
    else:
        reference = prepared["actor"]["items"]["rnn_states"]["items"][0]
        replacement = capture["arrays"][reference["name"]][:, :13, :]
    _replace_capture_array(capture, reference, replacement)

    with pytest.raises(RuntimeError, match=expected_error):
        harness.validate_capture(capture)


@pytest.mark.parametrize(
    ("drift", "expected_error"),
    [
        (
            "input-rnn-array",
            r"normal_fp32\.batch\.batch_dict\.rnn_states must be a two-item list tree",
        ),
        (
            "actor-rnn-array",
            r"normal_fp32\.output\.prepared\.actor\.rnn_states must be a two-item list tree",
        ),
        (
            "actor-rnn-mask-array",
            r"normal_fp32\.output\.prepared\.actor\.rnn_masks must be none",
        ),
        (
            "central-rnn-mask-array",
            r"normal_fp32\.output\.prepared\.central\.rnn_masks must be none",
        ),
    ],
)
def test_validate_capture_rejects_native_prepared_tree_kind_drift(
    drift: str, expected_error: str
) -> None:
    capture = complete_synthetic_capture()
    normal = capture["manifest"]["cases"][0]
    batch_dict = normal["input"]["batch"]["items"]["batch_dict"]["items"]
    prepared = normal["output"]["prepared"]
    if drift in {"input-rnn-array", "actor-rnn-array"}:
        container = batch_dict if drift == "input-rnn-array" else prepared["actor"]["items"]
        rnn_states = container["rnn_states"]
        kept, removed = rnn_states["items"]
        container["rnn_states"] = kept
        _drop_capture_array(capture, removed)
    else:
        owner = "actor" if drift == "actor-rnn-mask-array" else "central"
        name = _array_name("normal_fp32", "output", "prepared", owner, "rnn_masks", "unexpected")
        prepared[owner]["items"]["rnn_masks"] = _add_capture_array(
            capture,
            name,
            np.ones((56, 1), dtype=np.float32),
            "numeric",
        )

    with pytest.raises(RuntimeError, match=expected_error):
        harness.validate_capture(capture)


def test_synthetic_capture_has_case_and_role_scoped_array_ownership() -> None:
    capture = complete_synthetic_capture()
    manifest = capture["manifest"]
    arrays = capture["arrays"]
    metadata = manifest["npz_arrays"]
    exact = set(manifest["exact_comparison_inventory"])
    numeric = set(manifest["numeric_comparison_inventory"])
    assert exact.isdisjoint(numeric)
    assert exact | numeric == set(arrays) == set(metadata)
    assert {name for name, record in metadata.items() if record["comparison"] == "exact"} == exact
    assert {
        name for name, record in metadata.items() if record["comparison"] == "numeric"
    } == numeric

    all_case_references: set[str] = set()
    for case in manifest["cases"]:
        case_name = case["name"]
        input_references = _array_reference_names(case["input"])
        execution_references = _array_reference_names(case["execution"])
        output_references = _array_reference_names(case["output"])
        case_references = input_references + execution_references + output_references
        assert len(case_references) == len(set(case_references))
        assert all(name.startswith(f"{case_name}__") for name in case_references)
        assert all_case_references.isdisjoint(case_references)
        all_case_references.update(case_references)

        batch_items = case["input"]["batch"]["items"]
        batch_dict = batch_items["batch_dict"]
        assert batch_dict["kind"] == "dict"
        assert set(batch_dict["items"]) == set(BATCH_FIELDS)
        batch_dict_names = set(_array_reference_names(batch_dict))
        expected_batch_names = {
            _array_name(case_name, "input", "batch", "batch_dict", field)
            for field in set(BATCH_FIELDS) - {"rnn_states"}
        }
        expected_batch_names.update(
            _array_name(
                case_name,
                "input",
                "batch",
                "batch_dict",
                "rnn_states",
                str(index),
            )
            for index in range(2)
        )
        assert batch_dict_names == expected_batch_names
        ps_extras = batch_items["ps_extras"]
        assert ps_extras["kind"] == "dict"
        assert set(ps_extras["items"]) == set(PS_EXTRA_FIELDS)
        assert ps_extras["items"]["mb_intr_rewards"] == {"kind": "none"}
        ps_extra_names = set(_array_reference_names(ps_extras))
        assert ps_extra_names == {_array_name(case_name, "input", "batch", "ps_extras", "rewards")}
        assert batch_dict_names.isdisjoint(ps_extra_names)
        assert batch_dict_names | ps_extra_names <= exact
        for phase in ("input", "output"):
            rng_names = _array_reference_names(case[phase]["rng"])
            assert len(rng_names) == len(set(rng_names)) == 3
            assert set(rng_names) <= exact

        prepared_names: set[str] = set()
        for owner, fields in PREPARED_FIELDS.items():
            tree = case["output"]["prepared"][owner]
            assert tree["kind"] == "dict"
            assert set(tree["items"]) == set(fields)
            owner_names = set(_array_reference_names(tree))
            expected_owner_names = {
                _prepared_name(case_name, owner, field)
                for field in set(fields) - {"rnn_masks", "rnn_states"}
            }
            if owner == "actor":
                expected_owner_names.update(
                    _array_name(
                        case_name,
                        "output",
                        "prepared",
                        owner,
                        "rnn_states",
                        str(index),
                    )
                    for index in range(2)
                )
            assert owner_names == expected_owner_names
            assert prepared_names.isdisjoint(owner_names)
            prepared_names.update(owner_names)
        assert prepared_names <= exact | numeric

        native_return_names = set(_array_reference_names(case["execution"]["native_return"]))
        assert len(native_return_names) == (5 if case_name == "overflow_amp" else 34)
        assert native_return_names <= numeric

        for phase, comparison in (("input", "exact"), ("output", "numeric")):
            rms = case[phase]["rms"]
            assert tuple(rms) == RMS_ROLES
            rms_names = [
                rms[role][statistic]["name"] for role in RMS_ROLES for statistic in ("mean", "var")
            ]
            assert len(rms_names) == len(set(rms_names)) == 2 * len(RMS_ROLES)
            assert all(metadata[name]["comparison"] == comparison for name in rms_names)

        assert all(metadata[name]["comparison"] == "exact" for name in input_references)
        for name in output_references:
            comparison = "numeric" if np.issubdtype(arrays[name].dtype, np.floating) else "exact"
            assert metadata[name]["comparison"] == comparison

    assert all_case_references == set(arrays)
    harness.validate_capture(capture)


@pytest.mark.parametrize("drift", ["missing-rms-role", "input-float-domain"])
def test_validate_capture_rejects_case_role_or_domain_drift(drift: str) -> None:
    capture = complete_synthetic_capture()
    normal = capture["manifest"]["cases"][0]
    if drift == "missing-rms-role":
        del normal["output"]["rms"]["central_input"]
    else:
        input_names = _array_reference_names(normal["input"]["batch"]["items"]["batch_dict"])
        assert input_names
        input_name = input_names[0]
        capture["manifest"]["npz_arrays"][input_name]["comparison"] = "numeric"
        capture["manifest"]["exact_comparison_inventory"].remove(input_name)
        capture["manifest"]["numeric_comparison_inventory"].append(input_name)

    with pytest.raises(RuntimeError):
        harness.validate_capture(capture)


def test_validate_capture_accepts_source_observed_amp_step_skip_counts() -> None:
    capture = complete_synthetic_capture()
    amp = capture["manifest"]["cases"][1]
    amp["execution"]["actor_optimizer_steps"] = 6
    amp["execution"]["actor_scaler_skips"] = 2
    amp["output"]["optimizer"]["actor"] = _optimizer_state("normal_amp:observed", 6)
    amp["output"]["scaler"]["state_dict"]["scale"] = 16384.0
    harness.validate_capture(capture)


def test_validate_capture_accepts_overflow_value_loss_in_native_return() -> None:
    capture = complete_synthetic_capture()
    manifest = capture["manifest"]
    overflow = manifest["cases"][2]
    c_losses = overflow["execution"]["native_return"]["items"]["c_losses"]
    assert c_losses["kind"] == "list"
    assert len(c_losses["items"]) == 1
    reference = c_losses["items"][0]
    assert reference["kind"] == "array"
    name = reference["name"]
    assert manifest["npz_arrays"][name]["comparison"] == "numeric"
    assert np.issubdtype(capture["arrays"][name].dtype, np.floating)

    harness.validate_capture(capture)


@pytest.mark.parametrize(
    ("owner", "field"),
    [("actor", "sigma"), ("central", "rnn_masks")],
)
def test_validate_capture_rejects_missing_prepared_owner_field(owner: str, field: str) -> None:
    capture = complete_synthetic_capture()
    prepared = capture["manifest"]["cases"][0]["output"]["prepared"]
    del prepared[owner]["items"][field]

    with pytest.raises(
        RuntimeError,
        match=rf"normal_fp32 prepared {owner} field inventory mismatch",
    ):
        harness.validate_capture(capture)


@pytest.mark.parametrize(
    ("node", "expected_label"),
    [
        ("batch_dict", "normal_fp32.batch.batch_dict"),
        ("ps_extras", "normal_fp32.batch.ps_extras"),
        ("prepared_actor", "normal_fp32.output.prepared.actor"),
        ("prepared_central", "normal_fp32.output.prepared.central"),
    ],
)
def test_validate_capture_rejects_required_dict_tree_downgrade(
    node: str, expected_label: str
) -> None:
    capture = complete_synthetic_capture()
    manifest = capture["manifest"]
    normal = manifest["cases"][0]
    batch_items = normal["input"]["batch"]["items"]
    targets = {
        "batch_dict": (batch_items, "batch_dict"),
        "ps_extras": (batch_items, "ps_extras"),
        "prepared_actor": (normal["output"]["prepared"], "actor"),
        "prepared_central": (normal["output"]["prepared"], "central"),
    }
    container, key = targets[node]
    references = _array_reference_names(container[key])
    assert references
    kept_reference, *removed_references = references
    container[key] = {"kind": "array", "name": kept_reference}
    for name in removed_references:
        record = manifest["npz_arrays"].pop(name)
        capture["arrays"].pop(name)
        manifest[f"{record['comparison']}_comparison_inventory"].remove(name)

    with pytest.raises(RuntimeError, match=rf"{expected_label} must be a dict tree"):
        harness.validate_capture(capture)


def test_validate_capture_rejects_incomplete_optimizer_state_with_stale_aggregate() -> None:
    capture = complete_synthetic_capture()
    optimizer = capture["manifest"]["cases"][0]["output"]["optimizer"]["actor"]
    state = optimizer["state"]["weight"]
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    stale_aggregate = optimizer["aggregate_sha256"]
    del state["exp_avg"]
    assert optimizer["aggregate_sha256"] == stale_aggregate

    with pytest.raises(RuntimeError, match=r"optimizer state-key inventory mismatch"):
        harness.validate_capture(capture)


def test_validate_capture_rejects_enabled_scaler_with_scale_only_state() -> None:
    capture = complete_synthetic_capture()
    amp = capture["manifest"]["cases"][1]
    for phase in ("input", "output"):
        state = amp[phase]["scaler"]["state_dict"]
        assert set(state) == {
            "scale",
            "growth_factor",
            "backoff_factor",
            "growth_interval",
            "_growth_tracker",
        }
        amp[phase]["scaler"]["state_dict"] = {"scale": state["scale"]}

    with pytest.raises(RuntimeError, match=r"state_dict keys mismatch"):
        harness.validate_capture(capture)


def test_validate_capture_rejects_raw_native_return_root() -> None:
    capture = complete_synthetic_capture()
    manifest = capture["manifest"]
    normal = manifest["cases"][0]
    native_return = normal["execution"]["native_return"]
    references = _array_reference_names(native_return)
    assert len(references) == 34
    for name in references:
        record = manifest["npz_arrays"].pop(name)
        capture["arrays"].pop(name)
        manifest[f"{record['comparison']}_comparison_inventory"].remove(name)
    normal["execution"]["native_return"] = {
        "a_losses": [0.0] * 8,
        "c_losses": [0.0] * 8,
        "b_losses": [0.0] * 8,
        "entropies": [0.0] * 8,
        "kls": [0.0] * 2,
        "last_lr": 1e-4,
        "lr_mul": 1.0,
        "excluded_wall_clock_fields": ["play_time", "update_time", "total_time"],
    }

    with pytest.raises(RuntimeError, match=r"execution.native_return must be a dict tree"):
        harness.validate_capture(capture)


def test_validate_capture_rejects_tree_native_return_wrong_exclusions() -> None:
    capture = complete_synthetic_capture()
    native_return = capture["manifest"]["cases"][0]["execution"]["native_return"]
    excluded = native_return["items"]["excluded_wall_clock_fields"]
    assert [item["value"] for item in excluded["items"]] == [
        "play_time",
        "update_time",
        "total_time",
    ]
    excluded["items"][0]["value"] = "wrong"

    with pytest.raises(RuntimeError, match=r"excluded wall-clock fields mismatch"):
        harness.validate_capture(capture)


def test_validate_capture_rejects_exact_native_return_float_arrays() -> None:
    capture = complete_synthetic_capture()
    manifest = capture["manifest"]
    native_return = manifest["cases"][0]["execution"]["native_return"]
    references = _array_reference_names(native_return)
    assert len(references) == 34
    for name in references:
        assert manifest["npz_arrays"][name]["comparison"] == "numeric"
        manifest["npz_arrays"][name]["comparison"] = "exact"
        manifest["numeric_comparison_inventory"].remove(name)
        manifest["exact_comparison_inventory"].append(name)

    with pytest.raises(RuntimeError, match=r"native return must compare numerically"):
        harness.validate_capture(capture)


def test_validate_capture_rejects_parameter_record_with_stale_aggregate() -> None:
    capture = complete_synthetic_capture()
    parameters = capture["manifest"]["cases"][0]["output"]["model"]["actor"]
    stale_aggregate = parameters["aggregate_sha256"]
    parameters["parameters"]["weight"]["sha256"] = "f" * 64
    assert parameters["aggregate_sha256"] == stale_aggregate

    with pytest.raises(RuntimeError, match=r"parameter aggregate SHA256 mismatch"):
        harness.validate_capture(capture)


@pytest.mark.parametrize(
    "drift",
    [
        "missing-output",
        "owner-order",
        "training-env-abi",
        "step-skip-arithmetic",
        "owners-missing",
        "owners-extra",
        "owners-value",
        "numeric-inventory",
        "extra-top-level",
    ],
)
def test_validate_capture_rejects_synthetic_contract_drift(drift: str) -> None:
    capture = complete_synthetic_capture()
    normal = capture["manifest"]["cases"][0]
    if drift == "missing-output":
        del normal["output"]["rng"]
    elif drift == "owner-order":
        normal["execution"]["owner_call_order"] = ["actor", "central", "prepare"]
    elif drift == "training-env-abi":
        normal["execution"]["set_train_info_calls"] = []
    elif drift == "step-skip-arithmetic":
        normal["execution"]["actor_optimizer_steps"] = 7
    elif drift == "owners-missing":
        del normal["owners"]["central_value"]
    elif drift == "owners-extra":
        normal["owners"]["legacy"] = "rl_games.legacy.LegacyOwner"
    elif drift == "owners-value":
        normal["owners"]["agent"] = "rl_games.algos_torch.a2c_continuous.NotA2CAgent"
    elif drift == "numeric-inventory":
        capture["manifest"]["numeric_comparison_inventory"].remove(
            _prepared_name("overflow_amp", "actor")
        )
    else:
        capture["legacy_evidence"] = {}
    with pytest.raises(RuntimeError):
        harness.validate_capture(capture)


def test_validate_capture_rejects_cross_case_prepared_array_ownership() -> None:
    capture = complete_synthetic_capture()
    amp_actor = capture["manifest"]["cases"][1]["output"]["prepared"]["actor"]["items"]
    obs_reference = amp_actor["obs"]
    assert obs_reference["kind"] == "array"
    obs_reference["name"] = _prepared_name("normal_fp32", "actor", "obs")

    with pytest.raises(RuntimeError, match=r"case normal_amp owns an array from another case"):
        harness.validate_capture(capture)


def test_compare_capture_uses_exact_and_numeric_domains(monkeypatch) -> None:
    source, target = comparison_pair()
    numeric_name = source.manifest["numeric_comparison_inventory"][0]
    target["arrays"][numeric_name] += np.float32(5e-7)
    target["manifest"]["npz_arrays"][numeric_name]["sha256"] = _sha256(
        target["arrays"][numeric_name]
    )
    monkeypatch.setattr(harness, "validate_capture", lambda _capture: None, raising=False)

    result = harness.compare_capture(source, target)

    assert result.case_names == CASE_NAMES
    assert result.exact_array_count == len(source.manifest["exact_comparison_inventory"])
    assert result.numeric_array_count == len(source.manifest["numeric_comparison_inventory"])
    assert result.max_abs_error == pytest.approx(5e-7, rel=0.1)


def test_compare_capture_rejects_exact_array_drift(monkeypatch) -> None:
    source, target = comparison_pair()
    exact_name = source.manifest["exact_comparison_inventory"][0]
    target["arrays"][exact_name][0] += 1
    target["manifest"]["npz_arrays"][exact_name]["sha256"] = _sha256(target["arrays"][exact_name])
    monkeypatch.setattr(harness, "validate_capture", lambda _capture: None, raising=False)

    with pytest.raises(AssertionError, match=exact_name):
        harness.compare_capture(source, target)


def test_compare_capture_rejects_numeric_drift_outside_tolerance(monkeypatch) -> None:
    source, target = comparison_pair()
    numeric_name = source.manifest["numeric_comparison_inventory"][0]
    target["arrays"][numeric_name] += np.float32(0.1)
    target["manifest"]["npz_arrays"][numeric_name]["sha256"] = _sha256(
        target["arrays"][numeric_name]
    )
    monkeypatch.setattr(harness, "validate_capture", lambda _capture: None, raising=False)

    with pytest.raises(AssertionError):
        harness.compare_capture(source, target)


@pytest.mark.parametrize("drift", ["model", "optimizer", "scaler", "lr", "native_return"])
def test_compare_capture_rejects_json_only_case_state_drift(monkeypatch, drift: str) -> None:
    source, target = comparison_pair()
    target_case = next(case for case in target["manifest"]["cases"] if case["name"] == "normal_amp")
    if drift == "model":
        target_case["output"]["model"]["actor"]["aggregate_sha256"] = "f" * 64
    elif drift == "optimizer":
        target_case["output"]["optimizer"]["actor"]["aggregate_sha256"] = "e" * 64
    elif drift == "scaler":
        target_case["output"]["scaler"]["state_dict"]["scale"] = 123.0
    elif drift == "lr":
        target_case["output"]["lr"]["actor"]["last_lr"] = 2e-4
    else:
        target_case["execution"]["native_return"]["items"]["last_lr"]["value"] = 2e-4
    assert all(
        np.array_equal(source.arrays[name], value) for name, value in target["arrays"].items()
    )
    monkeypatch.setattr(harness, "validate_capture", lambda _capture: None, raising=False)

    with pytest.raises((AssertionError, RuntimeError)):
        harness.compare_capture(source, target)


def test_compare_capture_rejects_normal_amp_drift_against_same_case(monkeypatch) -> None:
    source, target = comparison_pair()
    amp_name = _prepared_name("normal_amp", "actor")
    fp32_name = _prepared_name("normal_fp32", "actor")
    assert not np.array_equal(source.arrays[amp_name], source.arrays[fp32_name])
    target["arrays"][amp_name] += np.float32(0.1)
    target["manifest"]["npz_arrays"][amp_name]["sha256"] = _sha256(target["arrays"][amp_name])
    monkeypatch.setattr(harness, "validate_capture", lambda _capture: None, raising=False)

    with pytest.raises(AssertionError):
        harness.compare_capture(source, target)


@pytest.mark.parametrize("drift", ["source-hash", "target-hash", "incomplete-inventory"])
def test_compare_capture_owns_inventory_validation(monkeypatch, drift: str) -> None:
    source, target = comparison_pair()
    numeric_name = source.manifest["numeric_comparison_inventory"][0]
    if drift == "source-hash":
        source.manifest["npz_arrays"][numeric_name]["sha256"] = "0" * 64
    elif drift == "target-hash":
        target["manifest"]["npz_arrays"][numeric_name]["sha256"] = "0" * 64
    else:
        omitted = source.manifest["exact_comparison_inventory"].pop()
        target["manifest"]["exact_comparison_inventory"].remove(omitted)
    monkeypatch.setattr(harness, "validate_capture", lambda _capture: None, raising=False)

    with pytest.raises(RuntimeError):
        harness.compare_capture(source, target)


@pytest.mark.parametrize("case_name", ["normal_amp", "overflow_amp"])
def test_compare_capture_rejects_missing_case_numeric_inventory(
    monkeypatch, case_name: str
) -> None:
    source, target = comparison_pair()
    omitted = _prepared_name(case_name, "actor")
    source.manifest["numeric_comparison_inventory"].remove(omitted)
    target["manifest"]["numeric_comparison_inventory"].remove(omitted)
    monkeypatch.setattr(harness, "validate_capture", lambda _capture: None, raising=False)

    with pytest.raises(RuntimeError):
        harness.compare_capture(source, target)


@pytest.mark.parametrize("has_finite_values", [True, False], ids=["finite-tail", "all-nonfinite"])
def test_compare_capture_accepts_matching_nonfinite_masks_with_finite_diagnostic(
    has_finite_values: bool,
) -> None:
    source, target = comparison_pair()
    numeric_name = _prepared_name("normal_fp32", "actor", "obs")
    source_array = source.arrays[numeric_name].copy()
    original_shape = source_array.shape
    original_dtype = source_array.dtype
    source_flat = source_array.reshape(-1)
    assert source_flat.size >= 6
    if has_finite_values:
        source_flat[:6] = [np.nan, np.inf, -np.inf, 1.0, 2.0, 3.0]
    else:
        nonfinite = np.array([np.nan, np.inf, -np.inf], dtype=original_dtype)
        source_flat[:] = nonfinite[np.arange(source_flat.size) % len(nonfinite)]
    target_array = source_array.copy()
    if has_finite_values:
        target_array.reshape(-1)[3:6] += np.float32(5e-7)
    assert source_array.shape == target_array.shape == original_shape
    assert source_array.dtype == target_array.dtype == original_dtype
    _set_array_payload(source.manifest, source.arrays, numeric_name, source_array)
    _set_array_payload(target["manifest"], target["arrays"], numeric_name, target_array)

    result = harness.compare_capture(source, target)

    assert np.isfinite(result.max_abs_error)
    expected_error = 5e-7 if has_finite_values else 0.0
    assert result.max_abs_error == pytest.approx(expected_error, rel=0.1)


@pytest.mark.parametrize(
    ("flat_index", "replacement"),
    [(0, 0.0), (1, -np.inf), (2, np.inf)],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_compare_capture_rejects_nonfinite_mask_mismatch(
    flat_index: int, replacement: float
) -> None:
    source, target = comparison_pair()
    numeric_name = _prepared_name("normal_fp32", "actor", "obs")
    source_array = source.arrays[numeric_name].copy()
    original_shape = source_array.shape
    original_dtype = source_array.dtype
    source_array.reshape(-1)[:3] = [np.nan, np.inf, -np.inf]
    target_array = source_array.copy()
    target_array.reshape(-1)[flat_index] = replacement
    assert source_array.shape == target_array.shape == original_shape
    assert source_array.dtype == target_array.dtype == original_dtype
    _set_array_payload(source.manifest, source.arrays, numeric_name, source_array)
    _set_array_payload(target["manifest"], target["arrays"], numeric_name, target_array)

    with pytest.raises(AssertionError, match=r"non-finite mask mismatch"):
        harness.compare_capture(source, target)


@pytest.mark.parametrize("drift", ["model", "optimizer", "scaler"])
def test_validate_capture_rejects_overflow_state_drift(drift: str) -> None:
    capture = complete_synthetic_capture()
    overflow = capture["manifest"]["cases"][2]
    if drift == "model":
        overflow["output"]["model"]["actor"]["aggregate_sha256"] = "f" * 64
    elif drift == "optimizer":
        overflow["output"]["optimizer"]["actor"]["aggregate_sha256"] = "e" * 64
    else:
        overflow["output"]["scaler"]["state_dict"]["scale"] = overflow["input"]["scaler"][
            "state_dict"
        ]["scale"]

    with pytest.raises(RuntimeError, match="overflow"):
        harness.validate_capture(capture)


def test_capture_native_case_closes_writer_when_batch_load_fails(monkeypatch) -> None:
    sentinel = RuntimeError("sentinel batch load failure")

    class Writer:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("writer close must not replace sentinel")

    writer = Writer()
    agent = SimpleNamespace(writer=writer, ppo_device="cpu")
    monkeypatch.setattr(harness, "_agent_and_env", lambda _params, _path: (agent, object()))

    def fail_batch_load(_device):
        raise sentinel

    monkeypatch.setattr(harness, "_load_code3_batch", fail_batch_load)

    with pytest.raises(RuntimeError) as error:
        harness._capture_native_case("normal_fp32", {"config": {}}, harness.SnapshotStore())

    assert error.value is sentinel
    assert writer.close_calls == 1


@pytest.mark.parametrize("failure_stage", ["init_tensors", "env_reset", "owner_check"])
def test_agent_and_env_closes_partial_writer_on_initialization_failure(
    monkeypatch, tmp_path: Path, failure_stage: str
) -> None:
    from rl_games import torch_runner

    sentinel = RuntimeError(f"sentinel {failure_stage} failure")

    class Writer:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class Agent:
        def __init__(self, writer: Writer) -> None:
            self.writer = writer

        def init_tensors(self) -> None:
            if failure_stage == "init_tensors":
                raise sentinel

        def env_reset(self):
            if failure_stage == "env_reset":
                raise sentinel
            return object()

    writer = Writer()
    agent = Agent(writer)

    class Factory:
        def create(self, *_args, **_kwargs):
            return agent

    class Runner:
        algo_name = "fake"

        def __init__(self) -> None:
            self.algo_factory = Factory()
            self.params = None

        def load(self, payload) -> None:
            self.params = payload["params"]

        def set_vec_env(self, _env) -> None:
            pass

    monkeypatch.setattr(torch_runner, "Runner", Runner)
    monkeypatch.setattr(harness, "SyntheticVecEnv", lambda *_args: object())
    if failure_stage == "owner_check":

        def fail_owner_check(_value):
            raise sentinel

        monkeypatch.setattr(harness, "_class_path", fail_owner_check)

    with pytest.raises(RuntimeError) as error:
        harness._agent_and_env({"config": {}}, tmp_path)

    assert error.value is sentinel
    assert writer.close_calls == 1


def test_patch_scope_restores_raw_objects_and_synthetic_vecenv_abi() -> None:
    class Owner:
        marker = object()

        def method(self):
            return "native"

    class SyntheticVecEnv:
        pass

    owner = Owner()
    env = SyntheticVecEnv()
    agent = SimpleNamespace(frame=48)
    calls = []
    raw_method = Owner.__dict__["method"]
    raw_marker = Owner.__dict__["marker"]

    def set_train_info(frame, candidate):
        assert (frame, candidate) == (agent.frame, agent)
        calls.append((frame, candidate is agent))

    with pytest.raises(RuntimeError, match="injected native failure"):
        with harness._Patches() as patches:
            patches.set_raw(Owner, "method", lambda _self: "patched")
            patches.set_raw(Owner, "marker", object())
            patches.set(env, "set_train_info", set_train_info)
            env.set_train_info(agent.frame, agent)
            raise RuntimeError("injected native failure")

    assert calls == [(48, True)]
    assert "set_train_info" not in vars(env)
    assert Owner.__dict__["method"] is raw_method
    assert Owner.__dict__["marker"] is raw_marker
    assert owner.method() == "native"
    assert patches.restored is True


def test_wrong_expected_package_root_is_rejected(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="exact package root"):
        harness.assert_loaded_namespace(tmp_path)


def test_generator_has_no_deleted_harness_api_references() -> None:
    generator_path = (
        Path(__file__).parents[3] / "scripts" / "generate_simtoolreal_sapg_update_fixture.py"
    )
    syntax = ast.parse(generator_path.read_text(), filename=str(generator_path))

    deleted_machinery = {
        "array_metadata",
        "validate_update_evidence_invariants",
        "event_ledger",
        "capability",
        "sentinel",
        "forensic",
        "rollback",
        "_link_backup",
        "BUDGET",
        "REQUIRED_GENERATION_ENV",
        "_observed_generation_invocation",
    }
    imported_names = {
        alias.name
        for node in ast.walk(syntax)
        if isinstance(node, ast.ImportFrom)
        and node.module
        in {
            "tests.algos.rlgames_sapg.source_rollout_harness",
            "tests.algos.rlgames_sapg.source_update_harness",
        }
        for alias in node.names
    }
    symbol_names = {
        name
        for node in ast.walk(syntax)
        for name in (
            [node.id]
            if isinstance(node, ast.Name)
            else [node.attr]
            if isinstance(node, ast.Attribute)
            else [node.name]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            else [node.arg]
            if isinstance(node, ast.arg)
            else []
        )
    }

    assert deleted_machinery.isdisjoint(imported_names)
    assert deleted_machinery.isdisjoint(symbol_names)


def test_generator_source_provenance_constants_are_exact() -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    assert generator.SOURCE_HEAD == EXPECTED_SOURCE_HEAD
    assert generator.SOURCE_RL_GAMES_TREE == EXPECTED_SOURCE_RL_GAMES_TREE
    assert generator.TRAIN_OWNER == tuple(EXPECTED_SOURCE_OWNERS["train"].values())
    assert generator.TASK_OWNER == tuple(EXPECTED_SOURCE_OWNERS["task"].values())
    assert generator.CODE3_ANCHORS == EXPECTED_CODE3_ANCHORS


def _canonical_generation_command_tokens() -> list[str]:
    return [
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=.",
        "CUDA_VISIBLE_DEVICES=0",
        "UV_INDEX=https://download.pytorch.org/whl/cu128",
        "UNILAB_REQUIRE_SAPG=1",
        "UNILAB_SAPG_ORACLE_MODE=source",
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--python",
        "3.11",
        "--with",
        "gym==0.26.2",
        "--with",
        "torch==2.7.0",
        "--with",
        "numpy==2.4.4",
        "--with",
        "omegaconf==2.3.0",
        "--with-editable",
        "/home/user/ws/lemon/simtoolreal/rl_games",
        "scripts/generate_simtoolreal_sapg_update_fixture.py",
        "--source",
        "/home/user/ws/lemon/simtoolreal",
        "--output",
        "tests/fixtures/simtoolreal_sapg",
    ]


def test_generator_canonical_generation_command_is_complete() -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    command_factory = getattr(generator, "_canonical_generation_command", None)
    assert callable(command_factory)
    command = command_factory()
    expected = _canonical_generation_command_tokens()

    assert "\n" not in command
    assert shlex.split(command) == expected
    assert shlex.join(expected) == command
    assert not any("<" in token or ">" in token for token in expected)


@pytest.mark.parametrize(
    ("binary", "stderr", "detail"),
    [
        (False, "fatal: text failure\n", "fatal: text failure"),
        (True, b"fatal: byte failure\n", "fatal: byte failure"),
    ],
)
def test_generator_git_failure_reports_checkout_operation_status_and_stderr(
    monkeypatch, tmp_path, binary: bool, stderr: str | bytes, detail: str
) -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    command = ["git", "rev-parse", "HEAD"]
    failure = subprocess.CalledProcessError(23, command, stderr=stderr)

    def fail_run(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(generator.subprocess, "run", fail_run)

    with pytest.raises(RuntimeError) as error:
        generator._git(checkout, "rev-parse", "HEAD", binary=binary)

    message = str(error.value)
    assert str(checkout) in message
    assert "git rev-parse HEAD" in message
    assert "exit status 23" in message
    assert detail in message
    assert error.value.__cause__ is failure


def _controlled_source_identity(monkeypatch, tmp_path, *, drift: str | None = None):
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    source = tmp_path / "source"
    source.mkdir()
    train_bytes = b"verified train owner\n"
    task_bytes = b"verified task owner\n"
    train_owner = ("train.yaml", "a" * 40, hashlib.sha256(train_bytes).hexdigest())
    task_owner = ("task.yaml", "b" * 40, hashlib.sha256(task_bytes).hexdigest())
    monkeypatch.setattr(generator, "TRAIN_OWNER", train_owner)
    monkeypatch.setattr(generator, "TASK_OWNER", task_owner)
    responses = {
        (("rev-parse", "HEAD"), False): generator.SOURCE_HEAD,
        (
            ("rev-parse", f"{generator.SOURCE_HEAD}:rl_games/rl_games"),
            False,
        ): generator.SOURCE_RL_GAMES_TREE,
        (
            ("rev-parse", f"{generator.SOURCE_HEAD}:{train_owner[0]}"),
            False,
        ): train_owner[1],
        (
            ("cat-file", "blob", f"{generator.SOURCE_HEAD}:{train_owner[0]}"),
            True,
        ): train_bytes,
        (
            ("rev-parse", f"{generator.SOURCE_HEAD}:{task_owner[0]}"),
            False,
        ): task_owner[1],
        (
            ("cat-file", "blob", f"{generator.SOURCE_HEAD}:{task_owner[0]}"),
            True,
        ): task_bytes,
    }
    drift_keys = {
        "head": (("rev-parse", "HEAD"), False),
        "tree": (
            ("rev-parse", f"{generator.SOURCE_HEAD}:rl_games/rl_games"),
            False,
        ),
        "train_blob": (
            ("rev-parse", f"{generator.SOURCE_HEAD}:{train_owner[0]}"),
            False,
        ),
        "train_bytes": (
            ("cat-file", "blob", f"{generator.SOURCE_HEAD}:{train_owner[0]}"),
            True,
        ),
        "task_blob": (
            ("rev-parse", f"{generator.SOURCE_HEAD}:{task_owner[0]}"),
            False,
        ),
        "task_bytes": (
            ("cat-file", "blob", f"{generator.SOURCE_HEAD}:{task_owner[0]}"),
            True,
        ),
    }
    if drift is not None:
        key = drift_keys[drift]
        responses[key] = b"drift" if key[1] else "drift"
    calls = []

    def controlled_git(candidate, *arguments, binary=False):
        calls.append((candidate, arguments, binary))
        return responses[(arguments, binary)]

    monkeypatch.setattr(generator, "_git", controlled_git)
    return generator, source, {"train": train_bytes, "task": task_bytes}, calls


def test_source_identity_returns_only_verified_owner_blobs(monkeypatch, tmp_path) -> None:
    generator, source, expected_blobs, calls = _controlled_source_identity(monkeypatch, tmp_path)

    assert generator._source_identity(source) == expected_blobs
    assert [binary for _source, _arguments, binary in calls] == [
        False,
        False,
        False,
        True,
        False,
        True,
    ]


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("head", "Source HEAD drift"),
        ("tree", "Source RL-Games tree drift"),
        ("train_blob", "Source train owner drift"),
        ("train_bytes", "Source train owner drift"),
        ("task_blob", "Source task owner drift"),
        ("task_bytes", "Source task owner drift"),
    ],
)
def test_source_identity_rejects_each_git_boundary_drift(
    monkeypatch, tmp_path, drift: str, message: str
) -> None:
    generator, source, _expected_blobs, _calls = _controlled_source_identity(
        monkeypatch, tmp_path, drift=drift
    )

    with pytest.raises(RuntimeError, match=message):
        generator._source_identity(source)


def _copy_code3_fixture(generator, output: Path) -> None:
    output.mkdir()
    for name in generator.CODE3_ANCHORS:
        shutil.copy2(generator.FIXTURE_ROOT / name, output / name)


def test_generator_verifies_real_code3_fixture_copies(tmp_path) -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    output = tmp_path / "code3"
    _copy_code3_fixture(generator, output)

    fixture = generator._verify_code3(output)

    assert fixture.manifest == json.loads(
        (generator.FIXTURE_ROOT / "source_rollout_manifest.json").read_bytes()
    )


@pytest.mark.parametrize("name", EXPECTED_CODE3_ANCHORS)
def test_generator_rejects_each_code3_anchor_drift(tmp_path, name: str) -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    output = tmp_path / "code3"
    _copy_code3_fixture(generator, output)
    path = output / name
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)

    with pytest.raises(RuntimeError) as error:
        generator._verify_code3(output)
    assert str(error.value) == f"Code #3 anchor drift: {name}"


def _minimal_source_owner_blobs(generator) -> dict[str, bytes]:
    dimensions = generator.EXPECTED_OWNER_DIMENSIONS
    train_config = {
        **generator.EXPECTED_SOURCE_DEFAULTS,
        "expl_coef_block_size": dimensions["expl_coef_block_size"],
        "horizon_length": dimensions["horizon_length"],
        "seq_length": dimensions["seq_length"],
        "minibatch_size": dimensions["minibatch_size"],
        "central_value_config": {"minibatch_size": dimensions["central_minibatch_size"]},
    }
    return {
        "train": json.dumps({"params": {"config": train_config}}).encode(),
        "task": json.dumps({"scene": {"num_envs": dimensions["num_envs"]}}).encode(),
    }


def _real_code3_rollout_manifest(generator) -> dict[str, object]:
    return json.loads((generator.FIXTURE_ROOT / "source_rollout_manifest.json").read_bytes())


def test_generator_accepts_verified_owner_and_code4_override_contracts() -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    generator._owner_and_override_contracts(
        _minimal_source_owner_blobs(generator),
        _real_code3_rollout_manifest(generator),
    )


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("source-default", "Source train owner defaults e_clip drift"),
        ("owner-dimension", "Source owner dimensions num_envs drift"),
        ("code4-override", "Code #4 actor boundary num_actors drift"),
    ],
)
def test_generator_rejects_owner_or_override_contract_drift(drift: str, message: str) -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    owner_blobs = _minimal_source_owner_blobs(generator)
    rollout_manifest = _real_code3_rollout_manifest(generator)
    if drift == "source-default":
        train = json.loads(owner_blobs["train"])
        train["params"]["config"]["e_clip"] = 0.2
        owner_blobs["train"] = json.dumps(train).encode()
    elif drift == "owner-dimension":
        task = json.loads(owner_blobs["task"])
        task["scene"]["num_envs"] -= 1
        owner_blobs["task"] = json.dumps(task).encode()
    else:
        rollout_manifest["runner_params"]["config"]["num_actors"] += 1

    with pytest.raises(RuntimeError, match=message):
        generator._owner_and_override_contracts(owner_blobs, rollout_manifest)


def _git_blob_oid(data: bytes) -> str:
    payload = b"blob " + str(len(data)).encode() + b"\0" + data
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()


def _controlled_source_python_tree(monkeypatch, tmp_path, *, file_count: int = 72):
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    source = tmp_path / "source"
    package_root = source / generator.SOURCE_PACKAGE_RELATIVE
    package_root.mkdir(parents=True)
    git_blobs: dict[str, bytes] = {}
    expected: dict[str, str] = {}
    working_paths: dict[str, Path] = {}
    tree_entries = []
    for index in range(file_count):
        relative = Path("common/example.py") if index == 0 else Path(f"module_{index:02d}.py")
        data = f"VALUE_{index} = {index}\n".encode()
        blob = _git_blob_oid(data)
        git_blobs[blob] = data
        relative_name = relative.as_posix()
        expected[relative_name] = hashlib.sha256(data).hexdigest()
        path = package_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        working_paths[relative_name] = path
        tree_entries.append(f"100644 blob {blob}\t{relative_name}\0".encode())
    tree_data = b"".join(sorted(tree_entries))

    def controlled_git(candidate, *arguments, binary=False):
        assert candidate == source
        if (
            arguments
            == (
                "ls-tree",
                "-r",
                "-z",
                generator.SOURCE_RL_GAMES_TREE,
            )
            and binary
        ):
            return tree_data
        if len(arguments) == 3 and arguments[:2] == ("cat-file", "blob") and binary:
            return git_blobs[arguments[2]]
        raise AssertionError((arguments, binary))

    monkeypatch.setattr(generator, "_git", controlled_git)
    return generator, source, expected, working_paths, git_blobs


def test_generator_preverifies_exact_source_python_tree(monkeypatch, tmp_path) -> None:
    generator, source, expected, _working_paths, _git_blobs = _controlled_source_python_tree(
        monkeypatch, tmp_path
    )

    verify = getattr(generator, "_verify_source_python_tree_before_capture", None)
    assert callable(verify)
    assert generator.EXPECTED_SOURCE_PYTHON_FILE_COUNT == 72
    assert verify(source) == expected


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("working-bytes", "Source Python working bytes drift"),
        ("missing", "Source Python working inventory drift"),
        ("extra", "Source Python working inventory drift"),
        ("symlink", "Source Python working tree contains symlink"),
        ("git-blob", "Source Python Git blob identity drift"),
        ("file-count", "Source Python file count drift"),
    ],
)
def test_generator_rejects_source_python_tree_drift(
    monkeypatch, tmp_path, drift: str, message: str
) -> None:
    file_count = 71 if drift == "file-count" else 72
    generator, source, _expected, working_paths, git_blobs = _controlled_source_python_tree(
        monkeypatch, tmp_path, file_count=file_count
    )
    first_name = sorted(working_paths)[0]
    first_path = working_paths[first_name]
    if drift == "working-bytes":
        first_path.write_bytes(b"WORKING_DRIFT = True\n")
    elif drift == "missing":
        first_path.unlink()
    elif drift == "extra":
        (source / generator.SOURCE_PACKAGE_RELATIVE / "extra.py").write_bytes(b"EXTRA = True\n")
    elif drift == "symlink":
        outside = tmp_path / "outside.py"
        outside.write_bytes(first_path.read_bytes())
        first_path.unlink()
        first_path.symlink_to(outside)
    elif drift == "git-blob":
        first_blob = next(blob for blob, data in git_blobs.items() if _git_blob_oid(data) == blob)
        git_blobs[first_blob] = b"GIT_DRIFT = True\n"

    verify = getattr(generator, "_verify_source_python_tree_before_capture", None)
    assert callable(verify)
    with pytest.raises(RuntimeError, match=message):
        verify(source)


@pytest.mark.parametrize("candidate", ["extension", "legacy-bytecode"])
def test_generator_rejects_untracked_source_import_candidate(
    monkeypatch, tmp_path, candidate: str
) -> None:
    generator, source, _expected, _working_paths, _git_blobs = _controlled_source_python_tree(
        monkeypatch, tmp_path
    )
    package_root = source / generator.SOURCE_PACKAGE_RELATIVE
    if candidate == "extension":
        suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
        (package_root / f"shadow{suffix}").write_bytes(b"extension shadow")
    else:
        (package_root / "shadow.pyc").write_bytes(b"legacy bytecode shadow")

    with pytest.raises(RuntimeError, match="Source Python import candidate drift"):
        generator._verify_source_python_tree_before_capture(source)


def test_generator_allows_existing_pycache_only_for_isolated_lookup(monkeypatch, tmp_path) -> None:
    generator, source, expected, _working_paths, _git_blobs = _controlled_source_python_tree(
        monkeypatch, tmp_path
    )
    cache = source / generator.SOURCE_PACKAGE_RELATIVE / "common/__pycache__"
    cache.mkdir()
    (cache / "example.cpython-311.pyc").write_bytes(b"existing Source cache")

    assert generator._verify_source_python_tree_before_capture(source) == expected


def test_generator_source_python_drift_fails_before_capture(monkeypatch, tmp_path) -> None:
    generator, source, _expected, working_paths, _git_blobs = _controlled_source_python_tree(
        monkeypatch, tmp_path
    )
    working_paths[sorted(working_paths)[0]].write_bytes(b"WORKING_DRIFT = True\n")
    output = tmp_path / "fixtures"
    output.mkdir()
    rollout_fixture = SimpleNamespace(manifest={"runner_params": {}})
    capture_calls = []

    monkeypatch.setattr(generator, "_validated_source_root", lambda _source: source)
    monkeypatch.setattr(
        generator,
        "_source_identity",
        lambda _source: {"train": b"train", "task": b"task"},
    )
    monkeypatch.setattr(generator, "_verify_code3", lambda _output: rollout_fixture)
    monkeypatch.setattr(generator, "_owner_and_override_contracts", lambda *_args: None)
    monkeypatch.setattr(generator, "_reject_preloaded_rl_games", lambda: None)

    def capture_spy(*_args):
        capture_calls.append(True)
        return {"manifest": {"provenance": {}}, "arrays": {}}

    monkeypatch.setattr(generator, "capture_update", capture_spy)
    monkeypatch.setattr(generator, "_validate_source_capture", lambda _capture: None)
    monkeypatch.setattr(generator, "_verify_modules", lambda *_args: None)
    monkeypatch.setattr(generator, "_serialize_capture", lambda _capture: object())

    try:
        with pytest.raises(RuntimeError, match="Source Python working bytes drift"):
            generator._build_source_artifacts(source, output)
    finally:
        assert capture_calls == []


def _controlled_module_verifier(monkeypatch, tmp_path, *, git_drift: str | None = None):
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    source = tmp_path / "source"
    relative = Path("common/example.py")
    working_path = source / generator.SOURCE_PACKAGE_RELATIVE / relative
    working_path.parent.mkdir(parents=True)
    module_bytes = b"VALUE = 1\n"
    working_path.write_bytes(module_bytes)
    object_name = f"{generator.SOURCE_HEAD}:rl_games/rl_games/{relative.as_posix()}"
    git_calls = []

    def controlled_git(candidate, *arguments, binary=False):
        assert candidate == source
        git_calls.append((arguments, binary))
        if arguments == ("rev-parse", object_name) and not binary:
            if git_drift == "git-oid":
                return "f" * 40
            return _git_blob_oid(module_bytes)
        if arguments == ("cat-file", "blob", object_name) and binary:
            if git_drift == "git-bytes":
                return b"GIT_DRIFT = True\n"
            return module_bytes
        raise AssertionError((arguments, binary))

    monkeypatch.setattr(generator, "_git", controlled_git)
    records = {
        "rl_games.common.example": {
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(module_bytes).hexdigest(),
        }
    }
    preverified = {
        relative.as_posix(): hashlib.sha256(module_bytes).hexdigest(),
    }
    return generator, source, records, working_path, preverified, git_calls


def test_generator_verifies_loaded_module_git_and_working_bytes(monkeypatch, tmp_path) -> None:
    generator, source, records, _working_path, preverified, git_calls = _controlled_module_verifier(
        monkeypatch, tmp_path
    )

    generator._verify_modules(source, records, preverified)
    object_name = f"{generator.SOURCE_HEAD}:rl_games/rl_games/common/example.py"
    assert git_calls == [
        (("rev-parse", object_name), False),
        (("cat-file", "blob", object_name), True),
    ]


@pytest.mark.parametrize("drift", ["git-oid", "git-bytes"])
def test_generator_rejects_loaded_module_git_identity_drift(
    monkeypatch, tmp_path, drift: str
) -> None:
    generator, source, records, _working_path, preverified, git_calls = _controlled_module_verifier(
        monkeypatch, tmp_path, git_drift=drift
    )

    with pytest.raises(RuntimeError, match="loaded Source module Git identity drift"):
        generator._verify_modules(source, records, preverified)

    object_name = f"{generator.SOURCE_HEAD}:rl_games/rl_games/common/example.py"
    assert git_calls == [
        (("rev-parse", object_name), False),
        (("cat-file", "blob", object_name), True),
    ]


@pytest.mark.parametrize("drift", ["manifest-digest", "working-bytes"])
def test_generator_rejects_loaded_module_byte_drift(monkeypatch, tmp_path, drift: str) -> None:
    generator, source, records, working_path, preverified, _git_calls = _controlled_module_verifier(
        monkeypatch, tmp_path
    )
    if drift == "manifest-digest":
        records["rl_games.common.example"]["sha256"] = "f" * 64
    else:
        working_path.write_bytes(b"VALUE = 2\n")

    with pytest.raises(RuntimeError, match="loaded Source module bytes drift"):
        generator._verify_modules(source, records, preverified)


def test_generator_rejects_loaded_module_preverified_inventory_mismatch(
    monkeypatch, tmp_path
) -> None:
    generator, source, records, _working_path, preverified, _git_calls = (
        _controlled_module_verifier(monkeypatch, tmp_path)
    )
    preverified["common/example.py"] = "f" * 64

    with pytest.raises(RuntimeError, match="loaded Source module bytes drift"):
        generator._verify_modules(source, records, preverified)


def test_generator_rejects_unpreverified_loaded_module_before_git(monkeypatch, tmp_path) -> None:
    generator, source, records, _working_path, preverified, git_calls = _controlled_module_verifier(
        monkeypatch, tmp_path
    )
    preverified.clear()
    preverified["common/other.py"] = "0" * 64

    with pytest.raises(RuntimeError, match="loaded Source module was not preverified"):
        generator._verify_modules(source, records, preverified)

    assert git_calls == []


def test_generator_rejects_noncanonical_loaded_module_path_before_git(
    monkeypatch, tmp_path
) -> None:
    generator, source, records, _working_path, preverified, git_calls = _controlled_module_verifier(
        monkeypatch, tmp_path
    )
    records["rl_games.common.example"]["path"] = "common//example.py"

    with pytest.raises(RuntimeError, match="loaded Source module path drift"):
        generator._verify_modules(source, records, preverified)

    assert git_calls == []


def test_generator_orchestration_orders_provenance_before_capture(
    monkeypatch, tmp_path, capsys
) -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    source = tmp_path / "source"
    output = tmp_path / "fixtures"
    source.mkdir()
    output.mkdir()
    monkeypatch.setattr(generator, "SOURCE_CHECKOUT", source)
    monkeypatch.setattr(generator, "FIXTURE_ROOT", output)
    monkeypatch.setenv("UNILAB_SAPG_ORACLE_MODE", "source")
    capture = complete_synthetic_capture()
    capture["manifest"]["platform"] = copy.deepcopy(generator.CANONICAL_PLATFORM)
    capture["manifest"]["canonical_platform"] = copy.deepcopy(generator.CANONICAL_PLATFORM)
    loaded_modules = {
        "rl_games.torch_runner": {"path": "torch_runner.py", "sha256": "0" * 64},
        "rl_games.algos_torch.a2c_continuous": {
            "path": "algos_torch/a2c_continuous.py",
            "sha256": "1" * 64,
        },
        "rl_games.common.datasets": {
            "path": "common/datasets.py",
            "sha256": "2" * 64,
        },
        "rl_games.algos_torch.central_value": {
            "path": "algos_torch/central_value.py",
            "sha256": "3" * 64,
        },
    }
    capture["manifest"]["provenance"]["loaded_rl_games_modules"] = loaded_modules
    runner_params = capture["manifest"]["runner_params"]
    capture_manifest_before = copy.deepcopy(capture["manifest"])
    original_pycache_prefix = sys.pycache_prefix
    original_dont_write_bytecode = sys.dont_write_bytecode
    verified_owner_blobs = {"train": b"verified train", "task": b"verified task"}
    preverified_python = {"common/example.py": "4" * 64}
    rollout_fixture = SimpleNamespace(manifest={"runner_params": runner_params})
    events = []

    def validate_source(candidate):
        events.append("source-root")
        assert candidate == source
        return source

    def validate_identity(candidate):
        events.append("source-identity")
        assert candidate == source
        return verified_owner_blobs

    def validate_code3(candidate):
        events.append("code3")
        assert candidate == output
        return rollout_fixture

    def validate_owner_values(blobs, manifest):
        events.append("owner-values")
        assert blobs is verified_owner_blobs
        assert manifest is rollout_fixture.manifest
        return {}, {}, {}

    def verify_source_python_tree(candidate):
        events.append("source-python-tree")
        assert candidate == source
        return preverified_python

    def reject_preloaded_namespace():
        events.append("namespace")

    def capture_source(params, package_root):
        events.append("capture")
        assert params == runner_params
        assert package_root == source / generator.SOURCE_PACKAGE_RELATIVE
        assert sys.pycache_prefix is not None
        assert sys.pycache_prefix != original_pycache_prefix
        assert list(Path(sys.pycache_prefix).iterdir()) == []
        assert sys.dont_write_bytecode is True
        return capture

    def verify_modules(candidate, records, preverified=None):
        events.append("modules")
        assert candidate == source
        assert records == loaded_modules
        assert preverified is preverified_python

    monkeypatch.setattr(generator, "_validated_source_root", validate_source)
    monkeypatch.setattr(generator, "_source_identity", validate_identity)
    monkeypatch.setattr(generator, "_verify_code3", validate_code3)
    monkeypatch.setattr(generator, "_owner_and_override_contracts", validate_owner_values)
    monkeypatch.setattr(
        generator,
        "_verify_source_python_tree_before_capture",
        verify_source_python_tree,
    )
    monkeypatch.setattr(generator, "_reject_preloaded_rl_games", reject_preloaded_namespace)
    monkeypatch.setattr(generator, "capture_update", capture_source)
    monkeypatch.setattr(generator, "_verify_modules", verify_modules)

    artifacts = generator._build_source_artifacts(source, output)

    assert events == [
        "source-root",
        "source-identity",
        "code3",
        "owner-values",
        "source-python-tree",
        "namespace",
        "capture",
        "modules",
    ]
    harness.validate_capture(generator._deserialize_capture(artifacts.npz, artifacts.manifest))
    assert artifacts.manifest_data["generation_command"] == shlex.join(
        _canonical_generation_command_tokens()
    )
    assert capture["manifest"] == capture_manifest_before
    assert sys.pycache_prefix == original_pycache_prefix
    assert sys.dont_write_bytecode is original_dont_write_bytecode
    generator._print_artifact_hashes(artifacts)
    stdout = capsys.readouterr().out
    assert "fixture_npz_sha256=" in stdout
    assert "fixture_manifest_sha256=" in stdout
    assert "canonical_payload_sha256=" in stdout


def test_generator_restores_pycache_isolation_when_capture_fails(monkeypatch, tmp_path) -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    source = tmp_path / "source"
    output = tmp_path / "fixtures"
    source.mkdir()
    output.mkdir()
    original_pycache_prefix = sys.pycache_prefix
    original_dont_write_bytecode = sys.dont_write_bytecode
    observed = []
    sentinel = RuntimeError("injected capture failure")
    rollout_fixture = SimpleNamespace(manifest={"runner_params": {}})

    monkeypatch.setattr(generator, "_validated_source_root", lambda _source: source)
    monkeypatch.setattr(
        generator,
        "_source_identity",
        lambda _source: {"train": b"train", "task": b"task"},
    )
    monkeypatch.setattr(generator, "_verify_code3", lambda _output: rollout_fixture)
    monkeypatch.setattr(generator, "_owner_and_override_contracts", lambda *_args: None)
    monkeypatch.setattr(
        generator,
        "_verify_source_python_tree_before_capture",
        lambda _source: {"common/example.py": "0" * 64},
    )
    monkeypatch.setattr(generator, "_reject_preloaded_rl_games", lambda: None)

    def fail_capture(*_args):
        observed.append((sys.pycache_prefix, sys.dont_write_bytecode))
        raise sentinel

    monkeypatch.setattr(generator, "capture_update", fail_capture)

    with pytest.raises(RuntimeError) as error:
        generator._build_source_artifacts(source, output)

    assert error.value is sentinel
    assert len(observed) == 1
    isolated_prefix, isolated_dont_write = observed[0]
    assert isolated_prefix is not None
    assert isolated_prefix != original_pycache_prefix
    assert isolated_dont_write is True
    assert sys.pycache_prefix == original_pycache_prefix
    assert sys.dont_write_bytecode is original_dont_write_bytecode


def test_generate_requires_explicit_source_only_mode(monkeypatch) -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    monkeypatch.delenv("UNILAB_SAPG_ORACLE_MODE", raising=False)

    with pytest.raises(RuntimeError, match="explicit Source-only generation mode"):
        generator._require_source_only_mode()


def test_generator_rejects_noncanonical_source_capture() -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    with pytest.raises(RuntimeError, match="Source update capture is not canonical"):
        generator._validate_source_capture(complete_synthetic_capture())


def _dotted_ast_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _dotted_ast_name(node.value)
        if owner is not None:
            return f"{owner}.{node.attr}"
    return None


def _update_generator_calls(path: Path) -> list[ast.Call]:
    target = "scripts.generate_simtoolreal_sapg_update_fixture"
    syntax = ast.parse(path.read_text(), filename=str(path))
    module_names: set[str] = set()
    direct_names: set[str] = set()
    for node in ast.walk(syntax):
        if isinstance(node, ast.Import):
            module_names.update(
                alias.asname or target for alias in node.names if alias.name == target
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "scripts":
            module_names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "generate_simtoolreal_sapg_update_fixture"
            )
        elif isinstance(node, ast.ImportFrom) and node.module == target:
            direct_names.update(
                alias.asname or alias.name for alias in node.names if alias.name == "generate"
            )

    module_calls = {f"{name}.generate" for name in module_names}
    return [
        node
        for node in ast.walk(syntax)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in direct_names)
            or _dotted_ast_name(node.func) in module_calls
        )
    ]


def test_ordinary_pytest_never_calls_update_fixture_generate() -> None:
    suite_root = Path(__file__).parent
    generate_calls = [
        (path.relative_to(suite_root).as_posix(), call.lineno)
        for path in sorted(suite_root.rglob("*.py"))
        for call in _update_generator_calls(path)
    ]

    assert generate_calls == []


def test_generator_rejects_symlink_output_component_without_touching_leaf(tmp_path) -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    outside = tmp_path / "outside"
    outside.mkdir()
    protected_leaf = outside / generator.FIXTURE_NAMES[0]
    protected_leaf.write_bytes(b"protected")
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        generator.validated_output_paths(alias)

    assert protected_leaf.read_bytes() == b"protected"


def test_generator_rejects_symlink_output_leaf(monkeypatch, tmp_path) -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    output = tmp_path / "fixtures"
    output.mkdir()
    monkeypatch.setattr(generator, "FIXTURE_ROOT", output)
    protected = tmp_path / "protected"
    protected.write_bytes(b"protected")
    (output / generator.FIXTURE_NAMES[0]).symlink_to(protected)

    with pytest.raises(RuntimeError, match="regular file"):
        generator.validated_output_paths(output)

    assert protected.read_bytes() == b"protected"


def test_generator_rejects_non_regular_output_leaf(monkeypatch, tmp_path) -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    output = tmp_path / "fixtures"
    output.mkdir()
    monkeypatch.setattr(generator, "FIXTURE_ROOT", output)
    (output / generator.FIXTURE_NAMES[1]).mkdir()

    with pytest.raises(RuntimeError, match="regular file"):
        generator.validated_output_paths(output)


def test_generator_accepts_absent_or_regular_output_leaves(monkeypatch, tmp_path) -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    output = tmp_path / "fixtures"
    output.mkdir()
    monkeypatch.setattr(generator, "FIXTURE_ROOT", output)
    expected = tuple(output / name for name in generator.FIXTURE_NAMES)

    assert generator.validated_output_paths(output) == expected
    for path in expected:
        path.write_bytes(b"existing regular fixture")
    assert generator.validated_output_paths(output) == expected


def test_generator_replaces_both_leaves_from_fsynced_same_directory_temporaries(
    monkeypatch, tmp_path
) -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    output = tmp_path / "fixtures"
    output.mkdir()
    monkeypatch.setattr(generator, "FIXTURE_ROOT", output)
    artifacts = generator._serialize_capture(complete_synthetic_capture())
    real_fsync = generator.os.fsync
    real_replace = generator.os.replace
    fsynced_modes = []
    replacements = []
    events = []

    def record_fsync(descriptor):
        mode = generator.os.fstat(descriptor).st_mode
        fsynced_modes.append(mode)
        events.append(("fsync", mode))
        real_fsync(descriptor)

    def record_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        events.append(("replace", Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(generator.os, "fsync", record_fsync)
    monkeypatch.setattr(generator.os, "replace", record_replace)

    generator._write_artifacts(output, artifacts)

    expected = tuple(output / name for name in generator.FIXTURE_NAMES)
    assert [destination for _, destination in replacements] == list(expected)
    assert all(source.parent == output for source, _ in replacements)
    assert all(destination.parent == output for _, destination in replacements)
    assert expected[0].read_bytes() == artifacts.npz
    assert expected[1].read_bytes() == artifacts.manifest
    assert sum(stat.S_ISREG(mode) for mode in fsynced_modes) == 2
    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)
    regular_fsync_indices = [
        index
        for index, (event, value) in enumerate(events)
        if event == "fsync" and stat.S_ISREG(value)
    ]
    replace_indices = [index for index, (event, _value) in enumerate(events) if event == "replace"]
    directory_fsync_indices = [
        index
        for index, (event, value) in enumerate(events)
        if event == "fsync" and stat.S_ISDIR(value)
    ]
    assert max(regular_fsync_indices) < min(replace_indices)
    assert max(replace_indices) < min(directory_fsync_indices)
    assert sorted(path.name for path in output.iterdir()) == sorted(generator.FIXTURE_NAMES)


def test_generator_cleans_temporary_files_when_replace_fails(monkeypatch, tmp_path) -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    output = tmp_path / "fixtures"
    output.mkdir()
    monkeypatch.setattr(generator, "FIXTURE_ROOT", output)
    artifacts = generator._serialize_capture(complete_synthetic_capture())

    def fail_replace(_source, _destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(generator.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        generator._write_artifacts(output, artifacts)

    assert list(output.iterdir()) == []


def test_generator_output_root_is_fixed_to_checked_in_fixture_directory(
    monkeypatch, tmp_path
) -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    expected = tuple(generator.FIXTURE_ROOT / name for name in generator.FIXTURE_NAMES)
    assert generator.validated_output_paths(generator.FIXTURE_ROOT) == expected
    monkeypatch.chdir(generator.REPO_ROOT)
    assert generator.validated_output_paths(Path("tests/fixtures/simtoolreal_sapg")) == expected
    with pytest.raises(RuntimeError, match="exact.*fixture|fixture.*exact"):
        generator.validated_output_paths(tmp_path)


def test_generator_serializes_complete_capture_and_roundtrips_in_memory() -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    capture = complete_synthetic_capture()

    artifacts = generator._serialize_capture(capture)
    roundtrip = generator._deserialize_capture(artifacts.npz, artifacts.manifest)

    assert json.loads(artifacts.manifest) == artifacts.manifest_data
    assert artifacts.manifest.endswith(b"\n")
    assert roundtrip["manifest"] == artifacts.manifest_data
    assert set(roundtrip["arrays"]) == set(capture["arrays"])
    for name, expected in capture["arrays"].items():
        np.testing.assert_array_equal(roundtrip["arrays"][name], expected)
    assert (
        artifacts.manifest_data["canonical_payload_sha256"]
        == hashlib.sha256(harness.canonical_payload(artifacts.manifest_data)).hexdigest()
    )
    harness.validate_capture(roundtrip)


def _generator_artifacts_with_array_drift(drift: str):
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    artifacts = generator._serialize_capture(complete_synthetic_capture())
    roundtrip = generator._deserialize_capture(artifacts.npz, artifacts.manifest)
    fixture = UpdateFixture(
        manifest=roundtrip["manifest"],
        arrays=roundtrip["arrays"],
    )
    if drift == "content":
        arrays = {name: value.copy() for name, value in fixture.arrays.items()}
        name = next(name for name, value in arrays.items() if value.ndim > 0)
        arrays[name].view(np.uint8).reshape(-1)[0] ^= 1
    else:
        arrays = mutated_arrays(fixture, drift)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return generator, artifacts, stream.getvalue()


def test_generator_rejects_missing_array_in_memory() -> None:
    generator, artifacts, npz_bytes = _generator_artifacts_with_array_drift("missing")

    with pytest.raises(RuntimeError, match="missing snapshot arrays"):
        generator._deserialize_capture(npz_bytes, artifacts.manifest)


def test_generator_rejects_extra_array_in_memory() -> None:
    generator, artifacts, npz_bytes = _generator_artifacts_with_array_drift("extra")

    with pytest.raises(RuntimeError, match="extra snapshot arrays"):
        generator._deserialize_capture(npz_bytes, artifacts.manifest)


def test_generator_rejects_array_shape_corruption_in_memory() -> None:
    generator, artifacts, npz_bytes = _generator_artifacts_with_array_drift("shape")

    with pytest.raises(RuntimeError, match="array shape drift"):
        generator._deserialize_capture(npz_bytes, artifacts.manifest)


def test_generator_rejects_array_dtype_corruption_in_memory() -> None:
    generator, artifacts, npz_bytes = _generator_artifacts_with_array_drift("dtype")

    with pytest.raises(RuntimeError, match="array dtype drift"):
        generator._deserialize_capture(npz_bytes, artifacts.manifest)


def test_generator_rejects_array_content_corruption_in_memory() -> None:
    generator, artifacts, npz_bytes = _generator_artifacts_with_array_drift("content")

    with pytest.raises(RuntimeError, match="array content drift"):
        generator._deserialize_capture(npz_bytes, artifacts.manifest)


def test_generator_rejects_canonical_payload_corruption_in_memory() -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    artifacts = generator._serialize_capture(complete_synthetic_capture())
    manifest = copy.deepcopy(artifacts.manifest_data)
    manifest["canonical_payload_sha256"] = "f" * 64
    manifest_bytes = generator._strict_manifest_json_bytes(manifest)

    with pytest.raises(RuntimeError, match="canonical payload drift"):
        generator._deserialize_capture(artifacts.npz, manifest_bytes)


def test_generator_rejects_duplicate_strict_json_keys_in_memory() -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        generator._strict_manifest_json_loads(b'{"schema_version":2,"schema_version":2}')


def test_generator_strict_json_loads_finite_float_in_memory() -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    assert generator._strict_manifest_json_loads(b'{"value":1.25}') == {"value": 1.25}


def test_generator_rejects_overflowing_strict_json_float_in_memory() -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    with pytest.raises(RuntimeError, match="non-finite JSON number"):
        generator._strict_manifest_json_loads(b'{"value":1e400}')


def test_fixture_files_and_canonical_payload_match_external_anchors() -> None:
    assert _file_sha256(harness.FIXTURE_NPZ) == EXPECTED_UPDATE_NPZ_SHA256
    assert _file_sha256(harness.FIXTURE_MANIFEST) == EXPECTED_UPDATE_MANIFEST_SHA256
    manifest = json.loads(harness.FIXTURE_MANIFEST.read_text())
    assert hashlib.sha256(harness.canonical_payload(manifest)).hexdigest() == (
        EXPECTED_UPDATE_PAYLOAD_SHA256
    )


def test_source_code3_and_owner_provenance_is_fixed() -> None:
    from scripts import generate_simtoolreal_sapg_update_fixture as generator

    manifest = json.loads(harness.FIXTURE_MANIFEST.read_text())
    assert harness.SOURCE_HEAD == EXPECTED_SOURCE_HEAD
    assert harness.SOURCE_RL_GAMES_TREE == EXPECTED_SOURCE_RL_GAMES_TREE
    assert harness.CODE3_ANCHORS == EXPECTED_CODE3_ANCHORS
    assert manifest["generation_mode"] == "source-only"
    assert manifest["ordinary_pytest_regenerates"] is False
    assert manifest["provenance"]["source_head"] == EXPECTED_SOURCE_HEAD
    assert manifest["provenance"]["source_rl_games_tree"] == EXPECTED_SOURCE_RL_GAMES_TREE
    assert manifest["provenance"]["owners"] == EXPECTED_SOURCE_OWNERS
    assert generator.TRAIN_OWNER == tuple(EXPECTED_SOURCE_OWNERS["train"].values())
    assert generator.TASK_OWNER == tuple(EXPECTED_SOURCE_OWNERS["task"].values())
    assert manifest["code3_anchors"] == EXPECTED_CODE3_ANCHORS


def test_update_fixture_replays_native_target() -> None:
    result = replay_update_fixture(load_update_fixture())
    assert result.case_names == CASE_NAMES
    assert result.native_owner_paths
    assert result.numeric_array_count > 0
    assert np.isfinite(result.max_abs_error)
