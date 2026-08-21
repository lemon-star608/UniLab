from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tests.algos.rlgames_sapg.source_network_harness import (
    configure_canonical_execution,
    execution_platform,
    fill_parameters,
)
from tests.algos.rlgames_sapg.source_rollout_harness import CANONICAL_PLATFORM
from tests.algos.rlgames_sapg.source_update_harness import (
    CODE3_ANCHORS,
    SOURCE_HEAD,
    SOURCE_OWNERS,
    SOURCE_RL_GAMES_TREE,
    assert_loaded_namespace,
)

PLAYER_COUNTS = (6, 5, 7)
NETWORK_IDS = [50.0, 40.0, 30.0, 20.0, 10.0, 0.0]
EXPECTED_PLAYER_ROWS = {
    6: [0, 1, 2, 3, 4, 5],
    5: [0, 0, 0, 0, 5],
    7: [0, 0, 0, 0, 0, 0, 5],
}
CHECKPOINT_FILE_NAME = "source_checkpoint.pth"
MANIFEST_FILE_NAME = "source_checkpoint_manifest.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "simtoolreal_sapg"
SCHEMA_VERSION = 1
EXPECTED_CHECKPOINT_SHA256 = "bbe577dc7efed068bb38ce6f268e849de6a41e8ab6bb4a78fabeed9b0d7b5e02"
EXPECTED_MANIFEST_SHA256 = "8d55469d09095827587d502758d477913c76f13e8e9cd0baa23cb142d518c946"
EXPECTED_PAYLOAD_SHA256 = "1f652c3431aa80db02e3c852a3a4bedc56772b69714790f848a7683110980187"
MANIFEST_KEYS = frozenset(
    "schema_version generation_mode ordinary_pytest_regenerates provenance payload "
    "resume player comparison runner_params fixture_files canonical_payload_sha256 "
    "generation_command".split()
)
_SELF_HASH_FIELDS = frozenset(
    {"canonical_payload_sha256", "manifest_payload_sha256", "manifest_sha256"}
)


@dataclass(frozen=True)
class NativeCheckpoint:
    payload: bytes
    metadata: dict[str, object]
    loaded_modules: dict[str, dict[str, str]]


@dataclass(frozen=True)
class CheckpointReplay:
    player_counts: tuple[int, ...]
    observable_leaves: int
    exact_state_sections: int
    native_owner_paths: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointFixture:
    manifest: dict[str, object]
    payload: bytes


class CheckpointVecEnv:
    def __init__(self, spaces: object, device: torch.device, num_envs: int = 6) -> None:
        self.device = device
        self.env = self
        self.num_envs = num_envs
        self.step_index = 0
        self.set_env_state_calls: list[object] = []
        self.train_info_calls: list[tuple[int, object]] = []
        self.env_info = {
            "agents": 1,
            "value_size": 1,
            "observation_space": spaces.Box(-10.0, 10.0, (140,), dtype=np.float32),
            "state_space": spaces.Box(-10.0, 10.0, (162,), dtype=np.float32),
            "action_space": spaces.Box(-1.0, 1.0, (29,), dtype=np.float32),
        }

    def get_env_info(self) -> dict[str, object]:
        return self.env_info

    def get_env_state(self) -> None:
        return None

    def set_env_state(self, state: object) -> None:
        if state is not None:
            raise RuntimeError(f"checkpoint env only accepts env_state=None, got {state!r}")
        self.set_env_state_calls.append(state)

    def set_train_info(self, frame: int, owner: object) -> None:
        self.train_info_calls.append((int(frame), owner))

    def _matrix(self, width: int, offset: int) -> torch.Tensor:
        env = torch.arange(self.num_envs, dtype=torch.float32, device=self.device)[:, None]
        feature = torch.arange(width, dtype=torch.float32, device=self.device)[None, :]
        values = torch.remainder(env * 7 + feature * 3 + self.step_index + offset, 101)
        return (values - 50.0) / 50.0

    def _observation(self) -> dict[str, torch.Tensor]:
        return {
            "obs": self._matrix(140, 3),
            "states": self._matrix(162, 17),
        }

    def reset(self) -> dict[str, torch.Tensor]:
        self.step_index = 0
        return self._observation()

    def step(self, _actions: torch.Tensor):
        self.step_index += 1
        rewards = (
            torch.arange(self.num_envs, dtype=torch.float32, device=self.device) / 32.0
            + self.step_index / 10.0
        )
        dones = torch.zeros(self.num_envs, dtype=torch.uint8, device=self.device)
        dones[(self.step_index - 1) % self.num_envs] = 1
        return (
            self._observation(),
            rewards,
            dones,
            {"time_outs": torch.zeros_like(dones)},
        )


def _strict_json_bytes(value: object) -> bytes:
    """Encode a manifest using the one canonical, finite JSON representation."""
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("checkpoint manifest is not finite strict JSON") from error
    return encoded.encode("ascii")


def _strict_json_loads(data: bytes) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise RuntimeError(f"checkpoint manifest contains non-standard JSON constant: {value}")

    def reject_nonfinite(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError(f"checkpoint manifest contains non-finite JSON number: {value}")
        return number

    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"checkpoint manifest contains duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            data,
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
            parse_float=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("checkpoint manifest is not strict JSON") from error
    if not isinstance(decoded, dict):
        raise RuntimeError("checkpoint manifest must contain an object")
    return decoded


def canonical_payload(value: Mapping[str, object]) -> bytes:
    """Return the canonical manifest bytes with self-referential hashes removed."""
    if not isinstance(value, Mapping):
        raise RuntimeError("checkpoint manifest payload must be a mapping")
    payload = copy.deepcopy(dict(value))
    for name in _SELF_HASH_FIELDS:
        payload.pop(name, None)
    return _strict_json_bytes(payload)


def canonical_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_payload(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _state_key(value: object) -> dict[str, object]:
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    raise RuntimeError(f"unsupported checkpoint state key: {type(value)!r}")


def describe_state(value: object) -> dict[str, object]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return {
            "kind": "tensor",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _array_sha256(array),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "array",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _array_sha256(array),
        }
    if isinstance(value, Mapping):
        items = [
            {"key": _state_key(key), "value": describe_state(item)} for key, item in value.items()
        ]
        items.sort(key=lambda item: (str(item["key"]["kind"]), str(item["key"]["value"])))
        return {"kind": "dict", "items": items}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [describe_state(item) for item in value],
        }
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return {"kind": "none"}
    if isinstance(value, (bool, int, str)):
        return {"kind": type(value).__name__, "value": value}
    if isinstance(value, float) and math.isfinite(value):
        return {"kind": "float", "value": value}
    raise RuntimeError(f"unsupported checkpoint state: {type(value)!r}")


def _observable_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "+inf" if value > 0 else "-inf"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise RuntimeError(f"unsupported observable scalar: {type(value)!r}")


def observe_tree(value: object) -> dict[str, object]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "array",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _array_sha256(array),
            "data": [_observable_scalar(item) for item in array.reshape(-1)],
        }
    if isinstance(value, Mapping):
        items = [
            {"key": _state_key(key), "value": observe_tree(item)} for key, item in value.items()
        ]
        items.sort(key=lambda item: (str(item["key"]["kind"]), str(item["key"]["value"])))
        return {"kind": "dict", "items": items}
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [observe_tree(item) for item in value],
        }
    scalar = _observable_scalar(value)
    return {"kind": "scalar", "value": scalar}


def _decoded_float_data(record: Mapping[str, object]) -> np.ndarray:
    decoded = []
    for item in record["data"]:
        if item == "nan":
            decoded.append(float("nan"))
        elif item == "+inf":
            decoded.append(float("inf"))
        elif item == "-inf":
            decoded.append(-float("inf"))
        else:
            decoded.append(float(item))
    return np.asarray(decoded, dtype=np.float64)


def compare_observable(
    source: object,
    target: object,
    *,
    atol: float,
    rtol: float,
    path: str = "observable",
) -> int:
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise AssertionError(f"observable structure mismatch: {path}")
    if source.get("kind") != target.get("kind"):
        raise AssertionError(f"observable kind mismatch: {path}")
    kind = source["kind"]
    if kind == "dict":
        if [item["key"] for item in source["items"]] != [item["key"] for item in target["items"]]:
            raise AssertionError(f"observable key mismatch: {path}")
        return sum(
            compare_observable(
                left["value"],
                right["value"],
                atol=atol,
                rtol=rtol,
                path=f"{path}.{left['key']['value']}",
            )
            for left, right in zip(source["items"], target["items"], strict=True)
        )
    if kind in {"list", "tuple"}:
        if len(source["items"]) != len(target["items"]):
            raise AssertionError(f"observable length mismatch: {path}")
        return sum(
            compare_observable(
                left,
                right,
                atol=atol,
                rtol=rtol,
                path=f"{path}[{index}]",
            )
            for index, (left, right) in enumerate(
                zip(source["items"], target["items"], strict=True)
            )
        )
    if kind == "scalar":
        left, right = source.get("value"), target.get("value")
        if isinstance(left, float) and isinstance(right, float):
            if not math.isclose(left, right, abs_tol=atol, rel_tol=rtol):
                raise AssertionError(f"numeric observable mismatch: {path}")
        elif left != right:
            raise AssertionError(f"exact observable mismatch: {path}")
        return 1
    if kind != "array":
        raise AssertionError(f"unsupported observable kind at {path}: {kind!r}")
    if source.get("shape") != target.get("shape") or source.get("dtype") != target.get("dtype"):
        raise AssertionError(f"observable metadata mismatch: {path}")
    dtype = np.dtype(str(source["dtype"]))
    if not np.issubdtype(dtype, np.inexact):
        if source.get("data") != target.get("data"):
            raise AssertionError(f"exact observable mismatch: {path}")
        return 1
    left = _decoded_float_data(source)
    right = _decoded_float_data(target)
    if not (
        np.array_equal(np.isnan(left), np.isnan(right))
        and np.array_equal(np.isposinf(left), np.isposinf(right))
        and np.array_equal(np.isneginf(left), np.isneginf(right))
    ):
        raise AssertionError(f"non-finite observable mismatch: {path}")
    finite = np.isfinite(left) & np.isfinite(right)
    if np.any(finite):
        try:
            np.testing.assert_allclose(left[finite], right[finite], atol=atol, rtol=rtol)
        except AssertionError as error:
            raise AssertionError(f"numeric observable mismatch: {path}") from error
    return 1


def code5_runner_params(runner_params: Mapping[str, object]) -> dict[str, object]:
    params = copy.deepcopy(dict(runner_params))
    config = params["config"]
    config.update(
        {
            "num_actors": 6,
            "expl_coef_block_size": 1,
            "horizon_length": 4,
            "seq_length": 4,
            "minibatch_size": 12,
            "mini_epochs": 1,
            "mixed_precision": True,
            "use_others_experience": "none",
        }
    )
    params["network"]["mlp"]["units"] = [32, 32, 16, 16]
    params["network"]["rnn"]["units"] = 16
    central = config["central_value_config"]
    central["minibatch_size"] = 12
    central["mini_epochs"] = 1
    central["network"]["mlp"]["units"] = [32, 32, 16, 16]
    return params


def validate_player_routing(cases: object) -> None:
    if not isinstance(cases, list) or any(not isinstance(case, Mapping) for case in cases):
        raise RuntimeError("player case inventory drift")
    if [case.get("env_count") for case in cases] != list(PLAYER_COUNTS):
        raise RuntimeError("player case inventory drift")
    for case in cases:
        count = case["env_count"]
        if count not in EXPECTED_PLAYER_ROWS:
            raise RuntimeError("player case inventory drift")
        if not np.array_equal(
            np.asarray(case.get("network_ids"), dtype=np.float32),
            np.asarray(NETWORK_IDS, dtype=np.float32),
        ):
            raise RuntimeError(f"player network IDs drift for N={count}")
        expected_embedding = torch.linspace(50.0, 0.0, count).numpy()
        if not np.array_equal(
            np.asarray(case.get("embedding_ids"), dtype=np.float32), expected_embedding
        ):
            raise RuntimeError(f"player embedding IDs drift for N={count}")
        if case.get("selected_rows") != EXPECTED_PLAYER_ROWS[count]:
            raise RuntimeError(f"player selected-row routing drift for N={count}")


_PAYLOAD_KEYS = frozenset(
    "file_name bytes sha256 outer_keys state_keys state env_state_is_none rng_saved "
    "missing_rng_components central_optimizer_saved actor_optimizer_state_entries "
    "native_save_owner native_load_owner".split()
)
_RESUME_KEYS = frozenset(
    "loaded_state runner_before_update env_set_state_calls external_rng_before "
    "external_rng_after first_action_input first_action_output first_value_input "
    "first_value_output native_return final_state".split()
)
_RUNNER_STATE_KEYS = frozenset(
    "epoch_num frame last_lr actor_optimizer_group_lrs central_lr "
    "central_optimizer_group_lrs central_optimizer_state_entries scaler".split()
)
_PLAYER_KEYS = frozenset({"owner", "cases"})
_PLAYER_CASE_KEYS = frozenset(
    "env_count network_ids embedding_ids selected_rows loaded_model env_set_state_calls "
    "action_low action_high deterministic stochastic".split()
)
_PLAYER_MODE_KEYS = frozenset(
    "observation rnn_before model_output selected_action rnn_after external_rng_before "
    "external_rng_after".split()
)
_COMPARISON_KEYS = frozenset(
    "tolerances player_counts observable_leaves exact_state_sections native_owner_paths "
    "full_runtime_bit_exact".split()
)


def _manifest_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"checkpoint {label} must be a mapping")
    return value


def _manifest_exact_keys(value: object, expected: set[str] | frozenset[str], label: str):
    mapping = _manifest_mapping(value, label)
    if set(mapping) != set(expected):
        missing = sorted(set(expected) - set(mapping))
        extra = sorted(set(mapping) - set(expected))
        raise RuntimeError(f"checkpoint {label} key drift: missing={missing}, extra={extra}")
    return mapping


def _validate_module_inventory(value: object) -> None:
    modules = _manifest_mapping(value, "loaded module inventory")
    if not modules:
        raise RuntimeError("checkpoint loaded module inventory is empty")
    paths: set[str] = set()
    for name, record in sorted(modules.items()):
        if not isinstance(name, str) or (name != "rl_games" and not name.startswith("rl_games.")):
            raise RuntimeError(f"checkpoint namespace module drift: {name!r}")
        record_mapping = _manifest_exact_keys(record, {"path", "sha256"}, f"module {name}")
        path = record_mapping["path"]
        digest = record_mapping["sha256"]
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or path in paths
        ):
            raise RuntimeError(f"checkpoint namespace root/path drift: {name}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"checkpoint namespace module hash drift: {name}")
        try:
            int(digest, 16)
        except ValueError as error:
            raise RuntimeError(f"checkpoint namespace module hash drift: {name}") from error
        paths.add(path)


def _validate_resume_section(resume: object, payload: Mapping[str, object]) -> None:
    value = _manifest_exact_keys(resume, _RESUME_KEYS, "resume")
    if value["loaded_state"] != payload["state"]:
        raise RuntimeError("checkpoint resume loaded state drift")
    runner = _manifest_exact_keys(value["runner_before_update"], _RUNNER_STATE_KEYS, "runner state")
    if not isinstance(runner["epoch_num"], int) or not isinstance(runner["frame"], int):
        raise RuntimeError("checkpoint runner state drift")
    if value["env_set_state_calls"] != [None]:
        raise RuntimeError("checkpoint env_state restore boundary drift")
    # These are deliberately observable dictionaries, not an assertion that the
    # checkpoint contains RNG state.  Source restores the caller's RNG boundary.
    for name in ("external_rng_before", "external_rng_after"):
        _manifest_mapping(value[name], f"resume.{name}")
    for name in (
        "first_action_input",
        "first_action_output",
        "first_value_input",
        "first_value_output",
        "native_return",
        "final_state",
    ):
        _manifest_mapping(value[name], f"resume.{name}")


def _validate_player_section(player: object) -> None:
    value = _manifest_exact_keys(player, _PLAYER_KEYS, "player")
    if value["owner"] != "rl_games.algos_torch.players.PpoPlayerContinuous":
        raise RuntimeError("checkpoint player owner drift")
    cases = value["cases"]
    validate_player_routing(cases)
    for case in cases:
        case_value = _manifest_exact_keys(case, _PLAYER_CASE_KEYS, "player case")
        if case_value["env_set_state_calls"] != []:
            raise RuntimeError("checkpoint player env_state boundary drift")
        for mode in ("deterministic", "stochastic"):
            mode_value = _manifest_exact_keys(
                case_value[mode], _PLAYER_MODE_KEYS, f"player {mode} case"
            )
            for name in (
                "observation",
                "rnn_before",
                "model_output",
                "selected_action",
                "rnn_after",
            ):
                _manifest_mapping(mode_value[name], f"player.{mode}.{name}")
            _manifest_mapping(mode_value["external_rng_before"], f"player.{mode}.rng_before")
            _manifest_mapping(mode_value["external_rng_after"], f"player.{mode}.rng_after")


def build_fixture_manifest(
    checkpoint: NativeCheckpoint,
    runtime: Mapping[str, object],
    runner_params: Mapping[str, object],
    generation_command: str = "test-only",
) -> dict[str, object]:
    """Build the checked-in, source-owned manifest for one native payload."""
    if not isinstance(checkpoint, NativeCheckpoint):
        raise RuntimeError("checkpoint capture type drift")
    if not isinstance(runtime, Mapping) or set(runtime) != {"resume", "player"}:
        raise RuntimeError("checkpoint runtime capture section drift")
    if not isinstance(runner_params, Mapping):
        raise RuntimeError("checkpoint runner params must be a mapping")
    payload = copy.deepcopy(checkpoint.metadata)
    _manifest_exact_keys(payload, _PAYLOAD_KEYS, "payload")
    # Run the same structural and routing checks used by replay before freezing it.
    compare = compare_runtime(runtime, copy.deepcopy(runtime))
    loaded_modules = copy.deepcopy(checkpoint.loaded_modules)
    _validate_module_inventory(loaded_modules)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generation_mode": "source-only",
        "ordinary_pytest_regenerates": False,
        "provenance": {
            "source_head": SOURCE_HEAD,
            "source_rl_games_tree": SOURCE_RL_GAMES_TREE,
            "owners": copy.deepcopy(SOURCE_OWNERS),
            "code3_anchors": copy.deepcopy(CODE3_ANCHORS),
            "native_owner_paths": list(compare.native_owner_paths),
            "loaded_rl_games_modules": loaded_modules,
        },
        "payload": payload,
        "resume": copy.deepcopy(runtime["resume"]),
        "player": copy.deepcopy(runtime["player"]),
        "comparison": {
            "tolerances": {"atol": 1e-6, "rtol": 1e-5},
            "player_counts": list(compare.player_counts),
            "observable_leaves": compare.observable_leaves,
            "exact_state_sections": compare.exact_state_sections,
            "native_owner_paths": list(compare.native_owner_paths),
            "full_runtime_bit_exact": False,
        },
        "runner_params": copy.deepcopy(dict(runner_params)),
        "fixture_files": [CHECKPOINT_FILE_NAME, MANIFEST_FILE_NAME],
        "canonical_payload_sha256": "0" * 64,
        "generation_command": str(generation_command),
    }
    manifest["canonical_payload_sha256"] = canonical_sha256(manifest)
    validate_fixture(manifest, checkpoint.payload)
    return manifest


def validate_fixture(manifest: object, payload: bytes) -> None:
    """Validate the strict schema and all Source checkpoint boundaries."""
    value = _manifest_exact_keys(manifest, MANIFEST_KEYS, "manifest")
    if value["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("checkpoint schema version drift")
    if value["generation_mode"] != "source-only":
        raise RuntimeError("checkpoint generation mode drift")
    if value["ordinary_pytest_regenerates"] is not False:
        raise RuntimeError("ordinary pytest must not regenerate checkpoint fixtures")
    if not isinstance(payload, bytes) or not payload:
        raise RuntimeError("checkpoint payload is empty")
    provenance = _manifest_exact_keys(
        value["provenance"],
        {
            "source_head",
            "source_rl_games_tree",
            "owners",
            "code3_anchors",
            "native_owner_paths",
            "loaded_rl_games_modules",
        },
        "provenance",
    )
    if provenance["source_head"] != SOURCE_HEAD:
        raise RuntimeError("checkpoint Source HEAD drift")
    if provenance["source_rl_games_tree"] != SOURCE_RL_GAMES_TREE:
        raise RuntimeError("checkpoint Source RL-Games tree drift")
    if provenance["owners"] != SOURCE_OWNERS:
        raise RuntimeError("checkpoint Source owner provenance drift")
    if provenance["code3_anchors"] != CODE3_ANCHORS:
        raise RuntimeError("checkpoint Code #3 anchor drift")
    expected_owners = {
        "rl_games.torch_runner.Runner",
        "rl_games.algos_torch.a2c_continuous.A2CAgent",
        "rl_games.algos_torch.players.PpoPlayerContinuous",
        "rl_games.algos_torch.torch_ext.save_checkpoint",
        "rl_games.algos_torch.torch_ext.load_checkpoint",
    }
    if not expected_owners.issubset(set(provenance["native_owner_paths"])):
        raise RuntimeError("checkpoint native owner inventory drift")
    _validate_module_inventory(provenance["loaded_rl_games_modules"])

    payload_record = _manifest_exact_keys(value["payload"], _PAYLOAD_KEYS, "payload")
    if payload_record["file_name"] != CHECKPOINT_FILE_NAME:
        raise RuntimeError("checkpoint payload file inventory drift")
    payload_hash = hashlib.sha256(payload).hexdigest()
    if payload_record["sha256"] != payload_hash:
        raise RuntimeError("checkpoint payload hash drift")
    if payload_record["bytes"] != len(payload):
        raise RuntimeError("checkpoint payload byte-size drift")
    if payload_record["outer_keys"] != [{"kind": "int", "value": 0}]:
        raise RuntimeError("checkpoint outer-key inventory drift")
    if payload_record["env_state_is_none"] is not True:
        raise RuntimeError("checkpoint env_state boundary drift")
    if payload_record["rng_saved"] is not False:
        raise RuntimeError("checkpoint RNG boundary drift")
    if payload_record["missing_rng_components"] != ["python", "numpy", "torch_cpu", "torch_cuda"]:
        raise RuntimeError("checkpoint RNG component inventory drift")
    if payload_record["central_optimizer_saved"] is not False:
        raise RuntimeError("checkpoint central optimizer boundary drift")
    if not isinstance(payload_record["state_keys"], list) or not payload_record["state_keys"]:
        raise RuntimeError("checkpoint state-key inventory drift")
    if not isinstance(payload_record["state"], Mapping):
        raise RuntimeError("checkpoint state descriptor drift")

    _validate_resume_section(value["resume"], payload_record)
    _validate_player_section(value["player"])
    comparison = _manifest_exact_keys(value["comparison"], _COMPARISON_KEYS, "comparison")
    if comparison["player_counts"] != list(PLAYER_COUNTS):
        raise RuntimeError("checkpoint comparison player inventory drift")
    if comparison["full_runtime_bit_exact"] is not False:
        raise RuntimeError("checkpoint full-runtime boundary drift")
    tolerances = _manifest_exact_keys(comparison["tolerances"], {"atol", "rtol"}, "tolerances")
    if tolerances != {"atol": 1e-6, "rtol": 1e-5}:
        raise RuntimeError("checkpoint comparison tolerance drift")
    if comparison["native_owner_paths"] != provenance["native_owner_paths"]:
        raise RuntimeError("checkpoint comparison owner inventory drift")
    if not isinstance(value["runner_params"], Mapping):
        raise RuntimeError("checkpoint runner params drift")
    if value["fixture_files"] != [CHECKPOINT_FILE_NAME, MANIFEST_FILE_NAME]:
        raise RuntimeError("checkpoint fixture file inventory drift")
    declared = value["canonical_payload_sha256"]
    if not isinstance(declared, str) or len(declared) != 64:
        raise RuntimeError("checkpoint manifest hash drift")
    if declared != canonical_sha256(value):
        raise RuntimeError("checkpoint manifest hash drift")


def _read_regular_file(path: Path, label: str) -> bytes:
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise RuntimeError(f"{label} is unavailable: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return path.read_bytes()


def _require_real_directory(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except OSError as error:
            raise RuntimeError(f"{label} component is unavailable: {current}") from error
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise RuntimeError(f"{label} component must be a real directory: {current}")
    return absolute


def load_fixture(root: Path = FIXTURE_ROOT) -> CheckpointFixture:
    root = _require_real_directory(Path(root), "checkpoint fixture root")
    payload_path = root / CHECKPOINT_FILE_NAME
    manifest_path = root / MANIFEST_FILE_NAME
    payload = _read_regular_file(payload_path, "checkpoint payload")
    manifest_bytes = _read_regular_file(manifest_path, "checkpoint manifest")
    if (
        EXPECTED_CHECKPOINT_SHA256
        and hashlib.sha256(payload).hexdigest() != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("checkpoint payload file anchor drift")
    if (
        EXPECTED_MANIFEST_SHA256
        and hashlib.sha256(manifest_bytes).hexdigest() != EXPECTED_MANIFEST_SHA256
    ):
        raise RuntimeError("checkpoint manifest file anchor drift")
    manifest = _strict_json_loads(manifest_bytes)
    validate_fixture(manifest, payload)
    if EXPECTED_PAYLOAD_SHA256 and manifest["canonical_payload_sha256"] != EXPECTED_PAYLOAD_SHA256:
        raise RuntimeError("checkpoint canonical payload anchor drift")
    return CheckpointFixture(manifest=manifest, payload=payload)


def replay_fixture(fixture: CheckpointFixture) -> CheckpointReplay:
    from tests.algos.rlgames_sapg import _runtime_requirement

    if not isinstance(fixture, CheckpointFixture):
        raise RuntimeError("checkpoint fixture type drift")
    validate_fixture(fixture.manifest, fixture.payload)
    _runtime_requirement.require_simtoolreal_rl_games()
    expected_root = _runtime_requirement.VENDOR_PACKAGE_ROOT
    assert_loaded_namespace(expected_root)
    runner_params = fixture.manifest.get("runner_params")
    if not isinstance(runner_params, Mapping):
        raise RuntimeError("checkpoint fixture runner params drift")
    target = capture_runtime(fixture.payload, runner_params, expected_root)
    source = {"resume": fixture.manifest["resume"], "player": fixture.manifest["player"]}
    result = compare_runtime(source, target)
    assert_loaded_namespace(expected_root)
    return result


# Explicit aliases make the fixture API read naturally in tests and scripts.
load_checkpoint_fixture = load_fixture
replay_checkpoint_fixture = replay_fixture


def _class_path(value: object) -> str:
    owner = type(value)
    return f"{owner.__module__}.{owner.__qualname__}"


def _make_agent(runner_params: Mapping[str, object], train_dir: Path):
    from rl_games.common import a2c_common
    from rl_games.torch_runner import Runner

    params = copy.deepcopy(dict(runner_params))
    params["config"]["train_dir"] = str(train_dir)
    env = CheckpointVecEnv(a2c_common.gym.spaces, torch.device("cuda:0"), num_envs=6)
    configure_canonical_execution()
    runner = Runner()
    configure_canonical_execution()
    runner.load({"params": params})
    runner.set_vec_env(env)
    agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
    if _class_path(runner) != "rl_games.torch_runner.Runner":
        raise RuntimeError("native Runner owner drift")
    if _class_path(agent) != "rl_games.algos_torch.a2c_continuous.A2CAgent":
        raise RuntimeError("native A2CAgent owner drift")
    return runner, agent, env


def _close_writer(agent: object) -> None:
    writer = getattr(agent, "writer", None)
    if writer is not None:
        writer.close()


def create_native_checkpoint(
    runner_params: Mapping[str, object], expected_package_root: Path
) -> NativeCheckpoint:
    from rl_games.algos_torch import torch_ext

    if not torch.cuda.is_available():
        raise RuntimeError("SAPG checkpoint oracle requires CUDA")
    assert_loaded_namespace(expected_package_root)
    configure_canonical_execution()
    with tempfile.TemporaryDirectory(prefix="unilab-sapg-checkpoint-create-") as directory:
        _runner, agent, env = _make_agent(runner_params, Path(directory))
        try:
            agent.init_tensors()
            agent.obs = agent.env_reset()
            fill_parameters(agent.model, "actor")
            fill_parameters(agent.central_value_net.model, "central")
            native_return = agent.train_epoch()
            if not isinstance(native_return, tuple) or len(native_return) != 12:
                raise RuntimeError("native pre-save train_epoch return drift")
            if len(agent.optimizer.state) == 0 or len(agent.central_value_net.optimizer.state) == 0:
                raise RuntimeError("native pre-save optimizer state was not initialized")
            agent.epoch_num = 7
            agent.frame = int(agent.curr_frames)
            base_path = Path(directory) / "source_checkpoint"
            agent.save(str(base_path))
            path = base_path.with_suffix(".pth")
            payload = path.read_bytes()
            checkpoint = torch_ext.load_checkpoint(str(path))
        finally:
            _close_writer(agent)

    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {0}:
        raise RuntimeError("native checkpoint outer rank layout drift")
    state = checkpoint[0]
    if not isinstance(state, Mapping):
        raise RuntimeError("native checkpoint rank state is not a mapping")
    state_keys = sorted(state)
    rng_keys = {
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
    }
    optimizer = state.get("optimizer")
    if not isinstance(optimizer, Mapping) or not isinstance(optimizer.get("state"), Mapping):
        raise RuntimeError("native actor optimizer payload drift")
    metadata = {
        "file_name": CHECKPOINT_FILE_NAME,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "outer_keys": [_state_key(key) for key in checkpoint],
        "state_keys": state_keys,
        "state": describe_state(state),
        "env_state_is_none": state.get("env_state", object()) is None,
        "rng_saved": bool(rng_keys.intersection(state)),
        "missing_rng_components": ["python", "numpy", "torch_cpu", "torch_cuda"],
        "central_optimizer_saved": "central_optimizer" in state,
        "actor_optimizer_state_entries": len(optimizer["state"]),
        "native_save_owner": "rl_games.algos_torch.a2c_continuous.A2CAgent.save",
        "native_load_owner": "rl_games.algos_torch.torch_ext.load_checkpoint",
    }
    if metadata["env_state_is_none"] is not True or metadata["rng_saved"] is not False:
        raise RuntimeError("native checkpoint boundary drift")
    if execution_platform() != CANONICAL_PLATFORM:
        raise RuntimeError("canonical checkpoint platform drift")
    loaded_modules = assert_loaded_namespace(expected_package_root)
    return NativeCheckpoint(
        payload=payload,
        metadata=metadata,
        loaded_modules=loaded_modules,
    )


def _seed_external_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rng_hashes() -> dict[str, object]:
    numpy_state = np.random.get_state()
    numpy_payload = (
        numpy_state[0].encode()
        + np.ascontiguousarray(numpy_state[1]).tobytes()
        + repr(numpy_state[2:]).encode()
    )
    return {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": hashlib.sha256(numpy_payload).hexdigest(),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        "torch_cuda": [
            hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
            for state in torch.cuda.get_rng_state_all()
        ],
    }


def _algorithm_return(native_return: tuple[object, ...]) -> dict[str, object]:
    if len(native_return) != 12:
        raise RuntimeError("native resume train_epoch return layout drift")
    return {
        "a_losses": native_return[4],
        "c_losses": native_return[5],
        "b_losses": native_return[6],
        "entropies": native_return[7],
        "kls": native_return[8],
        "last_lr": native_return[9],
        "lr_mul": native_return[10],
        "excluded_wall_clock_fields": [
            "step_time",
            "play_time",
            "update_time",
            "total_time",
        ],
    }


def _capture_resume(payload_path: Path, runner_params: Mapping[str, object]) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="unilab-sapg-checkpoint-resume-") as directory:
        _runner, agent, env = _make_agent(runner_params, Path(directory))
        original_action = agent.get_action_values
        original_value = agent.get_values
        first_action_input: dict[str, object] | None = None
        first_action_output: dict[str, object] | None = None
        first_value_input: dict[str, object] | None = None
        first_value_output: dict[str, object] | None = None
        try:
            agent.restore(str(payload_path))
            loaded_state = describe_state(agent.get_full_state_weights())
            runner_before_update = {
                "epoch_num": int(agent.epoch_num),
                "frame": int(agent.frame),
                "last_lr": float(agent.last_lr),
                "actor_optimizer_group_lrs": [
                    float(group["lr"]) for group in agent.optimizer.param_groups
                ],
                "central_lr": float(agent.central_value_net.lr),
                "central_optimizer_group_lrs": [
                    float(group["lr"]) for group in agent.central_value_net.optimizer.param_groups
                ],
                "central_optimizer_state_entries": len(agent.central_value_net.optimizer.state),
                "scaler": describe_state(agent.scaler.state_dict()),
            }
            agent.init_tensors()
            if agent.obs is not None:
                agent.obs = agent.obs_to_tensors(agent.obs)

            def observed_action(obs, rnn_states=None):
                nonlocal first_action_input, first_action_output
                result = original_action(obs, rnn_states)
                if first_action_input is None:
                    first_action_input = observe_tree({"obs": obs, "rnn_states": rnn_states})
                    first_action_output = observe_tree(result)
                return result

            def observed_value(obs, rnn_states):
                nonlocal first_value_input, first_value_output
                result = original_value(obs, rnn_states)
                if first_value_input is None:
                    first_value_input = observe_tree({"obs": obs, "rnn_states": rnn_states})
                    first_value_output = observe_tree(result)
                return result

            agent.get_action_values = observed_action
            agent.get_values = observed_value
            _seed_external_rng(20260821)
            rng_before = _rng_hashes()
            native_return = agent.train_epoch()
            rng_after = _rng_hashes()
            final_state = describe_state(agent.get_full_state_weights())
        finally:
            agent.get_action_values = original_action
            agent.get_values = original_value
            _close_writer(agent)

    if first_action_input is None or first_action_output is None:
        raise RuntimeError("native resume did not execute an action")
    if first_value_input is None or first_value_output is None:
        raise RuntimeError("native resume did not execute a value call")
    if len(env.set_env_state_calls) != 1 or env.set_env_state_calls[0] is not None:
        raise RuntimeError("native resume env_state=None boundary drift")
    return {
        "loaded_state": loaded_state,
        "runner_before_update": runner_before_update,
        "env_set_state_calls": list(env.set_env_state_calls),
        "external_rng_before": rng_before,
        "external_rng_after": rng_after,
        "first_action_input": first_action_input,
        "first_action_output": first_action_output,
        "first_value_input": first_value_input,
        "first_value_output": first_value_output,
        "native_return": observe_tree(_algorithm_return(native_return)),
        "final_state": final_state,
    }


def _selected_extra_parameter_rows(player: object, rnn_input: torch.Tensor) -> list[int]:
    flattened = rnn_input.transpose(0, 1).reshape(rnn_input.shape[1], -1)
    extra_parameters = player.model.a2c_network.extra_params.detach()
    width = extra_parameters.shape[1]
    embedded = flattened[:, -width:]
    result: list[int] = []
    for row in embedded:
        matches = [
            index for index, parameter in enumerate(extra_parameters) if torch.equal(row, parameter)
        ]
        if len(matches) != 1:
            raise RuntimeError("native player extra-parameter routing was not observable")
        result.append(matches[0])
    return result


def _capture_player_mode(player: object, env: CheckpointVecEnv, deterministic: bool):
    model_outputs: list[object] = []
    rnn_inputs: list[torch.Tensor] = []

    def model_hook(_module, _arguments, output):
        model_outputs.append(observe_tree(output))

    def rnn_pre_hook(_module, arguments):
        rnn_inputs.append(arguments[0].detach().clone())

    player.reset()
    env.step_index = 0
    player.has_batch_dimension = True
    observation = player.env_reset(env)
    rnn_before = observe_tree(player.states)
    _seed_external_rng(20260821 + int(deterministic))
    rng_before = _rng_hashes()
    model_handle = player.model.register_forward_hook(model_hook)
    rnn_handle = player.model.a2c_network.rnn.register_forward_pre_hook(rnn_pre_hook)
    try:
        selected_action = player.get_action(observation, is_deterministic=deterministic)
    finally:
        rnn_handle.remove()
        model_handle.remove()
    rng_after = _rng_hashes()
    if len(model_outputs) != 1 or len(rnn_inputs) != 1:
        raise RuntimeError("native player forward observation count drift")
    return {
        "observation": observe_tree(observation),
        "rnn_before": rnn_before,
        "model_output": model_outputs[0],
        "selected_action": observe_tree(selected_action),
        "rnn_after": observe_tree(player.states),
        "external_rng_before": rng_before,
        "external_rng_after": rng_after,
    }, _selected_extra_parameter_rows(player, rnn_inputs[0])


def _capture_player_case(
    payload_path: Path,
    runner_params: Mapping[str, object],
    env_count: int,
) -> dict[str, object]:
    from rl_games.common import a2c_common
    from rl_games.torch_runner import Runner

    params = copy.deepcopy(dict(runner_params))
    params["config"]["num_actors"] = env_count
    env = CheckpointVecEnv(
        a2c_common.gym.spaces,
        torch.device("cuda:0"),
        num_envs=env_count,
    )
    params["config"]["env_info"] = env.get_env_info()
    params["config"]["vec_env"] = env
    params["config"]["player"]["use_vecenv"] = True
    configure_canonical_execution()
    runner = Runner()
    configure_canonical_execution()
    runner.load({"params": params})
    player = runner.create_player()
    if _class_path(player) != "rl_games.algos_torch.players.PpoPlayerContinuous":
        raise RuntimeError("native PpoPlayerContinuous owner drift")
    player.restore(str(payload_path))
    deterministic, deterministic_rows = _capture_player_mode(player, env, True)
    stochastic, stochastic_rows = _capture_player_mode(player, env, False)
    if deterministic_rows != stochastic_rows:
        raise RuntimeError("native player routing differs by action mode")
    network_ids = player.model.a2c_network.param_ids.detach().cpu().tolist()
    embedding_ids = player.intr_reward_coef_embd[:, 0].detach().cpu().tolist()
    return {
        "env_count": env_count,
        "network_ids": network_ids,
        "embedding_ids": embedding_ids,
        "selected_rows": deterministic_rows,
        "loaded_model": describe_state(player.model.state_dict()),
        "env_set_state_calls": list(env.set_env_state_calls),
        "action_low": observe_tree(player.actions_low),
        "action_high": observe_tree(player.actions_high),
        "deterministic": deterministic,
        "stochastic": stochastic,
    }


def capture_runtime(
    payload: bytes,
    runner_params: Mapping[str, object],
    expected_package_root: Path,
) -> dict[str, object]:
    if not isinstance(payload, bytes) or not payload:
        raise RuntimeError("checkpoint payload is empty")
    assert_loaded_namespace(expected_package_root)
    configure_canonical_execution()
    with tempfile.TemporaryDirectory(prefix="unilab-sapg-checkpoint-runtime-") as directory:
        payload_path = Path(directory) / CHECKPOINT_FILE_NAME
        payload_path.write_bytes(payload)
        resume = _capture_resume(payload_path, runner_params)
        cases = [
            _capture_player_case(payload_path, runner_params, count) for count in PLAYER_COUNTS
        ]
    validate_player_routing(cases)
    if execution_platform() != CANONICAL_PLATFORM:
        raise RuntimeError("canonical checkpoint runtime platform drift")
    assert_loaded_namespace(expected_package_root)
    return {
        "resume": resume,
        "player": {
            "owner": "rl_games.algos_torch.players.PpoPlayerContinuous",
            "cases": cases,
        },
    }


def _require_equal(left: object, right: object, label: str) -> None:
    if left != right:
        raise AssertionError(f"{label} mismatch")


def compare_runtime(source: object, target: object) -> CheckpointReplay:
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise AssertionError("checkpoint runtime structure mismatch")
    if set(source) != {"resume", "player"} or set(target) != set(source):
        raise AssertionError("checkpoint runtime section mismatch")
    source_resume = source["resume"]
    target_resume = target["resume"]
    if not isinstance(source_resume, Mapping) or not isinstance(target_resume, Mapping):
        raise AssertionError("resume structure mismatch")

    exact_sections = 0
    _require_equal(
        source_resume.get("loaded_state"),
        target_resume.get("loaded_state"),
        "resume loaded state",
    )
    exact_sections += 1
    _require_equal(
        source_resume.get("runner_before_update"),
        target_resume.get("runner_before_update"),
        "resume runner state",
    )
    exact_sections += 1
    for name, label in (
        ("env_set_state_calls", "resume env state calls"),
        ("external_rng_before", "resume external RNG input"),
        ("external_rng_after", "resume external RNG output"),
        ("final_state", "resume final state"),
    ):
        _require_equal(source_resume.get(name), target_resume.get(name), label)
        exact_sections += 1

    observable_leaves = 0
    for name in (
        "first_action_input",
        "first_action_output",
        "first_value_input",
        "first_value_output",
        "native_return",
    ):
        observable_leaves += compare_observable(
            source_resume.get(name),
            target_resume.get(name),
            atol=1e-6,
            rtol=1e-5,
            path=f"resume.{name}",
        )

    source_player = source["player"]
    target_player = target["player"]
    if not isinstance(source_player, Mapping) or not isinstance(target_player, Mapping):
        raise AssertionError("player structure mismatch")
    _require_equal(source_player.get("owner"), target_player.get("owner"), "player owner")
    source_cases = source_player.get("cases")
    target_cases = target_player.get("cases")
    validate_player_routing(source_cases)
    validate_player_routing(target_cases)
    if not isinstance(source_cases, list) or not isinstance(target_cases, list):
        raise AssertionError("player case structure mismatch")
    if len(source_cases) != len(target_cases):
        raise AssertionError("player case count mismatch")
    for source_case, target_case in zip(source_cases, target_cases, strict=True):
        count = source_case["env_count"]
        for name in (
            "env_count",
            "network_ids",
            "embedding_ids",
            "selected_rows",
            "loaded_model",
            "env_set_state_calls",
        ):
            _require_equal(
                source_case.get(name),
                target_case.get(name),
                f"player N={count} {name}",
            )
        exact_sections += 2
        for name in ("action_low", "action_high"):
            observable_leaves += compare_observable(
                source_case.get(name),
                target_case.get(name),
                atol=1e-6,
                rtol=1e-5,
                path=f"player.N{count}.{name}",
            )
        for mode in ("deterministic", "stochastic"):
            source_mode = source_case.get(mode)
            target_mode = target_case.get(mode)
            if not isinstance(source_mode, Mapping) or not isinstance(target_mode, Mapping):
                raise AssertionError(f"player N={count} {mode} structure mismatch")
            for rng_name in ("external_rng_before", "external_rng_after"):
                _require_equal(
                    source_mode.get(rng_name),
                    target_mode.get(rng_name),
                    f"player N={count} {mode} {rng_name}",
                )
            for name in (
                "observation",
                "rnn_before",
                "model_output",
                "selected_action",
                "rnn_after",
            ):
                observable_leaves += compare_observable(
                    source_mode.get(name),
                    target_mode.get(name),
                    atol=1e-6,
                    rtol=1e-5,
                    path=f"player.N{count}.{mode}.{name}",
                )

    return CheckpointReplay(
        player_counts=tuple(case["env_count"] for case in source_cases),
        observable_leaves=observable_leaves,
        exact_state_sections=exact_sections,
        native_owner_paths=(
            "rl_games.torch_runner.Runner",
            "rl_games.algos_torch.a2c_continuous.A2CAgent",
            "rl_games.algos_torch.players.PpoPlayerContinuous",
            "rl_games.algos_torch.torch_ext.save_checkpoint",
            "rl_games.algos_torch.torch_ext.load_checkpoint",
        ),
    )
