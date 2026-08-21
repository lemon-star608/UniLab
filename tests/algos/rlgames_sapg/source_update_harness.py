from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from tests.algos.rlgames_sapg.source_network_harness import (
    configure_canonical_execution,
    execution_platform,
    fill_parameters,
)
from tests.algos.rlgames_sapg.source_rollout_harness import (
    CANONICAL_PLATFORM,
    SyntheticVecEnv,
    load_rollout_fixture,
)

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "simtoolreal_sapg"
FIXTURE_NPZ = FIXTURE_ROOT / "source_update_fp32.npz"
FIXTURE_MANIFEST = FIXTURE_ROOT / "source_update_manifest.json"

SCHEMA_VERSION = 2
EXPECTED_UPDATE_NPZ_SHA256 = "df58bb09d67edd24a19f2a164a4851fa24b9f2d305e9826c10433635cee78463"
EXPECTED_UPDATE_MANIFEST_SHA256 = "748be517553df7689ee4a06991241e37fc205336f6a5638f2bdd168735d57e45"
EXPECTED_UPDATE_PAYLOAD_SHA256 = "686331c200b809b66b0978b855c75d359e8fffb51f918ca9f5ee2312dd44f397"

SOURCE_HEAD = "2a9917533bfea70419ed2667a511d7238e5b3abc"
SOURCE_RL_GAMES_TREE = "7a6a0bb090998d00565aaefa6ab9f2b3d356ace2"
CODE3_ANCHORS = {
    "source_rollout_fp32.npz": ("3573cc3d0a3700b1c5985b63d7b16175d5ccee3dc8f4b7ef1a7bd7b6676819e8"),
    "source_rollout_manifest.json": (
        "785443d10e2037e0ca4e4b044dd1dc8207b438ea69555726eac9501ad8207d3f"
    ),
}

CASE_NAMES = ("normal_fp32", "normal_amp", "overflow_amp")
OWNERS = {
    "runner": "rl_games.torch_runner.Runner",
    "agent": "rl_games.algos_torch.a2c_continuous.A2CAgent",
    "actor_dataset": "rl_games.common.datasets.PPODataset",
    "central_value": "rl_games.algos_torch.central_value.CentralValueTrain",
}
SOURCE_OWNERS = {
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
RMS_ROLES = (
    "actor_input",
    "central_input",
    "actor_model_value",
    "active_central_value",
)
MANIFEST_KEYS = frozenset(
    "schema_version generation_mode ordinary_pytest_regenerates provenance platform "
    "canonical_platform code3_anchors runner_params capture_contract cases npz_arrays "
    "exact_comparison_inventory numeric_comparison_inventory tolerances fixture_files "
    "canonical_payload_sha256 generation_command".split()
)
CASE_KEYS = frozenset("name config owners input execution output restore".split())
INPUT_KEYS = frozenset("batch model optimizer scaler rms lr rng".split())
OUTPUT_KEYS = frozenset("prepared model optimizer scaler rms lr rng".split())
EXECUTION_KEYS = frozenset(
    "identity_shuffle_calls owner_call_order actor_update_attempts "
    "actor_optimizer_steps actor_scaler_skips central_optimizer_steps native_return "
    "overflow_mutation set_train_info_calls autocast".split()
)


class SnapshotStore:
    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.metadata: dict[str, dict[str, object]] = {}

    def tree(self, prefix: str, value: object, *, comparison: str) -> dict[str, object]:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if prefix in self.arrays:
                raise RuntimeError(f"duplicate snapshot array: {prefix}")
            array = np.ascontiguousarray(value)
            domain = comparison if np.issubdtype(array.dtype, np.inexact) else "exact"
            if domain not in {"numeric", "exact"}:
                raise RuntimeError(f"invalid comparison domain: {domain}")
            self.arrays[prefix] = array
            self.metadata[prefix] = {
                "name": prefix,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _array_sha256(array),
                "comparison": domain,
            }
            return {"kind": "array", "name": prefix}
        if isinstance(value, dict):
            return {
                "kind": "dict",
                "items": {
                    str(name): self.tree(f"{prefix}__{name}", item, comparison=comparison)
                    for name, item in sorted(value.items())
                },
            }
        if isinstance(value, (list, tuple)):
            return {
                "kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [
                    self.tree(f"{prefix}__{index}", item, comparison=comparison)
                    for index, item in enumerate(value)
                ],
            }
        if isinstance(value, np.generic):
            value = value.item()
        if value is None:
            return {"kind": "none"}
        if isinstance(value, (bool, int, str)):
            return {"kind": "scalar", "value": value}
        if isinstance(value, float):
            if math.isnan(value):
                value = {"nonfinite": "nan"}
            elif math.isinf(value):
                value = {"nonfinite": "+inf" if value > 0 else "-inf"}
            return {"kind": "scalar", "value": value}
        raise RuntimeError(f"unsupported snapshot value at {prefix}: {type(value)!r}")


@dataclass(frozen=True)
class UpdateFixture:
    manifest: dict[str, object]
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class ReplayResult:
    case_names: tuple[str, ...]
    native_owner_paths: tuple[str, ...]
    exact_array_count: int
    numeric_array_count: int
    max_abs_error: float


class _Patches:
    def __init__(self) -> None:
        self._undo: list[tuple[object, str, bool, object]] = []
        self._active = False
        self.restored = False

    def __enter__(self) -> _Patches:
        _require(not self._active, "patch scope is already active")
        self._active = True
        self.restored = False
        return self

    def _record_and_set(self, target: object, name: str, value: object) -> None:
        _require(self._active, "patch scope is not active")
        namespace = vars(target)
        existed = name in namespace
        previous = namespace.get(name)
        self._undo.append((target, name, existed, previous))
        setattr(target, name, value)

    def set_raw(self, owner: object, name: str, value: object) -> None:
        self._record_and_set(owner, name, value)

    def set(self, target: object, name: str, value: object) -> None:
        self._record_and_set(target, name, value)

    def restore(self) -> None:
        if self.restored:
            return
        first_error: BaseException | None = None
        for target, name, existed, previous in reversed(self._undo):
            try:
                if existed:
                    setattr(target, name, previous)
                else:
                    delattr(target, name)
            except BaseException as error:  # pragma: no cover - defensive cleanup
                if first_error is None:
                    first_error = error
        self._undo.clear()
        self._active = False
        self.restored = first_error is None
        if first_error is not None:
            raise RuntimeError("failed to restore native patch scope") from first_error

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.restore()
        return False


def canonical_payload(value: object) -> bytes:
    if isinstance(value, Mapping):
        value = dict(value)
        value.pop("canonical_payload_sha256", None)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_payload(value)).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _tensor_record(tensor: torch.Tensor) -> dict[str, object]:
    array = tensor.detach().cpu().numpy()
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _array_sha256(array),
    }


def snapshot_parameters(module: torch.nn.Module) -> dict[str, object]:
    parameters: dict[str, dict[str, object]] = {}
    ordered_records: list[dict[str, object]] = []
    for name, parameter in sorted(module.named_parameters()):
        record = _tensor_record(parameter)
        parameters[name] = record
        ordered_records.append({"name": name, **record})
    return {
        "parameters": parameters,
        "aggregate_sha256": _canonical_sha256(ordered_records),
    }


def _strict_json_value(value: object, *, location: str) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite JSON scalar at {location}")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_value(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            str(name): _strict_json_value(item, location=f"{location}.{name}")
            for name, item in sorted(value.items())
        }
    raise RuntimeError(f"unsupported JSON value at {location}: {type(value)!r}")


def snapshot_optimizer(
    optimizer: torch.optim.Optimizer,
    named_parameters: Mapping[str, torch.nn.Parameter],
) -> dict[str, object]:
    names_by_id = {id(parameter): name for name, parameter in named_parameters.items()}
    if len(names_by_id) != len(named_parameters):
        raise RuntimeError("optimizer parameter names are not unique")

    param_groups: list[dict[str, object]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        recorded_group: dict[str, object] = {}
        for key, value in group.items():
            if key == "params":
                try:
                    recorded_group[key] = [names_by_id[id(parameter)] for parameter in value]
                except KeyError as error:
                    raise RuntimeError(
                        f"unnamed optimizer parameter in group {group_index}"
                    ) from error
            else:
                recorded_group[key] = _strict_json_value(
                    value, location=f"optimizer.param_groups[{group_index}].{key}"
                )
        param_groups.append(recorded_group)

    state: dict[str, dict[str, object]] = {}
    for name, parameter in named_parameters.items():
        if parameter not in optimizer.state:
            continue
        parameter_state: dict[str, object] = {}
        for state_name, value in optimizer.state[parameter].items():
            key = str(state_name)
            if isinstance(value, torch.Tensor):
                record = _tensor_record(value)
                if key == "step" and value.ndim == 0:
                    record["value"] = int(value.item())
                parameter_state[key] = record
            else:
                parameter_state[key] = _strict_json_value(
                    value, location=f"optimizer.state.{name}.{key}"
                )
        state[name] = parameter_state

    uninitialized = sorted(
        name for name, parameter in named_parameters.items() if parameter not in optimizer.state
    )
    aggregate = {
        "param_groups": param_groups,
        "state": state,
        "uninitialized": uninitialized,
    }
    return {
        **aggregate,
        "aggregate_sha256": _canonical_sha256(aggregate),
    }


def snapshot_rms(
    store: SnapshotStore, roles: Mapping[str, torch.nn.Module], prefix: str
) -> dict[str, object]:
    path_parts = prefix.split("__")
    if "input" in path_parts and "output" not in path_parts:
        comparison = "exact"
    elif "output" in path_parts and "input" not in path_parts:
        comparison = "numeric"
    else:
        raise RuntimeError(f"RMS snapshot prefix must identify input or output: {prefix}")

    result: dict[str, object] = {}
    for role, rms in roles.items():
        mean = store.tree(
            f"{prefix}__{role}__mean",
            rms.running_mean,
            comparison=comparison,
        )
        variance = store.tree(
            f"{prefix}__{role}__var",
            rms.running_var,
            comparison=comparison,
        )
        if comparison == "numeric":
            store.metadata[str(mean["name"])]["comparison"] = "numeric"
            store.metadata[str(variance["name"])]["comparison"] = "numeric"
        result[role] = {
            "mean": mean,
            "var": variance,
            "count": float(rms.count.item()),
            "training": bool(rms.training),
        }
    return result


def snapshot_rng(store: SnapshotStore, prefix: str) -> dict[str, object]:
    numpy_state = np.random.get_state()
    torch_cuda_states = torch.cuda.get_rng_state_all()
    return {
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": store.tree(f"{prefix}__numpy__keys", numpy_state[1], comparison="exact"),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": store.tree(f"{prefix}__torch_cpu", torch.get_rng_state(), comparison="exact"),
        "torch_cuda": [
            store.tree(f"{prefix}__torch_cuda__{index}", state, comparison="exact")
            for index, state in enumerate(torch_cuda_states)
        ],
    }


def _class_path(value: object) -> str:
    owner = type(value)
    return f"{owner.__module__}.{owner.__qualname__}"


def snapshot_lr(agent: object) -> dict[str, object]:
    central = agent.central_value_net
    return {
        "actor": {
            "last_lr": float(agent.last_lr),
            "optimizer_group_lrs": [float(group["lr"]) for group in agent.optimizer.param_groups],
            "scheduler_class": _class_path(agent.scheduler),
            "scheduler_state": _strict_json_value(
                vars(agent.scheduler), location="actor.scheduler"
            ),
        },
        "central": {
            "lr": float(central.lr),
            "optimizer_group_lrs": [float(group["lr"]) for group in central.optimizer.param_groups],
            "scheduler_class": _class_path(central.scheduler),
            "scheduler_state": _strict_json_value(
                vars(central.scheduler), location="central.scheduler"
            ),
        },
    }


def snapshot_scaler(scaler: object) -> dict[str, object]:
    return {
        "enabled": bool(scaler.is_enabled()),
        "state_dict": _strict_json_value(
            scaler.state_dict(), location="gradient_scaler.state_dict"
        ),
    }


def _clone_tree(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {name: _clone_tree(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _raw_rng_state() -> tuple[tuple[object, ...], torch.Tensor, list[torch.Tensor]]:
    numpy_state = np.random.get_state()
    copied_numpy = (
        numpy_state[0],
        numpy_state[1].copy(),
        numpy_state[2],
        numpy_state[3],
        numpy_state[4],
    )
    return (
        copied_numpy,
        torch.get_rng_state().clone(),
        [state.clone() for state in torch.cuda.get_rng_state_all()],
    )


def _same_raw_rng_state(
    left: tuple[tuple[object, ...], torch.Tensor, list[torch.Tensor]],
    right: tuple[tuple[object, ...], torch.Tensor, list[torch.Tensor]],
) -> bool:
    left_numpy, left_cpu, left_cuda = left
    right_numpy, right_cpu, right_cuda = right
    return bool(
        left_numpy[0] == right_numpy[0]
        and np.array_equal(left_numpy[1], right_numpy[1])
        and left_numpy[2:] == right_numpy[2:]
        and torch.equal(left_cpu, right_cpu)
        and len(left_cuda) == len(right_cuda)
        and all(
            torch.equal(left_state, right_state)
            for left_state, right_state in zip(left_cuda, right_cuda, strict=True)
        )
    )


def _load_code3_batch(device: torch.device) -> dict[str, object]:
    from tests.algos.rlgames_sapg import source_rollout_harness

    for filename, expected_sha256 in CODE3_ANCHORS.items():
        path = source_rollout_harness.FIXTURE_ROOT / filename
        _require(
            hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256,
            f"Code #3 fixture anchor drift: {filename}",
        )
    fixture = load_rollout_fixture()
    prefix = "buffer_post_shuffle__"
    tensor_fields = (
        "actions",
        "dones",
        "mus",
        "neglogpacs",
        "obses",
        "off_policy_mask",
        "returns",
        "sigmas",
        "states",
        "values",
    )
    batch = {
        name: torch.as_tensor(fixture.arrays[f"{prefix}{name}"].copy(), device=device)
        for name in tensor_fields
    }
    batch["rnn_states"] = [
        torch.as_tensor(fixture.arrays[f"{prefix}rnn_states__{index}"].copy(), device=device)
        for index in range(2)
    ]
    batch["played_frames"] = 48
    batch["step_time"] = 0.0
    _require(len(batch["returns"]) == 56, "Code #3 batch row count drift")
    _require(batch["rnn_states"][0].shape[1] == 14, "Code #3 sequence count drift")
    return batch


def _declared_batch_pair(batch: Mapping[str, object], ps_extras: object) -> dict[str, object]:
    return {
        "batch_dict": {
            name: item for name, item in batch.items() if name not in {"played_frames", "step_time"}
        },
        "played_frames": batch["played_frames"],
        "step_time": batch["step_time"],
        "ps_extras": ps_extras,
    }


def _close_agent_writer(agent: object, *, active_error: BaseException | None) -> None:
    try:
        writer = agent.writer
        if writer is not None:
            writer.close()
    except BaseException:
        if active_error is None:
            raise


def _agent_and_env(runner_params: dict[str, object], train_dir: Path):
    from rl_games.common import a2c_common
    from rl_games.torch_runner import Runner

    configure_canonical_execution()
    params = copy.deepcopy(runner_params)
    params["config"]["train_dir"] = str(train_dir)
    env = SyntheticVecEnv(a2c_common.gym.spaces, torch.device("cuda:0"))
    runner = Runner()
    configure_canonical_execution()
    runner.load({"params": params})
    runner.set_vec_env(env)
    agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
    try:
        agent.init_tensors()
        agent.obs = agent.env_reset()
        _require(_class_path(runner) == OWNERS["runner"], "native Runner owner drift")
        _require(_class_path(agent) == OWNERS["agent"], "native A2CAgent owner drift")
        _require(
            _class_path(agent.dataset) == OWNERS["actor_dataset"],
            "native actor PPODataset owner drift",
        )
        _require(
            _class_path(agent.central_value_net) == OWNERS["central_value"],
            "native central-value owner drift",
        )
    except BaseException as error:
        _close_agent_writer(agent, active_error=error)
        raise
    return agent, env


@contextmanager
def _writer_owned_agent_and_env(
    runner_params: dict[str, object], train_dir: Path
) -> Iterator[tuple[object, object]]:
    agent, env = _agent_and_env(runner_params, train_dir)
    active_error: BaseException | None = None
    try:
        yield agent, env
    except BaseException as error:
        active_error = error
        raise
    finally:
        _close_agent_writer(agent, active_error=active_error)


def _rms_roles(agent: object) -> dict[str, torch.nn.Module]:
    roles = {
        "actor_input": agent.model.running_mean_std,
        "central_input": agent.central_value_net.model.running_mean_std,
        "actor_model_value": agent.model.value_mean_std,
        "active_central_value": agent.value_mean_std,
    }
    _require(len({id(module) for module in roles.values()}) == 4, "RMS roles are not distinct")
    _require(
        roles["active_central_value"] is agent.central_value_net.model.value_mean_std,
        "active value RMS is not the central-value RMS",
    )
    return roles


def _model_snapshots(agent: object) -> dict[str, object]:
    return {
        "actor": snapshot_parameters(agent.model),
        "central": snapshot_parameters(agent.central_value_net.model),
    }


def _optimizer_snapshots(agent: object) -> dict[str, object]:
    return {
        "actor": snapshot_optimizer(agent.optimizer, dict(agent.model.named_parameters())),
        "central": snapshot_optimizer(
            agent.central_value_net.optimizer,
            dict(agent.central_value_net.model.named_parameters()),
        ),
    }


def _dataset_batch_sizes(dataset: object) -> list[int]:
    sizes = [int(len(dataset[index]["obs"])) for index in range(len(dataset))]
    _require(sizes == [12, 12, 12, 20], f"native dataset batch-size drift: {sizes}")
    return sizes


def _tensor_difference_paths(before: torch.Tensor, after: torch.Tensor, prefix: str) -> list[str]:
    if before.shape != after.shape or before.dtype != after.dtype:
        return [prefix]
    equal = torch.eq(before, after)
    if before.is_floating_point() or before.is_complex():
        equal = equal | (torch.isnan(before) & torch.isnan(after))
    indices = torch.nonzero(~equal, as_tuple=False).detach().cpu().tolist()
    return [prefix + "".join(f"[{index}]" for index in coordinate) for coordinate in indices]


def _tree_difference_paths(before: object, after: object, prefix: str = "") -> list[str]:
    if isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor):
        return _tensor_difference_paths(before, after, prefix)
    if isinstance(before, dict) and isinstance(after, dict):
        if set(before) != set(after):
            return [prefix or "<root>"]
        return [
            path
            for name in before
            for path in _tree_difference_paths(
                before[name], after[name], f"{prefix}.{name}" if prefix else str(name)
            )
        ]
    if isinstance(before, (list, tuple)) and isinstance(after, type(before)):
        if len(before) != len(after):
            return [prefix or "<root>"]
        return [
            path
            for index, (left, right) in enumerate(zip(before, after, strict=True))
            for path in _tree_difference_paths(left, right, f"{prefix}[{index}]")
        ]
    if before != after:
        return [prefix or "<root>"]
    return []


def _validate_array_inventory(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]
) -> None:
    actual_names = set(arrays)
    expected_names = set(metadata)
    missing = sorted(expected_names - actual_names)
    if missing:
        raise RuntimeError(f"missing snapshot arrays: {missing}")
    extra = sorted(actual_names - expected_names)
    if extra:
        raise RuntimeError(f"extra snapshot arrays: {extra}")

    for name in sorted(expected_names):
        array = arrays[name]
        record = metadata[name]
        if not isinstance(array, np.ndarray):
            raise RuntimeError(f"array type drift for {name}: {type(array)!r}")
        if not isinstance(record, Mapping):
            raise RuntimeError(f"invalid array metadata for {name}")
        if set(record) != {"name", "shape", "dtype", "sha256", "comparison"}:
            raise RuntimeError(f"invalid array metadata keys for {name}")
        if record.get("name") != name:
            raise RuntimeError(f"array name drift for {name}")
        if record.get("comparison") not in {"exact", "numeric"}:
            raise RuntimeError(f"invalid array comparison domain for {name}")
        if list(array.shape) != record.get("shape"):
            raise RuntimeError(f"array shape drift for {name}")
        if str(array.dtype) != record.get("dtype"):
            raise RuntimeError(f"array dtype drift for {name}")
        if _array_sha256(array) != record.get("sha256"):
            raise RuntimeError(f"array content drift for {name}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _require_exact_keys(value: object, keys: set[str] | frozenset[str], label: str) -> Mapping:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    actual = set(value)
    _require(actual == set(keys), f"{label} keys mismatch: {sorted(actual ^ set(keys))}")
    return value


def _require_sha256(value: object, label: str) -> None:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA256",
    )


def _array_references(value: object, *, label: str) -> list[str]:
    if isinstance(value, Mapping):
        kind = value.get("kind")
        if kind == "array":
            _require_exact_keys(value, {"kind", "name"}, label)
            name = value["name"]
            _require(isinstance(name, str) and bool(name), f"{label} has invalid array name")
            return [name]
        if kind == "dict":
            _require_exact_keys(value, {"kind", "items"}, label)
            items = value["items"]
            _require(isinstance(items, Mapping), f"{label}.items must be a mapping")
            return [
                name
                for child_name, child in items.items()
                for name in _array_references(child, label=f"{label}.{child_name}")
            ]
        if kind in {"list", "tuple"}:
            _require_exact_keys(value, {"kind", "items"}, label)
            items = value["items"]
            _require(isinstance(items, list), f"{label}.items must be a list")
            return [
                name
                for index, child in enumerate(items)
                for name in _array_references(child, label=f"{label}[{index}]")
            ]
        if kind == "scalar":
            _require_exact_keys(value, {"kind", "value"}, label)
            scalar = value["value"]
            if isinstance(scalar, Mapping):
                _require_exact_keys(scalar, {"nonfinite"}, f"{label}.value")
                _require(
                    scalar["nonfinite"] in {"nan", "+inf", "-inf"},
                    f"{label} has invalid non-finite scalar tag",
                )
            else:
                _require(
                    scalar is None or isinstance(scalar, (bool, int, float, str)),
                    f"{label} has unsupported scalar",
                )
            return []
        if kind == "none":
            _require_exact_keys(value, {"kind"}, label)
            return []
        _require(kind is None, f"{label} has invalid snapshot kind: {kind!r}")
        return [
            name
            for child_name, child in value.items()
            for name in _array_references(child, label=f"{label}.{child_name}")
        ]
    if isinstance(value, list):
        return [
            name
            for index, child in enumerate(value)
            for name in _array_references(child, label=f"{label}[{index}]")
        ]
    return []


def _validate_parameter_snapshot(snapshot: object, label: str) -> set[str]:
    value = _require_exact_keys(snapshot, {"parameters", "aggregate_sha256"}, label)
    parameters = value["parameters"]
    _require(isinstance(parameters, Mapping) and bool(parameters), f"{label} is empty")
    names: set[str] = set()
    ordered_records: list[dict[str, object]] = []
    for name, record in sorted(parameters.items()):
        _require(isinstance(name, str) and bool(name), f"{label} has invalid parameter name")
        tensor = _require_exact_keys(record, {"shape", "dtype", "sha256"}, f"{label}.{name}")
        _require(
            isinstance(tensor["shape"], list)
            and all(isinstance(size, int) and size >= 0 for size in tensor["shape"]),
            f"{label}.{name} has invalid shape",
        )
        _require(isinstance(tensor["dtype"], str), f"{label}.{name} has invalid dtype")
        _require_sha256(tensor["sha256"], f"{label}.{name}.sha256")
        names.add(name)
        ordered_records.append({"name": name, **tensor})
    _require_sha256(value["aggregate_sha256"], f"{label}.aggregate_sha256")
    _require(
        value["aggregate_sha256"] == _canonical_sha256(ordered_records),
        f"{label} parameter aggregate SHA256 mismatch",
    )
    return names


def _validate_optimizer_snapshot(snapshot: object, parameter_names: set[str], label: str) -> int:
    value = _require_exact_keys(
        snapshot,
        {"param_groups", "state", "uninitialized", "aggregate_sha256"},
        label,
    )
    groups = value["param_groups"]
    _require(isinstance(groups, list) and bool(groups), f"{label}.param_groups is empty")
    grouped: list[str] = []
    for index, group in enumerate(groups):
        _require(isinstance(group, Mapping), f"{label}.param_groups[{index}] is invalid")
        params = group.get("params")
        _require(
            isinstance(params, list) and all(isinstance(name, str) for name in params),
            f"{label}.param_groups[{index}].params is invalid",
        )
        for group_name, group_value in group.items():
            _require(
                isinstance(group_name, str),
                f"{label}.param_groups[{index}] has a non-string key",
            )
            if group_name == "params":
                continue
            normalized = _strict_json_value(
                group_value,
                location=f"{label}.param_groups[{index}].{group_name}",
            )
            _require(
                type(normalized) is type(group_value) and normalized == group_value,
                f"{label}.param_groups[{index}].{group_name} is not strict JSON",
            )
        grouped.extend(params)
    _require(
        len(grouped) == len(set(grouped)) and set(grouped) == parameter_names,
        f"{label} parameter-group inventory is incomplete",
    )

    state = value["state"]
    uninitialized = value["uninitialized"]
    _require(isinstance(state, Mapping), f"{label}.state is invalid")
    _require(
        isinstance(uninitialized, list)
        and all(isinstance(name, str) for name in uninitialized)
        and uninitialized == sorted(uninitialized),
        f"{label}.uninitialized is invalid",
    )
    _require(set(state) <= parameter_names, f"{label}.state has unknown parameters")
    _require(
        len(uninitialized) == len(set(uninitialized))
        and set(state).isdisjoint(uninitialized)
        and set(state) | set(uninitialized) == parameter_names,
        f"{label} initialized-state inventory is incomplete",
    )

    steps: set[int] = set()
    for parameter_name, parameter_state in state.items():
        _require(
            isinstance(parameter_state, Mapping),
            f"{label}.state.{parameter_name} is invalid",
        )
        _require(
            set(parameter_state) == {"step", "exp_avg", "exp_avg_sq"},
            f"{label} optimizer state-key inventory mismatch for {parameter_name}",
        )
        for state_name, record in parameter_state.items():
            expected_keys = (
                {"shape", "dtype", "sha256", "value"}
                if state_name == "step"
                else {"shape", "dtype", "sha256"}
            )
            tensor = _require_exact_keys(
                record,
                expected_keys,
                f"{label}.state.{parameter_name}.{state_name}",
            )
            _require(
                isinstance(tensor["shape"], list)
                and all(isinstance(size, int) and size >= 0 for size in tensor["shape"]),
                f"{label}.state.{parameter_name}.{state_name} has invalid shape",
            )
            _require(
                isinstance(tensor["dtype"], str),
                f"{label}.state.{parameter_name}.{state_name} has invalid dtype",
            )
            _require_sha256(tensor["sha256"], f"{label}.state.{parameter_name}.{state_name}.sha256")
            if state_name == "step":
                _require(
                    tensor["shape"] == []
                    and isinstance(tensor["value"], int)
                    and not isinstance(tensor["value"], bool)
                    and tensor["value"] >= 0,
                    f"{label}.state.{parameter_name}.step must be a non-negative scalar",
                )
                steps.add(tensor["value"])
    _require(len(steps) <= 1, f"{label} has inconsistent optimizer steps")
    _require_sha256(value["aggregate_sha256"], f"{label}.aggregate_sha256")
    aggregate = {
        "param_groups": groups,
        "state": state,
        "uninitialized": uninitialized,
    }
    _require(
        value["aggregate_sha256"] == _canonical_sha256(aggregate),
        f"{label} optimizer aggregate SHA256 mismatch",
    )
    return next(iter(steps), 0)


def _validate_scaler(snapshot: object, *, enabled: bool, label: str) -> None:
    value = _require_exact_keys(snapshot, {"enabled", "state_dict"}, label)
    _require(value["enabled"] is enabled, f"{label} enabled mode mismatch")
    state = value["state_dict"]
    _require(isinstance(state, Mapping), f"{label}.state_dict is invalid")
    if enabled:
        state = _require_exact_keys(
            state,
            {
                "scale",
                "growth_factor",
                "backoff_factor",
                "growth_interval",
                "_growth_tracker",
            },
            f"{label}.state_dict",
        )
        _require(
            isinstance(state["scale"], float)
            and math.isfinite(state["scale"])
            and state["scale"] > 0,
            f"{label} is missing a finite positive scale",
        )
        _require(
            isinstance(state["growth_factor"], float)
            and math.isfinite(state["growth_factor"])
            and state["growth_factor"] > 1,
            f"{label} growth factor is invalid",
        )
        _require(
            isinstance(state["backoff_factor"], float)
            and math.isfinite(state["backoff_factor"])
            and 0 < state["backoff_factor"] < 1,
            f"{label} backoff factor is invalid",
        )
        _require(
            isinstance(state["growth_interval"], int)
            and not isinstance(state["growth_interval"], bool)
            and state["growth_interval"] > 0,
            f"{label} growth interval is invalid",
        )
        _require(
            isinstance(state["_growth_tracker"], int)
            and not isinstance(state["_growth_tracker"], bool)
            and state["_growth_tracker"] >= 0,
            f"{label} growth tracker is invalid",
        )
    else:
        _require(not state, f"{label} disabled scaler must have empty state")


def _validate_rms(
    rms: object,
    *,
    label: str,
    expected_domain: str,
    metadata: Mapping[str, object],
) -> None:
    _require(isinstance(rms, Mapping), f"{label} must be a mapping")
    _require(list(rms) == list(RMS_ROLES), f"{label} RMS roles mismatch")
    names: list[str] = []
    for role in RMS_ROLES:
        state = _require_exact_keys(
            rms[role], {"mean", "var", "count", "training"}, f"{label}.{role}"
        )
        role_names = _array_references(state["mean"], label=f"{label}.{role}.mean")
        role_names += _array_references(state["var"], label=f"{label}.{role}.var")
        _require(len(role_names) == 2, f"{label}.{role} must own mean and variance")
        for name in role_names:
            record = metadata.get(name)
            _require(
                isinstance(record, Mapping) and record.get("comparison") == expected_domain,
                f"{label}.{role} has wrong comparison domain",
            )
        names.extend(role_names)
        _require(
            isinstance(state["count"], (int, float)) and math.isfinite(float(state["count"])),
            f"{label}.{role}.count is invalid",
        )
        _require(isinstance(state["training"], bool), f"{label}.{role}.training is invalid")
    _require(len(names) == len(set(names)), f"{label} RMS arrays are aliased")


def _validate_rng(
    rng: object, *, label: str, metadata: Mapping[str, object], cuda_count: int | None
) -> None:
    value = _require_exact_keys(rng, {"numpy", "torch_cpu", "torch_cuda"}, label)
    numpy_state = _require_exact_keys(
        value["numpy"],
        {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"},
        f"{label}.numpy",
    )
    _require(isinstance(numpy_state["algorithm"], str), f"{label}.numpy algorithm invalid")
    _require(isinstance(numpy_state["position"], int), f"{label}.numpy position invalid")
    _require(isinstance(numpy_state["has_gauss"], int), f"{label}.numpy gaussian flag invalid")
    _require(
        isinstance(numpy_state["cached_gaussian"], (int, float))
        and math.isfinite(float(numpy_state["cached_gaussian"])),
        f"{label}.numpy cached gaussian invalid",
    )
    references = _array_references(numpy_state["keys"], label=f"{label}.numpy.keys")
    references += _array_references(value["torch_cpu"], label=f"{label}.torch_cpu")
    cuda = value["torch_cuda"]
    _require(isinstance(cuda, list), f"{label}.torch_cuda must be a list")
    if cuda_count is not None:
        _require(len(cuda) == cuda_count, f"{label}.torch_cuda device inventory mismatch")
    references += [
        name
        for index, state in enumerate(cuda)
        for name in _array_references(state, label=f"{label}.torch_cuda[{index}]")
    ]
    _require(len(references) == 2 + len(cuda), f"{label} RNG array inventory incomplete")
    for name in references:
        record = metadata.get(name)
        _require(
            isinstance(record, Mapping) and record.get("comparison") == "exact",
            f"{label} RNG state must compare exactly",
        )


def _validate_lr(snapshot: object, label: str) -> None:
    value = _require_exact_keys(snapshot, {"actor", "central"}, label)
    actor = _require_exact_keys(
        value["actor"],
        {"last_lr", "optimizer_group_lrs", "scheduler_class", "scheduler_state"},
        f"{label}.actor",
    )
    central = _require_exact_keys(
        value["central"],
        {"lr", "optimizer_group_lrs", "scheduler_class", "scheduler_state"},
        f"{label}.central",
    )
    for owner, state, lr_key in (
        ("actor", actor, "last_lr"),
        ("central", central, "lr"),
    ):
        _require(
            isinstance(state[lr_key], (int, float)) and math.isfinite(float(state[lr_key])),
            f"{label}.{owner}.{lr_key} is invalid",
        )
        group_lrs = state["optimizer_group_lrs"]
        _require(
            isinstance(group_lrs, list)
            and bool(group_lrs)
            and all(isinstance(lr, (int, float)) and math.isfinite(float(lr)) for lr in group_lrs),
            f"{label}.{owner}.optimizer_group_lrs is invalid",
        )
        _require(
            isinstance(state["scheduler_class"], str) and bool(state["scheduler_class"]),
            f"{label}.{owner}.scheduler_class is invalid",
        )
        _require(
            isinstance(state["scheduler_state"], Mapping),
            f"{label}.{owner}.scheduler_state is invalid",
        )


def _validate_native_return(
    value: object,
    *,
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, int]:
    items = _require_snapshot_dict_items(value, "execution.native_return")
    _require(
        set(items)
        == {
            "a_losses",
            "c_losses",
            "b_losses",
            "entropies",
            "kls",
            "last_lr",
            "lr_mul",
            "excluded_wall_clock_fields",
        },
        "execution.native_return keys mismatch",
    )
    lengths: dict[str, int] = {}
    for name in ("a_losses", "c_losses", "b_losses", "entropies", "kls"):
        sequence = _require_exact_keys(
            items[name],
            {"kind", "items"},
            f"execution.native_return.{name}",
        )
        _require(
            sequence["kind"] == "list" and isinstance(sequence["items"], list),
            f"execution.native_return.{name} must be a list tree",
        )
        lengths[name] = len(sequence["items"])
        for index, child in enumerate(sequence["items"]):
            reference = _require_exact_keys(
                child,
                {"kind", "name"},
                f"execution.native_return.{name}[{index}]",
            )
            array_name = reference["name"]
            _require(
                reference["kind"] == "array"
                and isinstance(array_name, str)
                and isinstance(arrays.get(array_name), np.ndarray),
                f"execution.native_return.{name}[{index}] must be an array reference",
            )
            record = metadata.get(array_name)
            _require(
                isinstance(record, Mapping)
                and record.get("comparison") == "numeric"
                and np.issubdtype(arrays[array_name].dtype, np.floating),
                f"execution.native_return.{name}[{index}] native return must compare numerically",
            )

    for name in ("last_lr", "lr_mul"):
        scalar = _require_exact_keys(
            items[name],
            {"kind", "value"},
            f"execution.native_return.{name}",
        )
        scalar_value = scalar["value"]
        _require(
            scalar["kind"] == "scalar"
            and isinstance(scalar_value, (int, float))
            and not isinstance(scalar_value, bool)
            and math.isfinite(float(scalar_value)),
            f"execution.native_return.{name} must be a finite scalar",
        )

    excluded = _require_exact_keys(
        items["excluded_wall_clock_fields"],
        {"kind", "items"},
        "execution.native_return.excluded_wall_clock_fields",
    )
    _require(
        excluded["kind"] == "list" and isinstance(excluded["items"], list),
        "execution.native_return.excluded_wall_clock_fields must be a list tree",
    )
    excluded_values: list[object] = []
    for index, child in enumerate(excluded["items"]):
        scalar = _require_exact_keys(
            child,
            {"kind", "value"},
            f"execution.native_return.excluded_wall_clock_fields[{index}]",
        )
        _require(
            scalar["kind"] == "scalar" and isinstance(scalar["value"], str),
            "execution.native_return excluded wall-clock field must be a string scalar",
        )
        excluded_values.append(scalar["value"])
    _require(
        excluded_values == ["play_time", "update_time", "total_time"],
        "execution.native_return excluded wall-clock fields mismatch",
    )
    return lengths


def _require_snapshot_dict_items(value: object, label: str) -> Mapping[str, object]:
    _require(
        isinstance(value, Mapping) and value.get("kind") == "dict",
        f"{label} must be a dict tree",
    )
    tree = _require_exact_keys(value, {"kind", "items"}, label)
    items = tree["items"]
    _require(isinstance(items, Mapping), f"{label}.items must be a mapping")
    return items


def _require_nonempty_array_reference(
    value: object,
    *,
    label: str,
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> np.ndarray:
    _require(
        isinstance(value, Mapping) and value.get("kind") == "array",
        f"{label} must be an array reference",
    )
    reference = _require_exact_keys(value, {"kind", "name"}, label)
    name = reference["name"]
    array = arrays.get(name) if isinstance(name, str) else None
    record = metadata.get(name) if isinstance(name, str) else None
    _require(
        isinstance(name, str)
        and bool(name)
        and isinstance(array, np.ndarray)
        and isinstance(record, Mapping),
        f"{label} must resolve to a captured array",
    )
    _require(
        array.ndim > 0 and array.size > 0 and all(size > 0 for size in array.shape),
        f"{label} must reference a non-empty, non-scalar array",
    )
    return array


def _require_row_array(
    value: object,
    *,
    label: str,
    rows: int,
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> None:
    array = _require_nonempty_array_reference(
        value,
        label=label,
        metadata=metadata,
        arrays=arrays,
    )
    _require(array.shape[0] == rows, f"{label} row count mismatch")


def _require_rnn_state_tree(
    value: object,
    *,
    label: str,
    sequences: int,
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
) -> None:
    _require(
        isinstance(value, Mapping)
        and value.get("kind") == "list"
        and isinstance(value.get("items"), list)
        and len(value["items"]) == 2,
        f"{label} must be a two-item list tree",
    )
    tree = _require_exact_keys(value, {"kind", "items"}, label)
    for index, reference in enumerate(tree["items"]):
        item_label = f"{label}[{index}]"
        array = _require_nonempty_array_reference(
            reference,
            label=item_label,
            metadata=metadata,
            arrays=arrays,
        )
        _require(array.ndim == 3, f"{item_label} must be a rank-3 array")
        _require(array.shape[1] == sequences, f"{item_label} sequence count mismatch")


def _validate_case(
    case: object,
    *,
    case_name: str,
    rows: int,
    sequences: int,
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    cuda_count: int | None,
) -> list[str]:
    value = _require_exact_keys(case, CASE_KEYS, f"case {case_name}")
    _require(value["name"] == case_name, f"case order/name mismatch for {case_name}")
    mixed_precision = case_name != "normal_fp32"
    overflow = case_name == "overflow_amp"
    _require(
        value["config"]
        == {
            "mixed_precision": mixed_precision,
            "mini_epochs": 2,
            "use_others_experience": "none",
        },
        f"case {case_name} config mismatch",
    )
    _require(value["owners"] == OWNERS, f"case {case_name} owners mismatch")
    input_state = _require_exact_keys(value["input"], INPUT_KEYS, f"case {case_name}.input")
    output_state = _require_exact_keys(value["output"], OUTPUT_KEYS, f"case {case_name}.output")
    _require(value["restore"] == {"patches": True, "hooks": True}, f"case {case_name} restore")

    batch = _require_exact_keys(input_state["batch"], {"kind", "items"}, f"{case_name}.batch")
    _require(batch["kind"] == "dict", f"{case_name}.batch must be a dict tree")
    batch_items = _require_exact_keys(
        batch["items"],
        {"batch_dict", "played_frames", "step_time", "ps_extras"},
        f"{case_name}.batch.items",
    )
    _require(
        batch_items["played_frames"] == {"kind": "scalar", "value": 48},
        f"{case_name} played_frames mismatch",
    )
    _require(
        batch_items["step_time"] == {"kind": "scalar", "value": 0.0},
        f"{case_name} step_time mismatch",
    )
    batch_dict_items = _require_snapshot_dict_items(
        batch_items["batch_dict"], f"{case_name}.batch.batch_dict"
    )
    _require(
        set(batch_dict_items)
        == {
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
        },
        f"{case_name} Code #3 batch field inventory mismatch",
    )
    for field in (
        "actions",
        "dones",
        "mus",
        "neglogpacs",
        "obses",
        "off_policy_mask",
        "returns",
        "sigmas",
        "states",
        "values",
    ):
        _require_row_array(
            batch_dict_items[field],
            label=f"{case_name}.batch.batch_dict.{field}",
            rows=rows,
            metadata=metadata,
            arrays=arrays,
        )
    _require_rnn_state_tree(
        batch_dict_items["rnn_states"],
        label=f"{case_name}.batch.batch_dict.rnn_states",
        sequences=sequences,
        metadata=metadata,
        arrays=arrays,
    )
    ps_extra_items = _require_snapshot_dict_items(
        batch_items["ps_extras"], f"{case_name}.batch.ps_extras"
    )
    _require(
        set(ps_extra_items) == {"mb_intr_rewards", "rewards"},
        f"{case_name} play-step extras inventory mismatch",
    )
    _require(
        ps_extra_items["mb_intr_rewards"] == {"kind": "none"},
        f"{case_name}.batch.ps_extras.mb_intr_rewards must be none",
    )
    _require_row_array(
        ps_extra_items["rewards"],
        label=f"{case_name}.batch.ps_extras.rewards",
        rows=rows,
        metadata=metadata,
        arrays=arrays,
    )

    input_models = _require_exact_keys(
        input_state["model"], {"actor", "central"}, f"{case_name}.input.model"
    )
    output_models = _require_exact_keys(
        output_state["model"], {"actor", "central"}, f"{case_name}.output.model"
    )
    input_model_names: dict[str, set[str]] = {}
    output_model_names: dict[str, set[str]] = {}
    for owner in ("actor", "central"):
        input_model_names[owner] = _validate_parameter_snapshot(
            input_models[owner], f"{case_name}.input.model.{owner}"
        )
        output_model_names[owner] = _validate_parameter_snapshot(
            output_models[owner], f"{case_name}.output.model.{owner}"
        )
        _require(
            input_model_names[owner] == output_model_names[owner],
            f"{case_name} {owner} parameter inventory changed",
        )
        for parameter_name in input_model_names[owner]:
            before = input_models[owner]["parameters"][parameter_name]
            after = output_models[owner]["parameters"][parameter_name]
            _require(
                before["shape"] == after["shape"] and before["dtype"] == after["dtype"],
                f"{case_name} {owner} parameter metadata changed",
            )

    input_optimizers = _require_exact_keys(
        input_state["optimizer"], {"actor", "central"}, f"{case_name}.input.optimizer"
    )
    output_optimizers = _require_exact_keys(
        output_state["optimizer"], {"actor", "central"}, f"{case_name}.output.optimizer"
    )
    optimizer_steps: dict[str, int] = {}
    for owner in ("actor", "central"):
        _validate_optimizer_snapshot(
            input_optimizers[owner],
            input_model_names[owner],
            f"{case_name}.input.optimizer.{owner}",
        )
        optimizer_steps[owner] = _validate_optimizer_snapshot(
            output_optimizers[owner],
            output_model_names[owner],
            f"{case_name}.output.optimizer.{owner}",
        )

    _validate_scaler(
        input_state["scaler"], enabled=mixed_precision, label=f"{case_name}.input.scaler"
    )
    _validate_scaler(
        output_state["scaler"], enabled=mixed_precision, label=f"{case_name}.output.scaler"
    )
    _validate_rms(
        input_state["rms"],
        label=f"{case_name}.input.rms",
        expected_domain="exact",
        metadata=metadata,
    )
    _validate_rms(
        output_state["rms"],
        label=f"{case_name}.output.rms",
        expected_domain="numeric",
        metadata=metadata,
    )
    _validate_lr(input_state["lr"], f"{case_name}.input.lr")
    _validate_lr(output_state["lr"], f"{case_name}.output.lr")
    _validate_rng(
        input_state["rng"],
        label=f"{case_name}.input.rng",
        metadata=metadata,
        cuda_count=cuda_count,
    )
    _validate_rng(
        output_state["rng"],
        label=f"{case_name}.output.rng",
        metadata=metadata,
        cuda_count=cuda_count,
    )

    prepared = _require_exact_keys(
        output_state["prepared"], {"actor", "central"}, f"{case_name}.output.prepared"
    )
    prepared_fields = {
        "actor": {
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
        },
        "central": {
            "actions",
            "advantages",
            "dones",
            "obs",
            "old_values",
            "returns",
            "rnn_masks",
        },
    }
    for owner in ("actor", "central"):
        prepared_items = _require_snapshot_dict_items(
            prepared[owner], f"{case_name}.output.prepared.{owner}"
        )
        _require(
            set(prepared_items) == prepared_fields[owner],
            f"{case_name} prepared {owner} field inventory mismatch",
        )
        _require(
            prepared_items["rnn_masks"] == {"kind": "none"},
            f"{case_name}.output.prepared.{owner}.rnn_masks must be none",
        )
        ordinary_fields = prepared_fields[owner] - {"rnn_masks", "rnn_states"}
        for field in sorted(ordinary_fields):
            _require_row_array(
                prepared_items[field],
                label=f"{case_name}.output.prepared.{owner}.{field}",
                rows=rows,
                metadata=metadata,
                arrays=arrays,
            )
        if owner == "actor":
            _require_rnn_state_tree(
                prepared_items["rnn_states"],
                label=f"{case_name}.output.prepared.actor.rnn_states",
                sequences=sequences,
                metadata=metadata,
                arrays=arrays,
            )
        references = _array_references(prepared[owner], label=f"{case_name}.prepared.{owner}")
        _require(bool(references), f"{case_name} prepared {owner} is empty")
        _require(
            all(
                isinstance(metadata.get(name), Mapping)
                and metadata[name].get("comparison")
                == ("numeric" if np.issubdtype(arrays[name].dtype, np.floating) else "exact")
                for name in references
            ),
            f"{case_name} prepared {owner} has wrong comparison domain",
        )

    execution = _require_exact_keys(value["execution"], EXECUTION_KEYS, f"{case_name}.execution")
    attempts = 1 if overflow else 8
    expected_order = ["prepare", "actor"] if overflow else ["prepare", "central", "actor"]
    _require(
        execution["identity_shuffle_calls"] == (0 if overflow else 1),
        f"{case_name} identity shuffle count mismatch",
    )
    _require(execution["owner_call_order"] == expected_order, f"{case_name} owner call order")
    _require(execution["actor_update_attempts"] == attempts, f"{case_name} actor attempts")
    actor_steps = execution["actor_optimizer_steps"]
    actor_skips = execution["actor_scaler_skips"]
    _require(
        isinstance(actor_steps, int)
        and isinstance(actor_skips, int)
        and actor_steps >= 0
        and actor_skips >= 0
        and actor_steps + actor_skips == attempts,
        f"{case_name} actor step/skip arithmetic mismatch",
    )
    _require(
        optimizer_steps["actor"] == actor_steps,
        f"{case_name} actor optimizer step mismatch",
    )
    expected_central_steps = 0 if overflow else 8
    _require(
        execution["central_optimizer_steps"] == expected_central_steps
        and optimizer_steps["central"] == expected_central_steps,
        f"{case_name} central optimizer step mismatch",
    )
    _require(
        execution["set_train_info_calls"]
        == ([] if overflow else [{"frame": 48, "owner_is_agent": True}]),
        f"{case_name} training env ABI call mismatch",
    )
    _require(
        execution["autocast"]
        == {
            "enabled": mixed_precision,
            "dtype": "torch.float16" if mixed_precision else None,
        },
        f"{case_name} autocast mismatch",
    )
    _require(
        execution["overflow_mutation"] == ("advantages[0]=+inf" if overflow else None),
        f"{case_name} overflow mutation mismatch",
    )
    lengths = _validate_native_return(execution["native_return"], metadata=metadata, arrays=arrays)
    expected_lengths = (
        {
            "a_losses": 1,
            "c_losses": 1,
            "b_losses": 1,
            "entropies": 1,
            "kls": 1,
        }
        if overflow
        else {
            "a_losses": 8,
            "c_losses": 8,
            "b_losses": 8,
            "entropies": 8,
            "kls": 2,
        }
    )
    _require(
        lengths == expected_lengths,
        f"{case_name} native return length mismatch",
    )

    if case_name == "normal_fp32":
        _require(actor_steps == 8 and actor_skips == 0, "normal_fp32 step/skip mismatch")
    if overflow:
        _require(
            input_models["actor"] == output_models["actor"],
            "overflow actor model changed",
        )
        _require(
            input_optimizers["actor"] == output_optimizers["actor"],
            "overflow actor optimizer changed",
        )
        before_scale = input_state["scaler"]["state_dict"]["scale"]
        after_scale = output_state["scaler"]["state_dict"]["scale"]
        _require(after_scale < before_scale, "overflow scaler did not back off")

    references = _array_references(value, label=f"case {case_name}")
    _require(len(references) == len(set(references)), f"case {case_name} reuses arrays")
    _require(
        all(name.startswith(f"{case_name}__") for name in references),
        f"case {case_name} owns an array from another case",
    )
    input_references = set(_array_references(input_state, label=f"{case_name}.input"))
    for name in input_references:
        _require(metadata[name]["comparison"] == "exact", f"{case_name} input must be exact")
    for name in _array_references(output_state, label=f"{case_name}.output"):
        domain = "numeric" if np.issubdtype(arrays[name].dtype, np.floating) else "exact"
        _require(metadata[name]["comparison"] == domain, f"{case_name} output domain mismatch")
    return references


def validate_capture(capture: object) -> None:
    value = _require_exact_keys(capture, {"manifest", "arrays"}, "capture")
    manifest = _require_exact_keys(value["manifest"], MANIFEST_KEYS, "manifest")
    arrays = value["arrays"]
    _require(isinstance(arrays, Mapping), "capture arrays must be a mapping")
    canonical_payload(manifest)
    _require(manifest["schema_version"] == 2, "schema version must be 2")
    _require(manifest["generation_mode"] == "source-only", "generation mode mismatch")
    _require(
        manifest["ordinary_pytest_regenerates"] is False,
        "ordinary pytest must not regenerate fixtures",
    )
    provenance = manifest["provenance"]
    _require(isinstance(provenance, Mapping), "provenance must be a mapping")
    _require(provenance.get("source_head") == SOURCE_HEAD, "Source HEAD mismatch")
    _require(
        provenance.get("source_rl_games_tree") == SOURCE_RL_GAMES_TREE,
        "Source RL-Games tree mismatch",
    )
    _require(provenance.get("owners") == SOURCE_OWNERS, "Source owner records mismatch")
    _require(
        provenance.get("native_owner_paths") == list(OWNERS.values()),
        "native owner paths mismatch",
    )
    _require(manifest["code3_anchors"] == CODE3_ANCHORS, "Code #3 anchors mismatch")
    if manifest["platform"] == CANONICAL_PLATFORM:
        loaded_modules = provenance.get("loaded_rl_games_modules")
        _require(
            isinstance(loaded_modules, Mapping) and bool(loaded_modules),
            "loaded rl_games module inventory is missing",
        )
        loaded_paths: set[str] = set()
        for module_name, record in loaded_modules.items():
            _require(
                isinstance(module_name, str)
                and (module_name == "rl_games" or module_name.startswith("rl_games.")),
                "loaded rl_games module name is invalid",
            )
            module_record = _require_exact_keys(
                record, {"path", "sha256"}, f"loaded module {module_name}"
            )
            path = module_record["path"]
            _require(
                isinstance(path, str)
                and bool(path)
                and not Path(path).is_absolute()
                and ".." not in Path(path).parts
                and path not in loaded_paths,
                f"loaded module path is invalid: {module_name}",
            )
            loaded_paths.add(path)
            _require_sha256(module_record["sha256"], f"loaded module {module_name}")
        _require(
            set(OWNERS.values())
            <= {
                f"{module_name}.{owner.rsplit('.', 1)[-1]}"
                for module_name, owner in (
                    ("rl_games.torch_runner", OWNERS["runner"]),
                    ("rl_games.algos_torch.a2c_continuous", OWNERS["agent"]),
                    ("rl_games.common.datasets", OWNERS["actor_dataset"]),
                    ("rl_games.algos_torch.central_value", OWNERS["central_value"]),
                )
                if module_name in loaded_modules
            },
            "native owner modules are missing from the loaded namespace inventory",
        )

    contract = _require_exact_keys(
        manifest["capture_contract"],
        {
            "case_names",
            "rows",
            "sequences",
            "actor_dataset_batch_sizes",
            "central_dataset_batch_sizes",
            "rms_roles",
            "rms_alias",
        },
        "capture_contract",
    )
    _require(contract["case_names"] == list(CASE_NAMES), "capture case names mismatch")
    _require(contract["rows"] == 56, "capture row count mismatch")
    _require(contract["sequences"] == 14, "capture sequence count mismatch")
    rows = contract["rows"]
    sequences = contract["sequences"]
    _require(
        rows % sequences == 0 and rows // sequences == 4,
        "capture row/sequence relationship mismatch",
    )
    expected_batches = [12, 12, 12, 20]
    _require(
        contract["actor_dataset_batch_sizes"] == expected_batches
        and contract["central_dataset_batch_sizes"] == expected_batches,
        "native dataset batch sizes mismatch",
    )
    _require(
        sum(contract["actor_dataset_batch_sizes"]) == rows
        and sum(contract["central_dataset_batch_sizes"]) == rows,
        "native dataset batch sizes do not cover capture rows",
    )
    _require(contract["rms_roles"] == list(RMS_ROLES), "capture RMS roles mismatch")
    _require(
        contract["rms_alias"] == {"four_roles_distinct": True, "active_is_central_value": True},
        "capture RMS alias contract mismatch",
    )

    metadata = manifest["npz_arrays"]
    _require(isinstance(metadata, Mapping), "npz array metadata must be a mapping")
    _validate_array_inventory(arrays, metadata)
    exact_inventory = manifest["exact_comparison_inventory"]
    numeric_inventory = manifest["numeric_comparison_inventory"]
    _require(
        isinstance(exact_inventory, list)
        and isinstance(numeric_inventory, list)
        and bool(exact_inventory)
        and bool(numeric_inventory)
        and all(isinstance(name, str) for name in exact_inventory + numeric_inventory),
        "comparison inventories must be non-empty string lists",
    )
    _require(
        len(exact_inventory) == len(set(exact_inventory))
        and len(numeric_inventory) == len(set(numeric_inventory)),
        "comparison inventory contains duplicates",
    )
    exact = set(exact_inventory)
    numeric = set(numeric_inventory)
    _require(exact.isdisjoint(numeric), "comparison inventories overlap")
    _require(exact | numeric == set(arrays), "comparison inventory is incomplete")
    _require(
        exact == {name for name, record in metadata.items() if record["comparison"] == "exact"}
        and numeric
        == {name for name, record in metadata.items() if record["comparison"] == "numeric"},
        "comparison inventory disagrees with metadata",
    )

    cases = manifest["cases"]
    _require(isinstance(cases, list) and len(cases) == 3, "capture must contain three cases")
    _require(
        [case.get("name") if isinstance(case, Mapping) else None for case in cases]
        == list(CASE_NAMES),
        "capture cases are not exactly ordered",
    )
    cuda_count = None
    platform = manifest["platform"]
    if platform == CANONICAL_PLATFORM:
        cuda_count = 1
    elif isinstance(platform, Mapping) and isinstance(platform.get("cuda_device_count"), int):
        cuda_count = platform["cuda_device_count"]
    all_references: list[str] = []
    for case_name, case in zip(CASE_NAMES, cases, strict=True):
        all_references.extend(
            _validate_case(
                case,
                case_name=case_name,
                rows=rows,
                sequences=sequences,
                metadata=metadata,
                arrays=arrays,
                cuda_count=cuda_count,
            )
        )
    _require(
        len(all_references) == len(set(all_references)),
        "snapshot arrays are shared between cases",
    )
    _require(set(all_references) == set(arrays), "snapshot array references are incomplete")


def _comparison_inventory(
    manifest: Mapping[str, object], arrays: Mapping[str, np.ndarray], label: str
) -> tuple[list[str], list[str], Mapping[str, object]]:
    metadata = manifest.get("npz_arrays")
    _require(isinstance(metadata, Mapping), f"{label} NPZ metadata is invalid")
    _require(set(arrays) == set(metadata), f"{label} NPZ name inventory mismatch")
    for name, array in arrays.items():
        _require(isinstance(array, np.ndarray), f"{label} array type drift for {name}")
        record = metadata[name]
        _require(
            isinstance(record, Mapping)
            and set(record) == {"name", "shape", "dtype", "sha256", "comparison"}
            and record["name"] == name
            and record["comparison"] in {"exact", "numeric"},
            f"{label} metadata is invalid for {name}",
        )
        _require(
            _array_sha256(array) == record["sha256"],
            f"{label} array content/hash drift for {name}",
        )
    exact = manifest.get("exact_comparison_inventory")
    numeric = manifest.get("numeric_comparison_inventory")
    _require(
        isinstance(exact, list)
        and isinstance(numeric, list)
        and bool(exact)
        and bool(numeric)
        and all(isinstance(name, str) for name in exact + numeric),
        f"{label} comparison inventories are empty or invalid",
    )
    _require(
        len(exact) == len(set(exact))
        and len(numeric) == len(set(numeric))
        and set(exact).isdisjoint(numeric),
        f"{label} comparison inventories overlap or contain duplicates",
    )
    _require(
        set(exact) | set(numeric) == set(arrays),
        f"{label} comparison inventory is incomplete",
    )
    _require(
        set(exact)
        == {name for name, record in metadata.items() if record.get("comparison") == "exact"}
        and set(numeric)
        == {name for name, record in metadata.items() if record.get("comparison") == "numeric"},
        f"{label} comparison inventory disagrees with metadata",
    )
    for name in numeric:
        _require(
            np.issubdtype(arrays[name].dtype, np.floating),
            f"{label} numeric array is not floating: {name}",
        )
    return exact, numeric, metadata


def _assert_json_equal(source: object, target: object, label: str) -> None:
    if canonical_payload(source) != canonical_payload(target):
        raise AssertionError(f"JSON state mismatch: {label}")


def compare_capture(source: UpdateFixture, target: dict[str, object]) -> ReplayResult:
    source_capture = {"manifest": source.manifest, "arrays": source.arrays}
    validate_capture(source_capture)
    validate_capture(target)

    source_manifest = source.manifest
    target_manifest = target.get("manifest")
    target_arrays = target.get("arrays")
    _require(isinstance(target_manifest, Mapping), "Target manifest is invalid")
    _require(isinstance(target_arrays, Mapping), "Target arrays are invalid")
    source_exact, source_numeric, source_metadata = _comparison_inventory(
        source_manifest, source.arrays, "Source"
    )
    target_exact, target_numeric, target_metadata = _comparison_inventory(
        target_manifest, target_arrays, "Target"
    )
    _require(source_exact == target_exact, "exact comparison inventory mismatch")
    _require(source_numeric == target_numeric, "numeric comparison inventory mismatch")

    for key in (
        "schema_version",
        "generation_mode",
        "ordinary_pytest_regenerates",
        "canonical_platform",
        "code3_anchors",
        "runner_params",
        "capture_contract",
        "tolerances",
    ):
        _assert_json_equal(source_manifest.get(key), target_manifest.get(key), f"manifest.{key}")
    _assert_json_equal(source_manifest.get("cases"), target_manifest.get("cases"), "manifest.cases")

    for name in source_exact:
        _assert_json_equal(source_metadata[name], target_metadata[name], f"metadata.{name}")
        np.testing.assert_array_equal(
            source.arrays[name], target_arrays[name], err_msg=f"exact array mismatch: {name}"
        )

    for name in source_numeric:
        source_record = source_metadata[name]
        target_record = target_metadata[name]
        _assert_json_equal(
            {key: source_record[key] for key in ("name", "shape", "dtype", "comparison")},
            {key: target_record[key] for key in ("name", "shape", "dtype", "comparison")},
            f"numeric metadata.{name}",
        )

    tolerances = source_manifest.get("tolerances")
    _require(isinstance(tolerances, Mapping), "Source tolerances are invalid")
    atol = tolerances.get("atol")
    rtol = tolerances.get("rtol")
    _require(
        isinstance(atol, (int, float))
        and isinstance(rtol, (int, float))
        and math.isfinite(float(atol))
        and math.isfinite(float(rtol))
        and atol >= 0
        and rtol >= 0,
        "Source tolerances are invalid",
    )

    max_abs_error = 0.0
    for name in source_numeric:
        source_array = source.arrays[name]
        target_array = target_arrays[name]
        source_nan = np.isnan(source_array)
        target_nan = np.isnan(target_array)
        source_posinf = np.isposinf(source_array)
        target_posinf = np.isposinf(target_array)
        source_neginf = np.isneginf(source_array)
        target_neginf = np.isneginf(target_array)
        if not (
            np.array_equal(source_nan, target_nan)
            and np.array_equal(source_posinf, target_posinf)
            and np.array_equal(source_neginf, target_neginf)
        ):
            raise AssertionError(f"non-finite mask mismatch: {name}")
        finite = np.isfinite(source_array) & np.isfinite(target_array)
        if not np.any(finite):
            continue
        source_finite = source_array[finite]
        target_finite = target_array[finite]
        np.testing.assert_allclose(
            source_finite,
            target_finite,
            atol=float(atol),
            rtol=float(rtol),
            err_msg=f"numeric array mismatch: {name}",
        )
        pair_error = float(
            np.max(np.abs(source_finite.astype(np.float64) - target_finite.astype(np.float64)))
        )
        _require(math.isfinite(pair_error), f"non-finite diagnostic error for {name}")
        max_abs_error = max(max_abs_error, pair_error)

    cases = source_manifest.get("cases")
    _require(isinstance(cases, list), "Source cases are invalid")
    case_names = tuple(case["name"] for case in cases)
    provenance = source_manifest.get("provenance")
    _require(isinstance(provenance, Mapping), "Source provenance is invalid")
    native_owner_paths = provenance.get("native_owner_paths")
    _require(
        isinstance(native_owner_paths, list)
        and all(isinstance(path, str) for path in native_owner_paths),
        "Source native owner paths are invalid",
    )
    return ReplayResult(
        case_names=case_names,
        native_owner_paths=tuple(native_owner_paths),
        exact_array_count=len(source_exact),
        numeric_array_count=len(source_numeric),
        max_abs_error=max_abs_error,
    )


def assert_loaded_namespace(expected_package_root: Path) -> dict[str, dict[str, str]]:
    import rl_games

    expected = Path(expected_package_root).resolve(strict=True)
    package_file = Path(rl_games.__file__).resolve(strict=True)
    actual = package_file.parent
    if actual != expected:
        raise RuntimeError(
            f"rl_games must load from the exact package root {expected}; loaded {actual}"
        )

    inventory: dict[str, dict[str, str]] = {}
    for module_name, module in sorted(sys.modules.items()):
        if module_name != "rl_games" and not module_name.startswith("rl_games."):
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"loaded rl_games module has no file: {module_name}")
        path = Path(module_file).resolve(strict=True)
        if not path.is_relative_to(expected):
            raise RuntimeError(
                f"loaded rl_games module is outside the exact package root: {module_name}={path}"
            )
        inventory[module_name] = {
            "path": path.relative_to(expected).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    _require("rl_games" in inventory, "rl_games namespace inventory is empty")
    return inventory


def _capture_native_case(
    case_name: str,
    runner_params: dict[str, object],
    store: SnapshotStore,
) -> tuple[dict[str, object], list[int], list[int]]:
    from rl_games.common import a2c_common

    mixed_precision = case_name != "normal_fp32"
    overflow = case_name == "overflow_amp"
    params = copy.deepcopy(runner_params)
    params["config"]["mixed_precision"] = mixed_precision
    params["config"]["use_others_experience"] = "none"
    params["config"]["mini_epochs"] = 2
    configure_canonical_execution()

    with (
        tempfile.TemporaryDirectory(prefix=f"unilab-sapg-update-{case_name}-") as train_dir,
        _writer_owned_agent_and_env(params, Path(train_dir)) as (agent, env),
    ):
        agent.frame = 48
        frozen_batch = _load_code3_batch(torch.device(agent.ppo_device))
        ps_extras = {
            "mb_intr_rewards": None,
            "rewards": torch.zeros((56, 1), dtype=torch.float32, device=agent.ppo_device),
        }

        rng_before_fill = _raw_rng_state()
        fill_parameters(agent.model, "actor")
        fill_parameters(agent.central_value_net.model, "central")
        rng_after_fill = _raw_rng_state()
        _require(
            _same_raw_rng_state(rng_before_fill, rng_after_fill),
            f"{case_name} deterministic parameter fill consumed RNG",
        )

        input_state = {
            "batch": store.tree(
                f"{case_name}__input__batch",
                _declared_batch_pair(frozen_batch, ps_extras),
                comparison="exact",
            ),
            "model": _model_snapshots(agent),
            "optimizer": _optimizer_snapshots(agent),
            "scaler": snapshot_scaler(agent.scaler),
            "rms": snapshot_rms(store, _rms_roles(agent), f"{case_name}__input__rms"),
            "lr": snapshot_lr(agent),
            "rng": snapshot_rng(store, f"{case_name}__input__rng"),
        }

        owner_call_order: list[str] = []
        identity_shuffle_calls: list[dict[str, object]] = []
        set_train_info_calls: list[dict[str, object]] = []
        actor_autocast: list[dict[str, object]] = []
        central_returns: list[float] = []
        prepared: dict[str, object] = {}
        actor_batch_sizes: list[int] = []
        central_batch_sizes: list[int] = []
        played_batches: list[dict[str, object]] = []
        patches = _Patches()
        hook = None
        hooks_restored = False

        original_prepare = agent.prepare_dataset
        original_central = agent.train_central_value

        def observe_prepare(batch_dict, *args, **kwargs):
            _require(not prepared, f"{case_name} prepare_dataset called more than once")
            result = original_prepare(batch_dict, *args, **kwargs)
            owner_call_order.append("prepare")
            prepared["actor"] = store.tree(
                f"{case_name}__output__prepared__actor",
                agent.dataset.values_dict,
                comparison="numeric",
            )
            prepared["central"] = store.tree(
                f"{case_name}__output__prepared__central",
                agent.central_value_net.dataset.values_dict,
                comparison="numeric",
            )
            actor_batch_sizes.extend(_dataset_batch_sizes(agent.dataset))
            central_batch_sizes.extend(_dataset_batch_sizes(agent.central_value_net.dataset))
            return result

        def observe_central():
            owner_call_order.append("central")
            result = original_central()
            central_returns.append(float(result))
            return result

        def actor_pre_hook(_module, _arguments):
            enabled = bool(torch.is_autocast_enabled("cuda"))
            observation = {
                "enabled": enabled,
                "dtype": str(torch.get_autocast_dtype("cuda")) if enabled else None,
            }
            actor_autocast.append(observation)
            if "actor" not in owner_call_order:
                owner_call_order.append("actor")

        def set_train_info(frame, owner):
            _require(
                frame == agent.frame and owner is agent,
                f"{case_name} synthetic training-env ABI call drift",
            )
            set_train_info_calls.append({"frame": int(frame), "owner_is_agent": True})

        def play_steps():
            batch = _clone_tree(frozen_batch)
            extras = _clone_tree(ps_extras)
            _require(isinstance(batch, dict), "cloned Code #3 batch is not a mapping")
            _require(
                not _tree_difference_paths(
                    _declared_batch_pair(frozen_batch, ps_extras),
                    _declared_batch_pair(batch, extras),
                ),
                f"{case_name} native play input drift",
            )
            played_batches.append(batch)
            return batch, extras

        def identity_shuffle(batch_dict, seq_length):
            before = _raw_rng_state()
            _require(seq_length == 4, f"{case_name} identity shuffle sequence drift")
            _require(
                len(played_batches) == 1 and batch_dict is played_batches[0],
                f"{case_name} identity shuffle did not receive the native play batch",
            )
            after = _raw_rng_state()
            _require(
                _same_raw_rng_state(before, after),
                f"{case_name} identity shuffle consumed RNG",
            )
            identity_shuffle_calls.append({"same_object": True, "seq_length": seq_length})
            return batch_dict

        try:
            hook = agent.model.register_forward_pre_hook(actor_pre_hook)
            with patches:
                patches.set(agent, "prepare_dataset", observe_prepare)
                patches.set(agent, "train_central_value", observe_central)
                if overflow:
                    update_batch = _clone_tree(frozen_batch)
                    _require(isinstance(update_batch, dict), "overflow batch clone drift")
                    update_batch.pop("played_frames")
                    agent.curr_frames = 48
                    agent.prepare_dataset(update_batch)
                    native_minibatch = _clone_tree(agent.dataset[0])
                    mutation_before = _clone_tree(native_minibatch)
                    _require(
                        isinstance(native_minibatch, dict)
                        and isinstance(native_minibatch.get("advantages"), torch.Tensor),
                        "overflow native actor minibatch lacks advantages",
                    )
                    native_minibatch["advantages"].reshape(-1)[0] = float("inf")
                    differences = _tree_difference_paths(mutation_before, native_minibatch)
                    _require(
                        differences == ["advantages[0]"],
                        f"overflow mutation tree drift: {differences}",
                    )
                    native_actor_return = agent.train_actor_critic(native_minibatch)
                    (
                        a_loss,
                        c_loss,
                        entropy,
                        kl,
                        last_lr,
                        lr_mul,
                        _mu,
                        _sigma,
                        b_loss,
                        _extras,
                    ) = native_actor_return
                    algorithm_return = {
                        "a_losses": [a_loss],
                        "c_losses": [c_loss],
                        "b_losses": [b_loss],
                        "entropies": [entropy],
                        "kls": [kl],
                        "last_lr": last_lr,
                        "lr_mul": lr_mul,
                        "excluded_wall_clock_fields": [
                            "play_time",
                            "update_time",
                            "total_time",
                        ],
                    }
                else:
                    patches.set_raw(a2c_common, "shuffle_batch", identity_shuffle)
                    patches.set(agent, "play_steps", play_steps)
                    patches.set(env, "set_train_info", set_train_info)
                    native_return = agent.train_epoch()
                    _require(
                        isinstance(native_return, tuple) and len(native_return) == 12,
                        f"{case_name} native train_epoch return layout drift",
                    )
                    (
                        step_time,
                        _play_time,
                        _update_time,
                        _total_time,
                        a_losses,
                        c_losses,
                        b_losses,
                        entropies,
                        kls,
                        last_lr,
                        lr_mul,
                        _extra_infos,
                    ) = native_return
                    _require(step_time == 0.0, f"{case_name} native step_time drift")
                    algorithm_return = {
                        "a_losses": a_losses,
                        "c_losses": c_losses,
                        "b_losses": b_losses,
                        "entropies": entropies,
                        "kls": kls,
                        "last_lr": last_lr,
                        "lr_mul": lr_mul,
                        "excluded_wall_clock_fields": [
                            "play_time",
                            "update_time",
                            "total_time",
                        ],
                    }

                native_return_snapshot = store.tree(
                    f"{case_name}__execution__native_return",
                    algorithm_return,
                    comparison="numeric",
                )
                output_models = _model_snapshots(agent)
                output_optimizers = _optimizer_snapshots(agent)
                output_state = {
                    "prepared": prepared,
                    "model": output_models,
                    "optimizer": output_optimizers,
                    "scaler": snapshot_scaler(agent.scaler),
                    "rms": snapshot_rms(store, _rms_roles(agent), f"{case_name}__output__rms"),
                    "lr": snapshot_lr(agent),
                    "rng": snapshot_rng(store, f"{case_name}__output__rng"),
                }
        finally:
            if hook is not None:
                hook.remove()
                hooks_restored = True

        attempts = len(algorithm_return["a_losses"])
        actor_steps = _validate_optimizer_snapshot(
            output_optimizers["actor"],
            set(output_models["actor"]["parameters"]),
            f"{case_name}.captured.actor_optimizer",
        )
        central_steps = _validate_optimizer_snapshot(
            output_optimizers["central"],
            set(output_models["central"]["parameters"]),
            f"{case_name}.captured.central_optimizer",
        )
        actor_skips = attempts - actor_steps
        _require(actor_skips >= 0, f"{case_name} optimizer steps exceed attempts")
        _require(
            len(actor_autocast) == attempts and len(set(map(str, actor_autocast))) == 1,
            f"{case_name} actor autocast observation drift",
        )
        if overflow:
            _require(not central_returns, "overflow unexpectedly trained central value")
        else:
            _require(
                len(central_returns) == 1 and math.isfinite(central_returns[0]),
                f"{case_name} central-value native return drift",
            )
            _require(
                len(played_batches) == 1,
                f"{case_name} native play call-count drift",
            )

        case = {
            "name": case_name,
            "config": {
                "mixed_precision": mixed_precision,
                "mini_epochs": 2,
                "use_others_experience": "none",
            },
            "owners": dict(OWNERS),
            "input": input_state,
            "execution": {
                "identity_shuffle_calls": len(identity_shuffle_calls),
                "owner_call_order": owner_call_order,
                "actor_update_attempts": attempts,
                "actor_optimizer_steps": actor_steps,
                "actor_scaler_skips": actor_skips,
                "central_optimizer_steps": central_steps,
                "native_return": native_return_snapshot,
                "overflow_mutation": "advantages[0]=+inf" if overflow else None,
                "set_train_info_calls": set_train_info_calls,
                "autocast": actor_autocast[0],
            },
            "output": output_state,
            "restore": {"patches": patches.restored, "hooks": hooks_restored},
        }
        return case, actor_batch_sizes, central_batch_sizes


def capture_update(
    runner_params: dict[str, object], expected_package_root: Path
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("canonical SAPG update capture requires CUDA")
    _require(isinstance(runner_params, dict), "runner params must be a mapping")
    canonical_payload(runner_params)
    configure_canonical_execution()
    assert_loaded_namespace(expected_package_root)

    store = SnapshotStore()
    cases: list[dict[str, object]] = []
    actor_batch_sizes: list[int] | None = None
    central_batch_sizes: list[int] | None = None
    for case_name in CASE_NAMES:
        case, observed_actor_sizes, observed_central_sizes = _capture_native_case(
            case_name, runner_params, store
        )
        if actor_batch_sizes is None:
            actor_batch_sizes = observed_actor_sizes
            central_batch_sizes = observed_central_sizes
        else:
            _require(
                observed_actor_sizes == actor_batch_sizes
                and observed_central_sizes == central_batch_sizes,
                f"{case_name} native dataset partition drift",
            )
        cases.append(case)

    platform_data = execution_platform()
    _require(
        platform_data == CANONICAL_PLATFORM,
        f"canonical SAPG update platform mismatch: {platform_data}",
    )
    loaded_modules = assert_loaded_namespace(expected_package_root)
    exact_inventory = [
        name for name, metadata in store.metadata.items() if metadata["comparison"] == "exact"
    ]
    numeric_inventory = [
        name for name, metadata in store.metadata.items() if metadata["comparison"] == "numeric"
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generation_mode": "source-only",
        "ordinary_pytest_regenerates": False,
        "provenance": {
            "source_head": SOURCE_HEAD,
            "source_rl_games_tree": SOURCE_RL_GAMES_TREE,
            "owners": copy.deepcopy(SOURCE_OWNERS),
            "native_owner_paths": list(OWNERS.values()),
            "loaded_rl_games_modules": loaded_modules,
        },
        "platform": platform_data,
        "canonical_platform": copy.deepcopy(CANONICAL_PLATFORM),
        "code3_anchors": dict(CODE3_ANCHORS),
        "runner_params": copy.deepcopy(runner_params),
        "capture_contract": {
            "case_names": list(CASE_NAMES),
            "rows": 56,
            "sequences": 14,
            "actor_dataset_batch_sizes": actor_batch_sizes,
            "central_dataset_batch_sizes": central_batch_sizes,
            "rms_roles": list(RMS_ROLES),
            "rms_alias": {
                "four_roles_distinct": True,
                "active_is_central_value": True,
            },
        },
        "cases": cases,
        "npz_arrays": store.metadata,
        "exact_comparison_inventory": exact_inventory,
        "numeric_comparison_inventory": numeric_inventory,
        "tolerances": {"atol": 1e-6, "rtol": 1e-5},
        "fixture_files": [FIXTURE_NPZ.name, FIXTURE_MANIFEST.name],
        "canonical_payload_sha256": "0" * 64,
        "generation_command": (
            "UNILAB_SAPG_ORACLE_MODE=source uv run --isolated --no-project "
            "--python 3.11 scripts/generate_simtoolreal_sapg_update_fixture.py"
        ),
    }
    capture = {"manifest": manifest, "arrays": store.arrays}
    validate_capture(capture)
    return capture


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


def load_update_fixture(root: Path = FIXTURE_ROOT) -> UpdateFixture:
    root = _require_real_directory(Path(root), "update fixture root")
    manifest_bytes = _read_regular_file(root / FIXTURE_MANIFEST.name, "update fixture manifest")
    npz_bytes = _read_regular_file(root / FIXTURE_NPZ.name, "update fixture NPZ")
    if hashlib.sha256(manifest_bytes).hexdigest() != EXPECTED_UPDATE_MANIFEST_SHA256:
        raise RuntimeError("update fixture manifest file anchor drift")
    if hashlib.sha256(npz_bytes).hexdigest() != EXPECTED_UPDATE_NPZ_SHA256:
        raise RuntimeError("update fixture NPZ file anchor drift")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("update fixture manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("update fixture schema drift: expected schema v2")
    payload_sha256 = hashlib.sha256(canonical_payload(manifest)).hexdigest()
    if payload_sha256 != EXPECTED_UPDATE_PAYLOAD_SHA256:
        raise RuntimeError("update fixture canonical payload anchor drift")
    if manifest.get("canonical_payload_sha256") != EXPECTED_UPDATE_PAYLOAD_SHA256:
        raise RuntimeError("update fixture declared canonical payload anchor drift")
    fixture_files = manifest.get("fixture_files")
    if isinstance(fixture_files, list):
        if fixture_files != [FIXTURE_NPZ.name, FIXTURE_MANIFEST.name]:
            raise RuntimeError("update fixture file inventory drift")
    elif isinstance(fixture_files, Mapping):
        npz_record = fixture_files.get("npz")
        if (
            not isinstance(npz_record, Mapping)
            or npz_record.get("name") != FIXTURE_NPZ.name
            or npz_record.get("sha256") != EXPECTED_UPDATE_NPZ_SHA256
        ):
            raise RuntimeError("update fixture declared NPZ anchor drift")
    else:
        raise RuntimeError("update fixture file inventory drift")
    try:
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
            arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as error:
        raise RuntimeError("update fixture NPZ payload is invalid") from error
    fixture = UpdateFixture(manifest=manifest, arrays=arrays)
    validate_capture({"manifest": fixture.manifest, "arrays": fixture.arrays})
    return fixture


def replay_update_fixture(fixture: UpdateFixture) -> ReplayResult:
    from tests.algos.rlgames_sapg import _runtime_requirement

    _runtime_requirement.require_simtoolreal_rl_games()
    assert_loaded_namespace(_runtime_requirement.VENDOR_PACKAGE_ROOT)
    target = capture_update(
        fixture.manifest["runner_params"], _runtime_requirement.VENDOR_PACKAGE_ROOT
    )
    assert_loaded_namespace(_runtime_requirement.VENDOR_PACKAGE_ROOT)
    return compare_capture(fixture, target)
