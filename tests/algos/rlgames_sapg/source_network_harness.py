from __future__ import annotations

import base64
import copy
import ctypes
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

SOURCE_HEAD = "2a9917533bfea70419ed2667a511d7238e5b3abc"
SOURCE_RL_GAMES_TREE = "7a6a0bb090998d00565aaefa6ab9f2b3d356ace2"
COEFFICIENT_IDS = (50, 40, 30, 20, 10, 0)
ATOL = 1e-6
RTOL = 1e-5
FIXTURE_SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/simtoolreal_sapg"
FIXTURE_NPZ = FIXTURE_ROOT / "source_network_fp32.npz"
FIXTURE_MANIFEST = FIXTURE_ROOT / "source_network_manifest.json"

MAPPED_TENSORS = (
    "actor_embedding",
    "central_embedding",
    "actor_lstm",
    "actor_layer_norm",
    "actor_mlp",
    "central_mlp",
    "mu",
    "conditional_sigma",
    "actor_shared_value",
    "central_value",
    "neglogp",
    "entropy",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value)


def array_sha256(value: np.ndarray | torch.Tensor) -> str:
    return _sha256(_array(value).tobytes(order="C"))


def array_metadata(value: np.ndarray | torch.Tensor) -> dict[str, Any]:
    data = _array(value)
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "sha256": array_sha256(data),
    }


def _sentinel_indices(name: str, size: int) -> list[int]:
    return [
        int.from_bytes(hashlib.sha256(f"{name}:{index}".encode()).digest()[:8], "big") % size
        for index in range(64)
    ]


def tensor_signature(name: str, value: torch.Tensor) -> dict[str, Any]:
    data = _array(value)
    flat = data.reshape(-1)
    as_float64 = flat.astype(np.float64)
    sentinels = []
    for flat_index in _sentinel_indices(name, flat.size):
        sentinels.append(
            {
                "flat_index": flat_index,
                "coordinate": [int(index) for index in np.unravel_index(flat_index, data.shape)],
                "value": float(flat[flat_index]),
            }
        )
    return {
        **array_metadata(data),
        "norm": float(np.linalg.norm(as_float64)),
        "sum": float(as_float64.sum()),
        "max": float(as_float64.max()),
        "max_abs": float(np.abs(as_float64).max()),
        "sentinels": sentinels,
    }


def parameter_signatures(module: torch.nn.Module, role: str) -> dict[str, dict[str, Any]]:
    return {
        name: tensor_signature(f"{role}:parameter:{name}", parameter)
        for name, parameter in module.named_parameters()
    }


def gradient_signatures(
    module: torch.nn.Module, role: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    signatures = {}
    absent = []
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            absent.append(name)
        else:
            signatures[name] = tensor_signature(f"{role}:gradient:{name}", parameter.grad)
    return signatures, absent


def parameter_hashes(module: torch.nn.Module) -> dict[str, str]:
    return {name: array_sha256(parameter) for name, parameter in module.named_parameters()}


def _close(actual: float, expected: float) -> bool:
    return bool(np.isclose(actual, expected, atol=ATOL, rtol=RTOL))


@dataclass(frozen=True)
class SignatureCollection:
    entries: dict[str, dict[str, Any]]
    require_hash: bool = False

    def matches(self, expected: SignatureCollection) -> bool:
        if set(self.entries) != set(expected.entries):
            return False
        for name, actual_entry in self.entries.items():
            expected_entry = expected.entries[name]
            if actual_entry["shape"] != expected_entry["shape"]:
                return False
            if actual_entry["dtype"] != expected_entry["dtype"]:
                return False
            if self.require_hash and actual_entry["sha256"] != expected_entry["sha256"]:
                return False
            for statistic in ("norm", "sum", "max", "max_abs"):
                if not _close(actual_entry[statistic], expected_entry[statistic]):
                    return False
            if len(actual_entry["sentinels"]) != 64 or len(expected_entry["sentinels"]) != 64:
                return False
            for actual, wanted in zip(
                actual_entry["sentinels"], expected_entry["sentinels"], strict=True
            ):
                if actual["flat_index"] != wanted["flat_index"]:
                    return False
                if actual["coordinate"] != wanted["coordinate"]:
                    return False
                if not _close(actual["value"], wanted["value"]):
                    return False
        return True


@dataclass(frozen=True)
class NetworkFixture:
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    tensors: dict[str, np.ndarray]
    actor_parameter_signatures: SignatureCollection
    actor_gradient_signatures: SignatureCollection
    central_parameter_signatures: SignatureCollection
    central_gradient_signatures: SignatureCollection


@dataclass(frozen=True)
class NetworkReplay:
    tensors: dict[str, np.ndarray]
    native_initialization_hashes_exact: bool
    weight_hashes_exact: bool
    input_hashes_exact: bool
    computed_hashes_exact: bool
    is_canonical_platform: bool
    actor_parameter_signatures: SignatureCollection
    actor_gradient_signatures: SignatureCollection
    central_parameter_signatures: SignatureCollection
    central_gradient_signatures: SignatureCollection
    max_abs_errors: dict[str, float]
    max_rel_errors: dict[str, float]


@dataclass(frozen=True)
class NetworkCapture:
    tensors: dict[str, np.ndarray]
    inputs: dict[str, np.ndarray]
    native_initialization_hashes: dict[str, dict[str, str]]
    deterministic_parameter_hashes: dict[str, dict[str, str]]
    actor_parameter_signatures: dict[str, dict[str, Any]]
    actor_gradient_signatures: dict[str, dict[str, Any]]
    actor_absent_gradients: list[str]
    central_parameter_signatures: dict[str, dict[str, Any]]
    central_gradient_signatures: dict[str, dict[str, Any]]
    central_absent_gradients: list[str]
    rng_states: dict[str, Any]
    platform: dict[str, Any]
    observed_contract: dict[str, Any]


def _canonical_manifest_payload(manifest: dict[str, Any]) -> bytes:
    payload = copy.deepcopy(manifest)
    payload.pop("manifest_payload_sha256", None)
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def _require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"fixture path must be a regular file: {path}")


def load_network_fixture(root: Path = FIXTURE_ROOT) -> NetworkFixture:
    manifest_path = root / FIXTURE_MANIFEST.name
    npz_path = root / FIXTURE_NPZ.name
    _require_regular_file(manifest_path)
    _require_regular_file(npz_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise RuntimeError("network fixture schema version drift")
    payload_hash = _sha256(_canonical_manifest_payload(manifest))
    if payload_hash != manifest.get("manifest_payload_sha256"):
        raise RuntimeError("network fixture manifest payload hash drift")
    if _sha256(npz_path.read_bytes()) != manifest["fixture_files"]["npz"]["sha256"]:
        raise RuntimeError("network fixture NPZ file hash drift")

    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    expected_arrays = manifest["npz_arrays"]
    if set(arrays) != set(expected_arrays):
        raise RuntimeError("network fixture NPZ inventory drift")
    for name, data in arrays.items():
        if array_metadata(data) != expected_arrays[name]:
            raise RuntimeError(f"network fixture array drift: {name}")

    tensors = {name: arrays[f"trace__{name}"] for name in MAPPED_TENSORS}
    signatures = manifest["signatures"]
    return NetworkFixture(
        manifest=manifest,
        arrays=arrays,
        tensors=tensors,
        actor_parameter_signatures=SignatureCollection(signatures["actor_parameters"]),
        actor_gradient_signatures=SignatureCollection(signatures["actor_gradients"]),
        central_parameter_signatures=SignatureCollection(signatures["central_parameters"]),
        central_gradient_signatures=SignatureCollection(signatures["central_gradients"]),
    )


def load_source_owner_contract(root: Path = FIXTURE_ROOT) -> dict[str, Any]:
    return load_network_fixture(root).manifest["owner_contract"]


def _pattern(shape: tuple[int, ...], name: str, denominator: int = 128) -> np.ndarray:
    size = int(np.prod(shape))
    offset = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big") % 257
    integers = (np.arange(size, dtype=np.int64) * 37 + offset) % 257 - 128
    return (integers.astype(np.float32) / denominator).reshape(shape)


def make_inputs(spec: dict[str, Any]) -> dict[str, np.ndarray]:
    case = spec["synthetic_case"]
    batch_size = case["batch_size"]
    sequence_count = case["sequence_count"]
    actor_obs = _pattern((batch_size, case["actor_carrier"]), "actor_obs")
    central_obs = _pattern((batch_size, case["central_carrier"]), "central_obs")
    block_ids = np.repeat(np.asarray(COEFFICIENT_IDS, dtype=np.int64), 2)
    actor_obs[:, -1] = block_ids.astype(np.float32)
    central_obs[:, -1] = block_ids.astype(np.float32)
    inputs = {
        "actor_obs": actor_obs,
        "central_obs": central_obs,
        "actions": _pattern((batch_size, case["actions"]), "actions", denominator=64),
        "block_ids": block_ids,
        "rnn_hidden": _pattern((1, sequence_count, 1024), "rnn_hidden", denominator=256),
        "rnn_cell": _pattern((1, sequence_count, 1024), "rnn_cell", denominator=256),
        "dones": np.zeros((batch_size,), dtype=np.float32),
        "cotangent_mu": _pattern((batch_size, case["actions"]), "cotangent_mu", 512),
        "cotangent_sigma": _pattern((batch_size, case["actions"]), "cotangent_sigma", 512),
        "cotangent_actor_value": _pattern((batch_size, 1), "cotangent_actor_value", 64),
        "cotangent_neglogp": _pattern((batch_size,), "cotangent_neglogp", 64),
        "cotangent_entropy": _pattern((batch_size,), "cotangent_entropy", 64),
        "cotangent_central_value": _pattern((batch_size, 1), "cotangent_central_value", 64),
    }
    if block_ids.tolist() != [50, 50, 40, 40, 30, 30, 20, 20, 10, 10, 0, 0]:
        raise RuntimeError("synthetic block-ID coverage drift")
    return inputs


def _fill_parameter(parameter: torch.Tensor, name: str) -> None:
    offset = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big") % 257
    flat = parameter.view(-1)
    chunk_size = 1_000_000
    with torch.no_grad():
        for start in range(0, flat.numel(), chunk_size):
            stop = min(start + chunk_size, flat.numel())
            values = torch.arange(start, stop, dtype=torch.int64, device=flat.device)
            values = ((values * 37 + offset) % 257 - 128).to(dtype=flat.dtype)
            flat[start:stop].copy_(values / 8192)


def fill_parameters(module: torch.nn.Module, role: str) -> None:
    for name, parameter in module.named_parameters():
        _fill_parameter(parameter, f"{role}:{name}")


def _encoded_bytes(value: torch.Tensor | np.ndarray) -> dict[str, Any]:
    data = _array(value).tobytes(order="C")
    return {"sha256": _sha256(data), "base64": base64.b64encode(data).decode()}


def rng_state() -> dict[str, Any]:
    numpy_name, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached = np.random.get_state()
    return {
        "numpy": {
            "algorithm": numpy_name,
            "keys": _encoded_bytes(numpy_keys),
            "position": int(numpy_position),
            "has_gauss": int(numpy_has_gauss),
            "cached_gaussian": float(numpy_cached),
        },
        "torch_cpu": _encoded_bytes(torch.get_rng_state()),
        "torch_cuda": [_encoded_bytes(state) for state in torch.cuda.get_rng_state_all()],
    }


def configure_canonical_execution() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def execution_platform() -> dict[str, Any]:
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime_version = ctypes.c_int()
    runtime_status = ctypes.CDLL("libcudart.so").cudaRuntimeGetVersion(
        ctypes.byref(runtime_version)
    )
    if runtime_status != 0:
        raise RuntimeError(f"cudaRuntimeGetVersion failed with status {runtime_status}")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_runtime": runtime_version.value,
        "cudnn": torch.backends.cudnn.version(),
        "driver": driver,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "flags": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
    }


def _assert_namespace(expected_package_root: Path) -> None:
    expected_package_root = expected_package_root.resolve()
    for name, module in sorted(__import__("sys").modules.items()):
        if name != "rl_games" and not name.startswith("rl_games."):
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"loaded {name} has no auditable __file__")
        path = Path(module_file).resolve()
        try:
            path.relative_to(expected_package_root)
        except ValueError as exc:
            raise RuntimeError(
                f"loaded {name} resolves outside {expected_package_root}: {path}"
            ) from exc


def _build_native_models(spec: dict[str, Any], device: torch.device):
    from rl_games.algos_torch.central_value import CentralValueTrain
    from rl_games.algos_torch.model_builder import ModelBuilder
    from rl_games.algos_torch.models import ModelA2CContinuousLogStd, ModelCentralValue

    coefficient_ids = torch.tensor(COEFFICIENT_IDS, dtype=torch.float32, device=device)
    actor_builder = ModelBuilder().load(
        {"model": copy.deepcopy(spec["model"]), "network": copy.deepcopy(spec["network"])}
    )
    if not isinstance(actor_builder, ModelA2CContinuousLogStd):
        raise RuntimeError("Source owner did not select ModelA2CContinuousLogStd")
    actor = actor_builder.build(
        {
            "actions_num": spec["actions"],
            "input_shape": (spec["actor_obs"] + spec["embedding_size"],),
            "num_seqs": spec["synthetic_case"]["sequence_count"],
            "value_size": 1,
            "normalize_value": spec["normalize_value"],
            "normalize_input": spec["normalize_input"],
            "type": "extra_param",
            "coef_ids": coefficient_ids,
            "coef_id_idx": spec["actor_obs"],
        }
    )
    if not isinstance(actor, ModelA2CContinuousLogStd.Network):
        raise RuntimeError("actor native model wrapper drift")

    central_builder = ModelBuilder().load(
        {
            "model": {"name": "central_value"},
            "network": copy.deepcopy(spec["central_network"]),
        }
    )
    central_config = copy.deepcopy(spec["central_training_config"])
    central_config["minibatch_size"] = spec["synthetic_case"]["batch_size"]
    central = CentralValueTrain(
        state_shape=(spec["critic_obs"] + spec["embedding_size"],),
        value_size=1,
        ppo_device="cpu",
        num_agents=1,
        horizon_length=1,
        num_actors=spec["synthetic_case"]["batch_size"],
        num_actions=spec["actions"],
        seq_length=1,
        normalize_value=spec["normalize_value"],
        network=central_builder,
        config=central_config,
        writter=None,
        max_epochs=1,
        multi_gpu=False,
        zero_rnn_on_done=True,
        type="extra_param",
        coef_ids=coefficient_ids,
        coef_id_idx=spec["critic_obs"],
    )
    if not isinstance(central.model, ModelCentralValue.Network):
        raise RuntimeError("central native model wrapper drift")
    return actor, central


def _observed_contract(actor: torch.nn.Module, central: torch.nn.Module) -> dict[str, Any]:
    actor_native = actor.a2c_network
    central_native = central.model.a2c_network
    observed = {
        "actor_embedding_shape": list(actor_native.extra_params.shape),
        "central_embedding_shape": list(central_native.extra_params.shape),
        "conditional_sigma_shape": list(actor_native.sigma.shape),
        "actor_lstm_input": actor_native.rnn.rnn.input_size,
        "actor_lstm_hidden": actor_native.rnn.rnn.hidden_size,
        "actor_mlp_input": actor_native.actor_mlp[0].in_features,
        "central_mlp_input": central_native.actor_mlp[0].in_features,
        "actor_has_layer_norm": hasattr(actor_native, "layer_norm"),
        "actor_is_rnn": actor_native.is_rnn(),
        "central_is_rnn": central_native.is_rnn(),
    }
    expected = {
        "actor_embedding_shape": [6, 32],
        "central_embedding_shape": [6, 32],
        "conditional_sigma_shape": [6, 29],
        "actor_lstm_input": 172,
        "actor_lstm_hidden": 1024,
        "actor_mlp_input": 1024,
        "central_mlp_input": 194,
        "actor_has_layer_norm": True,
        "actor_is_rnn": True,
        "central_is_rnn": False,
    }
    if observed != expected:
        raise RuntimeError(f"native network architecture drift: {observed}")
    return observed


def _hook(storage: dict[str, torch.Tensor], name: str):
    def capture(_module, _arguments, output):
        if isinstance(output, tuple):
            output = output[0]
        storage[name] = output.detach()

    return capture


def capture_network(
    spec: dict[str, Any], expected_package_root: Path, inputs: dict[str, np.ndarray] | None = None
) -> NetworkCapture:
    if not torch.cuda.is_available():
        raise RuntimeError("canonical SAPG network capture requires CUDA")
    device = torch.device("cuda:0")
    configure_canonical_execution()

    from rl_games.torch_runner import Runner

    runner = Runner()
    runner.load({"params": copy.deepcopy(spec["runner_params"])})
    configure_canonical_execution()
    after_native_seed = rng_state()

    actor, central = _build_native_models(spec, device)
    native_initialization_hashes = {
        "actor": parameter_hashes(actor),
        "central": parameter_hashes(central),
    }
    after_native_initialization = rng_state()
    observed_contract = _observed_contract(actor, central)

    fill_parameters(actor, "actor")
    fill_parameters(central, "central")
    deterministic_parameter_hashes = {
        "actor": parameter_hashes(actor),
        "central": parameter_hashes(central),
    }
    actor.to(device).train()
    central.to(device).train()
    actor.zero_grad(set_to_none=True)
    central.zero_grad(set_to_none=True)

    generated_inputs = make_inputs(spec)
    if inputs is not None:
        if set(inputs) != set(generated_inputs):
            raise RuntimeError("network input inventory drift")
        for name in inputs:
            if array_sha256(inputs[name]) != array_sha256(generated_inputs[name]):
                raise RuntimeError(f"network deterministic input drift: {name}")
        generated_inputs = {name: _array(value) for name, value in inputs.items()}
    tensors = {
        name: torch.as_tensor(value, device=device) for name, value in generated_inputs.items()
    }

    intermediates: dict[str, torch.Tensor] = {}
    actor_native = actor.a2c_network
    central_native = central.model.a2c_network
    handles = [
        actor_native.rnn.register_forward_hook(_hook(intermediates, "actor_lstm")),
        actor_native.layer_norm.register_forward_hook(_hook(intermediates, "actor_layer_norm")),
        actor_native.actor_mlp.register_forward_hook(_hook(intermediates, "actor_mlp")),
        central_native.actor_mlp.register_forward_hook(_hook(intermediates, "central_mlp")),
    ]
    actor_result = actor(
        {
            "is_train": True,
            "obs": tensors["actor_obs"],
            "prev_actions": tensors["actions"],
            "rnn_states": (tensors["rnn_hidden"], tensors["rnn_cell"]),
            "dones": tensors["dones"],
            "seq_length": spec["synthetic_case"]["sequence_length"],
        }
    )
    central_result = central.forward(
        {"is_train": True, "obs": tensors["central_obs"], "rnn_states": None}
    )
    for handle in handles:
        handle.remove()

    torch.autograd.backward(
        (
            actor_result["mus"],
            actor_result["sigmas"],
            actor_result["values"],
            actor_result["prev_neglogp"],
            actor_result["entropy"],
        ),
        (
            tensors["cotangent_mu"],
            tensors["cotangent_sigma"],
            tensors["cotangent_actor_value"],
            tensors["cotangent_neglogp"],
            tensors["cotangent_entropy"],
        ),
    )
    torch.autograd.backward((central_result["values"],), (tensors["cotangent_central_value"],))
    torch.cuda.synchronize(device)

    mapped = {
        "actor_embedding": _array(actor_native.extra_params),
        "central_embedding": _array(central_native.extra_params),
        "actor_lstm": _array(intermediates["actor_lstm"]),
        "actor_layer_norm": _array(intermediates["actor_layer_norm"]),
        "actor_mlp": _array(intermediates["actor_mlp"]),
        "central_mlp": _array(intermediates["central_mlp"]),
        "mu": _array(actor_result["mus"]),
        "conditional_sigma": _array(actor_result["sigmas"]),
        "actor_shared_value": _array(actor_result["values"]),
        "central_value": _array(central_result["values"]),
        "neglogp": _array(actor_result["prev_neglogp"]),
        "entropy": _array(actor_result["entropy"]),
    }
    if set(mapped) != set(MAPPED_TENSORS):
        raise RuntimeError("mapped network tensor inventory drift")
    actor_gradients, actor_absent = gradient_signatures(actor, "actor")
    central_gradients, central_absent = gradient_signatures(central, "central")
    _assert_namespace(expected_package_root)
    return NetworkCapture(
        tensors=mapped,
        inputs=generated_inputs,
        native_initialization_hashes=native_initialization_hashes,
        deterministic_parameter_hashes=deterministic_parameter_hashes,
        actor_parameter_signatures=parameter_signatures(actor, "actor"),
        actor_gradient_signatures=actor_gradients,
        actor_absent_gradients=actor_absent,
        central_parameter_signatures=parameter_signatures(central, "central"),
        central_gradient_signatures=central_gradients,
        central_absent_gradients=central_absent,
        rng_states={
            "after_native_seed": after_native_seed,
            "after_native_initialization": after_native_initialization,
            "after_capture": rng_state(),
        },
        platform=execution_platform(),
        observed_contract=observed_contract,
    )


def _signature_hashes(entries: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {name: signature["sha256"] for name, signature in entries.items()}


def replay_network_fixture(fixture: NetworkFixture) -> NetworkReplay:
    input_arrays = {
        name.removeprefix("input__"): value
        for name, value in fixture.arrays.items()
        if name.startswith("input__")
    }
    from tests.algos.rlgames_sapg import _runtime_requirement

    capture = capture_network(
        fixture.manifest["network_spec"],
        _runtime_requirement.VENDOR_PACKAGE_ROOT,
        inputs=input_arrays,
    )
    _runtime_requirement.require_simtoolreal_rl_games()
    current_platform = capture.platform
    canonical_platform = fixture.manifest["platform"]
    is_canonical = current_platform == canonical_platform
    max_abs_errors = {}
    max_rel_errors = {}
    for name in MAPPED_TENSORS:
        difference = np.abs(capture.tensors[name] - fixture.tensors[name])
        denominator = np.abs(fixture.tensors[name])
        relative = np.divide(
            difference,
            denominator,
            out=np.zeros_like(difference),
            where=denominator != 0,
        )
        max_abs_errors[name] = float(difference.max(initial=0.0))
        max_rel_errors[name] = float(relative.max(initial=0.0))
    computed_hashes_exact = all(
        array_sha256(capture.tensors[name])
        == fixture.manifest["npz_arrays"][f"trace__{name}"]["sha256"]
        for name in MAPPED_TENSORS
    )
    return NetworkReplay(
        tensors=capture.tensors,
        native_initialization_hashes_exact=(
            capture.native_initialization_hashes
            == fixture.manifest["native_initialization_parameter_hashes"]
        ),
        weight_hashes_exact=(
            capture.deterministic_parameter_hashes
            == fixture.manifest["deterministic_fill"]["parameter_hashes"]
        ),
        input_hashes_exact=all(
            array_sha256(value) == fixture.manifest["npz_arrays"][f"input__{name}"]["sha256"]
            for name, value in capture.inputs.items()
        ),
        computed_hashes_exact=computed_hashes_exact,
        is_canonical_platform=is_canonical,
        actor_parameter_signatures=SignatureCollection(
            capture.actor_parameter_signatures, require_hash=True
        ),
        actor_gradient_signatures=SignatureCollection(
            capture.actor_gradient_signatures, require_hash=is_canonical
        ),
        central_parameter_signatures=SignatureCollection(
            capture.central_parameter_signatures, require_hash=True
        ),
        central_gradient_signatures=SignatureCollection(
            capture.central_gradient_signatures, require_hash=is_canonical
        ),
        max_abs_errors=max_abs_errors,
        max_rel_errors=max_rel_errors,
    )
