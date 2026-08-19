#!/usr/bin/env python3
"""Capture the canonical Source RL-Games SAPG network through native objects."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.algos.rlgames_sapg.source_network_harness import (  # noqa: E402
    COEFFICIENT_IDS,
    FIXTURE_SCHEMA_VERSION,
    MAPPED_TENSORS,
    SOURCE_HEAD,
    SOURCE_RL_GAMES_TREE,
    _canonical_manifest_payload,
    array_metadata,
    capture_network,
)

TRAIN_OWNER_PATH = "isaacsimenvs/cfg/train/SimToolRealSAPG.yaml"
TRAIN_OWNER_BLOB = "f363d05d4a24b190b7837703b93270d8f3fe9a9c"
TRAIN_OWNER_SHA256 = "04f30820094b062412541764b3feeb1492097e75afe5ad0df3fd0e2853496d34"
TASK_OWNER_PATH = "isaacsimenvs/cfg/task/SimToolReal.yaml"
TASK_OWNER_BLOB = "6469d46867081b70edaa589dcb31c7090b64d45e"
TASK_OWNER_SHA256 = "9d2bf514f75cc8c72b20da1e8ec971163bbd4cbdf6fc74812aa4a509340acb5e"
OBS_UTILS_PATH = "isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py"
FIXTURE_BUDGET_BYTES = 8 * 1024 * 1024
CANONICAL_TORCH = "2.7.0+cu128"
CANONICAL_GPU = "NVIDIA GeForce RTX 4090"


def _run_git(source: Path, *arguments: str, binary: bool = False):
    result = subprocess.run(
        ["git", *arguments], cwd=source, check=True, capture_output=True, text=not binary
    )
    return result.stdout


def _git_blob(source: Path, source_path: str) -> tuple[str, bytes]:
    object_name = f"{SOURCE_HEAD}:{source_path}"
    blob = _run_git(source, "rev-parse", object_name).strip()
    data = _run_git(source, "cat-file", "blob", object_name, binary=True)
    return blob, data


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verified_owner(
    source: Path, source_path: str, expected_blob: str, expected_sha256: str
) -> bytes:
    blob, data = _git_blob(source, source_path)
    if blob != expected_blob:
        raise RuntimeError(f"Source owner blob drift for {source_path}: {blob}")
    if _sha256(data) != expected_sha256:
        raise RuntimeError(f"Source owner SHA256 drift for {source_path}")
    return data


def _evaluate_integer(node: ast.AST, names: dict[str, int]) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _evaluate_integer(node.left, names) * _evaluate_integer(node.right, names)
    raise RuntimeError(f"unsupported Source observation-size expression: {ast.dump(node)}")


def _source_observation_sizes(source: Path) -> tuple[dict[str, int], dict[str, str]]:
    blob, data = _git_blob(source, OBS_UTILS_PATH)
    tree = ast.parse(data.decode(), filename=OBS_UTILS_PATH)
    names: dict[str, int] = {}
    field_sizes: dict[str, int] | None = None
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                target, value = node.targets[0].id, node.value
        if target is None or value is None:
            continue
        if target.startswith("NUM_"):
            names[target] = _evaluate_integer(value, names)
        elif target == "OBS_FIELD_SIZES" and isinstance(value, ast.Dict):
            field_sizes = {
                ast.literal_eval(key): _evaluate_integer(item, names)
                for key, item in zip(value.keys, value.values, strict=True)
            }
    if field_sizes is None:
        raise RuntimeError("Source OBS_FIELD_SIZES was not found")
    return field_sizes, {
        "source_path": OBS_UTILS_PATH,
        "source_blob": blob,
        "sha256": _sha256(data),
    }


def _owner_contract(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    train_bytes = _verified_owner(source, TRAIN_OWNER_PATH, TRAIN_OWNER_BLOB, TRAIN_OWNER_SHA256)
    task_bytes = _verified_owner(source, TASK_OWNER_PATH, TASK_OWNER_BLOB, TASK_OWNER_SHA256)
    train = yaml.safe_load(train_bytes)
    task = yaml.safe_load(task_bytes)
    params = train["params"]
    config = params["config"]
    network = params["network"]
    central_config = config["central_value_config"]
    field_sizes, obs_utils_provenance = _source_observation_sizes(source)
    actor_obs = sum(field_sizes[name] for name in task["obs"]["obs_list"])
    critic_obs = sum(field_sizes[name] for name in task["obs"]["state_list"])
    num_envs = task["scene"]["num_envs"]
    block_size = config["expl_coef_block_size"]
    num_blocks = num_envs // block_size
    coefficient_ids = np.linspace(50, 0, num_blocks, dtype=np.float32).astype(int).tolist()

    if coefficient_ids != list(COEFFICIENT_IDS):
        raise RuntimeError(f"Source coefficient-ID contract drift: {coefficient_ids}")
    if actor_obs != 140 or critic_obs != 162 or task["action_space"] != 29:
        raise RuntimeError(
            f"Source task shape drift: actor={actor_obs}, critic={critic_obs}, "
            f"actions={task['action_space']}"
        )
    if network["space"]["continuous"]["fixed_sigma"] != "coef_cond":
        raise RuntimeError("Source owner conditional sigma contract drift")
    if network.get("separate") is not False:
        raise RuntimeError("Source owner actor shared-value contract drift")
    if network["rnn"] != {
        "name": "lstm",
        "units": 1024,
        "layers": 1,
        "before_mlp": True,
        "layer_norm": True,
    }:
        raise RuntimeError("Source owner actor RNN contract drift")
    if "rnn" in central_config["network"]:
        raise RuntimeError("Source owner central critic unexpectedly has an RNN")

    embedding_size = config["expl_reward_coef_embd_size"]
    contract = {
        "source_head": SOURCE_HEAD,
        "source_rl_games_tree": SOURCE_RL_GAMES_TREE,
        "train_owner_blob": TRAIN_OWNER_BLOB,
        "train_owner_sha256": TRAIN_OWNER_SHA256,
        "task_owner_blob": TASK_OWNER_BLOB,
        "task_owner_sha256": TASK_OWNER_SHA256,
        "coefficient_ids": coefficient_ids,
        "num_envs": num_envs,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "actor_obs": actor_obs,
        "actor_carrier": actor_obs + 1,
        "actor_embedded_input": actor_obs + embedding_size,
        "critic_obs": critic_obs,
        "central_carrier": critic_obs + 1,
        "central_embedded_input": critic_obs + embedding_size,
        "actions": task["action_space"],
        "actor_embedding_shape": [num_blocks, embedding_size],
        "central_embedding_shape": [num_blocks, embedding_size],
        "conditional_sigma_shape": [num_blocks, task["action_space"]],
        "actor_architecture": ["lstm", "layer_norm", "mlp"],
        "central_architecture": ["mlp"],
        "obs_utils_provenance": obs_utils_provenance,
    }
    spec = {
        "runner_params": copy.deepcopy(params),
        "model": copy.deepcopy(params["model"]),
        "network": copy.deepcopy(network),
        "central_network": copy.deepcopy(central_config["network"]),
        "central_training_config": copy.deepcopy(central_config),
        "normalize_input": config["normalize_input"],
        "normalize_value": config["normalize_value"],
        "actor_obs": actor_obs,
        "critic_obs": critic_obs,
        "actions": task["action_space"],
        "embedding_size": embedding_size,
        "synthetic_case": {
            "batch_size": 12,
            "sequence_count": 6,
            "sequence_length": 2,
            "actor_carrier": actor_obs + 1,
            "central_carrier": critic_obs + 1,
            "actions": task["action_space"],
            "rows_per_coefficient_id": 2,
        },
    }
    return contract, spec


def _canonical_platform_guard(platform_data: dict[str, Any]) -> None:
    required = {
        "python": "3.11",
        "torch": CANONICAL_TORCH,
        "cuda_build": "12.8",
        "gpu": CANONICAL_GPU,
        "compute_capability": [8, 9],
    }
    if not platform_data["python"].startswith(required["python"] + "."):
        raise RuntimeError(f"canonical Python platform unavailable: {platform_data['python']}")
    for field in ("torch", "cuda_build", "gpu", "compute_capability"):
        if platform_data[field] != required[field]:
            raise RuntimeError(
                f"canonical platform mismatch for {field}: "
                f"expected {required[field]!r}, got {platform_data[field]!r}"
            )


def _verify_source_modules(source: Path, package_root: Path) -> list[dict[str, str]]:
    import sys as runtime_sys

    records = []
    seen_paths = set()
    for module_name, module in sorted(runtime_sys.modules.items()):
        if module_name != "rl_games" and not module_name.startswith("rl_games."):
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"loaded Source module {module_name} has no __file__")
        path = Path(module_file)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"loaded Source module is not a regular file: {path}")
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(package_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"loaded Source module {module_name} resolves outside {package_root}: {resolved}"
            ) from exc
        source_path = f"rl_games/rl_games/{relative}"
        blob, git_bytes = _git_blob(source, source_path)
        current_bytes = resolved.read_bytes()
        if current_bytes != git_bytes:
            raise RuntimeError(f"loaded Source module bytes drifted: {source_path}")
        if source_path not in seen_paths:
            records.append(
                {
                    "module": module_name,
                    "source_path": source_path,
                    "source_blob": blob,
                    "sha256": _sha256(git_bytes),
                }
            )
            seen_paths.add(source_path)
    if not records:
        raise RuntimeError("Source capture did not load any rl_games modules")
    return records


def _write_fixture(output: Path, manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    if output.is_symlink():
        raise RuntimeError(f"fixture output directory cannot be a symlink: {output}")
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / "source_network_fp32.npz"
    manifest_path = output / "source_network_manifest.json"
    np.savez_compressed(npz_path, **arrays)
    manifest["fixture_files"] = {
        "npz": {"sha256": _sha256(npz_path.read_bytes()), "byte_size": npz_path.stat().st_size}
    }
    manifest["manifest_payload_sha256"] = _sha256(_canonical_manifest_payload(manifest))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    total_bytes = npz_path.stat().st_size + manifest_path.stat().st_size
    if total_bytes > FIXTURE_BUDGET_BYTES:
        raise RuntimeError(f"network fixture exceeds 8 MiB: {total_bytes} > {FIXTURE_BUDGET_BYTES}")
    print(f"fixture_npz_bytes={npz_path.stat().st_size}")
    print(f"fixture_manifest_bytes={manifest_path.stat().st_size}")
    print(f"fixture_total_bytes={total_bytes}")


def generate(source: Path, output: Path) -> None:
    if os.environ.get("UNILAB_SAPG_ORACLE_MODE") != "source":
        raise RuntimeError("Source fixture generation requires UNILAB_SAPG_ORACLE_MODE=source")
    source = source.resolve()
    if _run_git(source, "rev-parse", "HEAD").strip() != SOURCE_HEAD:
        raise RuntimeError("Source repository HEAD drift")
    if _run_git(source, "rev-parse", f"{SOURCE_HEAD}:rl_games/rl_games").strip() != (
        SOURCE_RL_GAMES_TREE
    ):
        raise RuntimeError("Source RL-Games tree drift")
    package_root = (source / "rl_games/rl_games").resolve()
    owner_contract, network_spec = _owner_contract(source)
    capture = capture_network(network_spec, package_root)
    _canonical_platform_guard(capture.platform)
    source_modules = _verify_source_modules(source, package_root)
    if capture.actor_absent_gradients or capture.central_absent_gradients:
        raise RuntimeError(
            "native VJP left named parameters without gradients: "
            f"actor={capture.actor_absent_gradients}, central={capture.central_absent_gradients}"
        )
    arrays = {
        **{f"input__{name}": value for name, value in capture.inputs.items()},
        **{f"trace__{name}": value for name, value in capture.tensors.items()},
    }
    manifest = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "provenance": {
            "source_repository": "https://github.com/tylerlum/simtoolreal.git",
            "source_head": SOURCE_HEAD,
            "source_rl_games_tree": SOURCE_RL_GAMES_TREE,
            "train_owner": {
                "path": TRAIN_OWNER_PATH,
                "blob": TRAIN_OWNER_BLOB,
                "sha256": TRAIN_OWNER_SHA256,
            },
            "task_owner": {
                "path": TASK_OWNER_PATH,
                "blob": TASK_OWNER_BLOB,
                "sha256": TASK_OWNER_SHA256,
            },
            "loaded_source_modules": source_modules,
        },
        "owner_contract": owner_contract,
        "network_spec": network_spec,
        "observed_native_contract": capture.observed_contract,
        "platform": capture.platform,
        "rng_states": capture.rng_states,
        "native_initialization_parameter_hashes": capture.native_initialization_hashes,
        "deterministic_fill": {
            "algorithm": (
                "For each role:name and flat index i, offset=uint32(sha256(role:name)[:4]) "
                "mod 257 and value=(((37*i+offset) mod 257)-128)/8192 in FP32."
            ),
            "parameter_hashes": capture.deterministic_parameter_hashes,
        },
        "input_definition": {
            "algorithm": (
                "Name-seeded integer affine patterns divided by powers of two; six IDs "
                "[50,40,30,20,10,0] each occupy two rows."
            ),
            "fixed_cotangent": (
                "Native actor outputs mu/sigma/shared-value/neglogp/entropy and native central "
                "value receive the input__cotangent_* tensors through torch.autograd.backward."
            ),
        },
        "mapped_tensors": list(MAPPED_TENSORS),
        "npz_arrays": {name: array_metadata(value) for name, value in arrays.items()},
        "signatures": {
            "actor_parameters": capture.actor_parameter_signatures,
            "actor_gradients": capture.actor_gradient_signatures,
            "central_parameters": capture.central_parameter_signatures,
            "central_gradients": capture.central_gradient_signatures,
        },
        "tolerances": {"atol": 1e-6, "rtol": 1e-5},
        "generation": {
            "mode": "source-only",
            "python": "3.11",
            "ordinary_pytest_regenerates": False,
        },
    }
    _write_fixture(output.resolve(), manifest, arrays)
    print(f"loaded_source_modules={len(source_modules)}")
    print(f"mapped_tensors={len(MAPPED_TENSORS)}")
    print(f"source_head={SOURCE_HEAD}")
    print(f"source_rl_games_tree={SOURCE_RL_GAMES_TREE}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    generate(arguments.source, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
