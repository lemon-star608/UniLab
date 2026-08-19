#!/usr/bin/env python3
"""Capture the canonical Source SAPG rollout through the native A2CAgent path."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.algos.rlgames_sapg.source_network_harness import (  # noqa: E402
    _canonical_manifest_payload,
    array_metadata,
)
from tests.algos.rlgames_sapg.source_rollout_harness import (  # noqa: E402
    CANONICAL_PLATFORM,
    FIXTURE_SCHEMA_VERSION,
    SOURCE_HEAD,
    SOURCE_RL_GAMES_TREE,
    capture_rollout,
)

from scripts.generate_simtoolreal_sapg_network_fixture import (  # noqa: E402
    TASK_OWNER_BLOB,
    TASK_OWNER_PATH,
    TASK_OWNER_SHA256,
    TRAIN_OWNER_BLOB,
    TRAIN_OWNER_PATH,
    TRAIN_OWNER_SHA256,
    _owner_contract,
    _run_git,
    _sha256,
    _verify_source_modules,
)

FIXTURE_BUDGET = 8 * 1024 * 1024
FIXTURE_NAMES = ("source_rollout_fp32.npz", "source_rollout_manifest.json")


def _runner_params(source: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    owner_contract, network_spec = _owner_contract(source)
    params = copy.deepcopy(network_spec["runner_params"])
    config = params["config"]
    owner_defaults = dict(
        num_envs=owner_contract["num_envs"],
        expl_coef_block_size=owner_contract["block_size"],
        horizon_length=config["horizon_length"],
        seq_length=config["seq_length"],
        minibatch_size=config["minibatch_size"],
        central_minibatch_size=config["central_value_config"]["minibatch_size"],
    )
    overrides = dict(
        num_actors=12,
        expl_coef_block_size=2,
        horizon_length=4,
        seq_length=4,
        minibatch_size=12,
        central_minibatch_size=12,
        full_experiment_name=config["name"],
    )
    config.update({key: overrides[key] for key in overrides if key != "central_minibatch_size"})
    config["central_value_config"]["minibatch_size"] = overrides["central_minibatch_size"]
    return params, owner_defaults, overrides


def _reject_symlink_components(output: Path) -> None:
    current = Path(output.anchor) if output.is_absolute() else Path.cwd()
    components = output.parts[1:] if output.is_absolute() else output.parts
    for component in components:
        if current.is_symlink():
            raise RuntimeError(f"rollout fixture output component cannot be a symlink: {current}")
        current = current.parent if component == ".." else current / component
    if current.is_symlink():
        raise RuntimeError(f"rollout fixture output component cannot be a symlink: {current}")


def _validated_fixture_paths(output: Path, *, create: bool = False) -> tuple[Path, Path]:
    _reject_symlink_components(output)
    if output.exists() and not output.is_dir():
        raise RuntimeError(f"rollout fixture output must be a real directory: {output}")
    if create:
        output.mkdir(parents=True, exist_ok=True)
    paths = tuple(output / name for name in FIXTURE_NAMES)
    for path in paths:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise RuntimeError(f"rollout fixture leaf {path.name} must be a regular file")
    return paths


def _write(output: Path, manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    npz_path, manifest_path = _validated_fixture_paths(output, create=True)
    np.savez_compressed(npz_path, **arrays)
    manifest["fixture_files"] = {
        "npz": {"byte_size": npz_path.stat().st_size, "sha256": _sha256(npz_path.read_bytes())}
    }
    manifest["manifest_payload_sha256"] = _sha256(_canonical_manifest_payload(manifest))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    total = npz_path.stat().st_size + manifest_path.stat().st_size
    if total > FIXTURE_BUDGET:
        raise RuntimeError(f"rollout fixture exceeds 8 MiB: {total}")
    print(f"fixture_npz_bytes={npz_path.stat().st_size}")
    print(f"fixture_manifest_bytes={manifest_path.stat().st_size}")
    print(f"fixture_total_bytes={total}")


def generate(source: Path, output: Path) -> None:
    if os.environ.get("UNILAB_SAPG_ORACLE_MODE") != "source":
        raise RuntimeError("rollout generation requires UNILAB_SAPG_ORACLE_MODE=source")
    _validated_fixture_paths(output)
    source = source.resolve()
    if _run_git(source, "rev-parse", "HEAD").strip() != SOURCE_HEAD:
        raise RuntimeError("Source repository HEAD drift")
    if _run_git(source, "rev-parse", f"{SOURCE_HEAD}:rl_games/rl_games").strip() != (
        SOURCE_RL_GAMES_TREE
    ):
        raise RuntimeError("Source RL-Games tree drift")
    package_root = (source / "rl_games/rl_games").resolve()
    runner_params, owner_defaults, overrides = _runner_params(source)
    capture = capture_rollout(runner_params, package_root)
    if capture.platform != CANONICAL_PLATFORM:
        raise RuntimeError("Source rollout capture is not canonical")
    loaded_modules = _verify_source_modules(source, package_root)
    synthetic_contract = json.loads(
        '{"actions":29,"actor_carrier":141,"actor_obs":140,"base_rows":48,"blocks":6,"central_carrier":163,"coefficient_ids":[50,40,30,20,10,0],"follower_rows":8,"horizon_length":4,"num_envs":12,"privileged_state":162,"seq_length":4}'
    )
    synthetic_contract.update(
        augmented_rows=capture.semantics["augmented_rows"],
        actor_dataset_batches=capture.semantics["actor_dataset_batches"],
        central_dataset_batches=capture.semantics["central_dataset_batches"],
    )
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
            "loaded_source_modules": loaded_modules,
        },
        "owner_defaults": owner_defaults,
        "test_only_overrides": overrides,
        "runner_params": runner_params,
        "platform": capture.platform,
        "synthetic_contract": synthetic_contract,
        "native_calls": "Runner.load|Runner.set_vec_env|Runner.algo_factory.create(a2c_continuous)|A2CAgent.init_tensors|A2CAgent.env_reset|A2CAgent.play_steps|ExperienceBuffer.tensor_dict raw snapshot|rl_games.common.custom_utils.swap_and_flatten01 raw-to-flatten transform|rl_games.common.custom_utils.filter_leader via augment_batch_for_mixed_expl|A2CAgent.discount_values|A2CAgent.augment_batch_for_mixed_expl(repeat_idxs=None)|rl_games.common.custom_utils.shuffle_batch|A2CAgent.prepare_dataset(train_value_mean_std=False)|PPODataset.__getitem__|ModelCentralValue native forward|ModelA2CContinuousLogStd native forward|RnnWithDones native forward|RunningMeanStd native forward".split(
            "|"
        ),
        "semantics": capture.semantics,
        "rng_states": capture.rng_states,
        "npz_arrays": {name: array_metadata(value) for name, value in capture.arrays.items()},
        "buffer_hashes": {
            name: array_metadata(value)
            for name, value in capture.arrays.items()
            if name.startswith("buffer_")
        },
        "tolerances": {"atol": 1e-6, "rtol": 1e-5},
        "diagnostic_boundary": json.loads(
            '{"central_before_actor":true,"loss_backward_optimizer_amp":false,"name":"diagnostic_first_miniepoch_forward","train_value_mean_std":false,"value_rms_and_full_train_epoch_deferred_to_code4":true}'
        ),
        "generation": json.loads(
            '{"mode":"source-only","ordinary_pytest_regenerates":false,"python":"3.11"}'
        ),
    }
    _write(output, manifest, capture.arrays)
    print(f"loaded_source_modules={len(loaded_modules)}")
    print(f"npz_arrays={len(capture.arrays)}")
    print(f"repeat_idxs={capture.semantics['repeat_idxs']}")
    print(f"permutation={capture.semantics['permutation']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    generate(arguments.source, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
