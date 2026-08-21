#!/usr/bin/env python3
"""Explicit generator for the Target-only real MuJoCo Code #8 T1 fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.envs.manipulation.simtoolreal.target_t1_harness import capture_target_t1

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_NAME = "target_t1_fp32.npz"
MANIFEST_NAME = "target_t1_manifest.json"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_payload(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(array_buffer, np.asanyarray(arrays[name]), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(
                member, array_buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )
    return output.getvalue()


def _array_inventory(arrays: dict[str, np.ndarray]) -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "shape": list(np.asarray(arrays[name]).shape),
            "dtype": str(np.asarray(arrays[name]).dtype),
            "sha256": _sha256_bytes(np.ascontiguousarray(arrays[name]).tobytes()),
        }
        for name in sorted(arrays)
    ]


def _runtime_identity() -> dict[str, object]:
    distribution = importlib.metadata.distribution("mujoco-uni-runtime")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    return {
        "python": os.sys.version.split()[0],
        "numpy": np.__version__,
        "mujoco": __import__("mujoco").__version__,
        "mujoco_uni_runtime": importlib.metadata.version("mujoco-uni-runtime"),
        "mujoco_uni_url": direct_url.get("url"),
        "mujoco_uni_source_sha": direct_url.get("vcs_info", {}).get("commit_id"),
    }


def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    arrays = capture_target_t1()
    npz_payload = _npz_bytes(arrays)
    npz_path = output / FIXTURE_NAME
    npz_path.write_bytes(npz_payload)

    production_files = [
        "src/unilab/envs/manipulation/__init__.py",
        "src/unilab/envs/manipulation/simtoolreal/__init__.py",
        "src/unilab/envs/manipulation/simtoolreal/config.py",
        "src/unilab/envs/manipulation/simtoolreal/env.py",
        "src/unilab/envs/manipulation/simtoolreal/dr_provider.py",
    ]
    harness_file = "tests/envs/manipulation/simtoolreal/target_t1_harness.py"
    generator_file = "scripts/generate_simtoolreal_task_t1_fixture.py"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "generation_mode": "target-real-mujoco-only",
        "ordinary_pytest_regenerates": False,
        "source_accessed": False,
        "code8_base": "eb19779ecef69bbfb495abee9e7e2c4d5988f3ac",
        "source_reference": {
            "head": "2a9917533bfea70419ed2667a511d7238e5b3abc",
            "task_owner": "isaacsimenvs/cfg/task/SimToolReal.yaml",
            "task_owner_blob": "6469d46867081b70edaa589dcb31c7090b64d45e",
        },
        "donor_reference": {
            "commit": "74075b3238e3176650a9440984a74be3629ff93f",
            "env_blob": "103089c1e8192e25326a58de46be516808853013",
        },
        "target_files_sha256": {
            path: _sha256_file(ROOT / path)
            for path in [*production_files, harness_file, generator_file]
        },
        "asset_provenance_sha256": _sha256_file(
            ROOT / "src/unilab/assets/robots/kuka_sharpa/ASSET_PROVENANCE"
        ),
        "runtime": _runtime_identity(),
        "cases": {
            "n": 6,
            "horizon": 8,
            "seed": 20260821,
            "pool": "12x50=600",
            "shuffle_seed": 42,
            "backend": "mujoco",
            "dtype": "CPU FP32",
        },
        "config": {
            "sim_dt": 1.0 / 120.0,
            "ctrl_dt": 1.0 / 60.0,
            "sim_substeps": 2,
            "episode_steps": 600,
            "action_dim": 29,
            "actor_obs_dim": 140,
            "critic_obs_dim": 162,
            "object_pool_enabled": True,
            "action_delay": True,
            "observation_delay": True,
            "object_state_delay_noise": True,
        },
        "event_script": [
            "init_state",
            "four scripted real steps",
            "selected reset rows [1,4] via NpEnv._reset_done_envs",
            "three scripted real steps",
            "set info steps[2]=599 and execute final action",
        ],
        "reward_term_order": [
            "fingertip_delta_rew",
            "lifting_rew",
            "lift_bonus_rew",
            "keypoint_rew",
            "kuka_actions_penalty",
            "hand_actions_penalty",
            "bonus_rew",
            "total_reward",
        ],
        "allowed_mapping": {
            "physics_pool_nmesh": 19,
            "complete_source_xml_nmesh": 40,
            "note": "MuJoCo discardvisual=true physics contract is retained; source XML direct compile remains 40 meshes.",
        },
        "arrays": _array_inventory(arrays),
        "discrete_exact_fields": [
            entry["name"]
            for entry in _array_inventory(arrays)
            if np.dtype(entry["dtype"]).kind in "biu"
        ],
        "fixture": {
            "npz_filename": FIXTURE_NAME,
            "manifest_filename": MANIFEST_NAME,
            "npz_sha256": _sha256_bytes(npz_payload),
            "canonical_payload_sha256": _canonical_payload(arrays),
            "canonical_command": "uv run --extra mujoco scripts/generate_simtoolreal_task_t1_fixture.py --output <dir> --target-only",
        },
        "float_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "contract_statement": "Target regression only; this is not Source/Target physical parity.",
    }
    manifest_payload = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    (output / MANIFEST_NAME).write_bytes(manifest_payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-only", action="store_true")
    args = parser.parse_args()
    if not args.target_only:
        parser.error("--target-only is required")
    generate(args.output)


if __name__ == "__main__":
    main()
