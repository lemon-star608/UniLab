#!/usr/bin/env python3
"""Generate the reviewed Source-native Code #7 T0 fixture."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.envs.manipulation.simtoolreal.source_t0_harness import (
    SOURCE_HEAD,
    generate_source_cases,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write stable uncompressed NPY members with fixed ZIP metadata."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.ascontiguousarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload.getvalue())


def _canonical_payload_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-only", action="store_true", required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if (
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        != SOURCE_HEAD
    ):
        raise SystemExit(f"Source HEAD must be {SOURCE_HEAD}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays, inventory, extra = generate_source_cases(source)
    npz_path = output / "source_t0_fp32.npz"
    _write_deterministic_npz(npz_path, arrays)
    task_owner = subprocess.check_output(
        ["git", "-C", str(source), "show", f"{SOURCE_HEAD}:isaacsimenvs/cfg/task/SimToolReal.yaml"]
    )
    harness_path = (
        Path(__file__).resolve().parents[1]
        / "tests/envs/manipulation/simtoolreal/source_t0_harness.py"
    )
    manifest = {
        "schema_version": 1,
        "generation_mode": "source-only",
        "ordinary_pytest_regenerates": False,
        "source": {
            "path": str(source),
            "head": SOURCE_HEAD,
            "task_owner_path": "isaacsimenvs/cfg/task/SimToolReal.yaml",
            "task_owner_blob": "6469d46867081b70edaa589dcb31c7090b64d45e",
            "task_owner_sha256": hashlib.sha256(task_owner).hexdigest(),
        },
        "loaded_source_modules": inventory,
        "stub": {
            "path": "tests/envs/manipulation/simtoolreal/source_t0_harness.py",
            "sha256": _sha256(harness_path),
            "symbols": extra["stub_symbols"],
            "restriction": "quaternion math only; no task formulas or IsaacLab installation",
        },
        "cases": {
            "n": 6,
            "dtype": "float32",
            "device": "cpu",
            "seed": 0,
            "names": extra["case_names"],
            "coverage": [
                "action-delay",
                "goal-keypoint",
                "observation-delay-noise",
                "raw-reward-trackers",
                "termination",
                "source-random-reset",
                "wrench-dr",
                "tool-distribution-spec",
            ],
        },
        "config": {
            "sim_dt": 0.008333333333333333,
            "ctrl_dt": 0.016666666666666666,
            "action_dim": 29,
            "actor_obs_dim": 140,
            "critic_obs_dim": 162,
            "source_tool_pool_per_distribution": 100,
            "target_tool_pool_per_distribution": 50,
            "tool_distribution_count": 12,
            "raw_reward_scales": [200.0, 20.0, 300.0, 50.0, 1000.0, 0.03, 0.003],
        },
        "mapping": {
            "table": "Source movable table-height sample -> Target fixed reference, z range 0.0",
            "tool_pool": (
                "Source-native 12x1 distribution representatives validate Target ToolSpec math; "
                "production mapping is 12x100 Source -> 12x50 Target"
            ),
            "observation_key": "policy -> obs",
        },
        "arrays": [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            }
            for name, value in arrays.items()
        ],
        "fixture": {
            "filename": npz_path.name,
            "sha256": _sha256(npz_path),
            "canonical_payload_sha256": _canonical_payload_sha256(arrays),
        },
        "canonical_generation_command": "uv run scripts/generate_simtoolreal_task_t0_fixture.py --source /home/user/ws/lemon/simtoolreal --output tests/fixtures/simtoolreal_task --source-only",
        "float_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "discrete_exact_fields": [
            name for name, value in arrays.items() if value.dtype.kind in "biu"
        ],
        "primitive_boundary": "Explicit uniform/orientation draws are injected; Torch and NumPy RNG state/sequence parity is not claimed.",
    }
    manifest_path = output / "source_t0_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
