from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from tests.envs.manipulation.simtoolreal.target_t1_harness import capture_target_t1

ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "simtoolreal_task"
NPZ_PATH = FIXTURE_DIR / "target_t1_fp32.npz"
MANIFEST_PATH = FIXTURE_DIR / "target_t1_manifest.json"
NPZ_SHA256 = "a416604b6c17a4dbc8b39cf9ef375fbe50589dd483020c67c1a285022343c788"
MANIFEST_SHA256 = "b5952b2ef2ffaa35cfe346ad5f0997df32ca04f78b0adf573ee2107cff878614"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_target_t1_fixture_replays_real_mujoco_capture() -> None:
    assert NPZ_PATH.is_file()
    assert MANIFEST_PATH.is_file()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["generation_mode"] == "target-real-mujoco-only"
    assert manifest["ordinary_pytest_regenerates"] is False
    assert manifest["source_accessed"] is False
    assert _sha256(NPZ_PATH) == NPZ_SHA256
    assert _sha256(MANIFEST_PATH) == MANIFEST_SHA256
    assert NPZ_SHA256 == manifest["fixture"]["npz_sha256"]
    for relative_path, expected_hash in manifest["target_files_sha256"].items():
        assert _sha256(ROOT / relative_path) == expected_hash

    distribution = importlib.metadata.distribution("mujoco-uni-runtime")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    assert manifest["runtime"] == {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "mujoco_uni_runtime": importlib.metadata.version("mujoco-uni-runtime"),
        "mujoco_uni_url": direct_url.get("url"),
        "mujoco_uni_source_sha": direct_url.get("vcs_info", {}).get("commit_id"),
    }

    expected = np.load(NPZ_PATH, allow_pickle=False)
    inventory = {entry["name"]: entry for entry in manifest["arrays"]}
    assert set(expected.files) == set(inventory)
    for name in expected.files:
        array = expected[name]
        entry = inventory[name]
        assert list(array.shape) == entry["shape"]
        assert str(array.dtype) == entry["dtype"]
        assert hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() == entry["sha256"]
        if array.dtype.kind == "f":
            assert np.isfinite(array).all()

    actual = capture_target_t1()
    assert set(actual) == set(expected.files)
    tolerance = manifest["float_tolerance"]
    for name in expected.files:
        lhs = expected[name]
        rhs = actual[name]
        if lhs.dtype.kind in "biu":
            np.testing.assert_array_equal(rhs, lhs, err_msg=name)
        else:
            np.testing.assert_allclose(
                rhs,
                lhs,
                rtol=tolerance["rtol"],
                atol=tolerance["atol"],
                err_msg=name,
            )
