from __future__ import annotations

import importlib.metadata
import inspect
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from mujoco_uni.batch_env import BatchEnvPool

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = "mujoco-uni-runtime"
_VERSION = "0.4.0.dev0"
_GIT_URL = "https://github.com/lemon-star608/mujoco_uni.git"
_REV = "54a2197be5b0cd65e9d71ff884d8415191925136"
_TREE = "771de554330b698bc12e5110682af1d8de433ee2"
_MANIFEST = _ROOT / "tests/fixtures/simtoolreal_sapg/m0_dev_manifest.json"


def test_mujoco_uni_dependency_identity_is_fixed_and_installed() -> None:
    pyproject_text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock_text = (_ROOT / "uv.lock").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)
    lock = tomllib.loads(lock_text)

    assert f"{_PACKAGE}=={_VERSION}" in pyproject["project"]["optional-dependencies"]["mujoco"]
    assert pyproject["tool"]["uv"]["sources"][_PACKAGE] == {
        "git": _GIT_URL,
        "rev": _REV,
    }

    packages = [package for package in lock["package"] if package["name"] == _PACKAGE]
    assert len(packages) == 1
    package = packages[0]
    assert package["version"] == _VERSION
    assert package["source"] == {"git": f"{_GIT_URL}?rev={_REV}#{_REV}"}

    forbidden_sources = ("../mujoco_uni", "/home/user/ws/lemon/mujoco_uni")
    for forbidden in forbidden_sources:
        assert forbidden not in pyproject_text
        assert forbidden not in lock_text

    dist = importlib.metadata.distribution(_PACKAGE)
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    assert dist.version == _VERSION
    assert direct_url == {
        "url": _GIT_URL,
        "vcs_info": {
            "vcs": "git",
            "commit_id": _REV,
            "requested_revision": _REV,
        },
    }
    assert "cpu_ids" in inspect.signature(BatchEnvPool.__init__).parameters
    assert callable(BatchEnvPool.worker_cpu_ids)
    assert isinstance(BatchEnvPool.was_autoreset, property)


def test_m0_dev_manifest_records_reviewed_provenance_and_abi() -> None:
    manifest_text = _MANIFEST.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    assert "/home/user/ws/lemon/mujoco_uni" not in manifest_text
    assert manifest["schema"] == "simtoolreal_sapg_m0_dev_manifest_v1"
    assert manifest["distribution"] == {
        "name": _PACKAGE,
        "version": _VERSION,
        "formal_release": "deferred",
    }
    assert manifest["source"] == {
        "url": _GIT_URL,
        "ref": "refs/heads/feat/geom-size-pos-per-env-fields",
        "commit": _REV,
        "tree": _TREE,
    }
    assert manifest["external_tests"]["focused_passed"] == 26
    assert manifest["external_tests"]["focused_skipped"] == 0
    assert manifest["external_tests"]["full_passed"] == 50
    assert manifest["external_tests"]["full_skipped"] == 0
    assert manifest["public_abi"] == {
        "batch_env_pool_accepts_cpu_ids": True,
        "batch_env_pool_worker_cpu_ids": True,
        "batch_env_pool_was_autoreset_property": True,
        "default_cpu_ids": None,
        "default_worker_cpu_ids": [],
    }
    assert manifest["target"]["direct_url"] == {
        "url": _GIT_URL,
        "requested_revision": _REV,
        "commit_id": _REV,
    }
    assert manifest["target"]["sibling_checkout_dependency"] is False
