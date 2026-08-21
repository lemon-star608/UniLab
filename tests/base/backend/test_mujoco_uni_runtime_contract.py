from __future__ import annotations

import importlib.metadata
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
_REV = "7205e070e983df90d520f0f8593853013e976746"


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

    assert importlib.metadata.version(_PACKAGE) == _VERSION
    assert isinstance(BatchEnvPool.was_autoreset, property)
