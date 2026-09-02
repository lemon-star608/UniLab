from __future__ import annotations

import importlib
import sys
from importlib import metadata

import pytest


def test_package_import_does_not_import_optional_runtime():
    before = {name for name in sys.modules if name == "rl_games" or name.startswith("rl_games.")}
    importlib.import_module("unilab.algos.torch.rlgames_sapg")
    after = {name for name in sys.modules if name == "rl_games" or name.startswith("rl_games.")}
    assert after == before


def test_guard_accepts_pinned_external_install():
    from unilab.algos.torch.rlgames_sapg.dependency import require_rlgames_sapg

    try:
        metadata.version("unilab-simtoolreal-rl-games")
    except metadata.PackageNotFoundError:
        pytest.skip("install --extra rlgames-sapg to test the optional runtime")
    identity = require_rlgames_sapg()
    assert identity.distribution == "unilab-simtoolreal-rl-games"
    assert identity.version == "1.6.1+simtoolreal.2a991753.compat2"
    assert identity.python_files == 72
    assert identity.compatibility_patches == 7
    assert identity.package_root.name == "rl_games"


def test_linux_aarch64_is_explicitly_unsupported(monkeypatch):
    from unilab.algos.torch.rlgames_sapg import dependency

    monkeypatch.setattr(dependency.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dependency.platform, "machine", lambda: "aarch64")
    with pytest.raises(RuntimeError, match="Linux/aarch64"):
        dependency.require_rlgames_sapg()


def test_guard_fails_closed_when_distribution_is_missing(monkeypatch):
    from importlib import metadata

    from unilab.algos.torch.rlgames_sapg import dependency

    def missing(_name):
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(dependency.metadata, "distribution", missing)
    with pytest.raises(RuntimeError, match="--extra mujoco --extra rlgames-sapg"):
        dependency.require_rlgames_sapg()
