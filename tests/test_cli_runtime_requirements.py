from __future__ import annotations

import pytest

from unilab import cli


def test_check_runtime_requirements_requires_mujoco_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name == "mujoco" else object())

    with pytest.raises(SystemExit, match="sim=mujoco requires the MuJoCo extra"):
        cli._check_runtime_requirements("ppo", "mujoco")


def test_check_runtime_requirements_requires_motrix_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name == "motrixsim" else object())

    with pytest.raises(SystemExit, match="sim=motrix requires the Motrix extra"):
        cli._check_runtime_requirements("ppo", "motrix")


def test_check_runtime_requirements_requires_exact_rlgames_extra(monkeypatch):
    monkeypatch.setattr(cli, "find_spec", lambda name: object())
    monkeypatch.setattr(
        cli,
        "require_rlgames_sapg",
        lambda: (_ for _ in ()).throw(RuntimeError("use --extra mujoco --extra rlgames-sapg")),
        raising=False,
    )
    with pytest.raises(SystemExit, match="--extra mujoco --extra rlgames-sapg"):
        cli._check_runtime_requirements("rlgames_sapg", "mujoco")


def test_check_runtime_requirements_rejects_linux_aarch64(monkeypatch):
    monkeypatch.setattr(cli, "find_spec", lambda name: object())
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.platform, "machine", lambda: "aarch64")
    with pytest.raises(SystemExit, match="Linux/aarch64"):
        cli._check_runtime_requirements("rlgames_sapg", "mujoco")
