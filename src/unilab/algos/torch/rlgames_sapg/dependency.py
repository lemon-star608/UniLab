"""Fail-closed identity guard for the checked-in RL-Games runtime."""

from __future__ import annotations

import importlib.util
import json
import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from urllib.parse import urlparse

EXPECTED_DISTRIBUTION = "unilab-simtoolreal-rl-games"
EXPECTED_VERSION = "1.6.1+simtoolreal.2a991753.compat2"
EXPECTED_SOURCE_URL = "https://github.com/lemon-star608/simtoolreal-rl-games.git"
EXPECTED_SOURCE_COMMIT = "3b5363053d228a1b8b2ae49d3b828a8e5231ea83"
INSTALL_HINT = "uv run --extra mujoco --extra rlgames-sapg ..."


@dataclass(frozen=True)
class RlGamesSapgIdentity:
    distribution: str
    version: str
    package_root: Path
    python_files: int
    compatibility_patches: int


def require_rlgames_sapg() -> RlGamesSapgIdentity:
    """Verify the supported platform and pinned external runtime before import."""
    if platform.system() == "Linux" and platform.machine() == "aarch64":
        raise RuntimeError("RL-Games SAPG is unsupported on Linux/aarch64")
    try:
        distribution = metadata.distribution(EXPECTED_DISTRIBUTION)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Missing pinned RL-Games SAPG runtime; use `{INSTALL_HINT}`") from exc
    if distribution.metadata["Name"] != EXPECTED_DISTRIBUTION:
        raise RuntimeError("wrong RL-Games SAPG distribution metadata name")
    if distribution.version != EXPECTED_VERSION:
        raise RuntimeError(f"unsupported RL-Games SAPG runtime version: {distribution.version}")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("RL-Games SAPG direct_url.json is invalid") from exc
        source_url = direct_url.get("url", "")
        vcs_info = direct_url.get("vcs_info", {})
        if source_url != EXPECTED_SOURCE_URL or vcs_info.get("vcs") != "git":
            raise RuntimeError("RL-Games SAPG source URL is not the pinned public repository")
        commit = vcs_info.get("commit_id")
        if commit != EXPECTED_SOURCE_COMMIT:
            raise RuntimeError("RL-Games SAPG Git commit is not the pinned runtime revision")
    spec = importlib.util.find_spec("rl_games")
    if spec is None or spec.origin is None:
        raise RuntimeError("RL-Games SAPG runtime does not provide the rl_games package")
    package_root = Path(spec.origin).resolve().parent
    return RlGamesSapgIdentity(EXPECTED_DISTRIBUTION, EXPECTED_VERSION, package_root, 72, 7)
