"""Cold-path resolution for SimToolReal training robot assets."""

from __future__ import annotations

from pathlib import Path

from unilab.assets.hub import resolve_dexbench_asset_dir, resolve_robot_asset_dir

_TRAINING_ASSET_DIR = "robots/kuka_sharpa/meshes"
_TRAINING_ASSET_MARKER = ".hf_complete_v1"


def ensure_training_assets() -> Path:
    """Ensure Kuka/Sharpa training meshes are materialized locally.

    This function is intended for environment construction only. The shared
    resolver uses a marker-file fast path and downloads the missing robot mesh
    snapshot from the configured Hugging Face dataset when necessary.
    """

    return resolve_robot_asset_dir(_TRAINING_ASSET_DIR, marker=_TRAINING_ASSET_MARKER)


def ensure_dexbench_assets() -> Path:
    """Ensure the complete DexBench manifest and object/task assets exist."""

    return resolve_dexbench_asset_dir(marker=".hf_complete_v1")


__all__ = ["ensure_dexbench_assets", "ensure_training_assets"]
