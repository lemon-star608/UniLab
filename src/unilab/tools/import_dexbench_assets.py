"""Offline DexToolBench importer owner."""

from __future__ import annotations

import argparse
from pathlib import Path

from unilab.assets import ASSETS_ROOT_PATH
from unilab.envs.manipulation.simtoolreal.dexbench_assets import import_dexbench_assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        default=ASSETS_ROOT_PATH / "dexbench",
    )
    parser.add_argument(
        "--common-scene",
        type=Path,
        default=ASSETS_ROOT_PATH / "robots" / "kuka_sharpa" / "scene.xml",
    )
    parser.add_argument("--source-version", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = import_dexbench_assets(
        args.source_root,
        args.destination,
        common_scene_file=args.common_scene,
        source_version=args.source_version,
        overwrite=args.overwrite,
    )
    print(manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
