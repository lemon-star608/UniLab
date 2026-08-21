#!/usr/bin/env python3
"""Generate the Source-only SAPG checkpoint/player oracle fixture."""

from __future__ import annotations

# The source-preflight import intentionally follows a state-preserving shim.
# ruff: noqa: I001

import argparse
import copy
import hashlib
import os
import stat
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.algos.rlgames_sapg import source_checkpoint_harness as harness  # noqa: E402

# Code #4 keeps its source-preflight helpers in the generator module.  Importing
# that module toggles bytecode suppression as a process-level convenience; keep
# the checkpoint generator's caller-visible interpreter state unchanged.
_IMPORT_DONT_WRITE_BYTECODE = sys.dont_write_bytecode
from scripts.generate_simtoolreal_sapg_update_fixture import (  # noqa: E402
    _owner_and_override_contracts,
    _reject_preloaded_rl_games,
    _source_identity,
    _validated_source_root,
    _verify_code3,
    _verify_modules,
    _verify_source_python_tree_before_capture,
)

sys.dont_write_bytecode = _IMPORT_DONT_WRITE_BYTECODE

FIXTURE_NAMES = (harness.CHECKPOINT_FILE_NAME, harness.MANIFEST_FILE_NAME)
FIXTURE_ROOT = harness.FIXTURE_ROOT
SOURCE_CHECKOUT = Path("/home/user/ws/lemon/simtoolreal")
SOURCE_PACKAGE_RELATIVE = Path("rl_games/rl_games")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _raw_components(path: Path) -> tuple[Path, ...]:
    path = Path(path)
    if path.is_absolute():
        current = Path(path.anchor)
        parts = path.parts[1:]
    else:
        cwd = Path.cwd()
        current = Path(cwd.anchor)
        parts = (*cwd.parts[1:], *path.parts)
    components = [current]
    for part in parts:
        if part in ("", "."):
            continue
        current = current.parent if part == ".." else current / part
        components.append(current)
    return tuple(components)


def _checked_existing(path: Path, *, kind: str, label: str) -> Path:
    components = _raw_components(path)
    for index, component in enumerate(components):
        leaf = index == len(components) - 1
        try:
            mode = os.lstat(component).st_mode
        except OSError as error:
            raise RuntimeError(f"{label} component is unavailable: {component}") from error
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"{label} component cannot be a symlink: {component}")
        expected = kind if leaf else "directory"
        if (expected == "directory" and not stat.S_ISDIR(mode)) or (
            expected != "directory" and not stat.S_ISREG(mode)
        ):
            raise RuntimeError(f"{label} component must be a {expected}: {component}")
    return components[-1]


def validated_output_paths(output: Path) -> tuple[Path, Path]:
    root = _checked_existing(output, kind="directory", label="fixture output")
    expected = _checked_existing(FIXTURE_ROOT, kind="directory", label="fixture root")
    if root != expected:
        raise RuntimeError(f"fixture output must be exactly {expected}: {root}")
    destinations = tuple(root / name for name in FIXTURE_NAMES)
    for destination in destinations:
        try:
            mode = os.lstat(destination).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise RuntimeError(f"fixture leaf {destination.name} must be a regular file")
    return destinations


def _canonical_generation_command() -> str:
    return (
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 "
        "UV_INDEX=https://download.pytorch.org/whl/cu128 "
        "UNILAB_REQUIRE_SAPG=1 UNILAB_SAPG_ORACLE_MODE=source "
        "uv run --isolated --no-project --python 3.11 "
        "--with gym==0.26.2 --with torch==2.7.0 --with numpy==2.4.4 "
        "--with omegaconf==2.3.0 "
        "--with-editable /home/user/ws/lemon/simtoolreal/rl_games "
        "scripts/generate_simtoolreal_sapg_checkpoint_fixture.py "
        "--source /home/user/ws/lemon/simtoolreal "
        "--output tests/fixtures/simtoolreal_sapg"
    )


def _require_source_only_mode() -> None:
    if os.environ.get("UNILAB_SAPG_ORACLE_MODE") != "source":
        raise RuntimeError("generator requires explicit Source-only generation mode")


def _build_source_artifacts(
    source: Path, output_root: Path
) -> tuple[bytes, bytes, dict[str, object]]:
    source = _validated_source_root(source)
    owner_blobs = _source_identity(source)
    rollout_fixture = _verify_code3(output_root)
    _owner_and_override_contracts(owner_blobs, rollout_fixture.manifest)
    preverified_python = _verify_source_python_tree_before_capture(source)
    runner_params = copy.deepcopy(rollout_fixture.manifest.get("runner_params"))
    if not isinstance(runner_params, dict):
        raise RuntimeError("Code #3 runner params must be a mapping")

    _reject_preloaded_rl_games()
    previous_pycache_prefix = sys.pycache_prefix
    previous_dont_write_bytecode = sys.dont_write_bytecode
    with tempfile.TemporaryDirectory(prefix="unilab-source-checkpoint-pycache-") as pycache:
        sys.pycache_prefix = pycache
        sys.dont_write_bytecode = True
        try:
            checkpoint = harness.create_native_checkpoint(
                harness.code5_runner_params(runner_params),
                source / SOURCE_PACKAGE_RELATIVE,
            )
            runtime = harness.capture_runtime(
                checkpoint.payload,
                harness.code5_runner_params(runner_params),
                source / SOURCE_PACKAGE_RELATIVE,
            )
            manifest = harness.build_fixture_manifest(
                checkpoint,
                runtime,
                harness.code5_runner_params(runner_params),
                generation_command=_canonical_generation_command(),
            )
            _verify_modules(
                source,
                manifest["provenance"]["loaded_rl_games_modules"],
                preverified_python,
            )
            harness.validate_fixture(manifest, checkpoint.payload)
        finally:
            sys.pycache_prefix = previous_pycache_prefix
            sys.dont_write_bytecode = previous_dont_write_bytecode

    manifest_bytes = harness._strict_json_bytes(manifest) + b"\n"
    return checkpoint.payload, manifest_bytes, manifest


def _write_artifacts(output: Path, payload: bytes, manifest: bytes) -> None:
    destinations = validated_output_paths(output)
    temporary_paths: list[Path] = []
    try:
        for data, suffix in zip((payload, manifest), (".pth", ".json"), strict=True):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".sapg-checkpoint-", suffix=f"{suffix}.tmp", dir=destinations[0].parent
            )
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(data)
                stream.flush()
                os.fchmod(stream.fileno(), 0o644)
                os.fsync(stream.fileno())
            if not stat.S_ISREG(os.lstat(temporary).st_mode):
                raise RuntimeError("temporary checkpoint fixture is not a regular file")

        validated_output_paths(output)
        for temporary, destination in zip(tuple(temporary_paths), destinations, strict=True):
            os.replace(temporary, destination)
            temporary_paths.remove(temporary)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        directory_fd = os.open(destinations[0].parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        for temporary in temporary_paths:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def generate(source: Path, output: Path) -> None:
    _require_source_only_mode()
    output_root = validated_output_paths(output)[0].parent
    payload, manifest_bytes, manifest = _build_source_artifacts(source, output_root)
    # Validate the serialized form before replacing either checked-in leaf.
    decoded = harness._strict_json_loads(manifest_bytes)
    harness.validate_fixture(decoded, payload)
    _write_artifacts(output_root, payload, manifest_bytes)
    print(f"fixture_checkpoint_sha256={_sha(payload)}")
    print(f"fixture_manifest_sha256={_sha(manifest_bytes)}")
    print(f"canonical_payload_sha256={manifest['canonical_payload_sha256']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    generate(arguments.source, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
