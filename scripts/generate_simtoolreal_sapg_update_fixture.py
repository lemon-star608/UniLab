#!/usr/bin/env python3
"""Generate the Source-only SAPG native update oracle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.machinery
import io
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.algos.rlgames_sapg.source_rollout_harness import (  # noqa: E402
    load_rollout_fixture,
)
from tests.algos.rlgames_sapg.source_update_harness import (  # noqa: E402
    CANONICAL_PLATFORM,
    CODE3_ANCHORS,
    SCHEMA_VERSION,
    SOURCE_HEAD,
    SOURCE_RL_GAMES_TREE,
    canonical_payload,
    capture_update,
    validate_capture,
)

FIXTURE_NAMES = ("source_update_fp32.npz", "source_update_manifest.json")
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/simtoolreal_sapg"
SOURCE_CHECKOUT = Path("/home/user/ws/lemon/simtoolreal")
SOURCE_PACKAGE_RELATIVE = Path("rl_games/rl_games")
EXPECTED_SOURCE_PYTHON_FILE_COUNT = 72
TRAIN_OWNER = (
    "isaacsimenvs/cfg/train/SimToolRealSAPG.yaml",
    "f363d05d4a24b190b7837703b93270d8f3fe9a9c",
    "04f30820094b062412541764b3feeb1492097e75afe5ad0df3fd0e2853496d34",
)
TASK_OWNER = (
    "isaacsimenvs/cfg/task/SimToolReal.yaml",
    "6469d46867081b70edaa589dcb31c7090b64d45e",
    "9d2bf514f75cc8c72b20da1e8ec971163bbd4cbdf6fc74812aa4a509340acb5e",
)

# These values are copied only after parsing the corresponding verified Git blobs.
EXPECTED_SOURCE_DEFAULTS = {
    "e_clip": 0.1,
    "critic_coef": 4.0,
    "bounds_loss_coef": 0.0001,
    "entropy_coef": 0.0,
    "learning_rate": "1e-4",
    "lr_schedule": "adaptive",
    "schedule_type": "standard",
    "kl_threshold": 0.016,
    "normalize_input": True,
    "normalize_value": True,
    "normalize_advantage": True,
    "truncate_grads": True,
    "grad_norm": 1.0,
    "gamma": 0.99,
    "tau": 0.95,
    "value_bootstrap": True,
    "ppo": True,
    "mixed_precision": True,
    "clip_value": True,
    "use_others_experience": "lf",
}
EXPECTED_OWNER_DIMENSIONS = {
    "num_envs": 24576,
    "expl_coef_block_size": 4096,
    "horizon_length": 16,
    "seq_length": 16,
    "minibatch_size": 98304,
    "central_minibatch_size": 98304,
}


@dataclass(frozen=True)
class _SerializedArtifacts:
    npz: bytes
    manifest: bytes
    manifest_data: dict[str, Any]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_oid(data: bytes) -> str:
    payload = b"blob " + str(len(data)).encode() + b"\0" + data
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()


def _git(source: Path, *arguments: str, binary: bool = False) -> str | bytes:
    command = ["git", *arguments]
    try:
        return subprocess.run(
            command,
            cwd=source,
            check=True,
            capture_output=True,
            text=not binary,
        ).stdout
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr
        if isinstance(stderr, bytes):
            detail = stderr.decode("utf-8", errors="replace").strip()
        elif isinstance(stderr, str):
            detail = stderr.strip()
        else:
            detail = ""
        detail = detail or "<no stderr>"
        operation = " ".join(command)
        raise RuntimeError(
            f"Git operation {operation} failed in checkout {source} "
            f"with exit status {exc.returncode}: {detail}"
        ) from exc


def _raw_path_components(path: Path) -> tuple[Path, ...]:
    """Return every lexically traversed component without resolving symlinks."""
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


def _checked_existing_path(path: Path, *, kind: str, label: str) -> Path:
    components = _raw_path_components(path)
    effective = components[-1]
    for index, component in enumerate(components):
        leaf = index == len(components) - 1
        try:
            mode = os.lstat(component).st_mode
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise RuntimeError(f"{label} component does not exist: {component}") from exc
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"{label} component cannot be a symlink: {component}")
        expected_kind = kind if leaf else "directory"
        valid = stat.S_ISDIR(mode) if expected_kind == "directory" else stat.S_ISREG(mode)
        if not valid:
            raise RuntimeError(f"{label} component must be a {expected_kind}: {component}")
    return effective


def validated_output_paths(output: Path) -> tuple[Path, Path]:
    root = _checked_existing_path(output, kind="directory", label="fixture output")
    expected_root = _raw_path_components(FIXTURE_ROOT)[-1]
    if root != expected_root:
        raise RuntimeError(f"fixture output must be exactly {expected_root}: {root}")
    npz_path, manifest_path = (root / name for name in FIXTURE_NAMES)
    for leaf in (npz_path, manifest_path):
        try:
            mode = os.lstat(leaf).st_mode
        except FileNotFoundError:
            continue
        except NotADirectoryError as exc:
            raise RuntimeError(f"fixture output is not a real directory: {root}") from exc
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise RuntimeError(f"fixture leaf {leaf.name} must be a regular file")
    return npz_path, manifest_path


def _read_regular_bytes(path: Path) -> bytes:
    path = _checked_existing_path(path, kind="regular file", label="input")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"input leaf could not be opened safely: {path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"input leaf must be a regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validated_source_root(source: Path) -> Path:
    source = _checked_existing_path(source, kind="directory", label="Source checkout")
    expected_source = _raw_path_components(SOURCE_CHECKOUT)[-1]
    if source != expected_source:
        raise RuntimeError(f"Source checkout must be exactly {expected_source}: {source}")
    _checked_existing_path(
        source / SOURCE_PACKAGE_RELATIVE,
        kind="directory",
        label="Source package root",
    )
    return source


def _source_identity(source: Path) -> dict[str, bytes]:
    if str(_git(source, "rev-parse", "HEAD")).strip() != SOURCE_HEAD:
        raise RuntimeError("Source HEAD drift")
    tree = str(_git(source, "rev-parse", f"{SOURCE_HEAD}:rl_games/rl_games")).strip()
    if tree != SOURCE_RL_GAMES_TREE:
        raise RuntimeError("Source RL-Games tree drift")

    blobs: dict[str, bytes] = {}
    for role, (path, expected_blob, expected_sha) in (
        ("train", TRAIN_OWNER),
        ("task", TASK_OWNER),
    ):
        object_name = f"{SOURCE_HEAD}:{path}"
        actual_blob = str(_git(source, "rev-parse", object_name)).strip()
        data = _git(source, "cat-file", "blob", object_name, binary=True)
        if not isinstance(data, bytes):
            raise RuntimeError(f"Source {role} owner was not read as bytes")
        if actual_blob != expected_blob or _sha(data) != expected_sha:
            raise RuntimeError(f"Source {role} owner drift")
        blobs[role] = data
    return blobs


def _verify_code3(output: Path):
    for name, expected_sha in CODE3_ANCHORS.items():
        if _sha(_read_regular_bytes(output / name)) != expected_sha:
            raise RuntimeError(f"Code #3 anchor drift: {name}")
    return load_rollout_fixture(output)


def _copy_exact_fields(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, wanted in expected.items():
        if name not in actual:
            raise RuntimeError(f"{label} is missing {name}")
        observed = actual[name]
        if type(observed) is not type(wanted) or observed != wanted:
            raise RuntimeError(
                f"{label} {name} drift: expected {wanted!r} "
                f"({type(wanted).__name__}), got {observed!r} "
                f"({type(observed).__name__})"
            )
        result[name] = copy.deepcopy(observed)
    return result


def _yaml_mapping(data: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"{label} owner blob is not valid UTF-8 YAML") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} owner blob must contain a mapping")
    return value


def _nested_mapping(value: Mapping[str, Any], *path: str, label: str) -> Mapping[str, Any]:
    current: Any = value
    for name in path:
        if not isinstance(current, Mapping) or not isinstance(current.get(name), Mapping):
            raise RuntimeError(f"{label} owner blob is missing mapping {'.'.join(path)}")
        current = current[name]
    return current


def _owner_contracts_from_verified_blobs(
    owner_blobs: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(owner_blobs) != {"train", "task"} or any(
        not isinstance(data, bytes) for data in owner_blobs.values()
    ):
        raise RuntimeError("verified Source owner blob inventory drift")
    train = _yaml_mapping(owner_blobs["train"], label="train")
    task = _yaml_mapping(owner_blobs["task"], label="task")
    config = _nested_mapping(train, "params", "config", label="train")
    central = _nested_mapping(config, "central_value_config", label="train")
    scene = _nested_mapping(task, "scene", label="task")
    source_defaults = _copy_exact_fields(
        config,
        EXPECTED_SOURCE_DEFAULTS,
        label="Source train owner defaults",
    )
    owner_dimensions = _copy_exact_fields(
        {
            "num_envs": scene.get("num_envs"),
            "expl_coef_block_size": config.get("expl_coef_block_size"),
            "horizon_length": config.get("horizon_length"),
            "seq_length": config.get("seq_length"),
            "minibatch_size": config.get("minibatch_size"),
            "central_minibatch_size": central.get("minibatch_size"),
        },
        EXPECTED_OWNER_DIMENSIONS,
        label="Source owner dimensions",
    )
    return source_defaults, owner_dimensions


def _owner_and_override_contracts(
    owner_blobs: Mapping[str, bytes], rollout_manifest: Mapping[str, Any]
) -> None:
    source_defaults, owner_dimensions = _owner_contracts_from_verified_blobs(owner_blobs)
    runner_params = rollout_manifest.get("runner_params")
    if not isinstance(runner_params, Mapping) or not isinstance(
        runner_params.get("config"), Mapping
    ):
        raise RuntimeError("Code #3 runner config is missing")
    config = runner_params["config"]
    code3_defaults = _copy_exact_fields(
        config,
        source_defaults,
        label="Code #3 copy of Source owner defaults",
    )
    code3_dimensions = _copy_exact_fields(
        rollout_manifest.get("owner_defaults", {}),
        owner_dimensions,
        label="Code #3 copy of Source owner dimensions",
    )
    if code3_defaults != source_defaults or code3_dimensions != owner_dimensions:
        raise RuntimeError("Code #3 owner contract differs from verified Source owner blobs")
    central = config.get("central_value_config")
    if not isinstance(central, Mapping):
        raise RuntimeError("Code #3 central value config is missing")
    _copy_exact_fields(
        config,
        {
            "num_actors": 12,
            "expl_coef_block_size": 2,
            "horizon_length": 4,
            "seq_length": 4,
            "minibatch_size": 12,
            "mini_epochs": 2,
        },
        label="Code #4 actor boundary",
    )
    _copy_exact_fields(
        central,
        {"minibatch_size": 12, "mini_epochs": 2},
        label="Code #4 central boundary",
    )
    blocks, remainder = divmod(config["num_actors"], config["expl_coef_block_size"])
    if (blocks, remainder) != (6, 0):
        raise RuntimeError("Code #4 six-block test boundary drift")


def _working_source_python_paths(package_root: Path) -> dict[str, Path]:
    package_root = _checked_existing_path(
        package_root,
        kind="directory",
        label="Source package root",
    )
    result: dict[str, Path] = {}
    pending = [package_root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise RuntimeError(
                f"Source Python working tree cannot be enumerated: {directory}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(package_root).as_posix()
            try:
                if entry.is_symlink():
                    raise RuntimeError(f"Source Python working tree contains symlink: {relative}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif any(
                    entry.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
                ) or (
                    any(
                        entry.name.endswith(suffix)
                        for suffix in importlib.machinery.BYTECODE_SUFFIXES
                    )
                    and path.parent.name != "__pycache__"
                ):
                    raise RuntimeError(f"Source Python import candidate drift: {relative}")
                elif entry.name.endswith(".py"):
                    if not entry.is_file(follow_symlinks=False):
                        raise RuntimeError(
                            f"Source Python working tree entry is not regular: {relative}"
                        )
                    if relative in result:
                        raise RuntimeError(
                            f"Source Python working inventory has duplicate path: {relative}"
                        )
                    result[relative] = path
            except OSError as exc:
                raise RuntimeError(
                    f"Source Python working tree entry is unavailable: {relative}"
                ) from exc
    return result


def _verify_source_python_tree_before_capture(source: Path) -> dict[str, str]:
    tree_data = _git(
        source,
        "ls-tree",
        "-r",
        "-z",
        SOURCE_RL_GAMES_TREE,
        binary=True,
    )
    if not isinstance(tree_data, bytes):
        raise RuntimeError("Source Python Git tree was not read as bytes")
    if tree_data and not tree_data.endswith(b"\0"):
        raise RuntimeError("Source Python Git tree record drift")

    tracked: dict[str, tuple[str, bytes, str]] = {}
    records = tree_data[:-1].split(b"\0") if tree_data else []
    for record in records:
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, kind, raw_blob = metadata.split(b" ")
            path = raw_path.decode("utf-8")
            blob = raw_blob.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("Source Python Git tree record drift") from exc
        relative = PurePosixPath(path)
        if relative.suffix != ".py":
            continue
        if (
            mode not in {b"100644", b"100755"}
            or kind != b"blob"
            or len(blob) != 40
            or any(character not in "0123456789abcdef" for character in blob)
            or relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.as_posix() != path
            or path in tracked
        ):
            raise RuntimeError(f"Source Python Git tree entry drift: {path!r}")
        blob_data = _git(source, "cat-file", "blob", blob, binary=True)
        if not isinstance(blob_data, bytes) or _git_blob_oid(blob_data) != blob:
            raise RuntimeError(f"Source Python Git blob identity drift: {path}")
        tracked[path] = (blob, blob_data, _sha(blob_data))

    if len(tracked) != EXPECTED_SOURCE_PYTHON_FILE_COUNT:
        raise RuntimeError(
            "Source Python file count drift: "
            f"expected {EXPECTED_SOURCE_PYTHON_FILE_COUNT}, got {len(tracked)}"
        )

    package_root = source / SOURCE_PACKAGE_RELATIVE
    working_paths = _working_source_python_paths(package_root)
    if set(working_paths) != set(tracked):
        missing = sorted(set(tracked) - set(working_paths))
        extra = sorted(set(working_paths) - set(tracked))
        raise RuntimeError(
            f"Source Python working inventory drift: missing={missing}, extra={extra}"
        )

    inventory: dict[str, str] = {}
    for path, (_blob, _blob_data, expected_sha) in sorted(tracked.items()):
        working_sha = _sha(_read_regular_bytes(working_paths[path]))
        if working_sha != expected_sha:
            raise RuntimeError(f"Source Python working bytes drift: {path}")
        inventory[path] = expected_sha
    return inventory


def _verify_modules(
    source: Path,
    records: Any,
    preverified_python: Mapping[str, str],
) -> None:
    if not isinstance(records, Mapping) or not records:
        raise RuntimeError("loaded Source module inventory is empty")
    if not isinstance(preverified_python, Mapping) or not preverified_python:
        raise RuntimeError("preverified Source Python inventory is empty")
    package_root = source / SOURCE_PACKAGE_RELATIVE
    modules: set[str] = set()
    paths: set[str] = set()
    for module, record in sorted(records.items()):
        if (
            not isinstance(module, str)
            or (module != "rl_games" and not module.startswith("rl_games."))
            or module in modules
            or not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
        ):
            raise RuntimeError(f"loaded Source module record drift: {module!r}")
        raw_path = record["path"]
        digest = record["sha256"]
        if not isinstance(raw_path, str) or not isinstance(digest, str):
            raise RuntimeError(f"loaded Source module identity drift: {module}")
        relative = PurePosixPath(raw_path)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.suffix != ".py"
            or relative.as_posix() != raw_path
            or relative.as_posix() in paths
        ):
            raise RuntimeError(f"loaded Source module path drift: {module}={raw_path!r}")
        preverified_sha = preverified_python.get(raw_path)
        if not isinstance(preverified_sha, str):
            raise RuntimeError(f"loaded Source module was not preverified: {module}")
        source_path = PurePosixPath("rl_games/rl_games") / relative
        object_name = f"{SOURCE_HEAD}:{source_path.as_posix()}"
        blob = str(_git(source, "rev-parse", object_name)).strip()
        blob_data = _git(source, "cat-file", "blob", object_name, binary=True)
        if (
            len(blob) != 40
            or any(character not in "0123456789abcdef" for character in blob)
            or not isinstance(blob_data, bytes)
            or _git_blob_oid(blob_data) != blob
        ):
            raise RuntimeError(f"loaded Source module Git identity drift: {module}")
        expected_sha = _sha(blob_data)
        working_sha = _sha(_read_regular_bytes(package_root.joinpath(*relative.parts)))
        if digest != expected_sha or working_sha != expected_sha or preverified_sha != expected_sha:
            raise RuntimeError(f"loaded Source module bytes drift: {module}")
        modules.add(module)
        paths.add(relative.as_posix())


def _strict_manifest_json_bytes(manifest: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            manifest,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("update manifest is not finite strict JSON") from exc
    return (text + "\n").encode()


def _strict_manifest_json_loads(data: bytes) -> dict[str, Any]:
    def reject_nonstandard_constant(value: str):
        raise RuntimeError(f"update manifest contains non-standard JSON constant: {value}")

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise RuntimeError(f"update manifest contains non-finite JSON number: {value}")
        return parsed

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise RuntimeError(f"update manifest contains duplicate JSON key: {name}")
            result[name] = value
        return result

    try:
        decoded = json.loads(
            data,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonstandard_constant,
            parse_float=parse_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("serialized update manifest is not strict JSON") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("serialized update manifest must contain an object")
    return decoded


def _deserialize_capture(npz_bytes: bytes, manifest_bytes: bytes) -> dict[str, Any]:
    manifest = _strict_manifest_json_loads(manifest_bytes)
    try:
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise RuntimeError("update NPZ contains duplicate array names")
            arrays = {
                name: np.array(archive[name], copy=True, order="C", subok=False)
                for name in archive.files
            }
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise RuntimeError("serialized update NPZ is invalid") from exc
    capture = {"manifest": manifest, "arrays": arrays}
    validate_capture(capture)
    if manifest.get("fixture_files") != list(FIXTURE_NAMES):
        raise RuntimeError("update fixture file inventory drift")
    declared_payload = manifest.get("canonical_payload_sha256")
    actual_payload = _sha(canonical_payload(manifest))
    if declared_payload != actual_payload:
        raise RuntimeError("update manifest canonical payload drift")
    return capture


def _serialize_capture(capture: Mapping[str, Any]) -> _SerializedArtifacts:
    validate_capture(capture)
    manifest = copy.deepcopy(capture["manifest"])
    if not isinstance(manifest, dict) or manifest.get("fixture_files") != list(FIXTURE_NAMES):
        raise RuntimeError("update fixture file inventory drift")
    array_mapping = capture["arrays"]
    if not isinstance(array_mapping, Mapping):
        raise RuntimeError("capture arrays must be a mapping")
    arrays = {
        name: np.array(value, copy=True, order="C", subok=False)
        for name, value in sorted(array_mapping.items())
    }
    try:
        manifest["canonical_payload_sha256"] = _sha(canonical_payload(manifest))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("update manifest canonical payload is not strict JSON") from exc
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    npz_bytes = stream.getvalue()
    manifest_bytes = _strict_manifest_json_bytes(manifest)
    roundtrip = _deserialize_capture(npz_bytes, manifest_bytes)
    if roundtrip["manifest"] != manifest:
        raise RuntimeError("update manifest serialization roundtrip drift")
    return _SerializedArtifacts(
        npz=npz_bytes,
        manifest=manifest_bytes,
        manifest_data=roundtrip["manifest"],
    )


def _write_artifacts(output: Path, artifacts: _SerializedArtifacts) -> None:
    destinations = validated_output_paths(output)
    temporary_paths: list[Path] = []
    try:
        for data, suffix in zip(
            (artifacts.npz, artifacts.manifest), (".npz", ".json"), strict=True
        ):
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".sapg-update-",
                suffix=f"{suffix}.tmp",
                dir=destinations[0].parent,
            )
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(data)
                stream.flush()
                os.fchmod(stream.fileno(), 0o644)
                os.fsync(stream.fileno())
            if not stat.S_ISREG(os.lstat(temporary).st_mode):
                raise RuntimeError("temporary fixture leaf must be a regular file")

        validated_output_paths(output)
        for temporary, destination in zip(tuple(temporary_paths), destinations, strict=True):
            os.replace(temporary, destination)
            temporary_paths.remove(temporary)

        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_fd = os.open(destinations[0].parent, flags)
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


def _reject_preloaded_rl_games() -> None:
    loaded = sorted(
        name for name in sys.modules if name == "rl_games" or name.startswith("rl_games.")
    )
    if loaded:
        raise RuntimeError(f"Source generator process already loaded rl_games: {loaded}")


def _validate_source_capture(capture: object) -> None:
    validate_capture(capture)
    if not isinstance(capture, Mapping) or not isinstance(capture.get("manifest"), Mapping):
        raise RuntimeError("Source update capture schema drift")
    manifest = capture["manifest"]
    if (
        manifest.get("platform") != CANONICAL_PLATFORM
        or manifest.get("canonical_platform") != CANONICAL_PLATFORM
    ):
        raise RuntimeError("Source update capture is not canonical")


def _require_source_only_mode() -> None:
    if os.environ.get("UNILAB_SAPG_ORACLE_MODE") != "source":
        raise RuntimeError("generator requires explicit Source-only generation mode")


def _canonical_generation_command() -> str:
    return (
        "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 "
        "UV_INDEX=https://download.pytorch.org/whl/cu128 "
        "UNILAB_REQUIRE_SAPG=1 UNILAB_SAPG_ORACLE_MODE=source "
        "uv run --isolated --no-project --python 3.11 "
        "--with gym==0.26.2 --with torch==2.7.0 --with numpy==2.4.4 "
        "--with omegaconf==2.3.0 "
        "--with-editable /home/user/ws/lemon/simtoolreal/rl_games "
        "scripts/generate_simtoolreal_sapg_update_fixture.py "
        "--source /home/user/ws/lemon/simtoolreal "
        "--output tests/fixtures/simtoolreal_sapg"
    )


def _build_source_artifacts(source: Path, output_root: Path) -> _SerializedArtifacts:
    source = _validated_source_root(source)
    verified_owner_blobs = _source_identity(source)
    rollout_fixture = _verify_code3(output_root)
    _owner_and_override_contracts(verified_owner_blobs, rollout_fixture.manifest)
    preverified_python = _verify_source_python_tree_before_capture(source)
    runner_params = copy.deepcopy(rollout_fixture.manifest.get("runner_params"))
    if not isinstance(runner_params, dict):
        raise RuntimeError("Code #3 runner params must be a mapping")

    # The native Source namespace is first imported inside capture_update, after
    # all fixed Git identities, owner blobs and executable Python bytes have passed.
    _reject_preloaded_rl_games()
    previous_pycache_prefix = sys.pycache_prefix
    previous_dont_write_bytecode = sys.dont_write_bytecode
    with tempfile.TemporaryDirectory(prefix="unilab-source-pycache-") as pycache_prefix:
        sys.pycache_prefix = pycache_prefix
        sys.dont_write_bytecode = True
        try:
            capture = copy.deepcopy(capture_update(runner_params, source / SOURCE_PACKAGE_RELATIVE))
            if not isinstance(capture, dict) or not isinstance(capture.get("manifest"), dict):
                raise RuntimeError("Source update capture schema drift")
            capture["manifest"]["generation_command"] = _canonical_generation_command()
            _validate_source_capture(capture)
            manifest = capture["manifest"]
            _verify_modules(
                source,
                manifest["provenance"].get("loaded_rl_games_modules"),
                preverified_python,
            )
        finally:
            sys.pycache_prefix = previous_pycache_prefix
            sys.dont_write_bytecode = previous_dont_write_bytecode
    return _serialize_capture(capture)


def _print_artifact_hashes(artifacts: _SerializedArtifacts) -> None:
    print(f"fixture_npz_sha256={_sha(artifacts.npz)}")
    print(f"fixture_manifest_sha256={_sha(artifacts.manifest)}")
    print(f"canonical_payload_sha256={artifacts.manifest_data['canonical_payload_sha256']}")


def generate(source: Path, output: Path) -> None:
    _require_source_only_mode()
    if SCHEMA_VERSION != 2:
        raise RuntimeError(f"update fixture schema must be v2, got {SCHEMA_VERSION}")
    output_root = validated_output_paths(output)[0].parent
    artifacts = _build_source_artifacts(source, output_root)
    _write_artifacts(output_root, artifacts)
    _print_artifact_hashes(artifacts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    generate(arguments.source, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
