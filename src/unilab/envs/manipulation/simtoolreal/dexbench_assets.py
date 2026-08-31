"""Cold-path adapter for the original SimToolReal DexToolBench assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .constants import DEFAULT_JOINT_POS, JOINT_NAMES_CANONICAL

DEXTOOLBENCH_DATA_STRUCTURE: dict[str, dict[str, tuple[str, ...]]] = {
    "hammer": {
        "claw_hammer": ("swing_down", "swing_side"),
        "mallet_hammer": ("swing_down", "swing_side"),
    },
    "marker": {
        "sharpie_marker": ("draw_smile", "write_c"),
        "staples_marker": ("draw_smile", "write_c"),
    },
    "eraser": {
        "flat_eraser": ("wipe_smile", "wipe_c"),
        "handle_eraser": ("wipe_smile", "wipe_c"),
    },
    "brush": {
        "blue_brush": ("sweep_forward", "sweep_right"),
        "red_brush": ("sweep_forward", "sweep_right"),
    },
    "spatula": {
        "flat_spatula": ("serve_plate", "flip_over"),
        "spoon_spatula": ("serve_plate", "flip_over"),
    },
    "screwdriver": {
        "long_screwdriver": ("spin_vertical", "spin_horizontal"),
        "short_screwdriver": ("spin_vertical", "spin_horizontal"),
    },
}

# Policy-normalized grasp boxes from the original ``dextoolbench.objects``.
DEXTOOLBENCH_OBJECT_SCALES: dict[str, tuple[float, float, float]] = {
    "mallet_hammer": (6.0, 0.75, 0.5),
    "claw_hammer": (2.5, 0.5625, 0.375),
    "long_screwdriver": (2.5, 0.75, 0.75),
    "short_screwdriver": (1.75, 0.875, 0.875),
    "handle_eraser": (2.25, 0.8, 0.25),
    "flat_eraser": (2.5, 0.7, 1.25),
    "flat_spatula": (5.0, 0.375, 0.1875),
    "spoon_spatula": (3.0, 0.5, 0.5),
    "sharpie_marker": (2.125, 0.55, 0.55),
    "staples_marker": (3.0, 0.45, 0.45),
    "red_brush": (2.5, 0.5, 0.375),
    "blue_brush": (3.0, 0.875, 0.5),
}


@dataclass(frozen=True)
class DexBenchTaskAssets:
    source_root: Path
    category: str
    object_name: str
    task_name: str
    object_urdf: Path
    decomposed_urdf: Path
    table_urdf: Path
    trajectory: Path
    object_scale: tuple[float, float, float]
    stable_id: str = ""
    manifest_path: Path | None = None
    materialized_mjcf: Path | None = None
    input_sha256: str = ""


@dataclass(frozen=True)
class DexBenchTrajectory:
    start_pose_wxyz: tuple[float, float, float, float, float, float, float]
    goal_pos: np.ndarray
    goal_quat_wxyz: np.ndarray


@dataclass(frozen=True)
class MaterializedDexBenchScene:
    model_files: tuple[str, ...]
    trajectory_file: str
    cleanup: tempfile.TemporaryDirectory[str]


MANIFEST_SCHEMA = "unilab_dextoolbench_manifest_v1"
MATERIALIZER_VERSION = "unilab_dextoolbench_mjcf_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_ids() -> tuple[set[str], set[str]]:
    object_ids: set[str] = set()
    task_ids: set[str] = set()
    for category, objects in DEXTOOLBENCH_DATA_STRUCTURE.items():
        for object_name, tasks in objects.items():
            object_ids.add(f"{category}/{object_name}")
            task_ids.update(f"{category}/{object_name}/{task}" for task in tasks)
    return object_ids, task_ids


def _manifest_relative(root: Path, value: object, *, stable_id: str, field: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{stable_id}: manifest field {field} must be a safe relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"{stable_id}: manifest field {field} escapes the asset root") from None
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(
            f"{stable_id}: manifest field {field} is missing: {candidate}; "
            "rerun `uv run scripts/import_dexbench_assets.py`"
        )
    return candidate


def _manifest_records(path: str | Path) -> tuple[Path, dict[str, object]]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(f"DexToolBench manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"DexToolBench manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"DexToolBench manifest has an unsupported schema: {manifest_path}")
    return manifest_path, payload


def validate_manifest(
    path: str | Path,
    *,
    verify_hashes: bool = True,
    expected_counts: tuple[int, int, int] = (6, 12, 24),
) -> None:
    """Fail closed on the checked-in DexToolBench directory contract."""
    manifest_path, payload = _manifest_records(path)
    root = manifest_path.parent
    objects = payload.get("objects")
    tasks = payload.get("tasks")
    if not isinstance(objects, list) or not isinstance(tasks, list):
        raise ValueError("DexToolBench manifest objects/tasks must be lists")

    expected_objects, expected_tasks = _stable_ids()
    object_ids = [str(item.get("id", "")) for item in objects if isinstance(item, dict)]
    task_ids = [str(item.get("id", "")) for item in tasks if isinstance(item, dict)]
    categories = {stable_id.split("/", 1)[0] for stable_id in object_ids}
    expected_categories, expected_objects_count, expected_tasks_count = expected_counts
    if (
        len(categories) != expected_categories
        or len(object_ids) != expected_objects_count
        or len(task_ids) != expected_tasks_count
    ):
        raise ValueError(
            "DexToolBench manifest has unexpected category/object/task counts; "
            f"got {len(categories)}/{len(object_ids)}/{len(task_ids)}"
        )
    if len(set(object_ids)) != len(object_ids) or set(object_ids) != expected_objects:
        raise ValueError("DexToolBench manifest has missing or duplicate object IDs")
    if len(set(task_ids)) != len(task_ids) or set(task_ids) != expected_tasks:
        raise ValueError("DexToolBench manifest has missing or duplicate task IDs")

    for record in [*objects, *tasks]:
        if not isinstance(record, dict):
            raise ValueError("DexToolBench manifest records must be mappings")
        stable_id = str(record["id"])
        sources = record.get("sources")
        if not isinstance(sources, dict) or not sources:
            raise ValueError(f"{stable_id}: sources must contain path-to-SHA-256 records")
        for relative, expected_hash in sources.items():
            candidate = _manifest_relative(
                root, relative, stable_id=stable_id, field=f"sources.{relative}"
            )
            if verify_hashes and _sha256(candidate) != str(expected_hash):
                raise ValueError(
                    f"{stable_id}: source hash mismatch for {relative}; "
                    "rerun `uv run scripts/import_dexbench_assets.py`"
                )
        fields = (
            ("visual_urdf", "decomposed_urdf", "visual_mesh")
            if stable_id.count("/") == 1
            else ("table_urdf", "trajectory", "materialized_mjcf")
        )
        for field in fields:
            if field not in record or not record[field]:
                raise ValueError(f"{stable_id}: manifest field {field} is required")
            _manifest_relative(root, record[field], stable_id=stable_id, field=field)
        if stable_id.count("/") == 1:
            scale = record.get("object_scale")
            try:
                scale_array = np.asarray(scale, dtype=np.float64)
            except (TypeError, ValueError):
                scale_array = np.empty((0,), dtype=np.float64)
            if (
                not isinstance(scale, list)
                or scale_array.shape != (3,)
                or not np.isfinite(scale_array).all()
            ):
                raise ValueError(f"{stable_id}: object_scale must be a finite 3-vector")
            if any(float(value) <= 0.0 for value in scale_array):
                raise ValueError(f"{stable_id}: object_scale values must be positive")
            try:
                mass = float(cast(Any, record.get("mass")))
                com = np.asarray(record.get("com"), dtype=np.float64)
                inertia = np.asarray(record.get("full_inertia"), dtype=np.float64)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{stable_id}: mass/com/full_inertia metadata is malformed"
                ) from None
            if (
                not np.isfinite(mass)
                or mass <= 0.0
                or com.shape != (3,)
                or inertia.shape != (6,)
                or not np.isfinite(com).all()
                or not np.isfinite(inertia).all()
            ):
                raise ValueError(f"{stable_id}: mass/com/full_inertia metadata is malformed")
            collision_meshes = record.get("collision_meshes")
            if not isinstance(collision_meshes, list) or not collision_meshes:
                raise ValueError(f"{stable_id}: collision_meshes must be a non-empty list")
            for index, value in enumerate(collision_meshes):
                _manifest_relative(
                    root,
                    value,
                    stable_id=stable_id,
                    field=f"collision_meshes[{index}]",
                )

    for record in tasks:
        stable_id = str(record["id"])
        if record.get("materializer_version") != payload.get("materializer_version"):
            raise ValueError(f"{stable_id}: materializer_version does not match manifest")
        trajectory = _manifest_relative(
            root, record["trajectory"], stable_id=stable_id, field="trajectory"
        )
        try:
            trajectory_payload = json.loads(trajectory.read_text(encoding="utf-8"))
            start = np.asarray(trajectory_payload["start_pose"], dtype=np.float64)
            goals = np.asarray(trajectory_payload["goals"], dtype=np.float64)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{stable_id}: trajectory is malformed: {trajectory}") from exc
        if (
            start.shape != (7,)
            or goals.ndim != 2
            or goals.shape[0] == 0
            or goals.shape[1] != 7
            or not np.isfinite(start).all()
            or not np.isfinite(goals).all()
        ):
            raise ValueError(f"{stable_id}: trajectory must contain finite 7D start/goals")
        input_hash = record.get("input_sha256")
        if input_hash:
            source_hashes = record.get("sources", {})
            if not isinstance(source_hashes, dict):
                raise ValueError(f"{stable_id}: sources must be a mapping")
            expected_input = hashlib.sha256(
                "".join(str(value) for _, value in sorted(source_hashes.items())).encode("ascii")
            ).hexdigest()
            if str(input_hash) != expected_input:
                raise ValueError(f"{stable_id}: materialized input_sha256 does not match sources")


def build_dexbench_eval_override(
    *,
    sim_dt: float,
    ctrl_dt: float,
    object_urdf: str | Path,
    object_scale: tuple[float, float, float],
    table_urdf: str | Path,
    trajectory_file: str | Path,
    trajectory: DexBenchTrajectory,
    materialized_scene: str | Path | None = None,
) -> dict[str, object]:
    """Return the task-owner evaluation override for one DexToolBench episode.

    This keeps the fixed-start/fixed-trajectory and evaluation DR contract in
    the SimToolReal asset owner.  The viewer only supplies the selected cold
    paths and composes the resulting mapping into the registry environment.
    """
    goals = int(trajectory.goal_pos.shape[0])
    if goals <= 0:
        raise ValueError("DexToolBench trajectory must contain at least one goal")
    return {
        "sim_dt": float(sim_dt),
        "ctrl_dt": float(ctrl_dt),
        "assets": {
            "object_name": "dexbench",
            "object_urdf": str(Path(object_urdf).resolve()),
            "object_scale": tuple(float(v) for v in object_scale),
            "table_urdf": str(Path(table_urdf).resolve()),
            "materialized_scene": (
                str(Path(materialized_scene).resolve()) if materialized_scene is not None else ""
            ),
            "object_pool_enabled": False,
        },
        "reset": {
            "fixed_start_pose": tuple(float(v) for v in trajectory.start_pose_wxyz),
            "fixed_trajectory_file": str(Path(trajectory_file).resolve()),
            "fixed_trajectory_count": 1,
            "start_arm_higher": True,
            "reset_position_noise_x": 0.0,
            "reset_position_noise_y": 0.0,
            "reset_position_noise_z": 0.0,
            "reset_dof_pos_random_interval_arm": 0.0,
            "reset_dof_pos_random_interval_fingers": 0.0,
            "reset_dof_vel_random_interval": 0.0,
        },
        "goal": {
            "eval_success_tolerance": 0.01,
            "success_steps": 1,
        },
        "termination": {
            "max_consecutive_successes": goals,
        },
        "domain_randomization": {
            "use_obs_delay": False,
            "use_action_delay": False,
            "use_object_state_delay_noise": False,
            "joint_velocity_obs_noise_std": 0.0,
            "object_scale_noise_multiplier_range": (1.0, 1.0),
            "force_scale": 0.0,
            "torque_scale": 0.0,
            "force_prob_range": (0.0001, 0.0001),
            "torque_prob_range": (0.0001, 0.0001),
        },
    }


def _external_tool_spec(decomposed_urdf: str | Path, object_scale: tuple[float, float, float]):
    """Create the task-owned catalog record used by observation/reward code."""
    from .tool_catalog import ToolSpec

    link = ET.parse(decomposed_urdf).getroot().find("link")
    if link is None:
        raise ValueError(f"DexToolBench object URDF must contain one link: {decomposed_urdf}")
    inertial = link.find("inertial")
    mass = None if inertial is None else inertial.find("mass")
    inertia = None if inertial is None else inertial.find("inertia")
    if mass is None or inertia is None:
        raise ValueError(f"decomposed object URDF has incomplete inertial: {decomposed_urdf}")
    assert inertial is not None
    com = tuple(float(v) for v in _origin(inertial)[0].split())
    diagonal = tuple(float(inertia.get(name, "0")) for name in ("ixx", "iyy", "izz"))
    return ToolSpec(
        type="dexbench",
        topology="external",
        authored_shape="mesh",
        collision_shape="mesh",
        handle_size=(0.0, 0.0, 0.0),
        head_size=(0.0, 0.0, 0.0),
        head_pos=(0.0, 0.0, 0.0),
        mass=float(mass.get("value", "0")),
        com=cast(tuple[float, float, float], com),
        diaginertia=cast(tuple[float, float, float], diagonal),
        object_scale=cast(tuple[float, float, float], tuple(float(v) for v in object_scale)),
    )


def resolve_dexbench_task(
    source_root: str | Path,
    category: str,
    object_name: str,
    task_name: str,
) -> DexBenchTaskAssets:
    """Resolve one DexToolBench task from a manifest or legacy source checkout.

    A manifest is the production interface.  The directory form is retained for
    the offline importer and backwards-compatible local fixtures only.
    """
    candidate = Path(source_root).expanduser()
    manifest = candidate if candidate.is_file() else candidate / "manifest.json"
    if manifest.is_file():
        manifest_path, payload = _manifest_records(manifest)
        validate_manifest(manifest_path)
        stable_id = f"{category}/{object_name}/{task_name}"
        task_payload = payload.get("tasks")
        object_payload = payload.get("objects")
        if not isinstance(task_payload, list) or not isinstance(object_payload, list):
            raise ValueError("DexToolBench manifest objects/tasks must be lists")
        records = {
            str(record.get("id")): record for record in task_payload if isinstance(record, dict)
        }
        object_records = {
            str(record.get("id")): record for record in object_payload if isinstance(record, dict)
        }
        task_record = records.get(stable_id)
        object_record = object_records.get(f"{category}/{object_name}")
        if task_record is None or object_record is None:
            raise ValueError(f"DexToolBench manifest has no task {stable_id}")
        root = manifest_path.parent
        return DexBenchTaskAssets(
            source_root=root,
            category=category,
            object_name=object_name,
            task_name=task_name,
            object_urdf=_manifest_relative(
                root, object_record["visual_urdf"], stable_id=stable_id, field="visual_urdf"
            ),
            decomposed_urdf=_manifest_relative(
                root, object_record["decomposed_urdf"], stable_id=stable_id, field="decomposed_urdf"
            ),
            table_urdf=_manifest_relative(
                root, task_record["table_urdf"], stable_id=stable_id, field="table_urdf"
            ),
            trajectory=_manifest_relative(
                root, task_record["trajectory"], stable_id=stable_id, field="trajectory"
            ),
            object_scale=cast(
                tuple[float, float, float],
                tuple(
                    float(v) for v in cast(list[float | int | str], object_record["object_scale"])
                ),
            ),
            stable_id=stable_id,
            manifest_path=manifest_path,
            materialized_mjcf=(
                _manifest_relative(
                    root,
                    task_record["materialized_mjcf"],
                    stable_id=stable_id,
                    field="materialized_mjcf",
                )
                if task_record.get("materialized_mjcf")
                else None
            ),
            input_sha256=str(task_record.get("input_sha256", "")),
        )

    # Legacy source checkout path, used only by the offline importer and old
    # tests.  Production Viser invocations pass a manifest file.
    root = candidate.resolve()
    root = Path(source_root).expanduser().resolve()
    tasks = DEXTOOLBENCH_DATA_STRUCTURE.get(category)
    if tasks is None:
        raise ValueError(f"unknown DexToolBench category: {category!r}")
    object_tasks = tasks.get(object_name)
    if object_tasks is None:
        raise ValueError(f"object {object_name!r} is not in category {category!r}")
    if task_name not in object_tasks:
        raise ValueError(f"task {task_name!r} is not valid for object {object_name!r}")

    object_dir = root / "assets" / "urdf" / "dextoolbench" / category / object_name
    object_urdf = object_dir / f"{object_name}.urdf"
    decomposed_urdf = object_dir / f"{object_name}_decomposed.urdf"
    table_urdf = (
        root
        / "assets"
        / "urdf"
        / "dextoolbench"
        / "environments"
        / category
        / object_name
        / f"{task_name}.urdf"
    )
    trajectory = (
        root / "dextoolbench" / "trajectories" / category / object_name / f"{task_name}.json"
    )
    required = (object_urdf, decomposed_urdf, table_urdf, trajectory)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing original DexToolBench assets: " + ", ".join(missing))
    return DexBenchTaskAssets(
        source_root=root,
        category=category,
        object_name=object_name,
        task_name=task_name,
        object_urdf=object_urdf,
        decomposed_urdf=decomposed_urdf,
        table_urdf=table_urdf,
        trajectory=trajectory,
        object_scale=DEXTOOLBENCH_OBJECT_SCALES[object_name],
        stable_id=f"{category}/{object_name}/{task_name}",
    )


def load_dexbench_trajectory(
    task: DexBenchTaskAssets, *, z_offset: float = 0.03
) -> DexBenchTrajectory:
    """Read original xyzw poses and return UniLab/MuJoCo wxyz poses."""
    with task.trajectory.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    start = np.asarray(payload["start_pose"], dtype=np.float64)
    goals = np.asarray(payload["goals"], dtype=np.float64)
    if start.shape != (7,) or goals.ndim != 2 or goals.shape[1] != 7 or goals.shape[0] == 0:
        raise ValueError(f"invalid DexToolBench trajectory schema: {task.trajectory}")
    start = start.copy()
    start[2] += float(z_offset)
    start_wxyz = cast(
        tuple[float, float, float, float, float, float, float],
        tuple(float(value) for value in (*start[:3], start[6], *start[3:6])),
    )
    quat = goals[:, [6, 3, 4, 5]].copy()
    return DexBenchTrajectory(
        start_pose_wxyz=start_wxyz,
        goal_pos=goals[:, :3].copy(),
        goal_quat_wxyz=quat,
    )


def _origin(element: ET.Element) -> tuple[str, str | None]:
    origin = element.find("origin")
    if origin is None:
        return "0 0 0", None
    xyz = origin.get("xyz", "0 0 0")
    rpy = origin.get("rpy", "0 0 0")
    return xyz, None if rpy == "0 0 0" else rpy


def _rgba(element: ET.Element, default: str) -> str:
    color = element.find("material/color")
    return default if color is None else color.get("rgba", default)


def _mesh_record(
    asset: ET.Element,
    geometry: ET.Element,
    *,
    urdf_dir: Path,
    name: str,
    explicit_scale: tuple[float, float, float] | None = None,
) -> str | None:
    mesh = geometry.find("mesh")
    if mesh is None:
        return None
    file_path = (urdf_dir / mesh.get("filename", "")).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"DexToolBench mesh is missing: {file_path}")
    attrib = {"name": name, "file": str(file_path)}
    scale = explicit_scale
    if scale is not None:
        attrib["scale"] = " ".join(f"{float(value):.17g}" for value in scale)
    elif mesh.get("scale"):
        attrib["scale"] = str(mesh.get("scale"))
    ET.SubElement(asset, "mesh", attrib)
    return name


def _mesh_sidecar_files(mesh_path: Path) -> tuple[Path, ...]:
    """Return texture/material files referenced by an OBJ mesh.

    MuJoCo and ``ViserUrdf`` resolve OBJ ``mtllib``/``map_*`` references
    relative to the OBJ directory.  Keeping only the OBJ silently drops the
    original DexToolBench appearance and leaves the viewer with a flat fallback
    color, so the offline importer copies these cold-path sidecars as part of
    the asset package.
    """
    discovered: list[Path] = []
    pending = [mesh_path]
    seen: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in seen or not current.is_file():
            continue
        seen.add(current)
        if current.suffix.lower() == ".obj":
            try:
                lines = current.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            references = [
                line.split(None, 1)[1].strip()
                for line in lines
                if line.strip().lower().startswith("mtllib ") and len(line.split(None, 1)) == 2
            ]
        elif current.suffix.lower() == ".mtl":
            try:
                lines = current.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            references = [
                line.split(None, 1)[1].strip()
                for line in lines
                if line.strip().lower().startswith("map_") and len(line.split(None, 1)) == 2
            ]
        else:
            references = []
        for value in references:
            # OBJ/MTL paths are local asset references; reject absolute paths
            # and parent traversal instead of copying outside the source root.
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            referenced = (current.parent / relative).resolve()
            if referenced.is_file() and referenced not in seen:
                discovered.append(referenced)
                pending.append(referenced)
    return tuple(discovered)


def _append_object(root: ET.Element, task: DexBenchTaskAssets) -> None:
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("SimToolReal scene must contain asset and worldbody")
    body = ET.SubElement(worldbody, "body", {"name": "object"})
    ET.SubElement(body, "freejoint", {"name": "object_joint"})

    visual_root = ET.parse(task.object_urdf).getroot()
    visual_link = visual_root.find("link")
    collision_root = ET.parse(task.decomposed_urdf).getroot()
    collision_link = collision_root.find("link")
    if visual_link is None or collision_link is None:
        raise ValueError(f"DexToolBench object URDF must contain one link: {task.object_urdf}")

    visual_geoms: list[dict[str, str]] = []
    for index, visual in enumerate(visual_link.findall("visual")):
        geometry = visual.find("geometry")
        if geometry is None:
            continue
        mesh_name = _mesh_record(
            asset,
            geometry,
            urdf_dir=task.object_urdf.parent,
            name=f"dex_object_visual_mesh_{index}",
            # DexToolBench's ``object_scale`` is the policy-normalized grasp
            # box used in observations.  It is not a physical mesh scale;
            # the source URDF meshes must stay at their authored dimensions.
            explicit_scale=None,
        )
        if mesh_name is None:
            continue
        xyz, rpy = _origin(visual)
        attrib = {
            "name": f"dex_object_visual_{index}",
            "type": "mesh",
            "mesh": mesh_name,
            "pos": xyz,
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
            "density": "0",
            "rgba": _rgba(visual, "0.72 0.56 0.32 1"),
        }
        if rpy is not None:
            attrib["euler"] = rpy
        ET.SubElement(body, "geom", attrib)
        visual_geoms.append(dict(attrib, name=f"dex_goal_visual_{index}"))

    for index, collision in enumerate(collision_link.findall("collision")):
        geometry = collision.find("geometry")
        if geometry is None:
            continue
        mesh_name = _mesh_record(
            asset,
            geometry,
            urdf_dir=task.decomposed_urdf.parent,
            name=f"dex_object_collision_mesh_{index}",
            explicit_scale=None,
        )
        if mesh_name is None:
            continue
        xyz, rpy = _origin(collision)
        attrib = {
            "name": f"dex_object_collision_{index}",
            "type": "mesh",
            "mesh": mesh_name,
            "pos": xyz,
            "contype": "1",
            "conaffinity": "1",
            "group": "3",
            "density": "0",
            "friction": "0.5 0.005 0.0001",
        }
        if rpy is not None:
            attrib["euler"] = rpy
        ET.SubElement(body, "geom", attrib)

    inertial = collision_link.find("inertial")
    if inertial is None:
        raise ValueError(f"decomposed object URDF has no inertial: {task.decomposed_urdf}")
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    if mass is None or inertia is None:
        raise ValueError(f"decomposed object URDF has incomplete inertial: {task.decomposed_urdf}")
    pos, rpy = _origin(inertial)
    attributes = {
        "pos": pos,
        "mass": mass.get("value", ""),
        "fullinertia": " ".join(
            inertia.get(name, "0") for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ),
    }
    if rpy is not None:
        attributes["euler"] = rpy
    ET.SubElement(body, "inertial", attributes)

    # The goal is a visual-only copy of the same asset.  It has no joint or
    # inertial, so the selected tool remains the sole dynamic object body.
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("SimToolReal scene must contain worldbody")
    goal = ET.SubElement(worldbody, "body", {"name": "goal_object"})
    for attrib in visual_geoms:
        ET.SubElement(goal, "geom", attrib)


def _append_table_geometry(
    parent: ET.Element,
    asset: ET.Element,
    element: ET.Element,
    *,
    urdf_dir: Path,
    name: str,
    visual: bool,
) -> None:
    geometry = element.find("geometry")
    if geometry is None:
        return
    xyz, rpy = _origin(element)
    attrib = {
        "name": name,
        "pos": xyz,
        "density": "0",
        "group": "2" if visual else "3",
        "contype": "0" if visual else "1",
        "conaffinity": "0" if visual else "1",
    }
    box = geometry.find("box")
    if box is not None:
        full = np.asarray([float(v) for v in box.get("size", "").split()], dtype=np.float64)
        if full.shape != (3,):
            raise ValueError("DexToolBench table box must have three size values")
        attrib.update(type="box", size=" ".join(f"{v * 0.5:.17g}" for v in full))
    else:
        mesh_name = _mesh_record(asset, geometry, urdf_dir=urdf_dir, name=f"{name}_mesh")
        if mesh_name is None:
            return
        attrib.update(type="mesh", mesh=mesh_name)
    if visual:
        attrib["rgba"] = _rgba(element, "0.7 0.7 0.7 1")
    else:
        attrib["friction"] = "0.5 0.005 0.0001"
    if rpy is not None:
        attrib["euler"] = rpy
    ET.SubElement(parent, "geom", attrib)


def _append_task_table(root: ET.Element, task: DexBenchTaskAssets) -> None:
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("SimToolReal scene must contain asset and worldbody")
    table_root = ET.parse(task.table_urdf).getroot()
    link = table_root.find("link")
    if link is None:
        raise ValueError(f"DexToolBench table URDF must contain one link: {task.table_urdf}")
    body = ET.SubElement(worldbody, "body", {"name": "dex_task_table", "pos": "0 0 0.38"})
    # Element zero is the main table, already owned by UniLab scene.xml.
    for index, visual in enumerate(link.findall("visual")[1:]):
        _append_table_geometry(
            body,
            asset,
            visual,
            urdf_dir=task.table_urdf.parent,
            name=f"dex_table_visual_{index}",
            visual=True,
        )
    for index, collision in enumerate(link.findall("collision")[1:]):
        _append_table_geometry(
            body,
            asset,
            collision,
            urdf_dir=task.table_urdf.parent,
            name=f"dex_table_collision_{index}",
            visual=False,
        )
    # Keep the task fixture body observable even for plain-table tasks.
    if not list(body):
        ET.SubElement(
            body,
            "geom",
            {
                "name": "dex_table_collision_base_contract",
                "type": "box",
                "size": "0.0001 0.0001 0.0001",
                "pos": "0 0 -1",
                "contype": "0",
                "conaffinity": "0",
                "group": "3",
                "density": "0",
            },
        )


def materialize_dexbench_scene(
    common_scene_file: str | Path,
    task: DexBenchTaskAssets,
    *,
    temp_root: str | Path | None = None,
) -> MaterializedDexBenchScene:
    """Create a complete temporary MuJoCo scene and fixed trajectory file."""
    return materialize_external_scene(
        common_scene_file,
        object_urdf=task.object_urdf,
        decomposed_urdf=task.decomposed_urdf,
        object_scale=task.object_scale,
        table_urdf=task.table_urdf,
        trajectory=load_dexbench_trajectory(task),
        temp_root=temp_root,
    )


def _relativize_scene(scene_path: Path, root: Path) -> None:
    """Rewrite generated absolute mesh/include paths relative to an asset root."""
    xml_root = ET.parse(scene_path).getroot()
    for element in xml_root.iter():
        if "file" not in element.attrib:
            continue
        value = Path(element.attrib["file"])
        if value.is_absolute():
            element.set("file", os.path.relpath(value, scene_path.parent))
    ET.ElementTree(xml_root).write(scene_path, encoding="unicode")


def _relativize_robot(robot_path: Path, source_root: Path, target_root: Path) -> None:
    xml_root = ET.parse(robot_path).getroot()
    for mesh in xml_root.iter("mesh"):
        value = Path(mesh.get("file", ""))
        target = (source_root / value).resolve()
        mesh.set("file", os.path.relpath(target, target_root))
    ET.ElementTree(xml_root).write(robot_path, encoding="unicode")


def import_dexbench_assets(
    source_root: str | Path,
    destination: str | Path,
    *,
    common_scene_file: str | Path,
    source_version: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Copy the six-by-two DexToolBench package and emit a verified manifest.

    This is an explicit offline operation.  It refuses to overwrite a non-empty
    destination so an imported asset package remains reproducible.
    """
    source = Path(source_root).expanduser().resolve()
    dest = Path(destination).expanduser().resolve()
    if dest.exists() and any(dest.iterdir()) and not overwrite:
        raise FileExistsError(f"destination is not empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    object_records: list[dict[str, object]] = []
    task_records: list[dict[str, object]] = []
    object_root = source / "assets" / "urdf" / "dextoolbench"
    trajectory_root = source / "dextoolbench" / "trajectories"
    (dest / ".tmp").mkdir(parents=True, exist_ok=True)

    def copy_file(src: Path, relative: Path) -> Path:
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        return target

    for category, objects in DEXTOOLBENCH_DATA_STRUCTURE.items():
        for object_name, task_names in objects.items():
            source_dir = object_root / category / object_name
            visual = source_dir / f"{object_name}.urdf"
            decomposed = source_dir / f"{object_name}_decomposed.urdf"
            if not visual.is_file() or not decomposed.is_file():
                raise FileNotFoundError(f"{category}/{object_name}: object URDF pair is incomplete")
            visual_rel = Path("objects") / category / object_name / visual.name
            decomp_rel = Path("objects") / category / object_name / decomposed.name
            copy_file(visual, visual_rel)
            copy_file(decomposed, decomp_rel)

            visual_root = ET.parse(visual).getroot().find("link")
            decomp_root = ET.parse(decomposed).getroot().find("link")
            if visual_root is None or decomp_root is None:
                raise ValueError(f"{category}/{object_name}: object URDF must contain one link")
            mesh_files: list[Path] = []
            sidecar_files: list[Path] = []
            for element in [*visual_root.findall("visual"), *decomp_root.findall("collision")]:
                mesh = element.find("geometry/mesh")
                if mesh is None:
                    continue
                mesh_src = (source_dir / mesh.get("filename", "")).resolve()
                if not mesh_src.is_file():
                    raise FileNotFoundError(f"{category}/{object_name}: missing mesh {mesh_src}")
                mesh_rel = (
                    Path("objects") / category / object_name / mesh_src.relative_to(source_dir)
                )
                copy_file(mesh_src, mesh_rel)
                mesh_files.append(mesh_rel)
                for sidecar_src in _mesh_sidecar_files(mesh_src):
                    sidecar_rel = (
                        Path("objects")
                        / category
                        / object_name
                        / sidecar_src.relative_to(source_dir)
                    )
                    copy_file(sidecar_src, sidecar_rel)
                    if sidecar_rel not in sidecar_files:
                        sidecar_files.append(sidecar_rel)
            inertial = decomp_root.find("inertial")
            mass = inertial.find("mass") if inertial is not None else None
            inertia = inertial.find("inertia") if inertial is not None else None
            if mass is None or inertia is None:
                raise ValueError(f"{category}/{object_name}: decomposed inertial is incomplete")
            source_map = {
                str(relative): _sha256(dest / relative)
                for relative in (visual_rel, decomp_rel, *mesh_files, *sidecar_files)
            }
            object_records.append(
                {
                    "id": f"{category}/{object_name}",
                    "category": category,
                    "object": object_name,
                    "visual_urdf": visual_rel.as_posix(),
                    "decomposed_urdf": decomp_rel.as_posix(),
                    "visual_mesh": mesh_files[0].as_posix(),
                    "collision_meshes": [path.as_posix() for path in mesh_files[1:]],
                    "object_scale": list(DEXTOOLBENCH_OBJECT_SCALES[object_name]),
                    "mass": float(mass.get("value", "0")),
                    "com": [float(v) for v in _origin(cast(ET.Element, inertial))[0].split()],
                    "full_inertia": [
                        float(inertia.get(name, "0"))
                        for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                    ],
                    "sources": source_map,
                }
            )

            for task_name in task_names:
                table_src = (
                    object_root / "environments" / category / object_name / f"{task_name}.urdf"
                )
                trajectory_src = trajectory_root / category / object_name / f"{task_name}.json"
                table_rel = Path("tasks") / category / object_name / table_src.name
                trajectory_rel = Path("tasks") / category / object_name / trajectory_src.name
                copy_file(table_src, table_rel)
                copy_file(trajectory_src, trajectory_rel)
                task = DexBenchTaskAssets(
                    source_root=dest,
                    category=category,
                    object_name=object_name,
                    task_name=task_name,
                    object_urdf=dest / visual_rel,
                    decomposed_urdf=dest / decomp_rel,
                    table_urdf=dest / table_rel,
                    trajectory=dest / trajectory_rel,
                    object_scale=DEXTOOLBENCH_OBJECT_SCALES[object_name],
                    stable_id=f"{category}/{object_name}/{task_name}",
                )
                materialized = materialize_dexbench_scene(
                    common_scene_file, task, temp_root=dest / ".tmp"
                )
                try:
                    scene_rel = Path("scenes") / f"scene_{category}_{object_name}_{task_name}.xml"
                    scene_target = dest / scene_rel
                    scene_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(materialized.model_files[0], scene_target)
                    robot_target = scene_target.parent / "robot.xml"
                    shutil.copy2(
                        Path(materialized.model_files[0]).parent / "robot.xml", robot_target
                    )
                    _relativize_robot(
                        robot_target,
                        Path(materialized.model_files[0]).parent,
                        scene_target.parent,
                    )
                    scene_root = ET.parse(scene_target).getroot()
                    include = scene_root.find("include")
                    if include is not None:
                        include.set("file", "robot.xml")
                    ET.ElementTree(scene_root).write(scene_target, encoding="unicode")
                finally:
                    materialized.cleanup.cleanup()
                _relativize_scene(scene_target, dest)
                source_map = {
                    table_rel.as_posix(): _sha256(dest / table_rel),
                    trajectory_rel.as_posix(): _sha256(dest / trajectory_rel),
                    scene_rel.as_posix(): _sha256(scene_target),
                }
                task_records.append(
                    {
                        "id": task.stable_id,
                        "category": category,
                        "object": object_name,
                        "task": task_name,
                        "table_urdf": table_rel.as_posix(),
                        "trajectory": trajectory_rel.as_posix(),
                        "materialized_mjcf": scene_rel.as_posix(),
                        "materializer_version": MATERIALIZER_VERSION,
                        "input_sha256": hashlib.sha256(
                            "".join(value for _, value in sorted(source_map.items())).encode(
                                "ascii"
                            )
                        ).hexdigest(),
                        "sources": source_map,
                    }
                )

    manifest_path = dest / "manifest.json"
    payload = {
        "schema": MANIFEST_SCHEMA,
        "materializer_version": MATERIALIZER_VERSION,
        "source_version": source_version or "unknown",
        "objects": sorted(object_records, key=lambda item: str(item["id"])),
        "tasks": sorted(task_records, key=lambda item: str(item["id"])),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_manifest(manifest_path)
    return manifest_path


def materialize_external_scene(
    common_scene_file: str | Path,
    *,
    object_urdf: str | Path,
    decomposed_urdf: str | Path,
    object_scale: tuple[float, float, float],
    table_urdf: str | Path | None = None,
    trajectory: DexBenchTrajectory | None = None,
    temp_root: str | Path | None = None,
) -> MaterializedDexBenchScene:
    """Materialize one explicit external object/table into a UniLab scene."""
    source = Path(common_scene_file).resolve()
    object_task = DexBenchTaskAssets(
        source_root=source.parent,
        category="external",
        object_name=Path(object_urdf).stem,
        task_name="external",
        object_urdf=Path(object_urdf).resolve(),
        decomposed_urdf=Path(decomposed_urdf).resolve(),
        table_urdf=(Path(table_urdf).resolve() if table_urdf is not None else source),
        trajectory=source,
        object_scale=cast(tuple[float, float, float], tuple(float(v) for v in object_scale)),
    )
    cleanup = tempfile.TemporaryDirectory(
        prefix="unilab_dexbench_",
        dir=str(temp_root) if temp_root is not None else None,
    )
    try:
        root = ET.parse(source).getroot()
        include = root.find("include")
        if include is None:
            raise ValueError(f"SimToolReal scene must include the robot XML: {source}")
        include_file = Path(include.get("file", ""))
        robot_source = include_file if include_file.is_absolute() else source.parent / include_file
        robot_source = robot_source.resolve()
        if not robot_source.is_file():
            raise FileNotFoundError(f"SimToolReal robot include is missing: {robot_source}")
        robot_root = ET.parse(robot_source).getroot()
        compiler = robot_root.find("compiler")
        if compiler is not None:
            robot_root.remove(compiler)
        # The source robot declares all joint limits and actuator defaults in
        # radians.  Its original compiler also carries ``meshdir="assets"``;
        # that path is invalid after we rewrite mesh files to point at the
        # temporary self-contained scene, so keep the unit contract explicitly
        # while dropping only the stale mesh directory.
        robot_root.insert(0, ET.Element("compiler", {"angle": "radian"}))
        for mesh in robot_root.iter("mesh"):
            value = Path(mesh.get("file", ""))
            mesh_path = (robot_source.parent / "assets" / value).resolve()
            mesh.set("file", os.path.relpath(mesh_path, Path(cleanup.name)))
        robot_copy = Path(cleanup.name) / "robot.xml"
        ET.ElementTree(robot_root).write(robot_copy, encoding="unicode")
        include.set("file", str(robot_copy))
        _append_object(root, object_task)
        if table_urdf is not None:
            _append_task_table(root, object_task)

        # Keyframes belong to the task-level scene.  Replace the generic scene
        # keyframe with one that includes the selected DexToolBench start pose.
        for keyframe in list(root.findall("keyframe")):
            root.remove(keyframe)
        keyframe = ET.SubElement(root, "keyframe")
        start = (
            trajectory.start_pose_wxyz
            if trajectory is not None
            else (0.0, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0)
        )
        robot_start = [DEFAULT_JOINT_POS[name] for name in JOINT_NAMES_CANONICAL]
        ET.SubElement(
            keyframe,
            "key",
            {
                "name": "dexbench_start",
                "qpos": " ".join(f"{value:.17g}" for value in [*robot_start, *start]),
                "ctrl": " ".join(f"{value:.17g}" for value in robot_start),
            },
        )
        model_path = Path(cleanup.name) / "scene.xml"
        # Copy the formal robot mesh bundle so the generated scene is
        # self-contained. External DexToolBench mesh references remain absolute
        # and are validated before this point.
        (Path(cleanup.name) / "assets").mkdir()
        ET.ElementTree(root).write(model_path, encoding="unicode")
        trajectory_path = Path(cleanup.name) / "trajectory.json"
        if trajectory is None:
            trajectory_path.write_text(
                json.dumps({"pos": [[[0.0, 0.0, 0.7]]], "quat_wxyz": [[[1.0, 0.0, 0.0, 0.0]]]}),
                encoding="utf-8",
            )
        else:
            trajectory_path.write_text(
                json.dumps(
                    {
                        "pos": trajectory.goal_pos[None, :, :].tolist(),
                        "quat_wxyz": trajectory.goal_quat_wxyz[None, :, :].tolist(),
                    }
                ),
                encoding="utf-8",
            )
        return MaterializedDexBenchScene((str(model_path),), str(trajectory_path), cleanup)
    except Exception:
        cleanup.cleanup()
        raise


__all__ = [
    "DEXTOOLBENCH_DATA_STRUCTURE",
    "DEXTOOLBENCH_OBJECT_SCALES",
    "build_dexbench_eval_override",
    "DexBenchTaskAssets",
    "DexBenchTrajectory",
    "MaterializedDexBenchScene",
    "load_dexbench_trajectory",
    "materialize_external_scene",
    "materialize_dexbench_scene",
    "resolve_dexbench_task",
]
