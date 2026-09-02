"""Cold-path materialization of complete SimToolReal MJCF tool variants."""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .tool_catalog import ToolSpec


@dataclass(frozen=True)
class MaterializedToolScenes:
    model_files: tuple[str, ...]
    cleanup: tempfile.TemporaryDirectory[str]


def _fmt(values: tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _tool_body(spec: ToolSpec) -> str:
    geoms: list[str] = []
    if spec.collision_shape == "capsule":
        radius, half_length, _ = spec.handle_size
        quat = "0.7071067811865476 0 -0.7071067811865476 0"
        geoms.append(
            f'<geom name="object_handle_cyl_visual" type="cylinder" size="{_fmt((radius, half_length))}" '
            'quat="0.7071067811865476 0 -0.7071067811865476 0" '
            'contype="0" conaffinity="0" density="0" friction="1.0 0.005 0.0001" '
            'rgba="0.55 0.27 0.07 1"/>'
        )
        geoms.append(
            f'<geom name="object_handle_cyl" type="capsule" size="{_fmt((radius, half_length))}" '
            f'quat="{quat}" friction="1.0 0.005 0.0001" density="0"/>'
        )
    else:
        geoms.append(
            f'<geom name="object_handle" type="box" size="{_fmt(spec.handle_size)}" '
            'friction="1.0 0.005 0.0001" density="0" rgba="0.55 0.27 0.07 1"/>'
        )
    if spec.head_size != (0.0, 0.0, 0.0):
        geoms.append(
            f'<geom name="object_head" type="box" size="{_fmt(spec.head_size)}" '
            f'pos="{_fmt(spec.head_pos)}" friction="1.0 0.005 0.0001" density="0" '
            'rgba="0.5 0.5 0.5 1"/>'
        )
    return (
        '<body name="object"><freejoint name="object_joint"/>'
        + "".join(geoms)
        + f'<inertial pos="{_fmt(spec.com)}" mass="{spec.mass:.17g}" '
        f'diaginertia="{_fmt(spec.diaginertia)}"/></body>'
    )


def materialize_tool_scenes(
    common_scene_file: str,
    tools: Sequence[ToolSpec],
    *,
    temp_root: str | Path | None = None,
) -> MaterializedToolScenes:
    """Write one complete scene per immutable tool specification.

    Files are created beside the source scene so relative robot/mesh includes
    retain their original resolution. All tool geometry and inertial fields are
    authored before MuJoCo compilation; only this cold path reads the XML.
    """
    source = Path(common_scene_file)
    text = source.read_text(encoding="utf-8")
    text = re.sub(
        r"file=\"([^\"]+)\"",
        lambda match: (
            f'file="{(source.parent / match.group(1)).resolve()}"'
            if not Path(match.group(1)).is_absolute()
            else match.group(0)
        ),
        text,
    )
    if "</worldbody>" not in text:
        raise ValueError(f"Common scene has no worldbody: {source}")
    cleanup = tempfile.TemporaryDirectory(
        prefix="simtoolreal_tools_",
        dir=str(temp_root) if temp_root is not None else None,
    )
    meshes_link = Path(cleanup.name) / "meshes"
    if source.parent.joinpath("meshes").exists():
        meshes_link.symlink_to(source.parent / "meshes", target_is_directory=True)
    model_files: list[str] = []
    try:
        for index, spec in enumerate(tools):
            generated = text.replace("</worldbody>", f"{_tool_body(spec)}</worldbody>", 1)
            path = Path(cleanup.name) / f"tool_{index:04d}.xml"
            path.write_text(generated, encoding="utf-8")
            model_files.append(str(path))
    except Exception:
        cleanup.cleanup()
        raise
    return MaterializedToolScenes(tuple(model_files), cleanup)


__all__ = ["MaterializedToolScenes", "materialize_tool_scenes"]
