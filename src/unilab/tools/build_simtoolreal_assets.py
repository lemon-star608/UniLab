#!/usr/bin/env python3
"""Build the SimToolReal MJCF assets (KUKA iiwa14 + Sharpa left hand, table, object).

Usage:
  uv run python -m unilab.tools.build_simtoolreal_assets --simtoolreal-root ~/code/simtoolreal/simtoolreal

Outputs, under ``src/unilab/assets``:

  robots/kuka_sharpa/kuka_sharpa.xml   robot MJCF (29 position actuators)
  robots/kuka_sharpa/scene.xml         composed scene: robot + table + object + floor
  robots/kuka_sharpa/assets/**         meshes (ASCII STL converted to binary)
  objects/simtoolreal/hammer_single.xml  single fixed handle+head object

``unilab.tools.import_robot`` is the general URDF importer, but it shells out to
``urdf-to-mjcf`` and ends in an interactive MuJoCo viewer
(``import_robot.py:632-637``), and it only emits a keyframe-only scene. This task
needs a non-interactive build and a composed multi-asset scene, so the conversion
runs through MuJoCo's own URDF compiler here instead.

Physics values are ported verbatim from the SimToolReal source; see
``unilab.envs.manipulation.simtoolreal.constants`` for the per-joint tables and
their source line numbers.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.envs.manipulation.simtoolreal.constants import (
    ARM_JOINT_DAMPING,
    ARM_JOINT_STIFFNESS,
    DEFAULT_JOINT_POS,
    FINGERTIP_LINK_NAMES,
    HAND_JOINT_ARMATURE,
    HAND_JOINT_DAMPING,
    HAND_JOINT_FRICTION,
    HAND_JOINT_STIFFNESS,
    JOINT_NAMES_CANONICAL,
    NUM_JOINTS,
    ROBOT_ROOT_POS,
    ROBOT_ROOT_QUAT_WXYZ,
)

ROBOT_URDF_REL = "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
MESH_SUBDIRS = ("left_sharpa_meshes", "new_iiwa14_meshes")

ROBOT_DIR_NAME = "kuka_sharpa"
ROBOT_XML_NAME = "kuka_sharpa.xml"
SCENE_XML_NAME = "scene.xml"
OBJECT_DIR_NAME = "simtoolreal"
OBJECT_XML_NAME = "hammer_single.xml"

# Static frictions, set once at asset creation (simtoolreal_env_cfg.py:98-101).
ROBOT_FRICTION = 0.5
FINGERTIP_FRICTION = 1.5
OBJECT_FRICTION = 0.5
TABLE_FRICTION = 0.5

# Table geometry from assets/urdf/table_narrow.urdf (box 0.475 x 0.4 x 0.3) and
# ResetCfg.table_reset_z (simtoolreal_env_cfg.py:378). Surface lands at
# 0.38 + 0.15 = 0.53 m.
TABLE_BOX_SIZE = (0.475, 0.4, 0.3)
TABLE_RESET_Z = 0.38
TABLE_OBJECT_Z_OFFSET = 0.25

# Single hammer drawn from the source pool generator with seed=42 and
# shuffle=False, i.e. pool[0] of the first matching ObjectSizeDistribution
# (the cuboid hammer, object_size_distributions.py:88-99). Reproduce with:
#   generate_handle_head_urdfs(("hammer",...), 100, seed=42, shuffle=False)
HAMMER_HANDLE_SIZE = (0.24630474692314314, 0.021682799299900978, 0.017424430711419206)
HAMMER_HEAD_SIZE = (0.0479264685607898, 0.08752674564408842, 0.03238110465145311)
HAMMER_HANDLE_DENSITY = 412.36203565420874
HAMMER_HEAD_DENSITY = 837.7150228240811

# Isaac Gym enables all robot self-collisions, then masks these adjacent pairs
# (adjacent_links.py:LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS). MuJoCo already
# skips pairs joined by a joint; the explicit excludes cover the rest.
ADJACENT_LINKS: dict[str, tuple[str, ...]] = {
    "iiwa14_link_0": ("iiwa14_link_1",),
    "iiwa14_link_1": ("iiwa14_link_0", "iiwa14_link_2"),
    "iiwa14_link_2": ("iiwa14_link_1", "iiwa14_link_3"),
    "iiwa14_link_3": ("iiwa14_link_2", "iiwa14_link_4"),
    "iiwa14_link_4": ("iiwa14_link_3", "iiwa14_link_5"),
    "iiwa14_link_5": ("iiwa14_link_4", "iiwa14_link_6"),
    "iiwa14_link_6": ("iiwa14_link_5", "iiwa14_link_7"),
    "iiwa14_link_7": (
        "iiwa14_link_6",
        "left_thumb_CMC_VL",
        "left_thumb_MC",
        "left_index_MCP_VL",
        "left_index_PP",
        "left_middle_MCP_VL",
        "left_middle_PP",
        "left_ring_MCP_VL",
        "left_ring_PP",
        "left_pinky_MC",
    ),
    "left_index_MCP_VL": ("iiwa14_link_7", "left_index_PP"),
    "left_index_PP": ("iiwa14_link_7", "left_index_MCP_VL", "left_index_MP"),
    "left_index_MP": ("left_index_PP", "left_index_DP"),
    "left_index_DP": ("left_index_MP",),
    "left_middle_MCP_VL": ("iiwa14_link_7", "left_middle_PP"),
    "left_middle_PP": ("iiwa14_link_7", "left_middle_MCP_VL", "left_middle_MP"),
    "left_middle_MP": ("left_middle_PP", "left_middle_DP"),
    "left_middle_DP": ("left_middle_MP",),
    "left_pinky_MC": ("iiwa14_link_7", "left_pinky_MCP_VL", "left_pinky_PP"),
    "left_pinky_MCP_VL": ("left_pinky_MC", "left_pinky_PP"),
    "left_pinky_PP": ("left_pinky_MC", "left_pinky_MCP_VL", "left_pinky_MP"),
    "left_pinky_MP": ("left_pinky_PP", "left_pinky_DP"),
    "left_pinky_DP": ("left_pinky_MP",),
    "left_ring_MCP_VL": ("iiwa14_link_7", "left_ring_PP"),
    "left_ring_PP": ("iiwa14_link_7", "left_ring_MCP_VL", "left_ring_MP"),
    "left_ring_MP": ("left_ring_PP", "left_ring_DP"),
    "left_ring_DP": ("left_ring_MP",),
    "left_thumb_CMC_VL": ("iiwa14_link_7", "left_thumb_MC"),
    "left_thumb_MC": (
        "iiwa14_link_7",
        "left_thumb_CMC_VL",
        "left_thumb_MCP_VL",
        "left_thumb_PP",
    ),
    "left_thumb_MCP_VL": ("left_thumb_MC", "left_thumb_PP"),
    "left_thumb_PP": ("left_thumb_MC", "left_thumb_MCP_VL", "left_thumb_DP"),
    "left_thumb_DP": ("left_thumb_PP",),
}


def _stl_is_ascii(path: Path) -> bool:
    """Return True when an STL file is ASCII rather than binary.

    A binary STL's 80-byte header sometimes begins with ``solid``, so the header
    text alone is not a reliable discriminator. The size check is: a binary file
    is exactly ``84 + 50 * n_triangles`` bytes.

    Args:
        path: STL file to classify.

    Returns:
        True if the file should be parsed as ASCII STL.
    """
    size = path.stat().st_size
    with open(path, "rb") as handle:
        header = handle.read(80)
        count_bytes = handle.read(4)
    if len(count_bytes) < 4:
        return True
    n_tri = struct.unpack("<I", count_bytes)[0]
    if size == 84 + 50 * n_tri:
        return False
    return header[:5].lower() == b"solid"


def _convert_ascii_stl_to_binary(src: Path, dst: Path) -> int:
    """Rewrite an ASCII STL as binary STL, which is all MuJoCo reads.

    Args:
        src: ASCII STL source path.
        dst: Destination path for the binary STL.

    Returns:
        Number of triangles written.

    Raises:
        ValueError: If the source contains no triangles or a partial triangle.
    """
    text = src.read_text(errors="replace")
    coords: list[tuple[float, float, float]] = []
    for token in text.split("vertex")[1:]:
        parts = token.split()[:3]
        if len(parts) < 3:
            raise ValueError(f"Malformed vertex record in {src}")
        coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not coords or len(coords) % 3 != 0:
        raise ValueError(f"{src} has {len(coords)} vertices, not a whole number of triangles")

    with open(dst, "wb") as handle:
        handle.write(b"\0" * 80)
        handle.write(struct.pack("<I", len(coords) // 3))
        for i in range(0, len(coords), 3):
            tri = np.asarray(coords[i : i + 3], dtype=np.float64)
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            norm = float(np.linalg.norm(normal))
            normal = normal / norm if norm > 0.0 else np.zeros(3)
            handle.write(struct.pack("<3f", *normal.astype(np.float32)))
            for vertex in tri:
                handle.write(struct.pack("<3f", *vertex.astype(np.float32)))
            handle.write(struct.pack("<H", 0))
    return len(coords) // 3


def _stage_meshes(urdf_dir: Path, mesh_root: Path) -> int:
    """Copy the robot meshes into the asset tree, converting ASCII STL to binary.

    Directory structure is preserved so the URDF's relative ``filename``
    references keep resolving against ``meshdir``.

    Args:
        urdf_dir: Directory holding the source URDF and its mesh folders.
        mesh_root: Destination mesh root (``<robot dir>/assets``).

    Returns:
        Count of files converted from ASCII to binary.
    """
    if mesh_root.exists():
        shutil.rmtree(mesh_root)
    converted = 0
    for subdir in MESH_SUBDIRS:
        for src in sorted((urdf_dir / subdir).rglob("*")):
            if not src.is_file():
                continue
            dst = mesh_root / src.relative_to(urdf_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix.lower() == ".stl" and _stl_is_ascii(src):
                n_tri = _convert_ascii_stl_to_binary(src, dst)
                converted += 1
                print(f"  ascii->binary STL: {src.name} ({n_tri} triangles)")
            else:
                shutil.copyfile(src, dst)
    return converted


def _write_staging_urdf(urdf_src: Path, mesh_root: Path, staging_urdf: Path) -> None:
    """Copy the URDF next to the staged meshes and add MuJoCo compiler hints.

    MuJoCo reads a ``<mujoco><compiler .../></mujoco>`` extension block inside
    URDF. ``balanceinertia`` keeps marginally invalid inertia tensors from failing
    the compile, and ``discardvisual="false"`` preserves the visual geoms.

    Args:
        urdf_src: Source URDF path in the SimToolReal checkout.
        mesh_root: Directory the staged meshes were written to.
        staging_urdf: Destination path for the augmented URDF.
    """
    tree = ET.parse(urdf_src)
    root = tree.getroot()
    mujoco_ext = ET.Element("mujoco")
    ET.SubElement(
        mujoco_ext,
        "compiler",
        {
            "meshdir": str(mesh_root),
            "discardvisual": "false",
            "balanceinertia": "true",
            "strippath": "false",
            "angle": "radian",
            "fusestatic": "false",
        },
    )
    root.insert(0, mujoco_ext)
    staging_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(staging_urdf, encoding="unicode")


def _compile_urdf_to_mjcf(staging_urdf: Path, out_xml: Path) -> None:
    """Compile a URDF through MuJoCo and save the result as MJCF.

    Args:
        staging_urdf: URDF augmented by :func:`_write_staging_urdf`.
        out_xml: Destination MJCF path.

    Raises:
        RuntimeError: If MuJoCo cannot compile the URDF.
    """
    import mujoco

    mujoco_api: Any = mujoco

    try:
        model = mujoco_api.MjModel.from_xml_path(str(staging_urdf))
    except Exception as exc:  # pragma: no cover - surfaced to the operator
        raise RuntimeError(f"MuJoCo failed to compile {staging_urdf}: {exc}") from exc
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    mujoco_api.mj_saveLastXML(str(out_xml), model)


def _urdf_joint_efforts(urdf_src: Path) -> dict[str, float]:
    """Read per-joint effort limits from the URDF.

    Isaac Gym writes ``effort`` for the arm only and leaves the hand at its URDF
    value (isaacgymenvs/tasks/simtoolreal/utils.py:102); the URDF arm effort is
    already 300, so taking every value straight from the URDF matches both paths.

    Args:
        urdf_src: Source URDF path.

    Returns:
        Mapping of revolute joint name to effort limit.
    """
    root = ET.parse(urdf_src).getroot()
    efforts: dict[str, float] = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        name = joint.get("name")
        if limit is not None and name is not None and limit.get("effort") is not None:
            efforts[name] = float(str(limit.get("effort")))
    return efforts


def _iter_bodies(root: ET.Element) -> Iterable[ET.Element]:
    """Yield every ``<body>`` element in an MJCF tree."""
    return root.iter("body")


def _welded_subtree(body: ET.Element) -> list[ET.Element]:
    """Return ``body`` plus descendants attached to it without a joint.

    Isaac imports the robot with ``collapse_fixed_joints=True``, which merges
    fixed-joint children into their parent link, so the fingertip elastomer pads
    inherit the fingertip material. MuJoCo keeps them as separate welded bodies,
    so the friction override has to walk the welded subtree explicitly.

    Args:
        body: Root of the subtree.

    Returns:
        List containing ``body`` and every jointless descendant.
    """
    collected = [body]
    for child in body.findall("body"):
        if child.find("joint") is None and child.find("freejoint") is None:
            collected.extend(_welded_subtree(child))
    return collected


def _set_geom_friction(body: ET.Element, sliding: float) -> int:
    """Override the sliding friction of every collision geom on one body.

    Visual-only geoms (``contype=0`` and ``conaffinity=0``) are skipped: they
    never contact anything, and leaving them untouched keeps the friction audit
    unambiguous. MuJoCo's torsional and rolling coefficients keep their defaults;
    only the sliding term is ported, since that is all Isaac's
    ``rigid_shape_props.friction`` sets.

    Args:
        body: Body whose geoms are rewritten.
        sliding: Sliding friction coefficient to apply.

    Returns:
        Number of geoms modified.
    """
    count = 0
    for geom in body.findall("geom"):
        if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
            continue
        existing = (geom.get("friction") or "").split()
        torsional = existing[1] if len(existing) > 1 else "0.005"
        rolling = existing[2] if len(existing) > 2 else "0.0001"
        geom.set("friction", f"{sliding:g} {torsional} {rolling}")
        count += 1
    return count


def _postprocess_robot_xml(xml_path: Path, urdf_src: Path) -> dict[str, int]:
    """Apply the SimToolReal physics parameters to a freshly compiled robot MJCF.

    Steps, each mirroring a specific source behaviour:

    1. Root body pose ``(0, 0.8, 0)`` with identity orientation, fixed base
       (scene_utils.py:163-164; no free joint is added).
    2. ``gravcomp="1"`` on every robot body — Isaac sets
       ``disable_gravity=True`` on the whole articulation
       (isaacgymenvs .../env.py:1790).
    3. Per-joint ``armature``/``frictionloss`` for the hand only, and
       ``damping="0"`` everywhere: the source's damping numbers are PD gains on
       Isaac's ``ImplicitActuator``, so they belong on the actuator's ``kv``, not
       on passive joint damping. The arm intentionally gets no armature and no
       friction (isaacgymenvs .../utils.py:100-101).
    4. One position actuator per canonical joint, in canonical order, with
       ``kp``/``kv`` from the stiffness/damping tables, ``inheritrange="1"`` so
       the ctrl range tracks the joint range, and ``forcerange`` from the URDF
       effort limit.
    5. Sliding friction ``0.5`` on robot geoms, ``1.5`` on the five fingertip
       links and their welded pads (scene_utils.py:1584).
    6. ``<exclude>`` pairs for the adjacent links Isaac masks.
    7. ``integrator="implicitfast"``, which is what MuJoCo recommends for stiff
       position actuators.

    Args:
        xml_path: MJCF file to rewrite in place.
        urdf_src: Source URDF, read for effort limits.

    Returns:
        Counters describing what was touched, for the build log.

    Raises:
        ValueError: If the compiled model is missing an expected joint or body.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    root.set("model", "kuka_sharpa")

    stats = {"joints": 0, "fingertip_geoms": 0, "robot_geoms": 0, "excludes": 0}

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    # The staging URDF needs an absolute meshdir to compile, and mj_saveLastXML
    # writes that absolute path straight into the MJCF. Rewrite it relative to
    # the XML's own directory so the asset stays machine-independent, matching
    # every other robot in assets/robots/.
    compiler.set("meshdir", "assets")

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(list(root).index(compiler) + 1, option)
    option.set("integrator", "implicitfast")

    bodies = {body.get("name"): body for body in _iter_bodies(root)}

    # 1. Root pose + fixed base.
    root_body = bodies.get("iiwa14_link_0")
    if root_body is None:
        raise ValueError("Compiled MJCF has no 'iiwa14_link_0' body")
    root_body.set("pos", " ".join(f"{v:g}" for v in ROBOT_ROOT_POS))
    root_body.set("quat", " ".join(f"{v:g}" for v in ROBOT_ROOT_QUAT_WXYZ))
    if root_body.find("freejoint") is not None:
        raise ValueError("Robot root must stay fixed-base; found a free joint")

    # 2. Gravity compensation on every robot body.
    for body in _iter_bodies(root):
        body.set("gravcomp", "1")

    # 3. Per-joint dynamics.
    joints = {
        joint.get("name"): joint for body in _iter_bodies(root) for joint in body.findall("joint")
    }
    for name in JOINT_NAMES_CANONICAL:
        joint = joints.get(name)
        if joint is None:
            raise ValueError(f"Compiled MJCF is missing joint '{name}'")
        joint.set("damping", "0")
        if name in HAND_JOINT_ARMATURE:
            joint.set("armature", repr(HAND_JOINT_ARMATURE[name]))
            joint.set("frictionloss", repr(HAND_JOINT_FRICTION[name]))
        else:
            joint.set("armature", "0")
            joint.set("frictionloss", "0")
        stats["joints"] += 1
    # 4. One position actuator per canonical joint, in canonical order.
    for existing in root.findall("actuator"):
        root.remove(existing)
    actuator_root = ET.SubElement(root, "actuator")
    efforts = _urdf_joint_efforts(urdf_src)
    for name in JOINT_NAMES_CANONICAL:
        stiffness = ARM_JOINT_STIFFNESS.get(name, HAND_JOINT_STIFFNESS.get(name))
        damping = ARM_JOINT_DAMPING.get(name, HAND_JOINT_DAMPING.get(name))
        if stiffness is None or damping is None:
            raise ValueError(f"No PD gains defined for joint '{name}'")
        attrs = {
            "name": f"{name}_ctrl",
            "joint": name,
            "kp": repr(stiffness),
            "kv": repr(damping),
            "inheritrange": "1",
        }
        effort = efforts.get(name)
        if effort is not None:
            attrs["forcerange"] = f"{-effort:g} {effort:g}"
        ET.SubElement(actuator_root, "position", attrs)

    # 5. Geom friction: fingertips (and their welded pads) high, rest low.
    fingertip_bodies: set[int] = set()
    for link in FINGERTIP_LINK_NAMES:
        fingertip_body = bodies.get(link)
        if fingertip_body is None:
            raise ValueError(f"Compiled MJCF is missing fingertip body '{link}'")
        for member in _welded_subtree(fingertip_body):
            fingertip_bodies.add(id(member))
            stats["fingertip_geoms"] += _set_geom_friction(member, FINGERTIP_FRICTION)
    for body in _iter_bodies(root):
        if id(body) in fingertip_bodies:
            continue
        stats["robot_geoms"] += _set_geom_friction(body, ROBOT_FRICTION)

    # 6. Adjacent-link contact excludes.
    for existing_contact in root.findall("contact"):
        root.remove(existing_contact)
    contact_root = ET.SubElement(root, "contact")
    emitted: set[tuple[str, str]] = set()
    for link, neighbors in ADJACENT_LINKS.items():
        if link not in bodies:
            continue
        for neighbor in neighbors:
            if neighbor not in bodies:
                continue
            pair = tuple(sorted((link, neighbor)))
            if pair in emitted:
                continue
            emitted.add(pair)  # type: ignore[arg-type]
            ET.SubElement(
                contact_root,
                "exclude",
                {"name": f"exclude_{pair[0]}__{pair[1]}", "body1": pair[0], "body2": pair[1]},
            )
            stats["excludes"] += 1

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="unicode")
    return stats


def _format_floats(values: Sequence[float] | np.ndarray) -> str:
    """Render a float sequence as a space-separated MJCF attribute value."""
    return " ".join(f"{float(v):.17g}" for v in values)


def _box_inertia(size: Sequence[float], density: float) -> tuple[float, float, float, float]:
    """Return ``(mass, ixx, iyy, izz)`` for a solid box.

    Matches ``_compute_mass_and_inertia`` in the source generator
    (generate_objects.py:133-145).

    Args:
        size: Full box extents ``(lx, ly, lz)``.
        density: Uniform density in kg/m^3.

    Returns:
        Mass and the three principal inertia components about the box centroid.
    """
    lx, ly, lz = (float(v) for v in size)
    mass = lx * ly * lz * float(density)
    ixx = (1.0 / 12.0) * mass * (ly * ly + lz * lz)
    iyy = (1.0 / 12.0) * mass * (lx * lx + lz * lz)
    izz = (1.0 / 12.0) * mass * (lx * lx + ly * ly)
    return mass, ixx, iyy, izz


def _write_hammer_xml(out_xml: Path) -> dict[str, float]:
    """Write the single fixed handle+head object as an includable MJCF fragment.

    Geometry, densities, and the composite inertia reproduce the source's
    ``_handle_head_urdf_variable_density`` (generate_objects.py:166-259): two
    boxes with the head offset along ``+x``, one link, and a parallel-axis
    shifted inertia about the composite centre of mass.

    Args:
        out_xml: Destination path for ``hammer_single.xml``.

    Returns:
        The derived mass properties, for the build log.
    """
    handle_mass, handle_ixx, handle_iyy, handle_izz = _box_inertia(
        HAMMER_HANDLE_SIZE, HAMMER_HANDLE_DENSITY
    )
    head_mass, head_ixx, head_iyy, head_izz = _box_inertia(HAMMER_HEAD_SIZE, HAMMER_HEAD_DENSITY)

    x_offset = HAMMER_HANDLE_SIZE[0] / 2.0 + HAMMER_HEAD_SIZE[0] / 2.0
    total_mass = handle_mass + head_mass
    com_x = head_mass * x_offset / total_mass
    d_handle = -com_x
    d_head = x_offset - com_x

    ixx = handle_ixx + head_ixx
    iyy = (handle_iyy + handle_mass * d_handle**2) + (head_iyy + head_mass * d_head**2)
    izz = (handle_izz + handle_mass * d_handle**2) + (head_izz + head_mass * d_head**2)

    handle_half = [v / 2.0 for v in HAMMER_HANDLE_SIZE]
    head_half = [v / 2.0 for v in HAMMER_HEAD_SIZE]
    friction = f"{OBJECT_FRICTION:g} 0.005 0.0001"
    object_z = TABLE_RESET_Z + TABLE_OBJECT_Z_OFFSET

    xml = f"""<mujoco model="simtoolreal_hammer_single">
  <!--
    Single handle+head object (cuboid hammer). Values come from the SimToolReal
    pool generator with seed=42, shuffle=False, i.e. pool[0] of the first
    matching ObjectSizeDistribution (object_size_distributions.py:88-99):
      handle_scale   = {HAMMER_HANDLE_SIZE}
      head_scale     = {HAMMER_HEAD_SIZE}
      handle_density = {HAMMER_HANDLE_DENSITY}
      head_density   = {HAMMER_HEAD_DENSITY}
    Mass/inertia follow generate_objects.py:219-231 (parallel-axis shift to the
    composite COM). Only the handle bbox feeds the `phi` observation scale
    (migration guide section 2), which is why the handle geom is named
    `object_handle` and read back via get_geom_size().

    Task T8 replaces this single object with the 1200-entry pool.
  -->
  <worldbody>
    <body name="object" pos="0 0 {object_z:g}">
      <freejoint name="object_joint"/>
      <inertial pos="{com_x:.17g} 0 0" mass="{total_mass:.17g}"
        diaginertia="{ixx:.17g} {iyy:.17g} {izz:.17g}"/>
      <geom name="object_handle" type="box" size="{_format_floats(handle_half)}"
        friction="{friction}" rgba="0.55 0.27 0.07 1.0"/>
      <geom name="object_head" type="box" size="{_format_floats(head_half)}"
        pos="{x_offset:.17g} 0 0" friction="{friction}" rgba="0.5 0.5 0.5 1.0"/>
    </body>
  </worldbody>
</mujoco>
"""
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_xml.write_text(xml)
    return {
        "handle_mass": handle_mass,
        "head_mass": head_mass,
        "total_mass": total_mass,
        "com_x": com_x,
    }


def _scene_xml_text(robot_include: str, object_include: str, keyframe: str | None) -> str:
    """Render the composed scene MJCF.

    Include order fixes the state layout: the robot's 29 hinges land on
    ``qpos[0:29]`` and the object free joint on ``qpos[29:36]``. The floor and
    table are static (no joints), so they do not shift the layout.
    ``SimToolRealEnv._build_state_layout`` asserts this.

    Args:
        robot_include: Include path for the robot MJCF.
        object_include: Include path for the object MJCF.
        keyframe: Rendered ``<keyframe>`` block, or ``None`` on the first pass
            (the keyframe needs the compiled joint order).

    Returns:
        The scene XML text.
    """
    table_half = _format_floats([v / 2.0 for v in TABLE_BOX_SIZE])
    keyframe_block = f"\n{keyframe}\n" if keyframe else "\n"
    return f"""<mujoco model="simtoolreal_scene">
  <!--
    SimToolReal scene: KUKA iiwa14 + Sharpa left hand, narrow table, one
    handle+head object. Generated by unilab.tools.build_simtoolreal_assets;
    edit that generator rather than this file.

    qpos layout (nq = 29 robot + 7 object = 36):
      [0:29]   robot hinge joints, canonical order
      [29:32]  object position xyz
      [32:36]  object quaternion wxyz
    Include order below is what guarantees that layout.

    The table is welded to the world, matching Isaac's kinematic_enabled=True +
    disable_gravity=True table (scene_utils.py:1751-1758). Per-env table pose
    randomization (ResetCfg.table_reset_z_range, default +/-0.01) has no
    mechanism under the backend whitelist, since UniLab exposes no
    set_body_pose; see the T0 report.
  -->
  <include file="{robot_include}"/>

  <statistic center="0 0.3 0.5" extent="1.5"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.2 0.2 0.2" specular="0.9 0.9 0.9"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
      width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 2.0" dir="0 0 -1" directional="true"/>
    <geom name="floor" pos="0 0 0" size="0 0 0.05" type="plane"
      material="groundplane" friction="{TABLE_FRICTION:g} 0.005 0.0001"/>

    <!-- Static table. Box 0.475 x 0.4 x 0.3 from assets/urdf/table_narrow.urdf,
         centred at ResetCfg.table_reset_z = 0.38, so the surface sits at 0.53. -->
    <body name="table" pos="0 0 {TABLE_RESET_Z:g}">
      <geom name="table_box" type="box" size="{table_half}"
        friction="{TABLE_FRICTION:g} 0.005 0.0001" rgba="0.82 0.56 0.35 1.0"/>
    </body>
  </worldbody>

  <include file="{object_include}"/>
{keyframe_block}</mujoco>
"""


def _render_keyframe(scene_xml: Path) -> str:
    """Compile the scene and render a ``home`` keyframe for the default pose.

    Reading the joint order back from the compiled model, rather than assuming
    it, means the keyframe stays correct even if the URDF's joint order changes.

    Args:
        scene_xml: Scene file to compile.

    Returns:
        The ``<keyframe>`` XML block.

    Raises:
        RuntimeError: If the scene does not compile.
        ValueError: If the compiled model's joint or actuator layout is not the
            expected 29 hinges plus one object free joint.
    """
    import mujoco

    mujoco_api: Any = mujoco

    try:
        model = mujoco_api.MjModel.from_xml_path(str(scene_xml))
    except Exception as exc:  # pragma: no cover - surfaced to the operator
        raise RuntimeError(f"scene {scene_xml} failed to compile: {exc}") from exc

    if model.nu != NUM_JOINTS:
        raise ValueError(f"Scene must expose {NUM_JOINTS} actuators, got {model.nu}")

    qpos = np.zeros((model.nq,), dtype=np.float64)
    for joint_name, value in DEFAULT_JOINT_POS.items():
        jid = mujoco_api.mj_name2id(model, mujoco_api.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"Compiled scene is missing joint '{joint_name}'")
        qpos[model.jnt_qposadr[jid]] = value

    object_jid = mujoco_api.mj_name2id(model, mujoco_api.mjtObj.mjOBJ_JOINT, "object_joint")
    if object_jid < 0:
        raise ValueError("Compiled scene is missing the 'object_joint' free joint")
    obj_adr = int(model.jnt_qposadr[object_jid])
    object_z = TABLE_RESET_Z + TABLE_OBJECT_Z_OFFSET
    qpos[obj_adr : obj_adr + 7] = [0.0, 0.0, object_z, 1.0, 0.0, 0.0, 0.0]

    # Position actuators hold the default pose, so ctrl mirrors qpos per joint.
    ctrl = np.zeros((model.nu,), dtype=np.float64)
    for act_id in range(model.nu):
        jid = int(model.actuator_trnid[act_id, 0])
        ctrl[act_id] = qpos[model.jnt_qposadr[jid]]

    return (
        "  <keyframe>\n"
        f'    <key name="home" qpos="{_format_floats(qpos)}"\n'
        f'      ctrl="{_format_floats(ctrl)}"/>\n'
        "  </keyframe>"
    )


def build(simtoolreal_root: Path, assets_root: Path = ASSETS_ROOT_PATH) -> dict[str, Path]:
    """Build every SimToolReal MJCF asset.

    Args:
        simtoolreal_root: Root of the SimToolReal checkout holding
            ``assets/urdf/kuka_sharpa_description``.
        assets_root: UniLab assets root to write into.

    Returns:
        Mapping of artifact name to the path written.

    Raises:
        FileNotFoundError: If the source URDF is missing.
    """
    urdf_src = simtoolreal_root / ROBOT_URDF_REL
    if not urdf_src.is_file():
        raise FileNotFoundError(f"SimToolReal robot URDF not found: {urdf_src}")

    robot_dir = assets_root / "robots" / ROBOT_DIR_NAME
    object_dir = assets_root / "objects" / OBJECT_DIR_NAME
    robot_dir.mkdir(parents=True, exist_ok=True)
    object_dir.mkdir(parents=True, exist_ok=True)

    mesh_root = robot_dir / "assets"
    robot_xml = robot_dir / ROBOT_XML_NAME
    scene_xml = robot_dir / SCENE_XML_NAME
    object_xml = object_dir / OBJECT_XML_NAME

    print(f"[simtoolreal-assets] staging meshes -> {mesh_root}")
    converted = _stage_meshes(urdf_src.parent, mesh_root)
    print(f"[simtoolreal-assets] staged meshes ({converted} ASCII STL converted)")

    staging_urdf = mesh_root.parent / "_staging_kuka_sharpa.urdf"
    _write_staging_urdf(urdf_src, mesh_root, staging_urdf)
    print("[simtoolreal-assets] compiling URDF through MuJoCo")
    _compile_urdf_to_mjcf(staging_urdf, robot_xml)
    staging_urdf.unlink()

    stats = _postprocess_robot_xml(robot_xml, urdf_src)
    print(
        "[simtoolreal-assets] robot MJCF: "
        f"{stats['joints']} joints, {stats['fingertip_geoms']} fingertip geoms @ "
        f"{FINGERTIP_FRICTION}, {stats['robot_geoms']} other geoms @ {ROBOT_FRICTION}, "
        f"{stats['excludes']} contact excludes"
    )

    mass_info = _write_hammer_xml(object_xml)
    print(
        "[simtoolreal-assets] object MJCF: "
        f"mass={mass_info['total_mass']:.6f} kg (handle {mass_info['handle_mass']:.6f} + "
        f"head {mass_info['head_mass']:.6f}), com_x={mass_info['com_x']:.6f}"
    )

    robot_include = robot_xml.name
    object_include = str(Path("..") / ".." / "objects" / OBJECT_DIR_NAME / object_xml.name)

    # First pass without a keyframe, so the joint order can be read back from the
    # compiled model; second pass bakes the keyframe in.
    scene_xml.write_text(_scene_xml_text(robot_include, object_include, keyframe=None))
    keyframe = _render_keyframe(scene_xml)
    scene_xml.write_text(_scene_xml_text(robot_include, object_include, keyframe=keyframe))
    _render_keyframe(scene_xml)  # re-compile as a self-check
    print(f"[simtoolreal-assets] scene MJCF verified -> {scene_xml}")

    return {"robot": robot_xml, "scene": scene_xml, "object": object_xml, "meshes": mesh_root}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simtoolreal-root",
        default="~/code/simtoolreal/simtoolreal",
        help="Root of the SimToolReal checkout (default: %(default)s).",
    )
    parser.add_argument(
        "--assets-root",
        default=str(ASSETS_ROOT_PATH),
        help="UniLab assets root to write into (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    try:
        written = build(
            Path(args.simtoolreal_root).expanduser().resolve(),
            Path(args.assets_root).expanduser().resolve(),
        )
    except Exception as exc:
        print(f"[simtoolreal-assets] error: {exc}", file=sys.stderr)
        return 1

    for name, path in written.items():
        print(f"[simtoolreal-assets] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
