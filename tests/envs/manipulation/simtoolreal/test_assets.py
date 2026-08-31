from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ASSET_ROOT = (
    Path(__file__).resolve().parents[4] / "src" / "unilab" / "assets" / "robots" / "kuka_sharpa"
)
MESH_ROOT = ASSET_ROOT / "assets"
EXPECTED_MESHES = {
    *(f"new_iiwa14_meshes/collision/link_{i}.stl" for i in range(8)),
    *(f"new_iiwa14_meshes/visual/link_{i}.stl" for i in range(8)),
    "left_sharpa_meshes/left_hand_C_MC_visual.STL",
    "left_sharpa_meshes/left_thumb_CMC_VL.STL",
    "left_sharpa_meshes/left_thumb_MC_visual.STL",
    "left_sharpa_meshes/left_thumb_MCP_VL_visual.STL",
    "left_sharpa_meshes/left_thumb_PP_visual.STL",
    "left_sharpa_meshes/left_thumb_DP_visual.STL",
    "left_sharpa_meshes/thumb_elastomer_surface.STL",
    "left_sharpa_meshes/left_MCP_VL_visual.STL",
    "left_sharpa_meshes/left_PP_visual.STL",
    "left_sharpa_meshes/left_MP_visual.STL",
    "left_sharpa_meshes/left_DP_visual.STL",
    "left_sharpa_meshes/elastomer_surface.STL",
    "left_sharpa_meshes/left_pinky_MC_visual.STL",
    "left_sharpa_meshes/left_pinky_MC.STL",
    *(f"menagerie_sharpa_wave/palm/palm{i:03d}.obj" for i in range(32)),
}
EXPECTED_ANCILLARY = {
    "menagerie_sharpa_wave/LICENSE",
    "menagerie_sharpa_wave/SOURCE.md",
}
RETIRED_COLLISION_MESHES = {
    "left_hand_C_MC.STL",
    "left_thumb_MC.STL",
    "left_thumb_PP.STL",
    "left_thumb_DP.STL",
    "thumb_elastomer.STL",
    "MCP_VL.STL",
    "left_PP.STL",
    "left_MP.STL",
    "left_DP.STL",
    "elastomer.STL",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_asset_inventory_is_closed() -> None:
    provenance = json.loads((ASSET_ROOT / "ASSET_PROVENANCE").read_text(encoding="utf-8"))
    assert provenance["schema_version"] == 1
    robot_xml = ASSET_ROOT / "kuka_sharpa.xml"
    scene_xml = ASSET_ROOT / "scene.xml"
    assert robot_xml.is_file()
    assert scene_xml.is_file()
    assert "unilab.tools.build_simtoolreal_assets" not in scene_xml.read_text(encoding="utf-8")
    refs = [element.attrib["file"] for element in ET.parse(robot_xml).iter("mesh")]
    assert len(refs) == 62
    assert len(set(refs)) == 62
    assert set(refs) == EXPECTED_MESHES
    final_files = {
        path.relative_to(MESH_ROOT).as_posix() for path in MESH_ROOT.rglob("*") if path.is_file()
    }
    assert final_files == EXPECTED_MESHES | EXPECTED_ANCILLARY
    for ref in refs:
        candidate = MESH_ROOT / ref
        assert candidate.is_file() and not candidate.is_symlink()
        path = candidate.resolve()
        assert MESH_ROOT.resolve() in path.parents
    assert not any(
        (MESH_ROOT / "left_sharpa_meshes" / name).exists() for name in RETIRED_COLLISION_MESHES
    )
    assert not list(ET.parse(robot_xml).iter("keyframe"))
    assert len(list(ET.parse(scene_xml).iter("keyframe"))) == 1
    assert provenance["mesh_census"] == {"total": 62, "byte_identical": 61, "different": 1}
    assert {entry["target"] for entry in provenance["meshes"]} == EXPECTED_MESHES
    for entry in provenance["meshes"]:
        target = MESH_ROOT / entry["target"]
        assert _sha256(target) == entry["target_sha256"]
    assert _sha256(robot_xml) == provenance["xml"]["robot"]["target_sha256"]
    assert _sha256(scene_xml) == provenance["xml"]["scene"]["target_sha256"]
    assert _sha256(ASSET_ROOT / "LICENSE.simtoolreal") == provenance["licenses"][0]["sha256"]
    assert _sha256(ASSET_ROOT / "LICENSE.kuka_iiwa") == provenance["licenses"][1]["sha256"]
    menagerie_license = next(
        entry
        for entry in provenance["licenses"]
        if entry["target"] == "assets/menagerie_sharpa_wave/LICENSE"
    )
    assert _sha256(MESH_ROOT / "menagerie_sharpa_wave/LICENSE") == menagerie_license["sha256"]
    ancillary = {entry["target"]: entry for entry in provenance["additional_assets"]}
    for target in EXPECTED_ANCILLARY - {"menagerie_sharpa_wave/LICENSE"}:
        assert _sha256(MESH_ROOT / target) == ancillary[target]["sha256"]
    assert provenance["menagerie_sharpa_wave"]["repository_commit"] == (
        "da76818e269b82289eba39808e2fb91d679d6994"
    )
    assert provenance["menagerie_sharpa_wave"]["sharpa_directory_commit"] == (
        "c1a4eeb85694ae1dffe33ff1797d4e528928a133"
    )
    special = next(
        entry
        for entry in provenance["meshes"]
        if entry["target"] == "left_sharpa_meshes/left_hand_C_MC_visual.STL"
    )
    assert special["donor_blob"] == "4eaa0d5d0d57fb42b50e8e66e91bf3904f9a47fa"
    assert special["source_blob"] == "ec9632db49c79c84e25868a28c2796b350121360"
    assert special["byte_identical"] is False
    assert special["adaptation"] == {
        "donor_owner": (
            "src/unilab/tools/build_simtoolreal_assets.py:_convert_ascii_stl_to_binary"
        ),
        "geometry_preserved": True,
        "kind": "ascii-stl-to-binary-stl",
        "source_size_bytes": 1_254_651,
        "target_size_bytes": 298_284,
        "triangle_count": 5_964,
    }
    assert provenance["xml"]["scene"]["adaptation"] == (
        "comment-only: removed stale reference to non-vendored build_simtoolreal_assets.py"
    )


def test_robot_xml_compiles_with_source_contract() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(ASSET_ROOT / "kuka_sharpa.xml"))
    assert (model.nq, model.nv, model.nu, model.nmesh) == (29, 29, 29, 62)


EXPECTED_JOINTS = (
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
    "left_1_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_2_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_3_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_4_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_5_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
)
EXPECTED_ACTUATORS = tuple(f"{name}_ctrl" for name in EXPECTED_JOINTS)


def _body_geoms(element: ET.Element, body_name: str | None = None):
    for child in element:
        if child.tag == "body":
            current = child.attrib.get("name", body_name)
            yield from _body_geoms(child, current)
        elif child.tag == "geom" and body_name is not None:
            yield body_name, child
        else:
            yield from _body_geoms(child, body_name)


def test_robot_xml_preserves_hand_structure_and_scene_ownership() -> None:
    root = ET.parse(ASSET_ROOT / "kuka_sharpa.xml").getroot()
    assert [joint.attrib["name"] for joint in root.iter("joint")] == list(EXPECTED_JOINTS)
    assert [actuator.attrib["name"] for actuator in root.iter("position")] == list(
        EXPECTED_ACTUATORS
    )
    excludes = [(item.attrib["body1"], item.attrib["body2"]) for item in root.iter("exclude")]
    assert len(excludes) == 50
    assert not list(root.iter("keyframe"))
    assert not list(root.iter("option")), "scene.xml must be the sole MuJoCo option owner"
    forbidden_names = {"floor", "table", "table_box", "object", "goal_object"}
    assert not any(element.attrib.get("name") in forbidden_names for element in root.iter())

    hand_geoms = list(_body_geoms(root))
    hand_geoms = [(name, geom) for name, geom in hand_geoms if name.startswith("left_")]
    assert hand_geoms
    for body_name, geom in hand_geoms:
        contype = geom.attrib.get("contype", "1")
        conaffinity = geom.attrib.get("conaffinity", "1")
        if contype == "0" and conaffinity == "0":
            assert geom.attrib.get("group") == "2", (body_name, geom.attrib)


def test_sharpa_collision_geoms_have_explicit_role_contact_profile() -> None:
    root = ET.parse(ASSET_ROOT / "kuka_sharpa.xml").getroot()
    hand_geoms = [(name, geom) for name, geom in _body_geoms(root) if name.startswith("left_")]
    collisions = [
        (name, geom)
        for name, geom in hand_geoms
        if geom.attrib.get("contype", "1") != "0" or geom.attrib.get("conaffinity", "1") != "0"
    ]
    assert collisions
    for body_name, geom in collisions:
        attrs = geom.attrib
        assert attrs.get("contype") == "1"
        assert attrs.get("conaffinity") == "1"
        assert attrs.get("group") == "3"
        assert float(attrs["density"]) == pytest.approx(0.0)
        assert int(attrs["condim"]) == 3
        np.testing.assert_allclose(
            tuple(float(value) for value in attrs["friction"].split()),
            (1.0, 0.005, 0.0001),
        )
        np.testing.assert_allclose(
            tuple(float(value) for value in attrs["solimp"].split()),
            (0.9, 0.95, 0.001, 0.5, 2.0),
        )
        assert float(attrs["margin"]) == pytest.approx(0.0)
        assert float(attrs["gap"]) == pytest.approx(0.0)
        np.testing.assert_allclose(
            tuple(float(value) for value in attrs["solref"].split()),
            (0.02, 1.0),
            err_msg=f"unexpected solref for {body_name}: {attrs}",
        )
