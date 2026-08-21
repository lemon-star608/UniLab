from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ASSET_ROOT = (
    Path(__file__).resolve().parents[4] / "src" / "unilab" / "assets" / "robots" / "kuka_sharpa"
)
MESH_ROOT = ASSET_ROOT / "assets"
EXPECTED_MESHES = {
    *(f"new_iiwa14_meshes/collision/link_{i}.stl" for i in range(8)),
    *(f"new_iiwa14_meshes/visual/link_{i}.stl" for i in range(8)),
    "left_sharpa_meshes/left_hand_C_MC_visual.STL",
    "left_sharpa_meshes/left_hand_C_MC.STL",
    "left_sharpa_meshes/left_thumb_CMC_VL.STL",
    "left_sharpa_meshes/left_thumb_MC_visual.STL",
    "left_sharpa_meshes/left_thumb_MC.STL",
    "left_sharpa_meshes/left_thumb_MCP_VL_visual.STL",
    "left_sharpa_meshes/left_thumb_PP_visual.STL",
    "left_sharpa_meshes/left_thumb_PP.STL",
    "left_sharpa_meshes/left_thumb_DP_visual.STL",
    "left_sharpa_meshes/left_thumb_DP.STL",
    "left_sharpa_meshes/thumb_elastomer_surface.STL",
    "left_sharpa_meshes/thumb_elastomer.STL",
    "left_sharpa_meshes/left_MCP_VL_visual.STL",
    "left_sharpa_meshes/MCP_VL.STL",
    "left_sharpa_meshes/left_PP_visual.STL",
    "left_sharpa_meshes/left_PP.STL",
    "left_sharpa_meshes/left_MP_visual.STL",
    "left_sharpa_meshes/left_MP.STL",
    "left_sharpa_meshes/left_DP_visual.STL",
    "left_sharpa_meshes/left_DP.STL",
    "left_sharpa_meshes/elastomer_surface.STL",
    "left_sharpa_meshes/elastomer.STL",
    "left_sharpa_meshes/left_pinky_MC_visual.STL",
    "left_sharpa_meshes/left_pinky_MC.STL",
}
PROVENANCE_SHA256 = "ad12eeec35d7e33e8f4a00011aaa56a56b04acaf87af45c84697b3e9b94b4b42"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_asset_inventory_is_closed() -> None:
    provenance = json.loads((ASSET_ROOT / "ASSET_PROVENANCE").read_text(encoding="utf-8"))
    assert _sha256(ASSET_ROOT / "ASSET_PROVENANCE") == PROVENANCE_SHA256
    assert provenance["schema_version"] == 1
    robot_xml = ASSET_ROOT / "kuka_sharpa.xml"
    scene_xml = ASSET_ROOT / "scene.xml"
    assert robot_xml.is_file()
    assert scene_xml.is_file()
    assert "unilab.tools.build_simtoolreal_assets" not in scene_xml.read_text(encoding="utf-8")
    refs = [element.attrib["file"] for element in ET.parse(robot_xml).iter("mesh")]
    assert len(refs) == 40
    assert len(set(refs)) == 40
    assert set(refs) == EXPECTED_MESHES
    assert {
        path.relative_to(MESH_ROOT).as_posix() for path in MESH_ROOT.rglob("*") if path.is_file()
    } == EXPECTED_MESHES
    for ref in refs:
        candidate = MESH_ROOT / ref
        assert candidate.is_file() and not candidate.is_symlink()
        path = candidate.resolve()
        assert MESH_ROOT.resolve() in path.parents
    assert not (MESH_ROOT / "left_sharpa_meshes/left_hand_C_MC_visual_.STL").exists()
    assert not (MESH_ROOT / "left_sharpa_meshes/left_thumb_MC_modified.STL").exists()
    assert not list(ET.parse(robot_xml).iter("keyframe"))
    assert len(list(ET.parse(scene_xml).iter("keyframe"))) == 1
    assert provenance["mesh_census"] == {"total": 40, "byte_identical": 39, "different": 1}
    assert {entry["target"] for entry in provenance["meshes"]} == EXPECTED_MESHES
    for entry in provenance["meshes"]:
        target = MESH_ROOT / entry["target"]
        assert _sha256(target) == entry["target_sha256"]
    assert _sha256(robot_xml) == provenance["xml"]["robot"]["target_sha256"]
    assert _sha256(scene_xml) == provenance["xml"]["scene"]["target_sha256"]
    assert _sha256(ASSET_ROOT / "LICENSE.simtoolreal") == provenance["licenses"][0]["sha256"]
    assert _sha256(ASSET_ROOT / "LICENSE.kuka_iiwa") == provenance["licenses"][1]["sha256"]
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
    assert (model.nq, model.nv, model.nu, model.nmesh) == (29, 29, 29, 40)
