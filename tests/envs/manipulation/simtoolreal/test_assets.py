from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ASSET_ROOT = (
    Path(__file__).resolve().parents[4] / "src" / "unilab" / "assets" / "robots" / "kuka_sharpa"
)
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


def test_ensure_training_assets_uses_robot_hub(monkeypatch, tmp_path: Path) -> None:
    from unilab.envs.manipulation.simtoolreal import assets

    calls: list[tuple[str, str]] = []

    def fake_resolve(directory: str, *, marker: str) -> Path:
        calls.append((directory, marker))
        return tmp_path

    monkeypatch.setattr(assets, "resolve_robot_asset_dir", fake_resolve)
    assert assets.ensure_training_assets() == tmp_path
    assert calls == [("robots/kuka_sharpa/meshes", ".hf_complete_v1")]


def test_ensure_dexbench_assets_uses_shared_hub(monkeypatch, tmp_path: Path) -> None:
    from unilab.envs.manipulation.simtoolreal import assets

    calls: list[str] = []

    def fake_resolve(*, marker: str) -> Path:
        calls.append(marker)
        return tmp_path

    monkeypatch.setattr(assets, "resolve_dexbench_asset_dir", fake_resolve)
    assert assets.ensure_dexbench_assets() == tmp_path
    assert calls == [".hf_complete_v1"]


def test_training_asset_inventory_is_closed() -> None:
    robot_xml = ASSET_ROOT / "kuka_sharpa.xml"
    scene_xml = ASSET_ROOT / "scene.xml"
    assert robot_xml.is_file()
    assert scene_xml.is_file()
    assert "unilab.tools.build_simtoolreal_assets" not in scene_xml.read_text(encoding="utf-8")
    assert ET.parse(robot_xml).getroot().find("compiler").attrib["meshdir"] == "meshes"
    refs = [element.attrib["file"] for element in ET.parse(robot_xml).iter("mesh")]
    assert len(refs) == 62
    assert len(set(refs)) == 62
    assert set(refs) == EXPECTED_MESHES
    assert not list(ET.parse(robot_xml).iter("keyframe"))
    assert len(list(ET.parse(scene_xml).iter("keyframe"))) == 1


def test_training_asset_notices_point_to_hf() -> None:
    notices = (ASSET_ROOT / "ASSET_NOTICES.md").read_text(encoding="utf-8")
    assert "unilabsim/unilab-robots" in notices
    assert "LICENSE.simtoolreal" in notices
    assert "LICENSE.kuka_iiwa" in notices
    assert "Apache-2.0" in notices


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
