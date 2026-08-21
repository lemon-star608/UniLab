from __future__ import annotations

from pathlib import Path

import mujoco

from unilab.envs.manipulation.simtoolreal.tool_assets import materialize_tool_scenes
from unilab.envs.manipulation.simtoolreal.tool_catalog import build_tool_catalog


def test_representative_shipped_scene_compiles_for_each_topology(tmp_path: Path) -> None:
    scene = Path(__file__).resolve().parents[4] / "src/unilab/assets/robots/kuka_sharpa/scene.xml"
    catalog = build_tool_catalog(
        ("hammer", "marker", "eraser"), num_per_type=1, seed=42, shuffle=False
    )
    representatives = {spec.topology: spec for spec in catalog}
    assert set(representatives) == {"box_box", "capsule_box", "box_only"}
    materialized = materialize_tool_scenes(
        str(scene), tuple(representatives.values()), temp_root=tmp_path
    )
    try:
        for model_file in materialized.model_files:
            model = mujoco.MjModel.from_xml_path(model_file)
            assert (model.nq, model.nv, model.nu, model.nmesh) == (36, 35, 29, 40)
    finally:
        materialized.cleanup.cleanup()
