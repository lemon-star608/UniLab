from __future__ import annotations

import tempfile
from pathlib import Path

import mujoco
import numpy as np

from unilab.base.backend.mujoco.playback import (
    materialize_visual_playback_model,
    resolve_render_play_model_files,
)


def _write(path: Path, object_geoms: str) -> str:
    path.write_text(
        f"""<mujoco model="{path.stem}">
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="robot"><geom name="robot_visual" type="sphere" size=".02" contype="0" conaffinity="0" rgba="1 0 0 1"/></body>
    <body name="object" pos="0 0 .3"><freejoint/>{object_geoms}</body>
  </worldbody>
</mujoco>""",
        encoding="utf-8",
    )
    return str(path)


def _contact(model):
    result = {}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if (
            name
            and name.startswith("object_")
            and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        ):
            result[name] = (int(model.geom_type[geom_id]), model.geom_size[geom_id].copy())
    return result


def test_generic_visual_materializer_synchronizes_contact_topology(tmp_path):
    visual = _write(
        tmp_path / "visual.xml",
        '<geom name="object_handle" type="box" size=".1 .02 .03"/>'
        '<geom name="object_head" type="box" size=".03 .08 .04" pos=".13 0 0"/>'
        '<geom name="object_decoration" type="sphere" size=".01" contype="0" conaffinity="0"/>',
    )
    variants = {
        "box_box": '<geom name="object_handle" type="box" size=".12 .025 .035"/>'
        '<geom name="object_head" type="box" size=".04 .09 .05" pos=".16 0 0"/>',
        "capsule_box": '<geom name="object_handle_cyl" type="capsule" size=".025 .12" quat=".70710678 0 -.70710678 0"/>'
        '<geom name="object_head" type="box" size=".035 .07 .045" pos=".15 0 0"/>',
        "box_only": '<geom name="object_handle" type="box" size=".09 .04 .02"/>',
    }
    expected_names = {
        "box_box": {"object_handle", "object_head"},
        "capsule_box": {"object_handle_cyl", "object_head"},
        "box_only": {"object_handle"},
    }
    visual_base = mujoco.MjModel.from_xml_path(visual)
    for topology, geoms in variants.items():
        physics_path = _write(tmp_path / f"{topology}.xml", geoms)
        output = tmp_path / f"{topology}.mjb"
        materialize_visual_playback_model(
            visual_model_file=visual,
            visual_base_model=visual_base,
            playback_model=mujoco.MjModel.from_xml_path(physics_path),
            output_path=output,
        )
        model = mujoco.MjModel.from_binary_path(str(output))
        assert set(_contact(model)) == expected_names[topology]
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_visual") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_decoration") >= 0


def test_real_simtoolreal_indexes_keep_assigned_40_mesh_visual_topology():
    from unilab.base import registry
    from unilab.base.registry import ensure_registries
    from unilab.envs.manipulation.simtoolreal.tool_catalog import ALL_TYPES, build_tool_catalog

    ensure_registries()
    env = registry.make("SimToolReal", num_envs=8, sim_backend="mujoco")
    paths: list[Path] = []
    try:
        visual_path = env.get_scene_visual_model_file()
        assert visual_path is not None
        visual = mujoco.MjModel.from_xml_path(visual_path)
        mesh_names = {
            mujoco.mj_id2name(visual, mujoco.mjtObj.mjOBJ_MESH, idx) for idx in range(visual.nmesh)
        }
        catalog = build_tool_catalog(ALL_TYPES, num_per_type=50, seed=42, shuffle=True)
        with tempfile.TemporaryDirectory(prefix="visual-topology-test-") as directory:
            resolved = resolve_render_play_model_files(env, num_envs=8, tmp_dir=directory)
            assert isinstance(resolved, list)
            paths = [Path(path) for path in resolved]
            for index in (0, 1, 7):
                model = mujoco.MjModel.from_binary_path(str(paths[index]))
                assert (model.nq, model.nv, model.nu, model.nmesh) == (36, 35, 29, 40)
                assert {
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, idx)
                    for idx in range(model.nmesh)
                } == mesh_names
                spec = catalog[index]
                contact = _contact(model)
                expected = {
                    "object_handle_cyl" if spec.collision_shape == "capsule" else "object_handle"
                }
                if spec.topology != "box_only":
                    expected.add("object_head")
                assert set(contact) == expected
                handle = next(value for name, value in contact.items() if name != "object_head")
                np.testing.assert_allclose(handle[1], spec.handle_size, rtol=1e-5, atol=1e-8)
                if "object_head" in contact:
                    np.testing.assert_allclose(
                        contact["object_head"][1], spec.head_size, rtol=1e-5, atol=1e-8
                    )
            assert all(path.is_file() for path in paths)
        assert all(not path.exists() for path in paths)
    finally:
        env.close()
