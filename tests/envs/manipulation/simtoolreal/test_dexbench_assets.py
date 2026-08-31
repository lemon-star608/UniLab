from functools import partial
from pathlib import Path

import mujoco
import numpy as np
import yourdfpy

from unilab.envs.manipulation.simtoolreal.config import SimToolRealCfg
from unilab.envs.manipulation.simtoolreal.dexbench_assets import (
    DEXTOOLBENCH_DATA_STRUCTURE,
    materialize_dexbench_scene,
    resolve_dexbench_task,
    validate_manifest,
)
from unilab.envs.manipulation.simtoolreal.env import SimToolRealEnv

SOURCE_ROOT = Path("/home/user/ws/lemon/simtoolreal")
MANIFEST = Path(__file__).resolve().parents[4] / "src/unilab/assets/dexbench/manifest.json"


def test_checked_in_manifest_has_complete_dexbench_package() -> None:
    validate_manifest(MANIFEST)
    payload = __import__("json").loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(payload["objects"]) == 12
    assert len(payload["tasks"]) == 24
    assert {record["category"] for record in payload["objects"]} == set(DEXTOOLBENCH_DATA_STRUCTURE)
    claw = next(record for record in payload["objects"] if record["id"] == "hammer/claw_hammer")
    assert "objects/hammer/claw_hammer/material.mtl" in claw["sources"]
    assert "objects/hammer/claw_hammer/material_0.png" in claw["sources"]


def test_catalog_matches_dexbench_and_resolves_trajectory() -> None:
    assert set(DEXTOOLBENCH_DATA_STRUCTURE) == {
        "hammer",
        "marker",
        "eraser",
        "brush",
        "spatula",
        "screwdriver",
    }
    assert sum(len(objects) for objects in DEXTOOLBENCH_DATA_STRUCTURE.values()) == 12
    task = resolve_dexbench_task(SOURCE_ROOT, "hammer", "claw_hammer", "swing_side")
    assert task.object_urdf.name == "claw_hammer.urdf"
    assert task.decomposed_urdf.name == "claw_hammer_decomposed.urdf"
    assert task.trajectory.name == "swing_side.json"
    assert task.object_scale == (2.5, 0.5625, 0.375)


def test_dexbench_visual_urdf_keeps_original_textured_mesh_sidecars() -> None:
    task = resolve_dexbench_task(MANIFEST, "hammer", "claw_hammer", "swing_side")
    urdf = yourdfpy.URDF.load(
        task.object_urdf,
        build_scene_graph=True,
        load_meshes=True,
        filename_handler=partial(yourdfpy.filename_handler_magic, dir=task.object_urdf.parent),
    )
    assert urdf.scene is not None
    mesh = next(iter(urdf.scene.geometry.values()))
    assert mesh.visual.kind == "texture"
    assert mesh.visual.material.image.size[0] > 2
    assert mesh.visual.material.image.size[1] > 2


def test_dexbench_task_materializes_compilable_mujoco_scene(tmp_path: Path) -> None:
    task = resolve_dexbench_task(SOURCE_ROOT, "hammer", "claw_hammer", "swing_side")
    materialized = materialize_dexbench_scene(
        "/home/user/ws/lemon/rlgame-unilab/UniLab/src/unilab/assets/robots/kuka_sharpa/scene.xml",
        task,
        temp_root=tmp_path,
    )
    try:
        generated_assets = Path(materialized.model_files[0]).parent / "assets"
        assert generated_assets.is_dir() and not generated_assets.is_symlink()
        model = mujoco.MjModel.from_xml_path(materialized.model_files[0])
        assert model.nu == 29
        # The copied robot XML must retain MuJoCo's authored radian unit.
        # Without ``<compiler angle="radian">``, MuJoCo interprets the
        # numeric joint limits as degrees and silently clips the reset pose to
        # a nearly straight arm (about +/-0.05 rad).
        np.testing.assert_allclose(
            model.jnt_range[:7],
            np.array(
                [
                    [-2.96706, 2.96706],
                    [-2.0944, 2.0944],
                    [-2.96706, 2.96706],
                    [-2.0944, 2.0944],
                    [-2.96706, 2.96706],
                    [-2.0944, 2.0944],
                    [-3.05433, 3.05433],
                ]
            ),
            rtol=0.0,
            atol=1e-5,
        )
        # The task keyframe must be a usable full-scene start state too.  The
        # robot portion follows the policy's trained home pose; only the tool
        # pose comes from the selected DexToolBench trajectory.
        np.testing.assert_allclose(
            model.key_qpos[0, :7],
            np.array([-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308]),
            rtol=0.0,
            atol=1e-6,
        )
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object") >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table") >= 0
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)
        ]
        assert any(name.startswith("dex_object_visual") for name in names)
        assert any(name.startswith("dex_table_collision") for name in names)
        assert np.isfinite(model.qpos0).all()
        assert model.opt.timestep == 1.0 / 120.0
        assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
        assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
        assert model.opt.impratio == 10.0
        assert model.opt.iterations == 100
        assert model.opt.ls_iterations == 50
        assert not (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT))
        assert not (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD))
        for geom_id, name in enumerate(names):
            if name.startswith("dex_object_visual"):
                assert model.geom_contype[geom_id] == 0
                assert model.geom_conaffinity[geom_id] == 0
                assert model.geom_group[geom_id] == 2
            if name.startswith("dex_object_collision"):
                assert model.geom_contype[geom_id] == 1
                assert model.geom_conaffinity[geom_id] == 1
                assert model.geom_group[geom_id] == 3
                assert model.geom_condim[geom_id] == 3
                np.testing.assert_allclose(model.geom_friction[geom_id], [0.5, 0.005, 0.0001])
                np.testing.assert_allclose(model.geom_solimp[geom_id], [0.9, 0.95, 0.001, 0.5, 2.0])
                np.testing.assert_allclose(model.geom_solref[geom_id], [0.02, 1.0])
                assert model.geom_margin[geom_id] == 0.0
                assert model.geom_gap[geom_id] == 0.0
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        assert model.body_jntnum[object_id] == 1
        assert model.body_inertia[object_id].shape == (3,)
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal_object") >= 0
        assert model.nkey == 1
        mesh_scales = [
            model.mesh_scale[index].copy()
            for index in range(model.nmesh)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, index) or "").startswith(
                "dex_object_"
            )
        ]
        assert mesh_scales
        # object_scale is a policy-normalized observation scale, not a physical
        # mesh transform.  DexToolBench's URDF visual must remain unscaled.
        np.testing.assert_allclose(mesh_scales[0], [1.0, 1.0, 1.0])
    finally:
        materialized.cleanup.cleanup()


def test_simtoolreal_can_own_a_selected_dexbench_object() -> None:
    task = resolve_dexbench_task(SOURCE_ROOT, "hammer", "claw_hammer", "swing_side")
    cfg = SimToolRealCfg()
    cfg.assets.object_urdf = str(task.decomposed_urdf)
    cfg.assets.object_scale = task.object_scale
    cfg.assets.table_urdf = str(task.table_urdf)
    cfg.assets.object_pool_enabled = False
    cfg.reset.reset_position_noise_x = 0.0
    cfg.reset.reset_position_noise_y = 0.0
    cfg.reset.reset_position_noise_z = 0.0
    env = SimToolRealEnv(cfg, num_envs=1)
    try:
        assert env.resolve_object_scale().tolist() == [2.5, 0.5625, 0.375]
        assert env._tool_catalog[0].mass > 0.0
        assert env._backend.get_playback_model(0).nq == 36
        env.init_state()
        for _ in range(3):
            env.step(np.zeros((1, 29), dtype=np.float32))
        assert np.isfinite(env.get_physics_state_snapshot()).all()
    finally:
        env.close()


def test_dexbench_eval_start_arm_higher_matches_source_pose() -> None:
    task = resolve_dexbench_task(SOURCE_ROOT, "hammer", "claw_hammer", "swing_side")
    cfg = SimToolRealCfg()
    cfg.assets.object_urdf = str(task.decomposed_urdf)
    cfg.assets.object_scale = task.object_scale
    cfg.assets.object_pool_enabled = False
    cfg.reset.start_arm_higher = True
    cfg.reset.reset_position_noise_x = 0.0
    cfg.reset.reset_position_noise_y = 0.0
    cfg.reset.reset_position_noise_z = 0.0
    cfg.reset.reset_dof_pos_random_interval_arm = 0.0
    cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
    cfg.reset.reset_dof_vel_random_interval = 0.0
    env = SimToolRealEnv(cfg, num_envs=1)
    try:
        env.init_state()
        expected = env._default_joint_pos_canon[:7].copy()
        np.testing.assert_allclose(env.get_joint_pos_canon()[0, :7], expected, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(
            expected[[1, 3]],
            np.array([1.571, 1.376]) + np.array([-1.0, 1.0]) * np.deg2rad(10.0),
            rtol=0.0,
            atol=1e-6,
        )
    finally:
        env.close()
