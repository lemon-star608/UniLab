"""M0-dev combination gate for mixed layouts, autoreset, and CPU affinity."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.backend.mujoco.playback import resolve_render_play_model_files
from unilab.base.scene import SceneCfg
from unilab.envs.manipulation.simtoolreal import SimToolRealEnv
from unilab.envs.manipulation.simtoolreal.dr_provider import DSTAR_SENTINEL

_ROOT = Path(__file__).resolve().parents[4]
_MANIFEST = _ROOT / "tests/fixtures/simtoolreal_sapg/m0_dev_manifest.json"
_NUM_ENVS = 8
_SELECTED_ROW = 7


def _object_contacts(model: mujoco.MjModel) -> dict[str, tuple[int, np.ndarray]]:
    contacts: dict[str, tuple[int, np.ndarray]] = {}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if (
            name
            and name.startswith("object_")
            and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        ):
            contacts[name] = (int(model.geom_type[geom_id]), model.geom_size[geom_id].copy())
    return contacts


def _diverge_one_env(env: SimToolRealEnv, env_id: int) -> None:
    """Trip the engine autoreset through the public FULLPHYSICS state contract."""
    backend = env.backend
    nq, nv = int(backend.model.nq), int(backend.model.nv)
    physics = np.asarray(backend.get_physics_state(), dtype=np.float64)
    assert physics.shape[1] >= 1 + nq + nv
    qpos = physics[env_id : env_id + 1, 1 : 1 + nq].copy()
    qvel = physics[env_id : env_id + 1, 1 + nq : 1 + nq + nv].copy()
    qvel.fill(1.0e11)
    backend.set_state(np.asarray([env_id], dtype=np.int32), qpos, qvel)


def test_manifest_records_the_m0_dev_combination_gate() -> None:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["combination_gate"] == {
        "command": (
            "uv run --extra mujoco --extra rlgames-sapg pytest "
            "tests/envs/manipulation/simtoolreal/test_m0_dev_matrix.py -q"
        ),
        "passed": 3,
        "skipped": 0,
        "covers": [
            "mixed visual model layouts for catalog indexes 0, 1, and 7",
            "two-substep per-env autoreset latch and reset isolation",
            "explicit CPU affinity and default OS scheduling",
        ],
    }


def test_real_mixed_layout_autoreset_affinity_combination(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    available_cpus = sorted(os.sched_getaffinity(0))
    assert len(available_cpus) >= 2
    cpu_ids = available_cpus[:2]

    np.random.seed(20260821)
    registry.ensure_registries()
    env = registry.make(
        "SimToolReal",
        sim_backend="mujoco",
        num_envs=_NUM_ENVS,
        env_cfg_override={"cpu_ids": cpu_ids},
    )
    assert isinstance(env, SimToolRealEnv)
    backend = env.backend
    assert isinstance(backend, MuJoCoBackend)
    pool = backend._pool
    assert pool is not None
    tool_root = Path(env._tool_scenes.cleanup.name)
    visual_root: Path | None = None

    try:
        state = env.init_state()
        np.testing.assert_array_equal(env._tool_index, np.arange(_NUM_ENVS, dtype=np.int32))
        assert tuple(pool.cpu_ids) == tuple(cpu_ids)
        assert pool.worker_cpu_ids() == tuple(cpu_ids)
        np.testing.assert_array_equal(pool.was_autoreset, np.zeros(_NUM_ENVS, dtype=bool))
        np.testing.assert_array_equal(
            backend.get_step_autoreset_mask(), np.zeros(_NUM_ENVS, dtype=bool)
        )

        visual_source = env.get_scene_visual_model_file()
        assert visual_source is not None
        visual_base = mujoco.MjModel.from_xml_path(visual_source)
        visual_mesh_names = {
            mujoco.mj_id2name(visual_base, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
            for mesh_id in range(visual_base.nmesh)
        }
        visual_tmp = tempfile.TemporaryDirectory(prefix="m0-dev-visual-", dir=tmp_path)
        visual_root = Path(visual_tmp.name)
        try:
            resolved = resolve_render_play_model_files(
                env,
                num_envs=_NUM_ENVS,
                tmp_dir=visual_root,
            )
            assert isinstance(resolved, list)
            for env_id in (0, 1, 7):
                model = mujoco.MjModel.from_binary_path(resolved[env_id])
                assert (model.nq, model.nv, model.nu, model.nmesh) == (36, 35, 29, 62)
                assert {
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
                    for mesh_id in range(model.nmesh)
                } == visual_mesh_names

                spec = env._tool_catalog[int(env._tool_index[env_id])]
                contacts = _object_contacts(model)
                handle_name = (
                    "object_handle_cyl" if spec.collision_shape == "capsule" else "object_handle"
                )
                expected_names = {handle_name}
                if spec.topology != "box_only":
                    expected_names.add("object_head")
                assert set(contacts) == expected_names
                np.testing.assert_allclose(
                    contacts[handle_name][1], spec.handle_size, rtol=1e-5, atol=1e-8
                )
                if "object_head" in contacts:
                    np.testing.assert_allclose(
                        contacts["object_head"][1], spec.head_size, rtol=1e-5, atol=1e-8
                    )
        finally:
            visual_tmp.cleanup()
        assert not visual_root.exists()

        state.info["steps"].fill(0)
        state.info["goal_pos"][:] = np.asarray([9.0, 9.0, 9.0], dtype=np.float32)
        state.info["near_goal_steps"].fill(0)
        state.info["prev_targets"][_SELECTED_ROW].fill(9.0)
        state.info["cur_targets"][_SELECTED_ROW].fill(9.0)
        state.info["object_init_z"][_SELECTED_ROW] = 9.0
        state.info["closest_keypoint_max_dist"][_SELECTED_ROW] = 9.0
        state.info["closest_fingertip_dist"][_SELECTED_ROW].fill(9.0)
        state.info["lifted_object"][_SELECTED_ROW] = True
        env._state_cache_lifted_object.fill(False)
        env._state_cache_lifted_object[_SELECTED_ROW] = True
        env._action_queue[_SELECTED_ROW].fill(9.0)
        env._obs_queue[_SELECTED_ROW].fill(9.0)
        env._object_state_queue[_SELECTED_ROW].fill(9.0)

        keep = np.arange(_NUM_ENVS) != _SELECTED_ROW
        object_init_z_before = state.info["object_init_z"][keep].copy()
        goal_pos_before = state.info["goal_pos"][keep].copy()
        goal_quat_before = state.info["goal_quat"][keep].copy()

        assert env.cfg.sim_substeps == 2
        monkeypatch.chdir(tmp_path)
        _diverge_one_env(env, _SELECTED_ROW)
        state = env.step(np.zeros((_NUM_ENVS, 29), dtype=np.float32))

        expected = np.arange(_NUM_ENVS) == _SELECTED_ROW
        np.testing.assert_array_equal(backend.get_step_autoreset_mask(), expected)
        np.testing.assert_array_equal(env._autoreset_envs, expected)
        np.testing.assert_array_equal(state.terminated, expected)
        np.testing.assert_array_equal(state.info["_final_observation"], expected)
        assert state.final_observation is not None
        for group in env.obs_groups_spec:
            assert np.isfinite(state.final_observation[group][_SELECTED_ROW]).all()
            assert np.isfinite(state.obs[group][_SELECTED_ROW]).all()
            assert not np.array_equal(
                state.obs[group][_SELECTED_ROW],
                state.final_observation[group][_SELECTED_ROW],
            )

        assert np.isfinite(state.info["prev_targets"][_SELECTED_ROW]).all()
        assert np.isfinite(state.info["cur_targets"][_SELECTED_ROW]).all()
        assert not np.all(state.info["prev_targets"][_SELECTED_ROW] == 9.0)
        assert not np.all(state.info["cur_targets"][_SELECTED_ROW] == 9.0)
        assert state.info["object_init_z"][_SELECTED_ROW] != 9.0
        assert state.info["closest_keypoint_max_dist"][_SELECTED_ROW] == DSTAR_SENTINEL
        assert np.all(state.info["closest_fingertip_dist"][_SELECTED_ROW] == DSTAR_SENTINEL)
        assert not state.info["lifted_object"][_SELECTED_ROW]
        assert not env._state_cache_lifted_object[_SELECTED_ROW]
        assert np.all(env._action_queue[_SELECTED_ROW] == 0.0)
        assert not np.any(env._obs_queue[_SELECTED_ROW] == 9.0)
        assert not np.any(env._object_state_queue[_SELECTED_ROW] == 9.0)

        np.testing.assert_array_equal(state.info["object_init_z"][keep], object_init_z_before)
        np.testing.assert_array_equal(state.info["goal_pos"][keep], goal_pos_before)
        np.testing.assert_array_equal(state.info["goal_quat"][keep], goal_quat_before)

        state = env.step(np.zeros((_NUM_ENVS, 29), dtype=np.float32))
        np.testing.assert_array_equal(
            backend.get_step_autoreset_mask(), np.zeros(_NUM_ENVS, dtype=bool)
        )
        assert not env._autoreset_envs.any()
        assert not state.info["_final_observation"].any()
        assert np.isfinite(state.obs["obs"]).all()
    finally:
        pool.close()
        env.close()
        env.close()

    assert not tool_root.exists()
    assert visual_root is not None
    assert not visual_root.exists()


def test_default_cpu_ids_keeps_os_scheduling_through_public_backend_wiring() -> None:
    model_file = str(ASSETS_ROOT_PATH / "robots" / "go2_arm" / "scene_flat.xml")
    backend = create_backend(
        "mujoco",
        SceneCfg(model_file=model_file),
        num_envs=2,
        sim_dt=0.01,
        base_name="base",
        adaptive_chunk_size=False,
        cpu_ids=None,
    )
    assert isinstance(backend, MuJoCoBackend)
    backend.materialize()
    pool = backend._pool
    assert pool is not None
    try:
        assert backend._cpu_ids is None
        assert pool.cpu_ids is None
        assert pool.worker_cpu_ids() == ()
        backend.step(np.zeros((2, backend.num_actuators), dtype=np.float64))
    finally:
        pool.close()
        backend.cleanup_scene_assets()
