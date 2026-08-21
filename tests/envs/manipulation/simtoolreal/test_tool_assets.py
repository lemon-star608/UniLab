from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

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
            assert (model.nq, model.nv, model.nu, model.nmesh) == (36, 35, 29, 62)
            assert model.opt.timestep == pytest.approx(1.0 / 120.0)
            assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
            assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
            assert model.opt.impratio == pytest.approx(10.0)
            assert model.opt.iterations == 100
            assert model.opt.ls_iterations == 50
            for flag_name in ("mjDSBL_CONTACT", "mjDSBL_MULTICCD"):
                flag = getattr(mujoco.mjtDisableBit, flag_name, None)
                assert flag is not None, f"installed MuJoCo lacks public {flag_name}"
                assert not (int(model.opt.disableflags) & int(flag)), f"{flag_name} is disabled"

            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            assert np.isfinite(data.qpos).all()
            assert np.isfinite(data.qvel).all()
            assert np.isfinite(data.ctrl).all()
            for _ in range(4):
                mujoco.mj_step(model, data)
                assert np.isfinite(data.qpos).all()
                assert np.isfinite(data.qvel).all()
                assert np.isfinite(data.ctrl).all()
    finally:
        materialized.cleanup.cleanup()
