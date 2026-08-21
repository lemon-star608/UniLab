from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg
from unilab.dr.types import GeomSizeOverride, InitRandomizationPlan, ModelVariantSpec

_SMALL_GEOMS = '<geom name="tool" type="box" size="0.06 0.04 0.03" mass="1"/>'
_DOMINANT_GEOMS = """
<geom name="tool" type="box" size="0.06 0.04 0.03" pos="-0.08 0 0" mass="0.7"/>
<geom name="tool_aux" type="box" size="0.03 0.025 0.02" pos="0.12 0 0" mass="0.3"/>
"""


def _write_free_body_model(path: Path, geoms: str) -> str:
    path.write_text(
        f"""<mujoco model="{path.stem}">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="object" pos="0 0 0.3">
      <freejoint name="root"/>
      {geoms}
      <site name="object_site"/>
    </body>
  </worldbody>
  <sensor><framepos objtype="site" objname="object_site"/></sensor>
</mujoco>
""",
        encoding="utf-8",
    )
    return str(path)


def _close_backend(backend: MuJoCoBackend) -> None:
    if backend._pool is not None:
        backend._pool.close()


def test_source_model_file_is_part_of_variant_identity(tmp_path: Path) -> None:
    source = _write_free_body_model(tmp_path / "source.xml", _SMALL_GEOMS)

    assert not ModelVariantSpec(source_model_file=source).is_empty()


def test_mujoco_directly_compiles_source_topology_and_applies_geom_override(
    tmp_path: Path,
) -> None:
    sphere = _write_free_body_model(
        tmp_path / "sphere.xml",
        '<geom name="tool" type="sphere" size="0.05" mass="1"/>',
    )
    capsule = _write_free_body_model(
        tmp_path / "capsule.xml",
        '<geom name="tool" type="capsule" size="0.03 0.04" mass="1"/>',
    )
    backend = MuJoCoBackend(SceneCfg(model_file=sphere), 4, 0.002)
    try:
        backend.apply_init_randomization(
            InitRandomizationPlan(
                model_assignments=np.asarray([0, 1, 1, 0], dtype=np.int32),
                model_variants=(
                    ModelVariantSpec(source_model_file=sphere),
                    ModelVariantSpec(
                        source_model_file=capsule,
                        geom_size_overrides=(GeomSizeOverride("tool", (0.04, 0.06, 0.0)),),
                    ),
                ),
            )
        )
        sphere_model, capsule_model = backend._model_variants
        tool_id = mujoco.mj_name2id(capsule_model, mujoco.mjtObj.mjOBJ_GEOM, "tool")
        assert int(sphere_model.geom_type[tool_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        assert int(capsule_model.geom_type[tool_id]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        np.testing.assert_allclose(capsule_model.geom_size[tool_id, :2], [0.04, 0.06])

        backend.materialize()
        backend.step(np.zeros((4, int(backend.model.nu)), dtype=np.float64))
    finally:
        _close_backend(backend)


def test_twelve_source_distributions_materialize_dominant_layout_and_step(
    tmp_path: Path,
) -> None:
    source_files = tuple(
        _write_free_body_model(
            tmp_path / f"distribution_{index}.xml",
            _DOMINANT_GEOMS if index == 7 else _SMALL_GEOMS,
        )
        for index in range(12)
    )
    assert len(set(source_files)) == 12

    backend = MuJoCoBackend(SceneCfg(model_file=source_files[0]), 12, 0.002)
    try:
        assignments = np.arange(12, dtype=np.int32)
        backend.apply_init_randomization(
            InitRandomizationPlan(
                model_assignments=assignments,
                model_variants=tuple(
                    ModelVariantSpec(source_model_file=source_file) for source_file in source_files
                ),
            )
        )
        np.testing.assert_array_equal(backend._model_assignments, assignments)

        small = backend._model_variants[0]
        dominant = backend._model_variants[7]
        for field in ("ngeom", "nbvh", "nC", "nbuffer"):
            assert getattr(dominant, field) > getattr(small, field), field

        backend.materialize()
        backend.step(np.zeros((12, int(backend.model.nu)), dtype=np.float64), nsteps=3)
        assert np.isfinite(backend.get_physics_state()).all()
    finally:
        _close_backend(backend)
