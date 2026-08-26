from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.base.backend.base import SimBackend
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg

_NUM_ENVS = 4
_SIM_DT = 1.0 / 120.0
_DIVERGENT_QVEL = 1e11
_MINIMAL_XML = """<mujoco model="autoreset_probe">
  <option timestep="0.00833333" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="freebody" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom name="ball" type="sphere" size="0.05" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    xml_path = tmp_path / "autoreset_probe.xml"
    xml_path.write_text(_MINIMAL_XML, encoding="utf-8")
    current = MuJoCoBackend(SceneCfg(model_file=str(xml_path)), _NUM_ENVS, _SIM_DT)
    current.materialize()
    try:
        yield current
    finally:
        if current._pool is not None:
            current._pool.close()


def _zero_ctrl(backend: MuJoCoBackend) -> np.ndarray:
    return np.zeros((_NUM_ENVS, int(backend.model.nu)), dtype=np.float64)


def _rest_state(backend: MuJoCoBackend, rows: int) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.tile(
        np.asarray([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        (rows, 1),
    )
    qvel = np.zeros((rows, int(backend.model.nv)), dtype=np.float64)
    assert qpos.shape == (rows, int(backend.model.nq))
    return qpos, qvel


def _set_rest_state(backend: MuJoCoBackend) -> None:
    qpos, qvel = _rest_state(backend, _NUM_ENVS)
    backend.set_state(np.arange(_NUM_ENVS, dtype=np.int32), qpos, qvel)


def _diverge(backend: MuJoCoBackend, env_id: int) -> None:
    qpos, qvel = _rest_state(backend, 1)
    qvel[0, 0] = _DIVERGENT_QVEL
    backend.set_state(np.asarray([env_id], dtype=np.int32), qpos, qvel)


def test_real_pool_reports_settled_baseline_and_exact_diverged_env(
    backend: MuJoCoBackend,
) -> None:
    ctrl = _zero_ctrl(backend)
    _set_rest_state(backend)

    backend.step(ctrl)
    np.testing.assert_array_equal(
        backend.get_step_autoreset_mask(),
        np.asarray([False, False, False, False]),
    )

    _diverge(backend, env_id=1)
    backend.step(ctrl)
    np.testing.assert_array_equal(
        backend.get_step_autoreset_mask(),
        np.asarray([False, True, False, False]),
    )


def test_real_pool_or_latches_first_of_four_substeps_and_next_step_clears(
    backend: MuJoCoBackend,
) -> None:
    substep_calls: list[int] = []

    def passthrough(_backend: MuJoCoBackend, ctrl: np.ndarray) -> np.ndarray:
        substep_calls.append(1)
        return ctrl

    backend.set_pre_step_control(passthrough)
    ctrl = _zero_ctrl(backend)
    _diverge(backend, env_id=2)

    backend.step(ctrl, nsteps=4)

    assert len(substep_calls) == 4
    np.testing.assert_array_equal(
        backend.get_step_autoreset_mask(),
        np.asarray([False, False, True, False]),
    )

    _set_rest_state(backend)
    backend.step(ctrl)
    np.testing.assert_array_equal(
        backend.get_step_autoreset_mask(),
        np.asarray([False, False, False, False]),
    )


def test_default_backend_autoreset_contract_is_unknown() -> None:
    assert SimBackend.get_step_autoreset_mask(SimpleNamespace()) is None  # type: ignore[arg-type]
