from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.base.backend.base import SimBackend
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg

_MODEL_XML = """<mujoco model="body_wrench">
  <worldbody>
    <body name="body_a" pos="-0.2 0 0">
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
    <body name="body_b" pos="0.2 0 0">
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def backend(tmp_path: Path) -> MuJoCoBackend:
    model_file = tmp_path / "body_wrench.xml"
    model_file.write_text(_MODEL_XML, encoding="utf-8")
    return MuJoCoBackend(SceneCfg(model_file=str(model_file)), 3, 0.002)


def _wrench_inputs() -> tuple[np.ndarray, np.ndarray]:
    force = np.zeros((3, 2, 3), dtype=np.float32)
    torque = np.zeros((3, 2, 3), dtype=np.float32)
    force[0, 0] = [1.0, 2.0, 3.0]
    torque[0, 0] = [4.0, 5.0, 6.0]
    force[2, 1] = [-1.0, -2.0, -3.0]
    torque[2, 1] = [-4.0, -5.0, -6.0]
    return force, torque


def test_apply_body_wrench_isolates_env_rows_and_body_blocks_and_accumulates(
    backend: MuJoCoBackend,
) -> None:
    body_a, body_b = backend.get_body_ids(["body_a", "body_b"])
    body_ids = np.asarray([body_a, body_b], dtype=np.int64)
    force, torque = _wrench_inputs()

    backend.apply_body_wrench(body_ids, force, torque)

    staged = backend._pending_xfrc_applied.reshape(3, backend.model.nbody, 6)
    np.testing.assert_array_equal(staged[0, body_a], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(staged[2, body_b], [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    np.testing.assert_array_equal(staged[0, body_b], np.zeros(6))
    np.testing.assert_array_equal(staged[2, body_a], np.zeros(6))
    np.testing.assert_array_equal(staged[1], np.zeros((backend.model.nbody, 6)))
    np.testing.assert_array_equal(staged[:, 0], np.zeros((3, 6)))

    backend.apply_body_wrench(body_ids, force, torque)

    np.testing.assert_array_equal(staged[0, body_a], [2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    np.testing.assert_array_equal(staged[2, body_b], [-2.0, -4.0, -6.0, -8.0, -10.0, -12.0])


@pytest.mark.parametrize("invalid_input", ["force", "torque"])
def test_apply_body_wrench_rejects_each_shape_mismatch(
    backend: MuJoCoBackend,
    invalid_input: str,
) -> None:
    body_ids = backend.get_body_ids(["body_a", "body_b"])
    force, torque = _wrench_inputs()
    if invalid_input == "force":
        force = force[:, :1]
    else:
        torque = torque[:, :1]

    with pytest.raises(ValueError, match=rf"body wrench {invalid_input} must have shape"):
        backend.apply_body_wrench(body_ids, force, torque)


def test_default_backend_body_wrench_contract_is_unsupported() -> None:
    unsupported = SimpleNamespace()
    with pytest.raises(NotImplementedError, match="does not support body wrench perturbation"):
        SimBackend.apply_body_wrench(  # type: ignore[arg-type]
            unsupported,
            np.asarray([1], dtype=np.int32),
            np.zeros((1, 1, 3)),
            np.zeros((1, 1, 3)),
        )
