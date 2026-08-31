from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch


def test_normalize_checkpoint_converts_only_numpy_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.pth"
    target = tmp_path / "normalized.pth"
    model = OrderedDict(
        [
            ("actor_mlp.0.weight", torch.arange(12, dtype=torch.float32).reshape(3, 4)),
            ("mu.weight", torch.ones((2, 3), dtype=torch.float32)),
        ]
    )
    torch.save(
        {
            0: {
                "model": model,
                "meta": {"epoch": np.int64(4), "array": np.array([1, 2])},
                "env_state": None,
            }
        },
        source,
    )

    from unilab.algos.torch.rlgames_sapg.checkpoint_normalize import normalize_checkpoint

    assert normalize_checkpoint(source, target) == target.resolve()
    assert source.exists()
    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert isinstance(payload[0]["model"], OrderedDict)
    assert torch.equal(payload[0]["model"]["actor_mlp.0.weight"], model["actor_mlp.0.weight"])
    assert isinstance(payload[0]["meta"]["epoch"], int)
    assert payload[0]["meta"]["array"] == [1, 2]


def test_preflight_reports_dimensions_and_rejects_actor_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "model.pth"
    torch.save(
        {
            0: {
                "model": {
                    "actor_mlp.0.weight": torch.zeros((8, 4)),
                    "mu.weight": torch.zeros((2, 8)),
                },
                "env_state": None,
            }
        },
        path,
    )
    from unilab.algos.torch.rlgames_sapg.checkpoint_normalize import preflight_checkpoint

    report = preflight_checkpoint(path, expected_actor_dim=4)
    assert report.actor_input_dim == 4
    assert report.action_dim == 2
    with pytest.raises(ValueError, match="actor input dimension"):
        preflight_checkpoint(path, expected_actor_dim=140)


def test_preflight_error_points_to_normalize_for_unreadable_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "broken.pth"
    path.write_bytes(b"not a torch checkpoint")
    from unilab.algos.torch.rlgames_sapg.checkpoint_normalize import preflight_checkpoint

    with pytest.raises(ValueError, match="normalize_checkpoint"):
        preflight_checkpoint(path)
