from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[3]


def _cfg(log_root: Path, *overrides: str):
    common = [
        f"training.log_root={log_root}",
        "training.device=cuda:0",
        "algo.num_envs=6",
        "rl_games.params.config.expl_coef_block_size=1",
        "rl_games.params.config.horizon_length=4",
        "rl_games.params.config.seq_length=4",
        "rl_games.params.config.minibatch_size=12",
        "rl_games.params.config.mini_epochs=1",
        "rl_games.params.config.central_value_config.minibatch_size=12",
        "rl_games.params.config.central_value_config.mini_epochs=1",
        "rl_games.params.config.max_epochs=1",
        "rl_games.params.config.save_frequency=1",
    ]
    with initialize_config_dir(config_dir=str(ROOT / "conf/rlgames_sapg"), version_base="1.3"):
        return compose("config", overrides=[*common, *overrides])


def _temporary_artifacts() -> set[Path]:
    root = Path(tempfile.gettempdir())
    prefixes = ("simtoolreal_tools_", "unilab-mj-variant-", "unilab-playback-models-")
    return {path for path in root.iterdir() if path.name.startswith(prefixes)}


def _assert_finite_tensors(value: Any) -> None:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for child in value.values():
            _assert_finite_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_finite_tensors(child)


@pytest.mark.slow
def test_real_n6_native_train_checkpoint_player_and_video_vertical_slice(tmp_path):
    from scripts.train_rlgames_sapg import run_playback, run_training

    from unilab.algos.torch.rlgames_sapg.checkpoint import validate_native_checkpoint

    assert torch.cuda.is_available()
    before_temp = _temporary_artifacts()
    train = run_training(_cfg(tmp_path, "training.no_play=true"))

    assert train.native_result[1] == 1
    assert train.checkpoint is not None and train.checkpoint.is_file()
    assert train.checkpoint.stat().st_size > 0
    metadata = validate_native_checkpoint(train.checkpoint)
    required = {
        "model",
        "optimizer",
        "assymetric_vf_nets",
        "epoch",
        "frame",
        "env_state",
        "obs",
        "rnn_states",
        "dones",
        "current_rewards",
        "current_shaped_rewards",
        "current_lengths",
    }
    assert required <= set(metadata.state_keys)
    assert metadata.env_state_is_none
    payload = torch.load(train.checkpoint, map_location="cpu", weights_only=False)[0]
    assert payload["epoch"] == 1
    assert payload["frame"] == 24
    assert payload["optimizer"]["state"]
    _assert_finite_tensors(payload)

    assert (train.run_dir / "run_config.json").is_file()
    assert (train.run_dir / "run_summary.json").is_file()
    assert list((train.run_dir / "summaries").glob("events.out.tfevents.*"))
    assert [path for path in train.run_dir.parent.iterdir() if path.is_dir()] == [train.run_dir]
    train_config = (train.run_dir / "run_config.json").read_bytes()
    train_summary = json.loads((train.run_dir / "run_summary.json").read_text())
    assert train_summary["checkpoint"] == str(train.checkpoint)

    play = run_playback(
        _cfg(
            tmp_path,
            "training.play_only=true",
            "training.play_render_mode=record",
            "training.play_env_num=6",
            "training.play_steps=4",
            f"algo.load_run={train.run_dir.name}",
        )
    )
    assert play.source_run == train.run_dir
    assert play.checkpoint == train.checkpoint
    assert play.run_dir.parent == train.run_dir.parent
    assert play.run_dir.name.startswith("eval_")
    assert {path for path in train.run_dir.parent.iterdir() if path.is_dir()} == {
        train.run_dir,
        play.run_dir,
    }
    assert play.video is not None
    video = Path(play.video)
    assert video.is_file() and video.stat().st_size > 0
    assert (play.run_dir / "run_config.json").is_file()
    play_summary = json.loads((play.run_dir / "run_summary.json").read_text())
    assert play_summary["source_run"] == str(train.run_dir)
    assert play_summary["source_checkpoint"] == str(train.checkpoint)
    assert (train.run_dir / "run_config.json").read_bytes() == train_config
    assert _temporary_artifacts() == before_temp
