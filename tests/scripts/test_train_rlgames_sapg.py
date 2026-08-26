from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]


def _cfg(*overrides):
    with initialize_config_dir(config_dir=str(ROOT / "conf/rlgames_sapg"), version_base="1.3"):
        return compose("config", overrides=list(overrides))


class Resource:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class Tracker(Resource):
    def __init__(self, **kwargs):
        super().__init__()
        self.log_dir = Path(kwargs["log_dir"])
        self.start_calls = self.finish_calls = 0
        self.summaries = []
        self.videos = []
        self.run = None

    def start(self):
        self.start_calls += 1

    def update_summary(self, value):
        self.summaries.append(value)

    def log_video(self, value):
        self.videos.append(value)

    def finish(self):
        self.finish_calls += 1


def test_script_uses_single_tracker_and_cleans_env_writer_on_native_failure(tmp_path):
    from scripts.train_rlgames_sapg import run_training

    env = Resource()
    tracker_box = []
    observer_box = []

    def tracker_factory(**kwargs):
        tracker = Tracker(**kwargs)
        tracker_box.append(tracker)
        return tracker

    class Observer(Resource):
        def __init__(self, tracker):
            super().__init__()
            observer_box.append(self)

        close_writer = Resource.close

    def fail(*args, **kwargs):
        raise RuntimeError("native failure")

    with pytest.raises(RuntimeError, match="native failure"):
        run_training(
            _cfg(),
            root_dir=tmp_path,
            env_factory=lambda **kwargs: env,
            tracker_factory=tracker_factory,
            observer_factory=Observer,
            adapter_factory=lambda env, device: object(),
            executor=fail,
            verify_dependency=False,
            ensure_registry=lambda: None,
        )
    assert env.close_calls == 1
    assert observer_box[0].close_calls == 1
    assert tracker_box[0].start_calls == tracker_box[0].finish_calls == 1


def test_preflight_failure_happens_before_env_creation(tmp_path):
    from scripts.train_rlgames_sapg import run_training

    with pytest.raises(ValueError, match="MuJoCo"):
        run_training(
            _cfg("training.sim_backend=motrix"),
            root_dir=tmp_path,
            env_factory=lambda **kwargs: pytest.fail("env must not be created"),
            verify_dependency=False,
            ensure_registry=lambda: None,
        )


def test_training_returns_and_summarizes_checkpoint_from_current_run(tmp_path):
    from scripts.train_rlgames_sapg import run_training

    env = Resource()
    trackers = []

    def tracker_factory(**kwargs):
        tracker = Tracker(**kwargs)
        trackers.append(tracker)
        return tracker

    class Observer(Resource):
        def __init__(self, tracker):
            super().__init__()

        close_writer = Resource.close

    def executor(*args, train_dir, run_name, **kwargs):
        (Path(train_dir) / "batches").mkdir()
        checkpoint = Path(train_dir) / run_name / "nn/last_owner_ep_1_rew_-inf.pth"
        checkpoint.parent.mkdir(parents=True)
        torch.save({0: {"model": {"weight": torch.ones(1)}, "env_state": None}}, checkpoint)
        return SimpleNamespace(result=(-1_000_000_000, 1))

    result = run_training(
        _cfg("training.no_play=true"),
        root_dir=tmp_path,
        env_factory=lambda **kwargs: env,
        tracker_factory=tracker_factory,
        observer_factory=Observer,
        adapter_factory=lambda env, device: object(),
        executor=executor,
        verify_dependency=False,
        ensure_registry=lambda: None,
    )

    assert result.checkpoint == (result.run_dir / "nn/last_owner_ep_1_rew_-inf.pth").resolve()
    assert trackers[0].summaries[-1]["checkpoint"] == str(result.checkpoint)
    assert trackers[0].summaries[-1]["source_checkpoint"] is None
    assert env.close_calls == 1
    assert not (result.run_dir.parent / "batches").exists()


def test_training_releases_train_env_and_writer_before_after_train_playback(tmp_path):
    from scripts.train_rlgames_sapg import run_training

    env = Resource()
    observers = []

    class Observer(Resource):
        def __init__(self, tracker):
            super().__init__()
            observers.append(self)

        close_writer = Resource.close

    def executor(*args, train_dir, run_name, **kwargs):
        checkpoint = Path(train_dir) / run_name / "nn/last_owner_ep_1_rew_-inf.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"validated by injection")
        return SimpleNamespace(result=(-1_000_000_000, 1))

    def playback_runner(*args, **kwargs):
        assert env.close_calls == 1
        assert observers[0].close_calls == 1
        return SimpleNamespace(video="play_video.mp4")

    result = run_training(
        _cfg("training.play_render_mode=record"),
        root_dir=tmp_path,
        env_factory=lambda **kwargs: env,
        tracker_factory=Tracker,
        observer_factory=Observer,
        adapter_factory=lambda env, device: object(),
        executor=executor,
        checkpoint_validator=lambda path: None,
        playback_runner=playback_runner,
        verify_dependency=False,
        ensure_registry=lambda: None,
    )

    assert result.video == "play_video.mp4"
    assert env.close_calls == 1
    assert observers[0].close_calls == 1


def test_play_only_validates_before_env_and_uses_new_eval_sibling_tracker(tmp_path):
    from scripts.train_rlgames_sapg import run_playback

    task_root = tmp_path / "SimToolReal"
    source_run = task_root / "0_source_mujoco"
    source_run.mkdir(parents=True)
    source_config = source_run / "run_config.json"
    source_config.write_text('{"source": true}', encoding="utf-8")
    checkpoint = source_run / "nn/last_owner_ep_1_rew_-inf.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"trusted by injected validator")
    events = []
    trackers = []

    class PlayEnv(Resource):
        def run_playback_mode(self, *, initialize, step, output_video, play_steps, **kwargs):
            obs = initialize()
            for _ in range(play_steps):
                obs = step(obs)
            Path(output_video).write_bytes(b"mp4")
            return str(output_video)

    class Bridge:
        def initialize(self):
            events.append("initialize")
            return torch.zeros((6, 140))

        def step(self, obs):
            events.append("step")
            return obs

    def tracker_factory(**kwargs):
        tracker = Tracker(**kwargs)
        trackers.append(tracker)
        return tracker

    def validate(path):
        events.append("validate")
        assert Path(path) == checkpoint

    def env_factory(**kwargs):
        events.append("env")
        return PlayEnv()

    result = run_playback(
        _cfg(
            "training.play_only=true",
            "training.play_render_mode=record",
            "training.play_steps=2",
            f"training.log_root={tmp_path}",
            "algo.load_run=0_source_mujoco",
        ),
        root_dir=tmp_path,
        env_factory=env_factory,
        tracker_factory=tracker_factory,
        adapter_factory=lambda env, device: object(),
        player_builder=lambda **kwargs: Bridge(),
        checkpoint_validator=validate,
        sim2sim_resolver=lambda *args, **kwargs: args[1],
        verify_dependency=False,
        ensure_registry=lambda: None,
    )

    assert events == ["validate", "env", "initialize", "step", "step"]
    assert result.source_run == source_run.resolve()
    assert result.checkpoint == checkpoint.resolve()
    assert result.run_dir.parent == task_root.resolve()
    assert result.run_dir.name.startswith("eval_")
    assert Path(result.video).read_bytes() == b"mp4"
    assert trackers[0].log_dir == result.run_dir
    assert trackers[0].start_calls == trackers[0].finish_calls == 1
    assert source_config.read_text(encoding="utf-8") == '{"source": true}'
