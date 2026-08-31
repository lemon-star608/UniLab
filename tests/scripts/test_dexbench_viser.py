from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf
from scripts.play_dexbench_mujoco_viser import build_dexbench_env_override

from unilab.envs.manipulation.simtoolreal.dexbench_assets import (
    DexBenchTrajectory,
)
from unilab.visualization.dexbench_mujoco_playback import DexBenchPlaybackController


def test_dexbench_run_loop_uses_render_rate(monkeypatch) -> None:
    """The parent viewer must poll often enough to display worker frames."""
    import scripts.play_dexbench_mujoco_viser as viewer

    demo = viewer.DexBenchInteractiveDemo(
        OmegaConf.create({"dexbench": {"port": 8083}}), server=object()
    )
    monkeypatch.setattr(demo, "build_gui", lambda: None)
    monkeypatch.setattr(demo, "_poll_worker", lambda: None)
    monkeypatch.setattr(demo, "close", lambda: None)
    sleep_calls: list[float] = []

    def stop_after_one_poll(delay: float) -> None:
        sleep_calls.append(delay)
        raise KeyboardInterrupt

    monkeypatch.setattr(viewer.time, "sleep", stop_after_one_poll)
    demo.run()

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(1.0 / 60.0)


def test_dexbench_run_loop_honors_configured_render_rate(monkeypatch) -> None:
    import scripts.play_dexbench_mujoco_viser as viewer

    demo = viewer.DexBenchInteractiveDemo(
        OmegaConf.create({"dexbench": {"port": 8083, "render_hz": 30}}), server=object()
    )
    monkeypatch.setattr(demo, "build_gui", lambda: None)
    monkeypatch.setattr(demo, "_poll_worker", lambda: None)
    monkeypatch.setattr(demo, "close", lambda: None)
    sleep_calls: list[float] = []

    def stop_after_one_poll(delay: float) -> None:
        sleep_calls.append(delay)
        raise KeyboardInterrupt

    monkeypatch.setattr(viewer.time, "sleep", stop_after_one_poll)
    demo.run()

    assert sleep_calls == [pytest.approx(1.0 / 30.0)]


def test_dexbench_run_loop_rejects_non_positive_render_rate(monkeypatch) -> None:
    import scripts.play_dexbench_mujoco_viser as viewer

    demo = viewer.DexBenchInteractiveDemo(
        OmegaConf.create({"dexbench": {"port": 8083, "render_hz": 0}}), server=object()
    )
    monkeypatch.setattr(demo, "build_gui", lambda: None)

    with pytest.raises(ValueError, match="render_hz must be positive"):
        demo.run()


def test_dexbench_poll_worker_renders_only_latest_state(monkeypatch) -> None:
    """A slow render must skip stale queued states instead of replaying them."""
    import scripts.play_dexbench_mujoco_viser as viewer

    demo = viewer.DexBenchInteractiveDemo(OmegaConf.create({}), server=object())
    queued = [
        {"type": "state", "step": 1, "goal_index": 0, "goal_count": 2},
        {"type": "state", "step": 2, "goal_index": 0, "goal_count": 2},
        {"type": "state", "step": 3, "goal_index": 1, "goal_count": 2},
    ]
    demo.controller = SimpleNamespace(poll=lambda: queued)
    rendered: list[dict[str, object]] = []
    monkeypatch.setattr(demo, "_render_state", rendered.append)

    demo._poll_worker()

    assert [int(message["step"]) for message in rendered] == [3]


def test_dexbench_controller_requires_load_before_run() -> None:
    controller = DexBenchPlaybackController()
    assert controller.status == "empty"
    assert controller.request_run() is False
    assert controller.status == "empty"


def test_dexbench_controller_load_pause_stop_lifecycle(tmp_path: Path) -> None:
    controller = DexBenchPlaybackController()
    controller.load("hammer", "claw_hammer", "swing_side", tmp_path / "trajectory.json")
    assert controller.status == "ready"
    assert controller.request_run() is True
    assert controller.status == "running"
    controller.toggle_pause()
    assert controller.status == "paused"
    controller.toggle_pause()
    assert controller.status == "running"
    controller.stop()
    assert controller.status == "ready"


def test_dexbench_env_override_is_eval_only_and_keeps_policy_contract() -> None:
    cfg = OmegaConf.create({"env": {"sim_dt": 1.0 / 120.0, "ctrl_dt": 1.0 / 60.0}})
    trajectory = DexBenchTrajectory(
        start_pose_wxyz=(0.0, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0),
        goal_pos=np.asarray([[0.1, 0.2, 0.7], [0.2, 0.2, 0.7]], dtype=np.float64),
        goal_quat_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )
    override = build_dexbench_env_override(
        cfg,
        object_urdf="/tmp/object_decomposed.urdf",
        object_scale=(2.5, 0.5, 0.4),
        table_urdf="/tmp/table.urdf",
        trajectory_file="/tmp/trajectory.json",
        trajectory=trajectory,
    )
    assert override["sim_dt"] == 1.0 / 120.0
    assert override["ctrl_dt"] == 1.0 / 60.0
    assert override["assets"]["object_pool_enabled"] is False
    assert override["assets"]["object_urdf"].endswith("object_decomposed.urdf")
    assert override["reset"]["fixed_start_pose"] == trajectory.start_pose_wxyz
    assert override["reset"]["fixed_trajectory_count"] == 1
    assert override["goal"]["eval_success_tolerance"] == 0.01
    assert override["goal"]["success_steps"] == 1
    assert override["termination"]["max_consecutive_successes"] == 2
    dr = override["domain_randomization"]
    assert dr["use_obs_delay"] is False
    assert dr["use_action_delay"] is False
    assert dr["force_scale"] == 0.0
    assert dr["torque_scale"] == 0.0


def test_render_data_starts_from_task_keyframe() -> None:
    """The static scene must not show MuJoCo's all-zero pose while loading."""
    import mujoco
    from scripts.play_dexbench_mujoco_viser import _initialize_render_data

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body><joint name="joint"/><geom type="box" size="0.1 0.1 0.1"/></body>
          </worldbody>
          <keyframe><key name="home" qpos="0.7"/></keyframe>
        </mujoco>
        """
    )
    data = _initialize_render_data(model)
    np.testing.assert_allclose(data.qpos, [0.7])


def test_initial_visual_poses_follow_trajectory() -> None:
    from scripts.play_dexbench_mujoco_viser import _initial_visual_poses

    from unilab.envs.manipulation.simtoolreal.dexbench_assets import DexBenchTrajectory

    trajectory = DexBenchTrajectory(
        start_pose_wxyz=(0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0),
        goal_pos=np.asarray([[0.4, 0.5, 0.6]], dtype=np.float64),
        goal_quat_wxyz=np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float64),
    )
    start, goal = _initial_visual_poses(trajectory)
    assert start == trajectory.start_pose_wxyz
    assert goal == (0.4, 0.5, 0.6, 0.0, 1.0, 0.0, 0.0)
