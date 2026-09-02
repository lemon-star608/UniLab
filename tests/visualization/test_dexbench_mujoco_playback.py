from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


def _fake_worker(conn, spec):
    conn.send(
        {
            "type": "ready",
            "task_id": spec["task_id"],
            "joint_pos": [0.0] * 29,
            "tool_pose_wxyz": [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
            "goal_pose_wxyz": [0.1, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
            "goal_index": 0,
            "goal_count": 2,
            "step": 0,
            "episode_result": None,
        }
    )
    while True:
        if not conn.poll(0.05):
            continue
        command = conn.recv().get("command")
        if command == "run":
            conn.send({"type": "state", "status": "running", "step": 1})
        elif command == "pause":
            conn.send({"type": "state", "status": "paused", "step": 1})
        elif command == "resume":
            conn.send({"type": "state", "status": "running", "step": 1})
        elif command == "reset":
            conn.send({"type": "state", "status": "ready", "step": 0})
        elif command in {"stop", "quit"}:
            conn.send({"type": "stopped", "reason": command})
            return


def _wait_for(controller, kind: str):
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        messages = controller.poll()
        for message in messages:
            if message.get("type") == "error":
                raise AssertionError(f"worker error: {message}")
        for message in messages:
            if message.get("type") == kind:
                return message
        time.sleep(0.01)
    raise AssertionError(f"did not receive {kind}; status={controller.status}")


def _wait_for_post_step(controller):
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        for message in controller.poll():
            if message.get("type") == "error":
                raise AssertionError(f"worker error: {message}")
            if message.get("type") in {"state", "done"} and int(message.get("step", 0)) > 1:
                return message
        time.sleep(0.01)
    raise AssertionError(f"did not receive a post-step state; status={controller.status}")


def test_worker_command_poll_is_nonblocking() -> None:
    """Playback command checks must not add a control-period-sized wait."""
    from unilab.visualization.dexbench_mujoco_playback import _poll_worker_command

    class _Connection:
        def __init__(self) -> None:
            self.poll_args: list[tuple[object, ...]] = []

        def poll(self, *args: object) -> bool:
            self.poll_args.append(args)
            return False

        def recv(self) -> object:  # pragma: no cover - must not be called
            raise AssertionError("recv() called without a pending command")

    conn = _Connection()
    assert _poll_worker_command(conn) is None
    assert conn.poll_args == [()]


def test_spawn_worker_streams_ready_and_pause_resume_commands(tmp_path: Path) -> None:
    from unilab.visualization.dexbench_mujoco_playback import DexBenchPlaybackController

    controller = DexBenchPlaybackController(worker_target=_fake_worker, join_timeout=1.0)
    controller.start(
        category="hammer",
        object_name="claw_hammer",
        task_name="swing_side",
        trajectory_file=tmp_path / "trajectory.json",
    )
    try:
        ready = _wait_for(controller, "ready")
        assert ready["task_id"] == "hammer/claw_hammer/swing_side"
        assert controller.request_run() is True
        _wait_for(controller, "state")
        controller.toggle_pause()
        assert controller.status == "paused"
        _wait_for(controller, "state")
        controller.toggle_pause()
        assert controller.status == "running"
    finally:
        controller.stop()
    assert controller.process is None


def test_stop_and_reload_leave_no_orphan_worker(tmp_path: Path) -> None:
    from unilab.visualization.dexbench_mujoco_playback import DexBenchPlaybackController

    controller = DexBenchPlaybackController(worker_target=_fake_worker, join_timeout=1.0)
    kwargs = {
        "category": "hammer",
        "object_name": "claw_hammer",
        "task_name": "swing_side",
        "trajectory_file": tmp_path / "trajectory.json",
    }
    controller.start(**kwargs)
    _wait_for(controller, "ready")
    first = controller.process
    controller.reload(**kwargs)
    assert first is None or not first.is_alive()
    _wait_for(controller, "ready")
    controller.stop()
    assert controller.process is None


def test_registered_mujoco_worker_reaches_ready_state(tmp_path: Path) -> None:
    """Smoke the real cold path with the checked-in native SAPG checkpoint."""
    from hydra import compose, initialize_config_dir
    from scripts.play_dexbench_mujoco_viser import build_dexbench_env_override

    from unilab.algos.torch.rlgames_sapg.checkpoint_normalize import preflight_checkpoint
    from unilab.envs.manipulation.simtoolreal.assets import ensure_dexbench_assets
    from unilab.envs.manipulation.simtoolreal.dexbench_assets import (
        load_dexbench_trajectory,
        resolve_dexbench_task,
    )
    from unilab.visualization.dexbench_mujoco_playback import DexBenchPlaybackController

    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / "conf/rlgames_sapg"), version_base="1.3"):
        cfg = compose("dexbench_mujoco_viser")
    manifest = root / "src/unilab/assets/dexbench/manifest.json"
    if not manifest.is_file():
        try:
            manifest = ensure_dexbench_assets() / "manifest.json"
        except Exception as exc:
            pytest.skip(f"DexBench assets unavailable: {exc}")
    task = resolve_dexbench_task(manifest, "hammer", "claw_hammer", "swing_side")
    trajectory = load_dexbench_trajectory(task)
    trajectory_file = tmp_path / "trajectory.json"
    trajectory_file.write_text(
        json.dumps(
            {
                "pos": trajectory.goal_pos[None].tolist(),
                "quat_wxyz": trajectory.goal_quat_wxyz[None].tolist(),
            }
        ),
        encoding="utf-8",
    )
    checkpoint = (
        root
        / "logs/rlgames_sapg/SimToolReal/0_2026-08-27_22-38-06_mujoco/nn/0_simtoolreal_sapg.pth"
    )
    preflight_checkpoint(checkpoint)
    override = build_dexbench_env_override(
        cfg,
        object_urdf=task.decomposed_urdf,
        object_scale=task.object_scale,
        table_urdf=task.table_urdf,
        trajectory_file=trajectory_file,
        trajectory=trajectory,
        materialized_scene=task.materialized_mjcf,
    )
    controller = DexBenchPlaybackController(join_timeout=2.0)
    controller.start(
        category="hammer",
        object_name="claw_hammer",
        task_name="swing_side",
        trajectory_file=trajectory_file,
        cfg=cfg,
        checkpoint=checkpoint,
        env_override=override,
        model_file=task.materialized_mjcf,
    )
    try:
        message = _wait_for(controller, "ready")
        assert len(message["joint_pos"]) == 29
        assert len(message["tool_pose_wxyz"]) == 7
        assert message["goal_count"] == len(trajectory.goal_pos)
        assert controller.request_run() is True
        running = _wait_for(controller, "state")
        assert running.get("status") == "running"
        stepped = _wait_for_post_step(controller)
        assert int(stepped["step"]) > 1
        # Reset must preserve the native player's tensor observation ABI before
        # the next policy step; feeding adapter.step()'s dict here regresses to
        # the original ``unhashable type: 'slice'`` failure.
        controller.handle("reset")
        reset_state = _wait_for(controller, "state")
        assert reset_state.get("status") == "ready"
        assert controller.request_run() is True
        _wait_for(controller, "state")
        reset_stepped = _wait_for_post_step(controller)
        assert int(reset_stepped["step"]) > 1
    finally:
        controller.stop()
