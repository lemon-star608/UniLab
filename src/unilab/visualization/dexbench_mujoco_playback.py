"""MuJoCo worker boundary for the DexToolBench interactive viewer.

The parent process owns controls and rendering. Environment construction,
checkpoint loading, and policy stepping happen in a ``spawn`` child.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class DexBenchState:
    joint_pos: tuple[float, ...]
    tool_pose_wxyz: tuple[float, ...]
    goal_pose_wxyz: tuple[float, ...]
    goal_index: int
    goal_count: int
    step: int
    episode_result: str | None
    physics_state: tuple[float, ...] | None = None

    def as_message(self) -> dict[str, object]:
        message: dict[str, object] = {
            "type": "state",
            "joint_pos": list(self.joint_pos),
            "tool_pose_wxyz": list(self.tool_pose_wxyz),
            "goal_pose_wxyz": list(self.goal_pose_wxyz),
            "goal_index": self.goal_index,
            "goal_count": self.goal_count,
            "step": self.step,
            "episode_result": self.episode_result,
        }
        if self.physics_state is not None:
            message["physics_state"] = list(self.physics_state)
        return message


def _poll_worker_command(conn: Any) -> object | None:
    """Return one pending command without blocking the control loop."""
    if not conn.poll():
        return None
    return conn.recv()


def _state_from_env(env: Any, *, episode_result: str | None = None) -> DexBenchState:
    """Read a finite, env-local snapshot from the already-built environment."""
    state = env.state
    if state is None:
        raise RuntimeError("SimToolReal worker has no state after reset")
    joint = np.asarray(env.get_joint_pos_canon()[0], dtype=np.float64)
    tool_pos = np.asarray(env.get_object_pos()[0], dtype=np.float64)
    tool_quat = np.asarray(env.get_object_quat()[0], dtype=np.float64)
    goal_pos = np.asarray(state.info.get("goal_pos"), dtype=np.float64)
    goal_quat = np.asarray(state.info.get("goal_quat"), dtype=np.float64)
    if goal_pos.ndim == 2:
        goal_pos = goal_pos[0]
    if goal_quat.ndim == 2:
        goal_quat = goal_quat[0]
    goal_index = int(np.asarray(state.info.get("successes", [0]))[0])
    goal_count = int(getattr(env.cfg.termination, "max_consecutive_successes", 0))
    step = int(np.asarray(state.info.get("steps", [0]))[0])
    physics = np.asarray(env.get_physics_state_snapshot()[0], dtype=np.float64)
    values = (joint, tool_pos, tool_quat, goal_pos, goal_quat)
    valid_shapes = {(3,), (4,), (29,)}
    if any(value.shape not in valid_shapes or not np.isfinite(value).all() for value in values):
        raise RuntimeError("MuJoCo worker produced a non-finite or malformed state")
    if physics.ndim != 1 or not np.isfinite(physics).all():
        raise RuntimeError("MuJoCo worker produced a non-finite physics snapshot")
    return DexBenchState(
        joint_pos=tuple(float(value) for value in joint),
        tool_pose_wxyz=tuple(float(value) for value in (*tool_pos, *tool_quat)),
        goal_pose_wxyz=tuple(float(value) for value in (*goal_pos, *goal_quat)),
        goal_index=goal_index,
        goal_count=goal_count,
        step=step,
        episode_result=episode_result,
        physics_state=tuple(float(value) for value in physics),
    )


def _worker_env(spec: dict[str, object]):
    """Construct the registered environment and native policy on the cold path."""
    from omegaconf import OmegaConf

    from unilab.algos.torch.rlgames_sapg.checkpoint import validate_native_checkpoint
    from unilab.algos.torch.rlgames_sapg.checkpoint_normalize import preflight_checkpoint
    from unilab.algos.torch.rlgames_sapg.env_adapter import RlGamesNpEnvAdapter
    from unilab.algos.torch.rlgames_sapg.player import build_native_player_bridge
    from unilab.training import create_env, ensure_registries

    ensure_registries()
    cfg = OmegaConf.create(spec["cfg"])
    checkpoint = Path(str(spec["checkpoint"])).expanduser().resolve()
    preflight_checkpoint(checkpoint, expected_actor_dim=140)
    validate_native_checkpoint(checkpoint)
    override = spec.get("env_override")
    env = create_env(
        cfg,
        num_envs=1,
        env_cfg_override=override if isinstance(override, dict) else None,
        sim_backend="mujoco",
        task_name=str(OmegaConf.select(cfg, "training.task_name") or "SimToolReal"),
    )
    env.set_autoreset(False)
    adapter = RlGamesNpEnvAdapter(
        env, device=str(OmegaConf.select(cfg, "training.device") or "cpu")
    )
    bridge = build_native_player_bridge(
        cfg,
        adapter=adapter,
        checkpoint=checkpoint,
        verify_dependency=True,
        validate_checkpoint=False,
    )
    bridge.player.is_deterministic = bool(spec.get("deterministic", True))
    return env, adapter, bridge


def _playback_worker(conn: Any, spec: dict[str, object]) -> None:
    """Run the command loop in a spawned process."""
    env = adapter = bridge = None
    running = False
    paused = False
    try:
        env, adapter, bridge = _worker_env(spec)
        obs = bridge.initialize()
        import torch

        zero = torch.zeros((1, 29), dtype=torch.float32, device=adapter.device)
        # Match the original evaluator: one zero-action physics tick precedes
        # the first policy action. The adapter returns the post-tick obs.
        # Route the result through the native player's env_step adapter as
        # well: ``adapter.step`` returns the RL-Games dict ABI (``obs`` and
        # ``states``), while the native player expects its actor observation
        # tensor after ``BasePlayer.obs_to_torch`` normalization.
        obs, _reward, _done, _info = bridge.player.env_step(adapter, zero)
        state = _state_from_env(env)
        conn.send(
            {
                **state.as_message(),
                "type": "ready",
                "task_id": spec.get("task_id"),
                "model_file": spec.get("model_file"),
            }
        )
        while True:
            message = _poll_worker_command(conn)
            if message is not None:
                command = (
                    str(message.get("command", message))
                    if isinstance(message, dict)
                    else str(message)
                )
                if command == "quit":
                    conn.send({"type": "stopped", "reason": "quit"})
                    break
                if command == "stop":
                    running = False
                    paused = False
                    conn.send({"type": "stopped", "reason": "stop"})
                    break
                if command == "pause":
                    paused = True
                    conn.send({**state.as_message(), "type": "state", "status": "paused"})
                    continue
                if command == "resume":
                    paused = False
                    running = True
                    conn.send({**state.as_message(), "type": "state", "status": "running"})
                    continue
                if command == "run":
                    running = True
                    paused = False
                    conn.send({**state.as_message(), "type": "state", "status": "running"})
                    continue
                if command == "reset":
                    bridge.player.reset()
                    obs = bridge.initialize()
                    # Keep the post-reset observation on the native player ABI;
                    # ``adapter.step`` returns an ``{"obs", "states"}`` dict,
                    # which would become an unhashable slice operand on the
                    # next ``get_action`` call.
                    obs, _reward, _done, _info = bridge.player.env_step(adapter, zero)
                    state = _state_from_env(env)
                    running = False
                    conn.send({**state.as_message(), "type": "state", "status": "ready"})
                    continue
                if command == "load":
                    conn.send(
                        {
                            "type": "error",
                            "error": "load requires a new worker",
                            "recoverable": True,
                        }
                    )
                    continue
                conn.send(
                    {
                        "type": "error",
                        "error": f"unknown worker command: {command}",
                        "recoverable": True,
                    }
                )
            if not running or paused:
                continue
            with torch.inference_mode():
                obs = bridge.step(obs)
            done = bool(bridge.last_done is not None and bridge.last_done[0].item())
            if done:
                state = _state_from_env(env, episode_result="terminated")
                conn.send({**state.as_message(), "type": "done"})
                running = False
            else:
                state = _state_from_env(env)
                conn.send(state.as_message())
            ctrl_dt = float(getattr(env.cfg, "ctrl_dt", 0.0))
            if ctrl_dt > 0:
                time.sleep(min(ctrl_dt, 0.1))
    except BaseException as exc:
        try:
            conn.send(
                {
                    "type": "error",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "recoverable": False,
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


@dataclass
class DexBenchPlaybackController:
    """Parent-side worker lifecycle and backwards-compatible local status API."""

    status: str = "empty"
    category: str | None = None
    object_name: str | None = None
    task_name: str | None = None
    trajectory_file: Path | None = None
    episodes: int = 0
    total_goal_pct: float = 0.0
    total_steps: int = 0
    worker_target: Callable[[Any, dict[str, object]], None] = _playback_worker
    join_timeout: float = 3.0

    def __post_init__(self) -> None:
        self._process: Any | None = None
        self._conn: Any | None = None
        self._last_state: dict[str, object] | None = None
        self.last_error: str | None = None

    @property
    def process(self) -> Any | None:
        return self._process

    @property
    def last_state(self) -> dict[str, object] | None:
        """Most recent JSON-compatible state received from the worker."""
        return self._last_state

    def load(
        self, category: str, object_name: str, task_name: str, trajectory_file: str | Path
    ) -> None:
        self.category = str(category)
        self.object_name = str(object_name)
        self.task_name = str(task_name)
        self.trajectory_file = Path(trajectory_file)
        self.status = "ready"
        self.episodes = 0
        self.total_goal_pct = 0.0
        self.total_steps = 0

    def start(
        self,
        *,
        category: str,
        object_name: str,
        task_name: str,
        trajectory_file: str | Path,
        cfg: object | None = None,
        checkpoint: str | Path | None = None,
        env_override: dict[str, object] | None = None,
        model_file: str | Path | None = None,
        deterministic: bool = True,
    ) -> None:
        self.stop()
        if checkpoint is None and self.worker_target is _playback_worker:
            raise ValueError("checkpoint is required for a MuJoCo playback worker")
        self.load(category, object_name, task_name, trajectory_file)
        if cfg is None:
            cfg_payload: object = {}
        else:
            from omegaconf import OmegaConf

            cfg_payload = OmegaConf.to_container(cfg, resolve=False)
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        spec: dict[str, object] = {
            "task_id": f"{category}/{object_name}/{task_name}",
            "trajectory_file": str(Path(trajectory_file).resolve()),
            "cfg": cfg_payload,
            "checkpoint": str(Path(checkpoint).resolve()) if checkpoint is not None else "",
            "env_override": env_override or {},
            "model_file": str(Path(model_file).resolve()) if model_file is not None else None,
            "deterministic": bool(deterministic),
        }
        process = context.Process(target=self.worker_target, args=(child, spec), daemon=True)
        process.start()
        child.close()
        self._conn = parent
        self._process = process
        self.status = "loading"

    def handle(self, command: str) -> None:
        if self._conn is None:
            return
        if not self._process or not self._process.is_alive():
            return
        command = str(command).lower()
        if command not in {"load", "run", "pause", "resume", "stop", "reset", "quit"}:
            raise ValueError(f"unknown playback command: {command}")
        try:
            self._conn.send({"command": command})
        except (BrokenPipeError, EOFError, OSError):
            self.status = "error"

    def poll(self) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []
        if self._conn is None:
            return messages
        while True:
            try:
                if not self._conn.poll():
                    break
                message = self._conn.recv()
            except (EOFError, OSError):
                break
            if not isinstance(message, dict):
                continue
            messages.append(message)
            kind = str(message.get("type", ""))
            if kind in {"ready", "state"}:
                self._last_state = message
                default = "ready" if kind == "ready" else self.status
                self.status = str(message.get("status", default))
            elif kind == "done":
                self._last_state = message
                self.status = "ready"
                self.episodes += 1
                count = int(message.get("goal_count", 0))
                index = int(message.get("goal_index", 0))
                self.total_goal_pct += 100.0 * index / count if count else 0.0
                self.total_steps += int(message.get("step", 0))
            elif kind == "error":
                self.last_error = str(message.get("error", "worker error"))
                self.status = "error"
            elif kind == "stopped":
                self.status = "ready"
        if self._process is not None and not self._process.is_alive():
            if self.status not in {"ready", "error"}:
                self.status = "error"
                self.last_error = self.last_error or "playback worker exited unexpectedly"
            try:
                self._conn.close()
            except (OSError, AttributeError):
                pass
            self._conn = None
            self._process = None
        return messages

    def request_run(self) -> bool:
        if self.status not in {"ready", "paused"}:
            return False
        self.handle("run")
        self.status = "running"
        return True

    def toggle_pause(self) -> None:
        if self.status == "running":
            self.handle("pause")
            self.status = "paused"
        elif self.status == "paused":
            self.handle("resume")
            self.status = "running"

    def stop(self) -> None:
        process, conn = self._process, self._conn
        if process is None:
            if self.status in {"running", "paused"}:
                self.status = "ready"
            return
        try:
            if process.is_alive() and conn is not None:
                try:
                    conn.send({"command": "stop"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
                process.join(timeout=self.join_timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=self.join_timeout)
        finally:
            if conn is not None:
                conn.close()
            self._conn = None
            self._process = None
            self.status = "ready"

    def reload(self, **kwargs: object) -> None:
        self.stop()
        self.start(**kwargs)  # type: ignore[arg-type]

    def complete(self, *, goal_pct: float, steps: int) -> None:
        self.episodes += 1
        self.total_goal_pct += float(goal_pct)
        self.total_steps += int(steps)
        self.status = "ready"

    @property
    def avg_goal_pct(self) -> float:
        return self.total_goal_pct / self.episodes if self.episodes else 0.0

    @property
    def avg_steps(self) -> float:
        return self.total_steps / self.episodes if self.episodes else 0.0


__all__ = ["DexBenchPlaybackController", "DexBenchState"]
