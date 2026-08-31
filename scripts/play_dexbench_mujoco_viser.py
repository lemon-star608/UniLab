# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""DexToolBench task viewer backed by UniLab's MuJoCo SimToolReal env.

The original DexToolBench viewer exposes cascading category/object/task selectors
and an explicit Load/Run/Pause/Stop lifecycle.  This entrypoint keeps that
workflow while replacing Isaac Gym with the registered UniLab MuJoCo backend.
The policy is the native RL-Games SAPG player; each rendered frame is obtained
from the live environment physics state, never from a video or an offline
trajectory renderer.

Usage::

    uv sync --extra mujoco --extra viser --extra rlgames-sapg
    uv run scripts/play_dexbench_mujoco_viser.py \
      task=simtoolreal/mujoco_12k \
      algo.load_run=0_2026-08-22_11-50-43_mujoco \
      algo.checkpoint=nn/0_simtoolreal_sapg.pth \
      dexbench.port=8083
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.algos.torch.rlgames_sapg.checkpoint import resolve_native_checkpoint
from unilab.algos.torch.rlgames_sapg.checkpoint_normalize import preflight_checkpoint
from unilab.envs.manipulation.simtoolreal.dexbench_assets import (
    DexBenchTaskAssets,
    DexBenchTrajectory,
    build_dexbench_eval_override,
    load_dexbench_trajectory,
    resolve_dexbench_task,
)
from unilab.training import ensure_registries, get_log_root
from unilab.visualization.dexbench_mujoco_playback import DexBenchPlaybackController
from unilab.visualization.viser_scene import MujocoViserScene

try:
    from viser.extras import ViserUrdf
except ImportError:  # pragma: no cover - optional dependency
    ViserUrdf = None  # type: ignore[assignment,misc]

ensure_registries()

DEXBENCH_CATEGORY_DESCRIPTIONS = {
    "hammer": "Swing a hammer to hit a nail.",
    "spatula": "Flip or serve food with a spatula.",
    "eraser": "Wipe a whiteboard with an eraser.",
    "screwdriver": "Drive a screw from the top or side.",
    "marker": "Write shapes on a whiteboard.",
    "brush": "Sweep debris forward across the table.",
}

# The worker advances at ``ctrl_dt=1/60`` by default.  Keep the parent-side
# polling cadence in the same range so state messages reach the browser without
# introducing a visible one-second burst/update delay.
VISER_REFRESH_HZ = 60.0


def _snake_to_title(value: str) -> str:
    return value.replace("_", " ").title()


def _display_to_snake(value: str) -> str:
    return value.lower().replace(" ", "_")


def _resolved_device(cfg: DictConfig) -> str:
    configured = str(OmegaConf.select(cfg, "training.device") or "cpu")
    if configured.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return configured


def _initialize_render_data(model: Any) -> Any:
    """Create render data in the task's declared initial pose.

    ``MjData(model)`` starts from ``model.qpos0`` and does not implicitly apply
    a task-level keyframe.  Imported DexToolBench scenes store the authored
    robot/object start pose in keyframe 0, so apply it on the cold path before
    the first worker state arrives.  The worker's live physics snapshot still
    remains authoritative once rollout begins.
    """
    import mujoco

    data = mujoco.MjData(model)
    if int(model.nkey) > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return data


def _initial_visual_poses(
    trajectory: DexBenchTrajectory,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the selected task's start and first-goal poses for Viser frames."""
    start = tuple(float(value) for value in trajectory.start_pose_wxyz)
    goal_pos = np.asarray(trajectory.goal_pos[0], dtype=np.float64).reshape(3)
    goal_quat = np.asarray(trajectory.goal_quat_wxyz[0], dtype=np.float64).reshape(4)
    goal = tuple(float(value) for value in (*goal_pos, *goal_quat))
    return start, goal


def build_dexbench_env_override(
    cfg: DictConfig,
    *,
    object_urdf: str | Path,
    object_scale: tuple[float, float, float],
    table_urdf: str | Path,
    trajectory_file: str | Path,
    trajectory: DexBenchTrajectory,
    materialized_scene: str | Path | None = None,
) -> dict[str, Any]:
    """Build the single-task evaluation override without changing task owners.

    All paths are consumed by the environment's cold path.  The returned
    mapping deliberately leaves reward, tool generation, and frequency owners
    untouched except for evaluation success/termination fields.
    """
    sim_dt = float(OmegaConf.select(cfg, "env.sim_dt") or (1.0 / 120.0))
    ctrl_dt = float(OmegaConf.select(cfg, "env.ctrl_dt") or (1.0 / 60.0))
    return build_dexbench_eval_override(
        sim_dt=sim_dt,
        ctrl_dt=ctrl_dt,
        object_urdf=object_urdf,
        object_scale=object_scale,
        table_urdf=table_urdf,
        trajectory_file=trajectory_file,
        trajectory=trajectory,
        materialized_scene=materialized_scene,
    )


class DexBenchInteractiveDemo:
    """Own the DexToolBench GUI and one live MuJoCo policy environment."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        root_dir: str | Path = ROOT_DIR,
        server: Any | None = None,
        **_legacy_factories: Any,
    ) -> None:
        self.cfg = cfg
        self.root_dir = Path(root_dir).resolve()
        self._server = server
        self.controller = DexBenchPlaybackController()
        self._task: DexBenchTaskAssets | None = None
        self._trajectory: DexBenchTrajectory | None = None
        self._trajectory_tmp: tempfile.TemporaryDirectory[str] | None = None
        self._scene: MujocoViserScene | None = None
        self._goal_handle: Any | None = None
        self._object_visual: Any | None = None
        self._goal_visual: Any | None = None
        self._object_visual_frame: Any | None = None
        self._goal_visual_frame: Any | None = None
        self._render_model: Any | None = None
        self._render_data: Any | None = None

        self._dd_cat: Any | None = None
        self._dd_obj: Any | None = None
        self._dd_task: Any | None = None
        self._btn_pause: Any | None = None
        self._md_status: Any | None = None
        self._md_task: Any | None = None
        self._md_progress: Any | None = None
        self._md_stats: Any | None = None
        self._md_object: Any | None = None

    @property
    def server(self) -> Any:
        if self._server is None:
            try:
                import viser
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("DexBench Viser requires: uv sync --extra viser") from exc
            port = int(OmegaConf.select(self.cfg, "dexbench.port") or 8080)
            self._server = viser.ViserServer(host="0.0.0.0", port=port)
        return self._server

    def _set_markdown(self, handle: Any | None, value: str) -> None:
        if handle is not None:
            handle.content = value

    def _category_key(self) -> str | None:
        if self._dd_cat is None:
            return None
        display = str(self._dd_cat.value)
        for category in DEXBENCH_CATEGORY_DESCRIPTIONS:
            if _snake_to_title(category) == display:
                return category
        return None

    def build_gui(self) -> None:
        """Create the same cascading selectors and lifecycle buttons as DexBench."""
        gui = self.server.gui
        gui.add_markdown("# DexToolBench\n### UniLab MuJoCo Interactive Policy Demo")
        placeholder = "-- Select --"
        with gui.add_folder("Dataset Selection", expand_by_default=True):
            categories = [placeholder] + [
                _snake_to_title(value) for value in sorted(DEXBENCH_CATEGORY_DESCRIPTIONS)
            ]
            configured_category = str(OmegaConf.select(self.cfg, "dexbench.category") or "")
            initial_category = (
                _snake_to_title(configured_category)
                if configured_category in DEXBENCH_CATEGORY_DESCRIPTIONS
                else placeholder
            )
            self._dd_cat = gui.add_dropdown(
                "Tool Category", options=categories, initial_value=initial_category
            )
            self._dd_obj = gui.add_dropdown(
                "Object Instance", options=[placeholder], initial_value=placeholder
            )
            self._dd_task = gui.add_dropdown(
                "Task", options=[placeholder], initial_value=placeholder
            )
            description = gui.add_markdown("*Select a tool category to begin.*")
            self._md_status = gui.add_markdown("**Status:** Ready")
            load_button = gui.add_button("Load Environment")
            load_button.on_click(lambda _: self.load_selected_task())
            self._dd_cat.on_update(lambda _: self._on_category_change(description))
            self._dd_obj.on_update(lambda _: self._on_object_change())
            if initial_category != placeholder:
                self._on_category_change(description)
                configured_object = str(OmegaConf.select(self.cfg, "dexbench.object_name") or "")
                if (
                    self._dd_obj.options
                    and _snake_to_title(configured_object) in self._dd_obj.options
                ):
                    self._dd_obj.value = _snake_to_title(configured_object)
                    self._on_object_change()
                configured_task = str(OmegaConf.select(self.cfg, "dexbench.task_name") or "")
                if (
                    self._dd_task.options
                    and _snake_to_title(configured_task) in self._dd_task.options
                ):
                    self._dd_task.value = _snake_to_title(configured_task)

        with gui.add_folder("Episode Controls", expand_by_default=True):
            run_button = gui.add_button("Run Episode")
            run_button.on_click(lambda _: self.run_episode())
            self._btn_pause = gui.add_button("Pause")
            self._btn_pause.on_click(lambda _: self.toggle_pause())
            stop_button = gui.add_button("Stop Episode")
            stop_button.on_click(lambda _: self.stop_episode())

        with gui.add_folder("Status", expand_by_default=True):
            self._md_task = gui.add_markdown("**Task:** --")
            self._md_progress = gui.add_markdown("**Progress:** --")
            self._md_stats = gui.add_markdown("**Stats:** No episodes yet")
            self._md_object = gui.add_markdown("**Object Pos:** --")

    def _on_category_change(self, description: Any | None = None) -> None:
        category = self._category_key()
        if category is None or self._dd_obj is None or self._dd_task is None:
            return
        from unilab.envs.manipulation.simtoolreal.dexbench_assets import DEXTOOLBENCH_DATA_STRUCTURE

        objects = DEXTOOLBENCH_DATA_STRUCTURE[category]
        object_values = [_snake_to_title(value) for value in objects]
        self._dd_obj.options = object_values
        self._dd_obj.value = object_values[0]
        task_values = [_snake_to_title(value) for value in objects[next(iter(objects))]]
        self._dd_task.options = task_values
        self._dd_task.value = task_values[0]
        if description is not None:
            description.content = f"*{DEXBENCH_CATEGORY_DESCRIPTIONS[category]}*"

    def _on_object_change(self) -> None:
        category = self._category_key()
        if category is None or self._dd_obj is None or self._dd_task is None:
            return
        from unilab.envs.manipulation.simtoolreal.dexbench_assets import DEXTOOLBENCH_DATA_STRUCTURE

        object_name = _display_to_snake(str(self._dd_obj.value))
        tasks = DEXTOOLBENCH_DATA_STRUCTURE.get(category, {}).get(object_name, ())
        task_values = [_snake_to_title(value) for value in tasks]
        if task_values:
            self._dd_task.options = task_values
            self._dd_task.value = task_values[0]

    def _clear_scene(self) -> None:
        for visual, frame in (
            (self._object_visual, self._object_visual_frame),
            (self._goal_visual, self._goal_visual_frame),
        ):
            if visual is not None:
                try:
                    visual.remove()
                except Exception:
                    pass
            if frame is not None:
                try:
                    frame.remove()
                except Exception:
                    pass
        self._object_visual = None
        self._goal_visual = None
        self._object_visual_frame = None
        self._goal_visual_frame = None
        if self._scene is not None:
            self._scene.close()
            self._scene = None
        if self._goal_handle is not None:
            try:
                self._goal_handle.remove()
            except Exception:
                pass
            self._goal_handle = None

    def _close_environment(self) -> None:
        self.controller.stop()
        self._clear_scene()
        if self._trajectory_tmp is not None:
            self._trajectory_tmp.cleanup()
            self._trajectory_tmp = None
        self.controller.status = "empty"

    def load_selected_task(self) -> None:
        """Resolve and load one task, keeping all heavy work off the hot path."""
        category = self._category_key()
        if category is None or self._dd_obj is None or self._dd_task is None:
            self._set_markdown(self._md_status, "**Status:** Please select a tool category first.")
            return
        object_name = _display_to_snake(str(self._dd_obj.value))
        task_name = _display_to_snake(str(self._dd_task.value))
        source_root = Path(
            str(
                OmegaConf.select(self.cfg, "dexbench.manifest")
                or (self.root_dir / "src" / "unilab" / "assets" / "dexbench" / "manifest.json")
            )
        ).expanduser()
        if not source_root.is_absolute():
            source_root = (self.root_dir / source_root).resolve()
        label = f"{_snake_to_title(category)} / {_snake_to_title(object_name)} / {_snake_to_title(task_name)}"
        self._set_markdown(self._md_status, f"**Status:** Loading *{label}* ...")
        self._set_markdown(self._md_task, f"**Task:** {label}")
        try:
            self._close_environment()
            task = resolve_dexbench_task(source_root, category, object_name, task_name)
            configured_z_offset = OmegaConf.select(self.cfg, "dexbench.z_offset")
            z_offset = 0.03 if configured_z_offset is None else float(configured_z_offset)
            trajectory = load_dexbench_trajectory(task, z_offset=z_offset)
            temp_root = OmegaConf.select(self.cfg, "dexbench.temp_root")
            self._trajectory_tmp = tempfile.TemporaryDirectory(
                prefix="unilab_dexbench_traj_",
                dir=str(temp_root) if temp_root else None,
            )
            trajectory_file = Path(self._trajectory_tmp.name) / "trajectory.json"
            trajectory_file.write_text(
                json.dumps(
                    {
                        "pos": trajectory.goal_pos[None, :, :].tolist(),
                        "quat_wxyz": trajectory.goal_quat_wxyz[None, :, :].tolist(),
                    }
                ),
                encoding="utf-8",
            )
            env_cfg = self.cfg
            checkpoint, source_run = resolve_native_checkpoint(
                get_log_root(self.root_dir, env_cfg) / str(env_cfg.training.task_name),
                load_run=str(env_cfg.algo.load_run),
                checkpoint=str(env_cfg.algo.checkpoint),
            )
            preflight_checkpoint(checkpoint, expected_actor_dim=140)
            override = build_dexbench_env_override(
                env_cfg,
                object_urdf=task.decomposed_urdf,
                object_scale=task.object_scale,
                table_urdf=task.table_urdf,
                trajectory_file=trajectory_file,
                trajectory=trajectory,
                materialized_scene=task.materialized_mjcf,
            )
            self._task = task
            self._trajectory = trajectory
            self._build_live_scene()
            self.controller.start(
                category=category,
                object_name=object_name,
                task_name=task_name,
                trajectory_file=trajectory_file,
                cfg=env_cfg,
                checkpoint=checkpoint,
                env_override=override,
                model_file=task.materialized_mjcf,
                deterministic=bool(
                    OmegaConf.select(env_cfg, "dexbench.deterministic") is not False
                ),
            )
            self._set_markdown(self._md_status, "**Status:** Ready -- click **Run Episode**")
            self._set_markdown(self._md_stats, "**Stats:** No episodes yet")
            print(
                f"[dexbench-viser] loaded {label} checkpoint={checkpoint} source_run={source_run}"
            )
        except Exception as exc:
            self._close_environment()
            self._set_markdown(self._md_status, f"**Status:** Error -- {exc}")
            print(f"[dexbench-viser] load failed: {exc}")

    def _base_scene_file(self) -> Path:
        value = OmegaConf.select(self.cfg, "env.scene.model_file")
        if value:
            return Path(str(value)).expanduser().resolve()
        return (self.root_dir / "src/unilab/assets/robots/kuka_sharpa/scene.xml").resolve()

    def _build_live_scene(self) -> None:
        if self._task is None or self._task.materialized_mjcf is None:
            return
        self._clear_scene()
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(self._task.materialized_mjcf))
        self._render_model = model
        self._render_data = _initialize_render_data(model)
        self._scene = MujocoViserScene(self.server, model, name_prefix="/dexbench/mujoco")
        self._scene.update(self._render_data)
        # The MuJoCo bridge renders geometry-only meshes.  DexToolBench's
        # authored OBJ has MTL/PNG sidecars, so overlay the source visual URDF
        # for textured fidelity and hide only the compiled object/goal visuals.
        # Collision groups are physics-only in the source viewer and would
        # otherwise expose the convex decomposition over the authored mesh.
        self._scene.set_geom_group_visibility((3,), False)
        self._scene.set_geom_visibility(("dex_object_visual_", "dex_goal_visual_"), False)
        if ViserUrdf is not None:
            start_pose, goal_pose = _initial_visual_poses(self._trajectory)
            self._object_visual_frame = self.server.scene.add_frame(
                "/dexbench/object",
                position=start_pose[:3],
                wxyz=start_pose[3:],
                show_axes=False,
            )
            self._object_visual = ViserUrdf(
                self.server,
                self._task.object_urdf,
                root_node_name="/dexbench/object",
            )
            self._goal_visual_frame = self.server.scene.add_frame(
                "/dexbench/goal",
                position=goal_pose[:3],
                wxyz=goal_pose[3:],
                show_axes=False,
            )
            self._goal_visual = ViserUrdf(
                self.server,
                self._task.object_urdf,
                root_node_name="/dexbench/goal",
                mesh_color_override=(0, 255, 0, 0.5),
            )
        self._goal_handle = self.server.scene.add_icosphere(
            "/dexbench/goal_marker", radius=0.025, color=(40, 220, 80), opacity=0.65
        )
        if self._trajectory is not None:
            _, goal_pose = _initial_visual_poses(self._trajectory)
            self._goal_handle.position = tuple(float(value) for value in goal_pose[:3])

    def _render_state(self, message: dict[str, object]) -> None:
        if self._scene is None or self._render_model is None or self._render_data is None:
            return
        import mujoco

        goal_pose = np.asarray(message.get("goal_pose_wxyz", []), dtype=np.float64)
        if goal_pose.shape == (7,):
            goal_id = mujoco.mj_name2id(self._render_model, mujoco.mjtObj.mjOBJ_BODY, "goal_object")
            if goal_id >= 0:
                self._render_model.body_pos[goal_id] = goal_pose[:3]
                self._render_model.body_quat[goal_id] = goal_pose[3:]
        snapshot = message.get("physics_state")
        if isinstance(snapshot, list):
            values = np.asarray(snapshot, dtype=np.float64)
            try:
                mujoco.mj_setState(
                    self._render_model,
                    self._render_data,
                    values,
                    mujoco.mjtState.mjSTATE_FULLPHYSICS,
                )
                mujoco.mj_forward(self._render_model, self._render_data)
            except (ValueError, RuntimeError):
                pass
        self._scene.update(self._render_data)
        if self._goal_handle is not None and goal_pose.shape == (7,):
            self._goal_handle.position = tuple(float(v) for v in goal_pose[:3])
        tool_pose = np.asarray(message.get("tool_pose_wxyz", []), dtype=np.float64)
        if tool_pose.shape == (7,):
            if self._object_visual_frame is not None:
                self._object_visual_frame.position = tuple(float(v) for v in tool_pose[:3])
                self._object_visual_frame.wxyz = tuple(float(v) for v in tool_pose[3:])
            if self._goal_visual_frame is not None and goal_pose.shape == (7,):
                self._goal_visual_frame.position = tuple(float(v) for v in goal_pose[:3])
                self._goal_visual_frame.wxyz = tuple(float(v) for v in goal_pose[3:])
            self._set_markdown(
                self._md_object,
                f"**Object Pos:** {tool_pose[0]:.3f}, {tool_pose[1]:.3f}, {tool_pose[2]:.3f}",
            )

    def _poll_worker(self) -> None:
        messages = self.controller.poll()
        latest_render_message: dict[str, object] | None = None
        for message in messages:
            kind = str(message.get("type", ""))
            if kind in {"ready", "state", "done"}:
                # Rendering every queued state causes stale frames to be sent
                # one by one when the browser or scene update is slower than
                # the worker.  The newest state is the only one that matters
                # for an interactive viewer; lifecycle/status messages are
                # still processed below.
                latest_render_message = message
            if kind == "error":
                self._set_markdown(
                    self._md_status, f"**Status:** Error -- {message.get('error', '')}"
                )
            elif kind == "done":
                self._set_markdown(self._md_status, "**Status:** Episode done")
        if latest_render_message is not None:
            self._render_state(latest_render_message)
            count = int(latest_render_message.get("goal_count", 0))
            index = int(latest_render_message.get("goal_index", 0))
            step = int(latest_render_message.get("step", 0))
            self._set_markdown(
                self._md_progress,
                f"**Step:** {step} &nbsp;|&nbsp; **Goal:** {index}/{count}",
            )

    def run_episode(self) -> None:
        if not self.controller.request_run():
            self._set_markdown(self._md_status, "**Status:** Load an environment first.")
            return
        self._btn_pause.name = "Pause" if self._btn_pause is not None else "Pause"
        self._set_markdown(self._md_status, "**Status:** Running episode...")

    def toggle_pause(self) -> None:
        if self.controller.status not in {"running", "paused"}:
            return
        self.controller.toggle_pause()
        paused = self.controller.status == "paused"
        if self._btn_pause is not None:
            self._btn_pause.name = "Resume" if paused else "Pause"
        self._set_markdown(
            self._md_status, "**Status:** Paused" if paused else "**Status:** Running episode..."
        )

    def stop_episode(self) -> None:
        if self.controller.status in {"running", "paused"}:
            self.controller.stop()
            self._set_markdown(self._md_status, "**Status:** Episode stopped.")

    def run(self) -> None:
        self.build_gui()
        port = int(OmegaConf.select(self.cfg, "dexbench.port") or 8080)
        configured_render_hz = OmegaConf.select(self.cfg, "dexbench.render_hz")
        render_hz = (
            VISER_REFRESH_HZ if configured_render_hz is None else float(configured_render_hz)
        )
        if render_hz <= 0.0:
            raise ValueError(f"dexbench.render_hz must be positive, got {render_hz}")
        print(f"[dexbench-viser] server running at http://localhost:{port}")
        print(
            "[dexbench-viser] Select category/object/task, then Load Environment and Run Episode."
        )
        try:
            while True:
                self._poll_worker()
                time.sleep(1.0 / render_hz)
        except KeyboardInterrupt:
            self.close()

    def close(self) -> None:
        self._close_environment()


def run_dexbench_viser(cfg: DictConfig) -> DexBenchInteractiveDemo:
    demo = DexBenchInteractiveDemo(cfg)
    demo.run()
    return demo


@hydra.main(
    version_base="1.3",
    config_path="../conf/rlgames_sapg",
    config_name="dexbench_mujoco_viser",
)
def main(cfg: DictConfig) -> None:
    run_dexbench_viser(cfg)


if __name__ == "__main__":
    main()
