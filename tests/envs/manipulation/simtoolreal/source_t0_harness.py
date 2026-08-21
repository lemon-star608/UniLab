"""Source-only T0 harness for the Code #7 task-math oracle.

The harness materializes pinned Source utility files into a temporary namespace,
installs only the small ``isaaclab.utils.math`` quaternion surface those files
import, and drives them with a tensor carrier.  It deliberately does not import
any target task module.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import torch

SOURCE_HEAD = "2a9917533bfea70419ed2667a511d7238e5b3abc"
SOURCE_TASK_ROOT = "isaacsimenvs/tasks/simtoolreal"
SOURCE_MODULES = (
    "utils/action_utils.py",
    "utils/goal_sampling.py",
    "utils/obs_utils.py",
    "utils/reward_utils.py",
    "utils/reset_utils.py",
    "utils/termination_utils.py",
    "utils/object_size_distributions.py",
    "utils/generate_objects.py",
)

_PERM_CANON_TO_BACKEND = np.concatenate(
    (np.array([2, 0, 6, 3, 1, 5, 4], dtype=np.int64), np.arange(28, 6, -1, dtype=np.int64))
)
_PERM_BACKEND_TO_CANON = np.argsort(_PERM_CANON_TO_BACKEND)


def _convert_quat(q: torch.Tensor, to: str = "xyzw") -> torch.Tensor:
    if to != "xyzw":
        raise ValueError(to)
    return torch.cat((q[..., 1:], q[..., :1]), dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., 1:]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q_xyz, t, dim=-1)


def _quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    half = angle * 0.5
    return torch.cat((torch.cos(half).unsqueeze(-1), axis * torch.sin(half).unsqueeze(-1)), dim=-1)


def _random_orientation(n: int, device: torch.device | None = None) -> torch.Tensor:
    # Explicit deterministic primitive draw used by the fixture, not a second
    # random sampler.  The generator replaces this function per case.
    q = torch.zeros((n, 4), device=device, dtype=torch.float32)
    q[:, 0] = 1.0
    return q


def _install_math_stub() -> None:
    isaaclab = types.ModuleType("isaaclab")
    utils = types.ModuleType("isaaclab.utils")
    math_mod = types.ModuleType("isaaclab.utils.math")
    math_mod.convert_quat = _convert_quat
    math_mod.quat_apply = _quat_apply
    math_mod.quat_from_angle_axis = _quat_from_angle_axis
    math_mod.quat_mul = _quat_mul
    math_mod.random_orientation = _random_orientation
    isaaclab.utils = utils
    utils.math = math_mod
    sys.modules.update(
        {"isaaclab": isaaclab, "isaaclab.utils": utils, "isaaclab.utils.math": math_mod}
    )


def _load_source_modules(source: Path, temp: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    package = "code7_source_t0"
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(temp)]
    sys.modules[package] = pkg
    utils_pkg = types.ModuleType(f"{package}.utils")
    utils_pkg.__path__ = [str(temp / "utils")]
    sys.modules[f"{package}.utils"] = utils_pkg
    scene_stub = types.ModuleType(f"{package}.utils.scene_utils")
    scene_stub.ARM_JOINT_REGEX = "iiwa14_joint_.*"
    scene_stub.HAND_JOINT_REGEX = "left_.*"
    scene_stub.FINGERTIP_BODY_REGEX = "left_(index|middle|ring|thumb|pinky)_DP"
    scene_stub.JOINT_NAMES_CANONICAL = tuple(f"joint_{i}" for i in range(29))
    scene_stub.PALM_BODY_NAME = "iiwa14_link_7"
    scene_stub._apply_camera_pose_rand_at_reset = lambda env, env_ids: None
    sys.modules[f"{package}.utils.scene_utils"] = scene_stub
    inventory: list[dict[str, str]] = []
    loaded: dict[str, Any] = {}
    # reset_utils is imported before termination_utils; its scene owner is
    # represented only by the narrow constants above and is never executed.
    for rel in SOURCE_MODULES:
        source_path = source / SOURCE_TASK_ROOT / rel
        copied = temp / rel
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(source_path.read_bytes())
        expected = (
            __import__("subprocess")
            .check_output(
                ["git", "-C", str(source), "rev-parse", f"{SOURCE_HEAD}:{SOURCE_TASK_ROOT}/{rel}"]
            )
            .decode()
            .strip()
        )
        actual = (
            __import__("subprocess")
            .check_output(["git", "-C", str(source), "hash-object", str(copied)])
            .decode()
            .strip()
        )
        if actual != expected:
            raise RuntimeError(f"Source blob mismatch for {rel}: {actual} != {expected}")
        name = f"{package}.{rel[:-3].replace('/', '.')}"
        spec = importlib.util.spec_from_file_location(name, copied)
        if spec is None or spec.loader is None:
            raise ImportError(name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[rel[:-3].replace("/", ".")] = module
        inventory.append(
            {
                "path": f"{SOURCE_TASK_ROOT}/{rel}",
                "blob": expected,
                "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
            }
        )
    return loaded, inventory


class _Carrier:
    pass


def _capture_source_tool_specs(module: Any, temp: Path) -> dict[str, np.ndarray]:
    """Parse Source-native 12x1 URDFs into backend-neutral ToolSpec inputs."""
    types_in_order = tuple(dict.fromkeys(dist.type for dist in module.OBJECT_SIZE_DISTRIBUTIONS))
    numpy_state = np.random.get_state()
    try:
        paths, scales = module.generate_handle_head_urdfs(
            types_in_order,
            num_per_type=1,
            out_dir=temp / "source_tool_specs",
            seed=42,
            shuffle=False,
        )
    finally:
        np.random.set_state(numpy_state)

    shape_codes: list[int] = []
    has_head: list[bool] = []
    handle_sizes: list[tuple[float, float, float]] = []
    head_sizes: list[tuple[float, float, float]] = []
    head_positions: list[tuple[float, float, float]] = []
    masses: list[float] = []
    centers: list[tuple[float, float, float]] = []
    inertias: list[tuple[float, float, float]] = []

    for path in paths:
        link = ET.parse(path).getroot().find("link")
        assert link is not None
        collisions = link.findall("collision")
        handle_geometry = collisions[0].find("geometry")
        assert handle_geometry is not None
        box = handle_geometry.find("box")
        cylinder = handle_geometry.find("cylinder")
        if box is not None:
            shape_codes.append(0)
            handle_sizes.append(tuple(float(v) for v in box.attrib["size"].split()))
        else:
            assert cylinder is not None
            shape_codes.append(1)
            length = float(cylinder.attrib["length"])
            diameter = 2.0 * float(cylinder.attrib["radius"])
            handle_sizes.append((length, diameter, diameter))

        has_head.append(len(collisions) == 2)
        if len(collisions) == 2:
            head_box = collisions[1].find("geometry/box")
            head_origin = collisions[1].find("origin")
            assert head_box is not None and head_origin is not None
            head_sizes.append(tuple(float(v) for v in head_box.attrib["size"].split()))
            head_positions.append(tuple(float(v) for v in head_origin.attrib["xyz"].split()))
        else:
            head_sizes.append((0.0, 0.0, 0.0))
            head_positions.append((0.0, 0.0, 0.0))

        inertial = link.find("inertial")
        assert inertial is not None
        origin = inertial.find("origin")
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        assert origin is not None and mass is not None and inertia is not None
        centers.append(tuple(float(v) for v in origin.attrib["xyz"].split()))
        masses.append(float(mass.attrib["value"]))
        inertias.append(tuple(float(inertia.attrib[name]) for name in ("ixx", "iyy", "izz")))

    return {
        "source_tool_authored_shape": np.asarray(shape_codes, dtype=np.int8),
        "source_tool_has_head": np.asarray(has_head, dtype=bool),
        "source_tool_handle_size_full": np.asarray(handle_sizes, dtype=np.float32),
        "source_tool_head_size_full": np.asarray(head_sizes, dtype=np.float32),
        "source_tool_head_pos": np.asarray(head_positions, dtype=np.float32),
        "source_tool_mass": np.asarray(masses, dtype=np.float32),
        "source_tool_com": np.asarray(centers, dtype=np.float32),
        "source_tool_diaginertia": np.asarray(inertias, dtype=np.float32),
        "source_tool_object_scale": np.asarray(scales, dtype=np.float32),
    }


def _source_carrier(n: int, action: np.ndarray, goal_draws: dict[str, np.ndarray]) -> _Carrier:
    env = _Carrier()
    env.num_envs = n
    env.device = torch.device("cpu")
    env.step_dt = 1.0 / 60.0
    env.perm_canon_to_lab = torch.as_tensor(_PERM_CANON_TO_BACKEND)
    env._perm_canon_to_lab = torch.as_tensor(_PERM_CANON_TO_BACKEND)
    env._arm_joint_ids = torch.arange(7)
    env._hand_joint_ids = torch.arange(7, 29)
    env._arm_lower = torch.full((n, 7), -1.0)
    env._arm_upper = torch.full((n, 7), 1.0)
    env._hand_lower = torch.full((n, 22), -1.0)
    env._hand_upper = torch.full((n, 22), 1.0)
    env._cur_targets = torch.zeros((n, 29))
    env._prev_targets = torch.zeros((n, 29))
    env._action_queue = torch.as_tensor(
        np.linspace(-0.2, 0.2, n * 3 * 29, dtype=np.float32).reshape(n, 3, 29)
    )
    env.episode_length_buf = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    env._successes = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    env._replay_target_lab_order = None
    env.cfg = types.SimpleNamespace(
        action=types.SimpleNamespace(
            dof_speed_scale=1.5, arm_moving_average=0.1, hand_moving_average=0.1
        ),
        domain_randomization=types.SimpleNamespace(use_action_delay=True, action_delay_max=3),
    )
    env._action_input = torch.as_tensor(action)
    env._goal_draws = goal_draws
    return env


def generate_source_cases(
    source: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, str]], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="code7_source_t0_") as tmp:
        temp = Path(tmp)
        _install_math_stub()
        modules, inventory = _load_source_modules(source, temp)
        action_mod = modules["utils.action_utils"]
        goal_mod = modules["utils.goal_sampling"]
        obs_mod = modules["utils.obs_utils"]
        reward_mod = modules["utils.reward_utils"]
        reset_mod = modules["utils.reset_utils"]
        term_mod = modules["utils.termination_utils"]
        tool_arrays = _capture_source_tool_specs(modules["utils.generate_objects"], temp)

        n = 6
        actions = np.linspace(-1.0, 1.0, n * 29, dtype=np.float32).reshape(n, 29)
        goal_pos_draw = np.full((n, 3), 0.5, dtype=np.float32)
        goal_u1 = np.full((n, 3), 0.25, dtype=np.float32)
        goal_u2 = np.full((n, 3), 0.75, dtype=np.float32)
        goal_draws = {
            "position_uniform": goal_pos_draw,
            "orientation_u1": goal_u1,
            "orientation_u2": goal_u2,
        }
        env = _source_carrier(n, actions, goal_draws)
        action_queue_initial = env._action_queue.numpy().copy()
        action_delay_idx = np.array([2, 1, 0, 2, 1, 0], dtype=np.int64)
        original_action_randint = action_mod.torch.randint
        action_mod.torch.randint = lambda *args, **kwargs: torch.as_tensor(
            action_delay_idx, device=kwargs.get("device")
        )
        action_mod.apply_action_pipeline(env, torch.as_tensor(actions))
        action_mod.torch.randint = original_action_randint
        action_target = env._cur_targets.detach().numpy().astype(np.float32)

        # Source native absolute goal utility with explicit uniform/orientation draws.
        original_rand = goal_mod.torch.rand
        original_random_orientation = goal_mod.random_orientation
        draws = [torch.as_tensor(goal_pos_draw), torch.as_tensor(goal_u1), torch.as_tensor(goal_u2)]

        def draw_rand(*shape, **kwargs):
            value = draws.pop(0)
            return value.to(kwargs.get("device", "cpu"))

        def draw_orientation(count, device=None):
            u1, u2 = (
                torch.as_tensor(goal_u1, device=device),
                torch.as_tensor(goal_u2, device=device),
            )
            q = torch.zeros((count, 4), dtype=torch.float32, device=device)
            q[:, 0] = torch.sqrt(u1[:, 0]) * torch.cos(2 * torch.pi * u2[:, 2])
            q[:, 1] = torch.sqrt(1 - u1[:, 0]) * torch.sin(2 * torch.pi * u2[:, 1])
            q[:, 2] = torch.sqrt(1 - u1[:, 0]) * torch.cos(2 * torch.pi * u2[:, 1])
            q[:, 3] = torch.sqrt(u1[:, 0]) * torch.sin(2 * torch.pi * u2[:, 2])
            return q

        goal_mod.torch.rand = draw_rand
        goal_mod.random_orientation = draw_orientation
        pos, quat = goal_mod.sample_absolute_goal_pose(
            (-0.35, -0.2, 0.6), (0.35, 0.2, 0.95), 1.0, n, torch.device("cpu")
        )
        goal_mod.torch.rand = original_rand
        goal_mod.random_orientation = original_random_orientation

        # Build a deterministic tensor carrier for Source obs/reward/termination.
        source_env = _Carrier()
        source_env.num_envs = n
        source_env.device = torch.device("cpu")
        source_env.scene = types.SimpleNamespace(env_origins=torch.zeros((n, 3)))
        source_env._perm_lab_to_canon = torch.as_tensor(_PERM_BACKEND_TO_CANON)
        source_env._joint_lower_canon = torch.full((29,), -1.0)
        source_env._joint_upper_canon = torch.full((29,), 1.0)
        source_env._prev_targets = env._prev_targets.clone()
        source_env.robot = types.SimpleNamespace(
            data=types.SimpleNamespace(
                joint_pos=torch.zeros((n, 29)),
                joint_vel=torch.zeros((n, 29)),
                body_state_w=torch.zeros((n, 6, 13)),
            )
        )
        source_env.robot.data.body_state_w[:, :, 3] = 1.0
        source_env._palm_body_id = 0
        source_env._fingertip_body_ids = torch.arange(1, 6)
        object_pos_input = np.zeros((n, 3), dtype=np.float32)
        object_pos_input[:, 2] = np.array([0.0, 0.05, 0.11, 0.2, 0.3, 0.4], dtype=np.float32)
        source_env.object = types.SimpleNamespace(
            data=types.SimpleNamespace(
                root_pos_w=torch.as_tensor(object_pos_input),
                root_quat_w=torch.tensor(np.tile([1, 0, 0, 0], (n, 1)), dtype=torch.float32),
                root_lin_vel_w=torch.zeros((n, 3)),
                root_ang_vel_w=torch.zeros((n, 3)),
            )
        )
        source_env.goal_viz = types.SimpleNamespace(
            data=types.SimpleNamespace(root_pos_w=pos, root_quat_w=quat)
        )
        corners = np.array([[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=np.float32)
        source_env._keypoint_offsets = torch.tensor(
            np.tile(corners[None, :, :] * 0.03, (n, 1, 1)), dtype=torch.float32
        )
        source_env._object_scale_multiplier = torch.ones((n, 3), dtype=torch.float32)
        source_env._object_scale_per_env = torch.ones((n, 3), dtype=torch.float32)
        source_env._closest_keypoint_max_dist = torch.full((n,), -1.0)
        source_env._closest_fingertip_dist = torch.full((n, 5), -1.0)
        source_env._lifted_object = torch.zeros(n, dtype=torch.bool)
        source_env.episode_length_buf = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        source_env._successes = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
        source_env.reward_buf = torch.linspace(0, 5, n)
        source_env.cfg = types.SimpleNamespace(
            obs=types.SimpleNamespace(
                state_list=(
                    "joint_pos",
                    "joint_vel",
                    "prev_action_targets",
                    "palm_pos",
                    "palm_rot",
                    "palm_vel",
                    "object_rot",
                    "object_vel",
                    "fingertip_pos_rel_palm",
                    "keypoints_rel_palm",
                    "keypoints_rel_goal",
                    "object_scales",
                    "closest_keypoint_max_dist",
                    "closest_fingertip_dist",
                    "lifted_object",
                    "progress",
                    "successes",
                    "reward",
                ),
                obs_list=(
                    "joint_pos",
                    "joint_vel",
                    "prev_action_targets",
                    "palm_pos",
                    "palm_rot",
                    "object_rot",
                    "fingertip_pos_rel_palm",
                    "keypoints_rel_palm",
                    "keypoints_rel_goal",
                    "object_scales",
                ),
                clamp_abs_observations=10.0,
            ),
            domain_randomization=types.SimpleNamespace(
                use_object_state_delay_noise=True,
                object_state_xyz_noise_std=0.01,
                object_state_rotation_noise_degrees=5.0,
                use_obs_delay=True,
                joint_velocity_obs_noise_std=0.1,
            ),
        )
        object_state_queue_initial = np.linspace(-0.3, 0.3, n * 3 * 13, dtype=np.float32).reshape(
            n, 3, 13
        )
        obs_queue_initial = np.linspace(-0.1, 0.1, n * 3 * 140, dtype=np.float32).reshape(n, 3, 140)
        source_env._object_state_queue = torch.as_tensor(object_state_queue_initial.copy())
        source_env._obs_queue = torch.as_tensor(obs_queue_initial.copy())
        object_delay_idx = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        obs_delay_idx = np.array([2, 1, 0, 2, 1, 0], dtype=np.int64)
        object_pos_normal = np.linspace(-1, 1, n * 3, dtype=np.float32).reshape(n, 3)
        object_quat_axis_normal = np.tile(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), (n, 1))
        object_quat_angle_deg = np.linspace(-5, 5, n, dtype=np.float32)
        joint_velocity_normal = np.linspace(-1, 1, n * 29, dtype=np.float32).reshape(n, 29)
        original_randint = obs_mod.torch.randint
        original_randn_like = obs_mod.torch.randn_like
        original_randn = obs_mod.torch.randn
        original_empty = obs_mod.torch.empty
        delay_draws = [object_delay_idx, obs_delay_idx]
        normal_like_draws = [object_pos_normal, joint_velocity_normal]

        class _ExplicitUniform:
            def uniform_(self, lo, hi):
                del lo, hi
                return torch.as_tensor(object_quat_angle_deg)

        obs_mod.torch.randint = lambda *args, **kwargs: torch.as_tensor(
            delay_draws.pop(0), device=kwargs.get("device")
        )
        obs_mod.torch.randn_like = lambda value: torch.as_tensor(
            normal_like_draws.pop(0), device=value.device, dtype=value.dtype
        )
        obs_mod.torch.randn = lambda *shape, **kwargs: torch.as_tensor(
            object_quat_axis_normal, device=kwargs.get("device")
        )
        obs_mod.torch.empty = lambda *shape, **kwargs: _ExplicitUniform()
        source_obs = obs_mod.build_observations(source_env)
        obs_mod.torch.randint = original_randint
        obs_mod.torch.randn_like = original_randn_like
        obs_mod.torch.randn = original_randn
        obs_mod.torch.empty = original_empty

        source_env.cfg.reward = types.SimpleNamespace(
            lifting_bonus_threshold=0.15,
            lifting_bonus=300.0,
            lifting_rew_scale=20.0,
            distance_delta_rew_scale=50.0,
            keypoint_rew_scale=200.0,
            kuka_actions_penalty_scale=0.03,
            hand_actions_penalty_scale=0.003,
            reach_goal_bonus=1000.0,
            fixed_size_keypoint_reward=True,
            keypoint_scale=1.5,
        )
        source_env.cfg.termination = types.SimpleNamespace(
            success_steps=10, force_consecutive_near_goal_steps=False, max_consecutive_successes=50
        )
        source_env._object_init_z = torch.zeros(n)
        source_env._keypoint_offsets_fixed = source_env._keypoint_offsets.clone()
        source_env._near_goal_steps = torch.zeros(n, dtype=torch.long)
        source_env._current_success_tolerance = 0.075
        source_env._arm_joint_ids = torch.arange(7)
        source_env._hand_joint_ids = torch.arange(7, 29)
        obs_mod.compute_intermediate_values(source_env)
        source_reward = reward_mod.compute_rewards(source_env)
        source_env.max_episode_length = 600
        source_term, source_trunc = term_mod.compute_terminations(source_env)

        # Source native full reset with explicit primitive draws. The fixed
        # MuJoCo table mapping is represented by a zero table-z draw; object
        # orientation still exercises full SO(3).
        reset_env = _Carrier()
        reset_env.num_envs = n
        reset_env.device = torch.device("cpu")
        reset_env.scene = types.SimpleNamespace(env_origins=torch.zeros((n, 3)))
        reset_env._arm_joint_ids = torch.arange(7)
        reset_env._hand_joint_ids = torch.arange(7, 29)
        reset_env._perm_lab_to_canon = torch.as_tensor(_PERM_BACKEND_TO_CANON)
        reset_env._perm_canon_to_lab = torch.as_tensor(_PERM_CANON_TO_BACKEND)
        reset_joint_uniform = np.linspace(0.05, 0.95, n * 29, dtype=np.float32).reshape(n, 29)
        reset_joint_velocity = np.linspace(-0.5, 0.5, n * 29, dtype=np.float32).reshape(n, 29)
        reset_object_uniform = np.linspace(-1.0, 1.0, n * 3, dtype=np.float32).reshape(n, 3)
        reset_orientation_uniform = np.linspace(0.1, 0.9, n * 3, dtype=np.float32).reshape(n, 3)
        reset_orientation = np.empty((n, 4), dtype=np.float32)
        reset_orientation[:, 0] = np.sqrt(reset_orientation_uniform[:, 0]) * np.cos(
            2 * np.pi * reset_orientation_uniform[:, 2]
        )
        reset_orientation[:, 1] = np.sqrt(1 - reset_orientation_uniform[:, 0]) * np.sin(
            2 * np.pi * reset_orientation_uniform[:, 1]
        )
        reset_orientation[:, 2] = np.sqrt(1 - reset_orientation_uniform[:, 0]) * np.cos(
            2 * np.pi * reset_orientation_uniform[:, 1]
        )
        reset_orientation[:, 3] = np.sqrt(reset_orientation_uniform[:, 0]) * np.sin(
            2 * np.pi * reset_orientation_uniform[:, 2]
        )
        reset_scale_multiplier = np.linspace(0.9, 1.1, n * 3, dtype=np.float32).reshape(n, 3)
        reset_force_prob = np.geomspace(0.001, 0.1, n).astype(np.float32)
        reset_torque_prob = np.geomspace(0.1, 0.001, n).astype(np.float32)
        captured: dict[str, torch.Tensor] = {}
        robot_data = types.SimpleNamespace(
            default_joint_pos=torch.zeros((n, 29)),
            joint_pos_limits=torch.stack(
                (torch.full((n, 29), -1.0), torch.full((n, 29), 1.0)), dim=-1
            ),
        )
        reset_env.robot = types.SimpleNamespace(
            data=robot_data,
            write_joint_state_to_sim=lambda q, qd, env_ids: captured.update(
                joint_pos=q.clone(), joint_vel=qd.clone()
            ),
        )
        reset_env.table = types.SimpleNamespace(
            write_root_pose_to_sim=lambda pose, env_ids: captured.update(table_pose=pose.clone())
        )
        reset_env.object = types.SimpleNamespace(
            write_root_pose_to_sim=lambda pose, env_ids: captured.update(object_pose=pose.clone()),
            write_root_velocity_to_sim=lambda velocity, env_ids: captured.update(
                object_velocity=velocity.clone()
            ),
        )
        reset_env.goal_viz = types.SimpleNamespace(
            write_root_pose_to_sim=lambda pose, env_ids: captured.update(goal_pose=pose.clone())
        )
        reset_env._table_z_per_env = torch.full((n,), 0.38)
        reset_env._object_init_z = torch.zeros(n)
        reset_env._prev_targets = torch.zeros((n, 29))
        reset_env._cur_targets = torch.zeros((n, 29))
        reset_env._prev_episode_successes = torch.zeros(n, dtype=torch.long)
        reset_env._successes = torch.arange(n, dtype=torch.long)
        reset_env._closest_keypoint_max_dist = torch.zeros(n)
        reset_env._closest_fingertip_dist = torch.zeros((n, 5))
        reset_env._near_goal_steps = torch.ones(n, dtype=torch.long)
        reset_env._lifted_object = torch.ones(n, dtype=torch.bool)
        reset_env._action_queue = torch.ones((n, 3, 29))
        reset_env._obs_queue = torch.ones((n, 3, 140))
        reset_env._object_state_queue = torch.ones((n, 3, 13))
        reset_env._object_forces = torch.ones((n, 1, 3))
        reset_env._object_torques = torch.ones((n, 1, 3))
        reset_env._random_force_prob = torch.zeros(n)
        reset_env._random_torque_prob = torch.zeros(n)
        reset_env._object_scale_multiplier = torch.ones((n, 3))
        reset_env.cfg = types.SimpleNamespace(
            reset=types.SimpleNamespace(
                reset_dof_pos_random_interval_arm=0.1,
                reset_dof_pos_random_interval_fingers=0.1,
                reset_dof_vel_random_interval=0.5,
                table_reset_z=0.38,
                table_reset_z_range=0.0,
                table_reset_xy_range_m=(0.0, 0.0),
                table_reset_yaw_range_deg=0.0,
                fixed_start_pose=None,
                reset_position_noise_x=0.1,
                reset_position_noise_y=0.1,
                reset_position_noise_z=0.02,
                table_object_z_offset=0.25,
                fixed_goal_pose=None,
                fixed_trajectory_file="",
                goal_sampling_type="absolute",
                target_volume_mins=(-0.35, -0.2, 0.6),
                target_volume_maxs=(0.35, 0.2, 0.95),
                target_volume_region_scale=1.0,
            ),
            domain_randomization=types.SimpleNamespace(
                force_prob_range=(0.001, 0.1),
                torque_prob_range=(0.001, 0.1),
                object_scale_noise_multiplier_range=(0.9, 1.1),
            ),
        )
        original_rand_like = reset_mod.torch.rand_like
        original_empty_like = reset_mod.torch.empty_like
        original_reset_empty = reset_mod.torch.empty
        original_reset_orientation = reset_mod.random_orientation
        original_goal_rand = goal_mod.torch.rand
        original_goal_orientation = goal_mod.random_orientation

        class _ResetUniform:
            def __init__(self, value):
                self.value = torch.as_tensor(value)

            def uniform_(self, lo, hi):
                del lo, hi
                return self.value

        empty_values = [
            np.zeros(n, dtype=np.float32),
            reset_object_uniform,
            np.log(reset_force_prob),
            np.log(reset_torque_prob),
            reset_scale_multiplier,
        ]
        reset_mod.torch.rand_like = lambda value: torch.as_tensor(
            reset_joint_uniform[:, _PERM_CANON_TO_BACKEND], dtype=value.dtype
        )
        reset_mod.torch.empty_like = lambda value: _ResetUniform(
            reset_joint_velocity[:, _PERM_CANON_TO_BACKEND]
        )
        reset_mod.torch.empty = lambda *shape, **kwargs: _ResetUniform(empty_values.pop(0))
        reset_mod.random_orientation = lambda count, device=None: torch.as_tensor(
            reset_orientation, device=device
        )
        goal_reset_draws = [torch.as_tensor(goal_pos_draw)]
        goal_mod.torch.rand = lambda *shape, **kwargs: goal_reset_draws.pop(0).to(
            kwargs.get("device", "cpu")
        )
        goal_mod.random_orientation = lambda count, device=None: torch.as_tensor(
            quat.numpy(), device=device
        )
        reset_mod.reset_env_state(reset_env, torch.arange(n))
        reset_mod.torch.rand_like = original_rand_like
        reset_mod.torch.empty_like = original_empty_like
        reset_mod.torch.empty = original_reset_empty
        reset_mod.random_orientation = original_reset_orientation
        goal_mod.torch.rand = original_goal_rand
        goal_mod.random_orientation = original_goal_orientation
        reset_qpos = np.concatenate(
            (captured["joint_pos"].numpy(), captured["object_pose"].numpy()), axis=1
        ).astype(np.float32)
        reset_qvel = np.concatenate(
            (captured["joint_vel"].numpy(), captured["object_velocity"].numpy()), axis=1
        ).astype(np.float32)

        # Source native wrench transform with explicit Bernoulli and normal draws.
        wrench_env = _Carrier()
        wrench_env.num_envs = n
        wrench_env.device = torch.device("cpu")
        wrench_env.step_dt = 1.0 / 60.0
        wrench_force_uniform = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1], dtype=np.float32)
        wrench_torque_uniform = np.array([0.1, 0.08, 0.06, 0.04, 0.02, 0.0], dtype=np.float32)
        wrench_force_normal = np.linspace(-1, 1, n * 3, dtype=np.float32).reshape(n, 1, 3)
        wrench_torque_normal = np.linspace(1, -1, n * 3, dtype=np.float32).reshape(n, 1, 3)
        wrench_env._object_forces = torch.zeros((n, 1, 3))
        wrench_env._object_torques = torch.zeros((n, 1, 3))
        wrench_env._random_force_prob = torch.as_tensor(reset_force_prob)
        wrench_env._random_torque_prob = torch.as_tensor(reset_torque_prob)
        wrench_env._object_mass = torch.linspace(1.0, 2.0, n).reshape(n, 1)
        wrench_env._lifted_object = torch.tensor([True, False, True, False, True, True])
        wrench_capture: dict[str, torch.Tensor] = {}
        wrench_env.object = types.SimpleNamespace(
            set_external_force_and_torque=lambda forces, torques, is_global: wrench_capture.update(
                force=forces.clone(), torque=torques.clone()
            )
        )
        wrench_env.cfg = types.SimpleNamespace(
            domain_randomization=types.SimpleNamespace(
                force_decay=0.0,
                torque_decay=0.0,
                force_decay_interval=0.08,
                torque_decay_interval=0.08,
                force_scale=20.0,
                torque_scale=2.0,
                force_only_when_lifted=True,
                torque_only_when_lifted=True,
            )
        )
        original_action_rand = action_mod.torch.rand
        original_action_randn = action_mod.torch.randn
        bernoulli_draws = [wrench_force_uniform, wrench_torque_uniform]
        wrench_normal_draws = [wrench_force_normal, wrench_torque_normal]
        action_mod.torch.rand = lambda *shape, **kwargs: torch.as_tensor(
            bernoulli_draws.pop(0), device=kwargs.get("device")
        )
        action_mod.torch.randn = lambda *shape, **kwargs: torch.as_tensor(
            wrench_normal_draws.pop(0), device=kwargs.get("device")
        )
        action_mod.apply_wrench_dr(wrench_env)
        action_mod.torch.rand = original_action_rand
        action_mod.torch.randn = original_action_randn

        arrays = {
            "perm_canon_to_backend": _PERM_CANON_TO_BACKEND,
            "perm_backend_to_canon": _PERM_BACKEND_TO_CANON,
            "actions_canonical": actions,
            "action_queue_initial": action_queue_initial,
            "action_delay_indices": action_delay_idx,
            "action_target_backend": action_target,
            "goal_pos": pos.numpy().astype(np.float32),
            "goal_quat_wxyz": quat.numpy().astype(np.float32),
            "source_obs_policy": source_obs["policy"].numpy().astype(np.float32),
            "source_critic": source_obs["critic"].numpy().astype(np.float32),
            "source_reward": source_reward.numpy().astype(np.float32),
            "source_reward_terms": np.stack(
                [
                    source_env._reward_terms[name].numpy().astype(np.float32)
                    for name in (
                        "fingertip_delta_rew",
                        "lifting_rew",
                        "lift_bonus_rew",
                        "keypoint_rew",
                        "kuka_actions_penalty",
                        "hand_actions_penalty",
                        "bonus_rew",
                        "total_reward",
                    )
                ],
                axis=1,
            ),
            "source_terminated": source_term.numpy().astype(bool),
            "source_truncated": source_trunc.numpy().astype(bool),
            "source_goal_mask": source_env._is_success.numpy().astype(bool),
            "source_reset_mask": (source_term | source_trunc).numpy().astype(bool),
            "object_position_input": object_pos_input,
            "object_quaternion_wxyz_input": source_env.object.data.root_quat_w.numpy().astype(
                np.float32
            ),
            "object_linear_velocity_input": source_env.object.data.root_lin_vel_w.numpy().astype(
                np.float32
            ),
            "object_angular_velocity_input": source_env.object.data.root_ang_vel_w.numpy().astype(
                np.float32
            ),
            "joint_position_input": source_env.robot.data.joint_pos.numpy().astype(np.float32),
            "joint_velocity_input": source_env.robot.data.joint_vel.numpy().astype(np.float32),
            "previous_targets_input": source_env._prev_targets.numpy().astype(np.float32),
            "object_scale_phi_input": source_env._object_scale_per_env.numpy().astype(np.float32),
            "keypoint_offsets_input": source_env._keypoint_offsets.numpy().astype(np.float32),
            "reward_object_init_z_input": source_env._object_init_z.numpy().astype(np.float32),
            "source_fingertip_distance": source_env._curr_fingertip_distances.numpy().astype(
                np.float32
            ),
            "source_keypoint_max_distance": source_env._keypoints_max_dist.numpy().astype(
                np.float32
            ),
            "source_near_goal_tracker": source_env._near_goal.numpy().astype(bool),
            "source_near_goal_steps_tracker": source_env._near_goal_steps.numpy().astype(np.int64),
            "source_is_success_tracker": source_env._is_success.numpy().astype(bool),
            "source_lifted_tracker": source_env._lifted_object.numpy().astype(bool),
            "source_keypoint_dstar": source_env._closest_keypoint_max_dist.numpy().astype(
                np.float32
            ),
            "source_fingertip_dstar": source_env._closest_fingertip_dist.numpy().astype(np.float32),
            "source_success_tracker": source_env._successes.numpy().astype(np.int64),
            "reset_joint_uniform_draws": reset_joint_uniform,
            "reset_joint_velocity_draws": reset_joint_velocity,
            "reset_object_uniform_draws": reset_object_uniform,
            "reset_orientation_uniform_draws": reset_orientation_uniform,
            "reset_orientation_wxyz_draws": reset_orientation,
            "source_reset_qpos": reset_qpos,
            "source_reset_qvel": reset_qvel,
            "source_reset_goal_pose": captured["goal_pose"].numpy().astype(np.float32),
            "source_reset_force_prob": reset_env._random_force_prob.numpy().astype(np.float32),
            "source_reset_torque_prob": reset_env._random_torque_prob.numpy().astype(np.float32),
            "source_reset_scale_multiplier": reset_env._object_scale_multiplier.numpy().astype(
                np.float32
            ),
            "wrench_force_uniform_draws": wrench_force_uniform,
            "wrench_torque_uniform_draws": wrench_torque_uniform,
            "wrench_force_normal_draws": wrench_force_normal,
            "wrench_torque_normal_draws": wrench_torque_normal,
            "wrench_object_mass": wrench_env._object_mass.numpy().astype(np.float32),
            "wrench_lifted_previous": wrench_env._lifted_object.numpy().astype(bool),
            "source_wrench_force": wrench_capture["force"].numpy().astype(np.float32),
            "source_wrench_torque": wrench_capture["torque"].numpy().astype(np.float32),
            "explicit_goal_position_uniform": goal_pos_draw,
            "explicit_orientation_u1": goal_u1,
            "explicit_orientation_u2": goal_u2,
            "object_state_queue_initial": object_state_queue_initial,
            "obs_queue_initial": obs_queue_initial,
            "object_state_delay_indices": object_delay_idx,
            "obs_delay_indices": obs_delay_idx,
            "object_position_normal_draws": object_pos_normal,
            "object_quat_axis_normal_draws": object_quat_axis_normal,
            "object_quat_angle_degree_draws": object_quat_angle_deg,
            "joint_velocity_normal_draws": joint_velocity_normal,
            **tool_arrays,
        }
        return (
            arrays,
            inventory,
            {
                "case_names": [f"case_{i}" for i in range(n)],
                "stub_symbols": [
                    "convert_quat",
                    "quat_apply",
                    "quat_from_angle_axis",
                    "quat_mul",
                    "random_orientation",
                ],
            },
        )
