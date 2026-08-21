from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tests.envs.manipulation.simtoolreal.source_t0_harness import (
    _quat_apply,
    _quat_from_angle_axis,
    _quat_mul,
)

from unilab.envs.manipulation.simtoolreal.action_pipeline import apply_action_pipeline
from unilab.envs.manipulation.simtoolreal.config import SimToolRealCfg
from unilab.envs.manipulation.simtoolreal.dr_provider import SimToolRealDRProvider
from unilab.envs.manipulation.simtoolreal.dr_wrench import apply_wrench_dr
from unilab.envs.manipulation.simtoolreal.episode_lifecycle import compute_terminations
from unilab.envs.manipulation.simtoolreal.goal_sampling import sample_absolute_goal
from unilab.envs.manipulation.simtoolreal.observations import build_observations
from unilab.envs.manipulation.simtoolreal.rewards import compute_rewards
from unilab.envs.manipulation.simtoolreal.tool_catalog import ALL_TYPES, build_tool_catalog

ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIR = ROOT / "tests/fixtures/simtoolreal_task"
NPZ_SHA256 = "5393583ed7a424910b24622867785e6d6431e29570da997a8064f654ec70624d"
MANIFEST_SHA256 = "d90453bec0db06046aa832615f52c9b8499bee735dc62b4ed9d0d7f107e387b6"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_env(arrays: dict[str, np.ndarray]) -> SimpleNamespace:
    n = 6
    info = {
        "steps": np.array([0, 0, 1, 1, 2, 2], dtype=np.uint32),
        "successes": np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
        "prev_targets": arrays["action_target_backend"].copy(),
        "cur_targets": np.zeros((n, 29), dtype=np.float32),
        "last_actions": np.zeros((n, 29), dtype=np.float32),
        "current_actions": np.zeros((n, 29), dtype=np.float32),
        "goal_pos": arrays["goal_pos"].copy(),
        "goal_quat": arrays["goal_quat_wxyz"].copy(),
        "object_scales": np.ones((n, 3), dtype=np.float32),
        "closest_keypoint_max_dist": np.full(n, -1.0, dtype=np.float32),
        "closest_fingertip_dist": np.full((n, 5), -1.0, dtype=np.float32),
        "lifted_object": np.zeros(n, dtype=bool),
        "reward": np.linspace(0, 5, n, dtype=np.float32),
        "object_init_z": np.zeros(n, dtype=np.float32),
    }
    cfg = SimpleNamespace(
        obs=SimpleNamespace(
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
        reward_config=SimpleNamespace(
            object_base_size=0.04,
            fixed_size=(0.141, 0.03025, 0.0271),
            keypoint_rew_scale=200.0,
            lifting_rew_scale=20.0,
            lifting_bonus=300.0,
            lifting_bonus_threshold=0.15,
            distance_delta_rew_scale=50.0,
            kuka_actions_penalty_scale=0.03,
            hand_actions_penalty_scale=0.003,
            reach_goal_bonus=1000.0,
        ),
        goal=SimpleNamespace(keypoint_scale=1.5, success_steps=10),
        domain_randomization=SimpleNamespace(
            use_object_state_delay_noise=True,
            object_state_xyz_noise_std=0.01,
            object_state_rotation_noise_degrees=5.0,
            use_obs_delay=True,
            joint_velocity_obs_noise_std=0.1,
            use_action_delay=True,
            action_delay_max=3,
        ),
        action=SimpleNamespace(
            clip_actions=1.0, dof_speed_scale=1.5, arm_moving_average=0.1, hand_moving_average=0.1
        ),
        ctrl_dt=1.0 / 60.0,
        termination=SimpleNamespace(force_consecutive_near_goal_steps=False),
    )

    def body_pos(ids: np.ndarray) -> np.ndarray:
        ids = np.atleast_1d(ids)
        out = np.zeros((n, len(ids), 3), dtype=np.float32)
        for column, body_id in enumerate(ids):
            if int(body_id) == 6:
                out[:, column] = arrays["object_position_input"]
        return out

    def body_quat(ids: np.ndarray) -> np.ndarray:
        out = np.zeros((n, len(np.atleast_1d(ids)), 4), dtype=np.float32)
        out[..., 0] = 1.0
        return out

    backend = SimpleNamespace(
        get_body_pos_w=body_pos,
        get_body_quat_w=body_quat,
        get_body_lin_vel_w=lambda ids: np.zeros((n, len(np.atleast_1d(ids)), 3), dtype=np.float32),
        get_body_ang_vel_w=lambda ids: np.zeros((n, len(np.atleast_1d(ids)), 3), dtype=np.float32),
    )
    env = SimpleNamespace(
        _num_envs=n,
        _np_dtype=np.dtype(np.float32),
        _state=SimpleNamespace(info=info),
        cfg=cfg,
        _joint_lower_canon=np.full(29, -1.0, dtype=np.float32),
        _joint_upper_canon=np.full(29, 1.0, dtype=np.float32),
        _perm_backend_to_canon=arrays["perm_backend_to_canon"].astype(np.intp),
        _perm_canon_to_backend=arrays["perm_canon_to_backend"].astype(np.intp),
        _arm_lower=np.full(7, -1.0, dtype=np.float32),
        _arm_upper=np.full(7, 1.0, dtype=np.float32),
        _hand_lower=np.full(22, -1.0, dtype=np.float32),
        _hand_upper=np.full(22, 1.0, dtype=np.float32),
        _palm_body_id=0,
        _fingertip_body_ids=np.arange(1, 6),
        _object_body_id=6,
        _palm_offset=np.array([0.0, -0.02, 0.16], dtype=np.float32),
        _fingertip_offset=np.array([0.02, 0.002, 0.0], dtype=np.float32),
        _arm_slice=slice(0, 7),
        _hand_slice=slice(7, 29),
        _backend=backend,
        _action_queue=arrays["action_queue_initial"].copy(),
        _obs_queue=arrays["obs_queue_initial"].copy(),
        _object_state_queue=arrays["object_state_queue_initial"].copy(),
        _object_scale_multiplier=np.ones((n, 3), dtype=np.float32),
        get_joint_pos_canon=lambda: np.zeros((n, 29), dtype=np.float32),
        get_joint_vel_canon=lambda: np.zeros((n, 29), dtype=np.float32),
        _action_input=arrays["actions_canonical"],
        _object_pos=np.zeros((n, 3), dtype=np.float32),
        _curr_fingertip_distances=np.ones((n, 5), dtype=np.float32),
        _keypoints_max_dist=np.linspace(0.01, 0.06, n, dtype=np.float32),
        _joint_vel=np.zeros((n, 29), dtype=np.float32),
        _near_goal=np.zeros(n, dtype=bool),
        _is_success=np.zeros(n, dtype=bool),
        _reward_terms={},
    )
    return env


def test_fixture_hashes_and_array_inventory_are_closed() -> None:
    assert _sha256(FIXTURE_DIR / "source_t0_fp32.npz") == NPZ_SHA256
    assert _sha256(FIXTURE_DIR / "source_t0_manifest.json") == MANIFEST_SHA256
    data = np.load(FIXTURE_DIR / "source_t0_fp32.npz", allow_pickle=False)
    manifest = json.loads((FIXTURE_DIR / "source_t0_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_mode"] == "source-only"
    assert manifest["cases"]["n"] == 6
    assert manifest["fixture"]["sha256"] == _sha256(FIXTURE_DIR / "source_t0_fp32.npz")
    assert set(data.files) == {entry["name"] for entry in manifest["arrays"]}
    discrete_fields = {
        entry["name"] for entry in manifest["arrays"] if np.dtype(entry["dtype"]).kind in "biu"
    }
    assert set(manifest["discrete_exact_fields"]) == discrete_fields
    loaded_source_modules = {
        (entry["path"], entry["blob"]) for entry in manifest["loaded_source_modules"]
    }
    assert (
        "isaacsimenvs/tasks/simtoolreal/utils/object_size_distributions.py",
        "015471e5f5ddab3438efe2d203e9ce062466353f",
    ) in loaded_source_modules
    assert (
        "isaacsimenvs/tasks/simtoolreal/utils/generate_objects.py",
        "73e2129fd21186061f8a69e8370d736d75523547",
    ) in loaded_source_modules
    perm = data["perm_canon_to_backend"]
    inverse = data["perm_backend_to_canon"]
    assert not np.array_equal(perm, np.arange(29))
    np.testing.assert_array_equal(perm[inverse], np.arange(29))
    np.testing.assert_array_equal(inverse[perm], np.arange(29))
    for entry in manifest["arrays"]:
        array = data[entry["name"]]
        assert list(array.shape) == entry["shape"]
        assert str(array.dtype) == entry["dtype"]
        assert hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest() == entry["sha256"]
        assert np.isfinite(array).all() if array.dtype.kind == "f" else True


def test_source_math_stub_known_quaternion_vectors() -> None:
    angle = torch.tensor([np.pi], dtype=torch.float32)
    axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    quat = _quat_from_angle_axis(angle, axis)
    rotated = _quat_apply(quat, torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32))
    np.testing.assert_allclose(rotated.numpy(), [[-1.0, 0.0, 0.0]], atol=1e-6)
    identity = _quat_mul(quat, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(identity.numpy(), quat.numpy(), atol=1e-6)


def test_target_tool_catalog_matches_source_t0() -> None:
    data = np.load(FIXTURE_DIR / "source_t0_fp32.npz", allow_pickle=False)
    catalog = build_tool_catalog(ALL_TYPES, num_per_type=1, seed=42, shuffle=False)
    assert len(catalog) == 12

    shape_codes = np.array(
        [0 if spec.authored_shape == "box" else 1 for spec in catalog], dtype=np.int8
    )
    has_head = np.array([spec.topology != "box_only" for spec in catalog], dtype=bool)
    np.testing.assert_array_equal(shape_codes, data["source_tool_authored_shape"])
    np.testing.assert_array_equal(has_head, data["source_tool_has_head"])

    expected_handle_size = data["source_tool_handle_size_full"].copy()
    box = shape_codes == 0
    expected_handle_size[box] *= 0.5
    expected_handle_size[~box] = np.column_stack(
        (
            data["source_tool_handle_size_full"][~box, 1] * 0.5,
            data["source_tool_handle_size_full"][~box, 0] * 0.5,
            np.zeros(np.count_nonzero(~box), dtype=np.float32),
        )
    )
    np.testing.assert_allclose(
        np.asarray([spec.handle_size for spec in catalog]),
        expected_handle_size,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray([spec.head_size for spec in catalog]),
        data["source_tool_head_size_full"] * 0.5,
        rtol=1e-5,
        atol=1e-6,
    )
    for field, values in (
        ("head_pos", [spec.head_pos for spec in catalog]),
        ("mass", [spec.mass for spec in catalog]),
        ("com", [spec.com for spec in catalog]),
        ("diaginertia", [spec.diaginertia for spec in catalog]),
        ("object_scale", [spec.object_scale for spec in catalog]),
    ):
        np.testing.assert_allclose(
            np.asarray(values), data[f"source_tool_{field}"], rtol=1e-5, atol=1e-6
        )


def test_target_replay_matches_source_t0(monkeypatch) -> None:
    data = dict(np.load(FIXTURE_DIR / "source_t0_fp32.npz", allow_pickle=False))
    env = _target_env(data)
    env._state.info["prev_targets"][:] = 0.0
    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: data["action_delay_indices"])
    apply_action_pipeline(env, data["actions_canonical"])
    np.testing.assert_allclose(
        env._state.info["cur_targets"], data["action_target_backend"], rtol=1e-5, atol=1e-6
    )

    replay = _target_env(data)
    delay_draws = [data["object_state_delay_indices"], data["obs_delay_indices"]]
    normal_draws = [
        data["object_position_normal_draws"],
        data["object_quat_axis_normal_draws"],
        data["joint_velocity_normal_draws"],
    ]
    monkeypatch.setattr(np.random, "randint", lambda *args, **kwargs: delay_draws.pop(0))
    monkeypatch.setattr(np.random, "randn", lambda *shape: normal_draws.pop(0))
    monkeypatch.setattr(
        np.random,
        "uniform",
        lambda *args, **kwargs: np.deg2rad(data["object_quat_angle_degree_draws"]).astype(
            np.float32
        ),
    )
    obs = build_observations(replay, replay._state)
    np.testing.assert_allclose(obs["obs"], data["source_obs_policy"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(obs["critic"], data["source_critic"], rtol=1e-5, atol=1e-6)

    replay._state.info["closest_fingertip_dist"][:] = data["source_fingertip_dstar"]
    replay._state.info["closest_keypoint_max_dist"][:] = data["source_keypoint_dstar"]
    replay._state.info["lifted_object"][:] = False
    replay._object_pos = data["object_position_input"]
    replay._curr_fingertip_distances = data["source_fingertip_distance"]
    replay._keypoints_max_dist = data["source_keypoint_max_distance"]
    replay._near_goal = data["source_near_goal_tracker"]
    replay._is_success = data["source_is_success_tracker"]
    rewards = compute_rewards(replay, replay._state.info)
    np.testing.assert_allclose(rewards, data["source_reward"], rtol=1e-5, atol=1e-6)
    target_terms = np.stack(
        [
            replay._reward_terms[name]
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
    )
    np.testing.assert_allclose(target_terms, data["source_reward_terms"], rtol=1e-5, atol=1e-6)
    replay._cfg = SimpleNamespace(termination=SimpleNamespace(max_consecutive_successes=50))
    replay._autoreset_envs = np.zeros(6, dtype=bool)
    replay.get_object_pos = lambda: data["object_position_input"]
    terminated, truncated = compute_terminations(replay, data["source_goal_mask"])
    np.testing.assert_array_equal(terminated, data["source_terminated"])
    np.testing.assert_array_equal(truncated, data["source_truncated"])
    np.testing.assert_array_equal(terminated | truncated, data["source_reset_mask"])

    cfg = SimToolRealCfg()
    reset_env = SimpleNamespace(
        _num_envs=6,
        nq=36,
        nv=35,
        cfg=cfg,
        _default_joint_pos_canon=np.zeros(29, dtype=np.float32),
        _joint_lower_canon=np.full(29, -1.0, dtype=np.float32),
        _joint_upper_canon=np.full(29, 1.0, dtype=np.float32),
        _dof_pos_idx_canon=data["perm_backend_to_canon"].astype(np.intp),
        _dof_vel_idx_canon=data["perm_backend_to_canon"].astype(np.intp),
        _obj_pos_slice=slice(29, 32),
        _obj_quat_slice=slice(32, 36),
        _perm_canon_to_backend=data["perm_canon_to_backend"].astype(np.intp),
        _object_forces=np.ones((6, 3), dtype=np.float32),
        _object_torques=np.ones((6, 3), dtype=np.float32),
        _action_queue=np.ones((6, 3, 29), dtype=np.float32),
        _obs_queue=np.ones((6, 3, 140), dtype=np.float32),
        _object_state_queue=np.ones((6, 3, 13), dtype=np.float32),
        _random_force_prob=np.zeros(6, dtype=np.float32),
        _random_torque_prob=np.zeros(6, dtype=np.float32),
        _object_scale_multiplier=np.ones((6, 3), dtype=np.float32),
        _state_cache_lifted_object=np.ones(6, dtype=bool),
        resolve_object_scale=lambda: np.ones((6, 3), dtype=np.float32),
        _state=SimpleNamespace(info={"successes": np.arange(6, dtype=np.int32)}),
    )
    cfg.reset.object_spawn_z_reference_range = 0.0
    cfg.domain_randomization.object_scale_noise_multiplier_range = (0.9, 1.1)
    goal_orientation_uniform = np.column_stack(
        (
            data["explicit_orientation_u1"][:, 0],
            data["explicit_orientation_u2"][:, 1],
            data["explicit_orientation_u2"][:, 2],
        )
    ).astype(np.float32)
    rand_draws = [
        data["reset_joint_uniform_draws"],
        data["reset_orientation_uniform_draws"],
        data["explicit_goal_position_uniform"],
        goal_orientation_uniform,
    ]
    uniform_draws = [
        data["reset_joint_velocity_draws"],
        np.zeros(6, dtype=np.float32),
        data["reset_object_uniform_draws"],
        np.log(data["source_reset_force_prob"]),
        np.log(data["source_reset_torque_prob"]),
        data["source_reset_scale_multiplier"],
    ]
    monkeypatch.setattr(np.random, "rand", lambda *shape: rand_draws.pop(0))
    monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: uniform_draws.pop(0))
    plan = SimToolRealDRProvider().build_reset_plan(reset_env, np.arange(6, dtype=np.intp))
    np.testing.assert_allclose(plan.qpos, data["source_reset_qpos"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(plan.qvel, data["source_reset_qvel"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        plan.info_updates["goal_pos"], data["source_reset_goal_pose"][:, :3], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        plan.info_updates["goal_quat"],
        data["source_reset_goal_pose"][:, 3:],
        rtol=1e-5,
        atol=1e-6,
    )

    backend_calls = []
    wrench_env = SimpleNamespace(
        num_envs=6,
        _object_forces=np.zeros((6, 3), dtype=np.float32),
        _object_torques=np.zeros((6, 3), dtype=np.float32),
        _random_force_prob=data["source_reset_force_prob"],
        _random_torque_prob=data["source_reset_torque_prob"],
        _object_mass=data["wrench_object_mass"].reshape(6),
        _state_cache_lifted_object=data["wrench_lifted_previous"],
        _object_body_id=7,
        cfg=SimpleNamespace(
            ctrl_dt=1.0 / 60.0,
            domain_randomization=SimpleNamespace(
                force_decay=0.0,
                torque_decay=0.0,
                force_decay_interval=0.08,
                torque_decay_interval=0.08,
                force_scale=20.0,
                torque_scale=2.0,
                force_only_when_lifted=True,
                torque_only_when_lifted=True,
            ),
        ),
        backend=SimpleNamespace(
            apply_body_wrench=lambda ids, force, torque: backend_calls.append(
                (force.copy(), torque.copy())
            )
        ),
    )
    bernoulli_draws = [data["wrench_force_uniform_draws"], data["wrench_torque_uniform_draws"]]
    wrench_normal_draws = [
        data["wrench_force_normal_draws"].reshape(6, 3),
        data["wrench_torque_normal_draws"].reshape(6, 3),
    ]
    monkeypatch.setattr(np.random, "random", lambda n: bernoulli_draws.pop(0))
    monkeypatch.setattr(np.random, "randn", lambda *shape: wrench_normal_draws.pop(0))
    apply_wrench_dr(wrench_env)
    np.testing.assert_allclose(
        backend_calls[0][0], data["source_wrench_force"], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        backend_calls[0][1], data["source_wrench_torque"], rtol=1e-5, atol=1e-6
    )
