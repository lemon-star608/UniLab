"""Real MuJoCo integration coverage for the registered SimToolReal env."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from unilab.base import registry
from unilab.envs.manipulation.simtoolreal import SimToolRealEnv
from unilab.envs.manipulation.simtoolreal.config import SimToolRealCfg
from unilab.envs.manipulation.simtoolreal.dr_provider import DSTAR_SENTINEL

NUM_ENVS = 6
REWARD_TERM_ORDER = (
    "fingertip_delta_rew",
    "lifting_rew",
    "lift_bonus_rew",
    "keypoint_rew",
    "kuka_actions_penalty",
    "hand_actions_penalty",
    "bonus_rew",
    "total_reward",
)


@pytest.fixture(scope="module")
def real_env() -> Any:
    """Amortize the required 600-model compile across all real tests."""
    np.random.seed(20260821)
    registry.ensure_registries()
    env = registry.make("SimToolReal", sim_backend="mujoco", num_envs=NUM_ENVS)
    assert isinstance(env, SimToolRealEnv)
    temp_root = Path(env._tool_scenes.cleanup.name)
    env.init_state()
    try:
        yield env
    finally:
        env.close()
        env.close()
        assert not temp_root.exists()


def test_construction_failure_cleans_materialized_tool_scenes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unilab.envs.manipulation.simtoolreal import env as env_module

    materialized_roots: list[Path] = []
    original_materialize = env_module.materialize_tool_scenes

    def record_materialized_root(*args: Any, **kwargs: Any) -> Any:
        scenes = original_materialize(*args, **kwargs)
        materialized_roots.append(Path(scenes.cleanup.name))
        return scenes

    monkeypatch.setattr(env_module, "materialize_tool_scenes", record_materialized_root)
    cfg = SimToolRealCfg()
    cfg.assets.object_pool_enabled = False

    try:
        SimToolRealEnv(cfg, num_envs=1, backend_type="invalid-test-backend")
    except ValueError as exc:
        assert "Unknown backend" in str(exc)
        assert len(materialized_roots) == 1
        assert not materialized_roots[0].exists()
    else:
        raise AssertionError("invalid backend must fail construction")


def _reset_rows(env: Any, env_ids: np.ndarray) -> None:
    """Exercise the base selected-row scatter path and leave masks ready to step."""
    state = env.state
    assert state is not None
    state.terminated.fill(False)
    state.truncated.fill(False)
    state.terminated[env_ids] = True
    env._reset_done_envs()
    state.terminated.fill(False)
    state.truncated.fill(False)


def _reset_all(env: Any) -> None:
    _reset_rows(env, np.arange(NUM_ENVS, dtype=np.int32))


def _scripted_action(step: int) -> np.ndarray:
    env_index = np.arange(NUM_ENVS, dtype=np.float64)[:, None]
    joint_index = np.arange(29, dtype=np.float64)[None, :]
    return (0.05 * np.sin(0.17 * step + 0.11 * env_index + 0.07 * joint_index)).astype(np.float32)


def _diverge_one_env(env: Any, env_id: int) -> None:
    """Use the public FULLPHYSICS state contract to trip one engine autoreset."""
    backend = env._backend
    nq, nv = int(backend.model.nq), int(backend.model.nv)
    physics = np.asarray(backend.get_physics_state(), dtype=np.float64)
    assert physics.shape[1] >= 1 + nq + nv
    qpos = physics[env_id : env_id + 1, 1 : 1 + nq].copy()
    qvel = physics[env_id : env_id + 1, 1 + nq : 1 + nq + nv].copy()
    qvel.fill(1.0e11)
    backend.set_state(np.asarray([env_id], dtype=np.int32), qpos, qvel)


def test_registered_env_compiles_complete_600_tool_pool(real_env: Any) -> None:
    """Near-risk inventory inspection is intentionally test-only."""
    env = real_env
    assert env.action_space.shape == (29,)
    assert env.action_space.dtype == np.float32
    assert env.obs_groups_spec == {"obs": 140, "critic": 162}
    assert len(env._tool_catalog) == 600
    assert len(env._tool_variant_files) == 600
    assert all(Path(model_file).is_file() for model_file in env._tool_variant_files)
    np.testing.assert_array_equal(env._tool_index, np.arange(NUM_ENVS, dtype=np.int32))

    plan = env.build_init_randomization_plan()
    assert len(plan.model_variants) == 600
    assert all(variant.source_model_file for variant in plan.model_variants)
    np.testing.assert_array_equal(plan.model_assignments, np.arange(NUM_ENVS, dtype=np.int32))

    # Private inventory access is confined to this test to prove all cold-path
    # source models reached the real runtime pool.
    models = env._backend._model_variants
    assignments = env._backend._model_assignments
    assert len(models) == 600
    np.testing.assert_array_equal(assignments, np.arange(NUM_ENVS, dtype=np.int32))

    topology = Counter()
    for spec, model in zip(env._tool_catalog, models, strict=True):
        topology[spec.topology] += 1
        # The complete source XML has 40 meshes.  The real physics pool keeps
        # the established backend ``discardvisual=true`` contract, so its
        # compiled variants intentionally contain only the 19 collision meshes.
        assert (model.nq, model.nv, model.nu, model.nmesh) == (36, 35, 29, 19)
        object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        assert object_body_id >= 0
        np.testing.assert_allclose(model.body_mass[object_body_id], spec.mass, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(model.body_ipos[object_body_id], spec.com, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(
            model.body_inertia[object_body_id], spec.diaginertia, rtol=1e-5, atol=1e-7
        )
        object_geoms = np.flatnonzero(np.asarray(model.geom_bodyid) == object_body_id)
        assert object_geoms.size == (1 if spec.topology == "box_only" else 2)
        expected_type = (
            mujoco.mjtGeom.mjGEOM_CAPSULE
            if spec.collision_shape == "capsule"
            else mujoco.mjtGeom.mjGEOM_BOX
        )
        assert int(model.geom_type[object_geoms[0]]) == int(expected_type)
        assert np.isfinite(model.body_inertia[object_body_id]).all()
        assert (model.body_inertia[object_body_id] > 0.0).all()

    assert topology == {"box_box": 250, "capsule_box": 300, "box_only": 50}


def test_first_reset_and_partial_reset_preserve_unselected_rows(real_env: Any) -> None:
    env = real_env
    _reset_all(env)
    state = env.state
    assert state is not None
    for group, width in env.obs_groups_spec.items():
        assert state.obs[group].shape == (NUM_ENVS, width)
        assert state.obs[group].dtype == np.float32
        assert np.isfinite(state.obs[group]).all()
        assert np.any(state.obs[group])

    for step in range(3):
        state = env.step(_scripted_action(step))

    reset_ids = np.asarray([2, 5], dtype=np.int32)
    keep_ids = np.asarray([0, 1, 3, 4], dtype=np.int32)
    obs_before = {key: value.copy() for key, value in state.obs.items()}
    info_before = {
        key: np.asarray(state.info[key]).copy()
        for key in ("goal_pos", "prev_targets", "cur_targets", "object_init_z")
    }
    action_queue_before = env._action_queue.copy()
    obs_queue_before = env._obs_queue.copy()
    object_queue_before = env._object_state_queue.copy()

    _reset_rows(env, reset_ids)

    for key in state.obs:
        np.testing.assert_array_equal(state.obs[key][keep_ids], obs_before[key][keep_ids])
    for key, before in info_before.items():
        np.testing.assert_array_equal(state.info[key][keep_ids], before[keep_ids])
    np.testing.assert_array_equal(env._action_queue[keep_ids], action_queue_before[keep_ids])
    np.testing.assert_array_equal(env._obs_queue[keep_ids], obs_queue_before[keep_ids])
    np.testing.assert_array_equal(env._object_state_queue[keep_ids], object_queue_before[keep_ids])
    assert np.isfinite(state.obs["obs"][reset_ids]).all()
    assert np.isfinite(state.obs["critic"][reset_ids]).all()


def test_64_real_steps_keep_raw_reward_and_state_finite(real_env: Any) -> None:
    env = real_env
    _reset_all(env)
    for step in range(64):
        state = env.step(_scripted_action(step))
        assert np.isfinite(state.reward).all()
        assert state.reward.dtype == np.float32
        assert not state.truncated.any()
        for values in state.obs.values():
            assert values.dtype == np.float32
            assert np.isfinite(values).all()
        assert tuple(env._reward_terms) == REWARD_TERM_ORDER
        component_sum = sum(env._reward_terms[name] for name in REWARD_TERM_ORDER[:-1])
        np.testing.assert_allclose(state.reward, component_sum, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(state.info["reward"], state.reward)
        np.testing.assert_array_equal(env._reward_terms["total_reward"], state.reward)


def test_controlled_success_advances_goal_without_done(real_env: Any) -> None:
    env = real_env
    _reset_all(env)
    state = env.state
    assert state is not None
    original_success_steps = env.cfg.goal.success_steps
    original_tolerance = env._current_success_tolerance
    try:
        env.cfg.goal.success_steps = 1
        env._current_success_tolerance = 0.075
        state.info["near_goal_steps"].fill(0)
        state.info["goal_pos"][:] = np.asarray([9.0, 9.0, 9.0], dtype=np.float32)
        state.info["goal_pos"][0] = env.get_object_pos()[0]
        state.info["goal_quat"][0] = env.get_object_quat()[0]
        goal_before = state.info["goal_pos"][0].copy()

        state = env.step(np.zeros((NUM_ENVS, 29), dtype=np.float32))

        assert state.info["successes"][0] == 1
        assert state.info["steps"][0] == 1
        assert not state.terminated[0]
        assert not state.truncated[0]
        assert not np.array_equal(state.info["goal_pos"][0], goal_before)
        assert state.info["near_goal_steps"][0] == 0
    finally:
        env.cfg.goal.success_steps = original_success_steps
        env._current_success_tolerance = original_tolerance
        _reset_all(env)


def test_exact_timeout_preserves_terminal_obs_then_resets_selected_row(real_env: Any) -> None:
    env = real_env
    _reset_all(env)
    state = env.state
    assert state is not None
    timeout_row = 2
    state.info["steps"].fill(17)
    state.info["steps"][timeout_row] = 599

    state = env.step(np.zeros((NUM_ENVS, 29), dtype=np.float32))

    expected_mask = np.arange(NUM_ENVS) == timeout_row
    np.testing.assert_array_equal(state.truncated, expected_mask)
    assert not state.terminated[timeout_row]
    np.testing.assert_array_equal(state.info["_final_observation"], expected_mask)
    assert state.final_observation is not None
    for group in env.obs_groups_spec:
        assert np.isfinite(state.final_observation[group][timeout_row]).all()
        assert not np.array_equal(
            state.obs[group][timeout_row], state.final_observation[group][timeout_row]
        )
    assert state.info["steps"][timeout_row] == 0
    np.testing.assert_array_equal(
        state.info["steps"][~expected_mask], np.full(NUM_ENVS - 1, 18, dtype=np.uint32)
    )


def test_public_wrench_handoff_is_lift_gated_and_row_isolated(
    real_env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = real_env
    _reset_all(env)
    lifted = np.asarray([False, True, False, False, True, False])
    force_prob = env._random_force_prob.copy()
    torque_prob = env._random_torque_prob.copy()
    try:
        env._random_force_prob.fill(1.0)
        env._random_torque_prob.fill(1.0)
        env._state_cache_lifted_object[:] = lifted
        env._object_forces.fill(0.0)
        env._object_torques.fill(0.0)

        calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        original_apply_body_wrench = env._backend.apply_body_wrench

        def record_public_wrench(
            body_ids: np.ndarray, force: np.ndarray, torque: np.ndarray
        ) -> None:
            calls.append((body_ids.copy(), force.copy(), torque.copy()))
            original_apply_body_wrench(body_ids, force, torque)

        monkeypatch.setattr(env._backend, "apply_body_wrench", record_public_wrench)
        targets = env.apply_action(np.zeros((NUM_ENVS, 29), dtype=np.float32), env.state)

        assert np.any(env._object_forces[lifted] != 0.0)
        assert np.any(env._object_torques[lifted] != 0.0)
        assert np.all(env._object_forces[~lifted] == 0.0)
        assert np.all(env._object_torques[~lifted] == 0.0)
        assert len(calls) == 1
        body_ids, forces, torques = calls[0]
        np.testing.assert_array_equal(body_ids, np.asarray([env._object_body_id], dtype=np.int32))
        assert np.any(forces[lifted, 0] != 0.0)
        assert np.any(torques[lifted, 0] != 0.0)
        assert np.all(forces[~lifted, 0] == 0.0)
        assert np.all(torques[~lifted, 0] == 0.0)
        env._backend.step(targets, env.cfg.sim_substeps)
    finally:
        env._random_force_prob[:] = force_prob
        env._random_torque_prob[:] = torque_prob
        _reset_all(env)


def test_real_engine_autoreset_marks_exact_row_and_clears_caches(
    real_env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = real_env
    _reset_all(env)
    state = env.state
    assert state is not None
    flagged = 1
    state.info["prev_targets"][flagged].fill(9.0)
    state.info["cur_targets"][flagged].fill(9.0)
    state.info["object_init_z"][flagged] = 9.0
    state.info["closest_keypoint_max_dist"][flagged] = 9.0
    state.info["closest_fingertip_dist"][flagged].fill(9.0)
    state.info["lifted_object"][flagged] = True
    env._state_cache_lifted_object[flagged] = True

    # MuJoCo's divergence warning is cwd-relative. Keep the generated log out
    # of the repository while exercising the real backend autoreset path.
    monkeypatch.chdir(tmp_path)
    _diverge_one_env(env, flagged)
    state = env.step(np.zeros((NUM_ENVS, 29), dtype=np.float32))

    expected = np.arange(NUM_ENVS) == flagged
    backend_mask = env._backend.get_step_autoreset_mask()
    assert backend_mask is not None
    np.testing.assert_array_equal(backend_mask, expected)
    np.testing.assert_array_equal(env._autoreset_envs, expected)
    assert state.terminated[flagged]
    assert state.info["object_init_z"][flagged] != 9.0
    assert state.info["closest_keypoint_max_dist"][flagged] == DSTAR_SENTINEL
    assert np.all(state.info["closest_fingertip_dist"][flagged] == DSTAR_SENTINEL)
    assert not state.info["lifted_object"][flagged]
    assert not env._state_cache_lifted_object[flagged]

    state = env.step(np.zeros((NUM_ENVS, 29), dtype=np.float32))
    assert not env._autoreset_envs.any()
    assert not env._state_cache_lifted_object[flagged]
    assert np.isfinite(state.obs["obs"]).all()
