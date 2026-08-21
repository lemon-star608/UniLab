"""Target-only real MuJoCo capture harness for the Code #8 T1 fixture."""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.base import registry

N = 6
H = 8
SEED = 20260821
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


def _actions() -> np.ndarray:
    t = np.arange(H, dtype=np.float64)[:, None, None]
    env = np.arange(N, dtype=np.float64)[None, :, None]
    joint = np.arange(29, dtype=np.float64)[None, None, :]
    return (0.25 * np.sin(0.37 * t + 0.11 * env + 0.07 * joint)).astype(np.float32)


def _copy_info(info: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(info[key])
    return value.copy()


def capture_target_t1() -> dict[str, np.ndarray]:
    """Run the fixed eight-event Target-only capture and return named arrays."""
    np.random.seed(SEED)
    registry.ensure_registries()
    env = registry.make("SimToolReal", sim_backend="mujoco", num_envs=N)
    try:
        actions = _actions()
        state = env.init_state()
        initial_obs = state.obs["obs"].copy()
        initial_critic = state.obs["critic"].copy()
        initial_goal = _copy_info(state.info, "goal_pos")
        initial_goal_quat = _copy_info(state.info, "goal_quat")
        initial_object_scales = _copy_info(state.info, "object_scales")
        initial_prev_targets = _copy_info(state.info, "prev_targets")
        initial_object_pose = np.concatenate((env.get_object_pos(), env.get_object_quat()), axis=-1)

        tool_indices = np.asarray(env._tool_index, dtype=np.int32).copy()
        object_scales = np.asarray(env.resolve_object_scale()).copy()
        signatures = np.asarray(
            [
                [
                    env.get_playback_model(index).nq,
                    env.get_playback_model(index).nv,
                    env.get_playback_model(index).nu,
                    env.get_playback_model(index).nmesh,
                    env.get_playback_model(index).ngeom,
                ]
                for index in range(N)
            ],
            dtype=np.int32,
        )

        obs_steps: list[np.ndarray] = []
        critic_steps: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        terminated: list[np.ndarray] = []
        truncated: list[np.ndarray] = []
        steps: list[np.ndarray] = []
        successes: list[np.ndarray] = []
        near_goal: list[np.ndarray] = []
        lifted: list[np.ndarray] = []
        d_star: list[np.ndarray] = []
        fingertip_d_star: list[np.ndarray] = []
        goal_pose: list[np.ndarray] = []
        object_pose: list[np.ndarray] = []
        prev_targets: list[np.ndarray] = []
        cur_targets: list[np.ndarray] = []
        reward_terms = {name: [] for name in REWARD_TERM_ORDER}
        backend_autoreset: list[np.ndarray] = []

        partial_ids = np.asarray([1, 4], dtype=np.int32)
        partial_obs: np.ndarray | None = None
        partial_critic: np.ndarray | None = None
        partial_terminal_obs: np.ndarray | None = None
        partial_terminal_critic: np.ndarray | None = None
        partial_mask: np.ndarray | None = None
        partial_goal: np.ndarray | None = None
        partial_object: np.ndarray | None = None
        partial_targets: np.ndarray | None = None

        timeout_row = np.asarray([2], dtype=np.int32)
        timeout_obs: np.ndarray | None = None
        timeout_critic: np.ndarray | None = None
        timeout_terminal_obs: np.ndarray | None = None
        timeout_terminal_critic: np.ndarray | None = None
        timeout_mask: np.ndarray | None = None

        for index, action in enumerate(actions):
            if index == 4:
                state.terminated[:] = False
                state.truncated[:] = False
                state.terminated[partial_ids] = True
                env._reset_done_envs()
                assert state.final_observation is not None
                partial_terminal_obs = state.final_observation["obs"].copy()
                partial_terminal_critic = state.final_observation["critic"].copy()
                partial_mask = state.info["_final_observation"].copy()
                partial_obs = state.obs["obs"].copy()
                partial_critic = state.obs["critic"].copy()
                partial_goal = _copy_info(state.info, "goal_pos")
                partial_object = np.concatenate(
                    (env.get_object_pos(), env.get_object_quat()), axis=-1
                )
                partial_targets = _copy_info(state.info, "cur_targets")
                state.terminated[:] = False
                state.truncated[:] = False

            if index == 7:
                state.info["steps"][2] = 599

            state = env.step(action)
            obs_steps.append(state.obs["obs"].copy())
            critic_steps.append(state.obs["critic"].copy())
            rewards.append(state.reward.copy())
            terminated.append(state.terminated.copy())
            truncated.append(state.truncated.copy())
            steps.append(_copy_info(state.info, "steps"))
            successes.append(_copy_info(state.info, "successes"))
            near_goal.append(np.asarray(env._near_goal, dtype=bool).copy())
            lifted.append(_copy_info(state.info, "lifted_object"))
            d_star.append(_copy_info(state.info, "closest_keypoint_max_dist"))
            fingertip_d_star.append(_copy_info(state.info, "closest_fingertip_dist"))
            goal_pose.append(
                np.concatenate(
                    (_copy_info(state.info, "goal_pos"), _copy_info(state.info, "goal_quat")),
                    axis=-1,
                )
            )
            object_pose.append(
                np.concatenate((env.get_object_pos(), env.get_object_quat()), axis=-1)
            )
            prev_targets.append(_copy_info(state.info, "prev_targets"))
            cur_targets.append(_copy_info(state.info, "cur_targets"))
            for name in reward_terms:
                reward_terms[name].append(np.asarray(env._reward_terms[name]).copy())
            backend_autoreset.append(np.asarray(env._autoreset_envs, dtype=bool).copy())

            if index == 7:
                assert state.final_observation is not None
                timeout_obs = state.obs["obs"].copy()
                timeout_critic = state.obs["critic"].copy()
                timeout_terminal_obs = state.final_observation["obs"].copy()
                timeout_terminal_critic = state.final_observation["critic"].copy()
                timeout_mask = state.info["_final_observation"].copy()

        assert partial_obs is not None and partial_critic is not None
        assert partial_terminal_obs is not None and partial_terminal_critic is not None
        assert partial_mask is not None and partial_goal is not None
        assert partial_object is not None and partial_targets is not None
        assert timeout_obs is not None and timeout_critic is not None
        assert timeout_terminal_obs is not None and timeout_terminal_critic is not None
        assert timeout_mask is not None
        result: dict[str, np.ndarray] = {
            "actions": actions,
            "partial_reset_ids": partial_ids,
            "timeout_row": timeout_row,
            "tool_indices": tool_indices,
            "object_scales": object_scales,
            "model_signatures": signatures,
            "initial_obs": initial_obs,
            "initial_critic": initial_critic,
            "initial_goal": initial_goal,
            "initial_goal_quat": initial_goal_quat,
            "initial_object_scales": initial_object_scales,
            "initial_prev_targets": initial_prev_targets,
            "initial_object_pose": initial_object_pose,
            "partial_reset_obs": partial_obs,
            "partial_reset_critic": partial_critic,
            "partial_terminal_obs": partial_terminal_obs,
            "partial_terminal_critic": partial_terminal_critic,
            "partial_reset_mask": partial_mask,
            "partial_goal": partial_goal,
            "partial_object_pose": partial_object,
            "partial_targets": partial_targets,
            "step_obs": np.stack(obs_steps),
            "step_critic": np.stack(critic_steps),
            "step_reward": np.stack(rewards),
            "step_terminated": np.stack(terminated),
            "step_truncated": np.stack(truncated),
            "step_steps": np.stack(steps),
            "step_successes": np.stack(successes),
            "step_near_goal": np.stack(near_goal),
            "step_lifted": np.stack(lifted),
            "step_d_star": np.stack(d_star),
            "step_fingertip_d_star": np.stack(fingertip_d_star),
            "step_goal_pose": np.stack(goal_pose),
            "step_object_pose": np.stack(object_pose),
            "step_prev_targets": np.stack(prev_targets),
            "step_cur_targets": np.stack(cur_targets),
            "backend_autoreset": np.stack(backend_autoreset),
            "timeout_obs": timeout_obs,
            "timeout_critic": timeout_critic,
            "timeout_terminal_obs": timeout_terminal_obs,
            "timeout_terminal_critic": timeout_terminal_critic,
            "timeout_terminal_mask": timeout_mask,
        }
        result.update({f"reward_{name}": np.stack(values) for name, values in reward_terms.items()})
        return result
    finally:
        env.close()


__all__ = ["capture_target_t1"]
