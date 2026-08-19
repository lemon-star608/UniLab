from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from collections import namedtuple
from pathlib import Path
from types import MethodType
from typing import Any
from unittest.mock import Mock

import numpy as np
import torch
from tests.algos.rlgames_sapg.source_network_harness import (
    _array,
    _canonical_manifest_payload,
    _require_regular_file,
    array_metadata,
    configure_canonical_execution,
    execution_platform,
    rng_state,
)

SOURCE_HEAD = "2a9917533bfea70419ed2667a511d7238e5b3abc"
SOURCE_RL_GAMES_TREE = "7a6a0bb090998d00565aaefa6ab9f2b3d356ace2"
FIXTURE_SCHEMA_VERSION = 1
EXPECTED_NPZ_SHA256 = "3573cc3d0a3700b1c5985b63d7b16175d5ccee3dc8f4b7ef1a7bd7b6676819e8"
EXPECTED_MANIFEST_PAYLOAD_SHA256 = (
    "7d88cb01dce4607391a39d1fb31b21d8366d2bdadae2e0dce6eb02323c06901d"
)
CANONICAL_PLATFORM = json.loads(
    '{"compute_capability":[8,9],"cuda_build":"12.8","cuda_runtime":13020,"cudnn":90701,"driver":"580.173.02","flags":{"cudnn_allow_tf32":false,"cudnn_benchmark":false,"cudnn_deterministic":true,"deterministic_algorithms":true,"float32_matmul_precision":"highest","matmul_allow_tf32":false},"gpu":"NVIDIA GeForce RTX 4090","python":"3.11.15","torch":"2.7.0+cu128"}'
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/simtoolreal_sapg"
FIXTURE_NPZ = FIXTURE_ROOT / "source_rollout_fp32.npz"
FIXTURE_MANIFEST = FIXTURE_ROOT / "source_rollout_manifest.json"
RAW_EXPERIENCE_SCHEMA = json.loads(
    '{"actions":[[4,12,29],"float32"],"dones":[[4,12],"uint8"],"intr_rewards":[[4,12,1],"float32"],"mus":[[4,12,29],"float32"],"neglogpacs":[[4,12],"float32"],"obses":[[4,12,141],"float32"],"rewards":[[4,12,1],"float32"],"sigmas":[[4,12,29],"float32"],"states":[[4,12,163],"float32"],"values":[[4,12,1],"float32"]}'
)


def _clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(item) for item in value)
    return copy.deepcopy(value)


def _put(arrays: dict[str, np.ndarray], name: str, value: Any) -> None:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        arrays[name] = _array(value)
    elif hasattr(value, "_asdict"):
        _put(arrays, name, value._asdict())
    elif isinstance(value, dict):
        for key, item in sorted(value.items()):
            _put(arrays, f"{name}__{key}", item)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _put(arrays, f"{name}__{index}", item)


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _rng_changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return [name for name in ("numpy", "torch_cpu", "torch_cuda") if before[name] != after[name]]


def _rng_component_hashes(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "numpy": _digest(state["numpy"]),
        "torch_cpu": state["torch_cpu"]["sha256"],
        "torch_cuda": [item["sha256"] for item in state["torch_cuda"]],
    }


def _native_raw_transforms(raw: dict[str, torch.Tensor], base: dict[str, Any], fields: list[str]):
    from rl_games.common.custom_utils import swap_and_flatten01

    observed = {name: [list(value.shape), str(_array(value).dtype)] for name, value in raw.items()}
    if observed != RAW_EXPERIENCE_SCHEMA:
        raise RuntimeError(f"raw ExperienceBuffer inventory/metadata drift: {observed}")
    returned_fields = [name for name in fields if name in base]
    if returned_fields != fields:
        raise RuntimeError(f"play_steps returned tensor-list drift: {returned_fields}")
    flattened = {name: swap_and_flatten01(value) for name, value in raw.items()}
    for name in returned_fields:
        if not torch.equal(flattened[name], base[name]):
            raise RuntimeError(f"native raw-to-returned transform drift: {name}")
    return flattened, returned_fields


def _validate_target_array_metadata(
    name: str, actual: np.ndarray, expected: dict[str, Any]
) -> None:
    observed = array_metadata(actual)
    for key, label in (("shape", "shape"), ("dtype", "dtype"), ("sha256", "content hash")):
        if observed[key] != expected[key]:
            raise RuntimeError(f"Target rollout array {name} {label} drift")


def _target_array_diagnostics(actual_arrays, expected_arrays, metadata):
    actual_names, expected_names = set(actual_arrays), set(expected_arrays)
    missing, extra = sorted(expected_names - actual_names), sorted(actual_names - expected_names)
    if missing or extra:
        raise RuntimeError(f"Target array inventory drift: missing={missing}, extra={extra}")
    for name in sorted(expected_names):
        _validate_target_array_metadata(name, actual_arrays[name], metadata[name])
    max_abs_errors, max_rel_errors = {}, {}
    for name in sorted(expected_names):
        actual, expected = actual_arrays[name], expected_arrays[name]
        if np.issubdtype(expected.dtype, np.floating):
            difference = np.abs(actual - expected)
            relative = np.zeros_like(difference)
            np.divide(difference, np.abs(expected), out=relative, where=expected != 0)
            max_abs_errors[name] = float(difference.max(initial=0.0))
            max_rel_errors[name] = float(relative.max(initial=0.0))
    return max_abs_errors, max_rel_errors


def _delegate_rnn_and_record(owner, records, arguments):
    input_value, states, done_masks, bptt_len = arguments
    frozen_input, frozen_states, frozen_dones = _clone((input_value, states, done_masks))
    result = owner(input_value, states, done_masks, bptt_len)
    records.append(RnnCall(frozen_input, frozen_states, frozen_dones, _clone(result), bptt_len))
    return result


def _rnn_state_changed(masked: RnnCall, unmasked: dict[str, Any]) -> bool:
    return any(
        not torch.equal(*states)
        for states in zip(masked.output[1], unmasked["output"][1], strict=True)
    )


def _validate_augmented_state_carriers(arrays, follower_base_indices):
    state_carrier = arrays["buffer_pre_shuffle__states"][:, -1]
    observation_carrier = arrays["buffer_pre_shuffle__obses"][:, -1]
    if state_carrier.shape != (56,) or observation_carrier.shape != (56,):
        raise RuntimeError("augmented carrier shape drift")
    if not np.all(np.isfinite(state_carrier)):
        raise RuntimeError("augmented state carrier must be finite")
    if not np.array_equal(state_carrier, observation_carrier):
        raise RuntimeError("augmented state carrier differs from native observation carrier")
    original = arrays["buffer_base__states"][follower_base_indices, -1]
    if not np.all(state_carrier[-8:] != original):
        raise RuntimeError("follower state carrier was not relabelled")
    return True


def _rms(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        "running_mean": module.running_mean.detach().clone(),
        "running_var": module.running_var.detach().clone(),
        "count": module.count.detach().clone(),
    }


def _rms_snapshot(agent: Any) -> dict[str, dict[str, torch.Tensor]]:
    models = {"actor": agent.model, "central": agent.central_value_net.model}
    return {
        f"{role}_{kind}": _rms(getattr(model, attribute))
        for role, model in models.items()
        for kind, attribute in {"input": "running_mean_std", "value": "value_mean_std"}.items()
    }


class SyntheticVecEnv:
    def __init__(self, spaces: Any, device: torch.device):
        self.device = device
        self.env = self
        self.num_envs = 12
        self.step_index = 0
        self.inputs: dict[str, np.ndarray] = {}
        self.env_info = {
            "agents": 1,
            "value_size": 1,
            "observation_space": spaces.Box(-10.0, 10.0, (140,), dtype=np.float32),
            "state_space": spaces.Box(-10.0, 10.0, (162,), dtype=np.float32),
            "action_space": spaces.Box(-1.0, 1.0, (29,), dtype=np.float32),
        }

    def get_env_info(self) -> dict[str, Any]:
        return self.env_info

    def _matrix(self, time_index: int, width: int, offset: int) -> np.ndarray:
        env = np.arange(self.num_envs, dtype=np.int64)[:, None]
        feature = np.arange(width, dtype=np.int64)[None, :]
        values = (env * 37 + feature * 11 + time_index * 19 + offset) % 257 - 128
        result = values.astype(np.float32) / 128.0
        result[:, 0] = (env[:, 0] * 8 + time_index).astype(np.float32) / 128.0
        return result

    def _observation(self, time_index: int) -> dict[str, torch.Tensor]:
        obs = self._matrix(time_index, 140, 3)
        states = self._matrix(time_index, 162, 71)
        self.inputs[f"obs_t{time_index}"] = obs
        self.inputs[f"states_t{time_index}"] = states
        return {
            "obs": torch.as_tensor(obs, device=self.device),
            "states": torch.as_tensor(states, device=self.device),
        }

    def reset(self) -> dict[str, torch.Tensor]:
        self.step_index = 0
        return self._observation(0)

    def step(self, _actions: torch.Tensor):
        time_index = self.step_index
        env = np.arange(self.num_envs, dtype=np.float32)
        rewards = (time_index * 0.25 + env / 64.0 - 0.125).astype(np.float32)
        dones = np.zeros(self.num_envs, dtype=np.uint8)
        timeouts = np.zeros(self.num_envs, dtype=np.uint8)
        done_env = ((0, 1), (2,), (3,), (4, 5))[time_index]
        timeout_env = ((0,), (2,), (), (4,))[time_index]
        dones[list(done_env)] = 1
        timeouts[list(timeout_env)] = 1
        self.inputs[f"reward_t{time_index}"] = rewards
        self.inputs[f"dones_t{time_index}"] = dones
        self.inputs[f"timeouts_t{time_index}"] = timeouts
        self.step_index += 1
        return (
            self._observation(self.step_index),
            torch.as_tensor(rewards, device=self.device),
            torch.as_tensor(dones, device=self.device),
            {"time_outs": torch.as_tensor(timeouts, device=self.device)},
        )


RolloutCapture = namedtuple("RolloutCapture", "arrays semantics rng_states platform")
RolloutFixture = namedtuple("RolloutFixture", "manifest arrays")
RolloutReplay = namedtuple("RolloutReplay", "is_canonical_platform max_abs_errors max_rel_errors")
RnnCall = namedtuple("RnnCall", "input states dones output bptt_len")
RnnProbe = namedtuple("RnnProbe", "records masked unmasked delegate_calls before after")


def _agent_and_env(runner_params: dict[str, Any], train_dir: Path):
    from rl_games.common import a2c_common
    from rl_games.torch_runner import Runner

    params = copy.deepcopy(runner_params)
    params["config"]["train_dir"] = str(train_dir)
    env = SyntheticVecEnv(a2c_common.gym.spaces, torch.device("cuda:0"))
    runner = Runner()
    configure_canonical_execution()
    runner.load({"params": params})
    after_runner_seed = rng_state()
    runner.set_vec_env(env)
    agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
    agent.init_tensors()
    agent.obs = agent.env_reset()
    return agent, env, after_runner_seed


def _install_agent_spies(agent: Any):
    action_calls: list[dict[str, Any]] = []
    value_calls: list[dict[str, Any]] = []
    actor_shared_values: list[torch.Tensor] = []
    reward_calls: list[dict[str, torch.Tensor]] = []
    phase = {"name": "play"}

    original_action = agent.get_action_values
    original_values = agent.get_values
    original_shaper = agent.rewards_shaper

    def action_spy(_self, obs, rnn_states=None):
        record = {"obs": _clone(obs), "rnn_in": _clone(rnn_states)}
        result = original_action(obs, rnn_states)
        record["result"] = _clone(result)
        action_calls.append(record)
        return result

    def value_spy(_self, obs, rnn_states):
        result = original_values(obs, rnn_states)
        value_calls.append(
            {
                "phase": phase["name"],
                "obs": _clone(obs),
                "rnn_states": _clone(rnn_states),
                "output": _clone(result),
            }
        )
        return result

    def reward_spy(rewards):
        result = original_shaper(rewards)
        reward_calls.append({"input": _clone(rewards), "output": _clone(result)})
        return result

    def actor_hook(_module, _arguments, output):
        actor_shared_values.append(output["values"].detach().clone())

    agent.get_action_values = MethodType(action_spy, agent)
    agent.get_values = MethodType(value_spy, agent)
    agent.rewards_shaper = reward_spy
    handle = agent.model.register_forward_hook(actor_hook)
    return phase, action_calls, value_calls, actor_shared_values, reward_calls, handle


def _native_advantages(agent: Any, extras: dict[str, Any], tail_value: torch.Tensor):
    values = agent.experience_buffer.tensor_dict["values"].detach().clone()
    rewards = agent.experience_buffer.tensor_dict["rewards"].detach().clone()
    args = (
        extras["last_dones"].detach().clone(),
        tail_value.detach().clone(),
        extras["dones"].detach().clone(),
        values,
        rewards,
    )
    advantages = agent.discount_values(*args)
    tau = agent.tau
    try:
        agent.tau = 0.0
        delta = agent.discount_values(*args)
    finally:
        agent.tau = tau
    return delta, advantages


def _choice_spy(agent: Any, base: dict[str, Any], extras: dict[str, Any]):
    records: list[dict[str, Any]] = []
    original = np.random.choice

    def delegate(candidates, size=None, replace=True, p=None):
        result = original(candidates, size=size, replace=replace, p=p)
        records.append(
            {
                "candidates": list(candidates),
                "size": size,
                "replace": replace,
                "result": np.asarray(result).astype(int).tolist(),
                "rng_after": rng_state(),
            }
        )
        return result

    np.random.choice = delegate
    try:
        result = agent.augment_batch_for_mixed_expl(base, extras, repeat_idxs=None)
    finally:
        np.random.choice = original
    return result, records[0]


def _shuffle_spy(batch: dict[str, Any]):
    from rl_games.common.custom_utils import shuffle_batch

    records: list[torch.Tensor] = []
    original = torch.randperm

    def delegate(*args, **kwargs):
        result = original(*args, **kwargs)
        records.append(result.detach().clone())
        records.append(rng_state())
        return result

    torch.randperm = delegate
    try:
        result = shuffle_batch(batch, 4)
    finally:
        torch.randperm = original
    return result, records[0], records[1]


def _dataset_rows(batches: list[dict[str, Any]]) -> tuple[list[int], np.ndarray]:
    sizes = [len(batch["obs"]) for batch in batches]
    rows = np.concatenate(
        [np.rint(_array(batch["obs"])[:, 0] * 128).astype(np.int64) for batch in batches]
    )
    return sizes, rows


def _rms_probe(agent: Any, actor_batches: list[dict[str, Any]], central_batches):
    records: dict[str, Any] = {}
    central = agent.central_value_net.model
    central.train()
    for index, batch in enumerate(central_batches):
        records[f"central_states_{index}"] = batch["obs"].detach().clone()
        central(
            {
                "is_train": True,
                "obs": batch["obs"],
                "actions": batch["actions"],
                "rnn_states": None,
            }
        )
    central.eval()

    native_rnn = agent.model.a2c_network.rnn
    original_forward = native_rnn.forward
    counted_forward = Mock(wraps=original_forward)
    rnn_calls: list[RnnCall] = []

    def rnn_spy(_self, input_value, states, done_masks=None, bptt_len=0):
        return _delegate_rnn_and_record(
            counted_forward,
            rnn_calls,
            (input_value, states, done_masks, bptt_len),
        )

    native_rnn.forward = MethodType(rnn_spy, native_rnn)
    agent.model.train()
    try:
        for index, batch in enumerate(actor_batches):
            records[f"actor_obs_{index}"] = batch["obs"].detach().clone()
            result = agent.model(
                {
                    "is_train": True,
                    "prev_actions": batch["actions"],
                    "obs": batch["obs"],
                    "rnn_states": batch["rnn_states"],
                    "dones": batch["dones"],
                    "seq_length": 4,
                }
            )
            records[f"actor_returned_rnn_{index}"] = result["rnn_states"]
    finally:
        native_rnn.forward = original_forward
        agent.model.eval()
    delegate_calls = counted_forward.call_count
    if delegate_calls != len(rnn_calls) or len(rnn_calls) != len(actor_batches):
        raise RuntimeError("RNN wrapper/owner delegate count drift")
    before = rng_state()
    unmasked_calls = [
        {
            "output": _clone(
                native_rnn.forward(_clone(call.input), _clone(call.states), None, call.bptt_len)
            )
        }
        for call in rnn_calls
    ]
    after = rng_state()
    if _rng_changes(before, after):
        raise RuntimeError("isolated unmasked RNN diagnostic consumed RNG")
    return RnnProbe(records, rnn_calls, unmasked_calls, delegate_calls, before, after)


def capture_rollout(runner_params: dict[str, Any], expected_package_root: Path) -> RolloutCapture:
    if not torch.cuda.is_available():
        raise RuntimeError("canonical SAPG rollout capture requires CUDA")
    configure_canonical_execution()
    arrays: dict[str, np.ndarray] = {}
    rng_states: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="unilab-sapg-rollout-") as train_dir:
        agent, env, rng_states["after_runner_seed"] = _agent_and_env(runner_params, Path(train_dir))
        initial_rms = _rms_snapshot(agent)
        rng_states["after_agent_initialization"] = rng_state()
        spies = _install_agent_spies(agent)
        phase, action_calls, value_calls, actor_shared, reward_calls, actor_handle = spies
        rng_states["before_play"] = rng_state()
        try:
            base, extras = agent.play_steps()
            raw_experience = _clone(agent.experience_buffer.tensor_dict)
            rng_states["after_play"] = rng_state()
            base_snapshot = _clone(base)
            raw_flattened, returned_fields = _native_raw_transforms(
                raw_experience, base_snapshot, agent.tensor_list
            )
            delta, advantages = _native_advantages(agent, extras, value_calls[0]["output"])
            rms_after_play = _rms_snapshot(agent)
            shaped_rewards = torch.stack([call["output"] for call in reward_calls[::2]], dim=0)
            stored_rewards = agent.experience_buffer.tensor_dict["rewards"].detach().clone()
            timeout_bonus = stored_rewards - shaped_rewards

            phase["name"] = "augment"
            rng_states["before_augment"] = rng_state()
            original_counterfactual_states = _clone(
                (extras["states"], extras["last_obs"]["states"])
            )
            augmented, choice = _choice_spy(agent, base, extras)
            rng_states["after_follower_choice"] = choice.pop("rng_after")
            rng_states["after_augment"] = rng_state()
            rms_after_augment = _rms_snapshot(agent)
            pre_shuffle = _clone(augmented)

            rng_states["before_shuffle"] = rng_state()
            shuffled, permutation, after_randperm = _shuffle_spy(_clone(pre_shuffle))
            rng_states["after_trajectory_randperm"] = after_randperm
            rng_states["after_shuffle"] = rng_state()
            rms_after_shuffle = _rms_snapshot(agent)

            agent.prepare_dataset(shuffled, train_value_mean_std=False)
            rng_states["after_prepare"] = rng_state()
            rms_after_prepare = _rms_snapshot(agent)
            actor_batches = [agent.dataset[index] for index in range(len(agent.dataset))]
            central_dataset = agent.central_value_net.dataset
            central_batches = [central_dataset[index] for index in range(len(central_dataset))]
            actor_sizes, actor_rows = _dataset_rows(actor_batches)
            central_sizes, central_rows = _dataset_rows(central_batches)
            rnn_diagnostic = _rms_probe(agent, actor_batches, central_batches)
            rng_states["before_unmasked_rnn_diagnostic"] = rnn_diagnostic.before
            rng_states["after_unmasked_rnn_diagnostic"] = rnn_diagnostic.after
            rng_states["after_input_rms_probe"] = rng_state()
            rms_after_probe = _rms_snapshot(agent)
        finally:
            actor_handle.remove()
            agent.writer.close()

    for name, value in sorted(env.inputs.items()):
        _put(arrays, f"input__{name}", value)
    _put(arrays, "buffer_raw_experience", raw_experience)
    _put(arrays, "diagnostic_raw_swap_and_flatten", raw_flattened)
    _put(arrays, "buffer_base", base_snapshot)
    _put(arrays, "buffer_pre_shuffle", pre_shuffle)
    _put(arrays, "buffer_post_shuffle", shuffled)
    _put(arrays, "rollout_delta", delta)
    _put(arrays, "rollout_advantage", advantages)
    _put(arrays, "rollout_raw_reward", torch.stack([call["input"] for call in reward_calls[::2]]))
    _put(arrays, "rollout_shaped_reward_before_bootstrap", shaped_rewards)
    _put(arrays, "rollout_timeout_bonus", timeout_bonus)
    _put(arrays, "rollout_stored_reward", stored_rewards)
    _put(arrays, "rollout_mb_fdones", extras["dones"])
    _put(arrays, "rollout_fdones", extras["last_dones"])
    _put(arrays, "rollout_tail_value", value_calls[0]["output"])
    _put(arrays, "rollout_tail_obs", extras["last_obs"])
    _put(arrays, "rollout_tail_rnn", extras["last_rnn_states"])
    _put(arrays, "rollout_rnn_state_buffer", extras["rnn_states"])
    post_reset = [
        [
            action_calls[step + 1]["rnn_in"][part] if step < 3 else extras["last_rnn_states"][part]
            for step in range(4)
        ]
        for part in range(2)
    ]
    _put(arrays, "rollout_rnn_post_reset", [torch.stack(part) for part in post_reset])
    _put(arrays, "instrument_action", action_calls)
    _put(arrays, "instrument_actor_shared_value", actor_shared)
    _put(arrays, "instrument_value", value_calls)
    _put(arrays, "instrument_reward_shaper", reward_calls)
    _put(arrays, "shuffle_permutation", permutation)
    _put(arrays, "dataset_actor_rows", actor_rows)
    _put(arrays, "dataset_central_rows", central_rows)
    _put(arrays, "diagnostic_rms_probe", rnn_diagnostic.records)
    _put(arrays, "diagnostic_rnn_masked_probe", rnn_diagnostic.masked)
    _put(arrays, "diagnostic_rnn_isolated_unmasked", rnn_diagnostic.unmasked)
    for phase_name, snapshot in {
        "initial": initial_rms,
        "after_play": rms_after_play,
        "after_augment": rms_after_augment,
        "after_shuffle": rms_after_shuffle,
        "after_prepare": rms_after_prepare,
        "after_probe": rms_after_probe,
    }.items():
        _put(arrays, f"rms_{phase_name}", snapshot)

    repeat_idxs = [0, *choice["result"]]
    base_rows = np.rint(arrays["buffer_base__obses"][:, 0] * 128).astype(int).tolist()
    follower_rows = np.rint(arrays["buffer_pre_shuffle__obses"][-8:, 0] * 128).astype(int).tolist()
    raw_rows = np.rint(_array(raw_experience["obses"][..., 0]) * 128).astype(int).tolist()
    if raw_rows != [list(range(offset, 92, 8)) for offset in range(4)]:
        raise RuntimeError(f"raw ExperienceBuffer row identity drift: {raw_rows}")
    counterfactual_calls = [call for call in value_calls if call["phase"] == "augment"]
    privileged_unchanged = (
        len(counterfactual_calls) == 2
        and torch.equal(
            counterfactual_calls[0]["obs"]["states"],
            original_counterfactual_states[0].reshape(48, 163),
        )
        and torch.equal(counterfactual_calls[1]["obs"]["states"], original_counterfactual_states[1])
    )
    follower_base_indices = [base_rows.index(row) for row in follower_rows]
    augmented_states_relabelled = _validate_augmented_state_carriers(arrays, follower_base_indices)
    extras_intrinsic_none = extras["mb_intr_rewards"] is None
    if not (privileged_unchanged and augmented_states_relabelled and extras_intrinsic_none):
        raise RuntimeError("counterfactual native-call semantics drift")
    rms_paths = json.loads(
        '{"actor_initial":"rms_initial__actor_input__count","actor_after_prepare":"rms_after_prepare__actor_input__count","actor_after_probe":"rms_after_probe__actor_input__count","central_initial":"rms_initial__central_input__count","central_after_prepare":"rms_after_prepare__central_input__count","central_after_probe":"rms_after_probe__central_input__count","actor_value_after_probe":"rms_after_probe__actor_value__count","central_value_after_probe":"rms_after_probe__central_value__count"}'
    )
    rng_pairs = json.loads(
        '{"play":["before_play","after_play"],"native_gae_and_delta":["after_play","before_augment"],"follower_choice":["before_augment","after_follower_choice"],"counterfactual_values":["after_follower_choice","after_augment"],"trajectory_randperm":["before_shuffle","after_trajectory_randperm"],"shuffle_after_randperm":["after_trajectory_randperm","after_shuffle"],"prepare_and_rms_probe":["after_shuffle","after_input_rms_probe"],"isolated_unmasked_rnn_diagnostic":["before_unmasked_rnn_diagnostic","after_unmasked_rnn_diagnostic"]}'
    )
    semantics = {
        "raw_experience_axes": ["time", "env"],
        "raw_obs_row_ids": raw_rows,
        "returned_base_fields": returned_fields,
        "raw_to_returned_base_native_exact": returned_fields,
        "raw_intr_rewards_present": "intr_rewards" in raw_experience,
        "extras_mb_intr_rewards_is_none": extras_intrinsic_none,
        "base_rows": len(base_rows),
        "base_row_ids": base_rows,
        "flattened_row_labels": [f"e{row // 8}t{row % 8}" for row in base_rows],
        "candidate_set": choice["candidates"],
        "choice_size": choice["size"],
        "choice_replace": choice["replace"],
        "repeat_idxs": repeat_idxs,
        "follower_source_block": repeat_idxs[1] - 1,
        "follower_row_ids": follower_rows,
        "permutation": _array(permutation).astype(int).tolist(),
        "augmented_rows": len(arrays["buffer_pre_shuffle__returns"]),
        "played_frames": int(base_snapshot["played_frames"]),
        "actor_dataset_batches": actor_sizes,
        "central_dataset_batches": central_sizes,
        "get_values_phases": [call["phase"] for call in value_calls],
        "counterfactual_current_and_tail_calls": len(counterfactual_calls),
        "counterfactual_privileged_states_relabelled_before_value": not privileged_unchanged,
        "augmented_states_relabelled_for_training": augmented_states_relabelled,
        "intr_reward_model_is_none": agent.intr_reward_model is None,
        "counterfactual_intrinsic_reward_used": not extras_intrinsic_none,
        "rnn_internal_done_batches": [
            bool(torch.any(call.dones[1:] != 0).item()) for call in rnn_diagnostic.masked
        ],
        "rnn_delegate_original_calls": rnn_diagnostic.delegate_calls,
        "rnn_isolated_unmasked_calls": len(rnn_diagnostic.unmasked),
        "rnn_mask_changes_returned_state": [
            _rnn_state_changed(masked, unmasked)
            for masked, unmasked in zip(rnn_diagnostic.masked, rnn_diagnostic.unmasked, strict=True)
        ],
        "input_rms_counts": {name: float(arrays[path].item()) for name, path in rms_paths.items()},
        "rng_hashes": {name: _digest(state) for name, state in rng_states.items()},
        "rng_component_hashes": {
            name: _rng_component_hashes(state) for name, state in rng_states.items()
        },
        "rng_consumption": {
            name: _rng_changes(rng_states[before], rng_states[after])
            for name, (before, after) in rng_pairs.items()
        },
        "step_time_excluded": True,
        "train_value_mean_std": False,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "visible_cuda_device": torch.cuda.get_device_name(0),
    }
    platform_data = execution_platform()
    if platform_data != CANONICAL_PLATFORM:
        raise RuntimeError(f"canonical rollout platform mismatch: {platform_data}")
    expected_package_root = expected_package_root.resolve()
    for name, module in sorted(__import__("sys").modules.items()):
        if name == "rl_games" or name.startswith("rl_games."):
            module_path = Path(module.__file__).resolve()
            try:
                module_path.relative_to(expected_package_root)
            except ValueError as exc:
                raise RuntimeError(f"loaded {name} outside {expected_package_root}") from exc
    return RolloutCapture(arrays, semantics, rng_states, platform_data)


def load_rollout_fixture(root: Path = FIXTURE_ROOT) -> RolloutFixture:
    manifest_path = root / FIXTURE_MANIFEST.name
    npz_path = root / FIXTURE_NPZ.name
    _require_regular_file(manifest_path)
    _require_regular_file(npz_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise RuntimeError("rollout fixture schema drift")
    if manifest.get("provenance", {}).get("source_head") != SOURCE_HEAD:
        raise RuntimeError("rollout fixture Source HEAD drift")
    if manifest.get("provenance", {}).get("source_rl_games_tree") != SOURCE_RL_GAMES_TREE:
        raise RuntimeError("rollout fixture Source tree drift")
    if manifest.get("platform") != CANONICAL_PLATFORM:
        raise RuntimeError("rollout fixture canonical platform drift")
    if hashlib.sha256(_canonical_manifest_payload(manifest)).hexdigest() != (
        EXPECTED_MANIFEST_PAYLOAD_SHA256
    ):
        raise RuntimeError("rollout fixture manifest payload drift")
    if hashlib.sha256(npz_path.read_bytes()).hexdigest() != EXPECTED_NPZ_SHA256:
        raise RuntimeError("rollout fixture NPZ hash drift")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    if set(arrays) != set(manifest["npz_arrays"]):
        raise RuntimeError("rollout fixture array inventory drift")
    for name, value in arrays.items():
        if array_metadata(value) != manifest["npz_arrays"][name]:
            raise RuntimeError(f"rollout fixture array drift: {name}")
    return RolloutFixture(manifest, arrays)


def replay_rollout_fixture(fixture: RolloutFixture) -> RolloutReplay:
    from tests.algos.rlgames_sapg import _runtime_requirement

    capture = capture_rollout(
        fixture.manifest["runner_params"], _runtime_requirement.VENDOR_PACKAGE_ROOT
    )
    _runtime_requirement.require_simtoolreal_rl_games()
    max_abs_errors, max_rel_errors = _target_array_diagnostics(
        capture.arrays, fixture.arrays, fixture.manifest["npz_arrays"]
    )
    if capture.platform != CANONICAL_PLATFORM:
        raise RuntimeError("Target rollout canonical platform drift")
    if capture.rng_states != fixture.manifest["rng_states"]:
        raise RuntimeError("Target rollout RNG state drift")
    if capture.semantics != fixture.manifest["semantics"]:
        raise RuntimeError("Target rollout semantic contract drift")
    return RolloutReplay(True, max_abs_errors, max_rel_errors)
