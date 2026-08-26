from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "conf/rlgames_sapg"
def _compose(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        return compose("config", overrides=list(overrides))


def _native(cfg: DictConfig) -> dict:
    value = OmegaConf.to_container(cfg.rl_games.params, resolve=True)
    assert isinstance(value, dict)
    return value


def test_base_owner_exposes_native_sapg_resource_contract():
    params = _native(_compose())
    config = params["config"]
    central = config["central_value_config"]
    assert config["num_actors"] == 24576
    assert config["expl_coef_block_size"] == 4096
    assert config["num_actors"] // config["expl_coef_block_size"] == 6
    assert config["full_experiment_name"] == "0_simtoolreal_sapg"
    assert config["horizon_length"] == 16
    assert config["seq_length"] == 16
    assert config["minibatch_size"] == 98304
    assert central["minibatch_size"] == 98304
    assert params["network"]["mlp"]["units"] == [1024, 1024, 512, 512]
    assert central["network"]["mlp"]["units"] == [1024, 1024, 512, 512]


def test_12k_profile_changes_only_actor_and_block_size():
    base = _native(_compose())
    profile = _native(_compose("task=simtoolreal/mujoco_12k"))
    assert profile["config"]["num_actors"] == 12288
    assert profile["config"]["expl_coef_block_size"] == 2048
    profile["config"]["num_actors"] = base["config"]["num_actors"]
    profile["config"]["expl_coef_block_size"] = base["config"]["expl_coef_block_size"]
    assert profile == base


def test_owner_exposes_raw_reward_and_single_native_scale_boundary():
    cfg = _compose()
    assert OmegaConf.to_container(cfg.reward, resolve=True) == {
        "keypoint_rew_scale": 200.0,
        "object_base_size": 0.04,
        "fixed_size": [0.141, 0.03025, 0.0271],
        "fixed_size_keypoint_reward": True,
        "lifting_rew_scale": 20.0,
        "lifting_bonus": 300.0,
        "lifting_bonus_threshold": 0.15,
        "distance_delta_rew_scale": 50.0,
        "reach_goal_bonus": 1000.0,
        "kuka_actions_penalty_scale": 0.03,
        "hand_actions_penalty_scale": 0.003,
    }
    assert cfg.rl_games.params.config.reward_shaper.scale_value == 0.01
    assert cfg.algo.obs_groups == {"actor": ["obs"], "critic": ["critic"]}
    assert list(cfg.algo.policy.actor_hidden_dims) == [1024, 1024, 512, 512]


def test_preflight_accepts_base_and_12k_without_importing_env_or_rl_games():
    from unilab.algos.torch.rlgames_sapg.config import preflight_config

    preflight_config(_compose())
    preflight_config(_compose("task=simtoolreal/mujoco_12k"))


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("training.sim_backend=motrix", "MuJoCo"),
        ("training.task_name=Wrong", "SimToolReal"),
        ("algo.num_envs=5", "six exploration blocks"),
        ("rl_games.params.config.expl_coef_block_size=2", "six exploration blocks"),
        ("rl_games.params.config.multi_gpu=true", "multi_gpu"),
        ("rl_games.params.load_checkpoint=true", "load_checkpoint"),
        ("rl_games.params.load_path=unsafe.pth", "load_path"),
        ("env.action_space=28", "29"),
        ("rl_games.params.config.horizon_length=3", "seq_length"),
    ],
)
def test_preflight_rejects_unsupported_config_before_env_construction(override, match):
    from unilab.algos.torch.rlgames_sapg.config import preflight_config

    with pytest.raises(ValueError, match=match):
        preflight_config(_compose(override))
