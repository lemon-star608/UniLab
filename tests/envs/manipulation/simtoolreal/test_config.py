from __future__ import annotations

from unilab.base import registry
from unilab.envs.manipulation.simtoolreal.config import SimToolRealCfg


def test_config_defaults_preserve_source_contract() -> None:
    cfg = SimToolRealCfg()
    cfg.validate()
    assert cfg.num_actor_obs == 140
    assert cfg.num_critic_obs == 162
    assert cfg.action_space == 29
    assert cfg.sim_dt == 1.0 / 120.0
    assert cfg.ctrl_dt == 1.0 / 60.0
    assert cfg.termination.episode_length == 600
    assert cfg.reset.object_spawn_z_reference_range == 0.0
    assert cfg.domain_randomization.force_only_when_lifted
    assert cfg.domain_randomization.torque_only_when_lifted
    assert (
        cfg.reward_config.keypoint_rew_scale,
        cfg.reward_config.lifting_rew_scale,
        cfg.reward_config.lifting_bonus,
        cfg.reward_config.distance_delta_rew_scale,
        cfg.reward_config.reach_goal_bonus,
        cfg.reward_config.kuka_actions_penalty_scale,
        cfg.reward_config.hand_actions_penalty_scale,
    ) == (
        200.0,
        20.0,
        300.0,
        50.0,
        1000.0,
        0.03,
        0.003,
    )
    assert registry.contains("SimToolReal")


def test_config_rejects_asset_owned_override() -> None:
    cfg = SimToolRealCfg()
    cfg.assets.robot_friction = 0.25
    try:
        cfg.validate()
    except ValueError as exc:
        assert "asset-owned" in str(exc)
    else:
        raise AssertionError("asset-owned friction override must fail closed")


def test_config_rejects_invalid_shapes_ranges_and_fixed_table_mapping() -> None:
    cases = []
    cfg = SimToolRealCfg()
    cfg.action_space = 28
    cases.append(cfg)
    cfg = SimToolRealCfg()
    cfg.domain_randomization.action_delay_max = -1
    cases.append(cfg)
    cfg = SimToolRealCfg()
    cfg.domain_randomization.force_prob_range = (0.2, 0.1)
    cases.append(cfg)
    cfg = SimToolRealCfg()
    cfg.reset.object_spawn_z_reference_range = 0.01
    cases.append(cfg)
    for invalid in cases:
        try:
            invalid.validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config must fail closed: {invalid}")
