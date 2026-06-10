"""Unit tests for the CSE-PPO concurrent state estimator and actor-critic."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from unilab.algos.torch.cse_ppo import CSEActorCritic, CSEEstimator  # noqa: E402


def _make_actor_critic(history=8, one_step=76, num_pred=12, latent=19):
    return CSEActorCritic(
        num_actor_obs=history * one_step,
        num_critic_obs=130,
        num_one_step_obs=one_step,
        num_actions=18,
        estimator={
            "num_pred": num_pred,
            "latent_dim": latent,
            "target_weights": [0.2] * 6 + [1.0] * 6,
            "target_start": 0,
        },
    )


def test_actor_critic_shapes():
    ac = _make_actor_critic(history=8, one_step=76, latent=19)
    batch = 32
    hist = torch.randn(batch, 8 * 76)
    crit = torch.randn(batch, 130)
    actions = ac.act(hist)
    values = ac.evaluate(crit)
    assert actions.shape == (batch, 18)
    assert values.shape == (batch, 1)
    # Actor input is [current single-step obs, latent].
    assert ac.estimator.num_latent == 19
    inferred = ac.act_inference(hist)
    assert inferred.shape == (batch, 18)


def test_estimator_latent_is_detached():
    est = CSEEstimator(temporal_steps=8, num_one_step_obs=76, num_pred=12, latent_dim=19)
    hist = torch.randn(16, 8 * 76)
    latent = est.get_latent(hist)
    assert latent.shape == (16, 19)
    assert not latent.requires_grad


def test_estimator_regression_loss_decreases():
    """On a fixed linear history->target map the weighted MSE should drop."""
    history, one_step, num_pred = 8, 76, 12
    est = CSEEstimator(
        temporal_steps=history,
        num_one_step_obs=one_step,
        num_pred=num_pred,
        latent_dim=24,
        target_weights=[0.2] * 6 + [1.0] * 6,
        target_start=0,
        learning_rate=1e-3,
    )
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(history * one_step, num_pred, generator=gen) * 0.01

    first_loss = None
    last_loss = None
    for i in range(150):
        h = torch.randn(128, history * one_step, generator=gen)
        target = h @ weight
        critic = torch.zeros(128, 130)
        critic[:, :num_pred] = target
        loss = est.update(h, critic, lr=1e-3)
        if i == 0:
            first_loss = loss
        last_loss = loss
    assert first_loss is not None and last_loss is not None
    assert last_loss < first_loss * 0.5


def test_estimator_target_slice_validation():
    est = CSEEstimator(temporal_steps=4, num_one_step_obs=10, num_pred=12, target_start=0)
    too_small = torch.zeros(4, 8)
    with pytest.raises(ValueError):
        est.update(torch.zeros(4, 4 * 10), too_small)


def test_estimator_default_lr_is_fixed_when_not_overridden():
    """The estimator keeps its own fixed LR (UniFP 1e-5) unless lr is passed."""
    est = CSEEstimator(temporal_steps=4, num_one_step_obs=10, num_pred=12, target_start=0)
    assert est.learning_rate == pytest.approx(1e-5)
    est.update(torch.zeros(8, 4 * 10), torch.randn(8, 12))  # no lr override
    assert est.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_actor_input_width_matches_task_architecture():
    """Task config uses latent_dim=64 (history x 2); actor input = one_step + latent."""
    ac = CSEActorCritic(
        num_actor_obs=32 * 76,
        num_critic_obs=138,
        num_one_step_obs=76,
        num_actions=18,
        estimator={
            "latent_dim": 64,
            "num_pred": 12,
            "enc_hidden_dims": [512, 256, 128],
            "dec_hidden_dims": [128, 64],
            "target_start": 0,
        },
    )
    assert ac.estimator.num_latent == 64
    assert ac.actor[0].in_features == 76 + 64
