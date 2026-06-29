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


def test_actor_path_backprops_into_estimator_encoder():
    """The actor receives a NON-detached latent, so the PPO/actor loss co-trains
    the estimator encoder. Detaching it would sever the encoder<->RL coupling that
    is the core of the concurrent state estimator.
    """
    ac = _make_actor_critic(history=8, one_step=76, latent=19)
    hist = torch.randn(4, 8 * 76)
    ac.update_distribution(hist)
    loss = ac.action_mean.pow(2).mean()
    loss.backward()
    enc_weight = ac.estimator.encoder[0].weight
    assert enc_weight.grad is not None
    assert torch.count_nonzero(enc_weight.grad) > 0


def test_estimator_loss_is_unifp_per_group_sum():
    """Estimator loss = sum over the 4 target groups of
    ``mse(pred_g * w_g, target_g * w_g)``, NOT a single weighted mean over all 12
    dims (a per-group sum changes the gradient scale by ~4x relative to a mean).
    """
    import torch.nn.functional as F

    est = CSEEstimator(
        temporal_steps=4,
        num_one_step_obs=10,
        num_pred=12,
        target_group_sizes=[3, 3, 3, 3],
        target_weights=[0.2, 0.2, 1.0, 1.0],
    )
    torch.manual_seed(0)
    pred = torch.randn(8, 12)
    target = torch.randn(8, 12)
    got = est._regression_loss(pred, target)
    groups = [(0, 3, 0.2), (3, 6, 0.2), (6, 9, 1.0), (9, 12, 1.0)]
    expected = sum(F.mse_loss(pred[:, s:e] * w, target[:, s:e] * w) for s, e, w in groups)
    assert torch.allclose(got, expected)
    # And it must differ from the old all-dim weighted mean (the migration bug).
    per_dim_w = torch.tensor([0.2] * 6 + [1.0] * 6)
    old = (per_dim_w * F.mse_loss(pred, target, reduction="none")).mean()
    assert not torch.allclose(got, old)


def test_cse_ppo_update_runs_end_to_end():
    """A full CSEPPO.update() cycle runs without error with the reordered
    PPO-step-then-estimator-step and the non-detached encoder, returns finite
    losses, and updates the network parameters.
    """
    import math

    from unilab.algos.torch.cse_ppo.algorithm import CSEPPO

    torch.manual_seed(0)
    num_envs, num_steps = 8, 6
    history, one_step, num_actions, critic_dim = 4, 10, 3, 12
    num_actor_obs = history * one_step

    ac = CSEActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=critic_dim,
        num_one_step_obs=one_step,
        num_actions=num_actions,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        estimator={
            "num_pred": 12,
            "latent_dim": 8,
            "enc_hidden_dims": [32],
            "dec_hidden_dims": [16],
            "target_group_sizes": [3, 3, 3, 3],
            "target_weights": [0.2, 0.2, 1.0, 1.0],
            "target_start": 0,
            "learning_rate": 1e-3,
        },
    )
    alg = CSEPPO(
        ac,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        schedule="fixed",
        desired_kl=None,
    )
    alg.init_storage(num_envs, num_steps, [num_actor_obs], [critic_dim], [num_actions])

    obs = torch.randn(num_envs, num_actor_obs)
    critic_obs = torch.randn(num_envs, critic_dim)
    for _ in range(num_steps):
        alg.act(obs, critic_obs)
        next_obs = torch.randn(num_envs, num_actor_obs)
        next_critic = torch.randn(num_envs, critic_dim)
        rewards = torch.randn(num_envs)
        dones = torch.zeros(num_envs, dtype=torch.bool)
        alg.process_env_step(next_obs, rewards, dones, {})
        obs, critic_obs = next_obs, next_critic
    alg.compute_returns(critic_obs)

    enc_before = ac.estimator.encoder[0].weight.detach().clone()
    actor_before = ac.actor[0].weight.detach().clone()
    value_loss, surrogate_loss, estimation_loss = alg.update()

    assert math.isfinite(value_loss)
    assert math.isfinite(surrogate_loss)
    assert math.isfinite(estimation_loss)
    # Both the encoder and the actor were updated during the cycle.
    assert not torch.allclose(enc_before, ac.estimator.encoder[0].weight)
    assert not torch.allclose(actor_before, ac.actor[0].weight)


def test_lr_adaptation_respects_configurable_clamp():
    """The adaptive-KL learning rate must clamp to the configured
    [min_learning_rate, max_learning_rate]. The floor used to be a hardcoded
    1e-5; raising it keeps late training productive instead of stalling.
    """
    from unilab.algos.torch.cse_ppo.algorithm import CSEPPO

    ac = _make_actor_critic(history=4, one_step=10, latent=8)
    alg = CSEPPO(
        ac,
        schedule="adaptive",
        desired_kl=0.01,
        learning_rate=1e-3,
        min_learning_rate=5e-5,
        max_learning_rate=1e-2,
    )
    # KL >> 2*desired repeatedly -> LR falls, but NOT below min_learning_rate.
    for _ in range(100):
        alg._adapt_learning_rate(1.0)
    assert alg.learning_rate == pytest.approx(5e-5)
    assert alg.optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    # KL << desired/2 repeatedly -> LR rises to max_learning_rate.
    for _ in range(100):
        alg._adapt_learning_rate(1e-9)
    assert alg.learning_rate == pytest.approx(1e-2)


def test_policy_std_is_clamped_positive():
    """The action-noise std is an unconstrained Parameter used as the Normal
    scale; a low entropy_coef + aggressive update can drive it <= 0, crashing
    ``Normal()`` ("std >= 0.0"). It must be clamped to a positive floor.
    """
    from unilab.algos.torch.cse_ppo.algorithm import CSEPPO

    ac = _make_actor_critic(history=4, one_step=10, latent=8)
    alg = CSEPPO(ac, min_policy_std=1e-2)
    ac.std.data.fill_(-0.5)  # simulate a bad optimizer step that overshot past 0
    alg._clamp_policy_std()
    assert bool((ac.std.data >= 1e-2 - 1e-9).all())
    assert bool(torch.isfinite(ac.std.data).all())


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
