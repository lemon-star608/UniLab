# SPDX-License-Identifier: BSD-3-Clause
#
# Concurrent State Estimator (CSE) for UniLab, after Ji et al. 2022
# ("Concurrent Training of a Control Policy and a State Estimator"), extended to
# estimate end-effector position and external forces as in UniFP (CoRL 2025).
#
# Unlike the HIM estimator, this is a plain supervised encoder->latent->decoder:
# the encoder compresses the proprioceptive history into a latent that feeds the
# actor, and the decoder regresses that latent to privileged targets
# [base_lin_vel, ee_pos_sphere, force_ee, force_base] with per-group weights.
# There is no contrastive / prototype term.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.ReLU,
        "silu": nn.SiLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


class CSEEstimator(nn.Module):
    """Encoder->latent->decoder state estimator trained by supervised regression.

    Args:
        temporal_steps: history length ``H`` (number of stacked single-step obs).
        num_one_step_obs: single-step actor observation dim.
        num_pred: total dim of the estimated targets (sum of target groups).
        target_weights: optional per-dim regression weights (len == num_pred).
        target_start: offset of the target block inside the critic observation.
    """

    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        num_pred: int = 12,
        enc_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        latent_dim: int = 19,
        dec_hidden_dims: list[int] | tuple[int, ...] = (64,),
        activation: str = "elu",
        learning_rate: float = 1e-3,
        max_grad_norm: float = 10.0,
        target_weights: list[float] | tuple[float, ...] | None = None,
        target_start: int = 0,
    ) -> None:
        super().__init__()
        if temporal_steps <= 0:
            raise ValueError("temporal_steps must be positive")
        if num_one_step_obs <= 0:
            raise ValueError("num_one_step_obs must be positive")
        if num_pred <= 0:
            raise ValueError("num_pred must be positive")

        self.temporal_steps = int(temporal_steps)
        self.num_one_step_obs = int(num_one_step_obs)
        self.num_pred = int(num_pred)
        self.num_latent = int(latent_dim)
        self.max_grad_norm = float(max_grad_norm)
        self.target_start = int(target_start)

        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers: list[nn.Module] = []
        last = enc_input_dim
        for hidden_dim in enc_hidden_dims:
            enc_layers += [nn.Linear(last, int(hidden_dim)), get_activation(activation)]
            last = int(hidden_dim)
        enc_layers += [nn.Linear(last, self.num_latent)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        last = self.num_latent
        for hidden_dim in dec_hidden_dims:
            dec_layers += [nn.Linear(last, int(hidden_dim)), get_activation(activation)]
            last = int(hidden_dim)
        dec_layers += [nn.Linear(last, self.num_pred)]
        self.decoder = nn.Sequential(*dec_layers)

        if target_weights is None:
            weights = torch.ones(self.num_pred)
        else:
            weights = torch.as_tensor(list(target_weights), dtype=torch.float32)
            if weights.numel() != self.num_pred:
                raise ValueError(
                    f"target_weights length {weights.numel()} != num_pred {self.num_pred}"
                )
        self.register_buffer("target_weights", weights)

        self.learning_rate = float(learning_rate)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def encode(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs_history)

    def get_latent(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs_history.detach()).detach()

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        return self.get_latent(obs_history)

    def predict(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Decode the estimated targets (for evaluation / debugging)."""
        return self.decoder(self.encoder(obs_history.detach())).detach()

    def update(
        self,
        obs_history: torch.Tensor,
        next_critic_obs: torch.Tensor,
        lr: float | None = None,
    ) -> float:
        if lr is not None:
            self.learning_rate = float(lr)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        start = self.target_start
        end = start + self.num_pred
        if next_critic_obs.shape[-1] < end:
            raise ValueError(
                "next_critic_obs is too small for the CSE estimator target slice: "
                f"shape={tuple(next_critic_obs.shape)}, target=[{start}:{end}]"
            )
        target = next_critic_obs[:, start:end].detach()
        pred = self.decoder(self.encoder(obs_history))
        weights = self.target_weights.to(pred.device)
        loss = (weights * F.mse_loss(pred, target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())
