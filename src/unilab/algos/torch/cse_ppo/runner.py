# SPDX-License-Identifier: BSD-3-Clause
#
# CSE-PPO OnPolicy Runner for UniLab. Mirrors the HIM-PPO runner interface so
# train_cse_ppo.py shares most logic with train_rsl_rl.py.

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Callable, cast

import torch
from tensordict import TensorDict

from unilab.algos.torch.cse_ppo.actor_critic import CSEActorCritic
from unilab.algos.torch.cse_ppo.algorithm import CSEPPO


class _CSELogger:
    def __init__(self) -> None:
        self.rewbuffer: deque[float] = deque(maxlen=100)
        self.lenbuffer: deque[float] = deque(maxlen=100)
        self.tot_timesteps: int = 0


class CSEOnPolicyRunner:
    """On-policy training runner for CSE-PPO."""

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.device = device
        self.log_dir = log_dir
        self.current_learning_iteration: int = 0
        self.logger = _CSELogger()

        cfg: dict[str, Any] = dict(train_cfg)
        num_one_step_obs = int(cfg["num_one_step_obs"])
        num_actor_history = int(cfg.get("num_actor_history", 1))
        num_actor_obs = num_actor_history * num_one_step_obs
        num_critic_obs = int(getattr(env, "num_privileged_obs", None) or env.num_obs)
        num_actions = int(env.num_actions)

        policy_cfg: dict[str, Any] = dict(cfg.get("policy") or {})
        estimator_cfg: dict[str, Any] = dict(cfg.get("estimator") or {})
        algo_cfg: dict[str, Any] = dict(cfg.get("algorithm") or {})

        self.actor_critic = CSEActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_one_step_obs=num_one_step_obs,
            num_actions=num_actions,
            actor_hidden_dims=list(policy_cfg.get("actor_hidden_dims", [512, 256, 128])),
            critic_hidden_dims=list(policy_cfg.get("critic_hidden_dims", [512, 256, 128])),
            activation=str(policy_cfg.get("activation", "elu")),
            init_noise_std=float(policy_cfg.get("init_noise_std", 1.0)),
            estimator=estimator_cfg,
        ).to(device)

        self.alg = CSEPPO(self.actor_critic, device=device, **algo_cfg)

        self.num_steps_per_env = int(cfg.get("num_steps_per_env", 24))
        self.save_interval = int(cfg.get("save_interval", 100))

        self.alg.init_storage(
            env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [num_actions],
        )

        self._ep_returns = torch.zeros(env.num_envs, device=device)
        self._ep_lengths = torch.zeros(env.num_envs, device=device)

        self._writer: Any = None
        if log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]

                self._writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                pass

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = True,
    ) -> None:
        obs_td, _ = self.env.reset()
        obs = obs_td["actor"].to(self.device)
        critic_obs = obs_td.get("critic", obs).to(self.device)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        self.alg.train_mode()
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        wall_start = time.time()

        for it in range(start_iter, tot_iter):
            infos: dict[str, Any] = {}
            collect_start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs_td, rewards, dones, infos = self.env.step(actions)
                    next_obs = obs_td["actor"].to(self.device)
                    next_critic_obs = obs_td.get("critic", next_obs).to(self.device)

                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    self._ep_returns += rewards.to(self.device)
                    self._ep_lengths += 1
                    if len(done_ids) > 0:
                        for idx in done_ids:
                            self.logger.rewbuffer.append(float(self._ep_returns[idx]))
                            self.logger.lenbuffer.append(float(self._ep_lengths[idx]))
                        self._ep_returns[done_ids] = 0.0
                        self._ep_lengths[done_ids] = 0.0

                    self.alg.process_env_step(obs_td, rewards, dones, infos)
                    obs = next_obs
                    critic_obs = next_critic_obs

                self.alg.compute_returns(critic_obs)
            collection_time = time.time() - collect_start

            learn_start = time.time()
            value_loss, surrogate_loss, estimation_loss = self.alg.update()
            learn_time = time.time() - learn_start

            self.current_learning_iteration = it + 1
            num_steps = self.num_steps_per_env * self.env.num_envs
            self.logger.tot_timesteps += num_steps
            iter_time = collection_time + learn_time
            fps = int(num_steps / iter_time) if iter_time > 0 else 0
            mean_std = float(self.actor_critic.std.mean().detach())
            stats = {
                "value_loss": value_loss,
                "surrogate_loss": surrogate_loss,
                "estimation_loss": estimation_loss,
                "learning_rate": float(self.alg.learning_rate),
                "mean_noise_std": mean_std,
                "fps": fps,
                "collection_time": collection_time,
                "learn_time": learn_time,
            }

            self._print_iter(it + 1, tot_iter, stats, time.time() - wall_start, infos)
            if self._writer is not None:
                step = self.current_learning_iteration
                self._writer.add_scalar("Loss/value", value_loss, step)
                self._writer.add_scalar("Loss/surrogate", surrogate_loss, step)
                self._writer.add_scalar("Loss/estimation", estimation_loss, step)
                self._writer.add_scalar("Loss/learning_rate", float(self.alg.learning_rate), step)
                self._writer.add_scalar("Policy/mean_noise_std", mean_std, step)
                self._writer.add_scalar("Perf/total_fps", fps, step)
                self._writer.add_scalar("Perf/collection_time", collection_time, step)
                self._writer.add_scalar("Perf/learning_time", learn_time, step)
                self._writer.add_scalar("Perf/total_timesteps", self.logger.tot_timesteps, step)
                if self.logger.rewbuffer:
                    mean_rew = sum(self.logger.rewbuffer) / len(self.logger.rewbuffer)
                    mean_len = sum(self.logger.lenbuffer) / len(self.logger.lenbuffer)
                    self._writer.add_scalar("Train/mean_reward", mean_rew, step)
                    self._writer.add_scalar("Train/mean_episode_length", mean_len, step)
                    self._writer.add_scalar("Train/mean_reward_per_step", mean_rew, self.logger.tot_timesteps)
                for k, v in (infos.get("log") or {}).items():
                    self._writer.add_scalar(k, v, step)

            if (
                self.log_dir is not None
                and self.current_learning_iteration % self.save_interval == 0
            ):
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iteration": self.current_learning_iteration,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor_critic.load_state_dict(ckpt["actor_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.alg.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "iteration" in ckpt:
            self.current_learning_iteration = int(ckpt["iteration"])

    def get_inference_policy(self, device: str | None = None) -> Callable[..., Any]:
        self.actor_critic.eval()
        if device is not None:
            self.actor_critic.to(device)
        return cast(Callable[..., Any], self.actor_critic.act_inference)

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the end-to-end CSE-PPO policy (estimator + actor) via tracing."""
        orig_device = next(self.actor_critic.parameters()).device
        ac = self.actor_critic.cpu().eval()
        num_one_step_obs = ac.num_one_step_obs

        class _PolicyExport(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.estimator = ac.estimator
                self.actor_mlp = ac.actor

            def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
                latent = self.estimator.get_latent(obs_history)
                # Current single-step obs is the newest (last) frame; see _actor_input.
                actor_input = torch.cat((obs_history[:, -num_one_step_obs:], latent), dim=-1)
                return self.actor_mlp(actor_input)

        model = _PolicyExport().eval()
        dummy = torch.zeros(1, ac.num_actor_obs)
        with torch.inference_mode():
            traced = cast(torch.jit.ScriptModule, torch.jit.trace(model, (dummy,)))
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        traced.save(save_path)
        self.actor_critic.to(orig_device)
        print(f"Exported CSE-PPO policy (JIT) to {save_path}")

    def _print_iter(
        self,
        it: int,
        tot: int,
        stats: dict[str, float],
        elapsed: float,
        infos: dict,
    ) -> None:
        sep = "-" * 80
        mean_rew = (
            sum(self.logger.rewbuffer) / len(self.logger.rewbuffer)
            if self.logger.rewbuffer
            else 0.0
        )
        mean_len = (
            sum(self.logger.lenbuffer) / len(self.logger.lenbuffer)
            if self.logger.lenbuffer
            else 0.0
        )
        time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        eta = elapsed / it * (tot - it) if it > 0 else 0.0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        print(sep)
        print(f"{'Iteration':>40}: {it}/{tot}")
        print(f"{'Computation (fps)':>40}: {int(stats['fps'])} steps/s")
        print(f"{'Mean value loss':>40}: {stats['value_loss']:.4f}")
        print(f"{'Mean surrogate loss':>40}: {stats['surrogate_loss']:.4f}")
        print(f"{'Mean estimation loss':>40}: {stats['estimation_loss']:.4f}")
        print(f"{'Learning rate':>40}: {stats['learning_rate']:.2e}")
        print(f"{'Mean action noise std':>40}: {stats['mean_noise_std']:.3f}")
        if mean_rew:
            print(f"{'Mean episode reward':>40}: {mean_rew:.4f}")
        if mean_len:
            print(f"{'Mean episode length':>40}: {mean_len:.1f}")
        # Per-term reward means and any other env scalars (reward/*, etc.).
        for k, v in sorted((infos.get("log") or {}).items()):
            print(f"{k:>40}: {v:.4f}")
        print(f"{'Total timesteps':>40}: {self.logger.tot_timesteps}")
        print(f"{'Iteration time':>40}: {stats['collection_time'] + stats['learn_time']:.2f}s")
        print(f"{'Time elapsed':>40}: {time_str}")
        print(f"{'ETA':>40}: {eta_str}")
        print(sep)
