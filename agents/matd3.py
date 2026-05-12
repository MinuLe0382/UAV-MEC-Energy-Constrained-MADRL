"""MATD3 — Multi-Agent Twin Delayed DDPG (Fujimoto et al. 2018, multi-agent variant).

Differences from MADDPG:
1. Twin critics per agent (Q1, Q2); target = min(Q1', Q2')   — clipped double Q
2. Target policy smoothing: noise added to target action
3. Delayed actor + target updates: every `policy_delay` critic updates

Shared with v3-style MADDPG: single Actor (parameter sharing), UAV ID in obs.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from config import RL
from agents.networks import Actor, CentralizedCritic
from agents.replay_buffer import MultiAgentReplayBuffer


class MATD3:
    def __init__(self, num_agents: int, obs_dim: int, act_dim: int,
                 device: str | torch.device = "cpu",
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_delay: int = 2):
        self.M = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(device)

        self.gamma = float(RL["gamma"])
        self.tau = float(RL["tau"])
        self.actor_lr = float(RL["actor_lr"])
        self.critic_lr = float(RL["critic_lr"])
        self.batch_size = int(RL["batch_size"])
        self.hidden_dim = int(RL["hidden_dim"])
        self.noise_sigma_init = float(RL["noise_sigma_init"])
        self.noise_sigma_final = float(RL["noise_sigma_final"])

        self.policy_noise = policy_noise   # target policy smoothing sigma
        self.noise_clip = noise_clip       # clamp range for smoothing noise
        self.policy_delay = policy_delay   # actor update frequency

        # ---- Shared actor ----
        self.actor = Actor(obs_dim, act_dim, self.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        for p in self.actor_target.parameters():
            p.requires_grad = False
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # ---- Twin centralized critics per agent ----
        self.critics1 = [
            CentralizedCritic(num_agents * obs_dim, num_agents * act_dim, self.hidden_dim).to(self.device)
            for _ in range(num_agents)
        ]
        self.critics2 = [
            CentralizedCritic(num_agents * obs_dim, num_agents * act_dim, self.hidden_dim).to(self.device)
            for _ in range(num_agents)
        ]
        self.critic1_targets = [copy.deepcopy(c) for c in self.critics1]
        self.critic2_targets = [copy.deepcopy(c) for c in self.critics2]
        for c in self.critic1_targets + self.critic2_targets:
            for p in c.parameters():
                p.requires_grad = False
        self.critic1_opts = [torch.optim.Adam(c.parameters(), lr=self.critic_lr) for c in self.critics1]
        self.critic2_opts = [torch.optim.Adam(c.parameters(), lr=self.critic_lr) for c in self.critics2]

        self.buffer = MultiAgentReplayBuffer(
            int(RL["buffer_size"]), num_agents, obs_dim, act_dim
        )
        self._noise_sigma = self.noise_sigma_init
        self._update_step = 0

    # ---------------- exploration ----------------
    def set_noise(self, progress: float) -> None:
        progress = max(0.0, min(1.0, progress))
        self._noise_sigma = (
            self.noise_sigma_init * (1 - progress) + self.noise_sigma_final * progress
        )

    @torch.no_grad()
    def select_actions(self, obs_list, noise: bool = True) -> np.ndarray:
        actions = np.zeros((self.M, self.act_dim), dtype=np.float32)
        for m, o in enumerate(obs_list):
            o_t = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = self.actor(o_t).cpu().numpy()[0]
            if noise:
                a = a + np.random.normal(0.0, self._noise_sigma, size=self.act_dim)
            actions[m] = np.clip(a, -1.0, 1.0)
        return actions

    # ---------------- learning ----------------
    def store(self, obs, act, rew, next_obs, done) -> None:
        self.buffer.store(obs, act, rew, next_obs, done)

    def update(self) -> dict | None:
        if len(self.buffer) < self.batch_size:
            return None

        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size)
        obs = torch.as_tensor(obs, device=self.device)
        act = torch.as_tensor(act, device=self.device)
        rew = torch.as_tensor(rew, device=self.device)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        done = torch.as_tensor(done, device=self.device)

        B = obs.shape[0]
        all_obs = obs.reshape(B, -1)
        all_act = act.reshape(B, -1)
        all_next_obs = next_obs.reshape(B, -1)

        # ---- Target actions with policy smoothing ----
        with torch.no_grad():
            next_obs_flat = next_obs.reshape(B * self.M, self.obs_dim)
            next_acts_flat = self.actor_target(next_obs_flat)
            smoothing = (torch.randn_like(next_acts_flat) * self.policy_noise
                         ).clamp(-self.noise_clip, self.noise_clip)
            next_acts_flat = (next_acts_flat + smoothing).clamp(-1.0, 1.0)
            all_next_act = next_acts_flat.reshape(B, self.M * self.act_dim)

        # ---- Critic updates (both Q1 and Q2 per agent) ----
        critic1_losses, critic2_losses = [], []
        for m in range(self.M):
            with torch.no_grad():
                q1_next = self.critic1_targets[m](all_next_obs, all_next_act).squeeze(-1)
                q2_next = self.critic2_targets[m](all_next_obs, all_next_act).squeeze(-1)
                q_next = torch.min(q1_next, q2_next)
                y = rew[:, m] + self.gamma * (1.0 - done.squeeze(-1)) * q_next

            q1 = self.critics1[m](all_obs, all_act).squeeze(-1)
            q2 = self.critics2[m](all_obs, all_act).squeeze(-1)
            c1_loss = F.mse_loss(q1, y)
            c2_loss = F.mse_loss(q2, y)

            self.critic1_opts[m].zero_grad(); c1_loss.backward(); self.critic1_opts[m].step()
            self.critic2_opts[m].zero_grad(); c2_loss.backward(); self.critic2_opts[m].step()

            critic1_losses.append(c1_loss.item())
            critic2_losses.append(c2_loss.item())

        self._update_step += 1
        actor_loss_val = 0.0

        # ---- Delayed actor + target updates ----
        if self._update_step % self.policy_delay == 0:
            obs_flat = obs.reshape(B * self.M, self.obs_dim)
            curr_acts_flat = self.actor(obs_flat)
            curr_acts = curr_acts_flat.reshape(B, self.M, self.act_dim)

            self.actor_opt.zero_grad()
            actor_losses = []
            for m in range(self.M):
                joint_act = torch.cat([
                    curr_acts[:, j, :] if j == m else act[:, j, :]
                    for j in range(self.M)
                ], dim=-1)
                # Use Q1 for the actor objective (standard TD3)
                actor_losses.append(-self.critics1[m](all_obs, joint_act).mean())
            total_actor_loss = sum(actor_losses)
            total_actor_loss.backward()
            self.actor_opt.step()
            actor_loss_val = float(total_actor_loss.item() / self.M)

            # Soft target updates (also delayed)
            self._soft_update(self.actor_target, self.actor)
            for m in range(self.M):
                self._soft_update(self.critic1_targets[m], self.critics1[m])
                self._soft_update(self.critic2_targets[m], self.critics2[m])

        return {
            "critic_loss": float(np.mean(critic1_losses + critic2_losses)),
            "actor_loss": actor_loss_val,
            "noise_sigma": self._noise_sigma,
        }

    def _soft_update(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)

    # ---------------- checkpoint ----------------
    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critics1": [c.state_dict() for c in self.critics1],
            "critics2": [c.state_dict() for c in self.critics2],
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        for c, sd in zip(self.critics1, ckpt["critics1"]):
            c.load_state_dict(sd)
        for c, sd in zip(self.critics2, ckpt["critics2"]):
            c.load_state_dict(sd)
        self.actor_target = copy.deepcopy(self.actor)
        for p in self.actor_target.parameters():
            p.requires_grad = False
        self.critic1_targets = [copy.deepcopy(c) for c in self.critics1]
        self.critic2_targets = [copy.deepcopy(c) for c in self.critics2]
        for c in self.critic1_targets + self.critic2_targets:
            for p in c.parameters():
                p.requires_grad = False
