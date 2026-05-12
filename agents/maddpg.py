"""MADDPG with parameter sharing (Algorithm 1 in Shi et al. 2026).

Changes from v1:
- Single shared Actor (all UAVs share weights); each UAV's one-hot ID in obs
  disambiguates identity, enabling specialization while sharing experience.
- M independent Critics (centralized: all obs + all actions).
- Slower noise floor: sigma_final 0.05 -> 0.10
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from config import RL
from agents.networks import Actor, CentralizedCritic
from agents.replay_buffer import MultiAgentReplayBuffer


class MADDPG:
    def __init__(self, num_agents: int, obs_dim: int, act_dim: int,
                 device: str | torch.device = "cpu"):
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

        # ---- Parameter sharing: ONE shared actor ----
        self.actor = Actor(obs_dim, act_dim, self.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        for p in self.actor_target.parameters():
            p.requires_grad = False
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # ---- Per-agent centralized critics (unchanged) ----
        self.critics = [
            CentralizedCritic(num_agents * obs_dim, num_agents * act_dim, self.hidden_dim).to(self.device)
            for _ in range(num_agents)
        ]
        self.critic_targets = [copy.deepcopy(c) for c in self.critics]
        for c in self.critic_targets:
            for p in c.parameters():
                p.requires_grad = False
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=self.critic_lr) for c in self.critics]

        self.buffer = MultiAgentReplayBuffer(
            int(RL["buffer_size"]), num_agents, obs_dim, act_dim
        )
        self._noise_sigma = self.noise_sigma_init

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
        obs = torch.as_tensor(obs, device=self.device)            # (B, M, obs)
        act = torch.as_tensor(act, device=self.device)            # (B, M, act)
        rew = torch.as_tensor(rew, device=self.device)            # (B, M)
        next_obs = torch.as_tensor(next_obs, device=self.device)  # (B, M, obs)
        done = torch.as_tensor(done, device=self.device)          # (B, 1)

        B = obs.shape[0]
        all_obs = obs.reshape(B, -1)
        all_act = act.reshape(B, -1)
        all_next_obs = next_obs.reshape(B, -1)

        # Target next actions: shared actor_target applied to each agent's next obs
        with torch.no_grad():
            next_obs_flat = next_obs.reshape(B * self.M, self.obs_dim)
            next_acts_flat = self.actor_target(next_obs_flat)           # (B*M, act)
            all_next_act = next_acts_flat.reshape(B, self.M * self.act_dim)

        critic_losses = []
        for m in range(self.M):
            with torch.no_grad():
                q_next = self.critic_targets[m](all_next_obs, all_next_act).squeeze(-1)
                y = rew[:, m] + self.gamma * (1.0 - done.squeeze(-1)) * q_next
            q = self.critics[m](all_obs, all_act).squeeze(-1)
            critic_loss = F.mse_loss(q, y)
            self.critic_opts[m].zero_grad()
            critic_loss.backward()
            self.critic_opts[m].step()
            critic_losses.append(critic_loss.item())

        # Actor update: shared actor, gradient averaged across all M critics
        # For each agent m, critic_m sees actions where agent m uses current actor
        # and others use stored actions (deterministic PG).
        obs_flat = obs.reshape(B * self.M, self.obs_dim)
        curr_acts_flat = self.actor(obs_flat)                          # (B*M, act)
        curr_acts = curr_acts_flat.reshape(B, self.M, self.act_dim)

        actor_losses = []
        self.actor_opt.zero_grad()
        for m in range(self.M):
            joint_act = torch.cat([
                curr_acts[:, j, :] if j == m else act[:, j, :]
                for j in range(self.M)
            ], dim=-1)
            actor_loss_m = -self.critics[m](all_obs, joint_act).mean()
            actor_losses.append(actor_loss_m)

        # Sum losses (gradients accumulate), then single backward + step
        total_actor_loss = sum(actor_losses)
        total_actor_loss.backward()
        self.actor_opt.step()

        # Soft target updates
        self._soft_update(self.actor_target, self.actor)
        for m in range(self.M):
            self._soft_update(self.critic_targets[m], self.critics[m])

        return {
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(total_actor_loss.item() / self.M),
            "noise_sigma": self._noise_sigma,
        }

    def _soft_update(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)

    # ---------------- checkpoint ----------------
    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critics": [c.state_dict() for c in self.critics],
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        # Support both old format (list of actors) and new (single actor)
        if "actor" in ckpt:
            self.actor.load_state_dict(ckpt["actor"])
        else:
            self.actor.load_state_dict(ckpt["actors"][0])
        for c, sd in zip(self.critics, ckpt["critics"]):
            c.load_state_dict(sd)
        self.actor_target = copy.deepcopy(self.actor)
        for p in self.actor_target.parameters():
            p.requires_grad = False
        self.critic_targets = [copy.deepcopy(c) for c in self.critics]
        for c in self.critic_targets:
            for p in c.parameters():
                p.requires_grad = False
