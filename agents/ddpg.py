"""Single-agent DDPG baseline (shared policy across UAVs, local critic)."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from config import RL
from agents.networks import Actor, LocalCritic
from agents.replay_buffer import MultiAgentReplayBuffer


class DDPG:
    """Shared-parameter DDPG: one actor and one critic shared by all UAVs.

    Each UAV uses its own local observation only — both actor input and critic
    input are local. Multi-agent transitions are stored separately.
    """

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

        self.actor = Actor(obs_dim, act_dim, self.hidden_dim).to(self.device)
        self.critic = LocalCritic(obs_dim, act_dim, self.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        for t in [self.actor_target, self.critic_target]:
            for p in t.parameters():
                p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.buffer = MultiAgentReplayBuffer(
            int(RL["buffer_size"]), num_agents, obs_dim, act_dim
        )

        self._noise_sigma = self.noise_sigma_init

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

    def store(self, obs, act, rew, next_obs, done) -> None:
        self.buffer.store(obs, act, rew, next_obs, done)

    def update(self) -> dict | None:
        if len(self.buffer) < self.batch_size:
            return None

        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size)
        obs = torch.as_tensor(obs, device=self.device)            # (B, M, obs)
        act = torch.as_tensor(act, device=self.device)            # (B, M, act)
        rew = torch.as_tensor(rew, device=self.device)            # (B, M)
        next_obs = torch.as_tensor(next_obs, device=self.device)
        done = torch.as_tensor(done, device=self.device)

        # Flatten across agents — treat each agent transition as an independent sample
        B = obs.shape[0] * self.M
        obs_f = obs.reshape(B, self.obs_dim)
        act_f = act.reshape(B, self.act_dim)
        rew_f = rew.reshape(B)
        next_obs_f = next_obs.reshape(B, self.obs_dim)
        done_f = done.repeat(1, self.M).reshape(B)

        # Critic update
        with torch.no_grad():
            next_a = self.actor_target(next_obs_f)
            q_next = self.critic_target(next_obs_f, next_a).squeeze(-1)
            y = rew_f + self.gamma * (1.0 - done_f) * q_next
        q = self.critic(obs_f, act_f).squeeze(-1)
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        actor_loss = -self.critic(obs_f, self.actor(obs_f)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Target soft update
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "noise_sigma": self._noise_sigma,
        }

    def _soft_update(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * sp.data)

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
