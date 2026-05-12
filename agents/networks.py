"""Actor and Critic networks for MADDPG / DDPG."""
from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CentralizedCritic(nn.Module):
    """Critic that observes all agents' (obs, action). Used by MADDPG."""

    def __init__(self, total_obs_dim: int, total_act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, all_obs: torch.Tensor, all_act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([all_obs, all_act], dim=-1)
        return self.net(x)


class LocalCritic(nn.Module):
    """Critic that only observes its own (obs, action). Used by DDPG baseline."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))
