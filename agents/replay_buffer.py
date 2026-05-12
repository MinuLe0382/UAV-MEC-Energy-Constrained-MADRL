"""Replay buffer for multi-agent DDPG."""
from __future__ import annotations

import numpy as np


class MultiAgentReplayBuffer:
    """Stores transitions (obs, action, reward, next_obs) for M agents.

    Each transition stores per-agent data as fixed-shape arrays.
    """

    def __init__(self, capacity: int, num_agents: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.M = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, num_agents, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, num_agents), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.size = 0

    def store(self, obs, act, rew, next_obs, done):
        i = self.idx
        self.obs[i] = np.asarray(obs)
        self.act[i] = np.asarray(act)
        self.rew[i] = np.asarray(rew)
        self.next_obs[i] = np.asarray(next_obs)
        self.done[i] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],        # (B, M, obs_dim)
            self.act[idx],        # (B, M, act_dim)
            self.rew[idx],        # (B, M)
            self.next_obs[idx],   # (B, M, obs_dim)
            self.done[idx],       # (B, 1)
        )

    def __len__(self) -> int:
        return self.size
