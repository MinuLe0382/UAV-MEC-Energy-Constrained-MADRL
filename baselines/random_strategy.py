"""RANDOM baseline: uniformly random actions."""
from __future__ import annotations

import numpy as np


class RandomStrategy:
    def __init__(self, num_agents: int, act_dim: int = 3, seed: int | None = None):
        self.M = num_agents
        self.act_dim = act_dim
        self.rng = np.random.default_rng(seed)

    def act(self, env=None) -> np.ndarray:
        # Uniformly random in [-1, 1] (env decodes into physical units).
        return self.rng.uniform(-1.0, 1.0, size=(self.M, self.act_dim)).astype(np.float32)
