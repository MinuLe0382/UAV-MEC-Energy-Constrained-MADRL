"""GREEDY baseline: each UAV heads to the SD with the largest queue (oracle access)."""
from __future__ import annotations

import numpy as np

from config import SYS


class GreedyStrategy:
    """Greedy oracle: knows L_k,t for all SDs and picks the largest one not causing collisions."""

    def __init__(self, num_agents: int, act_dim: int = 3):
        self.M = num_agents
        self.act_dim = act_dim
        self.d_max = float(SYS["d_max"])
        self.r_min = float(SYS["r_min"])
        self.f_max = float(SYS["f_max"])
        self.f_min = float(SYS["f_min"])

    def reset(self, env) -> None:
        pass

    def act(self, env) -> np.ndarray:
        actions = np.zeros((self.M, self.act_dim), dtype=np.float32)
        # Greedy assignment: order UAVs and let each pick the largest L_k SD not yet picked
        order = np.argsort(-env.queue)  # desc
        taken = np.zeros(env.K, dtype=bool)
        # Predict next positions to avoid collisions
        planned_pos = env.uav_pos.copy()

        for m in range(self.M):
            target_idx = None
            for k in order:
                if taken[k] or env.queue[k] <= 0:
                    continue
                # Try this target
                target = env.sd_pos[k]
                delta = target - env.uav_pos[m]
                dist = np.linalg.norm(delta)
                d = min(dist, self.d_max)
                theta = np.arctan2(delta[1], delta[0]) if dist > 1e-6 else 0.0
                tentative = env.uav_pos[m] + d * np.array([np.cos(theta), np.sin(theta)])
                # Check collision against already-planned positions
                ok = True
                for j in range(self.M):
                    if j == m:
                        continue
                    if np.linalg.norm(tentative - planned_pos[j]) < self.r_min:
                        ok = False
                        break
                if ok:
                    target_idx = k
                    planned_pos[m] = tentative
                    break

            if target_idx is None:
                # No reachable non-colliding SD: stay in place at minimum freq
                d, theta, f = 0.0, 0.0, self.f_max
            else:
                target = env.sd_pos[target_idx]
                delta = target - env.uav_pos[m]
                dist = np.linalg.norm(delta)
                d = min(dist, self.d_max)
                theta = np.arctan2(delta[1], delta[0]) if dist > 1e-6 else 0.0
                f = self.f_max
                taken[target_idx] = True

            d_norm = (d / self.d_max) * 2.0 - 1.0
            theta_norm = (theta % (2.0 * np.pi)) / (2.0 * np.pi) * 2.0 - 1.0
            f_norm = ((f - self.f_min) / (self.f_max - self.f_min)) * 2.0 - 1.0
            actions[m] = [d_norm, theta_norm, f_norm]
        return actions
