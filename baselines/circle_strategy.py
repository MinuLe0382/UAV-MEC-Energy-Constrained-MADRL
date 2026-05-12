"""CIRCLE baseline: each UAV orbits the centroid of its assigned SD cluster."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from config import SYS, UNSPEC


class CircleStrategy:
    """Cluster SDs into M groups; each UAV flies on a circle around its group centroid."""

    def __init__(self, num_agents: int, act_dim: int = 3):
        self.M = num_agents
        self.act_dim = act_dim
        self.r_max = float(SYS["r_max"])
        self.d_max = float(SYS["d_max"])
        self.f_max = float(SYS["f_max"])
        self.f_min = float(SYS["f_min"])
        self.f_fixed = float(UNSPEC["circle_baseline_freq"])
        self.l_max = float(SYS["l_max"])
        self.centroids = None
        self.angles = None
        self.assigned = None

    def reset(self, env) -> None:
        kmeans = KMeans(n_clusters=self.M, n_init=10, random_state=0).fit(env.sd_pos)
        centroids = kmeans.cluster_centers_

        # Assign each UAV to its nearest centroid (greedy)
        uav_pos = env.uav_pos.copy()
        assigned = -np.ones(self.M, dtype=int)
        used = np.zeros(self.M, dtype=bool)
        for m in range(self.M):
            dists = np.linalg.norm(centroids - uav_pos[m], axis=1)
            order = np.argsort(dists)
            for c in order:
                if not used[c]:
                    assigned[m] = c
                    used[c] = True
                    break
        self.centroids = centroids
        self.assigned = assigned
        # Each UAV starts at its own current angle relative to centroid
        self.angles = np.zeros(self.M, dtype=np.float32)
        for m in range(self.M):
            v = uav_pos[m] - centroids[assigned[m]]
            self.angles[m] = np.arctan2(v[1], v[0])

    def act(self, env) -> np.ndarray:
        if self.centroids is None:
            self.reset(env)
        actions = np.zeros((self.M, self.act_dim), dtype=np.float32)
        # Step angle so the chord length is approximately d_max/2 (gentle orbit)
        step_angle = 0.5 * self.d_max / max(self.r_max, 1e-3)
        for m in range(self.M):
            self.angles[m] += step_angle
            target = self.centroids[self.assigned[m]] + self.r_max * np.array(
                [np.cos(self.angles[m]), np.sin(self.angles[m])]
            )
            target = np.clip(target, 0.0, self.l_max)
            delta = target - env.uav_pos[m]
            dist = np.linalg.norm(delta)
            theta = np.arctan2(delta[1], delta[0]) if dist > 1e-6 else 0.0
            d = min(dist, self.d_max)

            # Encode back into [-1, 1] action space
            d_norm = (d / self.d_max) * 2.0 - 1.0
            theta_norm = (theta % (2.0 * np.pi)) / (2.0 * np.pi) * 2.0 - 1.0
            f_norm = ((self.f_fixed - self.f_min) / (self.f_max - self.f_min)) * 2.0 - 1.0
            actions[m] = [d_norm, theta_norm, f_norm]
        return actions
