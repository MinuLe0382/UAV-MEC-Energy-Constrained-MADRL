"""Multi-UAV assisted MEC environment (Shi et al. 2026, Sec. III).

Fully vectorized implementation: all per-UAV / per-SD inner loops are
replaced with NumPy array operations.
"""
from __future__ import annotations

import numpy as np

from config import SYS, UNSPEC


class UAVMECEnv:
    """Multi-UAV MEC simulation environment.

    Action per UAV (continuous, normalized in [-1, 1]):
        a[0] -> distance d in [0, d_max]
        a[1] -> direction theta in [0, 2*pi]
        a[2] -> CPU frequency f in [f_min, f_max]

    Observation per UAV (dim = 2 + K + M + (M-1)):
        - own (x, y) coordinates (2)
        - cumulative service counts to each SD by this UAV (K)
        - cumulative total service counts of every UAV (M)
        - distances to other UAVs (M - 1)
    """

    def __init__(self, alpha_dist: str = "uniform", alpha_params: tuple | None = None,
                 seed: int | None = None):
        self.M = SYS["num_uavs"]
        self.K = SYS["num_sds"]
        self.l_max = float(SYS["l_max"])
        self.T = int(SYS["T"])
        self.r_min = float(SYS["r_min"])
        self.r_max = float(SYS["r_max"])
        self.d_max = float(SYS["d_max"])
        self.v_uav = float(SYS["v_uav"])
        self.w_uav = float(SYS["w_uav"])
        self.e_uav = float(SYS["e_uav"])
        self.task_mean = float(SYS["task_mean"])
        self.task_std = float(SYS["task_std"])
        self.l_queue_max = float(SYS["l_queue_max"])
        self.trans_rate = float(SYS["trans_rate"])
        self.k1 = float(SYS["k1"])
        self.k2 = float(SYS["k2"])
        self.cyc = float(SYS["cyc"])
        self.f_max = float(SYS["f_max"])
        self.f_min = float(SYS["f_min"])
        self.penalty = float(SYS["penalty"])
        self.uav_init = np.array(SYS["uav_init_positions"], dtype=np.float32)

        self.p_tran = float(UNSPEC["p_tran_watt"])
        self.reward_scale = float(UNSPEC["reward_scale_N_divisor"])
        self.energy_scale = float(UNSPEC["energy_scale"])
        self.individual_bonus = float(UNSPEC.get("individual_bonus_coef", 0.0))
        self.use_fairness_reward = bool(UNSPEC.get("use_fairness_reward", True))
        self.movement_energy_mult = float(UNSPEC.get("movement_energy_mult", 1.0))
        self.energy_overrun_base = float(UNSPEC.get("energy_overrun_penalty_base", 1.0))
        self.energy_overrun_per_excess = float(UNSPEC.get("energy_overrun_penalty_per_excess", 0.0))

        self.alpha_dist = alpha_dist
        self.alpha_params = alpha_params or (0.5, 0.2)

        self.rng = np.random.default_rng(seed)

        # obs: coords(2) + per-SD service(K) + per-UAV total(M) + inter-UAV dist(M-1) + UAV id one-hot(M)
        self.obs_dim = 2 + self.K + self.M + (self.M - 1) + self.M
        self.act_dim = 3

        # Pre-compute mask used in observation: (M, M) without diagonal
        self._other_mask = ~np.eye(self.M, dtype=bool)
        # Pre-compute UAV one-hot IDs: (M, M)
        self._uav_ids = np.eye(self.M, dtype=np.float32)

        self.reset()

    # ---------------- public API ----------------
    def reset(self) -> list[np.ndarray]:
        # SD positions
        self.sd_pos = self.rng.uniform(0, self.l_max, size=(self.K, 2)).astype(np.float32)

        # Per-SD task occurrence rates
        if self.alpha_dist == "normal":
            mu, sigma = self.alpha_params
            self.alpha = np.clip(self.rng.normal(mu, sigma, size=self.K), 1e-3, 1.0)
        else:
            self.alpha = self.rng.uniform(0.0, 1.0, size=self.K)
        self.alpha = self.alpha.astype(np.float32)

        # UAV state
        self.uav_pos = self.uav_init.copy()
        self.energy_remaining = np.full(self.M, self.e_uav, dtype=np.float32)
        self.energy_total_used = np.zeros(self.M, dtype=np.float32)
        self.active = np.ones(self.M, dtype=bool)
        # Energy-budget violation tracking (per UAV across the episode)
        self.energy_violated = np.zeros(self.M, dtype=bool)        # did E_total exceed E_uav at any point
        self.violation_slot = -np.ones(self.M, dtype=np.int32)     # first slot of violation, -1 if none
        self.cumulative_excess = np.zeros(self.M, dtype=np.float32) # sum of per-slot excess across episode
        self.peak_excess = np.zeros(self.M, dtype=np.float32)       # max single-slot excess

        # Queues and cumulative service counts
        self.queue = np.zeros(self.K, dtype=np.float32)
        self.Z = np.zeros((self.M, self.K), dtype=np.float32)
        self.total_bits = 0.0
        # Total task volume generated per SD across the episode (for analysis)
        self.generated_per_sd = np.zeros(self.K, dtype=np.float32)
        # Cumulative bits served per SD across the episode (for analysis)
        self.served_per_sd = np.zeros(self.K, dtype=np.float32)

        self.t = 0
        return self._all_obs()

    def step(self, actions: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, bool, dict]:
        actions = np.asarray(actions, dtype=np.float32).reshape(self.M, self.act_dim)

        # 1. Generate new tasks
        self._generate_tasks()

        # 2. Decode actions
        d, theta, f = self._decode_actions(actions)

        # 3. Move UAVs (clip to boundary; track violations)
        new_pos, boundary_hit = self._move_uavs(d, theta)
        self.uav_pos = new_pos

        # 4. Detect collisions
        collision_hit = self._collision_mask(self.uav_pos)

        # 5. Service assignment (M, K) bool
        z = self._service_assignment(self.uav_pos)

        # 6. Tasks served per UAV
        served_load = np.where(z, self.queue[None, :], 0.0)
        N = served_load.sum(axis=1).astype(np.float32)

        # 7. Energy consumption — paper C7 is a soft constraint over the whole episode.
        # UAVs remain active throughout; exceeding the budget is penalized via reward.
        E_total = self._energy_consumption(d, f, N)
        prev_remaining = self.energy_remaining.copy()
        self.energy_remaining = self.energy_remaining - E_total
        self.energy_total_used += E_total
        # Excess this slot: how much we overshot E_uav this slot only
        # (prev_remaining is positive but smaller than E_total → overshoot = E_total - prev_remaining)
        slot_excess = np.maximum(E_total - np.maximum(prev_remaining, 0.0), 0.0).astype(np.float32)
        # Track first violation
        new_violators = (~self.energy_violated) & (slot_excess > 0)
        self.violation_slot = np.where(new_violators, self.t, self.violation_slot)
        self.energy_violated = self.energy_violated | (slot_excess > 0)
        self.cumulative_excess += slot_excess
        self.peak_excess = np.maximum(self.peak_excess, slot_excess)
        # Reward penalty signal: this slot is currently over budget?
        excess = np.maximum(-self.energy_remaining, 0.0).astype(np.float32)
        self.active = self.energy_remaining > 0

        # 9. Update cumulative service counts and clear served queues (vectorized)
        served_active = z & (N[:, None] > 0) & (self.queue[None, :] > 0)
        self.Z += served_active.astype(np.float32)
        served_any = served_active.any(axis=0)
        # Bits served per SD this slot (entire queue when served)
        self.served_per_sd += np.where(served_any, self.queue, 0.0).astype(np.float32)
        self.queue = np.where(served_any, 0.0, self.queue)
        self.total_bits += float(N.sum())

        # 10. Fairness indicators
        F_sd = self._fairness_sd()
        F_uav = self._fairness_uav()

        # 11. Rewards (vectorized)
        scaled_N = N / self.reward_scale
        if self.use_fairness_reward:
            rewards = scaled_N * F_sd * F_uav
        else:
            rewards = np.zeros_like(scaled_N)
        # Individual contribution bonus: direct positive signal proportional to own service.
        rewards = rewards + self.individual_bonus * scaled_N
        rewards = rewards - boundary_hit.astype(np.float32) * self.penalty
        rewards = rewards - collision_hit.astype(np.float32) * self.penalty
        # Energy overrun: fixed base penalty + per-unit-excess penalty (proportional).
        over_mask = (excess > 0).astype(np.float32)
        rewards = rewards - over_mask * self.energy_overrun_base
        rewards = rewards - excess * self.energy_overrun_per_excess
        rewards = rewards.astype(np.float32)

        self.t += 1
        done = self.t >= self.T

        info = {
            "N": N.copy(),
            "F_sd": F_sd.copy(),
            "F_uav": float(F_uav),
            "energy_remaining": self.energy_remaining.copy(),
            "energy_total_used": self.energy_total_used.copy(),
            "energy_violated": self.energy_violated.copy(),
            "violation_slot": self.violation_slot.copy(),
            "cumulative_excess": self.cumulative_excess.copy(),
            "peak_excess": self.peak_excess.copy(),
            "slot_excess": slot_excess.copy(),
            "active": self.active.copy(),
            "boundary_hit": boundary_hit.copy(),
            "collision_hit": collision_hit.copy(),
            "total_bits": self.total_bits,
        }
        return self._all_obs(), rewards, done, info

    # ---------------- internals (all vectorized) ----------------
    def _generate_tasks(self) -> None:
        gen = self.rng.uniform(size=self.K) < self.alpha
        sizes = np.clip(
            self.rng.normal(self.task_mean, self.task_std, size=self.K),
            0.0, self.l_queue_max,
        ).astype(np.float32)
        new_load = np.where(gen, sizes, 0.0).astype(np.float32)
        self.generated_per_sd += new_load
        self.queue = np.minimum(self.queue + new_load, self.l_queue_max)

    def _decode_actions(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = np.clip(actions, -1.0, 1.0)
        u = (a + 1.0) * 0.5
        d = u[:, 0] * self.d_max
        theta = u[:, 1] * 2.0 * np.pi
        f = self.f_min + u[:, 2] * (self.f_max - self.f_min)
        return d, theta, f

    def _move_uavs(self, d: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Vectorized movement: (M, 2) arrays
        delta = np.stack([d * np.cos(theta), d * np.sin(theta)], axis=1).astype(np.float32)
        tentative = self.uav_pos + delta
        clipped = np.clip(tentative, 0.0, self.l_max)
        # Boundary hit if clipping changed any coordinate
        boundary_hit = np.any(np.abs(clipped - tentative) > 1e-6, axis=1)
        return clipped.astype(np.float32), boundary_hit

    def _collision_mask(self, positions: np.ndarray) -> np.ndarray:
        # Pairwise distance matrix (M, M)
        diff = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        return np.any(dists < self.r_min, axis=1)

    def _service_assignment(self, positions: np.ndarray) -> np.ndarray:
        # (M, K) distance matrix
        diff = positions[:, None, :] - self.sd_pos[None, :, :]
        dist = np.linalg.norm(diff, axis=2)

        in_range = (dist <= self.r_max)
        has_task = self.queue > 0
        in_range &= has_task[None, :]

        # For each SD, pick the closest in-range UAV
        masked_dist = np.where(in_range, dist, np.inf)
        best_uav = np.argmin(masked_dist, axis=0)         # (K,)
        sd_served = np.min(masked_dist, axis=0) < np.inf  # (K,)

        z = np.zeros((self.M, self.K), dtype=bool)
        served_idx = np.where(sd_served)[0]
        z[best_uav[served_idx], served_idx] = True
        return z

    def _energy_consumption(self, d: np.ndarray, f: np.ndarray, N: np.ndarray) -> np.ndarray:
        E_tran = self.p_tran * N / self.trans_rate
        E_com = self.k1 * np.power(f, self.k2 - 1) * N * self.cyc
        T_tran = N / self.trans_rate
        T_com = N * self.cyc / np.maximum(f, 1e-9)
        T_move = d / self.v_uav
        # Apply movement energy multiplier ONLY to the movement portion of E_oper.
        # In the paper, movement is dwarfed by computation (ratio ~10^-9); the multiplier
        # restores meaningful trade-off between trajectory and frequency choices.
        T_oper_eff = T_tran + T_com + T_move * self.movement_energy_mult
        E_oper = 0.5 * self.w_uav * (self.v_uav ** 2) * T_oper_eff
        return ((E_tran + E_com + E_oper) / self.energy_scale).astype(np.float32)

    def _fairness_sd(self) -> np.ndarray:
        s1 = self.Z.sum(axis=1)
        s2 = (self.Z ** 2).sum(axis=1)
        denom = self.K * s2 + 1e-9
        F = np.where(s2 > 0, (s1 ** 2) / denom, 0.0)
        return F.astype(np.float32)

    def _fairness_uav(self) -> float:
        per_uav = self.Z.sum(axis=1)
        s1 = per_uav.sum()
        s2 = (per_uav ** 2).sum()
        if s2 <= 0:
            return 0.0
        return float((s1 ** 2) / (self.M * s2 + 1e-9))

    def _all_obs(self) -> list[np.ndarray]:
        per_uav_total = self.Z.sum(axis=1)
        # Pairwise UAV distance matrix (M, M); pull off-diagonals into (M, M-1)
        diff = self.uav_pos[:, None, :] - self.uav_pos[None, :, :]
        all_dists = np.linalg.norm(diff, axis=2).astype(np.float32)
        other_dists = all_dists[self._other_mask].reshape(self.M, self.M - 1)

        denom_t = max(self.t, 1)
        coords_n = self.uav_pos / self.l_max  # (M, 2)
        Z_n = self.Z / denom_t                # (M, K)
        share = np.broadcast_to(per_uav_total / denom_t / self.K, (self.M, self.M))  # (M, M)
        other_n = other_dists / self.l_max    # (M, M-1)

        obs_block = np.concatenate([coords_n, Z_n, share, other_n, self._uav_ids], axis=1).astype(np.float32)
        return [obs_block[m] for m in range(self.M)]

    # ---------------- evaluation summary ----------------
    def summary(self) -> dict:
        return {
            "computation_bits": self.total_bits,
            "F_sd_final": self._fairness_sd().tolist(),
            "F_uav_final": self._fairness_uav(),
            "energy_remaining": self.energy_remaining.tolist(),
            "energy_total_used": self.energy_total_used.tolist(),
            "energy_violated": self.energy_violated.tolist(),
            "n_violators": int(self.energy_violated.sum()),
            "violation_slot": self.violation_slot.tolist(),
            "cumulative_excess": self.cumulative_excess.tolist(),
            "peak_excess": self.peak_excess.tolist(),
        }
