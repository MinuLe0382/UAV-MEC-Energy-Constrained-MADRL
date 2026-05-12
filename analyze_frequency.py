"""Measure CPU frequency choices made by the agent across episodes.

Action a[2] in [-1, 1] decodes to f in [f_min, f_max].
This script runs N episodes and reports the distribution of chosen f per UAV.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from env import UAVMECEnv
from agents.maddpg import MADDPG
from agents.matd3 import MATD3
from baselines import RandomStrategy, CircleStrategy, GreedyStrategy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--maddpg-ckpt", default=None)
    p.add_argument("--algo", choices=["maddpg", "matd3", "random", "circle", "greedy"],
                   default="maddpg")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--label", default=None)
    p.add_argument("--out-dir", default="results/freq_analysis")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def make_strategy(args, env):
    if args.algo == "maddpg":
        agent = MADDPG(env.M, env.obs_dim, env.act_dim, device=args.device)
        agent.load(args.maddpg_ckpt)
        return None, lambda obs, e: agent.select_actions(obs, noise=False)
    if args.algo == "matd3":
        agent = MATD3(env.M, env.obs_dim, env.act_dim, device=args.device)
        agent.load(args.maddpg_ckpt)
        return None, lambda obs, e: agent.select_actions(obs, noise=False)
    if args.algo == "random":
        s = RandomStrategy(env.M, env.act_dim, seed=args.seed)
        return s, lambda obs, e: s.act(e)
    if args.algo == "circle":
        s = CircleStrategy(env.M, env.act_dim)
        return s, lambda obs, e: s.act(e)
    s = GreedyStrategy(env.M, env.act_dim)
    return s, lambda obs, e: s.act(e)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    label = args.label or args.algo

    env = UAVMECEnv(seed=args.seed)
    strategy_obj, action_fn = make_strategy(args, env)
    f_min, f_max = env.f_min, env.f_max
    M = env.M

    # Collect actions
    all_f = []  # list of (M,) arrays per slot
    for ep in range(args.episodes):
        env_run = UAVMECEnv(seed=args.seed + ep)
        env_run.reset()
        if strategy_obj is not None and hasattr(strategy_obj, "reset"):
            strategy_obj.reset(env_run)
        obs = env_run._all_obs()
        for t in range(env_run.T):
            a = action_fn(obs, env_run)
            # Decode frequency: u = (a + 1) / 2; f = f_min + u × (f_max - f_min)
            u = (np.clip(a, -1, 1) + 1) / 2
            f_chosen = f_min + u[:, 2] * (f_max - f_min)
            all_f.append(f_chosen.copy())
            obs, r, done, info = env_run.step(a)
            if done:
                break

    all_f = np.array(all_f)  # (n_slots, M)
    print(f"\n=== {label}  (episodes={args.episodes}, slots={len(all_f)}) ===")
    print(f"f_min={f_min:.0f}, f_max={f_max:.0f}, range={f_max - f_min:.0f}")
    print(f"{'UAV':<5}{'f_mean':<14}{'f_std':<14}{'f_min_seen':<14}{'f_max_seen':<14}{'%of_fmax':<10}")
    print("-" * 70)
    for m in range(M):
        f_vals = all_f[:, m]
        print(f"{m:<5}{f_vals.mean():.3e}    {f_vals.std():.3e}    "
              f"{f_vals.min():.3e}    {f_vals.max():.3e}    "
              f"{f_vals.mean()/f_max*100:.1f}%")
    print("-" * 70)
    print(f"Overall: f_mean={all_f.mean():.3e}  ({all_f.mean()/f_max*100:.1f}% of f_max)")

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    bins = np.linspace(f_min, f_max, 30)
    for m in range(M):
        ax.hist(all_f[:, m], bins=bins, alpha=0.5, color=colors[m % len(colors)],
                label=f"UAV {m} (mean={all_f[:, m].mean():.2e})",
                edgecolor="k", linewidth=0.5)
    ax.axvline(f_max, color="red", linestyle="--", linewidth=2, label=f"f_max={f_max:.0f}")
    ax.axvline(f_min, color="orange", linestyle="--", linewidth=2, label=f"f_min={f_min:.0f}")
    ax.set_xlabel("CPU frequency chosen")
    ax.set_ylabel("Slots")
    ax.set_title(f"Frequency choice distribution — {label}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(args.out_dir, f"freq_hist_{label}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[analysis] saved -> {out}")


if __name__ == "__main__":
    main()
