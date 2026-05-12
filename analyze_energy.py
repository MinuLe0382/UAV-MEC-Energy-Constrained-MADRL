"""Analyze per-UAV energy usage and constraint violations across many episodes.

Runs N evaluation episodes (deterministic policy, varying env seeds) and reports:
  - Per-UAV energy used (mean / std / max / min)
  - % of episodes with at least one violator
  - Avg / max cumulative excess per UAV
  - Energy budget utilization histogram
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
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--out-dir", default="results/energy_analysis")
    p.add_argument("--device", default="cpu")
    p.add_argument("--label", default=None)
    return p.parse_args()


def make_action_fn(args, env):
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


def run_episodes(args) -> dict:
    """Run N episodes; return per-episode arrays of metrics."""
    env = UAVMECEnv(seed=args.seed)
    strategy_obj, action_fn = make_action_fn(args, env)

    M = env.M
    n_eps = args.episodes
    energy_used = np.zeros((n_eps, M))     # (ep, M)
    cum_excess = np.zeros((n_eps, M))
    peak_excess = np.zeros((n_eps, M))
    violators = np.zeros((n_eps, M), dtype=bool)
    violation_slots = -np.ones((n_eps, M), dtype=int)
    bits = np.zeros(n_eps)
    f_uav = np.zeros(n_eps)

    for ep in range(n_eps):
        env_run = UAVMECEnv(seed=args.seed + ep)
        env_run.reset()
        if strategy_obj is not None and hasattr(strategy_obj, "reset"):
            strategy_obj.reset(env_run)

        obs = env_run._all_obs()
        for t in range(env_run.T):
            a = action_fn(obs, env_run)
            obs, r, done, info = env_run.step(a)
            if done:
                break

        s = env_run.summary()
        energy_used[ep] = s["energy_total_used"]
        cum_excess[ep] = s["cumulative_excess"]
        peak_excess[ep] = s["peak_excess"]
        violators[ep] = s["energy_violated"]
        violation_slots[ep] = s["violation_slot"]
        bits[ep] = s["computation_bits"]
        f_uav[ep] = s["F_uav_final"]

    return {
        "energy_used": energy_used,
        "cum_excess": cum_excess,
        "peak_excess": peak_excess,
        "violators": violators,
        "violation_slots": violation_slots,
        "bits": bits,
        "F_uav": f_uav,
        "M": M,
        "E_uav": env.e_uav,
    }


def print_summary(res: dict, label: str) -> str:
    M = res["M"]
    E_uav = res["E_uav"]
    n_eps = len(res["bits"])
    used = res["energy_used"]
    excess = res["cum_excess"]
    viol = res["violators"]

    lines = []
    lines.append(f"=== {label}  (episodes={n_eps}, E_uav={E_uav:.0f}) ===")
    lines.append(f"{'UAV':<5}{'energy_used (mean±std)':<26}{'max':<10}"
                 f"{'util%':<8}{'viol%':<8}{'cum_excess_max':<14}")
    lines.append("-" * 76)
    for m in range(M):
        u_mean = used[:, m].mean()
        u_std = used[:, m].std()
        u_max = used[:, m].max()
        util = u_mean / E_uav * 100
        v_rate = viol[:, m].mean() * 100
        ex_max = excess[:, m].max()
        lines.append(
            f"{m:<5}{u_mean:.1f} ± {u_std:.1f}".ljust(31)
            + f"{u_max:.1f}".ljust(10)
            + f"{util:.1f}%".ljust(8)
            + f"{v_rate:.1f}%".ljust(8)
            + f"{ex_max:.2f}"
        )
    lines.append("-" * 76)
    any_violation = viol.any(axis=1)
    lines.append(f"Episodes with ANY violation: {any_violation.sum()}/{n_eps} ({any_violation.mean()*100:.1f}%)")
    lines.append(f"bits  mean ± std: {res['bits'].mean():.3e} ± {res['bits'].std():.2e}")
    lines.append(f"F_uav mean ± std: {res['F_uav'].mean():.3f} ± {res['F_uav'].std():.3f}")
    return "\n".join(lines)


def plot_histogram(res: dict, label: str, out_path: str) -> None:
    M = res["M"]
    E_uav = res["E_uav"]
    used = res["energy_used"]   # (ep, M)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    bins = np.linspace(0, max(used.max(), E_uav * 1.1), 40)
    for m in range(M):
        ax.hist(used[:, m], bins=bins, alpha=0.5, color=colors[m % len(colors)],
                label=f"UAV {m}", edgecolor="k", linewidth=0.5)
    ax.axvline(E_uav, color="red", linestyle="--", linewidth=2,
               label=f"E_uav budget = {E_uav:.0f}")
    ax.set_xlabel("Total energy used per episode")
    ax.set_ylabel("Episodes")
    ax.set_title(f"Energy usage distribution — {label}")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[analysis] saved -> {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    label = args.label or args.algo

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    res = run_episodes(args)
    summary_str = print_summary(res, label)
    print(summary_str)

    with open(os.path.join(args.out_dir, f"energy_summary_{label}.txt"), "w") as f:
        f.write(summary_str + "\n")
    plot_histogram(res, label,
                   os.path.join(args.out_dir, f"energy_hist_{label}.png"))


if __name__ == "__main__":
    main()
