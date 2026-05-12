"""Analyze task processing: ratio of served vs generated, plus per-SD heatmap.

Compares MADDPG / MATD3 / baselines on the same env seed.
Outputs:
  - results/<tag>/task_heatmap.png : 4-panel heatmap (one per strategy)
  - results/<tag>/task_summary.txt : numerical processing ratios
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch

from env import UAVMECEnv
from agents.maddpg import MADDPG
from agents.ddpg import DDPG
from agents.matd3 import MATD3
from baselines import RandomStrategy, CircleStrategy, GreedyStrategy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--maddpg-ckpt", default=None,
                   help="Path to MADDPG / MATD3 checkpoint")
    p.add_argument("--algo", choices=["maddpg", "matd3"], default="maddpg")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/task_analysis")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def run_strategy(env: UAVMECEnv, name: str, strategy_obj, action_fn) -> dict:
    """Run one episode, return per-SD generated/served + UAV trajectory."""
    env.reset()
    if strategy_obj is not None and hasattr(strategy_obj, "reset"):
        strategy_obj.reset(env)

    uav_trace = [env.uav_pos.copy()]
    obs = env._all_obs()
    for t in range(env.T):
        a = action_fn(obs, env)
        obs, r, done, info = env.step(a)
        uav_trace.append(env.uav_pos.copy())
        if done:
            break

    generated = env.generated_per_sd.copy()
    served = env.served_per_sd.copy()
    leftover = env.queue.copy()
    ratio = float(served.sum() / max(generated.sum(), 1e-9))
    return {
        "name": name,
        "generated": generated,
        "served": served,
        "leftover": leftover,
        "ratio": ratio,
        "sd_pos": env.sd_pos.copy(),
        "alpha": env.alpha.copy(),
        "uav_trace": np.array(uav_trace),
        "total_bits": env.total_bits,
        "F_uav": env._fairness_uav(),
        "F_sd": float(np.mean(env._fairness_sd())),
    }


def build_strategies(env: UAVMECEnv, args) -> dict:
    """Return {name: (strategy_obj_or_None, action_fn)}."""
    strats = {}
    rs = RandomStrategy(env.M, env.act_dim, seed=args.seed)
    strats["RANDOM"] = (rs, lambda obs, e: rs.act(e))
    cs = CircleStrategy(env.M, env.act_dim)
    strats["CIRCLE"] = (cs, lambda obs, e: cs.act(e))
    gs = GreedyStrategy(env.M, env.act_dim)
    strats["GREEDY"] = (gs, lambda obs, e: gs.act(e))

    if args.maddpg_ckpt and os.path.exists(args.maddpg_ckpt):
        if args.algo == "matd3":
            agent = MATD3(env.M, env.obs_dim, env.act_dim, device=args.device)
        else:
            agent = MADDPG(env.M, env.obs_dim, env.act_dim, device=args.device)
        agent.load(args.maddpg_ckpt)
        label = args.algo.upper()
        strats[label] = (None, lambda obs, e: agent.select_actions(obs, noise=False))

    return strats


def plot_heatmaps(results: list, out_path: str) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5), squeeze=False)
    axes = axes[0]
    # Common color scale: leftover bits
    max_leftover = max(r["leftover"].max() for r in results) or 1.0

    for ax, res in zip(axes, results):
        sd_pos = res["sd_pos"]
        leftover = res["leftover"]
        sc = ax.scatter(sd_pos[:, 0], sd_pos[:, 1],
                        c=leftover, cmap="Reds", s=140,
                        edgecolor="k", linewidths=0.6,
                        vmin=0, vmax=max_leftover)
        # UAV trajectories
        trace = res["uav_trace"]   # (T+1, M, 2)
        colors = ["#1f77b4", "#2ca02c", "#9467bd"]
        for m in range(trace.shape[1]):
            ax.plot(trace[:, m, 0], trace[:, m, 1], "-",
                    color=colors[m % len(colors)], alpha=0.5, linewidth=1.5)
            ax.scatter(trace[0, m, 0], trace[0, m, 1], color=colors[m % len(colors)],
                       marker="o", s=60, edgecolor="k", zorder=5)
            ax.scatter(trace[-1, m, 0], trace[-1, m, 1], color=colors[m % len(colors)],
                       marker="*", s=120, edgecolor="k", zorder=5)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.set_title(
            f"{res['name']} — ratio={res['ratio']*100:.1f}%\n"
            f"bits={res['total_bits']:.2e}  F_uav={res['F_uav']:.3f}  F_sd={res['F_sd']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="leftover bits")

    fig.suptitle("Per-SD leftover task volume at episode end (red = unprocessed)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[analysis] saved -> {out_path}")
    plt.close(fig)


def plot_processing_bars(results: list, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r["name"] for r in results]
    served = [r["served"].sum() / 1e8 for r in results]
    leftover = [r["leftover"].sum() / 1e8 for r in results]
    x = np.arange(len(names))
    ax.bar(x, served, label="processed", color="#2ca02c")
    ax.bar(x, leftover, bottom=served, label="unprocessed", color="#d62728")
    for i, r in enumerate(results):
        ax.text(i, served[i] + leftover[i] + 0.2,
                f"{r['ratio']*100:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("bits (×1e8)")
    ax.set_title("Generated tasks: processed vs unprocessed (one episode)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[analysis] saved -> {out_path}")
    plt.close(fig)


def write_summary(results: list, out_path: str) -> None:
    lines = []
    lines.append(f"{'Strategy':<10}{'Generated':<16}{'Processed':<16}"
                 f"{'Leftover':<16}{'Ratio':<10}{'F_sd':<10}{'F_uav':<10}")
    lines.append("-" * 88)
    for r in results:
        lines.append(
            f"{r['name']:<10}"
            f"{r['generated'].sum():<16.3e}"
            f"{r['served'].sum():<16.3e}"
            f"{r['leftover'].sum():<16.3e}"
            f"{r['ratio']*100:<10.2f}"
            f"{r['F_sd']:<10.3f}"
            f"{r['F_uav']:<10.3f}"
        )
    out = "\n".join(lines)
    print(out)
    with open(out_path, "w") as f:
        f.write(out)
    print(f"[analysis] saved -> {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    env_seed = 100  # fixed eval seed

    # Build strategies once (loads agent)
    env = UAVMECEnv(seed=env_seed)
    strategies = build_strategies(env, args)

    results = []
    for name, (obj, fn) in strategies.items():
        env_run = UAVMECEnv(seed=env_seed)  # identical SD/alpha across runs
        res = run_strategy(env_run, name, obj, fn)
        results.append(res)

    write_summary(results, os.path.join(args.out_dir, "task_summary.txt"))
    plot_heatmaps(results, os.path.join(args.out_dir, "task_heatmap.png"))
    plot_processing_bars(results, os.path.join(args.out_dir, "task_bars.png"))


if __name__ == "__main__":
    main()
