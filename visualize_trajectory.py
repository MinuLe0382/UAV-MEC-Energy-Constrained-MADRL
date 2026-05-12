"""Visualize UAV trajectories for a trained MADDPG / DDPG agent or baselines.

Outputs:
  - PNG: static plot showing trajectories, SD heatmap, coverage circles, energy decay
  - GIF (optional, --animate): step-by-step animation
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import torch

from env import UAVMECEnv
from agents.maddpg import MADDPG
from agents.ddpg import DDPG
from agents.matd3 import MATD3
from baselines import RandomStrategy, CircleStrategy, GreedyStrategy


COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["maddpg", "ddpg", "matd3", "random", "circle", "greedy"],
                   default="maddpg")
    p.add_argument("--ckpt", default="checkpoints/maddpg_final.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-png", default="results/trajectory.png")
    p.add_argument("--out-gif", default="results/trajectory.gif")
    p.add_argument("--animate", action="store_true")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_strategy(args, env: UAVMECEnv):
    if args.algo == "maddpg":
        agent = MADDPG(env.M, env.obs_dim, env.act_dim, device=args.device)
        agent.load(args.ckpt)
        return ("MADDPG", lambda obs, env: agent.select_actions(obs, noise=False))
    if args.algo == "ddpg":
        agent = DDPG(env.M, env.obs_dim, env.act_dim, device=args.device)
        agent.load(args.ckpt)
        return ("DDPG", lambda obs, env: agent.select_actions(obs, noise=False))
    if args.algo == "matd3":
        agent = MATD3(env.M, env.obs_dim, env.act_dim, device=args.device)
        agent.load(args.ckpt)
        return ("MATD3", lambda obs, env: agent.select_actions(obs, noise=False))
    if args.algo == "random":
        s = RandomStrategy(env.M, env.act_dim, seed=args.seed)
        return ("RANDOM", lambda obs, env: s.act(env))
    if args.algo == "circle":
        s = CircleStrategy(env.M, env.act_dim); s.reset(env)
        return ("CIRCLE", lambda obs, env: s.act(env))
    s = GreedyStrategy(env.M, env.act_dim)
    return ("GREEDY", lambda obs, env: s.act(env))


def rollout(env: UAVMECEnv, action_fn) -> dict:
    """Run one full episode and record per-slot state."""
    traj = {
        "uav_pos": [env.uav_pos.copy()],
        "energy": [env.energy_remaining.copy()],
        "active": [env.active.copy()],
        "freq": [],
        "N": [],
        "F_sd": [],
        "F_uav": [],
        "boundary_hit": [],
        "collision_hit": [],
        "rewards": [],
    }
    obs = env._all_obs()  # current observations
    for t in range(env.T):
        a = action_fn(obs, env)
        # Decode chosen frequency for inspection
        u = (np.clip(a, -1, 1) + 1) / 2
        f_chosen = env.f_min + u[:, 2] * (env.f_max - env.f_min)
        traj["freq"].append(f_chosen.copy())

        obs, r, done, info = env.step(a)
        traj["uav_pos"].append(env.uav_pos.copy())
        traj["energy"].append(info["energy_remaining"].copy())
        traj["active"].append(info["active"].copy())
        traj["N"].append(info["N"].copy())
        traj["F_sd"].append(info["F_sd"].copy())
        traj["F_uav"].append(info["F_uav"])
        traj["boundary_hit"].append(info["boundary_hit"].copy())
        traj["collision_hit"].append(info["collision_hit"].copy())
        traj["rewards"].append(r.copy())
        if done:
            break

    for k in ["uav_pos", "energy", "active", "freq", "N", "F_sd",
              "boundary_hit", "collision_hit", "rewards"]:
        traj[k] = np.array(traj[k])
    traj["F_uav"] = np.array(traj["F_uav"])
    return traj


def plot_static(env: UAVMECEnv, traj: dict, label: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    ax_map, ax_metrics = axes

    # Heatmap of SD task occurrence rates (alpha)
    sc = ax_map.scatter(env.sd_pos[:, 0], env.sd_pos[:, 1],
                        c=env.alpha, cmap="YlOrRd", s=60, edgecolor="k",
                        linewidths=0.4, vmin=0, vmax=1, label="SDs (color = α)")
    plt.colorbar(sc, ax=ax_map, fraction=0.046, pad=0.04, label="α (task rate)")

    # UAV trajectories
    for m in range(env.M):
        path = traj["uav_pos"][:, m, :]
        ax_map.plot(path[:, 0], path[:, 1], "-", color=COLORS[m], linewidth=2,
                    alpha=0.7, label=f"UAV {m}")
        ax_map.scatter(path[0, 0], path[0, 1], color=COLORS[m], marker="o",
                       s=80, edgecolor="k", zorder=5)
        ax_map.scatter(path[-1, 0], path[-1, 1], color=COLORS[m], marker="*",
                       s=180, edgecolor="k", zorder=5)
        # Final coverage circle
        circ = Circle(path[-1], env.r_max, fill=False, edgecolor=COLORS[m],
                      linewidth=1, linestyle="--", alpha=0.5)
        ax_map.add_patch(circ)

    ax_map.set_xlim(0, env.l_max)
    ax_map.set_ylim(0, env.l_max)
    ax_map.set_aspect("equal")
    ax_map.set_title(f"UAV Trajectories — {label}\n(○ start, ★ end, dashed = R_max)")
    ax_map.set_xlabel("x (m)"); ax_map.set_ylabel("y (m)")
    ax_map.legend(loc="upper right", fontsize=8)
    ax_map.grid(alpha=0.3)

    # Right panel: metrics over time
    T = traj["energy"].shape[0]
    slots = np.arange(T)
    for m in range(env.M):
        ax_metrics.plot(slots, traj["energy"][:, m], color=COLORS[m],
                        label=f"UAV {m} energy")
    ax_metrics.set_xlabel("time slot")
    ax_metrics.set_ylabel("energy remaining", color="tab:blue")
    ax_metrics.legend(loc="upper left", fontsize=8)
    ax_metrics.grid(alpha=0.3)

    # Twin axis: F_uav
    ax2 = ax_metrics.twinx()
    ax2.plot(np.arange(1, T), traj["F_uav"], color="black", linestyle="--",
             linewidth=1.5, label="F_uav")
    ax2.set_ylabel("F_uav (load balancing)", color="black")
    ax2.set_ylim(0, 1.05)

    total_bits = float(np.sum(traj["N"]))
    bound_total = int(np.sum(traj["boundary_hit"]))
    coll_total = int(np.sum(traj["collision_hit"]))
    rew_total = float(np.sum(traj["rewards"]))
    fig.suptitle(
        f"{label} — bits={total_bits:.2e}  F_uav_T={traj['F_uav'][-1]:.3f}  "
        f"reward={rew_total:+.1f}  | boundary hits={bound_total}, collisions={coll_total}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[viz] saved -> {out_path}")
    plt.close(fig)


def animate(env: UAVMECEnv, traj: dict, label: str, out_path: str, fps: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    sc_sd = ax.scatter(env.sd_pos[:, 0], env.sd_pos[:, 1],
                       c=env.alpha, cmap="YlOrRd", s=60, edgecolor="k",
                       linewidths=0.4, vmin=0, vmax=1)
    plt.colorbar(sc_sd, ax=ax, fraction=0.046, pad=0.04, label="α")

    lines, points, circles = [], [], []
    for m in range(env.M):
        ln, = ax.plot([], [], "-", color=COLORS[m], linewidth=2, alpha=0.7,
                      label=f"UAV {m}")
        pt, = ax.plot([], [], marker="o", color=COLORS[m],
                      markersize=10, markeredgecolor="k")
        circ = Circle((0, 0), env.r_max, fill=False, edgecolor=COLORS[m],
                      linewidth=1, linestyle="--", alpha=0.5)
        ax.add_patch(circ)
        lines.append(ln); points.append(pt); circles.append(circ)

    ax.set_xlim(0, env.l_max); ax.set_ylim(0, env.l_max)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    title = ax.set_title("")

    T = traj["uav_pos"].shape[0]

    def update(frame: int):
        for m in range(env.M):
            xs = traj["uav_pos"][: frame + 1, m, 0]
            ys = traj["uav_pos"][: frame + 1, m, 1]
            lines[m].set_data(xs, ys)
            points[m].set_data([xs[-1]], [ys[-1]])
            circles[m].center = (xs[-1], ys[-1])
        if frame > 0:
            title.set_text(
                f"{label}  slot {frame}/{T-1}  "
                f"F_uav={traj['F_uav'][min(frame, len(traj['F_uav'])-1)]:.3f}"
            )
        else:
            title.set_text(f"{label}  slot 0/{T-1}")
        return lines + points + circles + [title]

    anim = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    print(f"[viz] animation -> {out_path}")
    plt.close(fig)


def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    env = UAVMECEnv(seed=args.seed)
    label, action_fn = make_strategy(args, env)
    if args.algo not in ("maddpg", "ddpg"):
        # Reset baseline state if needed (CIRCLE precomputes)
        env.reset()
        if args.algo == "circle":
            CircleStrategy(env.M, env.act_dim).reset(env)
    else:
        env.reset()

    traj = rollout(env, action_fn)
    plot_static(env, traj, label, args.out_png)
    if args.animate:
        animate(env, traj, label, args.out_gif, args.fps)


if __name__ == "__main__":
    main()
