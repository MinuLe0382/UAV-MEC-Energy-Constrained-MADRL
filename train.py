"""Training entry point for CEJOMU (MADDPG) and the DDPG baseline."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from config import RL, SYS
from env import UAVMECEnv
from agents.maddpg import MADDPG
from agents.ddpg import DDPG
from agents.matd3 import MATD3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["maddpg", "ddpg", "matd3"], default="maddpg")
    p.add_argument("--episodes", type=int, default=int(RL["episode_max"]))
    p.add_argument("--warmup", type=int, default=int(RL["episode_before"]))
    p.add_argument("--alpha-dist", choices=["uniform", "normal"], default="uniform")
    p.add_argument("--alpha-mu", type=float, default=0.5)
    p.add_argument("--alpha-sigma", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-file", default=None,
                   help="Path to JSONL log file (default: logs/<algo>.jsonl)")
    return p.parse_args()


def make_agent(algo: str, env: UAVMECEnv, device: str):
    if algo == "maddpg":
        return MADDPG(env.M, env.obs_dim, env.act_dim, device=device)
    if algo == "matd3":
        return MATD3(env.M, env.obs_dim, env.act_dim, device=device)
    return DDPG(env.M, env.obs_dim, env.act_dim, device=device)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = UAVMECEnv(
        alpha_dist=args.alpha_dist,
        alpha_params=(args.alpha_mu, args.alpha_sigma),
        seed=args.seed,
    )
    agent = make_agent(args.algo, env, args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    log_path = args.log_file or f"logs/{args.algo}.jsonl"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fp = open(log_path, "w")

    print(f"[train] algo={args.algo} device={args.device} episodes={args.episodes}")
    print(f"[train] obs_dim={env.obs_dim} act_dim={env.act_dim} M={env.M} K={env.K}")
    print(f"[train] log -> {log_path}")

    ep_returns = []
    t0 = time.time()

    for ep in range(args.episodes):
        progress = ep / max(args.episodes - 1, 1)
        agent.set_noise(progress)

        obs = env.reset()
        ep_reward = np.zeros(env.M, dtype=np.float32)
        ep_bits = 0.0
        loss_log = {}

        for t in range(env.T):
            actions = agent.select_actions(obs, noise=True)
            next_obs, rewards, done, info = env.step(actions)
            agent.store(obs, actions, rewards, next_obs, done)

            if ep > args.warmup:
                losses = agent.update()
                if losses is not None:
                    loss_log = losses

            ep_reward += rewards
            ep_bits = info["total_bits"]
            obs = next_obs
            if done:
                break

        ep_returns.append(float(ep_reward.sum()))

        # Write per-episode log
        summary = env.summary()
        log_entry = {
            "episode": ep + 1,
            "return": float(ep_reward.sum()),
            "bits": float(ep_bits),
            "F_uav": float(summary["F_uav_final"]),
            "n_violators": int(summary["n_violators"]),
            "energy_used_max": float(max(summary["energy_total_used"])),
            "energy_used_min": float(min(summary["energy_total_used"])),
            "cum_excess_max": float(max(summary["cumulative_excess"])),
        }
        if loss_log:
            log_entry.update({
                "actor_loss": float(loss_log.get("actor_loss", 0.0)),
                "critic_loss": float(loss_log.get("critic_loss", 0.0)),
                "noise_sigma": float(loss_log.get("noise_sigma", 0.0)),
            })
        log_fp.write(json.dumps(log_entry) + "\n")
        log_fp.flush()

        if (ep + 1) % args.log_every == 0:
            recent = np.mean(ep_returns[-args.log_every:])
            elapsed = time.time() - t0
            summary = env.summary()
            extra = ""
            if loss_log:
                extra = (
                    f" | actor_loss={loss_log.get('actor_loss', 0):+.3f}"
                    f" critic_loss={loss_log.get('critic_loss', 0):.3f}"
                    f" sigma={loss_log.get('noise_sigma', 0):.3f}"
                )
            print(
                f"[ep {ep+1:5d}] return={recent:+.2f} | bits={ep_bits:.2e}"
                f" | F_uav={summary['F_uav_final']:.3f}"
                f" | t={elapsed:.0f}s{extra}"
            )

        if (ep + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"{args.algo}_ep{ep+1}.pt")
            agent.save(ckpt_path)
            print(f"[ckpt] saved -> {ckpt_path}")

    final_path = os.path.join(args.ckpt_dir, f"{args.algo}_final.pt")
    agent.save(final_path)
    log_fp.close()
    print(f"[done] final checkpoint -> {final_path}")
    print(f"[done] log -> {log_path}")


if __name__ == "__main__":
    main()
