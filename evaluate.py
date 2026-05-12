"""Evaluate strategies and reproduce Tables IV / V style summaries."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from env import UAVMECEnv
from agents.maddpg import MADDPG
from agents.ddpg import DDPG
from baselines import CircleStrategy, GreedyStrategy, RandomStrategy


SCENARIOS = {
    "g1_normal_0.2_0.1": ("normal", (0.2, 0.1)),
    "g2_normal_0.2_0.3": ("normal", (0.2, 0.3)),
    "g3_normal_0.8_0.1": ("normal", (0.8, 0.1)),
    "g4_normal_0.8_0.3": ("normal", (0.8, 0.3)),
    "g5_uniform_0_1":    ("uniform", (0.0, 1.0)),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--maddpg-ckpt", default="checkpoints/maddpg_final.pt")
    p.add_argument("--ddpg-ckpt", default="checkpoints/ddpg_final.pt")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--scenario", choices=list(SCENARIOS.keys()) + ["all"], default="all")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run_episode(env: UAVMECEnv, strategy_name: str, agent_obj) -> dict:
    obs = env.reset()
    if hasattr(agent_obj, "reset"):
        agent_obj.reset(env)
    for t in range(env.T):
        if strategy_name == "MADDPG" or strategy_name == "DDPG":
            actions = agent_obj.select_actions(obs, noise=False)
        else:
            actions = agent_obj.act(env)
        obs, rewards, done, info = env.step(actions)
        if done:
            break
    s = env.summary()
    return {
        "bits": s["computation_bits"],
        "F_sd": float(np.mean(s["F_sd_final"])),
        "F_uav": s["F_uav_final"],
    }


def evaluate_scenario(scenario_key: str, args, maddpg_agent, ddpg_agent) -> dict:
    alpha_dist, alpha_params = SCENARIOS[scenario_key]
    results = {name: {"bits": [], "F_sd": [], "F_uav": []} for name in
               ["RANDOM", "CIRCLE", "GREEDY", "DDPG", "MADDPG"]}

    for r in range(args.repeats):
        env = UAVMECEnv(alpha_dist=alpha_dist, alpha_params=alpha_params,
                        seed=args.seed + r)
        strategies = {
            "RANDOM":  RandomStrategy(env.M, env.act_dim, seed=args.seed + r),
            "CIRCLE":  CircleStrategy(env.M, env.act_dim),
            "GREEDY":  GreedyStrategy(env.M, env.act_dim),
            "DDPG":    ddpg_agent,
            "MADDPG":  maddpg_agent,
        }
        for name, strat in strategies.items():
            env_run = UAVMECEnv(alpha_dist=alpha_dist, alpha_params=alpha_params,
                                seed=args.seed + r)
            res = run_episode(env_run, name, strat)
            results[name]["bits"].append(res["bits"])
            results[name]["F_sd"].append(res["F_sd"])
            results[name]["F_uav"].append(res["F_uav"])

    return results


def print_table(scenario_key: str, results: dict) -> None:
    print(f"\n=== Scenario: {scenario_key} ===")
    print(f"{'Strategy':<10} {'Bits (mean ± std)':<28} {'F_sd':<18} {'F_uav':<18}")
    print("-" * 76)
    for name in ["RANDOM", "CIRCLE", "GREEDY", "DDPG", "MADDPG"]:
        b = results[name]["bits"]; fs = results[name]["F_sd"]; fu = results[name]["F_uav"]
        print(
            f"{name:<10} {np.mean(b):.3e} ± {np.std(b):.2e}   "
            f"{np.mean(fs):.3f} ± {np.std(fs):.3f}   "
            f"{np.mean(fu):.3f} ± {np.std(fu):.3f}"
        )


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_template = UAVMECEnv(seed=args.seed)
    maddpg_agent = MADDPG(env_template.M, env_template.obs_dim, env_template.act_dim,
                          device=args.device)
    ddpg_agent = DDPG(env_template.M, env_template.obs_dim, env_template.act_dim,
                      device=args.device)
    if os.path.exists(args.maddpg_ckpt):
        maddpg_agent.load(args.maddpg_ckpt)
        print(f"[eval] loaded MADDPG -> {args.maddpg_ckpt}")
    else:
        print(f"[warn] MADDPG checkpoint not found: {args.maddpg_ckpt}")
    if os.path.exists(args.ddpg_ckpt):
        ddpg_agent.load(args.ddpg_ckpt)
        print(f"[eval] loaded DDPG -> {args.ddpg_ckpt}")
    else:
        print(f"[warn] DDPG checkpoint not found: {args.ddpg_ckpt}")

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    for scen in scenarios:
        results = evaluate_scenario(scen, args, maddpg_agent, ddpg_agent)
        print_table(scen, results)


if __name__ == "__main__":
    main()
