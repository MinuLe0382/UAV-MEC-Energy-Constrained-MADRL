"""Plot mean ± std training curves across multiple seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> dict:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    keys = rows[0].keys()
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="valid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    p.add_argument("--smooth", type=int, default=200)
    p.add_argument("--label", default="v3")
    p.add_argument("--out", default="results/training_multiseed.png")
    args = p.parse_args()

    runs = [load_jsonl(Path(p)) for p in args.logs]
    # Align episode lengths to the minimum
    n = min(len(r["episode"]) for r in runs)
    eps = runs[0]["episode"][:n]

    def stack_metric(key: str) -> np.ndarray:
        return np.stack([r[key][:n] for r in runs])  # (S, N)

    metrics = {
        "Episode Return": stack_metric("return"),
        "Computation Bits": stack_metric("bits"),
        "UAV Load Balancing F_uav": stack_metric("F_uav"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, M) in zip(axes, metrics.items()):
        mean = M.mean(axis=0)
        std = M.std(axis=0)
        if args.smooth > 1 and n >= args.smooth:
            mean_s = smooth(mean, args.smooth)
            std_s = smooth(std, args.smooth)
            x = eps[args.smooth - 1:]
            ax.plot(x, mean_s, label=f"{args.label} mean ({len(runs)} seeds)", linewidth=2)
            ax.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.3, label="±1 std")
        else:
            ax.plot(eps, mean, label=f"{args.label} mean", linewidth=2)
            ax.fill_between(eps, mean - std, mean + std, alpha=0.3, label="±1 std")
        ax.set_title(title)
        ax.set_xlabel("episode")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        if "F_uav" in title:
            ax.set_ylim(0, 1.05)

    fig.suptitle(f"{args.label}: mean ± std across {len(runs)} seeds (smooth={args.smooth})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[plot] saved -> {args.out}")


if __name__ == "__main__":
    main()
