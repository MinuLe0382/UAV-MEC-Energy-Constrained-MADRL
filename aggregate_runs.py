"""Aggregate per-seed training logs into mean ± std statistics.

Usage:
    # Single group
    python aggregate_runs.py logs/v5_s*.jsonl --label v5 --last 200

    # Side-by-side comparison of multiple versions
    python aggregate_runs.py --last 200 \\
        --group v3 'logs/v3_s*.jsonl' \\
        --group v4 'logs/v4_s*.jsonl' \\
        --group v5 'logs/v5_s*.jsonl'
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


# Metrics expected in JSONL rows; missing ones are skipped gracefully.
METRICS = [
    "return",
    "bits",
    "F_uav",
    "n_violators",
    "energy_used_max",
    "energy_used_min",
    "cum_excess_max",
    "actor_loss",
    "critic_loss",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def per_seed_stats(rows: list[dict], last_n: int) -> dict:
    """Average each metric over the last `last_n` episodes."""
    last = rows[-last_n:] if len(rows) >= last_n else rows
    stats = {}
    for m in METRICS:
        vals = [r.get(m) for r in last if r.get(m) is not None]
        if vals:
            stats[m] = float(np.mean(vals))
    stats["episodes"] = len(rows)
    return stats


def aggregate_group(label: str, paths: list[Path], last_n: int) -> dict:
    """Aggregate a group of seed log files into per-seed and across-seed stats."""
    per_seed = []
    for p in paths:
        rows = load_jsonl(p)
        if not rows:
            print(f"[warn] {p}: empty, skipping")
            continue
        per_seed.append((p.stem, per_seed_stats(rows, last_n)))

    if not per_seed:
        return {"label": label, "n_seeds": 0, "per_seed": [], "agg": {}}

    # Aggregate across seeds
    agg = {}
    for m in METRICS:
        vals = [s[1][m] for s in per_seed if m in s[1]]
        if vals:
            arr = np.array(vals)
            agg[m] = {"mean": float(arr.mean()), "std": float(arr.std()),
                      "min": float(arr.min()), "max": float(arr.max()),
                      "cv": float(arr.std() / abs(arr.mean())) if arr.mean() != 0 else 0.0}
    return {"label": label, "n_seeds": len(per_seed), "per_seed": per_seed, "agg": agg}


# ---------------- output ----------------

def fmt(v: float, key: str) -> str:
    if "bits" in key or abs(v) >= 1e5:
        return f"{v:.3e}"
    if "F_uav" in key or "violators" in key:
        return f"{v:.3f}"
    return f"{v:+.1f}" if "return" in key or "loss" in key else f"{v:.2f}"


def print_group(group: dict, last_n: int) -> None:
    label = group["label"]
    n = group["n_seeds"]
    print(f"\n=== {label}  (seeds={n}, last {last_n} episodes) ===")
    if n == 0:
        print("  no data")
        return

    available = [m for m in METRICS if m in group["agg"]]
    if not available:
        print("  no metrics")
        return

    # Per-seed table
    header = f"{'seed':<12}" + "".join(f"{m:<15}" for m in available)
    print(header)
    print("-" * len(header))
    for seed_name, stats in group["per_seed"]:
        row = f"{seed_name:<12}"
        for m in available:
            v = stats.get(m, float("nan"))
            row += f"{fmt(v, m):<15}"
        print(row)
    print("-" * len(header))

    # Aggregate row (mean ± std)
    agg_row = f"{'mean ± std':<12}"
    for m in available:
        a = group["agg"][m]
        agg_row += f"{fmt(a['mean'], m)}±{fmt(a['std'], m)}".ljust(15)
    print(agg_row)

    # CV row
    cv_row = f"{'CV':<12}"
    for m in available:
        cv = group["agg"][m]["cv"]
        cv_row += f"{cv*100:.1f}%".ljust(15)
    print(cv_row)


def print_comparison(groups: list[dict], last_n: int) -> None:
    """Side-by-side comparison of multiple groups."""
    available = set()
    for g in groups:
        available.update(g["agg"].keys())
    available = [m for m in METRICS if m in available]

    print(f"\n=== Comparison (last {last_n} episodes, mean ± std across seeds) ===")
    header = f"{'metric':<18}"
    for g in groups:
        header += f"{g['label']} (n={g['n_seeds']})".ljust(22)
    print(header)
    print("-" * len(header))
    for m in available:
        row = f"{m:<18}"
        for g in groups:
            if m in g["agg"]:
                a = g["agg"][m]
                row += f"{fmt(a['mean'], m)}±{fmt(a['std'], m)}".ljust(22)
            else:
                row += "—".ljust(22)
        print(row)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="*", help="JSONL log files for a single group")
    p.add_argument("--label", default="run", help="Label for single-group mode")
    p.add_argument("--last", type=int, default=200, help="Last N episodes to average")
    p.add_argument("--group", nargs=2, action="append", metavar=("LABEL", "PATTERN"),
                   default=[], help="Comparison mode: label + glob pattern. Repeatable.")
    args = p.parse_args()

    if args.group:
        groups = []
        for label, pattern in args.group:
            paths = sorted(Path(p) for p in glob.glob(pattern))
            g = aggregate_group(label, paths, args.last)
            groups.append(g)
            print_group(g, args.last)
        print_comparison(groups, args.last)
    elif args.logs:
        paths = sorted(Path(p) for p in args.logs)
        g = aggregate_group(args.label, paths, args.last)
        print_group(g, args.last)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
