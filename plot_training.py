"""Plot training curves from train.py JSONL logs.

Supports:
  - JSONL log file produced by train.py (preferred)
  - Plain text terminal log (fallback parser)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_jsonl(path: Path) -> dict:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}


def parse_text_log(path: Path) -> dict:
    """Parse the terminal output of train.py.

    Looks for lines like:
      [ep   250] return=-378.20 | bits=2.25e+08 | F_uav=0.970 | t=38s | actor_loss=...
    """
    pattern = re.compile(
        r"\[ep\s+(\d+)\]\s+return=([+-]?[\d.]+)\s+\|\s+bits=([\d.eE+-]+)"
        r"\s+\|\s+F_uav=([\d.]+)"
        r"(?:.*?actor_loss=([+-]?[\d.]+)\s+critic_loss=([\d.]+)\s+sigma=([\d.]+))?"
    )
    eps, rets, bits_, fuavs, als, cls, sigs = [], [], [], [], [], [], []
    for line in path.open():
        m = pattern.search(line)
        if not m:
            continue
        eps.append(int(m.group(1)))
        rets.append(float(m.group(2)))
        bits_.append(float(m.group(3)))
        fuavs.append(float(m.group(4)))
        als.append(float(m.group(5)) if m.group(5) else np.nan)
        cls.append(float(m.group(6)) if m.group(6) else np.nan)
        sigs.append(float(m.group(7)) if m.group(7) else np.nan)
    return {
        "episode": np.array(eps),
        "return": np.array(rets),
        "bits": np.array(bits_),
        "F_uav": np.array(fuavs),
        "actor_loss": np.array(als),
        "critic_loss": np.array(cls),
        "noise_sigma": np.array(sigs),
    }


def smooth(y: np.ndarray, window: int = 50) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+",
                   help="Paths to JSONL or text logs (multiple allowed for comparison)")
    p.add_argument("--labels", nargs="*", default=None,
                   help="One label per log; defaults to file stem")
    p.add_argument("--smooth", type=int, default=50,
                   help="Moving average window (1 = off)")
    p.add_argument("--out", default="results/training_curves.png")
    return p.parse_args()


def plot(rows_by_label: dict, smooth_w: int, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_r, ax_b, ax_f, ax_l = axes.flatten()

    for label, d in rows_by_label.items():
        ep = d["episode"]
        ax_r.plot(ep, d["return"], alpha=0.25)
        if smooth_w > 1 and len(ep) >= smooth_w:
            sm = smooth(d["return"], smooth_w)
            ep_s = ep[smooth_w - 1:]
            ax_r.plot(ep_s, sm, label=f"{label} (smoothed)", linewidth=2)

        ax_b.plot(ep, d["bits"], alpha=0.25)
        if smooth_w > 1 and len(ep) >= smooth_w:
            ax_b.plot(ep[smooth_w - 1:], smooth(d["bits"], smooth_w),
                      label=label, linewidth=2)

        ax_f.plot(ep, d["F_uav"], alpha=0.25)
        if smooth_w > 1 and len(ep) >= smooth_w:
            ax_f.plot(ep[smooth_w - 1:], smooth(d["F_uav"], smooth_w),
                      label=label, linewidth=2)

        if "critic_loss" in d and np.any(~np.isnan(d["critic_loss"])):
            valid = ~np.isnan(d["critic_loss"])
            ax_l.plot(ep[valid], d["critic_loss"][valid], alpha=0.25)
            if smooth_w > 1 and valid.sum() >= smooth_w:
                cl = d["critic_loss"][valid]
                ax_l.plot(ep[valid][smooth_w - 1:], smooth(cl, smooth_w),
                          label=f"{label} critic", linewidth=2)

    ax_r.set_title("Episode Return"); ax_r.set_xlabel("episode"); ax_r.set_ylabel("return")
    ax_r.grid(alpha=0.3); ax_r.legend(fontsize=8)
    ax_b.set_title("Computation Bits"); ax_b.set_xlabel("episode"); ax_b.set_ylabel("bits")
    ax_b.grid(alpha=0.3); ax_b.legend(fontsize=8)
    ax_f.set_title("UAV Load Balancing F_uav"); ax_f.set_xlabel("episode")
    ax_f.set_ylabel("F_uav"); ax_f.set_ylim(0, 1.05); ax_f.grid(alpha=0.3); ax_f.legend(fontsize=8)
    ax_l.set_title("Critic Loss"); ax_l.set_xlabel("episode"); ax_l.set_ylabel("loss")
    ax_l.set_yscale("log"); ax_l.grid(alpha=0.3); ax_l.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[plot] saved -> {out_path}")


def main():
    args = parse_args()
    rows_by_label = {}
    labels = args.labels or [Path(p).stem for p in args.logs]
    if len(labels) != len(args.logs):
        labels = [Path(p).stem for p in args.logs]

    for path_str, label in zip(args.logs, labels):
        path = Path(path_str)
        if path.suffix == ".jsonl":
            rows = parse_jsonl(path)
        else:
            rows = parse_text_log(path)
        if not rows:
            print(f"[warn] no data parsed from {path}")
            continue
        rows_by_label[label] = rows
        print(f"[load] {label}: {len(rows['episode'])} episodes from {path}")

    if not rows_by_label:
        raise SystemExit("No data to plot.")

    plot(rows_by_label, args.smooth, args.out)


if __name__ == "__main__":
    main()
