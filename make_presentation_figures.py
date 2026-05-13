"""Generate 4 presentation figures.

Run inside Docker:
    python make_presentation_figures.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

OUT = Path("results/presentation")
OUT.mkdir(parents=True, exist_ok=True)

# ── shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
GRAY   = "#9CA3AF"

# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Version F_uav bar chart
# ═══════════════════════════════════════════════════════════════════════════
def fig1_fuav_versions():
    versions = ["v1\n(Paper exact)", "v2\n(Param. sharing)", "v3\n(Indiv. bonus)", "v5\n(6-seed confirm)"]
    fuav     = [0.33, 0.78, 0.978, 0.977]
    colors   = [RED, ORANGE, GREEN, BLUE]
    annotations = [
        "Lazy Agent\n(only 1 active)",
        "Partial fix\n(fair↑, bits↓)",
        "Lazy Agent\nSolved!",
        "Confirmed\n6 seeds",
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(versions, fuav, color=colors, width=0.5, zorder=3,
                  edgecolor="white", linewidth=1.5)

    # Value labels on bars
    for bar, val, ann in zip(bars, fuav, annotations):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=bar.get_facecolor())
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                ann, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold",
                multialignment="center")

    # Reference line: random (1/3 ≈ 0.33)
    ax.axhline(1/3, color=RED, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(3.45, 1/3 + 0.01, "F_uav=1/3\n(only 1 of 3 active)", fontsize=9,
            color=RED, va="bottom", ha="right")
    ax.axhline(1.0, color=GREEN, linestyle="--", linewidth=1.2, zorder=2, alpha=0.4)
    ax.text(3.45, 1.0 - 0.02, "Perfect balance (1.0)", fontsize=9,
            color=GREEN, va="top", ha="right", alpha=0.7)

    ax.set_ylabel("UAV Load Fairness (F_uav)", labelpad=10)
    ax.set_title("UAV Load Fairness: Improvement Across Versions", pad=15)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    p = OUT / "fig1_fuav_versions.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — 5-scenario baseline comparison bar chart
# ═══════════════════════════════════════════════════════════════════════════
def fig2_baseline_comparison():
    scenarios = ["g1\n(low density\nconcentrated)", "g2\n(low density\ndispersed)",
                 "g3\n(high density\nconcentrated)", "g4\n(high density\ndispersed)",
                 "g5\n(uniform)"]

    data = {
        "RANDOM":  [2.308e8, 2.433e8, 7.225e8, 6.759e8, 4.893e8],
        "CIRCLE":  [3.079e8, 3.633e8, 1.230e9, 1.156e9, 8.112e8],
        "GREEDY":  [3.853e8, 4.527e8, 1.337e9, 1.261e9, 9.136e8],
        "MADDPG":  [3.661e8, 4.179e8, 1.304e9, 1.261e9, 8.767e8],
    }
    colors = {"RANDOM": GRAY, "CIRCLE": ORANGE, "GREEDY": RED, "MADDPG": BLUE}
    hatches = {"RANDOM": "", "CIRCLE": "", "GREEDY": "//", "MADDPG": ""}

    n_sc   = len(scenarios)
    n_strat = len(data)
    x = np.arange(n_sc)
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))

    offsets = np.linspace(-(n_strat - 1) / 2, (n_strat - 1) / 2, n_strat) * width
    for (label, vals), off in zip(data.items(), offsets):
        bars = ax.bar(x + off, np.array(vals) / 1e8, width=width,
                      label=label, color=colors[label],
                      hatch=hatches[label], edgecolor="white",
                      linewidth=0.8, zorder=3)

    # Annotate MADDPG/GREEDY ratio above MADDPG bars
    maddpg_vals = data["MADDPG"]
    greedy_vals = data["GREEDY"]
    maddpg_off  = offsets[list(data.keys()).index("MADDPG")]
    for i, (m, g) in enumerate(zip(maddpg_vals, greedy_vals)):
        ratio = m / g * 100
        color = GREEN if ratio >= 100 else BLUE
        ax.text(x[i] + maddpg_off, m / 1e8 + 0.15,
                f"{ratio:.0f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=color)

    # GREEDY annotation
    ax.annotate("GREEDY =\nOracle baseline\n(unfair upper bound)",
                xy=(x[0] + offsets[list(data.keys()).index("GREEDY")],
                    data["GREEDY"][0] / 1e8),
                xytext=(0.5, 6.5),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED, alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11)
    ax.set_ylabel("Computation bits (×10⁸)", labelpad=10)
    ax.set_title("Throughput Comparison Across 5 Scenarios  (% = MADDPG vs GREEDY)", pad=15)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, 17)

    fig.tight_layout()
    p = OUT / "fig2_baseline_comparison.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — probe_f_effect: v5 (flat) vs v6 (steep) side by side
# ═══════════════════════════════════════════════════════════════════════════
def fig3_f_effect():
    labels = ["f_min\n(0%)", "f_mid\n(50%)", "f_max\n(100%)"]
    x = np.arange(len(labels))
    width = 0.35

    # v5: bits same regardless of f
    v5_bits  = np.array([5.216e8, 5.216e8, 5.216e8])
    v5_energy = np.array([11.25, 269.47, 527.68])

    # v6: bits scale with f
    v6_bits  = np.array([1.165e6, 5.819e7, 1.130e8])
    v6_energy = np.array([6.05, 35.42, 118.99])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # ── Left: bits ──────────────────────────────────────────────────────────
    ax = axes[0]
    b1 = ax.bar(x - width/2, v5_bits / 1e8, width, label="v5 env (original)", color=GRAY,
                edgecolor="white", zorder=3)
    b2 = ax.bar(x + width/2, v6_bits / 1e8, width, label="v6 env (redesigned)", color=BLUE,
                edgecolor="white", zorder=3)

    # value labels
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10, color=GRAY)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f"{h:.2f}" if h > 0.1 else f"{h:.3f}", ha="center", va="bottom",
                fontsize=10, color=BLUE)

    # annotation
    ax.annotate("No change in bits\nregardless of f", xy=(x[2] - width/2, v5_bits[2]/1e8),
                xytext=(1.2, 4.0),
                fontsize=10, color=GRAY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GRAY),
                bbox=dict(boxstyle="round", fc="white", ec=GRAY, alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Computation bits (×10⁸)", labelpad=8)
    ax.set_title("Throughput vs. f Choice", pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, 7)

    # ── Right: energy ────────────────────────────────────────────────────────
    ax = axes[1]
    ax.bar(x - width/2, v5_energy, width, label="v5 env", color=GRAY,
           edgecolor="white", zorder=3)
    ax.bar(x + width/2, v6_energy, width, label="v6 env", color=BLUE,
           edgecolor="white", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total energy consumed (arb. unit)", labelpad=8)
    ax.set_title("Energy Cost vs. f Choice", pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, 700)
    # E_uav budget note as text box instead of axhline (2000 >> ylim)
    ax.text(0.02, 0.96, "Energy budget E_uav = 2000\n(far above — well within limit)",
            transform=ax.transAxes, fontsize=9, color=RED, va="top",
            bbox=dict(boxstyle="round", fc="white", ec=RED, alpha=0.8))

    fig.suptitle("Effect of CPU Frequency f — Before vs. After Environment Redesign",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUT / "fig3_f_effect_v5_vs_v6.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — v1 vs v5 trajectory side by side
# ═══════════════════════════════════════════════════════════════════════════
def fig4_trajectory_comparison():
    """Load trajectory PNGs and arrange them side by side with annotations."""
    from PIL import Image

    v1_path = Path("results/v1/trajectory_maddpg.png")
    v5_path = Path("results/v5_s0/trajectory.png")

    if not v1_path.exists() or not v5_path.exists():
        print(f"[warn] trajectory images not found, skipping fig4")
        print(f"  expected: {v1_path}, {v5_path}")
        return

    img_v1 = Image.open(v1_path)
    img_v5 = Image.open(v5_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    axes[0].imshow(img_v1)
    axes[0].axis("off")
    axes[0].set_title("v1 — Lazy Agent Problem\nF_uav = 0.33  (only 1 UAV active, 2 stationary)",
                      fontsize=13, fontweight="bold", color=RED, pad=10)

    # Annotations on v1
    axes[0].annotate("2 UAVs\nstationary",
                     xy=(0.25, 0.75), xytext=(0.05, 0.55),
                     xycoords="axes fraction", textcoords="axes fraction",
                     fontsize=11, color=RED, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    axes[1].imshow(img_v5)
    axes[1].axis("off")
    axes[1].set_title("v5 — Lazy Agent Resolved\nF_uav = 0.977  (all 3 UAVs actively flying)",
                      fontsize=13, fontweight="bold", color=GREEN, pad=10)

    fig.suptitle("UAV Trajectory: Before vs. After Training Improvement",
                 fontsize=16, fontweight="bold", y=1.01)

    # divider line
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.05, 0.95],
                              transform=fig.transFigure,
                              color=GRAY, linewidth=1.5, linestyle="--"))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = OUT / "fig4_trajectory_v1_vs_v5.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {p}")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating presentation figures...")
    fig1_fuav_versions()
    fig2_baseline_comparison()
    fig3_f_effect()
    try:
        fig4_trajectory_comparison()
    except ImportError:
        print("[warn] PIL not available, skipping fig4 (trajectory side-by-side)")
    print("\nDone. Files in results/presentation/")
