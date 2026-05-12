# UAV-MEC Energy-Constrained MADRL

PyTorch implementation of a multi-agent deep reinforcement learning approach for
joint trajectory and CPU-frequency optimization in energy-constrained multi-UAV
assisted Mobile Edge Computing (MEC) systems.

Based on:
> Shi et al. (2026), *A Deep Reinforcement Learning Based Approach for
> Optimizing Trajectory and Frequency in Energy Constrained Multi-UAV
> Assisted MEC System.*

This is a **modified re-implementation**, not an exact reproduction. See
[`PROJECT_NOTES.md`](PROJECT_NOTES.md) for detailed design decisions, deviations
from the paper, and experimental insights.

---

## Problem

Multiple UAVs serve ground IoT devices (Smart Devices) by collecting and
processing computation tasks. The objective is to maximize total computation
bits while preserving fairness (per-SD service balance and per-UAV load
balance) under an energy budget constraint.

- **State**: per-UAV coordinates, cumulative service history, inter-UAV
  distances, UAV ID one-hot
- **Action** (continuous, per UAV): flight distance, direction, CPU frequency
- **Reward**: `N × F_sd × F_uav − penalty` (paper) with optional individual
  bonus `+ β × N` (this implementation)

---

## Algorithms

| Algorithm | File | Notes |
|-----------|------|-------|
| MADDPG | `agents/maddpg.py` | Parameter-shared actor + per-agent centralized critic |
| MATD3 | `agents/matd3.py` | Twin critics, target policy smoothing, delayed updates |
| DDPG | `agents/ddpg.py` | Single-agent baseline (shared policy, local obs) |
| RANDOM | `baselines/random_strategy.py` | Uniform random actions |
| CIRCLE | `baselines/circle_strategy.py` | K-means clusters, orbit centroids |
| GREEDY | `baselines/greedy_strategy.py` | Oracle access to SD queues |

---

## Setup

### Option 1: Docker (recommended)

```bash
docker build -t uav-mec .
docker run --gpus all -it -v "$(pwd)":/workspace uav-mec bash
```

### Option 2: Local

```bash
pip install -r requirements.txt
```

Requires Python ≥3.10, PyTorch ≥2.2 with CUDA support recommended.

---

## Usage

### Training

```bash
# MADDPG, 50k episodes
python train.py --algo maddpg --episodes 50000 --seed 0 \
  --log-file logs/run.jsonl --ckpt-dir checkpoints/run

# MATD3
python train.py --algo matd3 --episodes 50000 --seed 0

# Multi-seed parallel (one per GPU / CPU core block)
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((s%2)) taskset -c $((s*7))-$((s*7+6)) \
    python train.py --algo maddpg --seed $s \
      --log-file logs/seed_$s.jsonl --ckpt-dir checkpoints/seed_$s &
done
```

### Evaluation (5 scenarios, baselines vs MADDPG)

```bash
python evaluate.py --maddpg-ckpt checkpoints/seed_0/maddpg_final.pt \
  --repeats 10 --seed 100
```

### Visualization

```bash
# Trajectory (PNG + optional GIF)
python visualize_trajectory.py --algo maddpg \
  --ckpt checkpoints/seed_0/maddpg_final.pt --animate

# Training curves
python plot_training.py logs/seed_0.jsonl --out training.png

# Multi-seed mean ± std
python plot_multiseed.py logs/seed_*.jsonl --out multiseed.png
```

### Analysis tools

```bash
# Per-SD task processing rate + leftover heatmap
python analyze_task_processing.py --maddpg-ckpt checkpoints/seed_0/maddpg_final.pt

# Per-UAV energy usage and constraint violations across many episodes
python analyze_energy.py --algo maddpg \
  --maddpg-ckpt checkpoints/seed_0/maddpg_final.pt --episodes 50

# Distribution of CPU frequency choices
python analyze_frequency.py --algo maddpg \
  --maddpg-ckpt checkpoints/seed_0/maddpg_final.pt --episodes 30
```

---

## Configuration

All hyperparameters live in [`config.json`](config.json):

- `system`: environment constants (num UAVs/SDs, map size, energy budget, etc.)
- `rl`: RL hyperparameters (lr, gamma, tau, batch size, noise schedule)
- `unspecified_in_paper`: design decisions where the paper is ambiguous
  (energy scaling, reward bonus, movement-energy multiplier, etc.)

Key toggles for ablation:

```json
"use_fairness_reward": true/false,    // include F_sd × F_uav term
"individual_bonus_coef": 0.5,         // direct N-based bonus
"movement_energy_mult": 1e12,         // scale movement energy cost
"energy_overrun_penalty_base": 3.0,
"energy_overrun_penalty_per_excess": 0.5
```

---

## Project Structure

```
.
├── env/uav_mec_env.py          # vectorized simulation environment
├── agents/                     # MADDPG, MATD3, DDPG, networks, replay buffer
├── baselines/                  # RANDOM, CIRCLE, GREEDY
├── train.py                    # training entry point
├── evaluate.py                 # 5-scenario evaluation table
├── visualize_trajectory.py     # static PNG + GIF
├── plot_training.py            # single-run training curves
├── plot_multiseed.py           # multi-seed mean ± std curves
├── analyze_task_processing.py  # processing rate + leftover heatmap
├── analyze_energy.py           # energy usage and violations
├── analyze_frequency.py        # frequency choice distribution
├── config.json                 # all hyperparameters
├── Dockerfile                  # CUDA + PyTorch runtime
├── PROJECT_NOTES.md            # design decisions and findings (Korean)
└── README.md                   # this file
```

---

## Deviations from the Paper

This implementation departs from the paper in several ways. The full list is
in [`PROJECT_NOTES.md`](PROJECT_NOTES.md), but headline changes:

- **Parameter sharing** across actors (paper presumably uses independent actors)
- **Individual contribution bonus** `+ β × N_m` added to the reward
- **Penalty reduced** from 10 to 1
- **Soft energy constraint**: UAVs remain active when over budget and pay a
  per-slot penalty, instead of being deactivated
- **Movement-energy multiplier** introduced to balance flight vs computation cost
  (the paper's raw formulas make movement ~10⁹ times cheaper than computation)
- Several unspecified parameters (e.g. `cycles/bit`, `f_min`, `p_tran`) are
  set by us

The paper's energy formula uses **mixed normalized units** (e.g. `E_uav` is in
mAh while `k1, f, p_tran` units are unspecified), so this implementation should
be viewed as a *normalized simulation* rather than a physically-grounded one.

---

## Key Findings

From experiments documented in `PROJECT_NOTES.md`:

1. **Lazy-agent problem is reward-driven, not energy-driven.** Reducing the
   penalty and adding a direct per-agent reward fixes it; parameter sharing
   alone does not.
2. **CPU-frequency choice is effectively unlearnable** in the paper's
   environment because changing `f` has no effect on per-slot throughput. The
   trained policy outputs a frequency distribution statistically
   indistinguishable from RANDOM.
3. **GREEDY uses oracle queue information**, so it is not a fair baseline.
   Despite this, MADDPG reaches 85–90 % of GREEDY in bits while being roughly
   2× more energy-efficient (largely as a side effect of #2).
4. **The energy budget `E_uav = 2000` is not a binding constraint** with the
   paper's parameters: even GREEDY uses only ~16 % of the budget on average,
   and 0 violations are observed across 50 episodes.
5. **Single-seed RL results are unreliable**: variance across 4 seeds is
   ~8 % CV for bits, ~13 % CV for return.

---

## Status

- ✅ MADDPG with parameter sharing — trained, evaluated
- ✅ Baselines (RANDOM, CIRCLE, GREEDY) — implemented
- ✅ Analysis tools (energy, frequency, task processing)
- ✅ MATD3 — implemented, not yet trained
- ⏳ Algorithm-level verification against standard MARL benchmarks — pending
- ⏳ Larger-N seed averages for tighter statistics — pending

---

## Citation

Original paper:
```
Shi et al., "A Deep Reinforcement Learning Based Approach for Optimizing
Trajectory and Frequency in Energy Constrained Multi-UAV Assisted MEC System,"
2026.
```

This re-implementation is for research / educational purposes.

---

## License

MIT (see `LICENSE` if added).
