"""Probe: does CPU frequency f affect throughput at all?

Hypothesis: in this env, when a UAV is in range of an SD, the SD's ENTIRE queue
is cleared in a single slot regardless of f. f only changes per-slot energy cost.

Test: run identical episodes (same SD layout, same actions for d/theta) but
sweep f across {f_min, f_max/2, f_max}. If hypothesis holds, bits should be
identical and only energy should differ.
"""
import numpy as np
from env import UAVMECEnv

EPISODES = 30
F_GRID = [
    ("f_min (0)",   -1.0),
    ("f_mid (0.5)",  0.0),
    ("f_max (1)",   +1.0),
]

rng = np.random.default_rng(0)

results = {label: {"bits": [], "energy_used": []} for label, _ in F_GRID}

for ep in range(EPISODES):
    # Use the same seed for all f variants -> same SD layout, same task gen
    for label, f_action in F_GRID:
        env = UAVMECEnv(seed=100 + ep)
        env.reset()
        # Use a deterministic d/theta sequence (same across variants), only f varies
        ep_rng = np.random.default_rng(1000 + ep)
        for t in range(env.T):
            actions = np.zeros((env.M, env.act_dim), dtype=np.float32)
            actions[:, 0] = ep_rng.uniform(-1, 1, size=env.M)   # d
            actions[:, 1] = ep_rng.uniform(-1, 1, size=env.M)   # theta
            actions[:, 2] = f_action                             # f fixed per variant
            _, _, done, _ = env.step(actions)
            if done:
                break
        s = env.summary()
        results[label]["bits"].append(s["computation_bits"])
        results[label]["energy_used"].append(sum(s["energy_total_used"]))

print(f"\n=== f-sweep over {EPISODES} episodes (same d/theta, only f differs) ===")
print(f"{'variant':<15} {'bits (mean ± std)':<26} {'energy_used (mean ± std)':<28}")
print("-" * 72)
for label, _ in F_GRID:
    b = np.array(results[label]["bits"])
    e = np.array(results[label]["energy_used"])
    print(f"{label:<15} {b.mean():.3e} ± {b.std():.2e}    {e.mean():.2f} ± {e.std():.2f}")

# Pairwise bits diff
b0 = np.array(results[F_GRID[0][0]]["bits"])
b1 = np.array(results[F_GRID[1][0]]["bits"])
b2 = np.array(results[F_GRID[2][0]]["bits"])
print(f"\nBits identical across f variants? "
      f"f_min vs f_mid: {np.allclose(b0, b1)}, f_min vs f_max: {np.allclose(b0, b2)}")
print(f"Max abs diff bits (f_min vs f_max): {np.abs(b0 - b2).max():.3e}")
