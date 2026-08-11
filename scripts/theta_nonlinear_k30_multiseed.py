#!/usr/bin/env python3
"""
Matches the rigor already applied at K=15 (three independent seeds, Table
13e) at K=30: two ADDITIONAL seeds on top of the existing K=30 replicate
(seed=20260731, scripts/theta_nonlinear_k30.py, Delta_allocator=-4.703),
giving three total at K=30 -- the same n=3 standard the investigation plan
already met at K=15 before moving on.

Reuses theta_nonlinear_multiseed.run_one_seed() unchanged (the verified,
paired, common-random-numbers methodology) -- only K_PATHS and the seeds
differ from the K=15 multiseed run.

Output: outputs/theta_nonlinear_k30_multiseed.json
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import solve_bellman_lsm as lsm
import theta_nonlinear_multiseed as tms

ROOT = Path(__file__).parent.parent
K_PATHS_K30 = 30
NEW_SEEDS = [3, 4]   # seed=20260731 already run (theta_nonlinear_k30.py)

PRIOR_K30 = {"seed": 20260731, "theta0_lp": 4.2683, "theta0_nonlinear": -0.4348,
             "delta_allocator": -4.7031}


def main():
    lsm.K_PATHS = K_PATHS_K30
    tms.K_PATHS_REDUCED = K_PATHS_K30
    print(f"K_PATHS={lsm.K_PATHS}, N_PERIODS={lsm.N_PERIODS} ({lsm.N_PERIODS * lsm.PERIOD_DAYS} days)")
    print(f"Prior K=30 seed (20260731): Delta_allocator={PRIOR_K30['delta_allocator']:+.4f}")
    print(f"Prior K=15 seeds: Delta_allocator = -5.164, -4.723, -4.278 (mean -4.722, SD 0.443)")

    results = []
    for seed in NEW_SEEDS:
        results.append(tms.run_one_seed(seed))

    deltas = [PRIOR_K30["delta_allocator"]] + [r["delta_allocator"] for r in results]
    thetas_nl = [PRIOR_K30["theta0_nonlinear"]] + [r["theta0_nonlinear"] for r in results]
    thetas_lp = [PRIOR_K30["theta0_lp"]] + [r["theta0_lp"] for r in results]
    print(f"\n=== Summary across {len(deltas)} seeds at K=30 ===")
    print(f"Theta_LP(0): {[round(t, 4) for t in thetas_lp]}, mean={np.mean(thetas_lp):+.4f}, SD={np.std(thetas_lp, ddof=1):.4f}")
    print(f"Theta_nonlinear(0): {[round(t, 4) for t in thetas_nl]}, mean={np.mean(thetas_nl):+.4f}, SD={np.std(thetas_nl, ddof=1):.4f}")
    print(f"Delta_allocator: {[round(d, 4) for d in deltas]}, mean={np.mean(deltas):+.4f}, SD={np.std(deltas, ddof=1):.4f}")

    out_path = ROOT / "outputs/theta_nonlinear_k30_multiseed.json"
    with open(out_path, "w") as f:
        json.dump({
            "k_paths": K_PATHS_K30,
            "prior_seed_20260731": PRIOR_K30,
            "new_seed_results": results,
            "all_theta_lp": thetas_lp, "all_theta_nonlinear": thetas_nl, "all_deltas": deltas,
            "mean_theta_lp": float(np.mean(thetas_lp)), "sd_theta_lp": float(np.std(thetas_lp, ddof=1)),
            "mean_theta_nonlinear": float(np.mean(thetas_nl)), "sd_theta_nonlinear": float(np.std(thetas_nl, ddof=1)),
            "mean_delta": float(np.mean(deltas)), "sd_delta": float(np.std(deltas, ddof=1)),
        }, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
