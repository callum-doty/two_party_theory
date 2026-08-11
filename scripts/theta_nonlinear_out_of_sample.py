#!/usr/bin/env python3
"""
Item (3) of Section 8.9's investigation plan: out-of-sample policy
evaluation for the nonlinear-throughout run, mirroring Section 8.8's
held-out-path methodology (train the continuation-value regression on a
subset of paths, evaluate the resulting stopping policy on paths whose own
realized future value never informed that regression).

held_out_frac uses a fixed-seed RNG (np.random.default_rng(999)) completely
independent of the state-path RNG that drives d/r/eps/G_t simulation --
confirmed by reading run_lsm()'s implementation before running anything --
so this is compatible with the paired, common-random-numbers methodology
already established (Table 13d) without any modification: the SAME
train/held-out split applies to both the LP and nonlinear runs, and adding
held_out_frac costs no extra nonlinear solver calls (same K, same per-
period allocator calls; only which rows feed the regression fit changes).

Run at K=30 (already shown stable across 3+ seeds, Table 13g) with 30%
held out, paired between LP and nonlinear -- the natural next check once
K is large enough that a 70/30 split still leaves a usable training set.

Output: outputs/theta_nonlinear_out_of_sample.json
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import solve_bellman_lsm as lsm

ROOT = Path(__file__).parent.parent
K_PATHS_OOS = 30
SEED = 7
HELD_OUT_FRAC = 0.3


def main():
    lsm.K_PATHS = K_PATHS_OOS
    print(f"K_PATHS={lsm.K_PATHS}, N_PERIODS={lsm.N_PERIODS} ({lsm.N_PERIODS * lsm.PERIOD_DAYS} days), "
          f"held_out_frac={HELD_OUT_FRAC}")

    races = lsm.build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]

    eta_rng = np.random.default_rng(SEED)
    eta_arr_by_path, resid_std_arr_by_path, boot_summary = lsm.bootstrap_eta_resid_paths(
        lsm.BOOTSTRAP_CYCLES, tiers_per_race, lsm.K_PATHS, eta_rng)

    print("\n=== nonlinear-throughout, out-of-sample ===")
    lsm.RNG = np.random.default_rng(SEED)
    t0 = time.time()
    res_nl = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path, "oos_nonlinear",
                          eta_summary=boot_summary, use_nonlinear_allocator=True,
                          held_out_frac=HELD_OUT_FRAC)
    elapsed_nl = time.time() - t0
    print(f"  -> wall time: {elapsed_nl/3600:.2f} hours")

    print("\n=== LP-throughout, out-of-sample, paired (same state RNG seed) ===")
    lsm.RNG = np.random.default_rng(SEED)
    t0 = time.time()
    res_lp = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path, "oos_lp",
                          eta_summary=boot_summary, use_nonlinear_allocator=False,
                          held_out_frac=HELD_OUT_FRAC)
    elapsed_lp = time.time() - t0

    t0_nl = res_nl["theta_by_period"][0]
    t0_lp = res_lp["theta_by_period"][0]
    print(f"\nNonlinear: in-sample Theta(0)={t0_nl['mean_theta']:+.4f}  "
          f"held-out Theta(0)={t0_nl['mean_theta_held_out']:+.4f}")
    print(f"LP:        in-sample Theta(0)={t0_lp['mean_theta']:+.4f}  "
          f"held-out Theta(0)={t0_lp['mean_theta_held_out']:+.4f}")
    print(f"Delta_allocator (in-sample): {t0_nl['mean_theta'] - t0_lp['mean_theta']:+.4f}")
    print(f"Delta_allocator (held-out):  {t0_nl['mean_theta_held_out'] - t0_lp['mean_theta_held_out']:+.4f}")

    out_path = ROOT / "outputs/theta_nonlinear_out_of_sample.json"
    with open(out_path, "w") as f:
        json.dump({
            "k_paths": K_PATHS_OOS, "seed": SEED, "held_out_frac": HELD_OUT_FRAC,
            "nonlinear": res_nl, "lp": res_lp,
            "nonlinear_wall_seconds": elapsed_nl, "lp_wall_seconds": elapsed_lp,
        }, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
