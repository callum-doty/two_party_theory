#!/usr/bin/env python3
"""
The decisive resolution the K-progression (15/30/50/100) was building
toward: now that the concave-envelope surrogate is validated both against
optimize_nonlinear()'s single-state objective (0.11-0.19 seats out of
~235-240, Table in theta_concave_surrogate.py) AND against the full
nonlinear-throughout Theta(0) estimate at K=15 (within ~0.03 seats at
every one of 3 seeds, all same-signed), it is fast enough (~0.03s/call vs.
40s-3,600s for the true solver) to run at the SAME K=2,000 headline
precision the original LP-based Tables 11-13 used, for all three
calibration scenarios -- addressing item (6) of the investigation plan at
full statistical power, not just the eta_bootstrap_all_cycles scenario
this session's K-progression focused on for tractability.

At K=2,000, N_PERIODS=7: (7+1)*2,000=16,000 surrogate calls per scenario,
~0.03s each => roughly 8-10 minutes per scenario. All three scenarios are
launched as independent parallel processes (not sequentially within one
script), so wall-clock time for all three is close to the time for one.

Note (2026-08): this script's result is no longer the ONLY place the
corrected calculation lives -- solve_bellman_lsm.py's own main() was found
to still default to the LP allocator (neither allocator flag was ever
passed to its run_lsm() calls), silently reproducing the superseded,
wrong-sign figure in outputs/theta_schedule.json even after this script's
decisive re-solve had already established the corrected answer. main() now
passes use_surrogate_allocator=True too, so outputs/theta_schedule.json
should match this script's per-scenario output (up to RNG-seed/K
differences from running all three scenarios in one process vs. as
separate parallel processes here). This script remains useful as a
standalone, single-scenario rerun and as the historical record of the
original decisive re-solve.

Output: outputs/theta_surrogate_headline_{scenario}.json
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import solve_bellman_lsm as lsm

ROOT = Path(__file__).parent.parent
K_HEADLINE = 2000
SEED = 20260716   # matches the original headline seed for continuity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["bootstrap", "eta_fit_2022", "eta_fit_2024"], required=True)
    args = ap.parse_args()

    lsm.K_PATHS = K_HEADLINE
    print(f"scenario={args.scenario} K_PATHS={lsm.K_PATHS} N_PERIODS={lsm.N_PERIODS} "
          f"({lsm.N_PERIODS * lsm.PERIOD_DAYS} days) allocator=surrogate")

    races = lsm.build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]

    if args.scenario == "bootstrap":
        eta_rng = np.random.default_rng(SEED)
        eta_arr_by_path, resid_std_arr_by_path, eta_summary = lsm.bootstrap_eta_resid_paths(
            lsm.BOOTSTRAP_CYCLES, tiers_per_race, lsm.K_PATHS, eta_rng)
        label = "eta_bootstrap_all_cycles_surrogate"
    else:
        fit_cycle = 2022 if args.scenario == "eta_fit_2022" else 2024
        eta_by_tier, resid_std_by_tier = lsm.fit_eta_and_resid(fit_cycle)
        eta_arr_by_path, resid_std_arr_by_path = lsm.tile_single_cycle(
            eta_by_tier, resid_std_by_tier, tiers_per_race, lsm.K_PATHS)
        eta_summary = {"single_cycle_fit": eta_by_tier}
        label = f"eta_fit_{fit_cycle}_surrogate"

    lsm.RNG = np.random.default_rng(SEED)
    t0 = time.time()
    result = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path, label,
                          eta_summary=eta_summary, use_surrogate_allocator=True)
    elapsed = time.time() - t0
    print(f"\n{args.scenario}: Theta(0)={result['theta_by_period'][0]['mean_theta']:+.4f}  "
          f"frac_deploy_now(0)={result['theta_by_period'][0]['frac_deploy_now']:.3f}  "
          f"wall_time={elapsed/60:.1f} min")

    out_path = ROOT / f"outputs/theta_surrogate_headline_{args.scenario}.json"
    with open(out_path, "w") as f:
        json.dump({"k_paths": K_HEADLINE, "seed": SEED, "scenario": args.scenario,
                    "wall_seconds": elapsed, "result": result}, f, indent=2, default=str)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
