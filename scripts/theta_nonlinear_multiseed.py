#!/usr/bin/env python3
"""
Item (2) of Section 8.9's investigation plan, first sub-step: before paying
for a more expensive K, check how much Delta_allocator = Theta_nonlinear(0)
- Theta_LP(0) varies seed-to-seed at the SAME K=15 already run once
(scripts/theta_nonlinear_throughout.py, seed 20260730, paired result
Delta_allocator=-5.164). This script adds ADDITIONAL independent replicates
at K=15 -- each replicate varies BOTH the eta/resid bootstrap draw and the
state-path randomness together (matching the convention Section 8.8's
existing 5-seed LP-only check already uses), with the LP and nonlinear
allocators paired via common random numbers WITHIN each replicate (the same
lsm.RNG-reset trick theta_nonlinear_throughout.py established and verified
bit-identical before trusting).

Rationale for starting here rather than immediately increasing K: if
Delta_allocator is already stable across seeds at K=15, that is itself
useful evidence the K=15 estimate's sign/magnitude is not a fluke of one
particular simulated world, cheaply, before committing to K=30's roughly
2x compute cost per replicate.

Output: outputs/theta_nonlinear_multiseed.json (one entry per NEW seed;
combine with theta_nonlinear_throughout.json's existing seed=20260730
result for the full picture).
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
K_PATHS_REDUCED = 15
NEW_SEEDS = [1, 2]   # additional replicates; seed 20260730 already run


def run_one_seed(seed: int) -> dict:
    races = lsm.build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]

    eta_rng = np.random.default_rng(seed)
    eta_arr_by_path, resid_std_arr_by_path, boot_summary = lsm.bootstrap_eta_resid_paths(
        lsm.BOOTSTRAP_CYCLES, tiers_per_race, lsm.K_PATHS, eta_rng)

    print(f"\n=== seed={seed}: nonlinear-throughout ===")
    lsm.RNG = np.random.default_rng(seed)
    t0 = time.time()
    res_nonlinear = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path,
                                 f"seed{seed}_nonlinear", eta_summary=boot_summary,
                                 use_nonlinear_allocator=True)
    elapsed_nonlinear = time.time() - t0
    print(f"  -> wall time: {elapsed_nonlinear/3600:.2f} hours")

    print(f"=== seed={seed}: LP-throughout, paired (same state RNG seed) ===")
    lsm.RNG = np.random.default_rng(seed)
    t0 = time.time()
    res_lp = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path,
                          f"seed{seed}_lp", eta_summary=boot_summary,
                          use_nonlinear_allocator=False)
    elapsed_lp = time.time() - t0

    theta0_nonlinear = res_nonlinear["theta_by_period"][0]["mean_theta"]
    theta0_lp = res_lp["theta_by_period"][0]["mean_theta"]
    delta = theta0_nonlinear - theta0_lp
    print(f"  seed={seed}: Theta_LP(0)={theta0_lp:+.4f}  Theta_nonlinear(0)={theta0_nonlinear:+.4f}  "
          f"Delta_allocator={delta:+.4f}")

    return {
        "seed": seed,
        "theta0_lp": theta0_lp,
        "theta0_nonlinear": theta0_nonlinear,
        "delta_allocator": delta,
        "frac_deploy_now_lp": res_lp["theta_by_period"][0]["frac_deploy_now"],
        "frac_deploy_now_nonlinear": res_nonlinear["theta_by_period"][0]["frac_deploy_now"],
        "nonlinear_wall_seconds": elapsed_nonlinear,
        "lp_wall_seconds": elapsed_lp,
        "nonlinear_full": res_nonlinear,
        "lp_full": res_lp,
    }


def run_one_seed_single_cycle(seed: int, fit_cycle: int) -> dict:
    """Item (6): the same paired LP-vs-nonlinear comparison, but for a
    single-cycle eta bracket (eta_fit_2022 or eta_fit_2024) instead of
    eta_bootstrap_all_cycles -- tile_single_cycle's identical (eta,
    resid_std) per tier across every path, rather than a per-path bootstrap
    draw. `seed` controls only the state-path RNG here (d/r/eps/G_t), since
    tile_single_cycle has no randomness of its own to seed."""
    races = lsm.build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]

    eta_by_tier, resid_std_by_tier = lsm.fit_eta_and_resid(fit_cycle)
    eta_arr_by_path, resid_std_arr_by_path = lsm.tile_single_cycle(
        eta_by_tier, resid_std_by_tier, tiers_per_race, lsm.K_PATHS)
    label = f"eta_fit_{fit_cycle}"

    print(f"\n=== {label}, seed={seed}: nonlinear-throughout ===")
    lsm.RNG = np.random.default_rng(seed)
    t0 = time.time()
    res_nonlinear = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path,
                                 f"{label}_seed{seed}_nonlinear",
                                 eta_summary={"single_cycle_fit": eta_by_tier},
                                 use_nonlinear_allocator=True)
    elapsed_nonlinear = time.time() - t0
    print(f"  -> wall time: {elapsed_nonlinear/3600:.2f} hours")

    print(f"=== {label}, seed={seed}: LP-throughout, paired ===")
    lsm.RNG = np.random.default_rng(seed)
    t0 = time.time()
    res_lp = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path,
                          f"{label}_seed{seed}_lp",
                          eta_summary={"single_cycle_fit": eta_by_tier},
                          use_nonlinear_allocator=False)
    elapsed_lp = time.time() - t0

    theta0_nonlinear = res_nonlinear["theta_by_period"][0]["mean_theta"]
    theta0_lp = res_lp["theta_by_period"][0]["mean_theta"]
    delta = theta0_nonlinear - theta0_lp
    print(f"  {label} seed={seed}: Theta_LP(0)={theta0_lp:+.4f}  "
          f"Theta_nonlinear(0)={theta0_nonlinear:+.4f}  Delta_allocator={delta:+.4f}")

    return {
        "scenario": label, "seed": seed,
        "theta0_lp": theta0_lp, "theta0_nonlinear": theta0_nonlinear,
        "delta_allocator": delta,
        "frac_deploy_now_lp": res_lp["theta_by_period"][0]["frac_deploy_now"],
        "frac_deploy_now_nonlinear": res_nonlinear["theta_by_period"][0]["frac_deploy_now"],
        "nonlinear_wall_seconds": elapsed_nonlinear, "lp_wall_seconds": elapsed_lp,
        "nonlinear_full": res_nonlinear, "lp_full": res_lp,
    }


def main():
    lsm.K_PATHS = K_PATHS_REDUCED
    print(f"K_PATHS={lsm.K_PATHS}, N_PERIODS={lsm.N_PERIODS} ({lsm.N_PERIODS * lsm.PERIOD_DAYS} days)")
    print(f"Prior result (seed=20260730, paired): Theta_LP(0)=+4.817, "
          f"Theta_nonlinear(0)=-0.347, Delta_allocator=-5.164")

    results = []
    for seed in NEW_SEEDS:
        results.append(run_one_seed(seed))

    deltas = [r["delta_allocator"] for r in results] + [-5.164]   # include the prior seed
    print(f"\n=== Summary across {len(deltas)} seeds (including prior seed=20260730) ===")
    print(f"Delta_allocator values: {[round(d, 4) for d in deltas]}")
    print(f"mean={np.mean(deltas):+.4f}, SD={np.std(deltas, ddof=1):.4f}, "
          f"min={min(deltas):+.4f}, max={max(deltas):+.4f}")

    out_path = ROOT / "outputs/theta_nonlinear_multiseed.json"
    with open(out_path, "w") as f:
        json.dump({
            "k_paths": K_PATHS_REDUCED,
            "new_seed_results": results,
            "prior_seed_20260730": {"theta0_lp": 4.8166, "theta0_nonlinear": -0.3472,
                                     "delta_allocator": -5.1638},
            "all_deltas": deltas,
            "mean_delta": float(np.mean(deltas)),
            "sd_delta": float(np.std(deltas, ddof=1)),
        }, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
