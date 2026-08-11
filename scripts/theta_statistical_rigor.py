#!/usr/bin/env python3
"""
Statistical rigor for the live Theta(0) result (Paper III revision,
reviewer-requested, 2026-07-28): the headline Table 11 numbers were single-
seed point estimates with no simulation-noise accounting and an in-sample
Longstaff-Schwartz continuation regression (look-ahead bias risk). This adds:

  1. Multiple independent seeds -> Monte Carlo standard error for Theta(0).
  2. Out-of-sample policy evaluation: refit with held_out_frac=0.3, so 30% of
     paths' reported theta/frac_deploy_now never had their own realized
     future value feed the regression that decided their stopping choice.
  3. K-sensitivity: rerun at K=2000 (baseline) and K=5000.

Run only against eta_bootstrap_all_cycles (the primary, cycle-pooled
calibration) to keep this tractable -- the point is characterizing
estimation uncertainty in the headline number, not re-deriving every bracket.

Output: outputs/theta_statistical_rigor.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import solve_bellman_lsm as lsm

ROOT = Path(__file__).parent.parent
SEEDS = [20260716, 1, 2, 3, 4]   # 5 independent seeds (first matches the original headline run)


def run_one(seed: int, k_paths: int, held_out_frac: float = 0.0):
    lsm.RNG = np.random.default_rng(seed)
    lsm.K_PATHS = k_paths
    races = lsm.build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]
    eta_arr_by_path, resid_std_arr_by_path, boot_summary = lsm.bootstrap_eta_resid_paths(
        lsm.BOOTSTRAP_CYCLES, tiers_per_race, k_paths, lsm.RNG)
    res = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path,
                       f"seed{seed}_K{k_paths}", eta_summary=boot_summary,
                       held_out_frac=held_out_frac)
    return res["theta_by_period"][0]   # period-0 entry (the live 98-day horizon)


def main():
    results = {"multi_seed_K2000": [], "out_of_sample_K2000": None, "k_sensitivity": {}}

    print("=== Multi-seed Monte Carlo error, K=2000 ===")
    for seed in SEEDS:
        entry = run_one(seed, k_paths=2000)
        print(f"  seed={seed}: Theta(0)={entry['mean_theta']:+.4f}, "
              f"frac_deploy_now={entry['frac_deploy_now']:.3f}")
        results["multi_seed_K2000"].append({"seed": seed, **entry})
    thetas = np.array([r["mean_theta"] for r in results["multi_seed_K2000"]])
    results["theta0_mc_mean"] = float(thetas.mean())
    results["theta0_mc_se"] = float(thetas.std(ddof=1) / np.sqrt(len(thetas)))
    results["theta0_mc_sd_across_seeds"] = float(thetas.std(ddof=1))
    print(f"  -> across {len(SEEDS)} seeds: mean Theta(0)={results['theta0_mc_mean']:+.4f}, "
          f"SD={results['theta0_mc_sd_across_seeds']:.4f}, SE={results['theta0_mc_se']:.4f}")

    print("\n=== Out-of-sample policy evaluation, K=2000, 30% held out ===")
    oos_entry = run_one(SEEDS[0], k_paths=2000, held_out_frac=0.3)
    results["out_of_sample_K2000"] = oos_entry
    print(f"  in-sample (all paths) Theta(0)={oos_entry['mean_theta']:+.4f}, "
          f"held-out-only Theta(0)={oos_entry.get('mean_theta_held_out'):+.4f}")

    print("\n=== K-sensitivity, single seed ===")
    for k in [2000, 5000]:
        entry = run_one(SEEDS[0], k_paths=k)
        print(f"  K={k}: Theta(0)={entry['mean_theta']:+.4f}, "
              f"frac_deploy_now={entry['frac_deploy_now']:.3f}, basis_r2={entry['basis_r2']:.3f}")
        results["k_sensitivity"][k] = entry

    out_path = ROOT / "outputs/theta_statistical_rigor.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
