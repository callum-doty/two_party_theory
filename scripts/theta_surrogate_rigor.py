#!/usr/bin/env python3
"""
Statistical rigor for the surrogate-based headline result (Section 8.8's
methodology, applied to the now-decisive surrogate-throughout Theta(0) at
K=2,000). Cheap to do properly now that the surrogate runs in ~9 minutes
per replicate instead of hours: multiple independent seeds for a Monte
Carlo SE, plus an out-of-sample (held-out-path) check, both on
eta_bootstrap_all_cycles -- the scenario closest to indifference (53.4%
deploy at the single-seed headline run) and therefore the one most in
need of a precise estimate; eta_fit_2022/2024 were unanimous (100% deploy)
at the single-seed headline run and are less sensitive to exact precision.

Output: outputs/theta_surrogate_rigor.json
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
K_PATHS = 2000
SEEDS = [1, 2, 3, 4]   # plus the headline seed=20260716 already run
OOS_SEED = 5
HELD_OUT_FRAC = 0.3


def run_seed(seed: int) -> dict:
    races = lsm.build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]
    eta_rng = np.random.default_rng(seed)
    eta_arr, resid_arr, boot_summary = lsm.bootstrap_eta_resid_paths(
        lsm.BOOTSTRAP_CYCLES, tiers_per_race, lsm.K_PATHS, eta_rng)
    lsm.RNG = np.random.default_rng(seed)
    t0 = time.time()
    res = lsm.run_lsm(eta_arr, resid_arr, f"surrogate_rigor_seed{seed}",
                       eta_summary=boot_summary, use_surrogate_allocator=True)
    elapsed = time.time() - t0
    t0_entry = res["theta_by_period"][0]
    print(f"  seed={seed}: Theta(0)={t0_entry['mean_theta']:+.4f} "
          f"frac_deploy={t0_entry['frac_deploy_now']:.3f} ({elapsed/60:.1f} min)")
    return {"seed": seed, "mean_theta": t0_entry["mean_theta"],
            "frac_deploy_now": t0_entry["frac_deploy_now"], "wall_seconds": elapsed}


def run_out_of_sample() -> dict:
    races = lsm.build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]
    eta_rng = np.random.default_rng(OOS_SEED)
    eta_arr, resid_arr, boot_summary = lsm.bootstrap_eta_resid_paths(
        lsm.BOOTSTRAP_CYCLES, tiers_per_race, lsm.K_PATHS, eta_rng)
    lsm.RNG = np.random.default_rng(OOS_SEED)
    t0 = time.time()
    res = lsm.run_lsm(eta_arr, resid_arr, "surrogate_rigor_oos", eta_summary=boot_summary,
                       use_surrogate_allocator=True, held_out_frac=HELD_OUT_FRAC)
    elapsed = time.time() - t0
    t0_entry = res["theta_by_period"][0]
    print(f"  out-of-sample: in-sample Theta(0)={t0_entry['mean_theta']:+.4f}  "
          f"held-out Theta(0)={t0_entry['mean_theta_held_out']:+.4f} ({elapsed/60:.1f} min)")
    return {"seed": OOS_SEED, "in_sample_theta": t0_entry["mean_theta"],
            "held_out_theta": t0_entry["mean_theta_held_out"], "wall_seconds": elapsed}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["seed", "oos"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    lsm.K_PATHS = K_PATHS
    print(f"K_PATHS={lsm.K_PATHS}, N_PERIODS={lsm.N_PERIODS} ({lsm.N_PERIODS * lsm.PERIOD_DAYS} days), "
          f"allocator=surrogate, mode={args.mode}, seed={args.seed}")

    if args.mode == "seed":
        result = run_seed(args.seed)
    else:
        global OOS_SEED
        OOS_SEED = args.seed
        result = run_out_of_sample()

    out_path = ROOT / f"outputs/theta_surrogate_rigor_{args.mode}_{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({"k_paths": K_PATHS, "mode": args.mode, "seed": args.seed, "result": result},
                   f, indent=2, default=str)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
