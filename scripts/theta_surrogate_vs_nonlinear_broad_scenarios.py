#!/usr/bin/env python3
"""
Paper III revision, next-step item 5 (continued): every existing exact
nonlinear-throughout vs. surrogate-throughout paired comparison (the
K=15/30/50/100 progression, Tables 13e/13g, and the surrogate's own
Theta-level validation reported in Section 8.9's item-(5) paragraph) was run
ONLY on the eta_bootstrap_all_cycles scenario. This script extends the same
paired, common-random-numbers methodology (scripts/theta_nonlinear_throughout.py)
to the two single-cycle brackets (eta_fit_2022, eta_fit_2024), at K=15, and
additionally reports PER-PATH agreement in the deploy-vs-wait classification
at t=0 (not just agreement of the aggregate mean Theta(0)) -- the "agreement
in deploy-versus-wait classification" the broader validation asks for.

Pairing methodology identical to theta_nonlinear_throughout.py: lsm.RNG is
reset to an identical seed immediately before each of the two run_lsm()
calls, so both allocators see bit-identical simulated d_paths, r_paths,
mu_paths, eps_cum, and g_paths -- isolating the allocator's own effect.

K=15 (not the headline K=2,000) because optimize_nonlinear() costs 40s to
over an hour per call; this is a targeted robustness check, not a
precision-matched replication of the surrogate headline (which is already
established at K=2,000 in Table 13i for these two scenarios).

Output: outputs/theta_surrogate_vs_nonlinear_{scenario}.json
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

import solve_bellman_lsm as lsm

ROOT = Path(__file__).parent.parent
K_PATHS_REDUCED = 15
STATE_SEED = 20260803


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["eta_fit_2022", "eta_fit_2024"], required=True)
    args = ap.parse_args()

    lsm.K_PATHS = K_PATHS_REDUCED
    print(f"scenario={args.scenario} K_PATHS={lsm.K_PATHS}, N_PERIODS={lsm.N_PERIODS} "
          f"({lsm.N_PERIODS * lsm.PERIOD_DAYS} days)")

    races = lsm.build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]
    fit_cycle = 2022 if args.scenario == "eta_fit_2022" else 2024
    eta_by_tier, resid_std_by_tier = lsm.fit_eta_and_resid(fit_cycle)
    eta_arr_by_path, resid_std_arr_by_path = lsm.tile_single_cycle(
        eta_by_tier, resid_std_by_tier, tiers_per_race, lsm.K_PATHS)
    eta_summary = {"single_cycle_fit": eta_by_tier}

    print("\n=== Surrogate-throughout (paired) ===")
    lsm.RNG = np.random.default_rng(STATE_SEED)
    t0 = time.time()
    res_surrogate = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path,
                                 f"{args.scenario}_surrogate", eta_summary=eta_summary,
                                 use_surrogate_allocator=True, return_period0_action=True)
    elapsed_surrogate = time.time() - t0
    print(f"  -> surrogate-throughout wall time: {elapsed_surrogate:.1f}s")

    print("\n=== Nonlinear-throughout, SAME state-path RNG seed (paired) ===")
    lsm.RNG = np.random.default_rng(STATE_SEED)
    t0 = time.time()
    res_nonlinear = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path,
                                 f"{args.scenario}_nonlinear", eta_summary=eta_summary,
                                 use_nonlinear_allocator=True, return_period0_action=True)
    elapsed_nonlinear = time.time() - t0
    print(f"  -> nonlinear-throughout wall time: {elapsed_nonlinear / 3600:.2f} hours")

    theta0_sur = res_surrogate["theta_by_period"][0]["mean_theta"]
    theta0_nl = res_nonlinear["theta_by_period"][0]["mean_theta"]
    action_sur = np.array(res_surrogate["period0_action_deploy_now"])
    action_nl = np.array(res_nonlinear["period0_action_deploy_now"])
    agreement = float(np.mean(action_sur == action_nl))

    print(f"\nTheta(0) surrogate-throughout: {theta0_sur:+.4f}  (frac_deploy={np.mean(action_sur):.3f})")
    print(f"Theta(0) nonlinear-throughout: {theta0_nl:+.4f}  (frac_deploy={np.mean(action_nl):.3f})")
    print(f"Per-path deploy/wait classification agreement at t=0: {agreement:.3f} "
          f"({int(agreement * K_PATHS_REDUCED)}/{K_PATHS_REDUCED} paths)")

    out = {
        "scenario": args.scenario, "k_paths": K_PATHS_REDUCED, "paired": True, "state_seed": STATE_SEED,
        "surrogate_throughout": res_surrogate, "nonlinear_throughout": res_nonlinear,
        "theta0_surrogate": theta0_sur, "theta0_nonlinear": theta0_nl,
        "period0_classification_agreement": agreement,
        "surrogate_wall_seconds": elapsed_surrogate, "nonlinear_wall_seconds": elapsed_nonlinear,
    }
    out_path = ROOT / f"outputs/theta_surrogate_vs_nonlinear_{args.scenario}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
