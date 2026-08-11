#!/usr/bin/env python3
"""
Reduced-K validation run for Item 4(b-d): confirms the hierarchical-Bayesian
eta scenario (4b) and the robust allocator (4c) both run end-to-end on the
live 2026 universe and produce finite, sane Theta figures, and reports
whether "deploy now" survives this additional scrutiny (4d) -- at K=100
(not the headline K=2000) given wall-clock constraints; a full headline run
is a natural follow-up, not done here.

Output: outputs/theta_item4bcd_validation.json
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import solve_bellman_lsm as lsm
from backtest.data.universe import build_universe
from backtest.estimation import eta_hierarchical as eh

K_PATHS_VALIDATION = 100
RNG = np.random.default_rng(20260807)


def main() -> None:
    races = build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]
    n = len(races)

    lsm.K_PATHS = K_PATHS_VALIDATION
    results = {}

    # --- 4(b): hierarchical Bayesian eta scenario ---
    print("=== Fitting hierarchical eta model (Stage 1: per-cell OLS; Stage 2: MixedLM) ===")
    cycles = [2012, 2014, 2016, 2018, 2020, 2022, 2024]
    per_cell = eh.fit_per_cell_eta(cycles, lsm.build_period_panel, lsm.build_delta_panel, lsm.TIERS)
    fit = eh.fit_hierarchical_eta(per_cell)
    print(f"  mu_global={fit['mu_global']:.4f}, tier_var={fit['tier_var']:.6f}, "
          f"cycle_var={fit['cycle_var']:.6f}, resid_var={fit['resid_var']:.6f}")

    eta_arr_by_path = eh.posterior_predictive_eta_draws(fit, tiers_per_race, K_PATHS_VALIDATION, RNG)
    # resid_std: hybrid per module docstrings' flagged open decision -- use
    # each tier's own Stage-1 per-cell mean resid_std (real, cycle-averaged
    # estimation noise), not the hierarchical model's resid_var (which is
    # noise in the POINT ESTIMATE across cells, a different quantity).
    tier_resid_std = per_cell.groupby("tier")["resid_std"].mean().to_dict()
    resid_std_arr_by_path = np.tile(
        np.array([tier_resid_std.get(t, 0.0) for t in tiers_per_race]), (K_PATHS_VALIDATION, 1)
    )

    print("=== Running eta_hierarchical_bayesian scenario (surrogate allocator, K=%d) ===" % K_PATHS_VALIDATION)
    res_hier = lsm.run_lsm(
        eta_arr_by_path, resid_std_arr_by_path, "eta_hierarchical_bayesian",
        eta_summary={"mu_global": fit["mu_global"], "tier_var": fit["tier_var"],
                     "cycle_var": fit["cycle_var"], "resid_var": fit["resid_var"]},
        use_surrogate_allocator=True,
    )
    results["eta_hierarchical_bayesian"] = res_hier

    # --- 4(c): robust allocator, on the existing eta_bootstrap_all_cycles scenario ---
    print("=== Running eta_bootstrap_all_cycles scenario with the ROBUST allocator (K=%d) ===" % K_PATHS_VALIDATION)
    eta_arr_boot, resid_std_boot, boot_summary = lsm.bootstrap_eta_resid_paths(
        lsm.BOOTSTRAP_CYCLES, tiers_per_race, K_PATHS_VALIDATION, RNG)
    with open(Path(__file__).parent.parent / "data/processed/eta_uncertainty.json") as f:
        eta_uncertainty_by_tier = json.load(f)
    try:
        res_robust = lsm.run_lsm(
            eta_arr_boot, resid_std_boot, "eta_bootstrap_all_cycles_robust_allocator",
            eta_summary=boot_summary, use_robust_allocator=True,
            eta_uncertainty_by_tier=eta_uncertainty_by_tier,
        )
        results["eta_bootstrap_all_cycles_robust_allocator"] = res_robust
    except RuntimeError as e:
        # Genuine finding, not a bug to hide: robust.py's post-hoc
        # monotonicity check (module docstring) is designed to raise rather
        # than return a silently-wrong "robust" answer when its max_D
        # min_eta -> single-call reduction doesn't hold. It fired on this
        # live run -- reported here honestly rather than caught and
        # discarded.
        print(f"\n*** Robust allocator's monotonicity check FAILED on the live 2026 universe: {e}")
        results["eta_bootstrap_all_cycles_robust_allocator"] = {
            "status": "monotonicity_check_failed", "error": str(e),
        }

    out_path = Path(__file__).parent.parent / "outputs/theta_item4bcd_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")

    print("\n=== Summary: does 'deploy now' survive? ===")
    for label, res in results.items():
        if res.get("status") == "monotonicity_check_failed":
            print(f"  {label}: FAILED (monotonicity check) -- {res['error'][:80]}...")
            continue
        t0 = res["theta_by_period"][0]
        print(f"  {label}: Theta(0)={t0['mean_theta']:+.4f}, frac_deploy_now={t0['frac_deploy_now']:.3f}, "
              f"allocator={res['allocator']}")


if __name__ == "__main__":
    main()
