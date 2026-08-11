#!/usr/bin/env python3
"""
K=2,000 headline confirmation of the eta_hierarchical_bayesian scenario
(docs/theta_followup_plan.md Section 13.3), which was only run at K=100
(validation scale) in scripts/theta_item4bcd_validation.py and found the
first non-degenerate wait/deploy split (44% wait at t=0) anywhere in this
project's history -- every other scenario, at every K, has recommended
deploy-now for ~100% of paths. This resolves whether that split is real or
a K=100 sampling artifact, using the same surrogate-allocator, same-seed
pattern scripts/theta_surrogate_headline.py already established for the
other three scenarios' own K=2,000 confirmation runs.

At K=2,000, N_PERIODS=7: (7+1)*2,000=16,000 surrogate calls, ~0.03s each,
~8-10 minutes wall clock (matches the other three scenarios' own headline
runs).

Output: outputs/theta_hierarchical_headline.json
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
from backtest.estimation import eta_hierarchical as eh

ROOT = Path(__file__).parent.parent
K_HEADLINE = 2000
SEED = 20260716   # matches the other three scenarios' headline seed, for continuity


def main():
    lsm.K_PATHS = K_HEADLINE
    print(f"scenario=eta_hierarchical_bayesian K_PATHS={lsm.K_PATHS} N_PERIODS={lsm.N_PERIODS} "
          f"({lsm.N_PERIODS * lsm.PERIOD_DAYS} days) allocator=surrogate")

    races = lsm.build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]

    print("Fitting hierarchical eta model (Stage 1: per-cell OLS; Stage 2: MixedLM)...")
    cycles = [2012, 2014, 2016, 2018, 2020, 2022, 2024]
    per_cell = eh.fit_per_cell_eta(cycles, lsm.build_period_panel, lsm.build_delta_panel, lsm.TIERS)
    fit = eh.fit_hierarchical_eta(per_cell)
    print(f"  mu_global={fit['mu_global']:.4f}, tier_var={fit['tier_var']:.6f}, "
          f"cycle_var={fit['cycle_var']:.6f}, resid_var={fit['resid_var']:.6f}")

    draw_rng = np.random.default_rng(SEED)
    eta_arr_by_path = eh.posterior_predictive_eta_draws(fit, tiers_per_race, K_HEADLINE, draw_rng)
    tier_resid_std = per_cell.groupby("tier")["resid_std"].mean().to_dict()
    resid_std_arr_by_path = np.tile(
        np.array([tier_resid_std.get(t, 0.0) for t in tiers_per_race]), (K_HEADLINE, 1)
    )
    eta_summary = {"mu_global": fit["mu_global"], "tier_var": fit["tier_var"],
                    "cycle_var": fit["cycle_var"], "resid_var": fit["resid_var"],
                    "tier_effects": fit["tier_effects"], "cycle_effects": fit["cycle_effects"]}

    lsm.RNG = np.random.default_rng(SEED)
    t0 = time.time()
    result = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path, "eta_hierarchical_bayesian_headline",
                          eta_summary=eta_summary, use_surrogate_allocator=True)
    elapsed = time.time() - t0

    t0_entry = result["theta_by_period"][0]
    print(f"\neta_hierarchical_bayesian: Theta(0)={t0_entry['mean_theta']:+.4f}  "
          f"frac_deploy_now(0)={t0_entry['frac_deploy_now']:.3f}  "
          f"wall_time={elapsed/60:.1f} min")

    out_path = ROOT / "outputs/theta_hierarchical_headline.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
