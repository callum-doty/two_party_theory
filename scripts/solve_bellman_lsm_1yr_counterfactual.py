#!/usr/bin/env python3
"""
Counterfactual requested while discussing the bootstrap scenario's
frac_deploy_now=91.5% at t=0: does the deploy probability change if the
Bellman process starts ~1 year before Election Day instead of the live
~98 days?

IMPORTANT design choice, stated plainly: this is NOT a reconstruction of
what the 2026 race universe actually looked like a year ago -- that would
require a dated candidate-committee floor panel, which docs/
theta_followup_plan.md Section 0.1.1 already established does not exist
anywhere in this repository (candidate_disbursements_{cycle}.csv is
cycle-cumulative-final only). Attempting a "real" 1-year-ago snapshot
would silently confound two things changing at once -- a different
starting spending level AND a different remaining horizon -- making it
impossible to isolate which one drives any change in the result.

Instead, this holds TODAY's actual observed starting state (candidate
floors, R_i,t, sigma_i, the structural mu_i inputs -- everything
build_universe(cycle=2026) returns) exactly fixed, and ONLY changes the
artificial "days until Election Day" parameter from ~98 to 364 (26
periods instead of 7). This isolates the pure time-horizon effect the
discussion was actually about, at the cost of not being a literal
history -- a deliberate trade-off, not an oversight.

Implementation: monkey-patches solve_bellman_lsm's module-level TODAY/
N_PERIODS globals before calling its own run_lsm()/tile_single_cycle()/
bootstrap_eta_resid_paths()/fit_eta_and_resid() -- every reference to
N_PERIODS inside run_lsm() is a bare global lookup resolved at call time,
so this works without touching scripts/solve_bellman_lsm.py itself (already
tested and re-run several times this session; not worth risking a second
code path).

Caveat also worth stating up front (Paper III Section 5.3): sigma_G's
random-walk approximation and the epsilon-decay proxy were validated as
good specifically over a 3-9 month window; a 12-month run sits at the
edge of, not comfortably inside, that validated range.

Output: outputs/theta_schedule_1yr_counterfactual.json
"""

from __future__ import annotations
import json
from datetime import timedelta
from pathlib import Path

import numpy as np

import solve_bellman_lsm as lsm

ROOT = Path(__file__).parent.parent
COUNTERFACTUAL_DAYS_REMAINING = 364   # 26 periods at PERIOD_DAYS=14, vs. live ~98d/7 periods


def main():
    original_today, original_n_periods = lsm.TODAY, lsm.N_PERIODS
    lsm.TODAY = lsm.ELECTION_DAY - timedelta(days=COUNTERFACTUAL_DAYS_REMAINING)
    lsm.N_PERIODS = max(1, (lsm.ELECTION_DAY - lsm.TODAY).days // lsm.PERIOD_DAYS)

    print(f"1-YEAR COUNTERFACTUAL: N_PERIODS={lsm.N_PERIODS} ({lsm.N_PERIODS * lsm.PERIOD_DAYS} days) "
          f"vs. live N_PERIODS={original_n_periods} ({original_n_periods * lsm.PERIOD_DAYS} days)")
    print("Starting financial state (floors, R_i,t, sigma_i) held IDENTICAL to the live run --")
    print("only the artificial remaining-horizon parameter changes. See module docstring.\n")

    races = lsm.build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]

    results = {}
    for label, fit_cycle in [("eta_fit_2022", 2022), ("eta_fit_2024", 2024)]:
        print(f"=== {label} (1yr counterfactual) ===")
        eta_by_tier, resid_std_by_tier = lsm.fit_eta_and_resid(fit_cycle)
        eta_arr_by_path, resid_std_arr_by_path = lsm.tile_single_cycle(
            eta_by_tier, resid_std_by_tier, tiers_per_race, lsm.K_PATHS)
        res = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path, label,
                           eta_summary={"single_cycle_fit": eta_by_tier})
        results[label] = res

    print("=== eta_bootstrap_all_cycles (1yr counterfactual) ===")
    eta_arr_by_path, resid_std_arr_by_path, boot_summary = lsm.bootstrap_eta_resid_paths(
        lsm.BOOTSTRAP_CYCLES, tiers_per_race, lsm.K_PATHS, lsm.RNG)
    res = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path, "eta_bootstrap_all_cycles",
                       eta_summary=boot_summary)
    results["eta_bootstrap_all_cycles"] = res

    out_path = ROOT / "outputs/theta_schedule_1yr_counterfactual.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")

    lsm.TODAY, lsm.N_PERIODS = original_today, original_n_periods   # restore, in case anything re-imports this module


if __name__ == "__main__":
    main()
