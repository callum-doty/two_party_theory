#!/usr/bin/env python3
"""
Mechanism decomposition of Theta(0) (Paper III revision, reviewer-requested,
2026-07-28): isolate how much of the live 98-day Theta(0) comes from genuine
information value (stochastic shocks resolving) versus deterministic
sequencing/crowd-out value (candidate-committee organic spending growth,
known in advance, that a myopic "deploy now" benchmark doesn't account for).

Five scenarios, toggling the three independent channels run_lsm() now
exposes (enable_trickle, enable_stochastic, enable_opponent_reaction):

  A: all off       -- pure static benchmark. Nothing evolves over the
                       horizon, so Theta(0) should be ~0: a direct sanity
                       check on the decomposition machinery itself.
  B: trickle+eta on, stochastic off -- deterministic sequencing value alone
                       (candidate organic growth + opponent's deterministic
                       reaction to it), with no genuine uncertainty at all.
  C: stochastic on, trickle+eta off -- pure information value (idiosyncratic
                       epsilon + G_t resolving), no organic growth channel.
  D: trickle+stochastic on, eta off -- information value plus organic growth,
                       but opponents never react to it.
  E: everything on  -- the full reported model (matches eta_bootstrap_all_cycles
                       in outputs/theta_schedule.json).

Run against the eta_bootstrap_all_cycles calibration only (the primary,
cycle-pooled scenario used elsewhere in the paper) to keep this tractable --
not all three eta brackets, since the point here is isolating mechanisms,
not re-deriving every calibration bracket's Theta.

Output: outputs/theta_mechanism_decomposition.json
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

SCENARIOS = [
    ("A_static_benchmark",       dict(enable_trickle=False, enable_stochastic=False, enable_opponent_reaction=False)),
    ("B_deterministic_sequencing", dict(enable_trickle=True,  enable_stochastic=False, enable_opponent_reaction=True)),
    ("C_pure_information",       dict(enable_trickle=False, enable_stochastic=True,  enable_opponent_reaction=False)),
    ("D_information_plus_growth", dict(enable_trickle=True,  enable_stochastic=True,  enable_opponent_reaction=False)),
    ("E_full_model",             dict(enable_trickle=True,  enable_stochastic=True,  enable_opponent_reaction=True)),
]


def main():
    races = lsm.build_universe(cycle=2026)
    tiers_per_race = [r.cook_rating for r in races]

    print("Fitting eta_bootstrap_all_cycles calibration (shared across all 5 scenarios)...")
    eta_arr_by_path, resid_std_arr_by_path, boot_summary = lsm.bootstrap_eta_resid_paths(
        lsm.BOOTSTRAP_CYCLES, tiers_per_race, lsm.K_PATHS, lsm.RNG)

    results = {}
    for name, flags in SCENARIOS:
        print(f"\n=== Scenario {name}: {flags} ===")
        res = lsm.run_lsm(eta_arr_by_path, resid_std_arr_by_path, name,
                           eta_summary=boot_summary, **flags)
        theta0 = res["theta_by_period"][0]["mean_theta"]
        frac0 = res["theta_by_period"][0]["frac_deploy_now"]
        print(f"  -> Theta(0)={theta0:+.4f} seats, frac_deploy_now(0)={frac0:.3f}")
        results[name] = {"flags": flags, "theta0": theta0, "frac_deploy_now_0": frac0,
                          "theta_by_period": res["theta_by_period"]}

    out_path = ROOT / "outputs/theta_mechanism_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n\n=== Summary: Theta(0) by mechanism ===")
    print(f"{'Scenario':<28} {'trickle':>8} {'stochastic':>11} {'opp_react':>10} {'Theta(0)':>10}")
    for name, flags in SCENARIOS:
        r = results[name]
        print(f"{name:<28} {str(flags['enable_trickle']):>8} {str(flags['enable_stochastic']):>11} "
              f"{str(flags['enable_opponent_reaction']):>10} {r['theta0']:>+10.4f}")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
