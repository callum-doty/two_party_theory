#!/usr/bin/env python3
"""
Builds two uncertainty-aware alternatives to the current two-point
(eta_fit_2022 / eta_fit_2024) bracketing in scripts/solve_bellman_lsm.py,
using the per-(tier,cycle) point estimates already computed in
outputs/eta_seven_cycle_extension.csv (scripts/reconcile_eta_sigma_g_instability.py,
which found 5 of 7 tiers show statistically significant cycle-to-cycle
variation in eta -- this script turns that finding into something a Monte
Carlo can actually sample from, instead of two hand-picked brackets).

Two methods, chosen for being immediately buildable from data already on
disk without imposing assumptions the data can't support:

1. Non-parametric bootstrap: resample each tier's per-cycle point estimates
   with replacement. No distributional assumption at all -- doesn't impose
   symmetry or thin tails on a parameter whose real data include a sign
   flip (Toss-Up went to -0.22 in 2016).

2. Random-effects meta-analysis (DerSimonian-Laird): treats each cycle's
   per-tier estimate as one "study" and explicitly separates between-cycle
   variance (tau^2) from each cycle's own estimation noise (se^2) -- the
   correct way to answer "how much does eta really vary cycle-to-cycle,"
   rather than the naive variance of the 7 point estimates, which would
   conflate real variation with sampling noise. Also reports I^2 (the
   share of cross-cycle spread that's real variation, not noise) and a
   predictive SD for a next, unobserved cycle (tau^2 + pooled SE^2) --
   this predictive SD, not the pooled SE alone, is what a Monte Carlo
   draw for "next cycle's eta" should actually sample from.

NOT yet wired into scripts/solve_bellman_lsm.py's Monte Carlo -- this
script only builds and saves the distributions. Replacing the LSM's
two-bracket eta with a per-path draw from these distributions is a
separate, larger follow-on step (would require re-running the ~10-minute
backward induction) and is not done here.

Output: data/processed/eta_uncertainty.json, outputs/eta_bootstrap_distribution.csv,
        printed summary.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
COOK_ORDER = ["Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"]
N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(20260717)


def load_seven_cycle_fits() -> pd.DataFrame:
    path = ROOT / "outputs/eta_seven_cycle_extension.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/reconcile_eta_sigma_g_instability.py first.")
    return pd.read_csv(path)


def bootstrap_distribution(tier_fits: pd.DataFrame, n_draws: int = N_BOOTSTRAP) -> dict:
    """Resample this tier's real per-cycle point estimates with replacement --
    no parametric shape imposed on a distribution we have no basis to assume
    is symmetric or thin-tailed."""
    values = tier_fits["eta"].values
    draws = RNG.choice(values, size=n_draws, replace=True)
    return {
        "n_cycles_observed": int(len(values)),
        "observed_values": [float(v) for v in values],
        "bootstrap_mean": float(np.mean(draws)),
        "bootstrap_sd": float(np.std(draws, ddof=1)),
        "p5": float(np.percentile(draws, 5)),
        "p25": float(np.percentile(draws, 25)),
        "median": float(np.percentile(draws, 50)),
        "p75": float(np.percentile(draws, 75)),
        "p95": float(np.percentile(draws, 95)),
    }


def der_simonian_laird(tier_fits: pd.DataFrame) -> dict:
    """Random-effects meta-analysis. Separates real between-cycle variance
    (tau^2) from each cycle's own estimation noise (se^2) -- the naive
    variance of the 7 point estimates would conflate the two and overstate
    (or understate) true cycle-to-cycle heterogeneity."""
    y = tier_fits["eta"].values
    v = tier_fits["se"].values ** 2
    k = len(y)

    w_fe = 1.0 / v
    y_fe = np.sum(w_fe * y) / np.sum(w_fe)
    q = np.sum(w_fe * (y - y_fe) ** 2)
    c = np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe)
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0

    w_re = 1.0 / (v + tau2)
    y_re = np.sum(w_re * y) / np.sum(w_re)
    se_pooled = np.sqrt(1.0 / np.sum(w_re))
    predictive_sd = np.sqrt(tau2 + se_pooled ** 2)   # for a NEW, unobserved cycle
    i_squared = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0

    return {
        "k_cycles": int(k),
        "fixed_effect_estimate": float(y_fe),
        "q_statistic": float(q),
        "tau_squared": float(tau2),
        "pooled_mean": float(y_re),
        "pooled_se": float(se_pooled),
        "predictive_sd_next_cycle": float(predictive_sd),
        "i_squared": float(i_squared),
    }


def main():
    fits = load_seven_cycle_fits()

    results = {}
    print(f"{'tier':10s} {'k':>2s}  {'bootstrap 90% range':>22s}  {'DL pooled':>10s}  {'tau':>6s}  {'pred SD':>8s}  {'I^2':>5s}")
    for tier in COOK_ORDER:
        tier_fits = fits[fits["tier"] == tier]
        if len(tier_fits) < 2:
            continue
        boot = bootstrap_distribution(tier_fits)
        dl = der_simonian_laird(tier_fits)
        results[tier] = {"bootstrap": boot, "random_effects": dl}
        print(f"{tier:10s} {dl['k_cycles']:2d}  "
              f"[{boot['p5']:+.2f}, {boot['p95']:+.2f}]{'':>6s}  "
              f"{dl['pooled_mean']:+.3f}     "
              f"{dl['tau_squared'] ** 0.5:.3f}   "
              f"{dl['predictive_sd_next_cycle']:.3f}    "
              f"{dl['i_squared']:.0%}")

    with open(ROOT / "data/processed/eta_uncertainty.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> data/processed/eta_uncertainty.json")

    rows = []
    for tier, r in results.items():
        for v in r["bootstrap"]["observed_values"]:
            rows.append({"tier": tier, "observed_eta": v})
    pd.DataFrame(rows).to_csv(ROOT / "outputs/eta_bootstrap_distribution.csv", index=False)
    print(f"Saved -> outputs/eta_bootstrap_distribution.csv")

    print("\n=== I^2 read: what fraction of the cross-cycle spread is REAL variation vs. noise? ===")
    for tier, r in results.items():
        dl = r["random_effects"]
        verdict = "mostly REAL variation" if dl["i_squared"] > 0.5 else "could still be mostly noise"
        print(f"  {tier}: I^2={dl['i_squared']:.0%} -- {verdict}")

    print("\nNOTE: this builds the distributions only. Wiring a per-path eta draw into")
    print("scripts/solve_bellman_lsm.py's Monte Carlo (replacing the current eta_fit_2022/")
    print("eta_fit_2024 brackets) is a separate follow-on step, not done here.")


if __name__ == "__main__":
    main()
