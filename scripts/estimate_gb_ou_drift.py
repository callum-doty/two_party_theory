#!/usr/bin/env python3
"""
Fit an Ornstein-Uhlenbeck model WITH drift to the pooled four-cycle
historical generic-ballot series (Paper III docs/theta_followup_plan.md
Section 2.1-2.2 -- "guess B", tested here as a prerequisite check before
deciding whether solve_bellman_lsm.py's G_t process needs a drift term).

Section 5.3 already found std/sqrt(days) essentially flat from 30-270 days
(0.183-0.201) and only declining past ~365 days -- evidence the volatility
term structure is RW-like at the ~110-day horizon relevant to today's
Theta(0) decision. This script checks the SEPARATE question: does the
level of G_t (not just its volatility) show mean-reverting drift, and if
so, does that drift matter at the live horizon?

Discretized OU regression: G_{t+dt} - G_t = kappa*dt*(Gbar - G_t) + noise,
i.e. regress delta_G on G_t (with a constant): delta_G = a + b*G_t + noise,
b = -kappa*dt, a = kappa*dt*Gbar => Gbar = -a/b, kappa = -b/dt.

Output: outputs/gb_ou_drift_fit.json
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from backtest import config
from estimate_gb_volatility import load_historical_series

ROOT = Path(__file__).parent.parent

# Single source of truth: config.yaml (generic ballot) and
# data/processed/live_2026_state.json (today/election day), written by
# scripts/plot_2026_live_allocation.py -- previously independent hardcoded
# literals here (Paper III audit, 2026-07-16).
TODAY_GB = config.generic_ballot_for_cycle(2026)
with open(ROOT / "data/processed/live_2026_state.json") as _f:
    _live_state = json.load(_f)
DAYS_REMAINING = _live_state["days_remaining"]
DT_DAYS = float(config.period_days())   # regression lag, matches the biweekly period grid used elsewhere


def fit_ou_drift(all_series: dict[str, pd.Series], dt_days: float = DT_DAYS) -> dict:
    """Pool delta_G ~ G_t across all cycles at lag dt_days, recover kappa, Gbar."""
    g_vals, dg_vals = [], []
    for name, s in all_series.items():
        s = s.asfreq("D").interpolate()
        lagged = s.shift(int(dt_days))
        delta = s - lagged
        df = pd.DataFrame({"g_lag": lagged, "delta_g": delta}).dropna()
        g_vals.append(df["g_lag"].values)
        dg_vals.append(df["delta_g"].values)

    g_all = np.concatenate(g_vals)
    dg_all = np.concatenate(dg_vals)

    X = sm.add_constant(g_all)
    fit = sm.OLS(dg_all, X).fit(cov_type="HC3")
    a, b = float(fit.params[0]), float(fit.params[1])
    se_a, se_b = float(fit.bse[0]), float(fit.bse[1])
    p_b = float(fit.pvalues[1])

    kappa = -b / dt_days
    g_bar = -a / b if abs(b) > 1e-12 else float("nan")

    return {
        "n_obs": int(len(g_all)), "dt_days": dt_days,
        "a": a, "b": b, "se_a": se_a, "se_b": se_b, "p_value_b": p_b,
        "kappa_per_day": kappa, "g_bar": g_bar, "r_squared": float(fit.rsquared),
    }


def sanity_check_historical_cycles(hist: dict[int, pd.Series], today_gb: float, tol: float = 1.5) -> list[dict]:
    """For each historical cycle, find days where that cycle's trajectory
    was within `tol` points of today's D+5.02, and report what happened
    between that point and Election Day in that cycle. Illustrative only
    (4 cycles, not a powered test) -- reported as such per the plan."""
    election_days = {2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08", 2024: "2024-11-05"}
    rows = []
    for cycle, s in hist.items():
        s = s.asfreq("D").interpolate()
        elec = pd.Timestamp(election_days[cycle])
        near = s[(s - today_gb).abs() <= tol]
        near = near[near.index <= elec]
        if near.empty:
            rows.append({"cycle": cycle, "found_match": False})
            continue
        # earliest qualifying date at least 60 days before election, for a meaningful forward window
        candidates = near[near.index <= elec - pd.Timedelta(days=60)]
        match_date = candidates.index.min() if not candidates.empty else near.index.min()
        g_at_match = float(s.asof(match_date))
        g_at_election = float(s.asof(elec))
        rows.append({
            "cycle": cycle, "found_match": True,
            "match_date": str(match_date.date()), "g_at_match": g_at_match,
            "days_to_election": int((elec - match_date).days),
            "g_at_election": g_at_election,
            "change_to_election": g_at_election - g_at_match,
        })
    return rows


def main():
    hist = load_historical_series()
    all_series = {f"{c}_538": s for c, s in hist.items()}

    print("=== OU-with-drift fit, pooled across 4 historical cycles ===")
    fit = fit_ou_drift(all_series)
    print(f"  n={fit['n_obs']}, dt={fit['dt_days']:.0f}d")
    print(f"  a={fit['a']:+.4f} (se={fit['se_a']:.4f}), b={fit['b']:+.5f} (se={fit['se_b']:.5f}, p={fit['p_value_b']:.4f})")
    print(f"  kappa={fit['kappa_per_day']:.5f}/day (tau={1/fit['kappa_per_day']:.1f}d), Gbar={fit['g_bar']:+.2f}, R2={fit['r_squared']:.4f}")

    gap = TODAY_GB - fit["g_bar"]
    implied_drift = fit["kappa_per_day"] * (fit["g_bar"] - TODAY_GB) * DAYS_REMAINING
    print(f"\n  Today's live GB (D+{TODAY_GB}) vs fitted Gbar (D+{fit['g_bar']:.2f}): gap = {gap:+.2f}")
    print(f"  Implied E[delta_G] over remaining {DAYS_REMAINING} days = kappa*(Gbar-G_today)*days = {implied_drift:+.3f} points")

    print("\n=== Sanity check: historical cycles at a similarly favorable GB level ===")
    sanity = sanity_check_historical_cycles(hist, TODAY_GB)
    for r in sanity:
        if r["found_match"]:
            print(f"  {r['cycle']}: matched {r['match_date']} (G={r['g_at_match']:+.2f}), "
                  f"{r['days_to_election']}d to Election Day -> G_election={r['g_at_election']:+.2f} "
                  f"(change={r['change_to_election']:+.2f})")
        else:
            print(f"  {r['cycle']}: no day within tolerance of D+{TODAY_GB}")

    out = {
        "ou_drift_fit": fit,
        "today_gb": TODAY_GB, "days_remaining": DAYS_REMAINING,
        "gap_today_vs_gbar": gap, "implied_drift_points": implied_drift,
        "historical_sanity_check": sanity,
    }
    out_path = ROOT / "outputs/gb_ou_drift_fit.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
