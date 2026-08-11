#!/usr/bin/env python3
"""
Estimate the opponent-reaction function eta(tier) (Paper III Section 4).

eta(tier) answers: when Democratic-aligned IE groups increase spending in a
district, how much do Republican-aligned IE groups subsequently increase
their own IE spending in that same district?

    delta_R_IE_{i,t} = eta(tier_i) * delta_D_IE_{i,t-1} + u_{i,t}

Scope, stated explicitly (Paper III Section 4.3): only independent-
expenditure spending is date-resolved in this repository (candidate
committee and coordinated-expenditure totals are cycle-cumulative
constants, per dynamic/ledger.py's RealizedSpendCommitmentSource docstring).
This regression therefore measures IE-to-IE reaction, not total-spend
reaction -- a real, legitimate, but narrower quantity than "opponent total
spending reacts to your total spending."

Race fixed effects are applied via within-district demeaning (not dummy
columns -- ~800 districts across two cycles makes a dummy matrix wasteful),
pooling 2022 and 2024 into one panel, with eta allowed to vary by Cook
rating tier via interaction terms.

Output: outputs/eta_reaction_estimates.csv
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest.data.universe import build_universe
from backtest.data.fec import load_ie_transactions_dated
from backtest.dynamic.periods import biweekly_periods
from datetime import date

OUT = Path(__file__).parent.parent / "outputs"

TIERS = ["Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"]


def build_period_panel(cycle: int) -> pd.DataFrame:
    """Return a long panel: district_id, period_index, d_ie_cum, r_ie_cum."""
    races = build_universe(cycle=cycle)
    tiers = {r.district_id: r.cook_rating for r in races}

    txns = load_ie_transactions_dated(cycle)
    periods = biweekly_periods(date(cycle, 1, 1), date(cycle, 11, 8))

    rows = []
    for p in periods:
        cutoff = pd.Timestamp(p.period_date)
        cum = txns[txns["exp_date"] <= cutoff]
        agg = cum.groupby(["district_id", "party"])["amount"].sum().unstack(fill_value=0.0)
        agg = agg.reindex(columns=["D", "R"], fill_value=0.0)
        agg = agg.reset_index()
        agg["period_index"] = p.index
        agg["period_date"] = p.period_date
        rows.append(agg)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.rename(columns={"D": "d_ie_cum", "R": "r_ie_cum"})
    panel["tier"] = panel["district_id"].map(tiers)
    panel = panel[panel["tier"].notna()].copy()   # keep only districts in the modeled universe
    panel["cycle"] = cycle
    return panel


def build_delta_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Convert a cumulative panel to period-over-period deltas, with the
    lagged D-side delta as the regressor and within-race demeaning applied
    per (cycle, district_id) group to sweep out race fixed effects."""
    panel = panel.sort_values(["cycle", "district_id", "period_index"])
    g = panel.groupby(["cycle", "district_id"])
    panel["d_ie_delta"] = g["d_ie_cum"].diff()
    panel["r_ie_delta"] = g["r_ie_cum"].diff()
    panel["d_ie_delta_lag"] = g["d_ie_delta"].shift(1)
    panel = panel.dropna(subset=["d_ie_delta_lag", "r_ie_delta"]).copy()

    # Within-race demeaning (fixed-effects transform) per (cycle, district_id)
    key = ["cycle", "district_id"]
    for col in ["r_ie_delta", "d_ie_delta_lag"]:
        panel[col + "_dm"] = panel[col] - panel.groupby(key)[col].transform("mean")

    return panel


def fit_tiered_eta(panel: pd.DataFrame) -> pd.DataFrame:
    """OLS of demeaned r_ie_delta on demeaned d_ie_delta_lag interacted with
    tier dummies, no intercept (already demeaned). Returns one eta per tier."""
    tier_dummies = pd.get_dummies(panel["tier"], prefix="tier").astype(float)
    X_cols = {}
    for tier in TIERS:
        col = f"tier_{tier}"
        if col in tier_dummies.columns:
            X_cols[f"eta_{tier}"] = tier_dummies[col] * panel["d_ie_delta_lag_dm"]
    X = pd.DataFrame(X_cols)
    y = panel["r_ie_delta_dm"]

    mask = X.abs().sum(axis=1) > 0   # drop rows with zero regressor (no tier match / no variation)
    fit = sm.OLS(y[mask], X[mask]).fit(cov_type="HC3")

    rows = []
    n_by_tier = panel["tier"].value_counts()
    for tier in TIERS:
        name = f"eta_{tier}"
        if name in fit.params:
            rows.append({
                "tier": tier,
                "n_obs": int((panel["tier"] == tier).sum()),
                "eta": float(fit.params[name]),
                "se": float(fit.bse[name]),
                "p_value": float(fit.pvalues[name]),
            })
    result = pd.DataFrame(rows)
    return result, fit


def main():
    print("Building period panels (2022, 2024)...")
    panels = [build_period_panel(c) for c in (2022, 2024)]
    full = pd.concat(panels, ignore_index=True)
    print(f"  cumulative-panel rows: {len(full)}")

    delta = build_delta_panel(full)
    print(f"  delta-panel rows (post-lag, non-null): {len(delta)}")
    print(f"  tier counts:\n{delta['tier'].value_counts()}")

    result, fit = fit_tiered_eta(delta)
    print("\n=== eta(tier) estimates ===")
    print(result.to_string(index=False))

    # Pooled (scalar) eta for comparison, same demeaned-regression approach
    pooled_fit = sm.OLS(delta["r_ie_delta_dm"], delta[["d_ie_delta_lag_dm"]]).fit(cov_type="HC3")
    pooled_eta = float(pooled_fit.params.iloc[0])
    pooled_se = float(pooled_fit.bse.iloc[0])
    print(f"\nPooled scalar eta (for comparison to current hand-set 0/0.5/1.0): "
          f"{pooled_eta:.3f} (SE={pooled_se:.3f})")

    result.to_csv(OUT / "eta_reaction_estimates.csv", index=False)
    print(f"\nSaved -> {OUT / 'eta_reaction_estimates.csv'}")


if __name__ == "__main__":
    main()
