#!/usr/bin/env python3
"""
Decompose the 2026 live +7.9-seat gain (decompose_retrospective_gain_2026_live.py)
into an exact per-race contribution, then aggregate by observable race
characteristics -- the "feature importance" for the projection.

There's no black-box model here to run SHAP/permutation importance against
-- mu_i is a transparent structural formula. What "feature importance"
means for this kind of model is a decomposition of the *outcome metric*
(expected seats) into exactly which races' reallocation produces it, then
grouped by the same characteristics Paper I already uses throughout
(Cook rating, incumbency) plus the one specific to this live-cycle context:
whether a race is one of the 18 DCCC has actually committed money to yet,
or one of the 416 it hasn't (mirroring Paper I's own selection-vs-intensity
decomposition, Table 5).

This is an EXACT decomposition, not an approximation: for each race,
delta_seats_i = P_win(model allocation_i) - P_win(DCCC-scaled allocation_i),
using the same real floors/opponent spending as
decompose_retrospective_gain_2026_live.py. Summing delta_seats_i across all
434 races reproduces the +7.9 total exactly.

Usage:
    python scripts/decompose_2026_gain_by_race.py

Output: outputs/gain_decomposition_2026_by_race.csv (per-race)
        outputs/gain_decomposition_2026_by_category.csv (grouped)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import config
from backtest.data.universe import build_universe
from backtest.dynamic.ledger import CapitalLedger, RealizedSpendCommitmentSource
from backtest.model.budget import estimate_budget_2026
from backtest.optimizer.allocator import (
    optimize_nonlinear, _precompute_race_arrays, _p_win_vec,
)
from run_backtest import load_processed_artifacts, build_dummy_factor_model  # type: ignore

BUDGET_2026 = estimate_budget_2026()


def main() -> None:
    _, coef, sigma_model = load_processed_artifacts(config.processed_path())
    races = build_universe(cycle=2026)
    gb = config.generic_ballot_for_cycle(2026)
    factor_model = build_dummy_factor_model(races, gb)
    cov_matrix = factor_model.race_covariance()

    as_of = datetime.now(timezone.utc).date()
    src = RealizedSpendCommitmentSource(cycle=2026, party="D")
    committed = src.committed_capital(0, as_of, races)
    committed_total = sum(committed.values())

    ledger = CapitalLedger(
        period=0, total_budget=BUDGET_2026, committed_by_race=committed,
        committed_total=committed_total, deployable_total=BUDGET_2026 - committed_total,
    )
    races_with_floor = ledger.apply_to_races(races)
    result = optimize_nonlinear(
        races_with_floor, coef, sigma_model, budget=BUDGET_2026, cov_matrix=cov_matrix,
        gamma=0.0, cap_fraction=0.15, party_budget=ledger.deployable_total,
    )
    floor = ledger.deployable_floor_for(races)
    committed_arr = np.array([committed.get(r.district_id, 0.0) for r in races])
    model_total_party = (result.allocations - floor) + committed_arr
    dccc_scaled_party = committed_arr * (BUDGET_2026 / committed_total)

    # Same per-race Phi(mu/sigma) evaluation nonlinear_expected_seats_at_party_dollars
    # sums -- computed here per-race instead of summed, so the total gain can be
    # attributed to individual races and groups.
    arrays = _precompute_race_arrays(races, coef, sigma_model, eta=0.0)
    p_win_model = _p_win_vec(model_total_party, arrays)
    p_win_dccc = _p_win_vec(dccc_scaled_party, arrays)
    delta = p_win_model - p_win_dccc

    total_gain = float(delta.sum())
    print(f"Total gain (sum of per-race deltas, should match the earlier +7.9 exactly): "
          f"{total_gain:+.3f}\n")

    is_committed = np.array([committed.get(r.district_id, 0.0) > 0 for r in races])

    df = pd.DataFrame({
        "district_id": [r.district_id for r in races],
        "cook_rating": [r.cook_rating for r in races],
        "incumb_status": [r.incumb_status for r in races],
        "pvi": [r.pvi for r in races],
        "is_currently_committed": is_committed,
        "model_party_dollars": model_total_party,
        "dccc_scaled_party_dollars": dccc_scaled_party,
        "delta_seats": delta,
    }).sort_values("delta_seats", ascending=False).reset_index(drop=True)

    race_path = config.outputs_path() / "gain_decomposition_2026_by_race.csv"
    df.to_csv(race_path, index=False)
    print(f"Saved: {race_path}")

    print("\nTop 10 races driving the gain (positive = model favors, DCCC-scaled doesn't):")
    print(df.head(10)[["district_id", "cook_rating", "incumb_status", "is_currently_committed",
                        "delta_seats"]].to_string(index=False))
    print("\nBottom 5 (DCCC-scaled favors, model doesn't):")
    print(df.tail(5)[["district_id", "cook_rating", "incumb_status", "is_currently_committed",
                       "delta_seats"]].to_string(index=False))

    # ── Group by category: Cook rating, incumbency, committed-status ───────
    groups = []
    for group_col, group_label in [("cook_rating", "cook_rating"),
                                     ("incumb_status", "incumb_status")]:
        g = df.groupby(group_col)["delta_seats"].agg(["sum", "count"]).reset_index()
        g["dimension"] = group_label
        g = g.rename(columns={group_col: "category"})
        groups.append(g[["dimension", "category", "sum", "count"]])

    committed_g = df.assign(
        category=np.where(df["is_currently_committed"], "18 currently-committed races",
                           "416 zero-committed races")
    ).groupby("category")["delta_seats"].agg(["sum", "count"]).reset_index()
    committed_g["dimension"] = "committed_status"
    groups.append(committed_g[["dimension", "category", "sum", "count"]])

    grouped_df = pd.concat(groups, ignore_index=True).rename(columns={"sum": "total_delta_seats"})
    cat_path = config.outputs_path() / "gain_decomposition_2026_by_category.csv"
    grouped_df.to_csv(cat_path, index=False)
    print(f"\nSaved: {cat_path}")
    print("\n" + grouped_df.to_string(index=False))


if __name__ == "__main__":
    main()
