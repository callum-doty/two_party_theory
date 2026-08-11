#!/usr/bin/env python3
"""
Apply the fitted DCCC-forecast model (fit_dccc_forecast_model.py) to the
live 2026 cycle, producing the reviewer's actually-requested comparison:

    E[Seats | model-optimal, X_t]  vs.  E[Seats | forecasted-DCCC, X_t]

replacing every earlier 2026 comparison in this investigation, which used
a proportional scale-up of DCCC's thin 18-race current pattern as the
baseline instead of a genuine forecast of where DCCC will end up.

Usage:
    python scripts/apply_dccc_forecast_2026.py
"""
from __future__ import annotations

import json
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
from backtest.optimizer.allocator import optimize_nonlinear, nonlinear_expected_seats_at_party_dollars
from run_backtest import load_processed_artifacts, build_dummy_factor_model  # type: ignore
from build_dccc_forecast_training_data import COOK_ORDINAL  # type: ignore

BUDGET_2026 = estimate_budget_2026()


def main() -> None:
    model_path = config.processed_path() / "dccc_forecast_model.json"
    if not model_path.exists():
        raise SystemExit(f"Missing {model_path} -- run fit_dccc_forecast_model.py first.")
    model = json.load(open(model_path))
    features = model["features"]

    _, coef, sigma_model = load_processed_artifacts(config.processed_path())
    races = build_universe(cycle=2026)
    gb = config.generic_ballot_for_cycle(2026)
    factor_model = build_dummy_factor_model(races, gb)
    cov_matrix = factor_model.race_covariance()

    as_of = datetime.now(timezone.utc).date()
    src = RealizedSpendCommitmentSource(cycle=2026, party="D")
    committed = src.committed_capital(0, as_of, races)
    committed_total = sum(committed.values())

    # ── Build today's feature vector per race, matching training exactly ───
    rows = []
    for r in races:
        rows.append({
            "district_id": r.district_id,
            "pvi": r.pvi, "abs_pvi": abs(r.pvi),
            "is_challenger": float(r.incumb_status == "Challenger"),
            "is_open": float(r.incumb_status == "Open"),
            "cook_ordinal": COOK_ORDINAL.get(r.cook_rating, 0),
            "cand_ratio_t": r.cand_d_total / BUDGET_2026,
            "r_ratio_t": r.r_total / BUDGET_2026,
            "generic_ballot": gb,
        })
    X = pd.DataFrame(rows)

    def linpred(params: dict) -> np.ndarray:
        s = np.full(len(X), params["const"])
        for f in features:
            s = s + params[f] * X[f].values
        return s

    logit_score = linpred(model["selection_params"])
    p_funded = 1.0 / (1.0 + np.exp(-logit_score))
    log_share = linpred(model["intensity_params"])
    raw_score = p_funded * np.exp(log_share)
    predicted_share = raw_score / raw_score.sum()
    forecasted_dollars = predicted_share * BUDGET_2026

    forecast_seats = nonlinear_expected_seats_at_party_dollars(races, coef, sigma_model, forecasted_dollars)
    print(f"Forecasted-DCCC E[Seats] (genuine forecast, {len(model['trained_on_cycles'])}-cycle-trained "
          f"model): {forecast_seats:.3f}")

    # ── Model-optimal side: identical setup to decompose_retrospective_gain_2026_live.py ──
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
    model_seats = nonlinear_expected_seats_at_party_dollars(races, coef, sigma_model, model_total_party)
    print(f"Model-optimal E[Seats]: {model_seats:.3f}")

    gain_vs_forecast = model_seats - forecast_seats
    print(f"\nTHE REVIEWER'S REQUESTED COMPARISON:")
    print(f"  E[Seats | model-optimal, X_t={as_of}]     = {model_seats:.3f}")
    print(f"  E[Seats | forecasted-DCCC, X_t={as_of}]   = {forecast_seats:.3f}")
    print(f"  Gain = {gain_vs_forecast:+.3f} seats")

    # ── For comparison, the old (naive scaled-pattern) baseline this replaces ──
    dccc_scaled_party = committed_arr * (BUDGET_2026 / committed_total)
    dccc_scaled_seats = nonlinear_expected_seats_at_party_dollars(races, coef, sigma_model, dccc_scaled_party)
    print(f"\n  For comparison -- old naive scaled-18-race baseline: {dccc_scaled_seats:.3f} "
          f"expected seats, gain={model_seats - dccc_scaled_seats:+.3f} "
          f"(this is the figure being replaced)")

    df = pd.DataFrame({
        "district_id": [r.district_id for r in races],
        "cook_rating": [r.cook_rating for r in races],
        "predicted_dccc_share": predicted_share,
        "predicted_dccc_dollars": forecasted_dollars,
        "model_optimal_dollars": model_total_party,
    })
    out_dir = config.outputs_path()
    df.to_csv(out_dir / "dccc_forecast_2026_allocation.csv", index=False)

    summary = {
        "as_of": as_of.isoformat(),
        "model_optimal_expected_seats": model_seats,
        "forecasted_dccc_expected_seats": forecast_seats,
        "gain_vs_forecast": gain_vs_forecast,
        "old_scaled_baseline_expected_seats": dccc_scaled_seats,
        "gain_vs_old_scaled_baseline": model_seats - dccc_scaled_seats,
    }
    with open(out_dir / "dccc_forecast_2026_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir / 'dccc_forecast_2026_allocation.csv'}")
    print(f"Saved: {out_dir / 'dccc_forecast_2026_summary.json'}")


if __name__ == "__main__":
    main()
