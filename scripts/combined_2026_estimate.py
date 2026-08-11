#!/usr/bin/env python3
"""
The combined 2026 estimate: floor-maturity-corrected model-optimal
(Section 8) evaluated against the genuine forecasted-DCCC baseline
(apply_dccc_forecast_2026.py) instead of the naive scaled-18-race pattern
every earlier number in this investigation used. The single most
defensible number this investigation can produce for the live cycle,
combining the two corrections that matter most: a ceiling that doesn't
extrapolate from noise-dominated thin floors, and a baseline that reflects
where DCCC is actually likely to end up rather than a crude scale-up of
its first few committed dollars.

floor_maturity is applied consistently to the model-optimal side (both the
optimizer's decision and its evaluation) and to the forecasted-DCCC side's
evaluation, so neither allocation is scored against a ceiling it wasn't
computed under.

Usage:
    python scripts/combined_2026_estimate.py
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
from backtest.model import ceiling as ceiling_mod
from backtest.optimizer.allocator import optimize_nonlinear, nonlinear_expected_seats_at_party_dollars
from run_backtest import load_processed_artifacts, build_dummy_factor_model  # type: ignore
from build_dccc_forecast_training_data import COOK_ORDINAL  # type: ignore

BUDGET_2026 = estimate_budget_2026()


def main() -> None:
    model_path = config.processed_path() / "dccc_forecast_model.json"
    if not model_path.exists():
        raise SystemExit(f"Missing {model_path} -- run fit_dccc_forecast_model.py first.")
    fmodel = json.load(open(model_path))
    features = fmodel["features"]

    _, coef, sigma_model = load_processed_artifacts(config.processed_path())
    races = build_universe(cycle=2026)
    gb = config.generic_ballot_for_cycle(2026)
    factor_model = build_dummy_factor_model(races, gb)
    cov_matrix = factor_model.race_covariance()
    reference = config.floor_maturity_reference_dollars()

    as_of = datetime.now(timezone.utc).date()
    src = RealizedSpendCommitmentSource(cycle=2026, party="D")
    committed = src.committed_capital(0, as_of, races)
    committed_total = sum(committed.values())

    # ── Forecasted-DCCC baseline (genuine forecast, not scaled pattern) ────
    X = pd.DataFrame([{
        "pvi": r.pvi, "abs_pvi": abs(r.pvi),
        "is_challenger": float(r.incumb_status == "Challenger"),
        "is_open": float(r.incumb_status == "Open"),
        "cook_ordinal": COOK_ORDINAL.get(r.cook_rating, 0),
        "cand_ratio_t": r.cand_d_total / BUDGET_2026,
        "r_ratio_t": r.r_total / BUDGET_2026,
        "generic_ballot": gb,
    } for r in races])

    def linpred(params: dict) -> np.ndarray:
        s = np.full(len(X), params["const"])
        for f in features:
            s = s + params[f] * X[f].values
        return s

    p_funded = 1.0 / (1.0 + np.exp(-linpred(fmodel["selection_params"])))
    raw_score = p_funded * np.exp(linpred(fmodel["intensity_params"]))
    forecasted_dollars = raw_score / raw_score.sum() * BUDGET_2026

    # ── Model-optimal, floor-maturity-corrected ─────────────────────────────
    ledger = CapitalLedger(
        period=0, total_budget=BUDGET_2026, committed_by_race=committed,
        committed_total=committed_total, deployable_total=BUDGET_2026 - committed_total,
    )
    races_with_floor = ledger.apply_to_races(races)
    floor_maturity_opt = ceiling_mod.maturity(
        np.array([r.cand_d_total for r in races_with_floor]),
        np.array([r.r_total for r in races_with_floor]), reference,
    )
    floor_maturity_eval = ceiling_mod.maturity(
        np.array([r.cand_d_total for r in races]), np.array([r.r_total for r in races]), reference,
    )
    result = optimize_nonlinear(
        races_with_floor, coef, sigma_model, budget=BUDGET_2026, cov_matrix=cov_matrix,
        gamma=0.0, cap_fraction=0.15, party_budget=ledger.deployable_total,
        floor_maturity=floor_maturity_opt,
    )
    floor = ledger.deployable_floor_for(races)
    committed_arr = np.array([committed.get(r.district_id, 0.0) for r in races])
    model_total_party = (result.allocations - floor) + committed_arr

    model_seats = nonlinear_expected_seats_at_party_dollars(
        races, coef, sigma_model, model_total_party, floor_maturity=floor_maturity_eval)
    forecast_seats = nonlinear_expected_seats_at_party_dollars(
        races, coef, sigma_model, forecasted_dollars, floor_maturity=floor_maturity_eval)
    combined_gain = model_seats - forecast_seats

    # ── Every version of this number, for one clear comparison table ───────
    dccc_scaled_party = committed_arr * (BUDGET_2026 / committed_total)
    model_seats_uncorrected = nonlinear_expected_seats_at_party_dollars(races, coef, sigma_model, model_total_party)
    dccc_scaled_seats_uncorrected = nonlinear_expected_seats_at_party_dollars(races, coef, sigma_model, dccc_scaled_party)
    forecast_seats_uncorrected = nonlinear_expected_seats_at_party_dollars(races, coef, sigma_model, forecasted_dollars)

    print("Every version of the 2026 gain estimate, from this investigation's full arc:")
    print(f"  1. Naive baseline, uncorrected ceiling:                        "
          f"{model_seats_uncorrected - dccc_scaled_seats_uncorrected:+.3f}  (the original +7.9)")
    print(f"  2. Naive baseline, floor-maturity-corrected ceiling:           "
          f"{nonlinear_expected_seats_at_party_dollars(races, coef, sigma_model, model_total_party, floor_maturity=floor_maturity_eval) - nonlinear_expected_seats_at_party_dollars(races, coef, sigma_model, dccc_scaled_party, floor_maturity=floor_maturity_eval):+.3f}  (Section 8's +4.2)")
    print(f"  3. Forecast baseline, uncorrected ceiling:                     "
          f"{model_seats_uncorrected - forecast_seats_uncorrected:+.3f}  (this session's forecast-model result)")
    print(f"  4. Forecast baseline, floor-maturity-corrected ceiling:        "
          f"{combined_gain:+.3f}  <-- COMBINED, most defensible current estimate")

    out = {
        "as_of": as_of.isoformat(),
        "naive_baseline_uncorrected": model_seats_uncorrected - dccc_scaled_seats_uncorrected,
        "forecast_baseline_uncorrected": model_seats_uncorrected - forecast_seats_uncorrected,
        "combined_forecast_baseline_maturity_corrected": combined_gain,
        "model_seats_combined": model_seats,
        "forecast_seats_combined": forecast_seats,
    }
    out_path = config.outputs_path() / "combined_2026_estimate.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
