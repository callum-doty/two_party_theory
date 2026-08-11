#!/usr/bin/env python3
"""
Scoped joint uncertainty simulation for the combined 2026 gain estimate
(combined_2026_estimate.py's +0.62 seats), per the reviewer's request for a
+/- range rather than a single point estimate, and the explicit scope
agreement: a full 9-source joint simulation re-solving the optimizer under
every draw is not tractable (each solve takes minutes; hundreds of draws
would take hours). This is the tractable version instead:

Three uncertainty sources, ALL grounded in quantities this project has
already fit or measured directly -- none invented for this script:

  1. Forecast-model parameter uncertainty: bootstrap-resample the 7
     training cycles with replacement, refit the hurdle model on each
     resample (fit_dccc_forecast_model.py's exact procedure), predict a
     fresh forecasted-DCCC allocation for 2026 each draw.
  2. Floor-maturity threshold uncertainty: resample from the 8 empirical
     values already computed in sweep_floor_maturity_reference.py
     (2024/2022 x {p10,p25,p50,p75}) rather than treating the shipped
     $6.9M default as exact.
  3. Generic-ballot uncertainty: today's live GB (D+5.02) is a point
     estimate with real sampling/drift uncertainty. Uses this project's own
     already-fitted term-structure volatility (data/processed/gb_dynamics.
     json's sigma_g_per_sqrt_day=0.184), scaled to today's actual
     days-remaining -- not an invented number.

Explicit, stated approximation: the optimizer's chosen allocation is held
FIXED at the point-estimate solution (re-solving under all ~1000 draws is
not tractable). Only the EVALUATION of that fixed allocation, and of the
bootstrap-refit forecast baseline, varies per draw. This means the
resulting interval understates true uncertainty (a different GB or
maturity threshold would, in reality, also change where the optimizer
puts money, not just how the same allocation scores) -- reported as a
scenario range under this approximation, not a rigorous confidence
interval.

Usage:
    python scripts/simulate_2026_gain_uncertainty.py --n-draws 1000
"""
from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

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
from fit_dccc_forecast_model import FEATURES, _prep, fit_hurdle  # type: ignore

BUDGET_2026 = estimate_budget_2026()


def features_for(races, gb: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "pvi": r.pvi, "abs_pvi": abs(r.pvi),
        "is_challenger": float(r.incumb_status == "Challenger"),
        "is_open": float(r.incumb_status == "Open"),
        "cook_ordinal": COOK_ORDINAL.get(r.cook_rating, 0),
        "cand_ratio_t": r.cand_d_total / BUDGET_2026,
        "r_ratio_t": r.r_total / BUDGET_2026,
        "generic_ballot": gb,
    } for r in races])


def predict_forecast_dollars(sel_model, inten_model, X: pd.DataFrame) -> np.ndarray:
    Xc = sm.add_constant(X[FEATURES], has_constant="add")
    p_funded = sel_model.predict(Xc)
    log_share = inten_model.predict(Xc)
    raw = p_funded * np.exp(log_share)
    return raw / raw.sum() * BUDGET_2026


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    _, coef, sigma_model = load_processed_artifacts(config.processed_path())
    races = build_universe(cycle=2026)
    live_gb = config.generic_ballot_for_cycle(2026)
    factor_model = build_dummy_factor_model(races, live_gb)
    cov_matrix = factor_model.race_covariance()
    reference_default = config.floor_maturity_reference_dollars()

    as_of = datetime.now(timezone.utc).date()
    election_day = datetime(2026, 11, 3).date()
    days_remaining = (election_day - as_of).days
    src = RealizedSpendCommitmentSource(cycle=2026, party="D")
    committed = src.committed_capital(0, as_of, races)
    committed_total = sum(committed.values())

    # ── GB uncertainty: this project's own fitted term-structure volatility ──
    gb_dynamics = json.load(open(config.processed_path() / "gb_dynamics.json"))
    sigma_g_per_sqrt_day = gb_dynamics["sigma_g_per_sqrt_day"]
    gb_sd = sigma_g_per_sqrt_day * np.sqrt(days_remaining)
    print(f"GB uncertainty: sigma_g_per_sqrt_day={sigma_g_per_sqrt_day:.4f}, "
          f"{days_remaining}d remaining -> SD={gb_sd:.3f} points around today's {live_gb:+.2f}")

    # ── Maturity threshold uncertainty: the 8 empirical sweep values ───────
    threshold_sweep = pd.read_csv(config.outputs_path() / "floor_maturity_reference_sweep.csv")
    threshold_values = threshold_sweep["threshold_dollars"].values
    print(f"Maturity threshold uncertainty: resampling from {len(threshold_values)} empirical values "
          f"(${threshold_values.min():,.0f} to ${threshold_values.max():,.0f})")

    # ── Forecast-model training data, for cycle-bootstrap resampling ───────
    train_df = _prep(pd.read_csv(config.processed_path() / "dccc_forecast_training_data.csv"))
    cycles = sorted(train_df["cycle"].unique())

    # ── Point-estimate model-optimal allocation, held fixed per draw ───────
    ledger = CapitalLedger(
        period=0, total_budget=BUDGET_2026, committed_by_race=committed,
        committed_total=committed_total, deployable_total=BUDGET_2026 - committed_total,
    )
    races_with_floor = ledger.apply_to_races(races)
    floor_maturity_opt_default = ceiling_mod.maturity(
        np.array([r.cand_d_total for r in races_with_floor]),
        np.array([r.r_total for r in races_with_floor]), reference_default,
    )
    result = optimize_nonlinear(
        races_with_floor, coef, sigma_model, budget=BUDGET_2026, cov_matrix=cov_matrix,
        gamma=0.0, cap_fraction=0.15, party_budget=ledger.deployable_total,
        floor_maturity=floor_maturity_opt_default,
    )
    floor = ledger.deployable_floor_for(races)
    committed_arr = np.array([committed.get(r.district_id, 0.0) for r in races])
    model_total_party_fixed = (result.allocations - floor) + committed_arr
    print(f"\nPoint-estimate model-optimal allocation computed once, held fixed across all draws.")

    print(f"\nRunning {args.n_draws} Monte Carlo draws...")
    gains = []
    for i in range(args.n_draws):
        gb_draw = float(rng.normal(live_gb, gb_sd))
        threshold_draw = float(rng.choice(threshold_values))

        races_gb = [dataclasses.replace(r, generic_ballot=gb_draw) for r in races]
        floor_maturity_eval = ceiling_mod.maturity(
            np.array([r.cand_d_total for r in races_gb]),
            np.array([r.r_total for r in races_gb]), threshold_draw,
        )

        boot_cycles = rng.choice(cycles, size=len(cycles), replace=True)
        boot_train = pd.concat([train_df[train_df["cycle"] == c] for c in boot_cycles], ignore_index=True)
        try:
            sel_b, inten_b = fit_hurdle(boot_train)
        except Exception:
            continue
        X_2026 = features_for(races_gb, gb_draw)
        forecast_dollars_b = predict_forecast_dollars(sel_b, inten_b, X_2026)

        model_seats = nonlinear_expected_seats_at_party_dollars(
            races_gb, coef, sigma_model, model_total_party_fixed, floor_maturity=floor_maturity_eval)
        forecast_seats = nonlinear_expected_seats_at_party_dollars(
            races_gb, coef, sigma_model, forecast_dollars_b, floor_maturity=floor_maturity_eval)
        gains.append(model_seats - forecast_seats)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{args.n_draws} draws...")

    gains = np.array(gains)
    p5, p10, p25, p50, p75, p90, p95 = np.percentile(gains, [5, 10, 25, 50, 75, 90, 95])
    mean, sd = gains.mean(), gains.std()

    print(f"\n{len(gains)} successful draws.")
    print(f"Mean gain: {mean:+.3f}  SD: {sd:.3f}")
    print(f"Median: {p50:+.3f}")
    print(f"50% interval: [{p25:+.3f}, {p75:+.3f}]")
    print(f"80% interval: [{p10:+.3f}, {p90:+.3f}]")
    print(f"90% interval: [{p5:+.3f}, {p95:+.3f}]")
    print(f"\nReported as: {mean:+.2f} seats, +/- {sd:.2f} (1 SD), "
          f"90% scenario range [{p5:+.2f}, {p95:+.2f}]")
    print(f"\nPct of draws with gain <= 0: {100*np.mean(gains <= 0):.1f}%")

    out_path = config.outputs_path() / "simulate_2026_gain_uncertainty.csv"
    pd.DataFrame({"gain": gains}).to_csv(out_path, index=False)
    summary = {
        "n_draws": int(len(gains)), "mean": float(mean), "sd": float(sd),
        "p5": float(p5), "p10": float(p10), "p25": float(p25), "p50": float(p50),
        "p75": float(p75), "p90": float(p90), "p95": float(p95),
        "pct_leq_zero": float(100 * np.mean(gains <= 0)),
        "gb_sd": float(gb_sd), "days_remaining": int(days_remaining),
    }
    with open(config.outputs_path() / "simulate_2026_gain_uncertainty_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {config.outputs_path() / 'simulate_2026_gain_uncertainty_summary.json'}")


if __name__ == "__main__":
    main()
