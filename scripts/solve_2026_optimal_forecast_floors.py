#!/usr/bin/env python3
"""
Re-run the live 2026 nonlinear optimizer using MATURE (trend-forecast)
candidate-committee floors instead of today's actual (mostly near-zero)
ones, to get a seat-gain estimate that isn't distorted by the marginal-
seat-gain formula's 1/D leverage spike at low spend
(scripts/investigate_msg_low_d_extrapolation.py's already-documented finding).

Motivation: plot_2026_live_allocation.py's run_live_allocation() uses
TODAY's actual floors, which are almost all near $0 this early in the cycle
(~100 days out) -- nearly every competitive race showed an implausibly
large model-recommended allocation as a result, and the naive "optimal vs.
no-further-deployment" seat gain computed from it (~27 seats) is not
trustworthy: the marginal-seat-gain formula's spending elasticity was
disproportionately estimated from Safe-tier low-spend races (85/118
repeat-challenger pairs), not competitive ones (14/118) -- applying it to
dozens of simultaneously near-zero-floor competitive races amplifies that
uncertainty rather than averaging it out.

This script instead projects BOTH parties' candidate-committee spend
forward to Election Day at their own calibrated per-tier trickle rate
(mirroring scripts/estimate_candidate_spend_trickle.py's D-side rate,
generalized here to both parties -- projecting D forward while holding R
frozen at today's near-zero value would introduce a new asymmetry, not fix
the old one), then re-solves the same nonlinear optimizer against that more
mature floor.

Two numbers are reported for the naive (today's-floor) and forecast-floor
cases side by side, so the size of the low-floor artifact is visible
directly, not just asserted.

Output: outputs/optimal_2026_forecast_floors.json, printed comparison.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import config
from backtest.data.universe import build_universe
from backtest.data import fec
from backtest.dynamic.ledger import RealizedSpendCommitmentSource
from backtest.dynamic.periods import biweekly_periods
from backtest.model.margin import MarginModelCoefficients
from backtest.model.win_prob import compute_outputs_batch
from backtest.model.budget import estimate_budget_2026
from backtest.optimizer.allocator import optimize_nonlinear
from backtest.types import SigmaModel
from run_backtest import build_dummy_factor_model

CANDIDATE_CYCLES = [2012, 2014, 2016, 2018, 2020, 2022, 2024]
TIERS = ["Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"]


def _has_dated_panel(cycle: int) -> bool:
    return (config.raw_path("fec") / f"candidate_periodic_reports_{cycle}.csv").exists()


def _spend_period_panel(cycle: int, party: str) -> pd.DataFrame:
    """Per-party analog of estimate_candidate_spend_trickle.py's
    build_d_spend_period_panel -- same construction, parameterized by party
    rather than hardcoded to D, so R's own organic growth can be projected
    on equal footing rather than left frozen while D's is not."""
    races = build_universe(cycle=cycle)
    tiers = {r.district_id: r.cook_rating for r in races}
    periods = biweekly_periods(date(cycle, 1, 1), date(cycle, 11, 8))
    rows = []
    for p in periods:
        cum = fec.cumulative_candidate_spend_as_of(cycle, p.period_date)
        c = cum[cum["party"] == party][["district_id", "disb_cum"]].rename(
            columns={"disb_cum": "cand_cum"})
        c["period_index"] = p.index
        rows.append(c)
    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["district_id", "cand_cum", "period_index"])
    panel["tier"] = panel["district_id"].map(tiers)
    panel = panel[panel["tier"].notna()].copy()
    panel["cycle"] = cycle
    return panel


def trickle_rate_by_tier(party: str) -> dict[str, float]:
    available = [c for c in CANDIDATE_CYCLES if _has_dated_panel(c)]
    panels = [_spend_period_panel(c, party) for c in available]
    full = pd.concat(panels, ignore_index=True).sort_values(["cycle", "district_id", "period_index"])
    g = full.groupby(["cycle", "district_id"])
    full["delta"] = g["cand_cum"].diff()
    full = full.dropna(subset=["delta"]).copy()
    full["rate_per_day"] = full["delta"] / 14.0

    pooled = full["rate_per_day"]
    rates = {"_pooled": float(pooled.mean()) if len(pooled) else 0.0}
    for tier in TIERS:
        sub = full[full["tier"] == tier]["rate_per_day"]
        rates[tier] = float(sub.mean()) if len(sub) else rates["_pooled"]
    return rates


def main() -> None:
    as_of = datetime.now(timezone.utc).date()
    election_day = config.election_day(2026)
    days_remaining = max(0, (election_day - as_of).days)
    print(f"As of {as_of}, {days_remaining} days remaining to Election Day.\n")

    races = build_universe(cycle=2026)
    district_ids = [r.district_id for r in races]
    tiers = [r.cook_rating for r in races]

    print("Fitting D and R candidate-committee trickle rates (2012-2024 panel)...")
    d_rates = trickle_rate_by_tier("D")
    r_rates = trickle_rate_by_tier("R")
    d_trickle = np.array([d_rates.get(t, d_rates["_pooled"]) for t in tiers])
    r_trickle = np.array([r_rates.get(t, r_rates["_pooled"]) for t in tiers])

    cand = fec.cumulative_candidate_spend_as_of(2026, as_of)
    cand_by_key = {(row.district_id, row.party): row.disb_cum for row in cand.itertuples()}
    committed_d = RealizedSpendCommitmentSource(cycle=2026, party="D").committed_capital(0, as_of, races)
    committed_r = RealizedSpendCommitmentSource(cycle=2026, party="R").committed_capital(0, as_of, races)

    cand_d_today = np.array([cand_by_key.get((did, "D"), 0.0) for did in district_ids])
    cand_r_today = np.array([cand_by_key.get((did, "R"), 0.0) for did in district_ids])
    cand_d_forecast = cand_d_today + d_trickle * days_remaining
    cand_r_forecast = cand_r_today + r_trickle * days_remaining

    committed_d_arr = np.array([committed_d.get(did, 0.0) for did in district_ids])
    committed_r_arr = np.array([committed_r.get(did, 0.0) for did in district_ids])
    forecast_d_total = cand_d_forecast + committed_d_arr
    forecast_r_total = cand_r_forecast + committed_r_arr

    # Forecast floor = projected candidate-committee spend PLUS already-committed
    # coordinated/IE party spend -- both are irreversible by the optimizer's
    # decision date, so both belong in the floor `optimize_nonlinear()` reads
    # from `cand_d_total` (it takes no separate floor argument). This matches
    # dynamic/ledger.py's `apply_to_races`/`deployable_floor_for`, which bakes
    # committed capital into `cand_d_total` the same way. Only the *remaining*
    # deployable party budget (party_budget = budget - committed_total, below)
    # is left for the optimizer to allocate on top of this floor.
    forecast_races = [
        replace(r, cand_d_total=float(forecast_d_total[i]),
                d_total=float(forecast_d_total[i]), r_total=float(forecast_r_total[i]))
        for i, r in enumerate(races)
    ]

    with open(ROOT / "data/processed/margin_model_coef.json") as f:
        cd = json.load(f)
    coef = MarginModelCoefficients(
        **{k: cd[k] for k in ["alpha0", "alpha1", "alpha2", "alpha3", "alpha4",
                               "beta1", "beta2", "beta3"]},
        alpha5=cd.get("alpha5", 0.0), beta1_open=cd.get("beta1_open"),
    )
    with open(ROOT / "data/processed/sigma_model.json") as f:
        sigma_model = SigmaModel(_coef=json.load(f))

    budget = estimate_budget_2026()
    gb = config.generic_ballot_for_cycle(2026)
    factor_model = build_dummy_factor_model(forecast_races, gb)
    cov_matrix = factor_model.race_covariance()
    committed_total = float(sum(committed_d_arr))
    party_budget = budget - committed_total   # F_t, same convention as run_live_allocation()

    print(f"Budget ${budget:,.0f}  |  Already committed (coordinated+IE) ${committed_total:,.0f}  |  "
          f"Deployable F_t ${party_budget:,.0f}\n")

    print("Solving nonlinear optimizer against forecast floors...")
    opt = optimize_nonlinear(forecast_races, coef, sigma_model, budget, cov_matrix,
                              gamma=0.0, cap_fraction=0.15, party_budget=party_budget)

    outputs_noop = compute_outputs_batch(forecast_races, coef, sigma_model)
    baseline_expected_seats = float(sum(o.p_win for o in outputs_noop))

    print(f"\nForecast-floor optimal expected seats:            {opt.expected_seats:.2f}")
    print(f"Forecast-floor no-further-deployment expected seats: {baseline_expected_seats:.2f}")
    print(f"Seat gain from optimal deployment (forecast floors): {opt.expected_seats - baseline_expected_seats:.2f}")

    result = {
        "as_of": as_of.isoformat(),
        "days_remaining": days_remaining,
        "budget": budget,
        "committed_total": committed_total,
        "deployable_total": party_budget,
        "optimal_expected_seats": opt.expected_seats,
        "no_further_deployment_expected_seats": baseline_expected_seats,
        "seat_gain": opt.expected_seats - baseline_expected_seats,
        "d_trickle_by_tier": d_rates,
        "r_trickle_by_tier": r_rates,
    }
    out_path = ROOT / "outputs" / "optimal_2026_forecast_floors.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
