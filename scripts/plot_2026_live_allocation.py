#!/usr/bin/env python3
"""
Live 2026 model-recommended party spending (top races by recommended spend).

This is NOT a backtest chart like plot_allocator_comparison.py (which
compares strategies against a completed cycle's known outcome). The 2026
cycle hasn't happened yet, so there is no "DCCC observed" or "outcome"
series to compare against — this chart shows a single, prospective
recommendation: given today's live data, how should the modeled remaining
budget be deployed.

Pipeline (mirrors the Paper II dynamic/ architecture, one live period):
  1. Real 2026 race universe (build_universe) — PVI uses the same
     (2016, 2020) proxy years as 2022/2024 (see CYCLE_TO_PRES_YEARS in
     data/pvi.py); several states have mid-decade 2026 redistricting not
     reflected here.
  2. Budget: $394.3M — inflation-adjusted average of the 2018 and 2022
     midterm party-controlled budgets (BLS CPI-U, see docs/paper2_draft.md
     for the derivation). NOT a live fundraising-pace projection.
  3. Generic ballot: a live point-estimate (21-day trailing average of
     VoteHub polls, scripts/fetch_polling.py), used as a single per-cycle
     constant exactly like every other cycle's static GB value — NOT fed
     in as a time-varying quantity (see the alpha3 identification note in
     dynamic/simulate.py).
  4. L_t: RealizedSpendCommitmentSource — real, already-disbursed DCCC
     coordinated + independent expenditures to date. Small this early in
     the cycle; expected to grow on later re-runs.
  5. optimize_nonlinear(), gamma=0 (risk-neutral), cap_fraction=0.15 —
     identical Paper I optimizer, unmodified.

Output: outputs/allocation_2026_live.png, outputs/allocation_2026_live.csv
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import config
from backtest.data.universe import build_universe
from backtest.dynamic.ledger import RealizedSpendCommitmentSource
from backtest.dynamic.updates import EMAStateUpdater
from backtest.dynamic.periods import ReportingPeriod
from backtest.dynamic.horizon import run_receding_horizon
from backtest.model.budget import estimate_budget_2026
from run_backtest import load_processed_artifacts, build_dummy_factor_model

BUDGET_2026 = estimate_budget_2026()   # single source of truth: backtest.model.budget
TOP_N = 25


def run_live_allocation() -> tuple[pd.DataFrame, dict]:
    _, coef, sigma_model = load_processed_artifacts(config.processed_path())
    races = build_universe(cycle=2026)
    gb = config.generic_ballot_for_cycle(2026)
    factor_model = build_dummy_factor_model(races, gb)
    cov_matrix = factor_model.race_covariance()

    as_of = datetime.now(timezone.utc).date()
    periods = [ReportingPeriod(index=0, period_date=as_of, label="2026-live")]
    commitment_source = RealizedSpendCommitmentSource(cycle=2026, party="D")

    results = run_receding_horizon(
        periods, races, coef, sigma_model,
        commitment_source, EMAStateUpdater(lam=config.dynamic_cfg()["ema_lambda"]),
        cov_matrix_fn=lambda rs: cov_matrix,
        gamma=0.0, cap_fraction=0.15,
        total_budget_fn=lambda t: BUDGET_2026,
        generic_ballot_national=gb,
    )
    res = results[0]
    opt = res.optimizer_result

    races_out = res.state.to_race_records()
    floor = res.ledger.deployable_floor_for(races_out)
    rows = []
    for i, r in enumerate(races_out):
        committed = res.ledger.committed_by_race.get(r.district_id, 0.0)
        recommended_total_party = opt.allocations[i] - floor[i] + committed
        rows.append({
            "district_id": r.district_id,
            "cook_rating": r.cook_rating,
            "pvi": r.pvi,
            "already_committed": committed,
            "recommended_additional": opt.allocations[i] - floor[i],
            "recommended_total_party": recommended_total_party,
        })
    df = pd.DataFrame(rows).sort_values("recommended_total_party", ascending=False).reset_index(drop=True)

    meta = {
        "as_of": as_of.isoformat(),
        "budget": BUDGET_2026,
        "generic_ballot": gb,
        "committed_total": res.ledger.committed_total,
        "deployable_total": res.ledger.deployable_total,
        "expected_seats": opt.expected_seats,
        "status": opt.status,
        "n_races": len(races_out),
    }
    return df, meta


def plot(df: pd.DataFrame, meta: dict) -> None:
    top = df.head(TOP_N).copy().iloc[::-1]   # reverse so #1 lands at the top of a horizontal chart

    C_COMMITTED = "#1a6faf"   # blue — matches C_DCCC in plot_allocator_comparison.py
    C_ADDITIONAL = "#2a9d4f"  # green — matches C_MODEL

    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(top))

    ax.barh(y, top["already_committed"] / 1e6, color=C_COMMITTED, label="Already committed (L_t)")
    ax.barh(y, top["recommended_additional"] / 1e6, left=top["already_committed"] / 1e6,
            color=C_ADDITIONAL, label="Model-recommended additional spend")

    labels = [f"{r.district_id}  ({r.cook_rating})" for r in top.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Party-controlled spending ($M)", fontsize=11)
    ax.set_title(
        f"2026 Live Model-Recommended Party Spending — Top {TOP_N} Races\n"
        f"Budget ${meta['budget']/1e6:.1f}M (2018/2022 midterm avg., inflation-adjusted)  |  "
        f"Generic ballot D{meta['generic_ballot']:+.2f} (live polling point-estimate)  |  "
        f"As of {meta['as_of']}  |  PROSPECTIVE — cycle not yet concluded",
        fontsize=10.5,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    out_png = ROOT / "outputs" / "allocation_2026_live.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_png}")


def main() -> None:
    df, meta = run_live_allocation()
    out_csv = ROOT / "outputs" / "allocation_2026_live.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")
    print(f"\nExpected seats: {meta['expected_seats']:.2f}  |  status: {meta['status']}  |  "
          f"L_t=${meta['committed_total']:,.0f}  F_t=${meta['deployable_total']:,.0f}")
    plot(df, meta)

    # --- Single source of truth for today/F0/election_day (Paper III audit,
    # 2026-07-16): scripts/solve_bellman_lsm.py and scripts/estimate_gb_ou_drift.py
    # previously re-typed these as independent literals (a stale-TODAY /
    # 98-vs-110-days-remaining drift already surfaced from this before the fix).
    as_of = date.fromisoformat(meta["as_of"])
    elec_day = config.election_day(2026)
    live_state = {
        "as_of": meta["as_of"],
        "election_day": elec_day.isoformat(),
        "days_remaining": (elec_day - as_of).days,
        "budget_2026": meta["budget"],
        "l_t_committed": meta["committed_total"],
        "f0": meta["deployable_total"],
        "generic_ballot": meta["generic_ballot"],
    }
    with open(ROOT / "data/processed/live_2026_state.json", "w") as f:
        json.dump(live_state, f, indent=2)
    print(f"\nSaved -> data/processed/live_2026_state.json "
          f"(F0=${live_state['f0']:,.0f}, {live_state['days_remaining']}d remaining)")


if __name__ == "__main__":
    main()
