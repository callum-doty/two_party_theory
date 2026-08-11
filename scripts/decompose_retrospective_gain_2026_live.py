#!/usr/bin/env python3
"""
The 2026 analog of the "real-world" evaluation in
decompose_retrospective_gain_by_information_date.py -- adapted for a cycle
that hasn't concluded, so there is no future "final" state to borrow.

Why this differs from Paper II's existing live run
(scripts/plot_2026_live_allocation.py): that script compares the model's
optimal recommendation against "deploy the remaining budget on nothing
further" -- a baseline no real committee would choose, and a different,
more extreme question than Paper I asks. For 2022/2024, this project's own
session established that the fair, decision-relevant comparison is the same
one Paper I always uses: the model's recommended allocation of the full
party budget vs. DCCC's own OBSERVED allocation of that same budget, both
evaluated against the same real floors and real opponent spending.

For a completed cycle, "DCCC observed" is DCCC's real, final per-race party
spend. For 2026, that doesn't exist yet -- the best available substitute is
DCCC's REAL per-race committed spend so far (L_t, from
RealizedSpendCommitmentSource, the same real data plot_2026_live_allocation.py
uses), scaled up proportionally to the same total budget the model gets.
This preserves DCCC's revealed relative targeting pattern (which races it is
currently prioritizing over others) without assuming anything about how it
will finish the cycle.

Important, unavoidable caveat this script cannot remove: L_t is currently
small (~$1.6-2M against a $394M budget, as of the live run this repeats),
so "DCCC observed, scaled" is built from a thin, early sample and is much
less trustworthy than the real, complete 2022/2024 baselines those cycles'
decomposition used. Read the gain this script reports as a current,
noisy read on DCCC's revealed pattern -- not as validated evidence the way
the 2022/2024 real-world figures are, and expect it to firm up (for better
or worse) on later re-runs as L_t grows.

Usage:
    python scripts/decompose_retrospective_gain_2026_live.py
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

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
from backtest.optimizer.allocator import nonlinear_expected_seats_at_party_dollars
from run_backtest import load_processed_artifacts, build_dummy_factor_model  # type: ignore

ELECTION_DAY_2026 = date(2026, 11, 3)
BUDGET_2026 = estimate_budget_2026()


def main() -> None:
    _, coef, sigma_model = load_processed_artifacts(config.processed_path())
    races = build_universe(cycle=2026)
    gb = config.generic_ballot_for_cycle(2026)
    factor_model = build_dummy_factor_model(races, gb)
    cov_matrix = factor_model.race_covariance()

    as_of = datetime.now(timezone.utc).date()
    days_before_election = (ELECTION_DAY_2026 - as_of).days
    print(f"As of {as_of} ({days_before_election} days before the 2026 general election)\n")

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
    floor = res.ledger.deployable_floor_for(races_out)  # cand_d_total + L_t, per race

    committed = np.array([
        res.ledger.committed_by_race.get(r.district_id, 0.0) for r in races_out
    ])
    committed_total = float(committed.sum())
    print(f"Real, currently-committed DCCC party spend (L_t): ${committed_total:,.0f} "
          f"({100 * committed_total / BUDGET_2026:.2f}% of the ${BUDGET_2026/1e6:.1f}M "
          f"budget estimate)")

    if committed_total <= 0:
        raise SystemExit(
            "L_t is $0 -- cannot scale a proportional DCCC-observed baseline from an "
            "all-zero pattern. Nothing meaningful to report yet this cycle."
        )

    # DCCC-observed, scaled: preserve DCCC's real relative pattern (which
    # races it is currently prioritizing), scaled up to the same total
    # budget the model gets. No cap is imposed -- this is meant to reflect
    # DCCC's real revealed behavior as-is, exactly as Paper I never caps
    # the "DCCC observed" row for a completed cycle either.
    dccc_scaled_party = committed * (BUDGET_2026 / committed_total)

    # Model-optimal: total party $ (already-committed + model-recommended
    # additional), same quantity plot_2026_live_allocation.py reports as
    # "recommended_total_party".
    model_additional = opt.allocations - floor
    model_total_party = model_additional + committed

    dccc_scaled_seats = nonlinear_expected_seats_at_party_dollars(
        races_out, coef, sigma_model, dccc_scaled_party, eta=0.0,
    )
    model_seats = nonlinear_expected_seats_at_party_dollars(
        races_out, coef, sigma_model, model_total_party, eta=0.0,
    )
    gain = model_seats - dccc_scaled_seats

    print(
        f"\nDCCC-observed (real pattern, scaled to ${BUDGET_2026/1e6:.1f}M): "
        f"{dccc_scaled_seats:.3f} expected seats"
    )
    print(f"Model-optimal (same budget, today's real floors):          {model_seats:.3f} expected seats")
    print(f"\nGain (model - DCCC-scaled): {gain:+.3f} seats")

    print(
        "\nContext from the validated 2022/2024 decomposition (NOT a computed 2026 number, "
        "an analogy only): at a comparable ~90-115 days before Election Day, the real-world-"
        "evaluated gain had already reached roughly 70% of that cycle's eventual full-hindsight "
        "figure in both 2024 (+1.98 of +2.83, 113 days out) and 2022 (+2.25 of +3.22, 116 days "
        "out) -- a striking, tight match across two independently-estimated cycles. If 2026 "
        "followed the same pattern, today's true (unknowable until after the election) gain "
        "would plausibly be in a similar range. This script's own number above should be read "
        "as a current, thin-sample read on DCCC's revealed pattern given real 2026 data, not as "
        "validated evidence the way the 2022/2024 figures are -- L_t is still small and this "
        "comparison will firm up (for better or worse) on later re-runs."
    )

    out = {
        "as_of": as_of.isoformat(),
        "days_before_election": days_before_election,
        "budget_2026": BUDGET_2026,
        "l_t_committed": committed_total,
        "pct_budget_committed": 100 * committed_total / BUDGET_2026,
        "dccc_observed_scaled_expected_seats": dccc_scaled_seats,
        "model_optimal_expected_seats": model_seats,
        "gain": gain,
    }
    out_path = config.outputs_path() / "retrospective_gain_2026_live.csv"
    pd.DataFrame([out]).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
