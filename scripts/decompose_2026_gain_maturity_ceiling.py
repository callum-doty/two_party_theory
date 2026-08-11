#!/usr/bin/env python3
"""
Rerun the 2026 live gain decomposition (decompose_2026_gain_by_race.py) with
the floor-maturity persuasion-ceiling correction (model/ceiling.py's
maturity()) applied -- this time on its actual intended use case.

The correction failed to help on the 2022/2024 retrospective checkpoint
sweep, but that failure turned out to be a red herring: the real driver
there was a since-fixed bug (frozen cand_d_total), unrelated to floor
maturity, and the correction was fighting a problem that wasn't the real
one. The 2026 live case has no such confound -- floors here are genuinely,
measurably thin (check_2026_likely_r_ceiling_balance.py: mean Phi0 of
0.19-0.25 for the top Likely R/Safe R contributors, driven by candidate
floors mostly under $1M against a $6.9M "mature" reference), with no future
"final" data to accidentally substitute for the real fix. This is the
correction's first clean test.

Applies floor_maturity consistently to BOTH the optimizer's decision and
the expected-seats evaluation of both allocations (model and DCCC-scaled)
-- scoring a maturity-corrected recommendation against an uncorrected
ceiling would evaluate it against a standard it was never optimized under.

Usage:
    python scripts/decompose_2026_gain_maturity_ceiling.py
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
from backtest.model import ceiling as ceiling_mod
from backtest.optimizer.allocator import (
    optimize_nonlinear, nonlinear_expected_seats_at_party_dollars,
    _precompute_race_arrays, _p_win_vec,
)
from run_backtest import load_processed_artifacts, build_dummy_factor_model  # type: ignore

BUDGET_2026 = estimate_budget_2026()


def _run(races, coef, sigma_model, cov_matrix, committed, committed_total, use_maturity: bool):
    ledger = CapitalLedger(
        period=0, total_budget=BUDGET_2026, committed_by_race=committed,
        committed_total=committed_total, deployable_total=BUDGET_2026 - committed_total,
    )
    races_with_floor = ledger.apply_to_races(races)

    floor_maturity_opt = None
    floor_maturity_eval = None
    if use_maturity:
        reference = config.floor_maturity_reference_dollars()
        floor_maturity_opt = ceiling_mod.maturity(
            np.array([r.cand_d_total for r in races_with_floor]),
            np.array([r.r_total for r in races_with_floor]),
            reference,
        )
        floor_maturity_eval = ceiling_mod.maturity(
            np.array([r.cand_d_total for r in races]),
            np.array([r.r_total for r in races]),
            reference,
        )

    result = optimize_nonlinear(
        races_with_floor, coef, sigma_model, budget=BUDGET_2026, cov_matrix=cov_matrix,
        gamma=0.0, cap_fraction=0.15, party_budget=ledger.deployable_total,
        floor_maturity=floor_maturity_opt,
    )
    floor = ledger.deployable_floor_for(races)
    committed_arr = np.array([committed.get(r.district_id, 0.0) for r in races])
    model_total_party = (result.allocations - floor) + committed_arr
    dccc_scaled_party = committed_arr * (BUDGET_2026 / committed_total)

    model_seats = nonlinear_expected_seats_at_party_dollars(
        races, coef, sigma_model, model_total_party, floor_maturity=floor_maturity_eval,
    )
    dccc_seats = nonlinear_expected_seats_at_party_dollars(
        races, coef, sigma_model, dccc_scaled_party, floor_maturity=floor_maturity_eval,
    )

    arrays = _precompute_race_arrays(races, coef, sigma_model, eta=0.0, floor_maturity=floor_maturity_eval)
    delta = _p_win_vec(model_total_party, arrays) - _p_win_vec(dccc_scaled_party, arrays)

    return model_seats, dccc_seats, model_seats - dccc_seats, delta, model_total_party


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

    print("Without floor-maturity correction (baseline, as previously reported):")
    m0, d0, gain0, delta0, party0 = _run(races, coef, sigma_model, cov_matrix, committed, committed_total, False)
    print(f"  model={m0:.3f}  dccc-scaled={d0:.3f}  gain={gain0:+.3f}")

    print("\nWith floor-maturity correction:")
    m1, d1, gain1, delta1, party1 = _run(races, coef, sigma_model, cov_matrix, committed, committed_total, True)
    print(f"  model={m1:.3f}  dccc-scaled={d1:.3f}  gain={gain1:+.3f}")

    print(f"\nGain change: {gain0:+.3f} -> {gain1:+.3f}  (delta {gain1 - gain0:+.3f})")

    # ── Cook-category breakdown, before vs after ────────────────────────────
    cook = [r.cook_rating for r in races]
    df = pd.DataFrame({
        "district_id": [r.district_id for r in races],
        "cook_rating": cook,
        "delta_seats_no_maturity": delta0,
        "delta_seats_with_maturity": delta1,
        "model_party_no_maturity": party0,
        "model_party_with_maturity": party1,
    })
    by_cat = df.groupby("cook_rating")[["delta_seats_no_maturity", "delta_seats_with_maturity"]] \
        .sum().reindex(["Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"])
    by_cat["change"] = by_cat["delta_seats_with_maturity"] - by_cat["delta_seats_no_maturity"]
    print("\nGain by Cook category, before vs. after:")
    print(by_cat.round(3).to_string())

    out_dir = config.outputs_path()
    df.to_csv(out_dir / "gain_decomposition_2026_maturity_comparison.csv", index=False)
    by_cat.to_csv(out_dir / "gain_by_category_2026_maturity_comparison.csv")
    print(f"\nSaved: {out_dir / 'gain_decomposition_2026_maturity_comparison.csv'}")
    print(f"Saved: {out_dir / 'gain_by_category_2026_maturity_comparison.csv'}")


if __name__ == "__main__":
    main()
