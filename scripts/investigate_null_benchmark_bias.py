#!/usr/bin/env python3
"""
Historical record: this script found why the Null (equal-weight) benchmark
edged out the nonlinear model optimizer in the 2022 OOS backtest (221.87 vs
221.66, -0.21 seats -- a result that should not have survived unexamined
given every other comparison in this project shows the model optimizer
ahead). Both mechanisms it identified are now fixed at the source in
comparison/benchmark.py::compare_allocators() and
permutation_test_allocation_efficiency() (2026-07-22) -- this script is kept
as a standalone diagnostic and regression check, not because either fix is
still pending.

Two mechanisms, checked directly rather than assumed:

  1. Evaluation-method asymmetry. Null/Cook are scored via the linearized
     MSG-delta approximation in comparison/benchmark.py::_expected_seats_at_shares(),
     whose own docstring says it "overestimates because MSG decays as
     spending increases." The Model row is separately replaced with the
     true nonlinear Phi(mu/sigma) evaluation (run_backtest.py's own
     comment: "the optimizer's Phi(mu/sigma) sum is authoritative"). Re-score
     Null with the SAME nonlinear method the Model gets credited with.

  2. Budget-scope asymmetry. null_equal_weight_shares()/cook_proportional_shares()
     scale to sum(r.d_total for r in races) -- the ENTIRE two-party spending
     pool across all races, including every candidate's own committee money
     in every safe seat, none of which the DCCC controls. The model
     optimizer, correctly, only reallocates the DCCC-controllable party
     budget, holding every candidate's own money fixed everywhere. Re-score
     Null/Cook under the model's actual constraint (party-budget-only,
     floors fixed) for a true apples-to-apples comparison. (Now the default
     behavior of compare_allocators() itself, per user instruction: "All
     models/methods when compared to each other should only use the DCCC
     budget, that is the whole point.")

Usage:
    python scripts/investigate_null_benchmark_bias.py --cycle 2022 --processed-dir data/processed_oos_2020
    python scripts/investigate_null_benchmark_bias.py --cycle 2024   # sanity check: same direction, smaller effect
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import config
from backtest.data.universe import build_universe
from backtest.model.margin import MarginModelCoefficients
from backtest.model.win_prob import compute_outputs_batch
from backtest.types import SigmaModel
from backtest.comparison.benchmark import (
    null_equal_weight_shares, cook_proportional_shares, _expected_seats_at_shares,
)
from backtest.optimizer.allocator import _precompute_race_arrays, _p_win_vec


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate Null/Cook benchmark bias vs. the model optimizer")
    parser.add_argument("--cycle", type=int, default=2022)
    parser.add_argument("--processed-dir", type=str, default="data/processed_oos_2020")
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    with open(processed / "margin_model_coef.json") as f:
        d = json.load(f)
    coef = MarginModelCoefficients(
        **{k: d[k] for k in ["alpha0", "alpha1", "alpha2", "alpha3", "alpha4", "beta1", "beta2", "beta3"]},
        alpha5=d.get("alpha5", 0.0), beta1_open=d.get("beta1_open"),
    )
    with open(processed / "sigma_model.json") as f:
        sigma_model = SigmaModel(_coef=json.load(f))

    races = build_universe(cycle=args.cycle)
    outputs = compute_outputs_batch(races, coef, sigma_model)

    total_budget = sum(r.d_total for r in races)
    floors = np.array([r.cand_d_total for r in races])
    party_budget_dccc = sum(r.d_total - r.cand_d_total for r in races)
    competitive = set(config.competitive_ratings())
    competitive_mask = np.array([r.cook_rating in competitive for r in races])

    print(f"Cycle {args.cycle}: n races = {len(races)}, n competitive = {competitive_mask.sum()}")
    print(f"Total D budget (all sources, all races) = ${total_budget:,.0f}")
    print(f"DCCC party-controllable budget           = ${party_budget_dccc:,.0f}")
    print(f"  -> Null/Cook are allowed to move the full ${total_budget/1e6:,.0f}M pool;")
    print(f"     the Model optimizer only moves the ${party_budget_dccc/1e6:,.0f}M party slice.\n")

    null_shares = null_equal_weight_shares(races)
    cook_shares = cook_proportional_shares(races)

    null_linear = _expected_seats_at_shares(races, outputs, null_shares)
    cook_linear = _expected_seats_at_shares(races, outputs, cook_shares)
    print("=== Mechanism 1: linearized vs. true nonlinear evaluation ===")
    print(f"Null, linearized (as reported in the comparison table) : {null_linear:.3f}")
    print(f"Cook, linearized (as reported in the comparison table) : {cook_linear:.3f}")

    arrays = _precompute_race_arrays(races, coef, sigma_model, eta=0.0)

    def nonlinear_seats_full_budget(shares: np.ndarray) -> float:
        """Same budget scope as Null/Cook (full total budget), scored with
        the true nonlinear Phi(mu/sigma) instead of the linear MSG-delta."""
        party = np.maximum(shares * total_budget - floors, 0.0)
        return float(_p_win_vec(party, arrays).sum())

    null_nonlinear_full = nonlinear_seats_full_budget(null_shares)
    cook_nonlinear_full = nonlinear_seats_full_budget(cook_shares)
    print(f"Null, true nonlinear, SAME (full) budget scope          : {null_nonlinear_full:.3f}")
    print(f"Cook, true nonlinear, SAME (full) budget scope          : {cook_nonlinear_full:.3f}")
    print(f"  -> linearization bias for Null: {null_linear - null_nonlinear_full:+.3f} seats\n")

    print("=== Mechanism 2: same decision space as the Model optimizer ===")

    def nonlinear_seats_party_budget_only(shares: np.ndarray) -> float:
        """Redistribute ONLY the DCCC party budget among competitive races
        per the given relative weights; every floor (incl. non-competitive
        candidate money) stays exactly where it is -- the model's actual
        constraint."""
        party = np.zeros(len(races))
        party[competitive_mask] = shares[competitive_mask] * party_budget_dccc
        return float(_p_win_vec(party, arrays).sum())

    dccc_party = np.maximum(np.array([r.d_total for r in races]) - floors, 0.0)
    dccc_nonlinear = float(_p_win_vec(dccc_party, arrays).sum())
    null_party_scope = nonlinear_seats_party_budget_only(null_shares)
    cook_party_scope = nonlinear_seats_party_budget_only(cook_shares)

    print(f"DCCC observed, true nonlinear                : {dccc_nonlinear:.3f}")
    print(f"Null, party-budget-only, true nonlinear       : {null_party_scope:.3f}  "
          f"(gain vs DCCC: {null_party_scope - dccc_nonlinear:+.3f})")
    print(f"Cook, party-budget-only, true nonlinear        : {cook_party_scope:.3f}  "
          f"(gain vs DCCC: {cook_party_scope - dccc_nonlinear:+.3f})")
    print("\nCompare against outputs/allocator_comparison_table*.csv's Model optimizer row "
          "(already true nonlinear, party-budget-only by construction) for the real "
          "apples-to-apples comparison.")


if __name__ == "__main__":
    main()
