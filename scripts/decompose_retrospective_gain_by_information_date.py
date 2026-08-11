#!/usr/bin/env python3
"""
Decompose Paper I's retrospective seat-gain finding (+2.83 seats, 2024;
+3.22, 2022 OOS) by information date: how much of that gain is visible
using only information genuinely available *during* the cycle, versus only
becoming visible with the full cycle's hindsight?

This directly answers docs/paper2_draft.md §6.3's first question ("does the
model's real-time recommendation converge to Paper I's full-hindsight
recommendation") but via a simpler, more robust route than reusing
dynamic/simulate.py's one_step_ahead() + dynamic/timing.py's per-period gap
summation. That machinery has two structural issues that make its existing
outputs (outputs/dynamic_timing_*.csv) untrustworthy for this question:

  1. config.yaml's dynamic.commitment_mode defaults to "zero"
     (ZeroCommitmentSource) -- L_t=0 at every period, so F_t never shrinks
     across periods to reflect what's already been spent or recommended.
  2. dynamic/timing.py compares the FULL remaining-budget recommendation at
     each period (allocations[i] - floor[i], i.e. "spend the entire rest of
     F_t on this race") against DCCC's actual *incremental* two-week spend,
     then sums that gap across ~38 periods -- inflating the apparent
     front-loading gap by roughly the number of periods, since the same
     near-total recommendation is effectively counted many times over.
     (Confirmed directly: outputs/dynamic_timing_2024.csv reports a total
     gap of $553M for a single race, larger than the entire 2024 party
     budget.)

Both are real, separate bugs in the receding-horizon/execution-policy
machinery (dynamic/horizon.py, dynamic/timing.py) -- worth fixing on their
own terms, but not required to answer this specific question. This script
instead reuses dynamic/simulate.py's dated-reconstruction helper
(_reconstruct_races_at, already correct and already handles the dated
candidate-panel / dated-IE / static-coordinated approximations) to build a
real, no-lookahead snapshot of each race's spending as of a historical
checkpoint date, then runs Paper I's own unmodified single-shot optimizer
(the same optimize_nonlinear() call run_backtest.py uses) against that
snapshot's floors -- asking "if a committee had to lock in its full-cycle
party-budget allocation today, using only information available as of
today, what would this model recommend, and how does its expected-seat
advantage over DCCC's actual final allocation compare to Paper I's
full-hindsight +2.83?"

This isolates the "is the signal visible early" question cleanly, holding
the total budget and the outcome evaluation fixed at their Paper I values;
it does NOT answer the separate, more operational question of how a
receding-horizon MPC policy should actually execute period to period
(that's dynamic/horizon.py, and Paper III's Theta) -- see the module-level
comparison table this script prints for exactly what is and isn't held
fixed across checkpoints.

Usage:
    python scripts/decompose_retrospective_gain_by_information_date.py --cycle 2024
    python scripts/decompose_retrospective_gain_by_information_date.py --cycle 2022 --processed-dir data/processed_oos_2020
"""

from __future__ import annotations
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from backtest import config
from backtest.data.universe import build_universe
from backtest.model.win_prob import compute_outputs_batch
from backtest.model import ceiling as ceiling_mod
from backtest.optimizer.allocator import optimize_nonlinear, nonlinear_expected_seats_at_party_dollars
from backtest.dynamic.simulate import (
    _reconstruct_races_at, _static_floor_totals,
    _has_dated_candidate_panel, _candidate_fallback_totals,
)
from backtest.dynamic.periods import fec_quarterly_periods
from run_backtest import load_processed_artifacts, build_dummy_factor_model  # type: ignore

ELECTION_DAY = {2022: date(2022, 11, 8), 2024: date(2024, 11, 5)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose Paper I's retrospective seat-gain finding by information date"
    )
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--floor-maturity-ceiling", action="store_true",
                        help="Apply the portfolio-level floor-maturity persuasion-ceiling "
                             "correction (model/ceiling.py's maturity()) at each checkpoint, "
                             "instead of the original unmodified ceiling.")
    args = parser.parse_args()

    processed = Path(args.processed_dir) if args.processed_dir else config.processed_path()
    cycle = args.cycle
    if cycle not in ELECTION_DAY:
        raise SystemExit(f"No Election Day date configured for cycle {cycle}")
    election_day = ELECTION_DAY[cycle]

    beta_rc, coef, sigma_model = load_processed_artifacts(processed)
    base_races = build_universe(cycle=cycle)

    factor_model = build_dummy_factor_model(base_races, config.generic_ballot_for_cycle(cycle))
    cov_matrix = factor_model.race_covariance()
    cap_baseline = config.optimizer_cfg()["cap_regimes"][-1]

    # ── Full-hindsight baseline: Paper I's own headline, recomputed here as
    # a sanity check that this script's setup matches run_backtest.py's. ──
    budget_final = sum(r.d_total for r in base_races)
    party_budget_final = sum(r.d_total - r.cand_d_total for r in base_races)
    outputs_final = compute_outputs_batch(base_races, coef, sigma_model)
    dccc_final_expected_seats = float(sum(o.p_win for o in outputs_final))

    print("Solving full-hindsight baseline (sanity check against Paper I's headline)...")
    final_result = optimize_nonlinear(
        base_races, coef, sigma_model, budget_final, cov_matrix, 0.0, cap_baseline,
        party_budget=party_budget_final, eta=args.eta,
    )
    full_hindsight_gain = final_result.expected_seats - dccc_final_expected_seats
    print(
        f"  Full-hindsight: model={final_result.expected_seats:.3f} vs "
        f"DCCC-final={dccc_final_expected_seats:.3f} => gain={full_hindsight_gain:+.3f} "
        f"(Paper I headline: +2.83 for 2024 / +3.22 for 2022 OOS -- should be close, "
        f"not necessarily bit-identical, since this script uses the current live pipeline)\n"
    )

    # ── Checkpoints spanning the cycle, using real historical filing dates ──
    static_totals = _static_floor_totals(cycle)
    use_dated = _has_dated_candidate_panel(cycle)
    fallback_totals = None if use_dated else _candidate_fallback_totals(cycle)
    if not use_dated:
        print(f"WARNING: no dated candidate periodic-reports panel for cycle {cycle} -- "
              "checkpoints will use the cycle-final candidate floor at every date, which "
              "defeats the purpose of this script. Results below are not meaningful.")

    periods = [p for p in fec_quarterly_periods(cycle) if p.period_date <= election_day]
    print(f"{len(periods)} checkpoints (real FEC quarterly filing dates, cycle-final party "
          f"budget ${party_budget_final:,.0f} used as F_t at every checkpoint; only each "
          f"race's real, dated floor changes across checkpoints):\n")

    reference_dollars = config.floor_maturity_reference_dollars()
    if args.floor_maturity_ceiling:
        print(f"Floor-maturity ceiling correction ON (reference=${reference_dollars:,.0f})\n")

    rows = []
    for rp in periods:
        races_t = _reconstruct_races_at(
            rp.index, rp.period_date, cycle, base_races, static_totals,
            use_dated, fallback_totals,
        )
        party_spent_to_date = float(sum(max(r.d_total - r.cand_d_total, 0.0) for r in races_t))

        floor_maturity = None
        if args.floor_maturity_ceiling:
            floor_maturity = ceiling_mod.maturity(
                np.array([r.cand_d_total for r in races_t]),
                np.array([r.r_total for r in races_t]),
                reference_dollars,
            )

        result_t = optimize_nonlinear(
            races_t, coef, sigma_model, budget_final, cov_matrix, 0.0, cap_baseline,
            party_budget=party_budget_final, eta=args.eta, floor_maturity=floor_maturity,
        )
        gain_t = result_t.expected_seats - dccc_final_expected_seats
        days_before_election = (election_day - rp.period_date).days
        pct_of_final_gain = (gain_t / full_hindsight_gain * 100) if full_hindsight_gain else float("nan")

        # Real-world evaluation: take the SAME checkpoint-informed party-dollar
        # decision (recommended D_i minus the checkpoint's own candidate floor)
        # but evaluate it against the real, final candidate floors and real,
        # final opponent spending -- exactly how Paper I's own counterfactual is
        # evaluated (swap only the party-money decision, hold everything else at
        # its true realized value). This isolates whether gain_t is driven by
        # the model picking genuinely wrong races given real-world outcomes, or
        # by an artifact of evaluating the recommendation inside its own
        # internally-consistent but immature (low-D, low-R) checkpoint world.
        checkpoint_floor = np.array([r.cand_d_total for r in races_t])
        recommended_party = result_t.allocations - checkpoint_floor
        real_world_seats = nonlinear_expected_seats_at_party_dollars(
            base_races, coef, sigma_model, recommended_party, eta=args.eta,
        )
        real_world_gain = real_world_seats - dccc_final_expected_seats

        row = {
            "period_date": rp.period_date.isoformat(),
            "days_before_election": days_before_election,
            "party_dollars_spent_to_date": round(party_spent_to_date, 0),
            "pct_of_final_party_budget_spent": round(100 * party_spent_to_date / party_budget_final, 1),
            "model_expected_seats": round(result_t.expected_seats, 3),
            "gain_vs_dccc_final": round(gain_t, 3),
            "pct_of_full_hindsight_gain": round(pct_of_final_gain, 1),
            "real_world_expected_seats": round(real_world_seats, 3),
            "real_world_gain_vs_dccc_final": round(real_world_gain, 3),
        }
        rows.append(row)
        print(
            f"  {rp.period_date} ({days_before_election:>4}d before election, "
            f"{row['pct_of_final_party_budget_spent']:>5.1f}% of final party $ spent): "
            f"gain={gain_t:+.3f} seats (own-environment eval) | "
            f"real-world gain={real_world_gain:+.3f} seats (same decision, real final floors+opponent)"
        )

    df = pd.DataFrame(rows)
    tag = "_maturity_ceiling" if args.floor_maturity_ceiling else ""
    out_path = config.outputs_path() / f"retrospective_gain_by_information_date_{cycle}{tag}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print(
        "\nInterpretation: if gain_vs_dccc_final stays small (a small % of the full-hindsight "
        "figure) until late in the cycle, the retrospective 2.83/3.22 headline is largely a "
        "hindsight artifact -- consistent with the live 2026 application's own small real-time "
        "estimate, and consistent with a well-specified model correctly pricing its own "
        "uncertainty early in a cycle. If the gain is already close to the full-hindsight value "
        "well before Election Day, the signal was genuinely visible in real time, and the small "
        "2026 live estimate would deserve a different explanation (e.g. the live universe's own "
        "documented approximations -- PVI proxy years, immature floors -- rather than a general "
        "property of real-time application)."
    )


if __name__ == "__main__":
    main()
