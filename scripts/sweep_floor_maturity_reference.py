#!/usr/bin/env python3
"""
Robustness sweep + cross-cycle calibration check for the floor-maturity
reference threshold (model/ceiling.py's maturity(), config.yaml's
persuasion_ceiling.floor_maturity_reference_dollars) -- the same discipline
Paper I already applied to c_max (Appendix E.1, 7-point sweep), not yet
applied to this threshold (docs/retrospective_vs_realtime_investigation.md
Section 10/11's own open item).

Two checks:

1. Percentile sweep: recompute the 2026 live floor-maturity-corrected gain
   at 8 candidate reference values -- {p10, p25, p50, p75} of combined
   competitive-race (cand_d_total + r_total) at cycle-final, computed
   separately from BOTH 2024 and 2022 (so this doubles as the cross-cycle
   calibration check: does using 2022's distribution instead of 2024's to
   set the threshold change the conclusion?). p25 (2024) = $6,915,158 is
   the value already shipped as the default.

2. Historical-checkpoint neutrality check: now that both bugs behind
   Section 3 are fixed, the 2022/2024 real-world-evaluated checkpoint sweep
   already converges correctly WITHOUT the maturity correction. Re-run it
   WITH each cross-calibrated threshold to confirm the correction, if left
   on, doesn't reintroduce the distortion the original (single-threshold)
   version caused in Section 3.1's now-superseded first test.

Usage:
    python scripts/sweep_floor_maturity_reference.py
"""
from __future__ import annotations

import dataclasses
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import config
from backtest.data.universe import build_universe, competitive_subset
from backtest.dynamic.ledger import CapitalLedger, RealizedSpendCommitmentSource
from backtest.model.budget import estimate_budget_2026
from backtest.model import ceiling as ceiling_mod
from backtest.optimizer.allocator import (
    optimize_nonlinear, nonlinear_expected_seats_at_party_dollars, _precompute_race_arrays, _p_win_vec,
)
from run_backtest import load_processed_artifacts, build_dummy_factor_model  # type: ignore

BUDGET_2026 = estimate_budget_2026()


def _thresholds() -> dict[str, float]:
    out = {}
    for cycle in (2024, 2022):
        races = build_universe(cycle=cycle)
        comp = competitive_subset(races)
        totals = np.array(sorted(r.cand_d_total + r.r_total for r in comp))
        for p in (10, 25, 50, 75):
            out[f"{cycle}-p{p}"] = float(np.percentile(totals, p))
    return out


def _2026_gain_at_threshold(races, coef, sigma_model, cov_matrix, committed, committed_total, reference):
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
        np.array([r.cand_d_total for r in races]),
        np.array([r.r_total for r in races]), reference,
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
        races, coef, sigma_model, model_total_party, floor_maturity=floor_maturity_eval)
    dccc_seats = nonlinear_expected_seats_at_party_dollars(
        races, coef, sigma_model, dccc_scaled_party, floor_maturity=floor_maturity_eval)

    arrays = _precompute_race_arrays(races, coef, sigma_model, eta=0.0, floor_maturity=floor_maturity_eval)
    delta = _p_win_vec(model_total_party, arrays) - _p_win_vec(dccc_scaled_party, arrays)
    cook = np.array([r.cook_rating for r in races])
    toss_up_gain = float(delta[cook == "Toss-Up"].sum())
    safe_r_gain = float(delta[cook == "Safe R"].sum())

    return model_seats - dccc_seats, toss_up_gain, safe_r_gain


def main() -> None:
    _, coef, sigma_model = load_processed_artifacts(config.processed_path())
    races2026 = build_universe(cycle=2026)
    gb = config.generic_ballot_for_cycle(2026)
    factor_model = build_dummy_factor_model(races2026, gb)
    cov_matrix = factor_model.race_covariance()

    as_of = datetime.now(timezone.utc).date()
    src = RealizedSpendCommitmentSource(cycle=2026, party="D")
    committed = src.committed_capital(0, as_of, races2026)
    committed_total = sum(committed.values())

    thresholds = _thresholds()
    print("Candidate reference thresholds (competitive-race combined D+R spend, cycle-final):")
    for label, val in thresholds.items():
        marker = "  <- current default" if abs(val - config.floor_maturity_reference_dollars()) < 1 else ""
        print(f"  {label}: ${val:,.0f}{marker}")

    print(f"\n[1/2] 2026 live gain at each threshold (uncorrected baseline = +7.90):")
    rows = []
    for label, ref in thresholds.items():
        gain, toss_up, safe_r = _2026_gain_at_threshold(
            races2026, coef, sigma_model, cov_matrix, committed, committed_total, ref,
        )
        rows.append({"threshold_label": label, "threshold_dollars": ref,
                      "gain_2026": gain, "toss_up_contribution": toss_up, "safe_r_contribution": safe_r})
        print(f"  {label:12s} (${ref:>11,.0f}): gain={gain:+.3f}  "
              f"Toss-Up contrib={toss_up:+.3f}  Safe R contrib={safe_r:+.3f}")

    df = pd.DataFrame(rows)
    gain_range = df["gain_2026"].max() - df["gain_2026"].min()
    print(f"\n  Gain range across all 8 thresholds: {df['gain_2026'].min():+.3f} to "
          f"{df['gain_2026'].max():+.3f} (spread {gain_range:.3f} seats)")
    print(f"  2024-derived vs 2022-derived thresholds (same percentile) -- cross-calibration check:")
    for p in (10, 25, 50, 75):
        g24 = df.loc[df.threshold_label == f"2024-p{p}", "gain_2026"].iloc[0]
        g22 = df.loc[df.threshold_label == f"2022-p{p}", "gain_2026"].iloc[0]
        print(f"    p{p}: 2024-calibrated={g24:+.3f}  2022-calibrated={g22:+.3f}  diff={g24-g22:+.3f}")

    out_path = config.outputs_path() / "floor_maturity_reference_sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ── 2. Historical-checkpoint neutrality check (already resolved case) ──
    print(f"\n[2/2] Historical-checkpoint neutrality check (2022/2024 real-world eval, "
          f"already correct without this correction -- confirming cross-calibrated "
          f"thresholds don't reintroduce distortion)...")
    from backtest.dynamic.simulate import (
        _reconstruct_races_at, _static_floor_totals, _has_dated_candidate_panel, _candidate_fallback_totals,
    )
    from datetime import date
    from backtest.dynamic.periods import fec_quarterly_periods

    ELECTION_DAY = {2022: date(2022, 11, 8), 2024: date(2024, 11, 5)}
    checkpoint_rows = []
    for eval_cycle, calib_cycle in [(2024, 2022), (2022, 2024)]:
        processed = config.processed_path() if eval_cycle == 2024 else Path("data/processed_oos_2020")
        _, coef_c, sigma_model_c = load_processed_artifacts(processed)
        base_races = build_universe(cycle=eval_cycle)
        budget_final = sum(r.d_total for r in base_races)
        party_budget_final = sum(r.d_total - r.cand_d_total for r in base_races)
        from backtest.model.win_prob import compute_outputs_batch
        outputs_final = compute_outputs_batch(base_races, coef_c, sigma_model_c)
        dccc_final_expected_seats = float(sum(o.p_win for o in outputs_final))
        factor_model_c = build_dummy_factor_model(base_races, config.generic_ballot_for_cycle(eval_cycle))
        cov_matrix_c = factor_model_c.race_covariance()
        cap_baseline = config.optimizer_cfg()["cap_regimes"][-1]

        cross_ref = thresholds[f"{calib_cycle}-p25"]
        static_totals = _static_floor_totals(eval_cycle)
        use_dated = _has_dated_candidate_panel(eval_cycle)
        fallback_totals = None if use_dated else _candidate_fallback_totals(eval_cycle)
        # Just the earliest checkpoint -- the one that showed the largest
        # distortion in Section 3.1's original (superseded) test.
        earliest = min(
            (p for p in fec_quarterly_periods(eval_cycle) if p.period_date <= ELECTION_DAY[eval_cycle]),
            key=lambda p: p.period_date,
        )
        races_t = _reconstruct_races_at(
            earliest.index, earliest.period_date, eval_cycle, base_races, static_totals,
            use_dated, fallback_totals,
        )
        floor_maturity_t = ceiling_mod.maturity(
            np.array([r.cand_d_total for r in races_t]), np.array([r.r_total for r in races_t]), cross_ref,
        )
        result_t = optimize_nonlinear(
            races_t, coef_c, sigma_model_c, budget_final, cov_matrix_c, 0.0, cap_baseline,
            party_budget=party_budget_final, floor_maturity=floor_maturity_t,
        )
        checkpoint_floor = np.array([r.cand_d_total for r in races_t])
        recommended_party = result_t.allocations - checkpoint_floor
        floor_maturity_eval_t = ceiling_mod.maturity(
            np.array([r.cand_d_total for r in base_races]), np.array([r.r_total for r in base_races]), cross_ref,
        )
        real_world_seats = nonlinear_expected_seats_at_party_dollars(
            base_races, coef_c, sigma_model_c, recommended_party, floor_maturity=floor_maturity_eval_t,
        )
        real_world_gain = real_world_seats - dccc_final_expected_seats
        checkpoint_rows.append({
            "eval_cycle": eval_cycle, "calibrated_on": calib_cycle,
            "checkpoint_date": earliest.period_date.isoformat(),
            "real_world_gain_with_cross_calibrated_maturity": real_world_gain,
        })
        print(f"  eval={eval_cycle}, threshold calibrated on {calib_cycle} (${cross_ref:,.0f}), "
              f"earliest checkpoint ({earliest.period_date}): real-world gain={real_world_gain:+.3f} "
              f"(uncorrected, from Section 3.4: {-1.2 if eval_cycle==2024 else -0.9:+.1f})")

    pd.DataFrame(checkpoint_rows).to_csv(
        config.outputs_path() / "floor_maturity_cross_calibration_checkpoint.csv", index=False)
    print(f"Saved: {config.outputs_path() / 'floor_maturity_cross_calibration_checkpoint.csv'}")


if __name__ == "__main__":
    main()
