#!/usr/bin/env python3
"""
Two robustness checks on the 2026 live +7.9-seat read
(decompose_retrospective_gain_2026_live.py), prompted directly by two
concerns: (1) the DCCC-observed-scaled baseline is built from only 18 of
434 races with any real committed $ yet -- how much does that number swing
just from which 18 races happen to be in the sample? (2) the live generic
ballot (D+5.02, a favorable environment for Democrats) enters every race's
mu_i via alpha3*GB -- how much of the +7.9 is the model finding genuine
targeting inefficiency vs. simply reflecting a favorable national mood?

1. Bootstrap CI on the DCCC-observed-scaled baseline: resample the 18
   committed races with replacement (1000 draws), rescale each resample to
   the full budget, and recompute the gain. The model side is NOT
   resampled -- it's evaluated on real floor data across the full 434-race
   universe, so the sampling noise lives entirely in the DCCC baseline.

2. Generic-ballot sensitivity: rerun the identical computation (today's
   real floors and committed pattern, held fixed) with generic_ballot
   overridden to each value in config.yaml's historical generic_ballot_by_
   cycle range, instead of the live D+5.02 estimate. This isolates how much
   of the gain is coming specifically from the current national environment
   versus the underlying spending-targeting mismatch Paper I's efficiency
   tests measure, which is a GB-independent, structural finding.

Usage:
    python scripts/robustness_2026_live_gain.py
"""
from __future__ import annotations

import dataclasses
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
from backtest.dynamic.ledger import CapitalLedger, RealizedSpendCommitmentSource
from backtest.model.budget import estimate_budget_2026
from backtest.optimizer.allocator import optimize_nonlinear, nonlinear_expected_seats_at_party_dollars
from run_backtest import load_processed_artifacts, build_dummy_factor_model  # type: ignore

BUDGET_2026 = estimate_budget_2026()
CAP_FRACTION = 0.15
N_BOOTSTRAP = 1000
SEED = 42

# Historical range, config.yaml's generic_ballot_by_cycle -- the most
# Democratic-favorable and most Republican-favorable points on record,
# plus the two most recent cycles, bracketing the live D+5.02 estimate.
GB_SWEEP = {
    "2014 (D-5.8, most R-favorable on record)": -5.8,
    "2022 (D-1.0)": -1.0,
    "2024 (D-1.2)": -1.2,
    "neutral (D+0.0)": 0.0,
    "2026 live (D+5.02, current estimate)": 5.02,
    "2020 (D+7.0)": 7.0,
    "2018 (D+8.6, most D-favorable on record)": 8.6,
}


def _gain_at(races, coef, sigma_model, committed_by_race, committed_total, gb_override=None):
    """Recompute the DCCC-scaled-vs-model-optimal gain, optionally with
    every race's generic_ballot overridden. `committed_by_race` may itself
    be a bootstrap resample rather than the real ledger -- this function is
    reused by both checks."""
    if gb_override is not None:
        races = [dataclasses.replace(r, generic_ballot=gb_override) for r in races]

    factor_model = build_dummy_factor_model(races, gb_override if gb_override is not None
                                              else races[0].generic_ballot)
    cov_matrix = factor_model.race_covariance()

    ledger = CapitalLedger(
        period=0, total_budget=BUDGET_2026,
        committed_by_race=committed_by_race, committed_total=committed_total,
        deployable_total=BUDGET_2026 - committed_total,
    )
    races_with_floor = ledger.apply_to_races(races)
    result = optimize_nonlinear(
        races_with_floor, coef, sigma_model, budget=BUDGET_2026, cov_matrix=cov_matrix,
        gamma=0.0, cap_fraction=CAP_FRACTION, party_budget=ledger.deployable_total,
    )
    floor = ledger.deployable_floor_for(races)
    model_total_party = (result.allocations - floor) + np.array(
        [committed_by_race.get(r.district_id, 0.0) for r in races]
    )
    model_seats = nonlinear_expected_seats_at_party_dollars(races, coef, sigma_model, model_total_party)

    dccc_scaled_party = np.array([committed_by_race.get(r.district_id, 0.0) for r in races]) \
        * (BUDGET_2026 / committed_total)
    dccc_seats = nonlinear_expected_seats_at_party_dollars(races, coef, sigma_model, dccc_scaled_party)

    return model_seats - dccc_seats, model_seats, dccc_seats


def main() -> None:
    _, coef, sigma_model = load_processed_artifacts(config.processed_path())
    races = build_universe(cycle=2026)
    as_of = datetime.now(timezone.utc).date()

    src = RealizedSpendCommitmentSource(cycle=2026, party="D")
    committed = src.committed_capital(0, as_of, races)
    nonzero = {k: v for k, v in committed.items() if v > 0}
    n_committed = len(nonzero)
    print(f"As of {as_of}: {n_committed} of {len(races)} races have real committed party $\n")

    # ── 1. Bootstrap CI on the DCCC-observed-scaled baseline ───────────────
    print(f"[1/2] Bootstrap ({N_BOOTSTRAP} resamples of the {n_committed} committed races)...")
    rng = np.random.default_rng(SEED)
    committed_ids = list(nonzero.keys())
    committed_vals = np.array([nonzero[d] for d in committed_ids])

    # Model side computed once -- it's evaluated on the full real universe,
    # not the thin 18-race sample, so it isn't part of what's being resampled.
    base_gain, base_model_seats, base_dccc_seats = _gain_at(
        races, coef, sigma_model, committed, sum(committed_vals),
    )
    print(f"  Point estimate (as reported before): gain={base_gain:+.3f} "
          f"(model={base_model_seats:.3f}, DCCC-scaled={base_dccc_seats:.3f})")

    boot_gains = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_committed, size=n_committed)
        boot_dict: dict[str, float] = {}
        for i in idx:
            did = committed_ids[i]
            boot_dict[did] = boot_dict.get(did, 0.0) + committed_vals[i]
        boot_total = sum(boot_dict.values())
        if boot_total <= 0:
            continue
        dccc_scaled_party = np.array([boot_dict.get(r.district_id, 0.0) for r in races]) \
            * (BUDGET_2026 / boot_total)
        dccc_seats_b = nonlinear_expected_seats_at_party_dollars(races, coef, sigma_model, dccc_scaled_party)
        boot_gains.append(base_model_seats - dccc_seats_b)

    boot_gains = np.array(boot_gains)
    p5, p25, p50, p75, p95 = np.percentile(boot_gains, [5, 25, 50, 75, 95])
    print(f"  Bootstrap distribution of the gain (model side held fixed at {base_model_seats:.3f}):")
    print(f"    5th pct: {p5:+.2f}   25th: {p25:+.2f}   median: {p50:+.2f}   "
          f"75th: {p75:+.2f}   95th pct: {p95:+.2f}")
    print(f"    90% CI: [{p5:+.2f}, {p95:+.2f}] seats -- width {p95 - p5:.2f} seats\n")

    # ── 2. Generic-ballot sensitivity ───────────────────────────────────────
    print("[2/2] Generic-ballot sensitivity (today's real floors/committed pattern held fixed)...")
    gb_rows = []
    for label, gb_val in GB_SWEEP.items():
        gain, model_seats, dccc_seats = _gain_at(
            races, coef, sigma_model, committed, sum(committed_vals), gb_override=gb_val,
        )
        gb_rows.append({"label": label, "gb": gb_val, "model_seats": model_seats,
                         "dccc_seats": dccc_seats, "gain": gain})
        print(f"  {label:45s} GB={gb_val:+5.2f}: model={model_seats:6.2f}  "
              f"dccc={dccc_seats:6.2f}  gain={gain:+6.2f}")

    gb_df = pd.DataFrame(gb_rows)
    gb_range = gb_df["gain"].max() - gb_df["gain"].min()
    print(f"\n  Gain range across the full historical GB sweep: {gb_df['gain'].min():+.2f} to "
          f"{gb_df['gain'].max():+.2f} (spread {gb_range:.2f} seats)")

    out_dir = config.outputs_path()
    pd.DataFrame({
        "boot_gain": boot_gains,
    }).to_csv(out_dir / "robustness_2026_bootstrap_gains.csv", index=False)
    gb_df.to_csv(out_dir / "robustness_2026_gb_sensitivity.csv", index=False)
    print(f"\nSaved: {out_dir / 'robustness_2026_bootstrap_gains.csv'}")
    print(f"Saved: {out_dir / 'robustness_2026_gb_sensitivity.csv'}")


if __name__ == "__main__":
    main()
