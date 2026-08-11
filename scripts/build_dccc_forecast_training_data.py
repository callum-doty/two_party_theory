#!/usr/bin/env python3
"""
Build the training dataset for a genuine DCCC-forecast model
f(x_i,t) -> predicted eventual DCCC party-dollar allocation, per the
reviewer's central point (docs/retrospective_vs_realtime_investigation.md
Section 11): the 2026 comparisons so far use a naive proportional scale-up
of DCCC's thin current pattern as the baseline, not a real forecast of
where DCCC will end up.

One (x_i,t, y_i) training row per race per historical cycle, using the
SAME checkpoint definition every cycle: 91 days before that cycle's
Election Day -- matching where the live 2026 decision actually sits today,
so a model trained this way is directly applicable to 2026 without any
horizon mismatch. Reuses the dated-reconstruction machinery already built
and validated earlier in this investigation (dynamic/simulate.py's
_reconstruct_races_at, now bug-fixed) -- no new data acquisition.

Features (x_i,t), each real and available as of the checkpoint:
  pvi, abs_pvi, incumb_status, cook_rating (Cook-tier ordinal),
  cand_ratio_t = candidate committee spend to date / that cycle's final
    party budget (normalizes across cycles of very different total size),
  r_ratio_t = opponent spend to date / that cycle's final party budget,
  generic_ballot (per-cycle constant, same simplification used throughout
    this project -- Section 3.3 of docs/paper2_draft.md).

Targets (y_i), from that SAME cycle's actual, complete, final outcome:
  funded = 1 if DCCC's final party $ for the race exceeds 0.1% of that
    cycle's party budget, else 0 (selection-stage target)
  party_share_final = DCCC's final party $ / that cycle's final party
    budget (intensity-stage target, defined only where funded==1)

Usage:
    python scripts/build_dccc_forecast_training_data.py

Output: data/processed/dccc_forecast_training_data.csv
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import config
from backtest.data.universe import build_universe
from backtest.dynamic.simulate import (
    _reconstruct_races_at, _static_floor_totals, _has_dated_candidate_panel, _candidate_fallback_totals,
)

ELECTION_DAY = {
    2012: date(2012, 11, 6), 2014: date(2014, 11, 4), 2016: date(2016, 11, 8),
    2018: date(2018, 11, 6), 2020: date(2020, 11, 3), 2022: date(2022, 11, 8),
    2024: date(2024, 11, 5),
}
CHECKPOINT_DAYS_OUT = 91  # matches where the live 2026 decision sits today
COOK_ORDINAL = {  # symmetric D-favorable-to-R-favorable scale, matches config.yaml's category ordering
    "Safe D": 3, "Likely D": 2, "Lean D": 1, "Toss-Up": 0, "Lean R": -1, "Likely R": -2, "Safe R": -3,
}


def build_cycle_rows(cycle: int) -> list[dict]:
    election_day = ELECTION_DAY[cycle]
    checkpoint_date = election_day - timedelta(days=CHECKPOINT_DAYS_OUT)

    base_races = build_universe(cycle=cycle)
    party_budget_final = sum(r.d_total - r.cand_d_total for r in base_races)
    final_party = {r.district_id: r.d_total - r.cand_d_total for r in base_races}

    static_totals = _static_floor_totals(cycle)
    use_dated = _has_dated_candidate_panel(cycle)
    if not use_dated:
        print(f"  WARNING: cycle {cycle} has no dated candidate panel -- skipping")
        return []
    fallback_totals = None

    races_t = _reconstruct_races_at(
        0, checkpoint_date, cycle, base_races, static_totals, use_dated, fallback_totals,
    )

    rows = []
    for race in races_t:
        cand_ratio_t = race.cand_d_total / party_budget_final if party_budget_final > 0 else 0.0
        r_ratio_t = race.r_total / party_budget_final if party_budget_final > 0 else 0.0
        party_share_final = final_party[race.district_id] / party_budget_final if party_budget_final > 0 else 0.0
        rows.append({
            "cycle": cycle,
            "district_id": race.district_id,
            "checkpoint_date": checkpoint_date.isoformat(),
            "pvi": race.pvi,
            "abs_pvi": abs(race.pvi),
            "incumb_status": race.incumb_status,
            "cook_ordinal": COOK_ORDINAL.get(race.cook_rating, 0),
            "cand_ratio_t": cand_ratio_t,
            "r_ratio_t": r_ratio_t,
            "generic_ballot": race.generic_ballot,
            "party_share_final": party_share_final,
            "funded": int(party_share_final > 0.001),
        })
    return rows


def main() -> None:
    all_rows = []
    for cycle in sorted(ELECTION_DAY):
        print(f"Building checkpoint features for cycle {cycle} "
              f"({CHECKPOINT_DAYS_OUT} days before {ELECTION_DAY[cycle]})...")
        rows = build_cycle_rows(cycle)
        print(f"  {len(rows)} races, {sum(r['funded'] for r in rows)} funded "
              f"({100*sum(r['funded'] for r in rows)/max(len(rows),1):.1f}%)")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = config.processed_path() / "dccc_forecast_training_data.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{len(df)} total training rows across {df['cycle'].nunique()} cycles")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
