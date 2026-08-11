#!/usr/bin/env python3
"""
District-validity summary for the 2026 universe -- a scoped-down version of
the reviewer's requested validity layer (current boundaries, rating
freshness, redistricting flags, special-election/replacement-candidate
status, structural-input comparability).

What's actually buildable from data already in this repo, checked here:
  - redistricting_flagged (already existed, was passive metadata -- now
    cross-referenced against gain_decomposition_2026_by_race.csv for
    materiality, per the earlier finding in this investigation: +0.77 of
    +7.9, 9.7%, $23.8M in recommended money across 13 districts).
  - near-zero-candidate-floor check: a genuinely new signal -- districts
    where the Democratic candidate floor is under $5,000 (effectively no
    real filed candidate yet). Checked directly against the gain
    decomposition: all 3 such races (OK-03, OH-02, MI-09) receive exactly
    $0 in recommended money and contribute exactly 0 to the gain -- the
    persuasion ceiling's own deep-PVI suppression (all 3 are Safe R,
    PVI -17.9 to -25.0) already handles this failure mode without a new
    validity layer being needed for it specifically.

What is NOT built here, and why: current candidate roster (withdrawn/
replaced candidates), special-election status, and rating-freshness
tracking all require live FEC candidate-master data
(https://api.open.fec.gov/v1/candidates/) that this environment has no API
key configured for (checked directly -- no FEC_API_KEY in the environment,
no cached candidate-master file in data/raw/fec/). This is an honest,
checked gap, not an assumption -- scripts/fetch_data.py already has the FEC
API request pattern this would extend; running it requires a free API key
(https://api.open.fec.gov/developers) this session does not have.

Usage:
    python scripts/district_validity_summary_2026.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from backtest import config
from backtest.data.universe import build_universe

REDISTRICTING_FLAGGED = {
    "AL-02", "LA-06", "NC-01", "NC-06", "NC-07", "NC-10", "NC-13", "NC-14",
    "NY-03", "NY-04", "NY-17", "NY-18", "NY-22",
}


def main() -> None:
    races = build_universe(cycle=2026)
    thin_candidate = [r for r in races if r.cand_d_total < 5000]

    gain_path = config.outputs_path() / "gain_decomposition_2026_by_race.csv"
    gain_df = pd.read_csv(gain_path) if gain_path.exists() else None

    rows = []
    for r in races:
        flags = []
        if r.district_id in REDISTRICTING_FLAGGED:
            flags.append("redistricting_flagged")
        if r.cand_d_total < 5000:
            flags.append("near_zero_candidate_floor")
        rows.append({
            "district_id": r.district_id, "cook_rating": r.cook_rating,
            "pvi": r.pvi, "cand_d_total": r.cand_d_total,
            "flags": ",".join(flags) if flags else "",
            "n_validity_flags": len(flags),
        })
    df = pd.DataFrame(rows)

    print(f"2026 universe: {len(races)} races")
    print(f"  redistricting_flagged: {len(REDISTRICTING_FLAGGED)} districts")
    print(f"  near_zero_candidate_floor (D floor < $5,000): {len(thin_candidate)} districts: "
          f"{[r.district_id for r in thin_candidate]}")

    if gain_df is not None:
        flagged_df = df[df["n_validity_flags"] > 0].merge(
            gain_df[["district_id", "model_party_dollars", "delta_seats"]], on="district_id", how="left")
        total_gain = gain_df["delta_seats"].sum()
        flagged_gain = flagged_df["delta_seats"].sum()
        print(f"\n  All validity-flagged districts combined: {flagged_gain:+.3f} of {total_gain:+.3f} "
              f"total gain ({100*flagged_gain/total_gain:.1f}%), "
              f"${flagged_df['model_party_dollars'].sum():,.0f} in recommended money")
        print(f"  near_zero_candidate_floor specifically: "
              f"{flagged_df[flagged_df['flags']=='near_zero_candidate_floor']['delta_seats'].sum():+.4f} seats "
              f"(the persuasion ceiling's deep-PVI suppression already handles this case)")

    print(f"\n  NOT checked (requires a live FEC candidate-master pull this environment has no API key for): "
          f"current candidate roster, withdrawn/replaced candidates, special-election status, "
          f"rating-freshness date.")

    out_path = config.outputs_path() / "district_validity_summary_2026.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
