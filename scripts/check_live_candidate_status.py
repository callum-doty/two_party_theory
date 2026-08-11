#!/usr/bin/env python3
"""
Live FEC candidate-status check -- the piece of the district-validity layer
(district_validity_summary_2026.py) that was previously blocked by no
configured API key. Checks the CURRENT (as of today) FEC candidate_status
and candidate_inactive flags for the districts that matter most: the top
gain-contributing races (gain_decomposition_2026_by_race.csv) plus every
already-flagged validity race (redistricting-flagged, near-zero-candidate-
floor), rather than all 434*2 candidates -- scoped to what's decision-
relevant, not exhaustive.

FEC candidate_status codes: C = statutory candidate (active), F = candidate
for a future election, N = not yet a statutory candidate (has filed
paperwork but hasn't crossed the $5,000 threshold or otherwise qualified),
P = candidate in a prior cycle only.

Does NOT persist the API key anywhere -- pass it only as a CLI argument at
invocation; nothing in this script or its output files contains it.

Usage:
    python scripts/check_live_candidate_status.py --api-key YOUR_KEY
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from backtest import config

FEC_API_BASE = "https://api.open.fec.gov/v1"


def lookup_candidate(candidate_id: str, api_key: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            resp = requests.get(
                f"{FEC_API_BASE}/candidate/{candidate_id}/", params={"api_key": api_key}, timeout=30,
            )
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            return results[0] if results else None
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    reports = pd.read_csv(config.raw_path("fec") / "candidate_periodic_reports_2026.csv", dtype=str)
    latest_id = reports.sort_values("coverage_end_date").groupby(["district_id", "party"])["fec_candidate_id"].last()

    gain_df = pd.read_csv(config.outputs_path() / "gain_decomposition_2026_by_race.csv")
    top_races = gain_df.reindex(gain_df["delta_seats"].abs().sort_values(ascending=False).index).head(args.top_n)

    validity_df = pd.read_csv(config.outputs_path() / "district_validity_summary_2026.csv")
    flagged_races = validity_df[validity_df["n_validity_flags"] > 0]["district_id"]

    check_districts = sorted(set(top_races["district_id"]) | set(flagged_races))
    print(f"Checking live candidate status for {len(check_districts)} districts "
          f"(top {args.top_n} by |gain| + all validity-flagged), both parties...\n")

    rows = []
    for did in check_districts:
        for party in ("D", "R"):
            key = (did, party)
            if key not in latest_id.index:
                continue
            cand_id = latest_id.loc[key]
            info = lookup_candidate(cand_id, args.api_key)
            time.sleep(0.05)
            if info is None:
                rows.append({"district_id": did, "party": party, "candidate_id": cand_id,
                             "name": None, "candidate_status": None, "candidate_inactive": None,
                             "note": "lookup failed"})
                continue
            status = info.get("candidate_status")
            inactive = info.get("candidate_inactive")
            flag = "" if (status == "C" and not inactive) else "REVIEW"
            rows.append({
                "district_id": did, "party": party, "candidate_id": cand_id,
                "name": info.get("name"), "candidate_status": status,
                "candidate_inactive": inactive, "note": flag,
            })
            if flag:
                print(f"  {did} ({party}): {info.get('name')} -- status={status}, "
                      f"inactive={inactive}  <-- REVIEW")

    df = pd.DataFrame(rows)
    n_review = (df["note"] == "REVIEW").sum()
    print(f"\n{len(df)} candidates checked, {n_review} flagged for review "
          f"(status != 'C' active, or marked inactive)")

    out_path = config.outputs_path() / "live_candidate_status_check_2026.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
