#!/usr/bin/env python3
"""
Live 2026 generic-ballot polling ingestion (Paper II §7, Phase 4B).

Fetches individual generic-ballot poll results from VoteHub's free public
API (https://api.votehub.com — no API key required) and persists both the
raw poll list and a simple trailing-average summary.

IMPORTANT — this data is a diagnostic/monitoring signal, not a margin-model
input. Per the Paper II implementation plan's Phase 4A finding: `coef.alpha3`
(the generic-ballot coefficient in `model.margin.estimate_from_panel`) is
identified entirely from *between-cycle* variation — one static GB value per
historical cycle, identical across every race in that cycle. It has never
been estimated against within-cycle GB movement. Feeding this script's
day-to-day trailing average into `RaceRecord.generic_ballot` (or any
dynamic/ `generic_ballot_t` field) as a moving quantity would apply alpha3
to an estimand it was never fit against — the same issue documented for the
historical harness in `dynamic/simulate.py::_reconstruct_races_at`, and it
applies just as much to a single live 2026 cycle. Re-estimating the margin
model on a panel with within-cycle GB observations (Option B in the plan)
would be required before this series could be a legitimate model input.
This script exists to give practitioners a live situational-awareness
signal and to feed future non-alpha3 uses (e.g. a Bayesian/Kalman
StateUpdater that models GB as a separate information channel), not to be
wired into the existing static model.

Usage:
    python scripts/fetch_polling.py
    python scripts/fetch_polling.py --window-days 21
    python scripts/fetch_polling.py --dry-run   # skip write

Outputs:
    data/live/generic_ballot_polls.csv    — raw poll-level rows
    data/live/generic_ballot_summary.json — trailing-average summary (diagnostic only)
    data/live/polling_fetch_log.jsonl     — append-only fetch audit trail
"""

from __future__ import annotations
import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("fetch_polling")

VOTEHUB_API_BASE = "https://api.votehub.com"
LIVE_DIR = Path(__file__).parent.parent / "data" / "live"


def fetch_generic_ballot_polls(subject: str = "2026") -> pd.DataFrame:
    """
    Fetch every generic-ballot poll VoteHub has for `subject` (default: the
    current cycle). No API key or pagination needed — the endpoint returns
    the full list in one response.

    Returns DataFrame with columns: pollster, start_date, end_date,
    sample_size, population, dem_pct, rep_pct, gb (dem_pct - rep_pct), url.
    """
    url = f"{VOTEHUB_API_BASE}/polls"
    params = {"poll_type": "generic-ballot"}
    logger.info(f"Fetching generic-ballot polls from {url} (subject={subject})…")
    resp = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=30)
    resp.raise_for_status()
    records = resp.json()
    logger.info(f"Received {len(records)} poll records")

    rows = []
    for r in records:
        if subject is not None and r.get("subject") != subject:
            continue
        answers = {a["choice"]: a["pct"] for a in r.get("answers", [])}
        dem = answers.get("Dem")
        rep = answers.get("Rep")
        if dem is None or rep is None:
            continue
        rows.append({
            "pollster": r.get("pollster"),
            "start_date": r.get("start_date"),
            "end_date": r.get("end_date"),
            "sample_size": r.get("sample_size"),
            "population": r.get("population"),
            "dem_pct": dem,
            "rep_pct": rep,
            "gb": dem - rep,
            "url": r.get("url"),
        })

    df = pd.DataFrame(rows)
    if len(df):
        df["end_date"] = pd.to_datetime(df["end_date"])
        df = df.sort_values("end_date").reset_index(drop=True)
    return df


def fetch_house_district_polls() -> pd.DataFrame:
    """
    Fetch every live 2026 district-level U.S. House race poll VoteHub has
    (poll_type="us-representative"), the same free, no-key API this module
    already uses for generic-ballot polls.

    New this session (Paper III's feature-catalog reviewer feedback, "Tier
    1" district polling): confirmed against live VoteHub data that this
    poll_type exists and returns real per-district subjects (e.g.
    "2026 PA-07"), NOT just national/generic-ballot polls. Checked:
    coverage is real but sparse (a small minority of House districts have
    any poll at all at any given time) and, more importantly, is LIVE
    2026-ONLY — no historical (2022/2024) district-level panel exists at
    this endpoint. This means district polling is usable as a live
    monitoring/diagnostic signal (this function's purpose) but NOT as a
    dataset to fit a historical epsilon_i,t process against — Paper III
    §6.2's conclusion that district-level idiosyncratic uncertainty cannot
    currently be estimated from data stands unchanged; this adds a new
    *live* state feature, not a new *historical estimation* dataset.

    Same IMPORTANT caveat as fetch_generic_ballot_polls's module docstring:
    this is a diagnostic signal, not wired into RaceState.mu_hat or any
    structural model input — there is no historical panel to validate a
    within-cycle coefficient against, so treating a poll average as if it
    were a calibrated model input would be exactly the same mistake
    already flagged there for generic-ballot polls.

    Returns DataFrame with columns: district_id, pollster, start_date,
    end_date, sample_size, margin (leading candidate's pct minus the
    next candidate's pct — sign not party-aligned, since VoteHub's
    `answers` list is by candidate name, not party; see
    house_district_poll_summary for the party-aligned trailing mean),
    dem_pct, rep_pct (None when a poll's answers can't be matched to a
    major-party candidate, e.g. a jungle primary or independent race), url.
    """
    url = f"{VOTEHUB_API_BASE}/polls"
    params = {"poll_type": "us-representative"}
    logger.info(f"Fetching district-level House polls from {url}…")
    resp = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=30)
    resp.raise_for_status()
    records = resp.json()
    logger.info(f"Received {len(records)} district-level poll records")

    rows = []
    for r in records:
        seat = r.get("seat_name")
        if not seat:
            continue
        answers = sorted(r.get("answers", []), key=lambda a: a.get("pct", 0), reverse=True)
        if len(answers) < 2:
            continue
        rows.append({
            "district_id": seat,
            "pollster": r.get("pollster"),
            "start_date": r.get("start_date"),
            "end_date": r.get("end_date"),
            "sample_size": r.get("sample_size"),
            "leader_pct": answers[0]["pct"],
            "second_pct": answers[1]["pct"],
            "margin": answers[0]["pct"] - answers[1]["pct"],
            "url": r.get("url"),
        })

    df = pd.DataFrame(rows)
    if len(df):
        df["end_date"] = pd.to_datetime(df["end_date"])
        df = df.sort_values(["district_id", "end_date"]).reset_index(drop=True)
    return df


def house_district_poll_summary(polls: pd.DataFrame, as_of: date) -> dict:
    """
    Per-district summary as of `as_of`: poll_mean (mean margin, most-recent-
    leader convention -- NOT signed to a party, since VoteHub's answer
    ordering is by candidate not party), poll_sigma (spread across polls,
    None if only one poll), poll_n (count), poll_trend (slope of margin
    over time via a simple linear fit against days-since-first-poll, None
    if fewer than 2 polls). This is the per-race analog of
    trailing_average_summary's generic-ballot rollup, feeding
    RaceState.poll_mean_t/poll_sigma_t/poll_n_t/poll_trend_t.
    """
    if not len(polls):
        return {}
    cutoff = pd.Timestamp(as_of)
    result: dict[str, dict] = {}
    for district_id, sub in polls[polls["end_date"] <= cutoff].groupby("district_id"):
        n = len(sub)
        entry = {
            "poll_n": int(n),
            "poll_mean": float(sub["margin"].mean()),
            "poll_sigma": float(sub["margin"].std()) if n > 1 else None,
            "latest_end_date": sub["end_date"].max().date().isoformat(),
        }
        days = (sub["end_date"] - sub["end_date"].min()).dt.days.astype(float)
        if n >= 2 and days.nunique() > 1:
            # nunique()>1 guard: polyfit's design matrix is singular (SVD
            # fails to converge) when every poll in the window shares the
            # same end_date -- e.g. two polls released the same day -- since
            # there is then zero variance in the x (days) column to fit a
            # slope against.
            slope = np.polyfit(days, sub["margin"].astype(float), 1)[0]
            entry["poll_trend"] = float(slope)
        else:
            entry["poll_trend"] = None
        result[district_id] = entry
    return result


def trailing_average_summary(polls: pd.DataFrame, window_days: int, as_of: date) -> dict:
    """
    Simple, unweighted trailing mean of `gb` over the last `window_days`
    ending at `as_of`. Deliberately the simplest possible aggregation (no
    pollster-quality weighting, no recency decay) — see the module
    docstring for why this stays a diagnostic rather than a calibrated
    polling average.
    """
    if not len(polls):
        return {"as_of": as_of.isoformat(), "window_days": window_days,
                "n_polls": 0, "gb_trailing_avg": None}

    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=window_days)
    window = polls[(polls["end_date"] > cutoff) & (polls["end_date"] <= pd.Timestamp(as_of))]
    if not len(window):
        return {"as_of": as_of.isoformat(), "window_days": window_days,
                "n_polls": 0, "gb_trailing_avg": None}

    return {
        "as_of": as_of.isoformat(),
        "window_days": window_days,
        "n_polls": int(len(window)),
        "gb_trailing_avg": float(window["gb"].mean()),
        "gb_trailing_min": float(window["gb"].min()),
        "gb_trailing_max": float(window["gb"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch live generic-ballot and district-level polls from VoteHub "
                     "(diagnostic only — see module docstring)"
    )
    parser.add_argument("--subject", type=str, default="2026",
                        help="VoteHub 'subject' (election cycle) to fetch (default: 2026)")
    parser.add_argument("--window-days", type=int, default=14,
                        help="Trailing-average window in days (default: 14)")
    parser.add_argument("--skip-district-polls", action="store_true",
                        help="Skip the district-level (us-representative) fetch; "
                             "generic-ballot only, the original behavior of this script.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and print the summary but don't write output files")
    args = parser.parse_args()

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    as_of = datetime.now(timezone.utc).date()

    polls = fetch_generic_ballot_polls(subject=args.subject)
    summary = trailing_average_summary(polls, args.window_days, as_of)

    print(f"\n{'─'*60}")
    print(f" GENERIC BALLOT — DIAGNOSTIC ONLY, NOT A MODEL INPUT")
    print(f" {summary['n_polls']} polls in trailing {args.window_days}d as of {summary['as_of']}")
    if summary["gb_trailing_avg"] is not None:
        print(f" Trailing average (Dem - Rep): {summary['gb_trailing_avg']:+.2f}")
    print(f"{'─'*60}\n")

    district_polls = pd.DataFrame()
    district_summary: dict = {}
    if not args.skip_district_polls:
        district_polls = fetch_house_district_polls()
        district_summary = house_district_poll_summary(district_polls, as_of)
        print(f"{'─'*60}")
        print(f" DISTRICT-LEVEL HOUSE POLLS — DIAGNOSTIC ONLY, NOT A MODEL INPUT")
        print(f" {len(district_polls)} polls across {district_polls['district_id'].nunique() if len(district_polls) else 0} districts as of {as_of.isoformat()}")
        print(f"{'─'*60}\n")

    if args.dry_run:
        logger.info("Dry run — no files written.")
        return

    polls_path = LIVE_DIR / "generic_ballot_polls.csv"
    polls.to_csv(polls_path, index=False)
    logger.info(f"Raw poll list → {polls_path}")

    summary_path = LIVE_DIR / "generic_ballot_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Trailing-average summary → {summary_path}")

    if not args.skip_district_polls:
        district_polls_path = LIVE_DIR / "house_district_polls.csv"
        district_polls.to_csv(district_polls_path, index=False)
        logger.info(f"Raw district-level poll list → {district_polls_path}")

        district_summary_path = LIVE_DIR / "house_district_polls_summary.json"
        with open(district_summary_path, "w") as f:
            json.dump({"as_of": as_of.isoformat(), "districts": district_summary}, f, indent=2)
        logger.info(f"Per-district summary → {district_summary_path}")

    log_path = LIVE_DIR / "polling_fetch_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "subject": args.subject,
            "window_days": args.window_days,
            "n_polls_fetched": int(len(polls)),
            "n_district_polls_fetched": int(len(district_polls)),
            "n_districts_with_coverage": int(district_polls["district_id"].nunique()) if len(district_polls) else 0,
            **summary,
        }) + "\n")


if __name__ == "__main__":
    main()
