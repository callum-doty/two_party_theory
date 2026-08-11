#!/usr/bin/env python3
"""
Live 2026 FEC IE ingestion — real-time MSG dashboard.

Polls the FEC /schedules/schedule_e/ endpoint for independent expenditures
filed in the last N hours (24h and 48h accelerated-reporting windows), updates
district-level D and R totals, and recalculates the MSG vector for all
competitive House races.

FEC mandatory accelerated reporting rules (11 C.F.R. § 104.4):
  • 48-hour notice: ≥ $10,000 IE disclosed in any single calendar day, filed
    within 48 hours of the date the expenditure is made.
  • 24-hour notice: within 20 days of a general election, ≥ $1,000 IE, filed
    within 24 hours.

Running this script daily during the late-cycle window captures all significant
capital deployments as they happen, before opponents can respond.

Usage:
    python scripts/fetch_live_ies.py --api-key YOUR_KEY
    python scripts/fetch_live_ies.py --api-key YOUR_KEY --lookback-hours 48
    python scripts/fetch_live_ies.py --api-key YOUR_KEY --dry-run   # skip write

Outputs:
    data/live/spending_live.json      — per-district {d_total, r_total}
    data/live/msg_live.csv            — competitive races with real-time MSG
    data/live/fetch_log.jsonl         — append-only fetch audit trail
    data/live/processed_ie_ids.json   — sub_ids already folded into spending_live.json;
                                         required because lookback windows overlap by
                                         design (e.g. daily runs at --lookback-hours 48),
                                         so the same filing is fetched more than once —
                                         this ledger is what keeps apply_new_ies() from
                                         double-counting it.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import requests

from backtest import config
from backtest.data.universe import build_universe
from backtest.model.margin import MarginModelCoefficients
from backtest.types import SigmaModel, RaceRecord
from backtest.optimizer.allocator import _precompute_race_arrays, _msg_vec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("fetch_live_ies")

FEC_API_BASE = "https://api.open.fec.gov/v1"

# Major IE committees to track (DCCC, NRCC, HMP, CLF, DCCC super PAC affiliates)
# Add CLF and HMP committee IDs — these are the dominant outside groups
#
# Corrected 2026-07-23 (codebase audit, verified against fec.gov/data/committee/<id>/):
#   old "C00075473" (NRCC)  -> actually CMS Energy Corp Employees for Better Government
#   old "C00500884" (HMP)   -> did not resolve to House Majority PAC
#   old "C00571372" (CLF)   -> actually Right to Rise USA (terminated Jeb Bush 2016 super PAC)
# Only "C00000935" (DCCC) was correct. All three R-side/other IDs below are
# fixed; any data/live/*.json or data/raw/fec IE files fetched under the old
# IDs must be re-fetched.
COMMITTEE_WATCH = {
    "C00000935": "D",   # DCCC
    "C00075820": "R",   # NRCC
    "C00495028": "D",   # House Majority PAC (HMP)
    "C00504530": "R",   # Congressional Leadership Fund (CLF)
}

LIVE_DIR = Path(__file__).parent.parent / "data" / "live"


# ─── FEC API ─────────────────────────────────────────────────────────────────

def _fec_get(session: requests.Session, endpoint: str, params: dict) -> list[dict]:
    """Paginate a FEC API endpoint and return all records."""
    records: list[dict] = []
    url = FEC_API_BASE + endpoint
    params = dict(params)
    params["per_page"] = 100
    page = 1

    while True:
        params["page"] = page
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            logger.warning("FEC rate limit — waiting 60s")
            import time; time.sleep(60)
            continue
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        records.extend(results)
        pages = data.get("pagination", {}).get("pages", 1)
        logger.debug(f"  page {page}/{pages}, {len(results)} records")
        if page >= pages:
            break
        page += 1

    return records


def fetch_recent_ies(
    api_key: str,
    lookback_hours: int = 24,
    cycle: int = 2026,
) -> pd.DataFrame:
    """
    Fetch IEs filed since (now - lookback_hours) for all watched committees.

    Returns DataFrame: district_id, party, amount (net of oppose/support)
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    logger.info(f"Fetching IEs filed since {cutoff_str} (last {lookback_hours}h)…")

    rows: list[dict] = []
    with requests.Session() as session:
        for committee_id, party in COMMITTEE_WATCH.items():
            records = _fec_get(session, "/schedules/schedule_e/", {
                "api_key":           api_key,
                "committee_id":      committee_id,
                "cycle":             cycle,
                "min_date":          cutoff_str,
                "candidate_office":  "H",
                "sort":              "-expenditure_date",
            })
            logger.info(f"  {committee_id} ({party}): {len(records)} new IEs")
            for r in records:
                state   = (r.get("candidate_office_state") or "").upper()
                dist    = str(r.get("candidate_office_district") or "00").zfill(2)
                if not state:
                    continue
                amount  = float(r.get("expenditure_amount") or 0)
                # sub_id is the FEC's own unique identifier for this individual
                # Schedule E line item (distinct from file_number, which
                # identifies the whole filing and can cover many line items).
                # Required for dedup: fetch windows overlap by design (24h/48h
                # lookback re-run on a cadence shorter than the lookback), so
                # the same expenditure is returned by more than one fetch.
                sub_id = r.get("sub_id")
                # Support = positive, Oppose = negative net effect on opposing party
                # We track D/R total spend (gross IE amounts)
                rows.append({
                    "district_id":     f"{state}-{dist}",
                    "party":           party,
                    "amount":          amount,
                    "sub_id":          str(sub_id) if sub_id is not None else "",
                    "filed_at":        r.get("file_number") or "",
                    "expenditure_date": r.get("expenditure_date") or "",
                    "support_oppose":   r.get("support_oppose_indicator") or "S",
                })

    if not rows:
        logger.warning("No new IE filings found in the lookback window.")
        return pd.DataFrame(columns=["district_id", "party", "amount", "sub_id"])

    return pd.DataFrame(rows)


# ─── Spending aggregation ─────────────────────────────────────────────────────

def load_baseline_spending(cycle: int) -> dict[str, dict]:
    """
    Load the most recent complete spending snapshot.

    Priority: data/live/spending_live.json > historical backtest CSVs.
    """
    live_path = LIVE_DIR / "spending_live.json"
    if live_path.exists():
        with open(live_path) as f:
            return json.load(f)

    logger.info("No live spending snapshot found — loading from historical backtest data.")
    from backtest.data import fec
    spend_df = fec.build_total_spend(cycle)
    return {
        row["district_id"]: {"d_total": row["d_total"], "r_total": row["r_total"]}
        for _, row in spend_df.iterrows()
    }


def load_processed_ie_ids() -> set[str]:
    """sub_ids already folded into spending_live.json by a prior run.

    Fetch windows (24h/48h lookback) overlap by design, so the same FEC
    Schedule E line item is returned by more than one call to
    fetch_recent_ies() — without this ledger, apply_new_ies() would add its
    amount into the cumulative total again on every overlapping run.
    """
    path = LIVE_DIR / "processed_ie_ids.json"
    if not path.exists():
        return set()
    with open(path) as f:
        return set(json.load(f))


def save_processed_ie_ids(ids: set[str]) -> None:
    path = LIVE_DIR / "processed_ie_ids.json"
    with open(path, "w") as f:
        json.dump(sorted(ids), f, indent=2)


def apply_new_ies(
    baseline: dict[str, dict],
    new_ies: pd.DataFrame,
    processed_ids: set[str],
) -> tuple[dict[str, dict], set[str]]:
    """
    Merge new IE amounts into the baseline spending snapshot.

    For the log-ratio model, what matters is the total gross spend per party
    per district.  We add new IE amounts to the existing totals — but only
    for sub_ids not already in `processed_ids`, so a transaction already
    folded into `baseline` by an earlier, overlapping fetch is never added
    twice.  Rows with a missing/blank sub_id can't be deduped and are always
    applied, with a warning, since skipping them silently would just as
    silently undercount instead.

    Returns (updated_spending, newly_applied_ids) — the caller persists
    newly_applied_ids into the processed-ids ledger.
    """
    updated = {k: dict(v) for k, v in baseline.items()}
    newly_applied: set[str] = set()

    n_skipped_dupe = 0
    n_missing_id = 0
    for _, row in new_ies.iterrows():
        sub_id = row.get("sub_id", "")
        if sub_id:
            if sub_id in processed_ids:
                n_skipped_dupe += 1
                continue
            newly_applied.add(sub_id)
        else:
            n_missing_id += 1

        did = row["district_id"]
        if did not in updated:
            updated[did] = {"d_total": 0.0, "r_total": 0.0}
        key = "d_total" if row["party"] == "D" else "r_total"
        updated[did][key] = updated[did].get(key, 0.0) + row["amount"]

    if n_skipped_dupe:
        logger.info(f"Skipped {n_skipped_dupe} IE(s) already applied in a prior fetch (overlapping window).")
    if n_missing_id:
        logger.warning(f"{n_missing_id} IE(s) had no sub_id — applied without dedup protection.")

    return updated, newly_applied


# ─── MSG computation ──────────────────────────────────────────────────────────

def compute_live_msg(
    spending: dict[str, dict],
    cycle: int = 2026,
) -> pd.DataFrame:
    """
    Load estimation artifacts and compute the real-time MSG vector.

    Returns a DataFrame of competitive races sorted by MSG descending.
    """
    processed = config.processed_path()
    with open(processed / "margin_model_coef.json") as f:
        d = json.load(f)
    coef = MarginModelCoefficients(**{k: d[k] for k in
                                      ["alpha0", "alpha1", "alpha2", "alpha3", "alpha4",
                                       "beta1", "beta2", "beta3"]},
                                   alpha5=d.get("alpha5", 0.0))
    with open(processed / "sigma_model.json") as f:
        sigma_coef = json.load(f)
    sigma_model = SigmaModel(_coef=sigma_coef)

    races = build_universe(cycle=cycle)

    # Override d_total / r_total from live spending snapshot
    updated_races = []
    for r in races:
        s = spending.get(r.district_id, {})
        d_live = s.get("d_total", r.d_total)
        r_live = s.get("r_total", r.r_total)
        updated_races.append(RaceRecord(
            district_id=r.district_id,
            state=r.state,
            district=r.district,
            cook_rating=r.cook_rating,
            incumb_status=r.incumb_status,
            pvi=r.pvi,
            d_total=d_live,
            r_total=r_live,
            cvap=r.cvap,
            generic_ballot=r.generic_ballot,
            redistricting_flagged=r.redistricting_flagged,
            outcome=r.outcome,
            cand_d_total=r.cand_d_total,
        ))

    arrays = _precompute_race_arrays(updated_races, coef, sigma_model, eta=0.0)
    party_obs = np.array([r.d_total - r.cand_d_total for r in updated_races], dtype=float)
    party_obs = np.maximum(party_obs, 0.0)
    msg = _msg_vec(party_obs, arrays)

    competitive_ratings = set(config.competitive_ratings())

    rows = []
    for i, (race, m) in enumerate(zip(updated_races, msg)):
        if race.cook_rating not in competitive_ratings:
            continue
        d = updated_races[i].d_total
        r_total = updated_races[i].r_total
        log_ratio = float(np.log(max(d, 1) / max(d + r_total, 2)))
        rows.append({
            "district_id":   race.district_id,
            "cook_rating":   race.cook_rating,
            "pvi":           race.pvi,
            "incumb_status": race.incumb_status,
            "d_total_m":     d / 1e6,
            "r_total_m":     r_total / 1e6,
            "log_ratio":     round(log_ratio, 4),
            "msg":           float(m),
            "msg_rank":      0,  # filled below
        })

    df = pd.DataFrame(rows).sort_values("msg", ascending=False).reset_index(drop=True)
    df["msg_rank"] = df.index + 1
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live FEC IEs and update MSG dashboard")
    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("FEC_API_KEY", "DEMO_KEY"),
                        help="FEC API key (env: FEC_API_KEY). "
                             "DEMO_KEY rate-limited to ~1,000 req/hr.")
    parser.add_argument("--lookback-hours", type=int, default=24,
                        help="Fetch IEs filed in the last N hours (default: 24)")
    parser.add_argument("--cycle", type=int, default=2026,
                        help="Election cycle (default: 2026)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print MSG but don't write output files")
    args = parser.parse_args()

    if args.api_key == "DEMO_KEY":
        logger.warning("Using DEMO_KEY — rate limited to 1,000 req/hr. "
                       "Register a free key at https://api.open.fec.gov/developers")

    LIVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch new IEs
    new_ies = fetch_recent_ies(args.api_key, args.lookback_hours, args.cycle)

    # 2. Load baseline + processed-ids ledger, apply updates (deduped)
    baseline = load_baseline_spending(args.cycle)
    processed_ids = load_processed_ie_ids()
    updated, newly_applied_ids = apply_new_ies(baseline, new_ies, processed_ids)

    # 3. Compute live MSG
    msg_df = compute_live_msg(updated, cycle=args.cycle)

    # 4. Display dashboard
    print(f"\n{'─'*70}")
    print(f" REAL-TIME MSG DASHBOARD  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
          f"{len(new_ies)} new IEs in last {args.lookback_hours}h")
    print(f"{'─'*70}")
    print(msg_df[["msg_rank", "district_id", "cook_rating", "d_total_m",
                  "r_total_m", "log_ratio", "msg"]].head(20).to_string(index=False))
    print(f"{'─'*70}\n")

    if args.dry_run:
        logger.info("Dry run — no files written.")
        return

    # 5. Persist
    spending_path = LIVE_DIR / "spending_live.json"
    with open(spending_path, "w") as f:
        json.dump(updated, f, indent=2)
    logger.info(f"Spending snapshot → {spending_path}")

    msg_path = LIVE_DIR / "msg_live.csv"
    msg_df.to_csv(msg_path, index=False)
    logger.info(f"MSG dashboard → {msg_path}")

    save_processed_ie_ids(processed_ids | newly_applied_ids)
    logger.info(f"Processed-IE ledger → {len(processed_ids | newly_applied_ids)} sub_ids "
                f"({len(newly_applied_ids)} new this run)")

    # Audit trail
    log_path = LIVE_DIR / "fetch_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle": args.cycle,
            "lookback_hours": args.lookback_hours,
            "new_ie_count": len(new_ies),
            "districts_updated": len(set(new_ies["district_id"])) if len(new_ies) else 0,
            "top_msg_district": msg_df["district_id"].iloc[0] if len(msg_df) else None,
        }) + "\n")


if __name__ == "__main__":
    main()
