#!/usr/bin/env python3
"""
Estimate a candidate-committee spending "trickle" rate (Paper III,
docs/theta_followup_plan.md Section 0.1.1's blocked fix).

Problem this solves: scripts/solve_bellman_lsm.py's "wait" branch holds
D_i,t (the Democratic candidate-committee floor) perfectly fixed while
waiting, because -- until data_catalog.md Section 2.7's dated periodic-
reports panel -- there was no per-filing-date source for candidate
disbursements anywhere in this repository. With D_i,t fixed, eta(tier) has
nothing to react to on the wait branch (eta * 0 = 0 regardless of eta's
value), which is the single biggest reason Theta showed almost no patience
anywhere in Paper III.

This script does NOT re-estimate eta itself (scripts/estimate_eta_reaction.py
already does that, from IE-to-IE reaction). It estimates a separate, much
simpler quantity: how fast does a Democratic candidate's own committee
disbursements grow, per day, absent any DCCC discretionary deployment
decision -- i.e. the campaign's own baseline "keep the lights on" spending
that happens regardless of what the party committee does. This is what
scripts/solve_bellman_lsm.py's wait branch grows D_i,t at; R_i,t then
reacts to that growth via the already-estimated eta(tier), exactly as
theta_followup_plan.md Section 0.1.1 proposed.

Design, mirroring estimate_eta_reaction.py's period-panel construction and
its "check both pooled and tiered, let the data decide" discipline: for
each historical cycle with a dated candidate-periodic-reports panel
available, build a biweekly cumulative-D-candidate-spend panel per
district (reusing dynamic/periods.py::biweekly_periods, the same grid
solve_bellman_lsm.py runs on), take period-over-period deltas, convert to
a $/day rate, and summarize per Cook tier.

Both median and mean are computed and reported, per this project's
established discipline of stating both rather than picking one a priori
(estimate_eta_reaction.py, paper3_draft.md Section 4.4). The a priori
expectation going in was that median would be preferred (candidate
disbursements are lumpy/bunched around FEC filing deadlines, so a few
large reports could dominate a mean) -- but checking the real 2022/2024
data directly overturned that expectation: **median is exactly $0.00/day
in every tier**, not a robust central-tendency estimate but a structural
artifact of comparing a quarterly filing cadence against a biweekly period
grid -- the large majority of 14-day windows contain no new periodic
report at all, so cumulative spend is flat (delta=0) between filing dates
and only jumps on the dates something was actually filed. A median across
mostly-zero deltas is zero regardless of how much real growth the
underlying committees experienced. **Mean is therefore the estimator this
script actually uses** (`preferred_estimator` below) -- it is pulled by
the genuine large jumps at filing dates in exactly the way needed to
represent an average daily growth rate over a period grid finer than the
underlying data's actual reporting frequency, which is the quantity
solve_bellman_lsm.py's wait branch needs (a smoothed drift to apply every
biweekly period), not a literal reproduction of the lumpy realization.

Output: data/processed/candidate_spend_trickle.json
"""

from __future__ import annotations
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import config
from backtest.data.universe import build_universe
from backtest.data import fec
from backtest.dynamic.periods import biweekly_periods

ROOT = Path(__file__).parent.parent
CANDIDATE_CYCLES = [2012, 2014, 2016, 2018, 2020, 2022, 2024]
TIERS = ["Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"]


def _has_dated_panel(cycle: int) -> bool:
    return (config.raw_path("fec") / f"candidate_periodic_reports_{cycle}.csv").exists()


def build_d_spend_period_panel(cycle: int) -> pd.DataFrame:
    """Return a long panel: district_id, tier, period_index, period_date,
    d_cand_cum -- cumulative Democratic candidate-committee disbursements
    as of each biweekly period, restricted to districts in the modeled
    competitive universe (same restriction estimate_eta_reaction.py's
    build_period_panel applies)."""
    races = build_universe(cycle=cycle)
    tiers = {r.district_id: r.cook_rating for r in races}

    periods = biweekly_periods(date(cycle, 1, 1), date(cycle, 11, 8))
    rows = []
    for p in periods:
        cum = fec.cumulative_candidate_spend_as_of(cycle, p.period_date)
        d_cum = cum[cum["party"] == "D"][["district_id", "disb_cum"]].rename(
            columns={"disb_cum": "d_cand_cum"}
        )
        d_cum["period_index"] = p.index
        d_cum["period_date"] = p.period_date
        rows.append(d_cum)

    panel = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["district_id", "d_cand_cum", "period_index", "period_date"]
    )
    panel["tier"] = panel["district_id"].map(tiers)
    panel = panel[panel["tier"].notna()].copy()
    panel["cycle"] = cycle
    return panel


def build_delta_rate_panel(panel: pd.DataFrame, period_days: int = 14) -> pd.DataFrame:
    """Period-over-period deltas in cumulative D candidate spend, converted
    to a $/day rate. No within-race demeaning here (unlike eta's
    regression) -- this is a direct rate estimate, not a reaction
    coefficient, so raw per-district-period growth rates are the unit of
    observation that gets aggregated to a tier median/mean below."""
    panel = panel.sort_values(["cycle", "district_id", "period_index"])
    g = panel.groupby(["cycle", "district_id"])
    panel["d_cand_delta"] = g["d_cand_cum"].diff()
    panel = panel.dropna(subset=["d_cand_delta"]).copy()
    panel["d_cand_rate_per_day"] = panel["d_cand_delta"] / period_days
    return panel


def summarize_by_tier(delta_panel: pd.DataFrame) -> dict:
    summary = {}
    for tier in TIERS:
        sub = delta_panel[delta_panel["tier"] == tier]["d_cand_rate_per_day"]
        if len(sub) == 0:
            continue
        summary[tier] = {
            "n_obs": int(len(sub)),
            "median_rate_per_day": float(sub.median()),
            "mean_rate_per_day": float(sub.mean()),
            "std_rate_per_day": float(sub.std()) if len(sub) > 1 else 0.0,
        }
    pooled = delta_panel["d_cand_rate_per_day"]
    summary["_pooled"] = {
        "n_obs": int(len(pooled)),
        "median_rate_per_day": float(pooled.median()) if len(pooled) else 0.0,
        "mean_rate_per_day": float(pooled.mean()) if len(pooled) else 0.0,
        "std_rate_per_day": float(pooled.std()) if len(pooled) > 1 else 0.0,
    }
    return summary


def main() -> None:
    available_cycles = [c for c in CANDIDATE_CYCLES if _has_dated_panel(c)]
    if not available_cycles:
        raise SystemExit(
            "No candidate_periodic_reports_{cycle}.csv found for any of "
            f"{CANDIDATE_CYCLES}. Run `python scripts/fetch_data.py --only "
            "fec-periodic --cycles <cycle> --fec-api-key YOUR_KEY` first "
            "(data_catalog.md Section 2.7) -- this script has nothing to "
            "fit without the dated panel."
        )
    skipped = sorted(set(CANDIDATE_CYCLES) - set(available_cycles))
    if skipped:
        print(f"Skipping cycles with no dated panel fetched yet: {skipped}")
    print(f"Fitting trickle rate from: {available_cycles}")

    panels = [build_d_spend_period_panel(c) for c in available_cycles]
    full = pd.concat(panels, ignore_index=True)
    print(f"  cumulative-panel rows: {len(full)}")

    delta = build_delta_rate_panel(full)
    print(f"  delta-panel rows (post-diff): {len(delta)}")

    summary = summarize_by_tier(delta)
    print("\n=== D candidate-committee spend trickle rate ($/day) ===")
    for tier, s in summary.items():
        print(f"  {tier:12s} n={s['n_obs']:5d}  median={s['median_rate_per_day']:10.2f}  "
              f"mean={s['mean_rate_per_day']:10.2f}  std={s['std_rate_per_day']:10.2f}")

    out = {
        "cycles_used": available_cycles,
        "cycles_skipped_no_data": skipped,
        "period_days": 14,
        "by_tier": summary,
        "preferred_estimator": "mean_rate_per_day",
        "note": (
            "mean preferred over median: checked directly against real 2022/2024 data, "
            "median_rate_per_day is exactly $0.00/day in every tier -- a structural "
            "artifact of comparing FEC's quarterly filing cadence against this "
            "project's biweekly period grid (most 14-day windows contain no new "
            "filing at all), not a robust central-tendency estimate. Mean captures the "
            "real filing-date jumps and is the smoothed drift rate solve_bellman_lsm.py's "
            "wait branch actually needs; see this script's module docstring."
        ),
    }
    out_path = config.processed_path() / "candidate_spend_trickle.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
