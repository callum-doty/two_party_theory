#!/usr/bin/env python3
"""
Strategic-window sweep (research-discussion follow-up, 2026-08-13): for the
same 12 top candidates already identified in compute_strategic_leverage.py,
hold tau=0 and delta=$1M fixed and vary the REFERENCE DATE t -- how many
days before Election Day the move is made -- using each race's TIER-POOLED
real committed capital (estimation.commitment_timing's competitive/lean/
safe_likely curves, not response_delay.py's single blended party curve).

This is the collapsed (1D) version of the full V_i(t, tau, delta) surface
discussed: fixing tau=0 answers "how much does the opponent's OWN
already-committed capital, as of the same date I move, constrain its
response" without the combinatorial cost of also varying tau on top --
scoped down explicitly after estimating the full grid at many hours of
exact-SLSQP solves. response_delay.py's separate tau-sweep (fixed t0=
Sept 1) remains the tool for "does additional delay beyond a fixed
commitment date matter."

Reference dates: 120/90/60/45/30/21/14/7 days before Election Day (2022:
Nov 8; 2024: Nov 5 -- true dates, not the project's Nov-8-for-both
approximation elsewhere, since exact commitment fractions are what's being
measured here).

Usage:
    python scripts/compute_strategic_window.py --cycle 2024
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_cycle_state import build_cycle_state  # noqa: E402
from estimation.commitment_timing import build_tiered_curves  # noqa: E402
from game import payoff  # noqa: E402
from game import strategic_window as sw  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("compute_strategic_window")

DELTA = 1_000_000.0
DAYS_BEFORE_COARSE = [120, 90, 60, 45, 30, 21, 14, 7]
# Weekly grid (2026-08-14, K=15-20 stress test's second check): 7-day steps from
# 120 days out down to the same 7-day mechanical-floor endpoint the coarse grid
# uses, so theta_final_week_sensitivity.py's "drop the last date" logic still
# isolates the same mechanical convergence effect at finer resolution.
DAYS_BEFORE_WEEKLY = list(range(120, 7, -7)) + [7]  # 120,113,...,8,7 (18 dates)
ELECTION_DAY = {2022: date(2022, 11, 8), 2024: date(2024, 11, 5)}
REDISTRICTING_FLAGGED = {
    "AL-02", "LA-06", "NC-01", "NC-06", "NC-07", "NC-10", "NC-13", "NC-14",
    "NY-03", "NY-04", "NY-17", "NY-18", "NY-22",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategic-window sweep (exact SLSQP, tiered locked capital)")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    parser.add_argument("--pool", choices=["curve", "primary", "union"], default="curve",
                         help="'curve' (default): the top-3-by-leverage candidates strategic_leverage.py ran the "
                              "full 4-delta curve for -- backward compatible, writes strategic_window_{cycle}.json. "
                              "'primary': the FULL top-4-swing+top-4-|Z| candidate pool (7-8 districts/side) "
                              "strategic_leverage.py solved once at $1M -- writes "
                              "strategic_window_expanded_{cycle}.json. "
                              "'union': the K~15-20 pre-screen from rank_candidate_races.py (top |Z| + top V_uni + "
                              "equilibrium swing + revealed-contested) -- writes strategic_window_union_{cycle}.json. "
                              "Run rank_candidate_races.py --cycle {cycle} first.")
    parser.add_argument("--exclude-redistricting", action="store_true",
                         help="Drop RaceRecord.redistricting_flagged districts (NC-06/13/14/etc.) from the pool "
                              "BEFORE sweeping, so no solves are spent on candidates already known to have an "
                              "unreliable baseline. Appends '_excl_redistricting' to the output filename.")
    parser.add_argument("--weekly", action="store_true",
                         help="Use the 18-point weekly grid (120,113,...,8,7 days out) instead of the coarse "
                              "8-point grid. Appends '_weekly' to the output filename.")
    args = parser.parse_args()
    days_before = DAYS_BEFORE_WEEKLY if args.weekly else DAYS_BEFORE_COARSE

    if args.pool == "union":
        union_path = REPO_ROOT / "results" / f"candidate_union_{args.cycle}.json"
        if not union_path.exists():
            raise SystemExit(f"{union_path} not found -- run rank_candidate_races.py --cycle {args.cycle} first")
        union = json.load(open(union_path))
        top_d = union["D"]["union"]
        top_r = union["R"]["union"]
        out_name = f"strategic_window_union_{args.cycle}.json"
    else:
        prior_path = REPO_ROOT / "results" / f"strategic_leverage_{args.cycle}.json"
        if not prior_path.exists():
            raise SystemExit(f"{prior_path} not found -- run compute_strategic_leverage.py --cycle {args.cycle} first")
        prior = json.load(open(prior_path))
        if args.pool == "curve":
            top_d = sorted({r["district_id"] for r in prior["leverage_D_curve"]})
            top_r = sorted({r["district_id"] for r in prior["leverage_R_curve"]})
            out_name = f"strategic_window_{args.cycle}.json"
        else:
            top_d = sorted({r["district_id"] for r in prior["leverage_D_primary"]})
            top_r = sorted({r["district_id"] for r in prior["leverage_R_primary"]})
            out_name = f"strategic_window_expanded_{args.cycle}.json"

    if args.exclude_redistricting:
        dropped_d = sorted(set(top_d) & REDISTRICTING_FLAGGED)
        dropped_r = sorted(set(top_r) & REDISTRICTING_FLAGGED)
        top_d = [d for d in top_d if d not in REDISTRICTING_FLAGGED]
        top_r = [d for d in top_r if d not in REDISTRICTING_FLAGGED]
        out_name = out_name.replace(".json", "_excl_redistricting.json")
        logger.info(f"--exclude-redistricting: dropped D={dropped_d}, R={dropped_r} before sweeping")
    if args.weekly:
        out_name = out_name.replace(".json", "_weekly.json")
    logger.info(f"Cycle {args.cycle} ({args.pool} pool): D candidates {top_d} ({len(top_d)}); "
                f"R candidates {top_r} ({len(top_r)})")

    election_day = ELECTION_DAY[args.cycle]
    dates = [election_day - timedelta(days=d) for d in days_before]
    logger.info(f"Election day {election_day}; {'WEEKLY' if args.weekly else 'coarse'} reference dates "
                f"(days-before -> date): {list(zip(days_before, dates))}")

    state = build_cycle_state(args.cycle, args.cap_fraction_d, args.cap_fraction_r)
    races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]
    cand_r_total, budget_d, budget_r = state["cand_r_total"], state["budget_d"], state["budget_r"]
    n = state["n_races"]
    d0 = np.array([r.d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    floors_d = np.array([r.cand_d_total for r in races])
    party_d_obs = np.maximum(d0 - floors_d, 0.0)
    party_r_obs = np.maximum(r0 - cand_r_total, 0.0)
    idx_by_district = {r.district_id: i for i, r in enumerate(races)}
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)

    logger.info("Building tier-pooled commitment curves (D and R)…")
    curves_d = build_tiered_curves(args.cycle, "D", races)
    curves_r = build_tiered_curves(args.cycle, "R", races)
    for party, curves in (("D", curves_d), ("R", curves_r)):
        for tier, c in curves.items():
            logger.info(f"  {party}/{tier}: {len(c)} distinct dates, "
                        f"{c.iloc[0]['exp_date'].date()} -> {c.iloc[-1]['exp_date'].date()}")

    d_rows: list[dict] = []
    baseline_cache_r: dict = {}
    for did in top_d:
        i = idx_by_district[did]
        logger.info(f"D-side {did}: sweeping reference date…")
        rows = sw.retention_by_date_d(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            race_idx=i, delta=DELTA, cycle=args.cycle, dates=dates,
            party_d_obs=party_d_obs, party_r_obs=party_r_obs, arrays=arrays,
            curves_r=curves_r, cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
            baseline_cache=baseline_cache_r,
        )
        for row, dbefore in zip(rows, days_before):
            logger.info(f"    t={dbefore:>3}d before ({row['ref_date']}): "
                        f"PSV={row['PSV']:+.4f} retention={row['retention_rate']:.1%} "
                        f"R_flexible=${row['flexible_budget_r']:,.0f}")
        d_rows.extend(rows)

    r_rows: list[dict] = []
    baseline_cache_d: dict = {}
    for did in top_r:
        i = idx_by_district[did]
        logger.info(f"R-side {did}: sweeping reference date…")
        rows = sw.retention_by_date_r(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            race_idx=i, delta=DELTA, cycle=args.cycle, dates=dates,
            party_d_obs=party_d_obs, party_r_obs=party_r_obs, arrays=arrays, n_races=n,
            curves_d=curves_d, cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
            baseline_cache=baseline_cache_d,
        )
        for row, dbefore in zip(rows, days_before):
            logger.info(f"    t={dbefore:>3}d before ({row['ref_date']}): "
                        f"PSV={row['PSV']:+.4f} retention={row['retention_rate']:.1%} "
                        f"D_flexible=${row['flexible_budget_d']:,.0f}")
        r_rows.extend(rows)

    # strategic_opening_date expects chronological (earliest-date-first) order;
    # `dates`/DAYS_BEFORE above go from farthest-out to closest-in, i.e. already
    # earliest-date-first -- no re-sort needed.
    opening_d = {did: sw.strategic_opening_date([r for r in d_rows if r["district_id"] == did], threshold=0.80)
                 for did in top_d}
    opening_r = {did: sw.strategic_opening_date([r for r in r_rows if r["district_id"] == did], threshold=0.80)
                 for did in top_r}
    logger.info(f"T_i^80 (D-side, days-before-election basis): {opening_d}")
    logger.info(f"T_i^80 (R-side, days-before-election basis): {opening_r}")

    out = dict(
        cycle=args.cycle, delta=DELTA, days_before=days_before, election_day=str(election_day),
        strategic_window_D=d_rows, strategic_window_R=r_rows,
        T80_D=opening_d, T80_R=opening_r,
    )
    out_path = REPO_ROOT / "results" / out_name
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
