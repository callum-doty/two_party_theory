#!/usr/bin/env python3
"""
Strategic leverage / response-displacement sweep (research-direction
discussion, 2026-08-13): "where can one dollar of pressure force the
opponent to incur the most costly possible response," rather than searching
for another static allocation the opponent's best-response solver would
just counter (project_spec.md's own equilibrium work already shows a
smarter static allocation isn't findable against a symmetric, unrestricted
opponent).

Scoped down from the literal "Republican opportunity cost" framing the
discussion proposed, after finding it is algebraically identical to
persistent_value.py's PSV in this constant-sum game -- see
src/game/strategic_leverage.py's module docstring for the derivation. What
this script actually adds on top of compute_persistent_value.py:

  1. The RESPONSE-DISPLACEMENT MAP: which specific races the opponent's
     best response pulls money out of (or into) -- not previously computed
     or exposed anywhere in this codebase, even though best_response
     already produces the full allocation vector needed for it.
  2. reshuffle_per_million: total opponent dollars disrupted per $1M
     committed -- distinct from PSV/leverage, since reshuffling can be
     large even when it nets to a small aggregate seat swing.
  3. A LEVERAGE CURVE across multiple delta values, for the top few
     candidates, to see whether PSV-per-dollar is roughly constant or
     declining as commitment size grows.

EXACT SLSQP ONLY, not the surrogate -- found out the hard way. An earlier
version of this script used best_response_surrogate.py (validated within
0.03-0.10 expected seats of exact SLSQP on AGGREGATE objective value) for
speed. First real run: WI-01 (2024, D-side, $1M) showed a surrogate-
reported reshuffle_l1 of $4.1M, but the exact re-solve found reshuffle_l1
of $16 -- R's true best response barely moves at all (retention 98.7%);
the "$4.1M reshuffle" was pure surrogate noise. This matches docs/
methodology.md's own documented caveat on the surrogate ("allocation-level
L1 differences are larger ($4-14M) than the objective-value agreement
would suggest") -- but that caveat was previously only about how much
worse the surrogate's ALLOCATION is as a descriptive object, never about
whether it could be trusted for a DIFFERENCE of two allocations (which is
exactly what a displacement map is). It can't: the surrogate's aggregate-
objective accuracy does not imply its allocation vector is accurate enough
to difference against another allocation vector. So this script pays the
exact-SLSQP cost throughout, and trims candidate/delta counts to keep total
runtime reasonable instead.

Candidate races are NOT a new arbitrary selection: the union of (a) each
side's own equilibrium swing-race list (equilibrium_support_composition_
{cycle}.json, already computed) and (b) the top-|Z| currently-funded races
compute_persistent_value.py already uses.

Two-phase design to keep exact-SLSQP cost bounded:
  Phase 1: every candidate, ONE reference delta ($1M) -- gives a trustworthy
    ranking and displacement map for the full candidate set.
  Phase 2: the top N_CURVE candidates per side (by Phase 1 leverage), the
    remaining delta values -- gives the leverage-vs-delta shape only for
    the races that matter most, not the whole candidate set.

Usage:
    python scripts/compute_strategic_leverage.py --cycle 2024
    python scripts/compute_strategic_leverage.py --cycle 2022 --n-swing 4 --n-z 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_cycle_state import build_cycle_state  # noqa: E402
from game import best_response as br  # noqa: E402
from game import exploitability, payoff  # noqa: E402
from game import strategic_leverage as lev  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("compute_strategic_leverage")

PRIMARY_DELTA = 1_000_000.0
CURVE_DELTAS = [250_000.0, 500_000.0, 2_000_000.0]  # additional points, top candidates only
N_CURVE = 3  # how many top-by-leverage candidates per side get the full curve


def _candidate_indices(races, cycle: int, side: str, surplus: dict,
                        n_swing: int, n_z: int, min_party_spend: float = 10_000.0) -> list[int]:
    """Union of the equilibrium's own swing-race list (already computed by
    equilibrium_support_composition.py) and the top-|Z| currently-funded
    races (the same funded-only filter compute_persistent_value.py uses,
    for the low-spend-MSG-artifact reason documented there)."""
    district_ids = [r.district_id for r in races]
    idx_by_district = {d: i for i, d in enumerate(district_ids)}

    swing_path = REPO_ROOT / "results" / f"equilibrium_support_composition_{cycle}.json"
    swing_ids: list[str] = []
    if swing_path.exists():
        comp = json.load(open(swing_path))
        key = "d_side" if side == "D" else "r_side"
        swing_ids = [row["district_id"] for row in comp[key]["top_swing_by_cv"][:n_swing]]
    else:
        logger.warning(f"{swing_path} not found -- skipping swing-race candidates for side {side}")

    party_obs = surplus["party_d_obs"] if side == "D" else surplus["party_r_obs"]
    z = surplus["Z_D"] if side == "D" else surplus["Z_R"]
    funded = np.where(party_obs > min_party_spend)[0]
    top_z_idx = funded[np.argsort(-np.abs(z[funded]))[:n_z]]
    top_z_ids = [district_ids[i] for i in top_z_idx]

    combined = list(dict.fromkeys(swing_ids + top_z_ids))  # de-dup, preserve discovery order
    missing = [d for d in combined if d not in idx_by_district]
    if missing:
        logger.warning(f"candidate district_ids not found in universe, skipped: {missing}")
    return [idx_by_district[d] for d in combined if d in idx_by_district]


def _print_ranked(label: str, rows: list[dict]) -> None:
    ranked = sorted(
        rows, key=lambda r: -r["leverage_seats_per_million"] if np.isfinite(r["leverage_seats_per_million"]) else 1e9,
    )
    logger.info(f"--- {label}-side leverage @ ${PRIMARY_DELTA/1e6:.0f}M, ranked ---")
    for r in ranked:
        logger.info(f"  {r['district_id']:8s} V_uni={r['V_uni']:+.4f}  PSV={r['PSV']:+.4f}  "
                    f"retention={r['retention_rate']:.1%}  leverage={r['leverage_seats_per_million']:+.4f} seats/$M  "
                    f"reshuffle=${r['reshuffle_l1']:,.0f} ({r['reshuffle_per_million']:.3f}/$M)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategic leverage / response-displacement sweep (exact SLSQP)")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    parser.add_argument("--n-swing", type=int, default=4, help="Per side, from the equilibrium swing list.")
    parser.add_argument("--n-z", type=int, default=4, help="Per side, top-|Z| funded races.")
    args = parser.parse_args()

    state = build_cycle_state(args.cycle, args.cap_fraction_d, args.cap_fraction_r)
    races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]
    cand_r_total, budget_d, budget_r = state["cand_r_total"], state["budget_d"], state["budget_r"]
    n = state["n_races"]
    d0 = np.array([r.d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    floors_d = np.array([r.cand_d_total for r in races])
    party_d_obs = np.maximum(d0 - floors_d, 0.0)
    party_r_obs = np.maximum(r0 - cand_r_total, 0.0)

    logger.info(f"Cycle {args.cycle}: {n} races, DCCC budget ${budget_d:,.0f}, NRCC budget ${budget_r:,.0f}")

    surplus = exploitability.race_level_surplus(
        races, coef, sigma_model, cand_r_total, budget_d, budget_r,
        args.cap_fraction_d, args.cap_fraction_r,
    )
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)

    logger.info("Computing shared isolated baselines: BR_R(D_obs) and BR_D(R_obs), exact SLSQP…")
    res_r_star = br.br_r(races, coef, sigma_model, party_d=party_d_obs, cand_r_total=cand_r_total,
                          budget_r=budget_r, cap_fraction_r=args.cap_fraction_r)
    party_r_star = res_r_star.party
    baseline_d = float(payoff.p_win_shared(party_d_obs, party_r_star, arrays).sum())

    res_d_star = br.br_d(races, coef, sigma_model, party_r=party_r_obs, cand_r_total=cand_r_total,
                          budget_d=budget_d, cap_fraction_d=args.cap_fraction_d)
    party_d_star = res_d_star.party
    baseline_r = float(n) - float(payoff.p_win_shared(party_d_star, party_r_obs, arrays).sum())
    logger.info(f"baseline_d={baseline_d:.3f}  baseline_r={baseline_r:.3f}  "
                f"(observed U_D={surplus['p_win_obs'].sum():.3f})")

    d_idx = _candidate_indices(races, args.cycle, "D", surplus, args.n_swing, args.n_z)
    r_idx = _candidate_indices(races, args.cycle, "R", surplus, args.n_swing, args.n_z)
    logger.info(f"D-side candidates ({len(d_idx)}): {[races[i].district_id for i in d_idx]}")
    logger.info(f"R-side candidates ({len(r_idx)}): {[races[i].district_id for i in r_idx]}")

    logger.info(f"Phase 1 -- D-side, {len(d_idx)} races @ ${PRIMARY_DELTA/1e6:.0f}M, exact SLSQP…")
    d_primary = [
        lev.leverage_curve_d(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            race_idx=i, deltas=[PRIMARY_DELTA],
            party_d_obs=party_d_obs, party_r_obs=party_r_obs, party_r_star=party_r_star,
            baseline_d=baseline_d, arrays=arrays,
            cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
            use_surrogate=False,
        )[0]
        for i in d_idx
    ]
    _print_ranked("D", d_primary)

    logger.info(f"Phase 1 -- R-side, {len(r_idx)} races @ ${PRIMARY_DELTA/1e6:.0f}M, exact SLSQP…")
    r_primary = [
        lev.leverage_curve_r(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            race_idx=i, deltas=[PRIMARY_DELTA],
            party_d_obs=party_d_obs, party_r_obs=party_r_obs, party_d_star=party_d_star,
            baseline_r=baseline_r, arrays=arrays, n_races=n,
            cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
            use_surrogate=False,
        )[0]
        for i in r_idx
    ]
    _print_ranked("R", r_primary)

    idx_by_district = {r.district_id: i for i, r in enumerate(races)}

    def _phase2(primary_rows: list[dict], idx_map: dict, is_d_side: bool) -> list[dict]:
        finite = [r for r in primary_rows if np.isfinite(r["leverage_seats_per_million"])]
        top = sorted(finite, key=lambda r: -r["leverage_seats_per_million"])[:N_CURVE]
        curve_rows = []
        for r in top:
            i = idx_map[r["district_id"]]
            logger.info(f"Phase 2 -- curve for {r['district_id']} ({'D' if is_d_side else 'R'}-side)…")
            if is_d_side:
                extra = lev.leverage_curve_d(
                    races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                    race_idx=i, deltas=CURVE_DELTAS,
                    party_d_obs=party_d_obs, party_r_obs=party_r_obs, party_r_star=party_r_star,
                    baseline_d=baseline_d, arrays=arrays,
                    cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
                    use_surrogate=False,
                )
            else:
                extra = lev.leverage_curve_r(
                    races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                    race_idx=i, deltas=CURVE_DELTAS,
                    party_d_obs=party_d_obs, party_r_obs=party_r_obs, party_d_star=party_d_star,
                    baseline_r=baseline_r, arrays=arrays, n_races=n,
                    cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
                    use_surrogate=False,
                )
            curve_rows.extend(extra)
            for row in extra:
                logger.info(f"    ${row['delta']/1e3:,.0f}K: PSV={row['PSV']:+.4f} "
                            f"leverage={row['leverage_seats_per_million']:+.4f} "
                            f"reshuffle=${row['reshuffle_l1']:,.0f}")
        return curve_rows

    d_curve = _phase2(d_primary, idx_by_district, True)
    r_curve = _phase2(r_primary, idx_by_district, False)

    out = dict(
        cycle=args.cycle, primary_delta=PRIMARY_DELTA, curve_deltas=CURVE_DELTAS,
        cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
        baseline_d=baseline_d, baseline_r=baseline_r,
        leverage_D_primary=d_primary, leverage_D_curve=d_curve,
        leverage_R_primary=r_primary, leverage_R_curve=r_curve,
    )
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"strategic_leverage_{args.cycle}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
