#!/usr/bin/env python3
"""
Large-delta extension of compute_strategic_leverage.py, for the same top-3-
per-side candidates already identified there (docs/methodology.md's
"Strategic leverage and response displacement" section) -- pushes delta up
toward each side's per-race cap to see where the shared "financing pool"
(the small, stable set of races each side draws on to fund a response,
found at delta <= $2M) actually runs out and the opponent is forced to
start touching races it would otherwise leave alone.

Deltas: $3M, $5M, $8M, $12M -- deliberately pushed past what's plausible for
a single race in practice, specifically to find the exhaustion point, not to
suggest committees actually move $12M into one district. Exact SLSQP
throughout (see strategic_leverage.py's module docstring for why the
surrogate isn't trustworthy for a displacement map). Capping-aware: at these
sizes the target race's own per-race cap (cap_fraction * budget: ~$15M for
D, $7-14M for R depending on cycle) can bind before delta is fully placed;
every row reports delta_requested/delta_deployed/capped, and leverage is
normalized by delta_deployed (src/game/strategic_leverage.py's docstring
has the full reasoning).

Usage:
    python scripts/compute_strategic_leverage_large_delta.py --cycle 2024
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
logger = logging.getLogger("compute_strategic_leverage_large_delta")

LARGE_DELTAS = [3_000_000.0, 5_000_000.0, 8_000_000.0, 12_000_000.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Large-delta strategic leverage sweep (exact SLSQP)")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    args = parser.parse_args()

    prior_path = REPO_ROOT / "results" / f"strategic_leverage_{args.cycle}.json"
    if not prior_path.exists():
        raise SystemExit(f"{prior_path} not found -- run compute_strategic_leverage.py --cycle {args.cycle} first")
    prior = json.load(open(prior_path))
    top_d = sorted({r["district_id"] for r in prior["leverage_D_curve"]})
    top_r = sorted({r["district_id"] for r in prior["leverage_R_curve"]})
    logger.info(f"Cycle {args.cycle}: extending top-3 D candidates {top_d} and R candidates {top_r} "
                f"to deltas {[d/1e6 for d in LARGE_DELTAS]} ($M)")

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

    cap_d = args.cap_fraction_d * budget_d
    cap_r = args.cap_fraction_r * budget_r
    logger.info(f"Per-race caps: D=${cap_d:,.0f}  R=${cap_r:,.0f}")

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

    d_rows: list[dict] = []
    for did in top_d:
        i = idx_by_district[did]
        headroom = cap_d - party_d_obs[i]
        logger.info(f"D-side {did}: headroom to cap = ${headroom:,.0f}")
        rows = lev.leverage_curve_d(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            race_idx=i, deltas=LARGE_DELTAS,
            party_d_obs=party_d_obs, party_r_obs=party_r_obs, party_r_star=party_r_star,
            baseline_d=baseline_d, arrays=arrays,
            cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
            use_surrogate=False,
        )
        for row in rows:
            flag = " [CAPPED]" if row["capped"] else ""
            logger.info(f"    ${row['delta']/1e6:.0f}M requested (${row['delta_deployed']/1e6:.2f}M deployed{flag}): "
                        f"PSV={row['PSV']:+.4f} leverage={row['leverage_seats_per_million']:+.4f} "
                        f"reshuffle=${row['reshuffle_l1']:,.0f}")
        d_rows.extend(rows)

    r_rows: list[dict] = []
    for did in top_r:
        i = idx_by_district[did]
        headroom = cap_r - party_r_obs[i]
        logger.info(f"R-side {did}: headroom to cap = ${headroom:,.0f}")
        rows = lev.leverage_curve_r(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            race_idx=i, deltas=LARGE_DELTAS,
            party_d_obs=party_d_obs, party_r_obs=party_r_obs, party_d_star=party_d_star,
            baseline_r=baseline_r, arrays=arrays, n_races=n,
            cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
            use_surrogate=False,
        )
        for row in rows:
            flag = " [CAPPED]" if row["capped"] else ""
            logger.info(f"    ${row['delta']/1e6:.0f}M requested (${row['delta_deployed']/1e6:.2f}M deployed{flag}): "
                        f"PSV={row['PSV']:+.4f} leverage={row['leverage_seats_per_million']:+.4f} "
                        f"reshuffle=${row['reshuffle_l1']:,.0f}")
        r_rows.extend(rows)

    out = dict(
        cycle=args.cycle, large_deltas=LARGE_DELTAS, cap_d=cap_d, cap_r=cap_r,
        baseline_d=baseline_d, baseline_r=baseline_r,
        leverage_D_large=d_rows, leverage_R_large=r_rows,
    )
    out_path = REPO_ROOT / "results" / f"strategic_leverage_large_delta_{args.cycle}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
