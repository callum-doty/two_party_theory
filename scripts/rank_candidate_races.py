#!/usr/bin/env python3
"""
Union candidate pre-screen for the K~15-20 sequential-game stress test
(2026-08-14 follow-up): the K~8 "primary" pool (compute_strategic_leverage.py's
top-4-swing + top-4-|Z|) already showed the screen matters -- FL-07 (2022 R)
had poor CURRENT leverage but a large, previously-invisible mid-season
strategic-window opportunity once its full trajectory was swept. This
script builds a broader, more principled screen so the K-expansion isn't
bottlenecked on one candidate-selection heuristic:

    union(top current |Z| (leverage), top V_uni, equilibrium swing races,
          "revealed contested-ness" proxy for likely-erosion/recovery slope)

Three of the four criteria are genuinely cheap (closed-form, no solves):
  - top |Z|: already-computed race_surplus_{cycle}.csv (Z_D/Z_R, a
    normalized marginal-surplus statistic), same source
    strategic_leverage.py's original screen used.
  - top V_uni: computed FRESH here across the full 433-race universe
    (persistent_value's own _finance_delta + one payoff.p_win_shared
    eval per race -- vectorized, no best-response solve) rather than
    reused from strategic_leverage.py's results, which only ever
    evaluated its OWN 8-candidate pool.
  - equilibrium swing: results/equilibrium_support_composition_{cycle}.json's
    already-computed top-CV races from the double-oracle mixed
    equilibrium's support.

**The fourth criterion, "largest strategic-window slope," is NOT
computed directly** -- doing so exactly would require the very per-race
date-sweep (BR_R at every reference date) this pre-screen exists to avoid
running on the full 433-race universe; that is precisely the circularity
this project has hit before when a diagnostic needs the thing it is
trying to select candidates FOR. Approximated instead by a "revealed
contested-ness" proxy: races where the OPPONENT's total observed spend is
large (revealed preference that the opponent considers this race a
priority, which is exactly what a full best response would defend hardest
early -- producing large initial erosion) among races with material
V_uni. FL-07's story matches this proxy directly: Democrats had already
committed serious money there in aggregate. Flagged explicitly as an
approximation; the pre-screen's real validation is whether the expensive
date-sweep run on the resulting union confirms genuine mid-season slope,
not the proxy score itself.

Usage:
    python scripts/rank_candidate_races.py --cycle 2024
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_cycle_state import build_cycle_state  # noqa: E402
from game import gradients, payoff  # noqa: E402
from game.persistent_value import RETENTION_MATERIALITY_THRESHOLD, _finance_delta  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("rank_candidate_races")

DELTA = 1_000_000.0
MIN_PARTY_SPEND = 10_000.0  # matches historical_backtest.py / persistent_value.py's own $0-spend-artifact fix
TOP_N_PER_CRITERION = 5


def v_uni_all(races, coef, sigma_model, cand_r_total, party_d_obs, party_r_obs, cap_d, cap_r, arrays):
    """V_uni_i for EVERY race, both sides, closed form (no best-response
    solve) -- one payoff.p_win_shared eval per race after financing a
    $1M deviation from that side's own lowest-MSG funded race."""
    n = len(races)
    baseline_d = float(payoff.p_win_shared(party_d_obs, party_r_obs, arrays).sum())
    msg_d = gradients.msg_d(party_d_obs, party_r_obs, arrays)
    msg_r = gradients.msg_r(party_d_obs, party_r_obs, arrays)

    v_uni_d = np.empty(n)
    v_uni_r = np.empty(n)
    for i in range(n):
        dev_d = _finance_delta(party_d_obs, msg_d, i, DELTA, cap_d)
        v_uni_d[i] = float(payoff.p_win_shared(dev_d, party_r_obs, arrays).sum()) - baseline_d

        dev_r = _finance_delta(party_r_obs, msg_r, i, DELTA, cap_r)
        e_d = float(payoff.p_win_shared(party_d_obs, dev_r, arrays).sum())
        v_uni_r[i] = (float(n) - e_d) - (float(n) - baseline_d)
    return v_uni_d, v_uni_r


def main() -> None:
    parser = argparse.ArgumentParser(description="Union candidate pre-screen for the K~15-20 stress test")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    args = parser.parse_args()

    state = build_cycle_state(args.cycle, args.cap_fraction_d, args.cap_fraction_r)
    races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]
    cand_r_total, budget_d, budget_r = state["cand_r_total"], state["budget_d"], state["budget_r"]
    cap_d, cap_r = state["cap_d"], state["cap_r"]
    d0 = np.array([r.d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    floors_d = np.array([r.cand_d_total for r in races])
    party_d_obs = np.maximum(d0 - floors_d, 0.0)
    party_r_obs = np.maximum(r0 - cand_r_total, 0.0)
    district_ids = [r.district_id for r in races]
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)

    logger.info("Computing V_uni for all 433 races, both sides (closed form, no solves)…")
    v_uni_d, v_uni_r = v_uni_all(races, coef, sigma_model, cand_r_total, party_d_obs, party_r_obs, cap_d, cap_r, arrays)

    surplus = pd.read_csv(REPO_ROOT / "results" / f"race_surplus_{args.cycle}.csv").set_index("district_id")
    equil = json.load(open(REPO_ROOT / "results" / f"equilibrium_support_composition_{args.cycle}.json"))

    def top_n(values: np.ndarray, mask: np.ndarray, n: int) -> list[str]:
        idx = np.where(mask)[0]
        idx = idx[np.argsort(-values[idx])][:n]
        return [district_ids[i] for i in idx]

    out = {}
    for side, party_own, party_opp, v_uni, z_col, side_key in (
        ("D", party_d_obs, party_r_obs, v_uni_d, "Z_D", "d_side"),
        ("R", party_r_obs, party_d_obs, v_uni_r, "Z_R", "r_side"),
    ):
        funded_mask = party_own > MIN_PARTY_SPEND
        z_vals = np.array([abs(surplus.loc[d, z_col]) if d in surplus.index else -np.inf for d in district_ids])
        top_z = top_n(z_vals, funded_mask, TOP_N_PER_CRITERION)

        top_vuni = top_n(v_uni, np.ones(len(district_ids), dtype=bool), TOP_N_PER_CRITERION)

        top_swing = [r["district_id"] for r in equil[side_key]["top_swing_by_cv"][:TOP_N_PER_CRITERION]]

        material_mask = v_uni > RETENTION_MATERIALITY_THRESHOLD
        top_contested = top_n(party_opp, material_mask, TOP_N_PER_CRITERION)

        union = sorted(set(top_z) | set(top_vuni) | set(top_swing) | set(top_contested))
        out[side] = dict(
            top_z=top_z, top_v_uni=top_vuni, equilibrium_swing=top_swing,
            revealed_contested=top_contested, union=union,
        )
        logger.info(f"{args.cycle} {side}-side:")
        logger.info(f"  top |Z|:              {top_z}")
        logger.info(f"  top V_uni:            {top_vuni}")
        logger.info(f"  equilibrium swing:    {top_swing}")
        logger.info(f"  revealed contested:   {top_contested}")
        logger.info(f"  UNION (K={len(union)}):    {union}")

    out_path = REPO_ROOT / "results" / f"candidate_union_{args.cycle}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
