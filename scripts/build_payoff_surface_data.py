#!/usr/bin/env python3
"""
Freeze one race's p_i(D_i, R_i) surface + the observed-point derivatives
(MSG_D, MSG_R) to results/ for visuals/scenes/race_payoff_surface.py to
consume -- per visuals/README.md's design rule, Manim never calculates the
result, it only reads a frozen file this script already computed.

Usage:
    python scripts/build_payoff_surface_data.py --cycle 2024 --district MI-08
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
from game import gradients, payoff  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("build_payoff_surface_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze one race's payoff surface + derivatives")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--district", type=str, required=True)
    parser.add_argument("--n-grid", type=int, default=40)
    parser.add_argument("--range-multiple", type=float, default=2.5,
                         help="Grid spans [0, range_multiple * max(observed_D, observed_R)].")
    args = parser.parse_args()

    state = build_cycle_state(args.cycle, 0.15, 0.15)
    races = state["races"]
    district_ids = [r.district_id for r in races]
    if args.district not in district_ids:
        raise ValueError(f"{args.district} not in the {args.cycle} universe")
    i = district_ids.index(args.district)

    floors_d = np.array([r.cand_d_total for r in races])
    d0 = np.array([r.d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    cand_r_total = state["cand_r_total"]
    party_d_obs = float(max(d0[i] - floors_d[i], 0.0))
    party_r_obs = float(max(r0[i] - cand_r_total[i], 0.0))

    d_max = args.range_multiple * max(party_d_obs, party_r_obs, 1.0)
    d_grid = np.linspace(0.0, d_max, args.n_grid)
    r_grid = np.linspace(0.0, d_max, args.n_grid)

    arrays = payoff.baseline_arrays(races, state["coef"], state["sigma_model"], cand_r_total)

    logger.info(f"Building {args.n_grid}x{args.n_grid} p_win grid for {args.district} "
                f"({args.cycle}), D/R in [0, ${d_max:,.0f}]…")
    p_grid = np.zeros((args.n_grid, args.n_grid))
    for a, party_d in enumerate(d_grid):
        party_d_vec = party_d_obs * np.ones(len(races))
        party_d_vec[i] = party_d
        for b, party_r in enumerate(r_grid):
            party_r_vec = party_r_obs * np.ones(len(races))
            party_r_vec[i] = party_r
            p = payoff.p_win_shared(party_d_vec, party_r_vec, arrays)
            p_grid[a, b] = float(p[i])

    party_d_full = np.maximum(d0 - floors_d, 0.0)
    party_r_full = np.maximum(r0 - cand_r_total, 0.0)
    msg_d_obs = float(gradients.msg_d(party_d_full, party_r_full, arrays)[i])
    msg_r_obs = float(gradients.msg_r(party_d_full, party_r_full, arrays)[i])
    p_win_obs = float(payoff.p_win_shared(party_d_full, party_r_full, arrays)[i])

    out = {
        "cycle": args.cycle,
        "district_id": args.district,
        "cook_rating": races[i].cook_rating,
        "pvi": races[i].pvi,
        "d_grid": d_grid.tolist(),
        "r_grid": r_grid.tolist(),
        "p_grid": p_grid.tolist(),
        "party_d_obs": party_d_obs,
        "party_r_obs": party_r_obs,
        "p_win_obs": p_win_obs,
        "MSG_D_obs": msg_d_obs,
        "MSG_R_obs": msg_r_obs,
    }
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"payoff_surface_{args.district}_{args.cycle}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"p_win_obs={p_win_obs:.4f}  MSG_D={msg_d_obs:.3e}  MSG_R={msg_r_obs:.3e}")
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
