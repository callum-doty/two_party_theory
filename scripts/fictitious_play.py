#!/usr/bin/env python3
"""
Continuous fictitious play for one cycle (docs' "Revised order of work"
#2): the cheap first mixed-strategy diagnostic to run before the full
double-oracle solver (scripts/double_oracle.py). Each side best-responds
to the OTHER side's TIME-AVERAGE allocation so far -- see
game/equilibrium.py's fictitious_play() docstring for what this is and
isn't a guarantee of in a continuous (non-matrix) game.

If the average pair's regret trends toward 0 and stabilizes there, that is
evidence the time-average IS converging toward something equilibrium-like
even though the last-iterate Gauss-Seidel dynamics (scripts/solve_nash.py)
cycle -- consistent with "no stable deterministic portfolio, but a stable
DISTRIBUTION over near-optimal portfolios" (the working thesis
minimize_pure_exploitability.py's E_min result is also testing).

Usage:
    python scripts/fictitious_play.py --cycle 2024
    python scripts/fictitious_play.py --cycle 2024 --rounds 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from build_cycle_state import build_cycle_state  # noqa: E402
from game import equilibrium  # noqa: E402
from game import exploitability  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("fictitious_play")


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous fictitious play for one cycle")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    parser.add_argument("--rounds", type=int, default=400)
    args = parser.parse_args()

    state = build_cycle_state(args.cycle, args.cap_fraction_d, args.cap_fraction_r)
    races, coef, sigma_model = state["races"], state["coef"], state["sigma_model"]
    cand_r_total = state["cand_r_total"]
    budget_d, budget_r = state["budget_d"], state["budget_r"]
    n = len(races)

    floors_d = np.array([r.cand_d_total for r in races])
    d0 = np.array([r.d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    party_d_obs = np.maximum(d0 - floors_d, 0.0)
    party_r_obs = np.maximum(r0 - cand_r_total, 0.0)

    starts = {
        "observed": (party_d_obs, party_r_obs),
        "uniform": (np.full(n, budget_d / n), np.full(n, budget_r / n)),
    }

    results = {}
    for name, (init_d, init_r) in starts.items():
        logger.info(f"[{name}] running {args.rounds}-round fictitious play (surrogate)…")
        fp = equilibrium.fictitious_play(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            args.cap_fraction_d, args.cap_fraction_r,
            init_party_d=init_d, init_party_r=init_r,
            rounds=args.rounds, use_surrogate=True,
        )
        last = fp["history"][-1]
        window = fp["history"][-20:]
        e_tail = [h["exploitability_avg"] for h in window]
        logger.info(f"[{name}] final round: E(avg)={last['exploitability_avg']:.4f} "
                    f"(RegretD={last['regret_D_avg']:.4f}, RegretR={last['regret_R_avg']:.4f}); "
                    f"last-20-round E(avg) range [{min(e_tail):.4f}, {max(e_tail):.4f}]")
        results[name] = fp

    best_name = min(results, key=lambda k: results[k]["history"][-1]["exploitability_avg"])
    best = results[best_name]
    avg_d, avg_r = best["avg_party_d"], best["avg_party_r"]

    logger.info(f"Exact SLSQP check of the {best_name}-start average pair…")
    exact = exploitability.regret_at(
        races, coef, sigma_model, cand_r_total, budget_d, budget_r,
        avg_d, avg_r, args.cap_fraction_d, args.cap_fraction_r, use_surrogate=False,
        x0_d=avg_d, x0_r=avg_r,
    )
    logger.info(f"Exact regret of average pair: E={exact['exploitability']:.4f} "
                f"(RegretD={exact['regret_D']:.4f}, RegretR={exact['regret_R']:.4f})")

    out = {
        "cycle": args.cycle,
        "config": {"rounds": args.rounds, "cap_fraction_d": args.cap_fraction_d,
                   "cap_fraction_r": args.cap_fraction_r},
        "per_start": {
            name: {
                "final_exploitability_avg_surrogate": r["history"][-1]["exploitability_avg"],
                "final_regret_D_avg_surrogate": r["history"][-1]["regret_D_avg"],
                "final_regret_R_avg_surrogate": r["history"][-1]["regret_R_avg"],
                "last20_min": min(h["exploitability_avg"] for h in r["history"][-20:]),
                "last20_max": max(h["exploitability_avg"] for h in r["history"][-20:]),
                "trajectory_exploitability_avg": [h["exploitability_avg"] for h in r["history"]],
            }
            for name, r in results.items()
        },
        "best_start": best_name,
        "exact_check_of_average": {
            "regret_D": exact["regret_D"], "regret_R": exact["regret_R"],
            "exploitability": exact["exploitability"],
            "e_seats_d": exact["e_seats_d"], "e_seats_r": exact["e_seats_r"],
        },
    }
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fictitious_play_{args.cycle}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    np.save(out_dir / f"fictitious_play_avg_party_d_{args.cycle}.npy", avg_d)
    np.save(out_dir / f"fictitious_play_avg_party_r_{args.cycle}.npy", avg_r)
    logger.info(f"Saved -> {out_path}")
    logger.info(f"SUMMARY cycle={args.cycle}: fictitious-play average-pair exact E="
                f"{exact['exploitability']:.3f} (best start: {best_name})")


if __name__ == "__main__":
    main()
