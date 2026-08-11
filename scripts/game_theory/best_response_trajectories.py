#!/usr/bin/env python3
"""
Best-response trajectories: for a handful of individual races, trace
D_0 -> R_1(D_0) -> D_2(R_1) -> ... round-by-round through the same
Gauss-Seidel best-response dynamics that produce the full Nash equilibrium
(src/backtest/optimizer/nash.py, scripts/solve_nash_equilibrium.py), and
plot each race's (D_i, R_i) path. The claim this is meant to make visible
at the race level: the apparent one-shot opportunity a race shows on move
one (RegretD/RegretR, scripts/game_theory/race_level_exploitability.py) is
largely competed away over subsequent best responses, not just in
aggregate but race-by-race.

This is a SEPARATE, SHORTER run from the citable +0.10-seat equilibrium
already saved in outputs/nash_equilibrium_2024.json (that run used
damping_theta=0.5, max_rounds=40, ~50 min wall clock per FINDINGS.md).
Here, damping_theta=1.0 (undamped) and max_rounds is capped small (default
6) deliberately -- each round costs ~80s (two full 433-race SLSQP solves),
so a full damped 40-round run would take almost an hour. This run is for
VISUALIZING the early convergence trend on a handful of races, not for
re-deriving the headline number; it will generally not have converged to
tol_dollars within max_rounds, and that's expected, not a bug.

Race selection: by default, the largest-|s_D|/|s_R| competitive races from
race_level_exploitability_competitive_only_{cycle}.csv (run that script
first), plus any --races explicitly named on the command line.

Usage:
    python scripts/game_theory/best_response_trajectories.py --cycle 2024
    python scripts/game_theory/best_response_trajectories.py --cycle 2024 --races NC-13,FL-27,GA-07 --max-rounds 8
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi": 150,
})

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

from backtest import config
from backtest.data import fec
from backtest.data.universe import build_universe
from backtest.optimizer import nash

import solve_bellman_lsm as lsm  # noqa: E402
from race_level_exploitability import load_cand_r_total  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("best_response_trajectories")


def pick_default_races(out_dir: Path, cycle: int, n_per_side: int = 3) -> list[str]:
    """Largest |s_D| and |s_R| competitive races from the exploitability
    table, if it's been run; falls back to an empty list (caller must pass
    --races) if not."""
    comp_path = out_dir / f"race_level_exploitability_competitive_only_{cycle}.csv"
    if not comp_path.exists():
        logger.warning(f"{comp_path} not found -- run race_level_exploitability.py first, "
                        "or pass --races explicitly.")
        return []
    df = pd.read_csv(comp_path)
    top_d = df.reindex(df["s_D"].abs().sort_values(ascending=False).index)["district_id"].head(n_per_side)
    top_r = df.reindex(df["s_R"].abs().sort_values(ascending=False).index)["district_id"].head(n_per_side)
    picks = list(dict.fromkeys(list(top_d) + list(top_r)))  # dedupe, preserve order
    logger.info(f"Auto-selected races from competitive-only exploitability table: {picks}")
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description="Race-level best-response trajectories")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--cap-fraction-d", type=float, default=0.15)
    parser.add_argument("--cap-fraction-r", type=float, default=0.15)
    parser.add_argument("--damping-theta", type=float, default=1.0)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--races", type=str, default=None,
                         help="Comma-separated district IDs, e.g. NC-13,FL-27. "
                              "Default: auto-pick from the competitive-only exploitability table.")
    args = parser.parse_args()

    engine_out_dir = config.outputs_path()
    out_dir = engine_out_dir / "game_theory"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading estimation artifacts…")
    coef, sigma_model = lsm.load_coef_and_sigma()

    logger.info(f"Building {args.cycle} race universe…")
    races = build_universe(cycle=args.cycle)
    n = len(races)
    district_ids = [r.district_id for r in races]

    requested = args.races.split(",") if args.races else pick_default_races(out_dir, args.cycle)
    if not requested:
        logger.error("No races selected (auto-pick found nothing and --races wasn't passed). Exiting.")
        return
    track_idx = {d: district_ids.index(d) for d in requested if d in district_ids}
    missing = set(requested) - set(track_idx)
    if missing:
        logger.warning(f"Not in universe, skipping: {sorted(missing)}")
    logger.info(f"Tracking {len(track_idx)} races: {list(track_idx.keys())}")

    cand_r_total = load_cand_r_total(races, args.cycle)
    floors_d = np.array([r.cand_d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    d0 = np.array([r.d_total for r in races])
    party_budget_d = float(np.sum(d0 - floors_d))
    party_budget_r = float(np.sum(r0 - cand_r_total))

    logger.info(f"Solving best-response dynamics with per-race tracking "
                f"(damping_theta={args.damping_theta}, max_rounds={args.max_rounds}, "
                f"~80s/round expected)…")
    result = nash.solve_best_response_dynamics(
        races, coef, sigma_model, cand_r_total, party_budget_d, party_budget_r,
        cap_fraction_d=args.cap_fraction_d, cap_fraction_r=args.cap_fraction_r,
        damping_theta=args.damping_theta, max_rounds=args.max_rounds,
        track_races=True,
    )
    logger.info(f"Ran {result.n_iterations} rounds (converged={result.converged}); "
                f"E[D seats] round trajectory: "
                f"{[round(h['e_seats_d'], 2) for h in result.history]}")

    # ---- assemble per-race trajectories: round 0 = observed, then each history round ----
    trajectories: dict[str, dict] = {}
    for district_id, i in track_idx.items():
        d_path = [float(d0[i])]
        r_path = [float(r0[i])]
        for h in result.history:
            d_path.append(float(floors_d[i] + h["party_d"][i]))
            r_path.append(float(cand_r_total[i] + h["party_r"][i]))
        trajectories[district_id] = {
            "cook_rating": races[i].cook_rating, "pvi": races[i].pvi,
            "d_path": d_path, "r_path": r_path,
        }

    json_path = out_dir / f"best_response_trajectories_{args.cycle}.json"
    with open(json_path, "w") as f:
        json.dump({
            "cycle": args.cycle, "damping_theta": args.damping_theta,
            "max_rounds": args.max_rounds, "n_rounds_run": result.n_iterations,
            "converged": result.converged,
            "aggregate_history": [{"round": h["round"], "e_seats_d": h["e_seats_d"],
                                    "e_seats_r": h["e_seats_r"]} for h in result.history],
            "trajectories": trajectories,
        }, f, indent=2)
    logger.info(f"Saved -> {json_path}")

    make_trajectory_plot(trajectories, args.cycle, out_dir)


def make_trajectory_plot(trajectories: dict[str, dict], cycle: int, out_dir: Path) -> None:
    """One panel per tracked race: D_i and R_i (party+floor totals, $) vs.
    best-response round. Round 0 = observed; each subsequent round
    alternates a D move then an R move within that round's Gauss-Seidel
    step, so what's plotted is the state AFTER each full round completes."""
    n_races = len(trajectories)
    ncols = min(3, n_races)
    nrows = int(np.ceil(n_races / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4 * nrows), squeeze=False)

    for idx, (district_id, t) in enumerate(trajectories.items()):
        ax = axes[idx // ncols][idx % ncols]
        rounds = list(range(len(t["d_path"])))
        ax.plot(rounds, np.array(t["d_path"]) / 1000, marker="o", ms=4, color="#2171b5", label="D total ($k)")
        ax.plot(rounds, np.array(t["r_path"]) / 1000, marker="o", ms=4, color="#de2d26", label="R total ($k)")
        ax.set_title(f"{district_id} ({t['cook_rating']}, PVI {t['pvi']:+.1f})", fontsize=10)
        ax.set_xlabel("best-response round (0 = observed)")
        ax.set_ylabel("$ thousands")
        ax.legend(fontsize=8, frameon=False)

    for idx in range(n_races, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"Race-level best-response trajectories — {cycle}\n"
                  "(illustrative early-round run, not the full damped equilibrium)", y=1.02)
    fig.tight_layout()
    fig_path = out_dir / f"best_response_trajectories_{cycle}.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved -> {fig_path}")


if __name__ == "__main__":
    main()
