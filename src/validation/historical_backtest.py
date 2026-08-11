"""
Per-cycle orchestration (project_spec.md Section 15): for one historical
cycle, compute observed allocations, both sides' unilateral best responses,
the iterated strategic equilibrium, regret/exploitability, race-level
(Z_D, Z_R), and persistent strategic value for a candidate race set --
then save everything needed for the cross-cycle replication check (spec
Section 26: "does the near-zero aggregate Nash result replicate?").

Expensive: one call runs multiple full-universe SLSQP best-response solves
(each ~1-2 min on the 433-race 2024 universe, per
scripts/game_theory/race_level_exploitability.py's own timing notes) plus a
damped multi-round Nash solve (~50 min per FINDINGS.md for the citable
result). Not run as part of scaffolding -- scripts/run_historical_backtest.py
is the CLI entry point for actually executing this per cycle.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from game import equilibrium, exploitability, persistent_value as pv

logger = logging.getLogger(__name__)

DEFAULT_N_PSV_RACES = 6  # per side; keeps a full cycle run tractable


def run_cycle(cycle: int, races, coef, sigma_model, cand_r_total: np.ndarray,
              cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15,
              damping_theta: float = 0.5, max_rounds: int = 40,
              n_psv_races: int = DEFAULT_N_PSV_RACES, psv_delta: float = 100_000.0,
              results_dir: Path | None = None) -> dict:
    """Full per-cycle pipeline. Returns the assembled results dict and, if
    results_dir is given, writes it to results_dir / f"cycle_{cycle}.json"."""
    n = len(races)
    floors_d = np.array([r.cand_d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    d0 = np.array([r.d_total for r in races])
    budget_d = float(np.sum(d0 - floors_d))
    budget_r = float(np.sum(r0 - cand_r_total))

    logger.info(f"[{cycle}] Race-level surplus + one-shot exploitability…")
    surplus = exploitability.race_level_surplus(
        races, coef, sigma_model, cand_r_total, budget_d, budget_r, cap_fraction_d, cap_fraction_r
    )
    exploit = exploitability.exploitability(
        races, coef, sigma_model, cand_r_total, budget_d, budget_r, cap_fraction_d, cap_fraction_r
    )

    logger.info(f"[{cycle}] Iterated Nash equilibrium (damping_theta={damping_theta}, "
                f"max_rounds={max_rounds})…")
    nash_result = equilibrium.solve_nash(
        races, coef, sigma_model, cand_r_total, budget_d, budget_r,
        cap_fraction_d=cap_fraction_d, cap_fraction_r=cap_fraction_r,
        damping_theta=damping_theta, max_rounds=max_rounds,
    )

    logger.info(f"[{cycle}] Persistent strategic value for top {n_psv_races} |Z_D| "
                f"and top {n_psv_races} |Z_R| races…")
    top_d_idx = np.argsort(-np.abs(surplus["Z_D"]))[:n_psv_races]
    top_r_idx = np.argsort(-np.abs(surplus["Z_R"]))[:n_psv_races]
    psv_d = [
        pv.persistent_strategic_value_d(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            race_idx=int(i), delta=psv_delta, cap_fraction_d=cap_fraction_d, cap_fraction_r=cap_fraction_r,
        )
        for i in top_d_idx
    ]
    psv_r = [
        pv.persistent_strategic_value_r(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            race_idx=int(i), delta=psv_delta, cap_fraction_d=cap_fraction_d, cap_fraction_r=cap_fraction_r,
        )
        for i in top_r_idx
    ]

    l1_distance = float(
        np.sum(np.abs(surplus["party_d_obs"] - nash_result.party_d))
        + np.sum(np.abs(surplus["party_r_obs"] - nash_result.party_r))
    )

    out = {
        "cycle": cycle,
        "n_races": n,
        "budget_d": budget_d,
        "budget_r": budget_r,
        "cap_fraction_d": cap_fraction_d,
        "cap_fraction_r": cap_fraction_r,
        "exploitability": exploit,
        "nash": {
            "converged": nash_result.converged,
            "n_iterations": nash_result.n_iterations,
            "e_seats_d": nash_result.e_seats_d,
            "e_seats_r": nash_result.e_seats_r,
            "multi_start_agreement": nash_result.multi_start_agreement,
        },
        "l1_distance_observed_to_nash": l1_distance,
        "persistent_value_D": psv_d,
        "persistent_value_R": psv_r,
        "quadrant_counts": _quadrant_counts(surplus),
    }

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"cycle_{cycle}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=_json_default)
        logger.info(f"[{cycle}] Saved -> {out_path}")

    return out


def _quadrant_counts(surplus: dict) -> dict:
    from game.exploitability import quadrant
    counts: dict[str, int] = {}
    for sd, sr in zip(surplus["S_D"], surplus["S_R"]):
        q = quadrant(float(sd), float(sr))
        counts[q] = counts.get(q, 0) + 1
    return counts


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)
