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

from game import equilibrium, exploitability, payoff
from game import persistent_value as pv

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

    logger.info(f"[{cycle}] Isolated PSV baselines: U_D(D, BR_R(D)) and U_R(BR_D(D), R)…")
    # See game/persistent_value.py's module docstring: the literal spec
    # formula (baseline = observed U_D(D,R)/U_R(D,R)) makes PSV nearly
    # race-invariant whenever observed spending is itself far from either
    # side's own unilateral optimum -- true here (RegretD/RegretR both
    # positive and O(seats), not O(0.01 seats)). race_level_surplus() above
    # already solved BR_D(R_obs) and BR_R(D_obs) (its own res_d/res_r) --
    # reused here rather than re-solving, so this costs zero extra
    # best-response solves.
    res_d_star, res_r_star = surplus["res_d"], surplus["res_r"]
    party_d_obs = np.maximum(d0 - floors_d, 0.0)
    party_r_obs = np.maximum(r0 - cand_r_total, 0.0)
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
    baseline_d = float(payoff.p_win_shared(party_d_obs, res_r_star.party, arrays).sum())

    e_d_at_d_star = float(payoff.p_win_shared(res_d_star.party, party_r_obs, arrays).sum())
    baseline_r = float(n) - e_d_at_d_star

    # Candidate pool restricted to races with REAL current party spend before
    # ranking by |Z| -- found necessary 2026-08-11 investigating a retention
    # >100% anomaly in an unrestricted first run. Root cause, verified by
    # tracing individual races end to end: with the corrected (small)
    # DCCC/NRCC control budgets, the vast majority of races get exactly $0
    # of party money (379/433 for D, 394/433 for R on the 2024 universe) --
    # MSG_D/MSG_R evaluated AT $0 sits at the steepest, most unstable point
    # of the persuasion-ceiling curve (the SAME "low-spend MSG artifact"
    # scripts/game_theory/race_level_exploitability.py's own scatter-plot
    # code already documents and excludes via its "competitive_only" cut).
    # An unrestricted top-|Z| selection is dominated by this zero-spend
    # population by sheer sample size. Because these races' TRUE unilateral
    # value (evaluated at a REAL delta injection, not the instantaneous
    # derivative) saturates almost immediately and stays small even as delta
    # grows 10x, while R's full 433-race reoptimization in response to ANY
    # change to D's allocation pattern produces a comparably-sized (or
    # larger) second-order reshuffling effect under the shared budget
    # constraint, PSV/V_uni becomes numerically unstable at a near-zero
    # denominator -- not evidence that "opponent optimization increases
    # value," just division instability. Restricting to races DCCC/NRCC are
    # ACTUALLY funding is also the more faithful reading of spec Section 13
    # itself: PSV asks whether a real, currently-active opportunity survives
    # optimal response, not an artifact of a boundary derivative.
    min_party_spend = 10_000.0
    funded_d = np.where(surplus["party_d_obs"] > min_party_spend)[0]
    funded_r = np.where(surplus["party_r_obs"] > min_party_spend)[0]
    logger.info(f"[{cycle}] Persistent strategic value for top {n_psv_races} |Z_D| "
                f"and top {n_psv_races} |Z_R| races (restricted to currently-funded races: "
                f"{len(funded_d)}/{n} D, {len(funded_r)}/{n} R)…")
    top_d_idx = funded_d[np.argsort(-np.abs(surplus["Z_D"][funded_d]))[:n_psv_races]]
    top_r_idx = funded_r[np.argsort(-np.abs(surplus["Z_R"][funded_r]))[:n_psv_races]]
    psv_d = [
        pv.persistent_strategic_value_d(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            race_idx=int(i), delta=psv_delta, cap_fraction_d=cap_fraction_d, cap_fraction_r=cap_fraction_r,
            baseline_e_seats=baseline_d,
        )
        for i in top_d_idx
    ]
    psv_r = [
        pv.persistent_strategic_value_r(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            race_idx=int(i), delta=psv_delta, cap_fraction_d=cap_fraction_d, cap_fraction_r=cap_fraction_r,
            baseline_e_seats_r=baseline_r,
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
        "psv_baseline": "isolated",
        "psv_baseline_e_seats_d": baseline_d,
        "psv_baseline_e_seats_r": baseline_r,
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
