"""
The strategic equilibrium (D*, R*) with D* = BR_D(R*), R* = BR_R(D*)
(project_spec.md Sections 11-12).

solve_nash() is a thin, renamed wrapper around the already-validated
backtest.optimizer.nash.find_nash_equilibrium_multi_start -- multi-start
agreement checking and non-convergence reporting come for free from that
solver (spec Section 12's "multiple starting allocations" and "multiple-
equilibrium detection" diagnostics).

iterate_best_response() adds the diagnostics that solver does NOT vary:
move order (D-first vs. R-first) and update timing (sequential Gauss-Seidel
vs. simultaneous Jacobi), per spec Section 12's explicit requirement to
check both rather than assume Gauss-Seidel D-first is the only reasonable
dynamic. "D_first"+"sequential" delegates to the validated solver directly;
the other three combinations are implemented here from the same br_d/br_r
building blocks, always scored via the SAME D-anchored formula
(payoff.p_win) for both sides' e_seats, replicating the like-for-like-
scoring fix documented in scripts/game_theory/race_level_exploitability.py
(mixing D-formula and R's-mirrored-formula scores produced a logically
impossible negative regret there).
"""

from __future__ import annotations

import numpy as np

from backtest.optimizer import nash as _nash_mod

from . import best_response as br
from . import payoff


def solve_nash(races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15,
                damping_theta: float = 1.0, max_rounds: int = 100,
                tol_dollars: float = 10_000.0):
    """(D*, R*) via multi-start Gauss-Seidel best-response dynamics, D-first,
    sequential -- the primary equilibrium object (spec Section 11)."""
    return _nash_mod.find_nash_equilibrium_multi_start(
        races, coef, sigma_model, cand_r_total, budget_d, budget_r,
        cap_fraction_d=cap_fraction_d, cap_fraction_r=cap_fraction_r,
        damping_theta=damping_theta, max_rounds=max_rounds, tol_dollars=tol_dollars,
    )


def _score(races, coef, sigma_model, party_d, floors_d, total_r):
    n = len(races)
    p = payoff.p_win(party_d, races, coef, sigma_model, total_r)
    e_d = payoff.expected_seats_d(p)
    return e_d, payoff.expected_seats_r(n, e_d)


def iterate_best_response(
    races, coef, sigma_model, cand_r_total, budget_d, budget_r,
    cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15,
    order: str = "D_first", mode: str = "sequential",
    init_party_d: np.ndarray | None = None, init_party_r: np.ndarray | None = None,
    damping_theta: float = 1.0, max_rounds: int = 100, tol_dollars: float = 10_000.0,
) -> dict:
    """order in {'D_first', 'R_first'}; mode in {'sequential', 'simultaneous'}.

    'sequential' == Gauss-Seidel: the second mover in a round sees the first
    mover's JUST-UPDATED allocation. 'simultaneous' == Jacobi: both sides
    best-respond to the OTHER side's allocation from the END of the PREVIOUS
    round, and both updates land together -- the more literal reading of
    "simultaneous move" (spec Section 12), typically slower to converge and
    more prone to cycling than Gauss-Seidel, which is exactly why both are
    required diagnostics rather than picking one silently.

    Returns a dict shaped like NashResult's fields, plus `cycle_detected`:
    True if the last 4 rounds' combined allocation delta failed to shrink
    monotonically (a simple, conservative tell for non-convergent orbits,
    not a proof of periodicity).
    """
    if order not in ("D_first", "R_first"):
        raise ValueError(f"order must be 'D_first' or 'R_first', got {order!r}")
    if mode not in ("sequential", "simultaneous"):
        raise ValueError(f"mode must be 'sequential' or 'simultaneous', got {mode!r}")

    if order == "D_first" and mode == "sequential":
        res = _nash_mod.solve_best_response_dynamics(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            cap_fraction_d, cap_fraction_r, init_party_d=init_party_d, init_party_r=init_party_r,
            damping_theta=damping_theta, max_rounds=max_rounds, tol_dollars=tol_dollars,
        )
        deltas = [h["max_delta_d"] + h["max_delta_r"] for h in res.history]
        return {
            "party_d": res.party_d, "party_r": res.party_r,
            "e_seats_d": res.e_seats_d, "e_seats_r": res.e_seats_r,
            "converged": res.converged, "n_iterations": res.n_iterations,
            "history": res.history, "cycle_detected": _detect_cycle(deltas),
        }

    n = len(races)
    floors_d = np.array([r.cand_d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    d0 = np.array([r.d_total for r in races])
    cand_r_total = np.asarray(cand_r_total, dtype=float)

    party_d = init_party_d.copy() if init_party_d is not None else np.maximum(d0 - floors_d, 0.0)
    party_r = init_party_r.copy() if init_party_r is not None else np.maximum(r0 - cand_r_total, 0.0)

    history: list[dict] = []
    converged = False
    it = 0
    for it in range(max_rounds):
        if mode == "sequential":  # R_first, sequential
            r_total_current = cand_r_total + party_r
            res_r = br.br_r(races, coef, sigma_model, total_d=floors_d + party_d,
                             cand_r_total=cand_r_total, budget_r=budget_r,
                             cap_fraction_r=cap_fraction_r, x0=party_r)
            party_r_new = damping_theta * res_r.party + (1.0 - damping_theta) * party_r
            r_total_after_r = cand_r_total + party_r_new
            res_d = br.br_d(races, coef, sigma_model, total_r=r_total_after_r,
                             budget_d=budget_d, cap_fraction_d=cap_fraction_d, x0=party_d)
            party_d_new = damping_theta * res_d.party + (1.0 - damping_theta) * party_d
        else:  # simultaneous, either order (order only matters for tie-breaking display)
            r_total_prev = cand_r_total + party_r
            d_total_prev = floors_d + party_d
            res_d = br.br_d(races, coef, sigma_model, total_r=r_total_prev,
                             budget_d=budget_d, cap_fraction_d=cap_fraction_d, x0=party_d)
            res_r = br.br_r(races, coef, sigma_model, total_d=d_total_prev,
                             cand_r_total=cand_r_total, budget_r=budget_r,
                             cap_fraction_r=cap_fraction_r, x0=party_r)
            party_d_new = damping_theta * res_d.party + (1.0 - damping_theta) * party_d
            party_r_new = damping_theta * res_r.party + (1.0 - damping_theta) * party_r

        max_delta_d = float(np.max(np.abs(party_d_new - party_d))) if n else 0.0
        max_delta_r = float(np.max(np.abs(party_r_new - party_r))) if n else 0.0
        party_d, party_r = party_d_new, party_r_new

        e_d, e_r = _score(races, coef, sigma_model, party_d, floors_d, cand_r_total + party_r)
        history.append({"round": it, "max_delta_d": max_delta_d, "max_delta_r": max_delta_r,
                         "e_seats_d": e_d, "e_seats_r": e_r})
        if max_delta_d < tol_dollars and max_delta_r < tol_dollars:
            converged = True
            break

    e_seats_d, e_seats_r = _score(races, coef, sigma_model, party_d, floors_d, cand_r_total + party_r)
    deltas = [h["max_delta_d"] + h["max_delta_r"] for h in history]
    return {
        "party_d": party_d, "party_r": party_r,
        "e_seats_d": e_seats_d, "e_seats_r": e_seats_r,
        "converged": converged, "n_iterations": it + 1,
        "history": history, "cycle_detected": _detect_cycle(deltas),
    }


def _detect_cycle(deltas: list[float], window: int = 4) -> bool:
    """Conservative non-convergence tell: over the last `window` rounds, the
    combined allocation delta never dropped below its value `window` rounds
    earlier -- consistent with an orbit rather than a shrinking approach to
    a fixed point. Not a formal periodicity proof."""
    if len(deltas) < window + 1:
        return False
    tail = deltas[-(window + 1):]
    return min(tail[1:]) >= tail[0]
