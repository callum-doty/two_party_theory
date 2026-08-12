"""
The strategic equilibrium (D*, R*) with D* = BR_D(R*), R* = BR_R(D*)
(project_spec.md Sections 11-12).

Both BR_D and BR_R now search AND score against the SAME shared, symmetric
payoff (payoff.p_win_shared, via game/best_response.py) -- see that
module's docstring for why this package no longer delegates to
backtest.optimizer.nash's fixed-point solver: that solver's D branch used
the old D-anchored formula and its R branch used a separately-derived
mirrored ceiling, and handing R the literal D-anchored formula as an
"exact" fix (rather than a genuinely shared one) let R exploit an
unregularized downward extrapolation. solve_nash/iterate_best_response below
implement best-response dynamics directly against this package's own
br_d/br_r, with no remaining dependency on nash.py's iteration logic.

iterate_best_response() provides the diagnostics project_spec.md Section 12
requires: move order (D-first vs. R-first) and update timing (sequential
Gauss-Seidel vs. simultaneous Jacobi). solve_nash() is the primary
equilibrium object -- D-first, sequential, from 3 starting points (observed,
uniform, zero), reporting whether they agree.
"""

from __future__ import annotations

import numpy as np

from backtest.optimizer.nash import NashResult

from . import best_response as br
from . import best_response_surrogate as brs
from . import payoff


def _score(arrays: dict, party_d: np.ndarray, party_r: np.ndarray) -> tuple[float, float]:
    p = payoff.p_win_shared(party_d, party_r, arrays)
    e_d = float(p.sum())
    return e_d, float(len(party_d)) - e_d


def _br_d(use_surrogate: bool, races, coef, sigma_model, *, party_r, cand_r_total,
           budget_d, cap_fraction_d, x0):
    """Dispatches to the exact SLSQP or the validated surrogate BR_D --
    the surrogate has no warm-start (x0) argument, greedy water-filling
    doesn't need one."""
    if use_surrogate:
        return brs.br_d_surrogate(races, coef, sigma_model, party_r=party_r, cand_r_total=cand_r_total,
                                   budget_d=budget_d, cap_fraction_d=cap_fraction_d)
    return br.br_d(races, coef, sigma_model, party_r=party_r, cand_r_total=cand_r_total,
                    budget_d=budget_d, cap_fraction_d=cap_fraction_d, x0=x0)


def _br_r(use_surrogate: bool, races, coef, sigma_model, *, party_d, cand_r_total,
           budget_r, cap_fraction_r, x0):
    if use_surrogate:
        return brs.br_r_surrogate(races, coef, sigma_model, party_d=party_d, cand_r_total=cand_r_total,
                                   budget_r=budget_r, cap_fraction_r=cap_fraction_r)
    return br.br_r(races, coef, sigma_model, party_d=party_d, cand_r_total=cand_r_total,
                    budget_r=budget_r, cap_fraction_r=cap_fraction_r, x0=x0)


def iterate_best_response(
    races, coef, sigma_model, cand_r_total, budget_d, budget_r,
    cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15,
    order: str = "D_first", mode: str = "sequential",
    init_party_d: np.ndarray | None = None, init_party_r: np.ndarray | None = None,
    damping_theta: float = 1.0, max_rounds: int = 100, tol_dollars: float = 10_000.0,
    use_surrogate: bool = False, track_trajectory: bool = False,
) -> dict:
    """order in {'D_first', 'R_first'}; mode in {'sequential', 'simultaneous'}.

    'sequential' == Gauss-Seidel: the second mover in a round sees the first
    mover's JUST-UPDATED allocation. 'simultaneous' == Jacobi: both sides
    best-respond to the OTHER side's allocation from the END of the PREVIOUS
    round, and both updates land together -- the more literal reading of
    "simultaneous move" (spec Section 12), typically slower to converge and
    more prone to cycling than Gauss-Seidel, which is exactly why both are
    required diagnostics rather than picking one silently.

    use_surrogate=True drives the dynamics with the validated concave-
    envelope surrogate (game/best_response_surrogate.py) instead of exact
    SLSQP -- ~500-1,000x faster, which is what makes multi-hundred-round,
    multi-start runs tractable at all (an exact 150-round run took ~2.3
    hours; docs/methodology.md's 2,000-round x 3-start numbers were a
    surrogate run). Always re-check anything reported as a final number
    with exact SLSQP at that specific point -- the surrogate is an
    approximation, not a substitute for verification.

    track_trajectory=True stashes a copy of (party_d, party_r) into each
    history entry (as "party_d"/"party_r") -- off by default since 2,000
    rounds x 433 races x 2 sides x 8 bytes is small per run but adds up
    across multi-start sweeps, and most callers only need the summary
    fields already in history.

    Returns a dict shaped like NashResult's fields, plus `cycle_detected`:
    True if the last 4 rounds' combined allocation delta failed to shrink
    monotonically (a simple, conservative tell for non-convergent orbits,
    not a proof of periodicity)."""
    if order not in ("D_first", "R_first"):
        raise ValueError(f"order must be 'D_first' or 'R_first', got {order!r}")
    if mode not in ("sequential", "simultaneous"):
        raise ValueError(f"mode must be 'sequential' or 'simultaneous', got {mode!r}")

    n = len(races)
    floors_d = np.array([r.cand_d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    d0 = np.array([r.d_total for r in races])
    cand_r_total = np.asarray(cand_r_total, dtype=float)
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)

    party_d = init_party_d.copy() if init_party_d is not None else np.maximum(d0 - floors_d, 0.0)
    party_r = init_party_r.copy() if init_party_r is not None else np.maximum(r0 - cand_r_total, 0.0)

    history: list[dict] = []
    converged = False
    it = 0
    for it in range(max_rounds):
        if mode == "sequential" and order == "D_first":
            res_d = _br_d(use_surrogate, races, coef, sigma_model, party_r=party_r, cand_r_total=cand_r_total,
                           budget_d=budget_d, cap_fraction_d=cap_fraction_d, x0=party_d)
            party_d_new = damping_theta * res_d.party + (1.0 - damping_theta) * party_d
            res_r = _br_r(use_surrogate, races, coef, sigma_model, party_d=party_d_new, cand_r_total=cand_r_total,
                           budget_r=budget_r, cap_fraction_r=cap_fraction_r, x0=party_r)
            party_r_new = damping_theta * res_r.party + (1.0 - damping_theta) * party_r
        elif mode == "sequential":  # R_first
            res_r = _br_r(use_surrogate, races, coef, sigma_model, party_d=party_d, cand_r_total=cand_r_total,
                           budget_r=budget_r, cap_fraction_r=cap_fraction_r, x0=party_r)
            party_r_new = damping_theta * res_r.party + (1.0 - damping_theta) * party_r
            res_d = _br_d(use_surrogate, races, coef, sigma_model, party_r=party_r_new, cand_r_total=cand_r_total,
                           budget_d=budget_d, cap_fraction_d=cap_fraction_d, x0=party_d)
            party_d_new = damping_theta * res_d.party + (1.0 - damping_theta) * party_d
        else:  # simultaneous, either order (order only matters for tie-breaking display)
            res_d = _br_d(use_surrogate, races, coef, sigma_model, party_r=party_r, cand_r_total=cand_r_total,
                           budget_d=budget_d, cap_fraction_d=cap_fraction_d, x0=party_d)
            res_r = _br_r(use_surrogate, races, coef, sigma_model, party_d=party_d, cand_r_total=cand_r_total,
                           budget_r=budget_r, cap_fraction_r=cap_fraction_r, x0=party_r)
            party_d_new = damping_theta * res_d.party + (1.0 - damping_theta) * party_d
            party_r_new = damping_theta * res_r.party + (1.0 - damping_theta) * party_r

        max_delta_d = float(np.max(np.abs(party_d_new - party_d))) if n else 0.0
        max_delta_r = float(np.max(np.abs(party_r_new - party_r))) if n else 0.0
        party_d, party_r = party_d_new, party_r_new

        e_d, e_r = _score(arrays, party_d, party_r)
        entry = {"round": it, "max_delta_d": max_delta_d, "max_delta_r": max_delta_r,
                 "e_seats_d": e_d, "e_seats_r": e_r}
        if track_trajectory:
            entry["party_d"] = party_d.copy()
            entry["party_r"] = party_r.copy()
        history.append(entry)
        if max_delta_d < tol_dollars and max_delta_r < tol_dollars:
            converged = True
            break

    e_seats_d, e_seats_r = _score(arrays, party_d, party_r)
    deltas = [h["max_delta_d"] + h["max_delta_r"] for h in history]
    return {
        "party_d": party_d, "party_r": party_r,
        "e_seats_d": e_seats_d, "e_seats_r": e_seats_r,
        "converged": converged, "n_iterations": it + 1,
        "history": history, "cycle_detected": _detect_cycle(deltas),
    }


def solve_nash(races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15,
                damping_theta: float = 1.0, max_rounds: int = 100,
                tol_dollars: float = 10_000.0) -> NashResult:
    """(D*, R*) via multi-start Gauss-Seidel best-response dynamics, D-first,
    sequential -- the primary equilibrium object (spec Section 11). Runs
    from 3 starting points (observed baseline, uniform-across-races,
    zero-party-spend) and reports whether they agree, within
    tol_dollars*10 per race (spec Section 12's multi-start requirement).
    Existence/uniqueness are NOT guaranteed for this non-convex game -- if
    the starts disagree, that is reported in the returned NashResult's
    multi_start_agreement field, not silently averaged away or discarded."""
    n = len(races)
    starts = {
        "observed": (None, None),
        "uniform": (np.full(n, budget_d / max(n, 1)), np.full(n, budget_r / max(n, 1))),
        "zero": (np.zeros(n), np.zeros(n)),
    }
    per_start: dict[str, dict] = {}
    for name, (init_d, init_r) in starts.items():
        per_start[name] = iterate_best_response(
            races, coef, sigma_model, cand_r_total, budget_d, budget_r,
            cap_fraction_d, cap_fraction_r, order="D_first", mode="sequential",
            init_party_d=init_d, init_party_r=init_r,
            damping_theta=damping_theta, max_rounds=max_rounds, tol_dollars=tol_dollars,
        )

    names = list(starts.keys())
    max_pairwise_d = max(
        (float(np.max(np.abs(per_start[a]["party_d"] - per_start[b]["party_d"])))
         for i, a in enumerate(names) for b in names[i + 1:]),
        default=0.0,
    )
    max_pairwise_r = max(
        (float(np.max(np.abs(per_start[a]["party_r"] - per_start[b]["party_r"])))
         for i, a in enumerate(names) for b in names[i + 1:]),
        default=0.0,
    )
    converged_all = all(r["converged"] for r in per_start.values())
    agree = converged_all and max_pairwise_d < tol_dollars * 10 and max_pairwise_r < tol_dollars * 10

    base = per_start["observed"]
    multi_start_agreement = {
        "converged_all": converged_all,
        "agree_within_tolerance": agree,
        "max_pairwise_party_d_diff": max_pairwise_d,
        "max_pairwise_party_r_diff": max_pairwise_r,
        "per_start_converged": {k: v["converged"] for k, v in per_start.items()},
        "per_start_e_seats_d": {k: v["e_seats_d"] for k, v in per_start.items()},
        "per_start_n_iterations": {k: v["n_iterations"] for k, v in per_start.items()},
    }
    return NashResult(
        party_d=base["party_d"], party_r=base["party_r"],
        e_seats_d=base["e_seats_d"], e_seats_r=base["e_seats_r"],
        converged=base["converged"], n_iterations=base["n_iterations"],
        history=base["history"], multi_start_agreement=multi_start_agreement,
    )


def fictitious_play(races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                     cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15,
                     init_party_d: np.ndarray | None = None, init_party_r: np.ndarray | None = None,
                     rounds: int = 200, use_surrogate: bool = True) -> dict:
    """Continuous-action fictitious play (docs' "Revised order of work" #2,
    the cheap first mixed-strategy diagnostic before double oracle): each
    round, both sides best-respond to the OTHER side's TIME-AVERAGE
    allocation so far (not its most recent play, which is what
    iterate_best_response's Gauss-Seidel/Jacobi dynamics do). The realized
    plays are folded back into a running average on each side, and it is
    the pair of AVERAGES -- not the last round's pure best responses --
    that is fictitious play's actual output object.

    This is the standard finite-matrix-game algorithm applied to a
    continuous allocation space: best-responding to the mean of an
    opponent's play distribution equals best-responding to the
    distribution itself only when payoff is linear (bilinear) in the
    opponent's action, which payoff.p_win_shared's tanh saturation is NOT
    globally -- so convergence here is an empirical question this function
    answers, not a theorem being invoked. Robinson (1951)'s finite zero-sum
    guarantee (time-averages -> the value of the game) does not
    automatically transfer to this continuous-action relaxation.

    Regret of the average pair is tracked every round via `regret_every`-
    round-strided evaluation using the (fast, validated-to-~0.03-0.10-seat)
    surrogate BR by default -- exact SLSQP is for the final refinement,
    same division of labor as minimize_pure_exploitability.py."""
    from . import exploitability as expl  # local import: avoids a cycle (exploitability imports payoff/best_response only)

    n = len(races)
    floors_d = np.array([r.cand_d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    d0 = np.array([r.d_total for r in races])
    party_d0 = init_party_d.copy() if init_party_d is not None else np.maximum(d0 - floors_d, 0.0)
    party_r0 = init_party_r.copy() if init_party_r is not None else np.maximum(r0 - cand_r_total, 0.0)

    sum_d, sum_r = party_d0.copy(), party_r0.copy()
    count = 1
    avg_d, avg_r = sum_d.copy(), sum_r.copy()

    history: list[dict] = []
    for t in range(rounds):
        res_d = _br_d(use_surrogate, races, coef, sigma_model, party_r=avg_r, cand_r_total=cand_r_total,
                       budget_d=budget_d, cap_fraction_d=cap_fraction_d, x0=avg_d)
        res_r = _br_r(use_surrogate, races, coef, sigma_model, party_d=avg_d, cand_r_total=cand_r_total,
                       budget_r=budget_r, cap_fraction_r=cap_fraction_r, x0=avg_r)
        sum_d += res_d.party
        sum_r += res_r.party
        count += 1
        avg_d = sum_d / count
        avg_r = sum_r / count

        reg = expl.regret_at(races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                              avg_d, avg_r, cap_fraction_d, cap_fraction_r, use_surrogate=use_surrogate)
        history.append({"round": t, "regret_D_avg": reg["regret_D"], "regret_R_avg": reg["regret_R"],
                         "exploitability_avg": reg["exploitability"],
                         "e_seats_d_avg": reg["e_seats_d"], "e_seats_r_avg": reg["e_seats_r"]})

    return {"avg_party_d": avg_d, "avg_party_r": avg_r, "history": history, "n_rounds": rounds}


def _detect_cycle(deltas: list[float], window: int = 4) -> bool:
    """Conservative non-convergence tell: over the last `window` rounds, the
    combined allocation delta never dropped below its value `window` rounds
    earlier -- consistent with an orbit rather than a shrinking approach to
    a fixed point. Not a formal periodicity proof."""
    if len(deltas) < window + 1:
        return False
    tail = deltas[-(window + 1):]
    return min(tail[1:]) >= tail[0]
