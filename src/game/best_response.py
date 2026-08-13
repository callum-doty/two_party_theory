"""
BR_D(R) and BR_R(D) (project_spec.md Section 9): each side's constrained
best response holding the opponent's spending fixed.

Both sides now search directly against payoff.p_win_shared -- the single,
symmetric, fixed-baseline payoff (see payoff.py's module docstring). This
package used to delegate to backtest.optimizer.nash.best_response(), whose
D branch reused the old D-anchored formula and whose R branch searched a
separately-derived, uncalibrated "mirrored ceiling" and only re-scored the
result against the D-anchored formula afterward. That mismatch was more than
cosmetic: handing BR_R the literal D-anchored formula as its own search
objective (the natural-seeming fix) let it exploit an unregularized
downward extrapolation -- dumping millions into races like HI-02 ($10 of R
candidate spending on record) and dragging a 99%-D seat to a modeled 33%.
docs/methodology.md has the incident writeup. payoff.p_win_shared's fixed,
two-sided ceiling removes that failure mode by construction, so there is no
longer any reason for the two sides to search different objectives: _solve
below is one function, parameterized only by which side's dollars are free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from . import payoff

SCALE = 1_000_000.0  # matches allocator.py / nash.py's SLSQP conditioning trick


@dataclass
class BestResponseResult:
    party: np.ndarray       # this side's own party-money allocation ($)
    e_seats_own: float      # this side's own expected seats at that allocation
    status: str


def _solve(side: str, arrays: dict, opp_party_fixed: np.ndarray, *,
           own_budget: float, own_cap_fraction: float,
           x0: np.ndarray | None, committed_own: np.ndarray | None = None) -> BestResponseResult:
    """One side's constrained best response under payoff.p_win_shared,
    holding the OTHER side's party allocation fixed at opp_party_fixed.
    side='D' maximizes sum(p); side='R' maximizes n - sum(p). Gradients are
    exact (payoff.grad_shared), not finite-differenced -- the shared
    formula's tanh saturation is smooth everywhere, so there's no ceiling
    kink to worry about the way the old exp-based one-sided ceiling had.

    committed_own (L_t, project_spec.md's irreversible-capital extension,
    2026-08-13): dollars per race ALREADY spent/locked, on top of the
    race's own uncontrolled floor -- src/backtest/dynamic/ledger.py's
    "committed capital enters as an addition to each race's existing
    spend floor" convention, applied here to the two-player game's own
    solver rather than the old single-player receding-horizon one. The
    search variable is the FLEXIBLE portion only (own_budget minus total
    committed = F_t); each race's flexible room is capped at
    `own_cap_fraction * own_budget - committed_own[i]` so total spend
    (committed + flexible) never exceeds the same per-race cap an
    uncommitted solve would respect. None (default) is IDENTICAL to no
    commitment (all-zero L_t): flexible room = the full cap, flexible
    budget = the full own_budget -- existing callers that never pass this
    argument get byte-identical behavior to before this parameter existed."""
    n = len(arrays["floor_d"])
    opp_party_fixed = np.asarray(opp_party_fixed, dtype=float)
    committed = np.zeros(n) if committed_own is None else np.asarray(committed_own, dtype=float)
    cap = own_cap_fraction * own_budget
    room = np.maximum(cap - committed, 0.0)          # max FLEXIBLE $ this race can still receive
    flexible_budget = max(own_budget - float(committed.sum()), 0.0)   # F_t

    x0_dollars = x0 if x0 is not None else np.zeros(n)
    x0_s = np.clip(x0_dollars, 0.0, room) / SCALE
    room_s = room / SCALE
    fb_s = flexible_budget / SCALE
    neg_ones = -np.ones(n)

    def p_at(flex_dollars: np.ndarray) -> np.ndarray:
        total_own = committed + flex_dollars
        if side == "D":
            return payoff.p_win_shared(total_own, opp_party_fixed, arrays)
        return payoff.p_win_shared(opp_party_fixed, total_own, arrays)

    def grad_at(flex_dollars: np.ndarray) -> np.ndarray:
        total_own = committed + flex_dollars
        if side == "D":
            dp_dxd, _ = payoff.grad_shared(total_own, opp_party_fixed, arrays)
            return dp_dxd
        _, dp_dxr = payoff.grad_shared(opp_party_fixed, total_own, arrays)
        return -dp_dxr  # d(n - sum p)/d(x_R) = -dp/dx_R

    def neg_obj(xs: np.ndarray) -> float:
        flex = xs * SCALE
        p = p_at(flex)
        obj = float(p.sum()) if side == "D" else float(n) - float(p.sum())
        return -obj

    def neg_grad(xs: np.ndarray) -> np.ndarray:
        flex = xs * SCALE
        return -(grad_at(flex) * SCALE)

    def budget_slack(xs: np.ndarray) -> float:
        return float(fb_s - xs.sum())

    result = minimize(
        neg_obj, x0=x0_s, method="SLSQP", jac=neg_grad,
        bounds=[(0.0, room_s[i]) for i in range(n)],
        constraints=[{"type": "ineq", "fun": budget_slack, "jac": lambda xs: neg_ones.copy()}],
        options={"maxiter": 3000, "ftol": 1e-12},
    )
    flex_final = np.maximum(result.x * SCALE, 0.0)
    party = committed + flex_final   # TOTAL spend (committed + flexible) -- matches uncommitted callers' semantics
    status = "optimal" if result.success else f"slsqp:{result.message}"
    p_final = p_at(flex_final)
    e_seats_own = float(p_final.sum()) if side == "D" else float(n) - float(p_final.sum())
    return BestResponseResult(party=party, e_seats_own=e_seats_own, status=status)


def br_d(races, coef, sigma_model, *, party_r: np.ndarray, cand_r_total: np.ndarray,
         budget_d: float, cap_fraction_d: float, x0: np.ndarray | None = None,
         committed_d: np.ndarray | None = None) -> BestResponseResult:
    """D's best response to a FIXED R-side allocation (party_r = x_R,
    dollars above R's own uncontrolled floor). committed_d: see _solve's
    docstring -- D's own already-locked capital, None = fully flexible
    (original behavior)."""
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
    return _solve("D", arrays, party_r, own_budget=budget_d,
                  own_cap_fraction=cap_fraction_d, x0=x0, committed_own=committed_d)


def br_r(races, coef, sigma_model, *, party_d: np.ndarray, cand_r_total: np.ndarray,
         budget_r: float, cap_fraction_r: float, x0: np.ndarray | None = None,
         committed_r: np.ndarray | None = None) -> BestResponseResult:
    """R's best response to a FIXED D-side allocation (party_d = x_D,
    dollars above D's own uncontrolled floor). committed_r: see _solve's
    docstring -- R's own already-locked capital, None = fully flexible
    (original behavior)."""
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
    return _solve("R", arrays, party_d, own_budget=budget_r,
                  own_cap_fraction=cap_fraction_r, x0=x0, committed_own=committed_r)
