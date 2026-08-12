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
           x0: np.ndarray | None) -> BestResponseResult:
    """One side's constrained best response under payoff.p_win_shared,
    holding the OTHER side's party allocation fixed at opp_party_fixed.
    side='D' maximizes sum(p); side='R' maximizes n - sum(p). Gradients are
    exact (payoff.grad_shared), not finite-differenced -- the shared
    formula's tanh saturation is smooth everywhere, so there's no ceiling
    kink to worry about the way the old exp-based one-sided ceiling had."""
    n = len(arrays["floor_d"])
    opp_party_fixed = np.asarray(opp_party_fixed, dtype=float)
    cap = own_cap_fraction * own_budget
    x0_dollars = x0 if x0 is not None else np.zeros(n)
    x0_s = np.clip(x0_dollars, 0.0, cap) / SCALE
    cap_s = cap / SCALE
    pb_s = own_budget / SCALE
    neg_ones = -np.ones(n)

    def p_at(own_party_dollars: np.ndarray) -> np.ndarray:
        if side == "D":
            return payoff.p_win_shared(own_party_dollars, opp_party_fixed, arrays)
        return payoff.p_win_shared(opp_party_fixed, own_party_dollars, arrays)

    def grad_at(own_party_dollars: np.ndarray) -> np.ndarray:
        if side == "D":
            dp_dxd, _ = payoff.grad_shared(own_party_dollars, opp_party_fixed, arrays)
            return dp_dxd
        _, dp_dxr = payoff.grad_shared(opp_party_fixed, own_party_dollars, arrays)
        return -dp_dxr  # d(n - sum p)/d(x_R) = -dp/dx_R

    def neg_obj(xs: np.ndarray) -> float:
        own = xs * SCALE
        p = p_at(own)
        obj = float(p.sum()) if side == "D" else float(n) - float(p.sum())
        return -obj

    def neg_grad(xs: np.ndarray) -> np.ndarray:
        own = xs * SCALE
        return -(grad_at(own) * SCALE)

    def budget_slack(xs: np.ndarray) -> float:
        return float(pb_s - xs.sum())

    result = minimize(
        neg_obj, x0=x0_s, method="SLSQP", jac=neg_grad,
        bounds=[(0.0, cap_s)] * n,
        constraints=[{"type": "ineq", "fun": budget_slack, "jac": lambda xs: neg_ones.copy()}],
        options={"maxiter": 3000, "ftol": 1e-12},
    )
    party = np.maximum(result.x * SCALE, 0.0)
    status = "optimal" if result.success else f"slsqp:{result.message}"
    p_final = p_at(party)
    e_seats_own = float(p_final.sum()) if side == "D" else float(n) - float(p_final.sum())
    return BestResponseResult(party=party, e_seats_own=e_seats_own, status=status)


def br_d(races, coef, sigma_model, *, party_r: np.ndarray, cand_r_total: np.ndarray,
         budget_d: float, cap_fraction_d: float, x0: np.ndarray | None = None) -> BestResponseResult:
    """D's best response to a FIXED R-side allocation (party_r = x_R,
    dollars above R's own uncontrolled floor)."""
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
    return _solve("D", arrays, party_r, own_budget=budget_d,
                  own_cap_fraction=cap_fraction_d, x0=x0)


def br_r(races, coef, sigma_model, *, party_d: np.ndarray, cand_r_total: np.ndarray,
         budget_r: float, cap_fraction_r: float, x0: np.ndarray | None = None) -> BestResponseResult:
    """R's best response to a FIXED D-side allocation (party_d = x_D,
    dollars above D's own uncontrolled floor)."""
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
    return _solve("R", arrays, party_d, own_budget=budget_r,
                  own_cap_fraction=cap_fraction_r, x0=x0)
