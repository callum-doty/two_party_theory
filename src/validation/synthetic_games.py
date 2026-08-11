"""
Level C validation (project_spec.md Section 20): "On synthetic games with
known equilibria, does the algorithm recover them?"

Deliberately standalone -- no dependency on backtest's estimated margin
model or RaceRecord type -- so this validates the GENERIC best-response /
Nash-finding algorithm itself (the iteration and convergence logic in
game/equilibrium.py's design), decoupled from whether the real election
model is well-specified. That question belongs to Level A (response model)
and Level D (historical behavior), not here.

Game: n independent "races," each a logistic contest in the SPENDING GAP,
p_i(D_i, R_i) = 1 / (1 + exp(-k_i * (D_i - R_i))), so
    dp_i/dD_i = -dp_i/dR_i = k_i * p_i * (1 - p_i),
maximized exactly at D_i = R_i (p_i = 0.5), independent of k_i. With equal
budgets (B_D = B_R = B) and identical k_i across races, symmetry gives a
closed-form Nash equilibrium: D_i* = R_i* = B/n for every race. Any
one-sided deviation strictly LOWERS that race's own marginal value below
its k_i/4 maximum without raising any other race's marginal value past ITS
own maximum, so no unilateral reallocation can improve either side's
expected seats -- this is a genuine, checkable equilibrium, not just a
plausible guess (see this module's docstring history for the full argument
if it needs re-deriving).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def p_win(total_d: np.ndarray, total_r: np.ndarray, k: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * (total_d - total_r)))


def msg(total_d: np.ndarray, total_r: np.ndarray, k: np.ndarray) -> np.ndarray:
    p = p_win(total_d, total_r, k)
    return k * p * (1.0 - p)


def _best_response(own_fixed_other: np.ndarray, k: np.ndarray, budget: float, cap: float,
                    maximize_own_side: str, x0: np.ndarray | None = None) -> np.ndarray:
    """Best response for one side (own vars are the free variable, the
    other side's totals are fixed), via generic SLSQP -- no closed-form
    shortcut, so this genuinely exercises the same kind of solve
    game/best_response.py performs on the real model."""
    n = len(k)
    sign = 1.0 if maximize_own_side == "D" else -1.0

    def neg_obj(x: np.ndarray) -> float:
        p = p_win(x, own_fixed_other, k) if maximize_own_side == "D" else p_win(own_fixed_other, x, k)
        return -float(np.sum(sign * p if maximize_own_side == "D" else (1.0 - p)))

    def neg_grad(x: np.ndarray) -> np.ndarray:
        g = msg(x, own_fixed_other, k) if maximize_own_side == "D" else msg(own_fixed_other, x, k)
        return -g

    x0 = x0 if x0 is not None else np.full(n, budget / n)
    result = minimize(
        neg_obj, x0=x0, jac=neg_grad, method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints=[{"type": "ineq", "fun": lambda x: budget - x.sum(),
                      "jac": lambda x: -np.ones(n)}],
        options={"maxiter": 500, "ftol": 1e-14},
    )
    return np.maximum(result.x, 0.0)


@dataclass
class SyntheticGameResult:
    party_d: np.ndarray
    party_r: np.ndarray
    converged: bool
    n_iterations: int
    max_error_vs_known_equilibrium: float


def solve_and_check(n_races: int = 5, k: float = 2e-6, budget: float = 1_000_000.0,
                     cap_fraction: float = 1.0, max_rounds: int = 200,
                     tol_dollars: float = 1.0) -> SyntheticGameResult:
    """Runs Gauss-Seidel best-response dynamics (D-first) on the symmetric
    logistic-contest game and checks the result against the known closed-form
    equilibrium D_i* = R_i* = budget/n_races."""
    k_vec = np.full(n_races, k)
    cap = cap_fraction * budget
    known_equilibrium = budget / n_races

    # Deliberately asymmetric starting point (all budget in race 0) so
    # convergence to the symmetric equilibrium is a real check, not a no-op
    # at an already-correct guess.
    party_d = np.zeros(n_races); party_d[0] = min(budget, cap)
    remaining_d = budget - party_d[0]
    if n_races > 1 and remaining_d > 0:
        party_d[1:] = remaining_d / (n_races - 1)
    party_r = np.full(n_races, budget / n_races)
    converged = False
    it = 0
    for it in range(max_rounds):
        new_d = _best_response(party_r, k_vec, budget, cap, "D", x0=party_d)
        new_r = _best_response(new_d, k_vec, budget, cap, "R", x0=party_r)
        delta = max(np.max(np.abs(new_d - party_d)), np.max(np.abs(new_r - party_r)))
        party_d, party_r = new_d, new_r
        if delta < tol_dollars:
            converged = True
            break

    max_error = float(max(
        np.max(np.abs(party_d - known_equilibrium)),
        np.max(np.abs(party_r - known_equilibrium)),
    ))
    return SyntheticGameResult(
        party_d=party_d, party_r=party_r, converged=converged,
        n_iterations=it + 1, max_error_vs_known_equilibrium=max_error,
    )


if __name__ == "__main__":
    res = solve_and_check()
    print(f"Converged: {res.converged} in {res.n_iterations} rounds")
    print(f"Max |allocation - known equilibrium|: ${res.max_error_vs_known_equilibrium:,.2f}")
    print(f"party_d: {res.party_d}")
    print(f"party_r: {res.party_r}")
