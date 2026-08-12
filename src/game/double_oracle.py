"""
Double-oracle mixed-strategy equilibrium solver (docs' "Revised order of
work" #3, project_spec.md Sections 11-12's open question of whether the
observed damped-Gauss-Seidel cycling reflects genuine non-existence of a
low-regret pure point -- see minimize_pure_exploitability.py -- or a
mixed-strategy equilibrium instead).

Treats a FULL 433-race allocation vector as one pure strategy ("portfolio").
Builds a payoff matrix over a small, growing pool of discovered D and R
portfolios, solves the finite zero-sum matrix game exactly via LP, then
computes each side's exact best response to the OTHER side's mixture and
adds it to the pool if it's a real improvement. Classic double-oracle
(McMahan, Gordon & Blum 2003) structure: alternately grow the strategy sets
and re-solve, rather than fixing a strategy space up front.

Because U_R = n - U_D (constant-sum in expected seats), this is a genuine
zero-sum game once expressed as a payoff matrix over portfolios -- the
finite-game minimax theorem applies EXACTLY to the matrix game at each
double-oracle round, even though best-responding to a MIXTURE over
opponent portfolios is not simply "best-respond to the mixture's average
allocation" (payoff.p_win_shared is not linear in the opponent's spending,
so E_{R~q}[U_D(D,R)] != U_D(D, E_{R~q}[R]) in general -- br_d_to_mixture
below optimizes the true expectation, not the average-opponent shortcut
game/equilibrium.py's fictitious_play() uses as a cheaper diagnostic).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize

from . import payoff

SCALE = 1_000_000.0


def load_solved(results_dir: Path, cycle: int) -> dict | None:
    """Reloads a completed double_oracle() run's pools + mixture from disk
    (scripts/double_oracle.py's saved output) -- shared by every downstream
    consumer of a solved equilibrium (scripts/level_d_benchmark.py,
    scripts/equilibrium_support_composition.py) so they agree on which file
    wins when both a primary and a resumed run exist for the same cycle.
    Prefers `double_oracle_{cycle}_resumed.json` (a continued run from a
    grown pool, e.g. 2022's non-convergent 25-round run resumed for 27 more)
    over `double_oracle_{cycle}.json` if both are present. Returns None if
    neither exists.

    Returns a dict with the raw pools/mixtures (d_pool, r_pool, p, q) as
    well as the LP value and each portfolio's mixture weight -- NOT just
    the mixture's expected/average portfolio, since p_win_shared is
    nonlinear in the opponent and downstream per-race variance analysis
    needs the individual portfolios and their weights, not just their
    weighted mean."""
    results_dir = Path(results_dir)
    resumed = results_dir / f"double_oracle_{cycle}_resumed.json"
    primary = results_dir / f"double_oracle_{cycle}.json"
    path = resumed if resumed.exists() else primary
    if not path.exists():
        return None
    with open(path) as f:
        meta = json.load(f)
    n_d = meta["final_d_pool_size"]
    n_r = meta["final_r_pool_size"]
    d_pool = [np.load(results_dir / f"double_oracle_d_portfolio_{i}_{cycle}.npy") for i in range(n_d)]
    r_pool = [np.load(results_dir / f"double_oracle_r_portfolio_{i}_{cycle}.npy") for i in range(n_r)]
    p = np.zeros(n_d)
    for s in meta["d_support"]:
        p[s["index"]] = s["weight"]
    q = np.zeros(n_r)
    for s in meta["r_support"]:
        q[s["index"]] = s["weight"]
    return {
        "d_pool": d_pool, "r_pool": r_pool, "p": p, "q": q,
        "value_e_seats_d": meta["value_e_seats_d"],
        "converged": meta.get("converged", True), "source_file": path.name,
    }


def payoff_matrix(d_pool: list[np.ndarray], r_pool: list[np.ndarray], arrays: dict) -> np.ndarray:
    """A[j, k] = E[D seats] when D plays portfolio j and R plays portfolio
    k -- pure forward evaluation of payoff.p_win_shared, no optimization,
    so this is cheap even for large pools."""
    j, k = len(d_pool), len(r_pool)
    a = np.empty((j, k))
    for jj, dp in enumerate(d_pool):
        for kk, rp in enumerate(r_pool):
            a[jj, kk] = float(payoff.p_win_shared(dp, rp, arrays).sum())
    return a


def solve_zero_sum_matrix_game(a: np.ndarray) -> dict:
    """Exact LP solve of max_p min_q p^T A q (row player D maximizes
    E[D seats]; column player R implicitly minimizes it, i.e. maximizes
    E[R seats] = n - E[D seats]). Returns both players' optimal mixtures
    and each LP's own value -- the two values should agree up to LP solver
    tolerance (von Neumann minimax theorem for finite zero-sum games); a
    persistent gap signals numerical trouble, not a modeling issue."""
    j, k = a.shape

    # Row player (D): max v s.t. A^T p >= v * 1, sum(p) = 1, p >= 0.
    c = np.zeros(j + 1)
    c[-1] = -1.0  # maximize v == minimize -v
    a_ub = np.hstack([-a.T, np.ones((k, 1))])  # v - sum_j p_j A[j,k] <= 0
    b_ub = np.zeros(k)
    a_eq = np.hstack([np.ones((1, j)), [[0.0]]])
    b_eq = [1.0]
    bounds = [(0.0, None)] * j + [(None, None)]
    res_p = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res_p.success:
        raise RuntimeError(f"Row-player LP failed: {res_p.message}")
    p = np.clip(res_p.x[:j], 0.0, None)
    p = p / p.sum()
    v_row = float(res_p.x[-1])

    # Column player (R, minimizing D's payoff): min v s.t. A q <= v * 1, sum(q) = 1, q >= 0.
    c2 = np.zeros(k + 1)
    c2[-1] = 1.0
    a_ub2 = np.hstack([a, -np.ones((j, 1))])  # sum_k A[j,k] q_k - v <= 0
    b_ub2 = np.zeros(j)
    a_eq2 = np.hstack([np.ones((1, k)), [[0.0]]])
    b_eq2 = [1.0]
    bounds2 = [(0.0, None)] * k + [(None, None)]
    res_q = linprog(c2, A_ub=a_ub2, b_ub=b_ub2, A_eq=a_eq2, b_eq=b_eq2, bounds=bounds2, method="highs")
    if not res_q.success:
        raise RuntimeError(f"Column-player LP failed: {res_q.message}")
    q = np.clip(res_q.x[:k], 0.0, None)
    q = q / q.sum()
    v_col = float(res_q.x[-1])

    return {"p": p, "q": q, "value_row": v_row, "value_col": v_col,
            "lp_gap": abs(v_row - v_col)}


def br_d_to_mixture(races, coef, sigma_model, cand_r_total, *, r_pool: list[np.ndarray],
                     q: np.ndarray, budget_d: float, cap_fraction_d: float,
                     x0: np.ndarray | None = None) -> dict:
    """D's exact best response to R's MIXTURE q over r_pool: maximizes
    E_{R~q}[sum_i p_i(x_D_i, R_i)] = sum_k q_k * sum_i p_i(x_D_i, R_k_i).
    Race-separable (each race's term depends only on that race's x_D_i), so
    this is one SLSQP solve over the same n-dim simplex-with-caps budget
    constraint as best_response.py's br_d, just with a K-term mixture
    objective instead of one fixed opponent."""
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
    n = len(races)
    active = [(w, rp) for w, rp in zip(q, r_pool) if w > 1e-12]
    cap = cap_fraction_d * budget_d
    x0_dollars = x0 if x0 is not None else np.zeros(n)
    x0_s = np.clip(x0_dollars, 0.0, cap) / SCALE
    cap_s = cap / SCALE
    pb_s = budget_d / SCALE
    neg_ones = -np.ones(n)

    def neg_obj(xs: np.ndarray) -> float:
        own = xs * SCALE
        total = 0.0
        for w, rp in active:
            total += w * float(payoff.p_win_shared(own, rp, arrays).sum())
        return -total

    def neg_grad(xs: np.ndarray) -> np.ndarray:
        own = xs * SCALE
        g = np.zeros(n)
        for w, rp in active:
            dp_dxd, _ = payoff.grad_shared(own, rp, arrays)
            g += w * dp_dxd
        return -(g * SCALE)

    def budget_slack(xs: np.ndarray) -> float:
        return float(pb_s - xs.sum())

    result = minimize(
        neg_obj, x0=x0_s, method="SLSQP", jac=neg_grad,
        bounds=[(0.0, cap_s)] * n,
        constraints=[{"type": "ineq", "fun": budget_slack, "jac": lambda xs: neg_ones.copy()}],
        options={"maxiter": 3000, "ftol": 1e-12},
    )
    party = np.maximum(result.x * SCALE, 0.0)
    value = -neg_obj(result.x)
    return {"party": party, "value_vs_mixture": value, "status": "optimal" if result.success else result.message}


def br_r_to_mixture(races, coef, sigma_model, cand_r_total, *, d_pool: list[np.ndarray],
                     p: np.ndarray, budget_r: float, cap_fraction_r: float,
                     x0: np.ndarray | None = None) -> dict:
    """R's exact best response to D's mixture p over d_pool: maximizes
    E_{D~p}[n - sum_i p_i(D_i, x_R_i)] = n - sum_j p_j * sum_i p_i(D_j_i, x_R_i)."""
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
    n = len(races)
    active = [(w, dp) for w, dp in zip(p, d_pool) if w > 1e-12]
    cap = cap_fraction_r * budget_r
    x0_dollars = x0 if x0 is not None else np.zeros(n)
    x0_s = np.clip(x0_dollars, 0.0, cap) / SCALE
    cap_s = cap / SCALE
    pb_s = budget_r / SCALE
    neg_ones = -np.ones(n)

    def neg_obj(xs: np.ndarray) -> float:
        # maximize n - e_d_mix(x_R) == minimize e_d_mix(x_R) - n
        own = xs * SCALE
        e_d_mix = 0.0
        for w, dp in active:
            e_d_mix += w * float(payoff.p_win_shared(dp, own, arrays).sum())
        return e_d_mix - float(n)

    def neg_grad(xs: np.ndarray) -> np.ndarray:
        # d(e_d_mix)/dx_R = sum_j w_j * dp_i/dx_R_i(D_j, x_R) -- same sign as neg_obj's derivative
        own = xs * SCALE
        g = np.zeros(n)
        for w, dp in active:
            _, dp_dxr = payoff.grad_shared(dp, own, arrays)
            g += w * dp_dxr
        return g * SCALE

    def budget_slack(xs: np.ndarray) -> float:
        return float(pb_s - xs.sum())

    result = minimize(
        neg_obj, x0=x0_s, method="SLSQP", jac=neg_grad,
        bounds=[(0.0, cap_s)] * n,
        constraints=[{"type": "ineq", "fun": budget_slack, "jac": lambda xs: neg_ones.copy()}],
        options={"maxiter": 3000, "ftol": 1e-12},
    )
    party = np.maximum(result.x * SCALE, 0.0)
    value = -neg_obj(result.x)  # n - e_d_mix(x_R*): R's own expected-seats value
    return {"party": party, "value_vs_mixture": value, "status": "optimal" if result.success else result.message}


@dataclass
class DoubleOracleResult:
    d_pool: list = field(default_factory=list)
    r_pool: list = field(default_factory=list)
    p: np.ndarray = None
    q: np.ndarray = None
    value: float = 0.0
    d_support: list = field(default_factory=list)   # indices into d_pool with p > tol
    r_support: list = field(default_factory=list)
    gain_history: list = field(default_factory=list)  # (round, d_gain, r_gain)
    converged: bool = False
    n_rounds: int = 0


def double_oracle(races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                   cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15,
                   init_d_pool: list[np.ndarray] | None = None,
                   init_r_pool: list[np.ndarray] | None = None,
                   max_rounds: int = 20, eps: float = 0.02,
                   support_tol: float = 1e-4) -> DoubleOracleResult:
    """Alternately: (1) solve the current matrix game exactly, (2) compute
    each side's exact best response to the opponent's mixture, (3) add any
    response that improves on the current value by more than `eps` seats.
    Stops when neither side can improve by more than eps, or at
    max_rounds -- whichever first (spec's own eps-equilibrium criterion)."""
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
    d_pool = [d.copy() for d in init_d_pool] if init_d_pool else []
    r_pool = [r.copy() for r in init_r_pool] if init_r_pool else []
    if not d_pool or not r_pool:
        raise ValueError("double_oracle needs at least one seed portfolio per side")

    gain_history = []
    solved = None
    for rnd in range(max_rounds):
        a = payoff_matrix(d_pool, r_pool, arrays)
        solved = solve_zero_sum_matrix_game(a)
        p, q, v = solved["p"], solved["q"], solved["value_row"]

        d_x0 = d_pool[int(np.argmax(p))]
        r_x0 = r_pool[int(np.argmax(q))]
        d_br = br_d_to_mixture(races, coef, sigma_model, cand_r_total, r_pool=r_pool, q=q,
                                budget_d=budget_d, cap_fraction_d=cap_fraction_d, x0=d_x0)
        r_br = br_r_to_mixture(races, coef, sigma_model, cand_r_total, d_pool=d_pool, p=p,
                                budget_r=budget_r, cap_fraction_r=cap_fraction_r, x0=r_x0)

        d_gain = d_br["value_vs_mixture"] - v
        r_gain = r_br["value_vs_mixture"] - (float(len(races)) - v)  # v is in D's-payoff units
        gain_history.append({"round": rnd, "d_gain": d_gain, "r_gain": r_gain,
                              "value": v, "lp_gap": solved["lp_gap"],
                              "d_pool_size": len(d_pool), "r_pool_size": len(r_pool)})

        improved = False
        if d_gain > eps:
            d_pool.append(d_br["party"])
            improved = True
        if r_gain > eps:
            r_pool.append(r_br["party"])
            improved = True
        if not improved:
            a_final = payoff_matrix(d_pool, r_pool, arrays)
            solved = solve_zero_sum_matrix_game(a_final)
            return DoubleOracleResult(
                d_pool=d_pool, r_pool=r_pool, p=solved["p"], q=solved["q"],
                value=solved["value_row"],
                d_support=[i for i, w in enumerate(solved["p"]) if w > support_tol],
                r_support=[i for i, w in enumerate(solved["q"]) if w > support_tol],
                gain_history=gain_history, converged=True, n_rounds=rnd + 1,
            )

    a_final = payoff_matrix(d_pool, r_pool, arrays)
    solved = solve_zero_sum_matrix_game(a_final)
    return DoubleOracleResult(
        d_pool=d_pool, r_pool=r_pool, p=solved["p"], q=solved["q"],
        value=solved["value_row"],
        d_support=[i for i, w in enumerate(solved["p"]) if w > support_tol],
        r_support=[i for i, w in enumerate(solved["q"]) if w > support_tol],
        gain_history=gain_history, converged=False, n_rounds=max_rounds,
    )
