"""
Fast water-filling surrogate for BR_D/BR_R under the shared payoff
(payoff.p_win_shared) -- project_spec.md Section 12: "the concave-envelope/
water-filling work from Paper III may provide a fast alternative after
symmetrical validation for both players."

The old surrogate (src/optimizer/concave_surrogate.py) only ever validated
D's side (against a fixed/reactive R, ~2,000-2,700x faster than SLSQP, see
its own module docstring); its R-side mirror (surrogate_allocate_r) was
built on the mirrored-ceiling formula game/best_response.py moved away from
on 2026-08-12, and was explicitly flagged as never validated. Since both
sides now search the SAME shared, symmetric payoff, a single surrogate
serves both here -- br_d_surrogate/br_r_surrogate differ only in which side
is free and how the objective is oriented (D maximizes sum(p), R maximizes
n - sum(p)).

Reuses build_concave_segments/greedy_allocate UNCHANGED from
optimizer.concave_surrogate -- that machinery only needs a payoff_fn(party,
arrays) -> array callable and doesn't care what formula it wraps. The
piecewise-linear concave envelope this constructs is the SAME regularizer
regardless of whether the true per-race curve is globally concave (it need
not be: Phi is S-shaped, so the OWN-side curve can be locally convex before
the ceiling saturates it) -- greedy_allocate finds the exact optimum of the
envelope relaxation, which is an approximation to the true nonlinear
optimum, not identical to it. Do not trust this for anything without
checking tests/test_best_response_surrogate.py's SLSQP-vs-surrogate
agreement first, exactly as the D-side surrogate was checked
(theta_concave_surrogate.py) before it was used anywhere real.
"""

from __future__ import annotations

import numpy as np

from optimizer.concave_surrogate import build_concave_segments, greedy_allocate

from . import payoff
from .best_response import BestResponseResult

N_GRID_DEFAULT = 40


def _payoff_fn(side: str, opp_party_fixed: np.ndarray):
    """Returns a payoff_fn(own_party, arrays) -> array closure for
    build_concave_segments, evaluating payoff.p_win_shared with the
    OTHER side's allocation held fixed."""
    if side == "D":
        def fn(own_party: np.ndarray, arrays: dict) -> np.ndarray:
            return payoff.p_win_shared(own_party, opp_party_fixed, arrays)
        return fn
    if side == "R":
        def fn(own_party: np.ndarray, arrays: dict) -> np.ndarray:
            return 1.0 - payoff.p_win_shared(opp_party_fixed, own_party, arrays)
        return fn
    raise ValueError(f"side must be 'D' or 'R', got {side!r}")


def br_d_surrogate(races, coef, sigma_model, *, party_r: np.ndarray, cand_r_total: np.ndarray,
                    budget_d: float, cap_fraction_d: float,
                    n_grid: int = N_GRID_DEFAULT) -> BestResponseResult:
    """Fast approximate BR_D(R), via the piecewise-linear concave envelope
    + greedy water-filling. See module docstring for validation status."""
    n = len(races)
    party_r = np.asarray(party_r, dtype=float)
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
    cap = cap_fraction_d * budget_d * np.ones(n)
    race_idx, width, slope, xstart = build_concave_segments(
        arrays, cap, n_grid, payoff_fn=_payoff_fn("D", party_r),
    )
    party = greedy_allocate(race_idx, width, slope, xstart, n, budget_d)
    e_seats_own = float(payoff.p_win_shared(party, party_r, arrays).sum())
    return BestResponseResult(party=party, e_seats_own=e_seats_own, status="surrogate")


def br_r_surrogate(races, coef, sigma_model, *, party_d: np.ndarray, cand_r_total: np.ndarray,
                    budget_r: float, cap_fraction_r: float,
                    n_grid: int = N_GRID_DEFAULT) -> BestResponseResult:
    """Fast approximate BR_R(D), via the piecewise-linear concave envelope
    + greedy water-filling. See module docstring for validation status."""
    n = len(races)
    party_d = np.asarray(party_d, dtype=float)
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
    cap = cap_fraction_r * budget_r * np.ones(n)
    race_idx, width, slope, xstart = build_concave_segments(
        arrays, cap, n_grid, payoff_fn=_payoff_fn("R", party_d),
    )
    party = greedy_allocate(race_idx, width, slope, xstart, n, budget_r)
    e_d = float(payoff.p_win_shared(party_d, party, arrays).sum())
    e_seats_own = float(n) - e_d
    return BestResponseResult(party=party, e_seats_own=e_seats_own, status="surrogate")
