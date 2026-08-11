"""
BR_D(R) and BR_R(D) (project_spec.md Section 9): each side's constrained
best response holding the opponent's spending fixed.

Thin, explicitly-named wrapper around backtest.optimizer.nash.best_response
-- that function is the already-validated D/R solver (D's branch reuses
optimize_nonlinear() unmodified; R's branch is the mirrored-ceiling SLSQP
solve, re-scored consistently -- see that module's docstring for the
2026-08-10 scoring fix this project depends on). Nothing here re-derives the
optimization; it only gives the new project's own API surface the spec's
naming (BR_D / BR_R / RegretD / RegretR all trace back to one solver).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backtest.optimizer import nash


@dataclass
class BestResponseResult:
    party: np.ndarray       # this side's own party-money allocation ($)
    e_seats_own: float      # this side's own expected seats at that allocation
    status: str


def br_d(races, coef, sigma_model, *, total_r: np.ndarray, budget_d: float,
         cap_fraction_d: float, x0: np.ndarray | None = None) -> BestResponseResult:
    """D's best response to a fixed R-side total allocation."""
    res = nash.best_response(
        "D", races, coef, sigma_model, opp_total_fixed=total_r,
        own_party_budget=budget_d, own_cap_fraction=cap_fraction_d, x0=x0,
    )
    return BestResponseResult(party=res.party, e_seats_own=res.e_seats_own, status=res.status)


def br_r(races, coef, sigma_model, *, total_d: np.ndarray, cand_r_total: np.ndarray,
         budget_r: float, cap_fraction_r: float, x0: np.ndarray | None = None) -> BestResponseResult:
    """R's best response to a fixed D-side total allocation."""
    res = nash.best_response(
        "R", races, coef, sigma_model, opp_total_fixed=total_d,
        own_party_budget=budget_r, own_cap_fraction=cap_fraction_r,
        cand_r_total=cand_r_total, x0=x0,
    )
    return BestResponseResult(party=res.party, e_seats_own=res.e_seats_own, status=res.status)
