"""
Shared two-player payoff model, p_i(D_i, R_i, X_i) = P(Democrat wins race i)
(project_spec.md Section 4).

Deliberately reuses ONE probability model for both sides rather than two
separately-calibrated formulas: backtest.optimizer.allocator's D-anchored
margin/ceiling construction, which IS empirically calibrated against real
DCCC spending-response data (see backtest/optimizer/nash.py's own module
docstring on why its R-side mirrored ceiling is NOT independently
calibrated). Constant-sum utilities (spec Section 4) follow directly:

    U_D(D, R) = sum_i p_i(D_i, R_i, X_i)
    U_R(D, R) = N - U_D(D, R)

R's spending enters p_i only through each race's r_total ($ total, not a
party-only split) -- callers pass the FULL R-side dollar total per race
(candidate-committee floor + party money), exactly as backtest.types.RaceRecord
already represents it.

eta (the old project's reactive-response scalar) is never set here: in this
game R is an endogenous decision variable, not a mechanical function of D
(spec Section 6's explicit warning against reusing eta this way).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from backtest.optimizer.allocator import _p_win_vec, _precompute_race_arrays


def race_arrays_at(races, coef, sigma_model, total_r: np.ndarray) -> dict:
    """Precompute the D-anchored race arrays at a specific R-side dollar
    total per race. Cheap relative to the SLSQP solves elsewhere in this
    package -- fine to call once per (D, R) evaluation."""
    races_at_r = [dataclasses.replace(r, r_total=float(total_r[i])) for i, r in enumerate(races)]
    return _precompute_race_arrays(races_at_r, coef, sigma_model, eta=0.0)


def p_win(party_d: np.ndarray, races, coef, sigma_model, total_r: np.ndarray) -> np.ndarray:
    """p_i(D_i, R_i, X_i) for every race, given full D-side PARTY dollars
    (floors are added internally, matching RaceRecord.cand_d_total) and full
    R-side TOTAL dollars (floor + party, i.e. race.r_total's own convention)."""
    arrays = race_arrays_at(races, coef, sigma_model, total_r)
    return _p_win_vec(np.maximum(party_d, 0.0), arrays)


def expected_seats_d(p: np.ndarray) -> float:
    """U_D(D, R) = sum_i p_i."""
    return float(np.sum(p))


def expected_seats_r(n_races: int, e_seats_d: float) -> float:
    """U_R(D, R) = N - U_D(D, R) (constant-sum, spec Section 4)."""
    return float(n_races) - e_seats_d
