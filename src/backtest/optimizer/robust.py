"""
Robust/max-min-over-eta optimization (docs/theta_followup_plan.md Section 6:
"robust optimization (max-min over eta in [eta_min, eta_max])... a
genuinely different, assumption-light paradigm worth prototyping separately
... not implemented here since it changes the optimizer's objective, not
just the calibration inputs").

Design, verified against the actual gradient formula rather than assumed:
allocator._msg_vec()'s eta-adjusted gradient is
    d(log_ratio)/d(D) = 1/d - (1+eta_eff)/t
which is STRICTLY DECREASING in eta wherever the reaction gates active
(party > party_obs, i.e. wherever eta actually multiplies anything). Since
MSG = (phi/sigma) * d(mu_raw)/d(D) * ceiling_grad_factor and phi/sigma and
the ceiling factor don't depend on eta's sign, MSG is monotonically
non-increasing in eta over that same region -- meaning D's expected-seats
objective, evaluated at ANY fixed candidate allocation, is monotonically
non-increasing in eta race-by-race. For a BOX uncertainty set
eta in [eta_low_tier, eta_high_tier] per race, the worst case for D is
therefore always eta = eta_high_tier, for every race simultaneously --
this collapses max_D min_eta E[Seats(D, eta)] to a SINGLE call of
optimize_nonlinear() at the worst-case eta array, not a new minimax solver.

This reduction is verified POST-HOC below (not just asserted in this
docstring) by recomputing the gradient's sign at the solution and confirming
it's non-increasing in eta there -- if that check ever fails (e.g. because
alpha4 someday becomes large enough to flip the sign of
d(mu_raw)/d(eta) = -c_spend/t + alpha4/t, which requires alpha4 > c_spend, not
true anywhere in the live-calibrated model but not permanently guaranteed
either), this module raises rather than silently returning a wrong "robust"
answer.
"""

from __future__ import annotations

import numpy as np

from .allocator import OptimizerResult, _msg_vec, _precompute_race_arrays, optimize_nonlinear


def eta_high_by_race(races: list, eta_uncertainty_by_tier: dict, quantile_key: str = "p95") -> np.ndarray:
    """Per-race worst-case eta, from data/processed/eta_uncertainty.json's
    per-tier bootstrap distribution (built by
    scripts/build_eta_uncertainty_distribution.py) -- written to disk but,
    before this module, never read anywhere in the codebase. Falls back to
    0.0 for a race whose tier has no entry (conservative in the OTHER
    direction -- a missing bound should not silently manufacture a
    worst-case penalty out of nothing)."""
    eta_high = np.zeros(len(races))
    for i, r in enumerate(races):
        tier_entry = eta_uncertainty_by_tier.get(r.cook_rating)
        if tier_entry is None:
            continue
        eta_high[i] = float(tier_entry.get("bootstrap", {}).get(quantile_key, 0.0))
    return np.maximum(eta_high, 0.0)   # eta is a reaction rate; never sensibly negative


def _verify_monotonicity(races, coef, sigma_model, eta_high: np.ndarray,
                          allocation: np.ndarray, tol: float = 1e-9) -> None:
    """Post-hoc check (not a silent assumption): at the solution's own
    allocation, MSG evaluated at eta_high must be <= MSG evaluated at
    eta=0 for every race where the reaction actually gates active
    (party > party_obs) -- confirming eta_high genuinely was the worst
    case for D, not accidentally a better one. Raises RuntimeError rather
    than returning a silently-wrong "robust" result if this ever fails."""
    arrays_0 = _precompute_race_arrays(races, coef, sigma_model, eta=0.0)
    arrays_high = _precompute_race_arrays(races, coef, sigma_model, eta=eta_high)
    floors = arrays_0["floors"]
    party = np.maximum(allocation - floors, 0.0)

    msg_0 = _msg_vec(party, arrays_0)
    msg_high = _msg_vec(party, arrays_high)
    active = party > arrays_0["party_obs"] + tol

    violation = active & (msg_high > msg_0 + tol)
    if np.any(violation):
        bad = np.where(violation)[0][:5]
        raise RuntimeError(
            f"optimize_nonlinear_robust: monotonicity assumption violated for "
            f"{int(violation.sum())} race(s) (first indices: {bad.tolist()}) -- "
            f"MSG at eta_high exceeds MSG at eta=0 where the reaction is active. "
            f"The max_D min_eta -> single-call-at-eta_high reduction this module "
            f"relies on does not hold for this coefficient set; a genuine "
            f"iterative minimax (alternating projected gradient on eta within the "
            f"box, D re-optimizing against it) would be needed instead, which this "
            f"module does NOT implement -- do not trust this result."
        )


def optimize_nonlinear_robust(
    races: list,
    coef,
    sigma_model,
    budget: float,
    cov_matrix: np.ndarray,
    gamma: float,
    cap_fraction: float,
    eta_uncertainty_by_tier: dict,
    party_budget: float | None = None,
    quantile_key: str = "p95",
) -> OptimizerResult:
    """max_D min_{eta in box} E[Seats(D, eta)], reduced (see module
    docstring, verified post-hoc via _verify_monotonicity) to a single
    optimize_nonlinear() call at the per-race worst-case eta
    (eta_high_by_race()). Returns the same OptimizerResult type as
    optimize_nonlinear() -- this is a drop-in, just with a more
    conservative eta input, not a different result type."""
    n = len(races)
    eta_high = eta_high_by_race(races, eta_uncertainty_by_tier, quantile_key=quantile_key)

    result = optimize_nonlinear(
        races, coef, sigma_model, budget=budget, cov_matrix=cov_matrix,
        gamma=gamma, cap_fraction=cap_fraction, party_budget=party_budget, eta=eta_high,
    )

    _verify_monotonicity(races, coef, sigma_model, eta_high, result.allocations)
    return result
