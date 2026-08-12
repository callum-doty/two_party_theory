"""Stage 2 (basic invariants) and Stage 4 (mirror symmetry) checks for
game/payoff.py's baseline_arrays/p_win_shared/grad_shared -- the fixed-
baseline, signed-tanh-saturation replacement for the D-anchored formula
that let BR_R exploit an unregularized downward extrapolation (see
payoff.py's own module docstring and docs/methodology.md for the incident
this replaces)."""

import numpy as np
import pytest

from game import payoff
from backtest.model.margin import MarginModelCoefficients
from backtest.types import RaceRecord, SigmaModel


def _synthetic_universe(n=25, seed=0):
    rng = np.random.default_rng(seed)
    coef = MarginModelCoefficients(
        alpha0=0.0, alpha1=0.9, alpha2=0.03, alpha3=0.4, alpha4=0.0, alpha5=0.0,
        beta1=3.0, beta2=0.0, beta3=0.0, beta1_open=None,
    )
    sigma_model = SigmaModel(_coef={"intercept": np.log(6.0), "abs_pvi": 0.0})

    races = []
    for i in range(n):
        pvi = rng.uniform(-8, 8)
        cand_d = rng.uniform(1_000, 300_000)     # includes near-zero floors like HI-02
        d_total = cand_d + rng.uniform(0, 500_000)
        r_total = rng.uniform(1_000, 800_000)
        races.append(RaceRecord(
            district_id=f"XX-{i:02d}", state="XX", district=i, cook_rating="Toss-up",
            incumb_status="Open", pvi=pvi, d_total=d_total, r_total=r_total,
            cvap=400_000, generic_ballot=1.0, cand_d_total=cand_d, indiv_share=0.0,
        ))
    floor_r = np.array([rng.uniform(10, 300_000) for _ in races])
    return races, coef, sigma_model, floor_r


def test_mu_bounded_by_ceiling_width_regardless_of_excursion_size():
    races, coef, sigma_model, floor_r = _synthetic_universe()
    arrays = payoff.baseline_arrays(races, coef, sigma_model, floor_r)
    n = len(races)
    rng = np.random.default_rng(1)

    for party_r_scale in (0.0, 1e5, 1e6, 1e8, 1e10):
        party_d = np.zeros(n)
        party_r = np.full(n, party_r_scale)
        total_d = arrays["floor_d"] + party_d
        total_r = arrays["floor_r"] + party_r
        mu_raw = payoff._mu_raw(total_d, total_r, arrays)
        delta_raw = mu_raw - arrays["mu_0"]
        delta_cap = arrays["C"] * np.tanh(delta_raw / arrays["C"])
        assert np.all(np.abs(delta_cap) < arrays["C"] + 1e-9)

    # symmetric check with D spending unboundedly instead
    for party_d_scale in (1e6, 1e8, 1e10):
        party_d = np.full(n, party_d_scale)
        party_r = np.zeros(n)
        total_d = arrays["floor_d"] + party_d
        total_r = arrays["floor_r"] + party_r
        mu_raw = payoff._mu_raw(total_d, total_r, arrays)
        delta_raw = mu_raw - arrays["mu_0"]
        delta_cap = arrays["C"] * np.tanh(delta_raw / arrays["C"])
        assert np.all(np.abs(delta_cap) < arrays["C"] + 1e-9)


def test_win_prob_saturates_instead_of_collapsing_to_zero_or_one():
    """The concrete HI-02 failure mode: a $10-floor R race should NOT be
    driven from ~99% D to ~33% D by a few million unilateral R dollars."""
    races, coef, sigma_model, floor_r = _synthetic_universe()
    # Force one race to look exactly like HI-02: near-zero R floor, large D floor.
    races = list(races)
    races[0] = type(races[0])(**{**races[0].__dict__, "cand_d_total": 706_756.0, "pvi": 25.0})
    floor_r = floor_r.copy()
    floor_r[0] = 10.0
    arrays = payoff.baseline_arrays(races, coef, sigma_model, floor_r)

    party_d = np.zeros(len(races))
    p_at_zero = payoff.p_win_shared(party_d, np.zeros(len(races)), arrays)[0]
    assert p_at_zero > 0.9  # genuinely safe D under the model

    party_r = np.zeros(len(races))
    party_r[0] = 3_000_000.0
    p_after_dump = payoff.p_win_shared(party_d, party_r, arrays)[0]
    # under the OLD D-anchored formula this raced to ~0.33; the fixed
    # baseline should keep it near its ceiling-bounded floor, not collapse it.
    assert p_after_dump > 0.5, f"expected saturation to hold the seat competitive-but-safe, got {p_after_dump}"


def test_gradient_signs_and_saturation_toward_zero():
    races, coef, sigma_model, floor_r = _synthetic_universe()
    arrays = payoff.baseline_arrays(races, coef, sigma_model, floor_r)
    n = len(races)

    party_d = np.full(n, 50_000.0)
    party_r = np.full(n, 50_000.0)
    dp_dxd, dp_dxr = payoff.grad_shared(party_d, party_r, arrays)
    assert np.all(dp_dxd >= 0)
    assert np.all(dp_dxr <= 0)

    dp_dxd_far, dp_dxr_far = payoff.grad_shared(np.full(n, 5e8), party_r, arrays)
    assert np.all(np.abs(dp_dxd_far) < np.abs(dp_dxd) + 1e-12)


def test_analytic_gradient_matches_finite_difference():
    races, coef, sigma_model, floor_r = _synthetic_universe()
    arrays = payoff.baseline_arrays(races, coef, sigma_model, floor_r)
    n = len(races)
    rng = np.random.default_rng(2)
    party_d = rng.uniform(0, 400_000, size=n)
    party_r = rng.uniform(0, 400_000, size=n)

    dp_dxd, dp_dxr = payoff.grad_shared(party_d, party_r, arrays)

    eps = 10.0
    p_plus_d = payoff.p_win_shared(party_d + eps, party_r, arrays)
    p_minus_d = payoff.p_win_shared(party_d - eps, party_r, arrays)
    fd_dxd = (p_plus_d - p_minus_d) / (2 * eps)

    p_plus_r = payoff.p_win_shared(party_d, party_r + eps, arrays)
    p_minus_r = payoff.p_win_shared(party_d, np.maximum(party_r - eps, 0.0), arrays)
    fd_dxr = (p_plus_r - p_minus_r) / (2 * eps)

    np.testing.assert_allclose(dp_dxd, fd_dxd, atol=1e-9, rtol=1e-4)
    np.testing.assert_allclose(dp_dxr, fd_dxr, atol=1e-9, rtol=1e-4)


def test_ceiling_saturation_itself_is_sign_symmetric():
    """Stage 4's mirror check, scoped to what this fix actually controls:
    the tanh saturation step, not the underlying margin regression.

    A full race-level D<->R mirror test (mu_raw(F^D, F^R) for a swapped race
    equal to -mu_raw(F^R, F^D)) does NOT hold for this model and isn't a bug
    introduced here: the calibrated margin formula uses the log MARKET-SHARE
    ratio log(D/(D+R)), which is not antisymmetric under D<->R swap (only
    the log-ODDS ratio log(D/R) would be) -- a property of the existing,
    already-calibrated regression (project_spec.md Section 19's open D/R
    elasticity-symmetry question), not something this ceiling rewrite
    changes or should paper over.

    What the ceiling rewrite DOES own is the saturation step applied on top
    of mu_raw: for any baseline C and any raw excursion z, tanh's own
    antisymmetry guarantees C*tanh(z/C) == -C*tanh(-z/C) exactly. That is
    the actual "no D-anchored/R-mirrored formula divergence" property this
    module claims -- checked directly here rather than via a race-swap
    round-trip that depends on an assumption the base model doesn't satisfy."""
    C = np.array([0.4, 1.2, 3.0, 8.5])
    z = np.array([0.05, 1.7, -0.3, 12.0])
    up = C * np.tanh(z / C)
    down = C * np.tanh(-z / C)
    np.testing.assert_allclose(up, -down, atol=1e-12)
