"""Checks for game/best_response.py's br_d/br_r against the shared, fixed-
baseline payoff (game/payoff.py): both sides search AND are scored by the
SAME payoff.p_win_shared, so BR_R can no longer exploit the unregularized
downward extrapolation the old D-anchored formula allowed (see payoff.py's
module docstring for the incident -- this file's
test_br_r_does_not_reproduce_the_hi02_extrapolation_bug is a permanent
regression test for it)."""

import numpy as np
import pytest

from game import best_response as br
from game import payoff
from backtest.model.margin import MarginModelCoefficients
from backtest.types import RaceRecord, SigmaModel


def _synthetic_universe(n=12, seed=0):
    rng = np.random.default_rng(seed)
    coef = MarginModelCoefficients(
        alpha0=0.0, alpha1=0.9, alpha2=0.03, alpha3=0.4, alpha4=0.0, alpha5=0.0,
        beta1=3.0, beta2=0.0, beta3=0.0, beta1_open=None,
    )
    sigma_model = SigmaModel(_coef={"intercept": np.log(6.0), "abs_pvi": 0.0})

    races = []
    for i in range(n):
        pvi = rng.uniform(-8, 8)
        cand_d = rng.uniform(20_000, 300_000)
        d_total = cand_d + rng.uniform(0, 500_000)
        r_total = rng.uniform(20_000, 300_000) + rng.uniform(0, 500_000)
        races.append(RaceRecord(
            district_id=f"XX-{i:02d}", state="XX", district=i, cook_rating="Toss-up",
            incumb_status="Open", pvi=pvi, d_total=d_total, r_total=r_total,
            cvap=400_000, generic_ballot=1.0, cand_d_total=cand_d, indiv_share=0.0,
        ))
    cand_r_total = np.array([rng.uniform(20_000, 300_000) for _ in races])
    return races, coef, sigma_model, cand_r_total


def test_br_r_is_a_local_optimum_of_shared_utility():
    races, coef, sigma_model, cand_r_total = _synthetic_universe()
    n = len(races)
    total_d = np.array([r.d_total for r in races])
    party_d = np.maximum(total_d - np.array([r.cand_d_total for r in races]), 0.0)
    budget_r, cap_fraction_r = 2_000_000.0, 0.5

    res = br.br_r(
        races, coef, sigma_model, party_d=party_d, cand_r_total=cand_r_total,
        budget_r=budget_r, cap_fraction_r=cap_fraction_r,
    )
    assert res.status == "optimal"

    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)

    def u_r(party_r: np.ndarray) -> float:
        p_d = payoff.p_win_shared(party_d, party_r, arrays)
        return float(n) - float(p_d.sum())

    u_at_zero = u_r(np.zeros(n))
    u_at_star = u_r(res.party)
    assert u_at_star >= u_at_zero - 1e-6
    assert res.e_seats_own == pytest.approx(u_at_star, abs=1e-6)

    rng = np.random.default_rng(2)
    cap = cap_fraction_r * budget_r
    for _ in range(20):
        i, j = rng.choice(n, size=2, replace=False)
        move = min(10_000.0, res.party[i], cap - res.party[j])
        if move <= 0:
            continue
        perturbed = res.party.copy()
        perturbed[i] -= move
        perturbed[j] += move
        assert u_r(perturbed) <= u_at_star + 1e-4


def test_br_d_is_a_local_optimum_of_shared_utility():
    races, coef, sigma_model, cand_r_total = _synthetic_universe()
    n = len(races)
    total_r = np.array([r.r_total for r in races])
    party_r = np.maximum(total_r - cand_r_total, 0.0)
    budget_d, cap_fraction_d = 2_000_000.0, 0.5

    res = br.br_d(
        races, coef, sigma_model, party_r=party_r, cand_r_total=cand_r_total,
        budget_d=budget_d, cap_fraction_d=cap_fraction_d,
    )
    assert res.status == "optimal"
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)

    def u_d(party_d: np.ndarray) -> float:
        return float(payoff.p_win_shared(party_d, party_r, arrays).sum())

    assert res.e_seats_own == pytest.approx(u_d(res.party), abs=1e-6)

    rng = np.random.default_rng(3)
    cap = cap_fraction_d * budget_d
    for _ in range(20):
        i, j = rng.choice(n, size=2, replace=False)
        move = min(10_000.0, res.party[i], cap - res.party[j])
        if move <= 0:
            continue
        perturbed = res.party.copy()
        perturbed[i] -= move
        perturbed[j] += move
        assert u_d(perturbed) <= res.e_seats_own + 1e-4


def test_br_r_does_not_reproduce_the_hi02_extrapolation_bug():
    """Regression test for the incident in payoff.py's module docstring:
    a near-zero-R-floor, safe-D race must NOT be driven from ~99% D to
    ~30-40% D by BR_R dumping a large unilateral sum into it."""
    races, coef, sigma_model, cand_r_total = _synthetic_universe(n=8)
    races = list(races)
    races[0] = RaceRecord(
        district_id="HI-02-like", state="XX", district=0, cook_rating="Safe D",
        incumb_status="Open", pvi=25.0, d_total=706_756.0, r_total=10.0,
        cvap=400_000, generic_ballot=1.0, cand_d_total=706_756.0, indiv_share=0.0,
    )
    cand_r_total = cand_r_total.copy()
    cand_r_total[0] = 10.0

    total_d = np.array([r.d_total for r in races])
    party_d = np.maximum(total_d - np.array([r.cand_d_total for r in races]), 0.0)

    res = br.br_r(
        races, coef, sigma_model, party_d=party_d, cand_r_total=cand_r_total,
        budget_r=20_000_000.0, cap_fraction_r=0.5,
    )
    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
    p_after = payoff.p_win_shared(party_d, res.party, arrays)[0]
    assert p_after > 0.5, f"HI-02-like race collapsed to {p_after:.3f} -- extrapolation bug reproduced"


def test_committed_none_matches_committed_zeros():
    """committed_r=None (the default, every pre-existing caller) must be
    byte-identical to explicitly passing an all-zero commitment array --
    the locked-capital extension must not change behavior for callers that
    never opt into it."""
    races, coef, sigma_model, cand_r_total = _synthetic_universe()
    n = len(races)
    total_d = np.array([r.d_total for r in races])
    party_d = np.maximum(total_d - np.array([r.cand_d_total for r in races]), 0.0)
    budget_r, cap_fraction_r = 2_000_000.0, 0.5

    res_none = br.br_r(races, coef, sigma_model, party_d=party_d, cand_r_total=cand_r_total,
                        budget_r=budget_r, cap_fraction_r=cap_fraction_r)
    res_zero = br.br_r(races, coef, sigma_model, party_d=party_d, cand_r_total=cand_r_total,
                        budget_r=budget_r, cap_fraction_r=cap_fraction_r, committed_r=np.zeros(n))
    np.testing.assert_allclose(res_none.party, res_zero.party, atol=1.0)
    assert res_none.e_seats_own == pytest.approx(res_zero.e_seats_own, abs=1e-9)


def test_committed_capital_is_a_floor_and_respects_flexible_budget():
    """Locked capital must (1) never be reduced below by the solve -- it is
    money already spent, not a soft preference -- and (2) the FLEXIBLE
    portion of the result must respect budget_r - sum(committed_r), not
    the full budget_r."""
    races, coef, sigma_model, cand_r_total = _synthetic_universe()
    n = len(races)
    total_d = np.array([r.d_total for r in races])
    party_d = np.maximum(total_d - np.array([r.cand_d_total for r in races]), 0.0)
    budget_r, cap_fraction_r = 2_000_000.0, 0.5

    rng = np.random.default_rng(7)
    committed_r = rng.uniform(0, 50_000, size=n)

    res = br.br_r(races, coef, sigma_model, party_d=party_d, cand_r_total=cand_r_total,
                   budget_r=budget_r, cap_fraction_r=cap_fraction_r, committed_r=committed_r)

    assert np.all(res.party >= committed_r - 1e-6), "locked capital was reduced -- it must be a hard floor"
    flexible_spent = float((res.party - committed_r).sum())
    flexible_budget = budget_r - float(committed_r.sum())
    assert flexible_spent <= flexible_budget + 1.0, "flexible spend exceeded budget_r - sum(committed_r)"


def test_fully_committed_leaves_no_room_to_reoptimize():
    """With committed_r pinned at the observed allocation and budget_r set
    so flexible_budget is ~0, the solver can't move money at all: the
    result should equal the committed allocation, and its objective should
    match evaluating the payoff directly at that allocation (not the
    higher value a free solve would find)."""
    races, coef, sigma_model, cand_r_total = _synthetic_universe()
    total_d = np.array([r.d_total for r in races])
    party_d = np.maximum(total_d - np.array([r.cand_d_total for r in races]), 0.0)
    total_r = np.array([r.r_total for r in races])
    party_r_obs = np.maximum(total_r - cand_r_total, 0.0)
    cap_fraction_r = 0.5
    budget_r = float(party_r_obs.sum())  # exactly enough to cover what's already committed, no more

    res = br.br_r(races, coef, sigma_model, party_d=party_d, cand_r_total=cand_r_total,
                   budget_r=budget_r, cap_fraction_r=cap_fraction_r, committed_r=party_r_obs)
    np.testing.assert_allclose(res.party, party_r_obs, atol=1.0)

    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
    n = len(races)
    expected = float(n) - float(payoff.p_win_shared(party_d, party_r_obs, arrays).sum())
    assert res.e_seats_own == pytest.approx(expected, abs=1e-6)
