"""SLSQP-vs-surrogate agreement check for game/best_response_surrogate.py
(project_spec.md Section 12: "symmetrical validation for both players"
before the concave-envelope surrogate is trusted for anything). The old
surrogate (src/optimizer/concave_surrogate.py) validated only D's side
against a fixed/reactive R; its R-side mirror was built on the mirrored-
ceiling formula game/best_response.py moved away from on 2026-08-12 and was
never benchmarked. Since both sides now share payoff.p_win_shared, a single
surrogate serves both -- this file is the validation both sides needed."""

import numpy as np
import pytest

from game import best_response as br
from game import best_response_surrogate as brs
from backtest.model.margin import MarginModelCoefficients
from backtest.types import RaceRecord, SigmaModel

pytestmark = pytest.mark.slow  # exact-side solves are real SLSQP calls, not instant


def _synthetic_universe(n=30, seed=0):
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


def test_surrogate_br_d_agrees_with_exact_slsqp():
    races, coef, sigma_model, cand_r_total = _synthetic_universe()
    total_r = np.array([r.r_total for r in races])
    party_r = np.maximum(total_r - cand_r_total, 0.0)
    budget_d, cap_fraction_d = 2_000_000.0, 0.5

    exact = br.br_d(races, coef, sigma_model, party_r=party_r, cand_r_total=cand_r_total,
                     budget_d=budget_d, cap_fraction_d=cap_fraction_d)
    surrogate = brs.br_d_surrogate(races, coef, sigma_model, party_r=party_r, cand_r_total=cand_r_total,
                                    budget_d=budget_d, cap_fraction_d=cap_fraction_d)

    assert surrogate.e_seats_own == pytest.approx(exact.e_seats_own, abs=0.15)
    assert surrogate.party.sum() <= budget_d + 1.0


def test_surrogate_br_r_agrees_with_exact_slsqp():
    races, coef, sigma_model, cand_r_total = _synthetic_universe()
    total_d = np.array([r.d_total for r in races])
    party_d = np.maximum(total_d - np.array([r.cand_d_total for r in races]), 0.0)
    budget_r, cap_fraction_r = 2_000_000.0, 0.5

    exact = br.br_r(races, coef, sigma_model, party_d=party_d, cand_r_total=cand_r_total,
                     budget_r=budget_r, cap_fraction_r=cap_fraction_r)
    surrogate = brs.br_r_surrogate(races, coef, sigma_model, party_d=party_d, cand_r_total=cand_r_total,
                                    budget_r=budget_r, cap_fraction_r=cap_fraction_r)

    assert surrogate.e_seats_own == pytest.approx(exact.e_seats_own, abs=0.15)
    assert surrogate.party.sum() <= budget_r + 1.0
