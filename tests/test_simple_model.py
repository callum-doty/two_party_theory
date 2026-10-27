"""Fast synthetic checks for simple_model's two jobs and the allocator."""

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from simple_model import baseline as bl
from simple_model import pvi as pvi_mod
from simple_model.allocate import allocate, win_prob
from simple_model.spending import SpendingEffect


def _frame(n=400, seed=0, noise=5.0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(dict(
        pvi=rng.uniform(-25, 25, n),
        status=rng.choice(bl.STATUSES, n),
        floor_d=rng.uniform(1e5, 5e6, n),
        floor_r=rng.uniform(1e5, 5e6, n),
        gb=-1.0,
    ))
    true = np.array([1.0, 1.8, 4.0, 2.0, 2.5])
    df["margin"] = bl._design(df) @ true + df["gb"] + rng.normal(0, noise, n)
    return df, true


def test_baseline_recovers_coefficients_and_sigma():
    df, true = _frame()
    base = bl.fit_frame(df)
    assert np.allclose(base.coef, true, atol=0.6)
    assert base.sigma == pytest.approx(5.0, rel=0.1)


def test_baseline_ignores_uncontested_races():
    df, true = _frame()
    df.loc[:9, "margin"] = 100.0
    assert np.allclose(bl.fit_frame(df).coef, true, atol=0.6)


def test_generic_ballot_is_uniform_swing():
    df, _ = _frame()
    base = bl.fit_frame(df)
    assert np.allclose(base.margin(df.assign(gb=2.0)) - base.margin(df), 3.0)


def _pres_table():
    rows = [  # lines_cycle, pres_year, district, d_share
        (2022, 2016, "AA-01", 0.60), (2022, 2020, "AA-01", 0.62),
        (2022, 2016, "AA-02", 0.40), (2022, 2020, "AA-02", 0.41),
        (2024, 2020, "AA-01", 0.62), (2024, 2024, "AA-01", 0.58),   # not redrawn
        (2024, 2020, "AA-02", 0.50), (2024, 2024, "AA-02", 0.47),   # redrawn
    ]
    return pd.DataFrame(rows, columns=["lines_cycle", "pres_year", "district_id", "d_share"])


def test_pvi_uses_only_elections_before_the_house_race(monkeypatch):
    monkeypatch.setattr(pvi_mod, "_table", _pres_table)
    nat = pvi_mod.NATIONAL_D2PARTY
    # 2024 House race: 2016 and 2020 count, the same-year 2024 result does not
    expected = ((0.60 - nat[2016]) + (0.62 - nat[2020])) / 2 * 100
    assert pvi_mod.load_pvi(2024)["AA-01"] == pytest.approx(expected)


def test_pvi_borrows_old_results_only_for_unchanged_districts(monkeypatch):
    monkeypatch.setattr(pvi_mod, "_table", _pres_table)
    # AA-02 was redrawn, so its 2016 result on the old map must not be used
    expected = (0.50 - pvi_mod.NATIONAL_D2PARTY[2020]) * 100
    assert pvi_mod.load_pvi(2024)["AA-02"] == pytest.approx(expected)


def test_spending_shift_zero_concave_and_per_voter():
    eff = SpendingEffect(scale_pts=1.5, saturation_per_voter=10.0)
    assert eff.shift(0.0, 5e5) == 0.0
    a, b, c = eff.shift([1e6, 2e6, 3e6], 5e5)
    assert 0 < b - a < a and c - b < b - a
    # same dollars per voter -> same shift, regardless of district size
    assert eff.shift(1e6, 5e5) == pytest.approx(eff.shift(2e6, 1e6))


def test_observed_allocation_reproduces_baseline():
    mu, obs = np.array([-5.0, 0.0, 8.0]), np.array([1e6, 3e6, 0.0])
    p = win_prob(mu, 6.0, SpendingEffect(), np.full(3, 5e5), obs, obs)
    assert np.allclose(p, norm.cdf(mu / 6.0))


def test_allocation_respects_budget_and_cap():
    rng = np.random.default_rng(1)
    mu = rng.uniform(-15, 15, 50)
    x = allocate(mu, 6.0, SpendingEffect(), np.full(50, 5e5), np.zeros(50),
                 budget=20e6, cap=3e6, step=1e5)
    assert x.sum() <= 20e6 + 1e-6
    assert x.max() <= 3e6 + 1e-6
    assert np.allclose(x / 1e5, np.round(x / 1e5))


def test_money_goes_to_the_closest_race():
    mu = np.array([-20.0, 0.0, 20.0])
    x = allocate(mu, 6.0, SpendingEffect(), np.full(3, 5e5), np.zeros(3),
                 budget=2e6, cap=2e6, step=1e5)
    assert x[1] == pytest.approx(2e6)


def test_allocation_on_top_of_existing_spend():
    mu, start = np.array([0.0, 0.0]), np.array([4e6, 0.0])
    x = allocate(mu, 6.0, SpendingEffect(), np.full(2, 5e5), start,
                 budget=1e6, cap=1e6, step=1e5, start=start)
    assert x.sum() == pytest.approx(start.sum() + 1e6)
    assert x[1] == pytest.approx(1e6)   # the race with nothing yet has the higher marginal value


def test_greedy_matches_brute_force_on_small_problem():
    mu, sigma, step = np.array([-9.0, -2.0, 4.0]), 5.0, 5e5
    eff, voters, obs = SpendingEffect(scale_pts=4.0), np.full(3, 4e5), np.zeros(3)
    budget, cap = 3e6, 2e6
    x = allocate(mu, sigma, eff, voters, obs, budget, cap, step=step)
    levels = np.arange(0, cap + step, step)
    best = max(win_prob(mu, sigma, eff, voters, np.array(c), obs).sum()
               for c in itertools.product(levels, repeat=3) if sum(c) <= budget)
    assert win_prob(mu, sigma, eff, voters, x, obs).sum() == pytest.approx(best, abs=1e-6)
