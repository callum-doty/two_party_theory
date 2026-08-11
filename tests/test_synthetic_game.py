"""Level C validation (project_spec.md Section 20): does the generic
best-response algorithm recover a synthetic game's known closed-form Nash
equilibrium?"""

from validation.synthetic_games import solve_and_check


def test_recovers_known_equilibrium_from_asymmetric_start():
    res = solve_and_check(n_races=5, budget=1_000_000.0)
    assert res.converged
    assert res.max_error_vs_known_equilibrium < 100.0  # within $100 of $200k/race


def test_scales_to_more_races():
    res = solve_and_check(n_races=20, budget=2_000_000.0)
    assert res.converged
    assert res.max_error_vs_known_equilibrium < 1000.0
