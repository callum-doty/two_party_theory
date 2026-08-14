"""Unit tests for game/unified_theta.py's decision rule and Bellman
recursion -- pure logic on synthetic inputs, no model dependency."""

import numpy as np

from game.unified_theta import deploy_value, solve_bellman


def test_deploy_value_picks_highest_estimated_psv_and_realizes_true_psv():
    candidates = [0, 1, 2]
    v_uni_noisy = {0: 0.05, 1: 0.10, 2: 0.02}
    retention = {0: 0.90, 1: 0.20, 2: 0.95}  # race 1 has high raw appeal but low retention
    psv_true = {0: 0.045, 1: 0.018, 2: 0.019}
    # estimated PSV: race0=0.045, race1=0.020, race2=0.019 -> race 0 picked
    idx, realized = deploy_value(candidates, v_uni_noisy, retention, psv_true)
    assert idx == 0
    assert realized == psv_true[0]


def test_deploy_value_treats_nan_retention_as_zero_not_a_crash():
    candidates = [0, 1]
    v_uni_noisy = {0: 0.05, 1: 0.20}
    retention = {0: 0.5, 1: float("nan")}  # race 1 immaterial V_uni upstream
    psv_true = {0: 0.025, 1: 0.20}
    idx, realized = deploy_value(candidates, v_uni_noisy, retention, psv_true)
    assert idx == 0  # race 1's nan retention -> estimated 0, never picked
    assert realized == psv_true[0]


def test_solve_bellman_final_date_has_no_waiting_option():
    dates = ["t0", "t1"]
    deploy = {"t0": 0.01, "t1": 0.05}
    out = solve_bellman(dates, deploy)
    assert out["V"][-1] == deploy["t1"]
    assert out["V"][0] == max(deploy["t0"], deploy["t1"])
    assert np.isclose(out["theta_t0"], out["V"][0] - deploy["t0"])


def test_solve_bellman_theta_zero_when_deploying_now_is_always_best():
    dates = ["t0", "t1", "t2"]
    deploy = {"t0": 0.10, "t1": 0.05, "t2": 0.02}  # monotonically declining -> never wait
    out = solve_bellman(dates, deploy)
    assert np.isclose(out["theta_t0"], 0.0)
    assert out["V"][0] == deploy["t0"]


def test_solve_bellman_propagates_a_later_peak_backward():
    dates = ["t0", "t1", "t2"]
    deploy = {"t0": 0.01, "t1": 0.01, "t2": 0.20}  # value only realized at the last date
    out = solve_bellman(dates, deploy)
    assert out["V"] == [0.20, 0.20, 0.20]
    assert np.isclose(out["theta_t0"], 0.19)
