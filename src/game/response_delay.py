"""
Response-delay sweep: does strategic leverage survive if the opponent
cannot respond instantaneously? (2026-08-13 research-discussion follow-up,
second major piece alongside best_response.py's locked-capital extension:
"the opportunity doesn't survive a fully informed instantaneous response,
but does survive long enough to matter electorally because response is not
instantaneous.")

Builds on strategic_leverage.py's PSV/leverage machinery but swaps the
opponent's FRICTIONLESS best response (BR_R(D'), searched over R's entire
budget every time, as if R could instantly and perfectly reallocate) for a
LOCKED-CAPITAL best response (best_response.br_r(..., committed_r=...)):
R can only reallocate whatever fraction of its budget is not already
committed as of the response date. Committed capital is REAL, dated FEC
data (estimation.commitment_timing.committed_capital_per_race), not a
synthetic schedule.

Design: one side moves delta at a reference date t0 (the other side
observes the move immediately -- this module tests RESPONSE DELAY, i.e.
how much of the opponent's budget is still flexible by the time it reacts,
not INFORMATION delay, i.e. whether the opponent notices at all). tau=0
uses whatever fraction of the opponent's budget was already committed at
t0 itself (not necessarily zero -- t0 is already partway through the
season); larger tau tightens the opponent's flexible budget and per-race
room further as more of its cycle spending becomes historical fact between
t0 and the response.

Baseline choice: for each tau, the baseline is U_D(D_obs, BR_R(D_obs;
committed_r_at(t0+tau))) -- the opponent's best response to the UNCHANGED
allocation, under the SAME commitment constraint being tested at that tau.
This isolates "how much does THIS tau's commitment level erode the specific
move," not conflated with "how much worse is the opponent's baseline
response in general as commitment grows" -- the same isolated-baseline
principle persistent_value.py established (compute once, reuse), here
parameterized by tau instead of being a single fixed quantity, since the
"once" is now once per tau rather than once per whole analysis.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from . import best_response as br
from . import gradients
from . import payoff
from .persistent_value import RETENTION_MATERIALITY_THRESHOLD, _finance_delta
from estimation.commitment_timing import (
    commitment_fraction_as_of,
    commitment_fraction_curve,
    committed_capital_per_race,
)


def leverage_by_response_delay_d(races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                                  race_idx: int, delta: float, cycle: int, t0: date, taus: list[int],
                                  party_d_obs: np.ndarray, party_r_obs: np.ndarray, arrays: dict,
                                  cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15) -> list[dict]:
    """Democratic move at race_idx, financed the same way strategic_
    leverage.py's leverage_curve_d does (D itself is NOT commitment-
    constrained here -- only R's RESPONSE is, isolating the response-delay
    question from D's own commitment mechanics). Sweeps R's reaction delay
    tau (days after t0) and reports how PSV/leverage/retention change as
    R's flexible budget shrinks."""
    baseline_obs = float(payoff.p_win_shared(party_d_obs, party_r_obs, arrays).sum())
    cap_d = cap_fraction_d * budget_d
    msg_d_obs = gradients.msg_d(party_d_obs, party_r_obs, arrays)
    party_d_dev = _finance_delta(party_d_obs, msg_d_obs, race_idx, delta, cap_d)
    v_uni = float(payoff.p_win_shared(party_d_dev, party_r_obs, arrays).sum()) - baseline_obs
    district_id = races[race_idx].district_id

    curve = commitment_fraction_curve(cycle, "R")
    rows = []
    for tau in taus:
        as_of = t0 + timedelta(days=tau)
        committed_r = committed_capital_per_race(cycle, "R", as_of, party_r_obs, curve=curve)
        frac = commitment_fraction_as_of(cycle, "R", as_of, curve=curve)
        flexible_budget_r = budget_r - float(committed_r.sum())

        res_r_baseline = br.br_r(races, coef, sigma_model, party_d=party_d_obs, cand_r_total=cand_r_total,
                                  budget_r=budget_r, cap_fraction_r=cap_fraction_r, committed_r=committed_r)
        baseline_d_tau = float(payoff.p_win_shared(party_d_obs, res_r_baseline.party, arrays).sum())

        res_r_prime = br.br_r(races, coef, sigma_model, party_d=party_d_dev, cand_r_total=cand_r_total,
                               budget_r=budget_r, cap_fraction_r=cap_fraction_r, committed_r=committed_r)
        psv = float(payoff.p_win_shared(party_d_dev, res_r_prime.party, arrays).sum()) - baseline_d_tau
        retention = float(psv / v_uni) if abs(v_uni) > RETENTION_MATERIALITY_THRESHOLD else float("nan")
        leverage = psv / (delta / 1e6)

        displacement = res_r_prime.party - res_r_baseline.party
        reshuffle_l1 = float(np.abs(displacement).sum())

        rows.append(dict(
            district_id=district_id, delta=delta, tau_days=tau, as_of_date=str(as_of),
            commitment_fraction_r=frac, flexible_budget_r=flexible_budget_r,
            V_uni=v_uni, PSV=psv, retention_rate=retention,
            leverage_seats_per_million=leverage, reshuffle_l1=reshuffle_l1,
        ))
    return rows


def leverage_by_response_delay_r(races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                                  race_idx: int, delta: float, cycle: int, t0: date, taus: list[int],
                                  party_d_obs: np.ndarray, party_r_obs: np.ndarray, arrays: dict,
                                  n_races: int, cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15
                                  ) -> list[dict]:
    """Mirror of leverage_by_response_delay_d: a Republican move, Democratic
    RESPONSE is commitment-constrained by tau."""
    baseline_d_obs = float(payoff.p_win_shared(party_d_obs, party_r_obs, arrays).sum())
    baseline_r_obs = float(n_races) - baseline_d_obs
    cap_r = cap_fraction_r * budget_r
    msg_r_obs = gradients.msg_r(party_d_obs, party_r_obs, arrays)
    party_r_dev = _finance_delta(party_r_obs, msg_r_obs, race_idx, delta, cap_r)
    e_d_uni = float(payoff.p_win_shared(party_d_obs, party_r_dev, arrays).sum())
    v_uni = (float(n_races) - e_d_uni) - baseline_r_obs
    district_id = races[race_idx].district_id

    curve = commitment_fraction_curve(cycle, "D")
    rows = []
    for tau in taus:
        as_of = t0 + timedelta(days=tau)
        committed_d = committed_capital_per_race(cycle, "D", as_of, party_d_obs, curve=curve)
        frac = commitment_fraction_as_of(cycle, "D", as_of, curve=curve)
        flexible_budget_d = budget_d - float(committed_d.sum())

        res_d_baseline = br.br_d(races, coef, sigma_model, party_r=party_r_obs, cand_r_total=cand_r_total,
                                  budget_d=budget_d, cap_fraction_d=cap_fraction_d, committed_d=committed_d)
        e_d_star_baseline = float(payoff.p_win_shared(res_d_baseline.party, party_r_obs, arrays).sum())
        baseline_r_tau = float(n_races) - e_d_star_baseline

        res_d_prime = br.br_d(races, coef, sigma_model, party_r=party_r_dev, cand_r_total=cand_r_total,
                               budget_d=budget_d, cap_fraction_d=cap_fraction_d, committed_d=committed_d)
        e_d_star = float(payoff.p_win_shared(res_d_prime.party, party_r_dev, arrays).sum())
        psv = (float(n_races) - e_d_star) - baseline_r_tau
        retention = float(psv / v_uni) if abs(v_uni) > RETENTION_MATERIALITY_THRESHOLD else float("nan")
        leverage = psv / (delta / 1e6)

        displacement = res_d_prime.party - res_d_baseline.party
        reshuffle_l1 = float(np.abs(displacement).sum())

        rows.append(dict(
            district_id=district_id, delta=delta, tau_days=tau, as_of_date=str(as_of),
            commitment_fraction_d=frac, flexible_budget_d=flexible_budget_d,
            V_uni=v_uni, PSV=psv, retention_rate=retention,
            leverage_seats_per_million=leverage, reshuffle_l1=reshuffle_l1,
        ))
    return rows
