"""
Persistent strategic value (project_spec.md Sections 13-14) -- the central
NEW object this project adds on top of the old trilogy: does a race stay
attractive after the OPPONENT is allowed to respond rationally, not just
after a one-shot unilateral deviation?

For race i, deviate D_i' = D_i + delta, financed by removing delta from D's
own current lowest-marginal-value funded race (spec Section 13's "removing
delta from their portfolio's current marginal use" -- operationalized here
literally as the lowest-MSG^D race still holding party money, cascading to
the next-lowest if it can't cover delta alone, since "the portfolio's
marginal use" is by definition wherever the portfolio's own optimizer would
cut first).

    V_uni_i    = U_D(D', R)              - U_D(D, R)     (opponent held fixed)
    PSV_i      = U_D(D', BR_R(D'))       - U_D(D, R)     (opponent best-responds)
    Erosion_i  = V_uni_i - PSV_i
    Retention_i = PSV_i / V_uni_i

All four use payoff.p_win_shared, the same fixed-baseline formula BR_D/BR_R
themselves search (game/best_response.py's module docstring) -- there is no
separate D-anchored evaluation left to diverge from the opponent's own
best-response objective.

A high V_uni with PSV near zero means the apparent opportunity is competed
away; a high PSV is a genuine candidate for persistent strategic mispricing
(spec Section 14). PSV_R is the mirror-image statistic for a Republican
deviation, financed the same way from R's own lowest-MSG^R funded race, D
best-responding.

Each call to persistent_strategic_value_d/_r involves ONE full best-response
solve (a 433-race SLSQP), so this is meant to be run over a small candidate
set (e.g. the top-|Z| races from exploitability.race_level_surplus), not the
full universe -- mirrors scripts/game_theory/best_response_trajectories.py's
same cost tradeoff.

BASELINE CHOICE, found empirically and worth flagging before trusting a run:
spec Section 14 writes PSV_i literally as UD(D', BR_R(D')) - UD(D, R) --
i.e. against the OBSERVED R baseline, same as V_uni. On the real 2024
universe this makes PSV nearly race-INVARIANT whenever RegretR itself is
large: BR_R(D') for almost any D' near D_obs recovers most of that same
RegretR-sized R-side swing, and every race's literal-formula PSV comes back
dominated by that constant -RegretR term rather than by the race-specific
signal Section 14's own worked example (+0.12 -> +0.01) is illustrating.
Both functions below default to the literal spec formula
(baseline_e_seats=None -> observed U_D(D,R)), but accept an explicit
baseline_e_seats so callers can instead pass U_D(D, BR_R(D_obs)) (R ALREADY
at its own best response, computed ONCE and reused across every race in a
candidate set) -- that isolates each race's own erosion from the shared,
race-independent RegretR term, and is the more informative comparison
whenever observed spending is itself far from either side's unilateral
optimum.

RETENTION-RATIO INSTABILITY, found 2026-08-11 investigating a retention
>100% anomaly in the 2022/2024 historical backtest (docs/methodology.md
has the full writeup): retention_rate = PSV / V_uni is a ratio of two small
numbers and is only a meaningful percentage when V_uni itself is large
enough to represent a real, material unilateral opportunity. Two distinct
failure modes were found empirically:

  1. Candidate races selected by top-|Z| (a NORMALIZED, dimensionless
     surplus measure) can have Z far from zero while the race is at exactly
     $0 current party spend -- MSG evaluated AT $0 sits at the steepest,
     most unstable point of the persuasion-ceiling curve (a known artifact,
     see scripts/game_theory/race_level_exploitability.py's own
     "low-spend MSG artifact" caveat). FIX: callers should restrict the
     candidate pool to races with real current party spend before ranking
     by |Z| -- see historical_backtest.py's min_party_spend filter. This
     module has no way to enforce that itself (it only computes PSV for a
     race index it's given), so it is the CALLER's responsibility.
  2. Even among currently-funded races, |V_uni| itself can be small enough
     (a few hundredths of a percent of a seat) that PSV/V_uni is dividing
     two near-zero, noisy numbers. RETENTION_MATERIALITY_THRESHOLD below
     (0.001 expected seats, i.e. 1/1000th of a seat) is the bar both
     functions use before reporting a numeric retention_rate at all --
     smaller than that, retention_rate is NaN ("no material unilateral
     opportunity here to begin with," not "highly retained").
"""

from __future__ import annotations

import numpy as np

from . import best_response as br
from . import gradients
from . import payoff

RETENTION_MATERIALITY_THRESHOLD = 0.001  # expected seats; below this, V_uni is too small for PSV/V_uni to be a meaningful percentage


def _finance_delta(party_own: np.ndarray, msg_own: np.ndarray, target_idx: int,
                    delta: float, cap: float) -> np.ndarray:
    """Return a new party allocation: +delta at target_idx (capped), financed
    by removing delta from the lowest-msg_own funded races (excluding
    target_idx), cascading across races if one alone can't cover it."""
    party_new = party_own.copy()
    take = min(delta, max(cap - party_new[target_idx], 0.0))
    party_new[target_idx] += take

    order = np.argsort(msg_own)  # lowest marginal value first
    remaining = take
    for j in order:
        if remaining <= 0:
            break
        if j == target_idx or party_new[j] <= 0:
            continue
        cut = min(party_new[j], remaining)
        party_new[j] -= cut
        remaining -= cut
    return party_new


def persistent_strategic_value_d(races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                                  race_idx: int, delta: float,
                                  cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15,
                                  baseline_e_seats: float | None = None) -> dict:
    """PSV_i^D, V_uni_i^D, erosion, retention rate for a Democratic deviation
    of `delta` dollars at race `race_idx`.

    baseline_e_seats: the U_D value both V_uni and PSV are measured against.
    None (default) = literal spec formula, U_D(D, R_observed). Pass
    U_D(D, BR_R(D_observed)) instead to isolate this race's own erosion from
    the shared RegretR term -- see module docstring. Compute that baseline
    ONCE (br.br_r + payoff.p_win_shared against party_d=party_d_obs) and
    reuse it across every race in a candidate set rather than recomputing it
    per call."""
    floors_d = np.array([r.cand_d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    d0 = np.array([r.d_total for r in races])
    party_d_obs = np.maximum(d0 - floors_d, 0.0)
    party_r_obs = np.maximum(r0 - cand_r_total, 0.0)
    cap_d = cap_fraction_d * budget_d

    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
    baseline_obs = float(payoff.p_win_shared(party_d_obs, party_r_obs, arrays).sum())
    baseline = baseline_obs if baseline_e_seats is None else baseline_e_seats

    msg_d_obs = gradients.msg_d(party_d_obs, party_r_obs, arrays)
    party_d_dev = _finance_delta(party_d_obs, msg_d_obs, race_idx, delta, cap_d)

    v_uni = float(payoff.p_win_shared(party_d_dev, party_r_obs, arrays).sum()) - baseline_obs

    res_r_prime = br.br_r(races, coef, sigma_model, party_d=party_d_dev, cand_r_total=cand_r_total,
                           budget_r=budget_r, cap_fraction_r=cap_fraction_r)
    psv = float(payoff.p_win_shared(party_d_dev, res_r_prime.party, arrays).sum()) - baseline

    erosion = v_uni - psv
    retention = float(psv / v_uni) if abs(v_uni) > RETENTION_MATERIALITY_THRESHOLD else float("nan")
    return dict(
        district_id=races[race_idx].district_id, delta=delta,
        V_uni=v_uni, PSV=psv, erosion=erosion, retention_rate=retention,
    )


def persistent_strategic_value_r(races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                                  race_idx: int, delta: float,
                                  cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15,
                                  baseline_e_seats_r: float | None = None) -> dict:
    """PSV_i^R, V_uni_i^R, erosion, retention rate for a Republican deviation
    of `delta` dollars at race `race_idx` -- mirror of the D-side function:
    R deviates, financed from R's own lowest-MSG^R funded race, D best-responds.

    baseline_e_seats_r: see persistent_strategic_value_d's docstring. None
    (default) = literal spec formula, U_R(D_observed, R). Pass
    U_R(BR_D(D_observed), R_observed) to isolate this race's own erosion from
    the shared RegretD term."""
    n = len(races)
    floors_d = np.array([r.cand_d_total for r in races])
    r0 = np.array([r.r_total for r in races])
    d0 = np.array([r.d_total for r in races])
    party_d_obs = np.maximum(d0 - floors_d, 0.0)
    party_r_obs = np.maximum(r0 - cand_r_total, 0.0)
    cap_r = cap_fraction_r * budget_r

    arrays = payoff.baseline_arrays(races, coef, sigma_model, cand_r_total)
    baseline_d_obs = float(payoff.p_win_shared(party_d_obs, party_r_obs, arrays).sum())
    baseline_r_obs = float(n) - baseline_d_obs
    baseline_r = baseline_r_obs if baseline_e_seats_r is None else baseline_e_seats_r

    msg_r_obs = gradients.msg_r(party_d_obs, party_r_obs, arrays)
    party_r_dev = _finance_delta(party_r_obs, msg_r_obs, race_idx, delta, cap_r)

    e_d_uni = float(payoff.p_win_shared(party_d_obs, party_r_dev, arrays).sum())
    v_uni = (float(n) - e_d_uni) - baseline_r_obs

    res_d_prime = br.br_d(races, coef, sigma_model, party_r=party_r_dev, cand_r_total=cand_r_total,
                           budget_d=budget_d, cap_fraction_d=cap_fraction_d)
    e_d_star = float(payoff.p_win_shared(res_d_prime.party, party_r_dev, arrays).sum())
    psv = (float(n) - e_d_star) - baseline_r

    erosion = v_uni - psv
    retention = float(psv / v_uni) if abs(v_uni) > RETENTION_MATERIALITY_THRESHOLD else float("nan")
    return dict(
        district_id=races[race_idx].district_id, delta=delta,
        V_uni=v_uni, PSV=psv, erosion=erosion, retention_rate=retention,
    )
