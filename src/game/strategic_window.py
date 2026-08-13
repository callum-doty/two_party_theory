"""
Strategic window: at what point in the cycle does a race's opportunity
become durable against the opponent's OWN currently-committed capital?
(2026-08-13, the "next question" identified after the response-delay
result: not "which race forces the biggest sacrifice" -- static forcing
mostly failed to replicate -- but "when does an opportunity become hard to
neutralize because the opponent has already committed too much capital.")

Collapses response_delay.py's separate (t0, tau) design to a single
reference-date axis: tau=0, i.e. the opponent's best response is evaluated
using whatever national-committee capital it has ACTUALLY committed as of
the SAME date the mover commits, not an extra artificial delay stacked on
top. This matches the descriptive question directly ("at 90 days out, X%
retained; at 7 days out, Y% retained") without the combinatorial cost of a
full (t, tau) grid -- a full cube at the scale discussed would be on the
order of 8000+ exact-SLSQP solves; this collapsed version is ~300.
response_delay.py's tau-at-fixed-t0 sweep remains the tool for the second
question (does ADDITIONAL delay beyond a fixed commitment date matter),
kept separate rather than merged into a slower combined sweep.

Uses estimation.commitment_timing's TIER-POOLED curves (competitive/lean/
safe_likely, by Cook rating) rather than response_delay.py's single
blended party curve -- checked for a real timing gradient and adequate
per-tier sample size before building this (docs/methodology.md).

For race i, this produces retention_i(t) -- the natural object for defining
T_i^80 = the earliest reference date at which the opponent's own
already-committed capital is enough that at least 80% of the deviation's
unilateral value survives its best response.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from . import best_response as br
from . import gradients
from . import payoff
from .persistent_value import RETENTION_MATERIALITY_THRESHOLD, _finance_delta
from estimation.commitment_timing import build_tiered_curves, committed_capital_per_race_tiered


def retention_by_date_d(races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                         race_idx: int, delta: float, cycle: int, dates: list[date],
                         party_d_obs: np.ndarray, party_r_obs: np.ndarray, arrays: dict,
                         curves_r: dict | None = None,
                         cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15,
                         baseline_cache: dict | None = None) -> list[dict]:
    """Democratic move at race_idx, financed the same way strategic_
    leverage.py/response_delay.py do (D is not itself commitment-
    constrained -- only R's response is, via R's REAL tier-pooled
    committed capital as of each date in `dates`).

    baseline_cache: R's best response to the UNCHANGED D allocation at each
    date depends only on (cycle, date, party_d_obs) -- NOT on which race is
    being tested -- so it is identical across every D-side candidate at a
    given date. Callers sweeping multiple races should pass the SAME dict
    across calls (mutated in place) so it is solved once per date and
    reused, instead of once per (race, date) -- an ~40% reduction in exact-
    SLSQP solves for a multi-race sweep. None (default) computes a private
    cache local to this one call, e.g. for a single-race spot check."""
    baseline_obs = float(payoff.p_win_shared(party_d_obs, party_r_obs, arrays).sum())
    cap_d = cap_fraction_d * budget_d
    msg_d_obs = gradients.msg_d(party_d_obs, party_r_obs, arrays)
    party_d_dev = _finance_delta(party_d_obs, msg_d_obs, race_idx, delta, cap_d)
    v_uni = float(payoff.p_win_shared(party_d_dev, party_r_obs, arrays).sum()) - baseline_obs
    district_id = races[race_idx].district_id

    if curves_r is None:
        curves_r = build_tiered_curves(cycle, "R", races)
    cache = {} if baseline_cache is None else baseline_cache

    rows = []
    for t in dates:
        key = str(t)
        if key not in cache:
            committed_r = committed_capital_per_race_tiered(cycle, "R", t, races, party_r_obs, curves=curves_r)
            res_r_baseline = br.br_r(races, coef, sigma_model, party_d=party_d_obs, cand_r_total=cand_r_total,
                                      budget_r=budget_r, cap_fraction_r=cap_fraction_r, committed_r=committed_r)
            baseline_d_t = float(payoff.p_win_shared(party_d_obs, res_r_baseline.party, arrays).sum())
            cache[key] = (committed_r, baseline_d_t)
        committed_r, baseline_d_t = cache[key]
        flexible_budget_r = budget_r - float(committed_r.sum())

        res_r_prime = br.br_r(races, coef, sigma_model, party_d=party_d_dev, cand_r_total=cand_r_total,
                               budget_r=budget_r, cap_fraction_r=cap_fraction_r, committed_r=committed_r)
        psv = float(payoff.p_win_shared(party_d_dev, res_r_prime.party, arrays).sum()) - baseline_d_t
        retention = float(psv / v_uni) if abs(v_uni) > RETENTION_MATERIALITY_THRESHOLD else float("nan")

        rows.append(dict(
            district_id=district_id, delta=delta, ref_date=str(t),
            flexible_budget_r=flexible_budget_r,
            V_uni=v_uni, PSV=psv, retention_rate=retention,
        ))
    return rows


def retention_by_date_r(races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                         race_idx: int, delta: float, cycle: int, dates: list[date],
                         party_d_obs: np.ndarray, party_r_obs: np.ndarray, arrays: dict, n_races: int,
                         curves_d: dict | None = None,
                         cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15,
                         baseline_cache: dict | None = None) -> list[dict]:
    """Mirror of retention_by_date_d: a Republican move, D's response
    constrained by D's own real tier-pooled committed capital.
    baseline_cache: see retention_by_date_d's docstring -- same reuse
    across every R-side candidate at a given cycle."""
    baseline_d_obs = float(payoff.p_win_shared(party_d_obs, party_r_obs, arrays).sum())
    baseline_r_obs = float(n_races) - baseline_d_obs
    cap_r = cap_fraction_r * budget_r
    msg_r_obs = gradients.msg_r(party_d_obs, party_r_obs, arrays)
    party_r_dev = _finance_delta(party_r_obs, msg_r_obs, race_idx, delta, cap_r)
    e_d_uni = float(payoff.p_win_shared(party_d_obs, party_r_dev, arrays).sum())
    v_uni = (float(n_races) - e_d_uni) - baseline_r_obs
    district_id = races[race_idx].district_id

    if curves_d is None:
        curves_d = build_tiered_curves(cycle, "D", races)
    cache = {} if baseline_cache is None else baseline_cache

    rows = []
    for t in dates:
        key = str(t)
        if key not in cache:
            committed_d = committed_capital_per_race_tiered(cycle, "D", t, races, party_d_obs, curves=curves_d)
            res_d_baseline = br.br_d(races, coef, sigma_model, party_r=party_r_obs, cand_r_total=cand_r_total,
                                      budget_d=budget_d, cap_fraction_d=cap_fraction_d, committed_d=committed_d)
            e_d_star_baseline = float(payoff.p_win_shared(res_d_baseline.party, party_r_obs, arrays).sum())
            baseline_r_t = float(n_races) - e_d_star_baseline
            cache[key] = (committed_d, baseline_r_t)
        committed_d, baseline_r_t = cache[key]
        flexible_budget_d = budget_d - float(committed_d.sum())

        res_d_prime = br.br_d(races, coef, sigma_model, party_r=party_r_dev, cand_r_total=cand_r_total,
                               budget_d=budget_d, cap_fraction_d=cap_fraction_d, committed_d=committed_d)
        e_d_star = float(payoff.p_win_shared(res_d_prime.party, party_r_dev, arrays).sum())
        psv = (float(n_races) - e_d_star) - baseline_r_t
        retention = float(psv / v_uni) if abs(v_uni) > RETENTION_MATERIALITY_THRESHOLD else float("nan")

        rows.append(dict(
            district_id=district_id, delta=delta, ref_date=str(t),
            flexible_budget_d=flexible_budget_d,
            V_uni=v_uni, PSV=psv, retention_rate=retention,
        ))
    return rows


def strategic_opening_date(rows: list[dict], threshold: float = 0.80) -> str | None:
    """T_i^threshold: the earliest ref_date (rows must be pre-sorted
    chronologically) at which retention_rate >= threshold and stays there
    for every later date in `rows` -- not just the first date that happens
    to clear the bar, in case retention is non-monotonic (the >100%-race
    counter-example in response_delay.py's findings shows retention can
    move in either direction, so a single crossing is not automatically
    permanent)."""
    for i, row in enumerate(rows):
        if all(np.isfinite(r["retention_rate"]) and r["retention_rate"] >= threshold for r in rows[i:]):
            return row["ref_date"]
    return None
