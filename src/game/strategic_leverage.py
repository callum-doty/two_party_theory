"""
Strategic leverage and response displacement -- follow-on to persistent_
value.py from the 2026-08-13 research-direction discussion ("where can one
dollar of pressure force the opponent to incur the most costly possible
response," rather than searching for another static allocation).

MATHEMATICAL NOTE, checked before writing any of this: the discussion
proposed a "counter-response cost" statistic, SL_i^D(delta) = U_R(D_obs,
BR_R(D_obs)) - U_R(D', BR_R(D')) -- how much D's move costs R, measured
against R's OWN best-response payoff. In a constant-sum game (U_R = n -
U_D, this project's payoff model throughout -- every p_win_shared caller
sums to n), that is ALGEBRAICALLY IDENTICAL to persistent_value.py's PSV_i^D
under the isolated baseline (baseline_e_seats = U_D(D_obs, BR_R(D_obs))):
substituting U_R = n - U_D into both terms and cancelling n leaves
U_D(D', BR_R(D')) - U_D(D_obs, BR_R(D_obs)), which is exactly PSV. So "how
much does this cost Republicans" is not new information sitting on top of
PSV -- it IS PSV, just read off the other side of the same zero-sum ledger.
This module does not recompute that scalar under a new name.

What's actually new, and not redundant with anything already computed:

  1. A LEVERAGE CURVE across multiple delta values per race
     (persistent_value.py's compute_persistent_value.py runs a single fixed
     delta per race) -- lets you see whether PSV-per-dollar-committed is
     roughly constant or declining as the commitment size grows.
  2. The RESPONSE-DISPLACEMENT MAP: BR_R(D') - BR_R(D_obs), race by race --
     which specific races R's optimal response pulls money OUT of (or INTO)
     to fund its reaction to a D move in race i. Nothing in this codebase
     computes or exposes this per-race breakdown; persistent_value.py only
     ever returns the aggregate PSV scalar, discarding the full allocation
     vector BR_R(D') that it already had to compute internally.
  3. reshuffle_l1 / reshuffle_per_million: total dollars of R's portfolio
     that move (sum of |displacement|), independent of whether that
     reshuffling nets out to a large or small aggregate PSV. This is
     genuinely distinct from PSV/leverage: R's reshuffling can be large
     (money pulled from several valuable races) even when the aggregate
     seat swing is small, if the reshuffling is roughly self-cancelling in
     the zero-sum accounting -- exactly the "decoy race" case the
     discussion described (low direct D value, but R still forced to
     substantially reallocate). PSV alone cannot distinguish a decoy race
     from a race R can simply ignore; reshuffle_l1 can.

CAPPING, relevant once delta approaches a race's per-race cap
(cap_fraction_d/r * budget): `_finance_delta` silently clips the amount it
actually places at the target race to `cap - current_spend` (best_response.py
enforces the same per-race cap in the opponent's own re-solve). At small
delta this never binds; at large delta it can, and dividing a fixed nominal
delta into PSV would then UNDERSTATE leverage (the denominator is bigger
than what was actually deployed). Every row therefore reports
delta_requested, delta_deployed, and a capped flag, and leverage/reshuffle
are normalized by delta_deployed, not the nominal request.

leverage_curve_d/_r are deliberately thin: they call the same
_finance_delta financing convention and the same isolated-baseline
convention persistent_value.py already established, and reuse
best_response_surrogate.py (already validated against exact SLSQP within
0.03-0.10 expected seats) for the opponent's re-solve, since a multi-delta,
multi-race sweep at exact-SLSQP cost (~20-30s/solve) would be impractical.
Callers should exact-check the single most interesting candidate before
reporting it, the same pattern compute_persistent_value.py's own
"Verification pitfall" note (docs/methodology.md) already established for
this kind of check.
"""

from __future__ import annotations

import numpy as np

from . import best_response as br
from . import best_response_surrogate as brs
from . import gradients
from . import payoff
from .persistent_value import RETENTION_MATERIALITY_THRESHOLD, _finance_delta

N_TOP_MOVERS = 8
MIN_MOVER_DOLLARS = 1.0  # ignore sub-$1 numerical noise when listing movers


def _top_movers(displacement: np.ndarray, races, exclude_idx: int) -> tuple[list[dict], list[dict]]:
    order = np.argsort(displacement)
    cuts = [
        dict(district_id=races[j].district_id, delta=float(displacement[j]))
        for j in order[:N_TOP_MOVERS]
        if displacement[j] < -MIN_MOVER_DOLLARS and j != exclude_idx
    ]
    adds = [
        dict(district_id=races[j].district_id, delta=float(displacement[j]))
        for j in order[::-1][:N_TOP_MOVERS]
        if displacement[j] > MIN_MOVER_DOLLARS and j != exclude_idx
    ]
    return cuts, adds


def leverage_curve_d(races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                      race_idx: int, deltas: list[float],
                      party_d_obs: np.ndarray, party_r_obs: np.ndarray, party_r_star: np.ndarray,
                      baseline_d: float, arrays: dict,
                      cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15,
                      use_surrogate: bool = True) -> list[dict]:
    """Sweep `deltas` for a Democratic deviation at `race_idx`. Per delta:
    V_uni, PSV, retention (persistent_value.py's own quantities), plus
    leverage_seats_per_million and the R-side response-displacement map.

    party_r_star must be BR_R(party_d_obs).party, computed ONCE by the
    caller and reused across every race/delta in a sweep -- same reuse
    persistent_value.py's isolated baseline already relies on. baseline_d
    must be U_D(party_d_obs, party_r_star), i.e. R already at its own best
    response."""
    baseline_obs = float(payoff.p_win_shared(party_d_obs, party_r_obs, arrays).sum())
    cap_d = cap_fraction_d * budget_d
    msg_d_obs = gradients.msg_d(party_d_obs, party_r_obs, arrays)
    district_id = races[race_idx].district_id

    rows = []
    for delta in deltas:
        party_d_dev = _finance_delta(party_d_obs, msg_d_obs, race_idx, delta, cap_d)
        deployed = float(party_d_dev[race_idx] - party_d_obs[race_idx])
        capped = deployed < delta - 1.0  # $1 tolerance
        denom_m = max(deployed, 1.0) / 1e6  # avoid /0 when the race is already at cap

        v_uni = float(payoff.p_win_shared(party_d_dev, party_r_obs, arrays).sum()) - baseline_obs

        if use_surrogate:
            res_r_prime = brs.br_r_surrogate(
                races, coef, sigma_model, party_d=party_d_dev, cand_r_total=cand_r_total,
                budget_r=budget_r, cap_fraction_r=cap_fraction_r,
            )
        else:
            res_r_prime = br.br_r(
                races, coef, sigma_model, party_d=party_d_dev, cand_r_total=cand_r_total,
                budget_r=budget_r, cap_fraction_r=cap_fraction_r,
            )

        psv = float(payoff.p_win_shared(party_d_dev, res_r_prime.party, arrays).sum()) - baseline_d
        retention = float(psv / v_uni) if abs(v_uni) > RETENTION_MATERIALITY_THRESHOLD else float("nan")
        leverage = (psv / denom_m) if deployed > 1.0 else float("nan")

        displacement = res_r_prime.party - party_r_star
        reshuffle_l1 = float(np.abs(displacement).sum())
        reshuffle_per_million = (reshuffle_l1 / denom_m) if deployed > 1.0 else float("nan")
        cuts, adds = _top_movers(displacement, races, race_idx)

        rows.append(dict(
            district_id=district_id, delta=delta,
            delta_requested=delta, delta_deployed=deployed, capped=capped,
            V_uni=v_uni, PSV=psv, retention_rate=retention,
            leverage_seats_per_million=leverage,
            reshuffle_l1=reshuffle_l1, reshuffle_per_million=reshuffle_per_million,
            r_top_cuts=cuts, r_top_adds=adds,
            solver="exact" if not use_surrogate else "surrogate",
        ))
    return rows


def leverage_curve_r(races, coef, sigma_model, cand_r_total, budget_d, budget_r,
                      race_idx: int, deltas: list[float],
                      party_d_obs: np.ndarray, party_r_obs: np.ndarray, party_d_star: np.ndarray,
                      baseline_r: float, arrays: dict, n_races: int,
                      cap_fraction_d: float = 0.15, cap_fraction_r: float = 0.15,
                      use_surrogate: bool = True) -> list[dict]:
    """Mirror of leverage_curve_d: a Republican deviation at `race_idx`,
    financed from R's own lowest-MSG^R funded race, D best-responds.
    party_d_star must be BR_D(party_r_obs).party; baseline_r must be
    U_R(party_d_star, party_r_obs) = n - U_D(party_d_star, party_r_obs)."""
    baseline_d_obs = float(payoff.p_win_shared(party_d_obs, party_r_obs, arrays).sum())
    baseline_r_obs = float(n_races) - baseline_d_obs
    cap_r = cap_fraction_r * budget_r
    msg_r_obs = gradients.msg_r(party_d_obs, party_r_obs, arrays)
    district_id = races[race_idx].district_id

    rows = []
    for delta in deltas:
        party_r_dev = _finance_delta(party_r_obs, msg_r_obs, race_idx, delta, cap_r)
        deployed = float(party_r_dev[race_idx] - party_r_obs[race_idx])
        capped = deployed < delta - 1.0  # $1 tolerance
        denom_m = max(deployed, 1.0) / 1e6  # avoid /0 when the race is already at cap

        e_d_uni = float(payoff.p_win_shared(party_d_obs, party_r_dev, arrays).sum())
        v_uni = (float(n_races) - e_d_uni) - baseline_r_obs

        if use_surrogate:
            res_d_prime = brs.br_d_surrogate(
                races, coef, sigma_model, party_r=party_r_dev, cand_r_total=cand_r_total,
                budget_d=budget_d, cap_fraction_d=cap_fraction_d,
            )
        else:
            res_d_prime = br.br_d(
                races, coef, sigma_model, party_r=party_r_dev, cand_r_total=cand_r_total,
                budget_d=budget_d, cap_fraction_d=cap_fraction_d,
            )

        e_d_star = float(payoff.p_win_shared(res_d_prime.party, party_r_dev, arrays).sum())
        psv = (float(n_races) - e_d_star) - baseline_r
        retention = float(psv / v_uni) if abs(v_uni) > RETENTION_MATERIALITY_THRESHOLD else float("nan")
        leverage = (psv / denom_m) if deployed > 1.0 else float("nan")

        displacement = res_d_prime.party - party_d_star
        reshuffle_l1 = float(np.abs(displacement).sum())
        reshuffle_per_million = (reshuffle_l1 / denom_m) if deployed > 1.0 else float("nan")
        cuts, adds = _top_movers(displacement, races, race_idx)

        rows.append(dict(
            district_id=district_id, delta=delta,
            delta_requested=delta, delta_deployed=deployed, capped=capped,
            V_uni=v_uni, PSV=psv, retention_rate=retention,
            leverage_seats_per_million=leverage,
            reshuffle_l1=reshuffle_l1, reshuffle_per_million=reshuffle_per_million,
            d_top_cuts=cuts, d_top_adds=adds,
            solver="exact" if not use_surrogate else "surrogate",
        ))
    return rows
