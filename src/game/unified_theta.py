"""
Unified sequential decision value V_t(X_t) = max(V_deploy_t, V_wait_t)
(2026-08-14 follow-up to strategic_window.py / value_of_waiting.py /
information_value.py).

Those three modules built two USEFUL but not yet COMPARABLE diagnostics:
strategic_window.py/value_of_waiting.py rank candidate races by PSV (the
game-theoretic, opponent-best-response-adjusted value) when deciding what
"deploy now" means; information_value.py ranks the SAME candidates by V_uni
(a cheap, closed-form proxy) when simulating a noisy real-time decision,
because the zero-noise sanity check there found V_uni and PSV rank races
DIFFERENTLY (docs/methodology.md, "Information option value" section) --
2024 D-side: V_uni picks CT-02, PSV picks WI-01. Two components computed
under two different decision rules are two separate diagnostics, not two
halves of one Bellman value; summing them as Theta = Theta_info + Theta_flex
was flagged as premature for exactly this reason.

This module fixes that by using ONE decision rule at every reference date,
in every counterfactual, throughout: `deploy_value()`'s
`v_uni_noisy[i] * retention[i]` is a closed-form (no new best-response
solve) estimate of race i's PSV under a noisy generic-ballot signal --
V_uni's cheap noise-simulation machinery (information_value.py's
`_noisy_best_pick`, generalized here to return every candidate's value
instead of just the argmax) combined with retention_i(t), the TRUE
(zero-noise) PSV/V_uni ratio at date t already computed by
strategic_window.py. The committee picks the race with the highest
ESTIMATED PSV under its noisy signal; the realized payoff is that race's
TRUE PSV. This is a real approximation (retention itself is estimated at
zero noise and held fixed while V_uni is perturbed, rather than
re-solving BR_R under the noisy race arrays, which would cost a
full SLSQP solve per Monte Carlo draw -- documented as impractical at this
project's scale in commitment_timing.py/strategic_window.py's own
docstrings), but it is the SAME approximation applied identically
everywhere, which is the property the two-module version lacked.

State X_t = (opponent's true committed capital at t, from
commitment_timing.py's tiered curves -- deterministic, since this is
retrospective on realized data; generic-ballot noise at t, from
gb_uncertainty.residual_gb_std). V_deploy_t is this state's one-shot
payoff; V_wait_t = V_{t+1}(X_{t+1}) (deterministic continuation -- there is
no additional source of randomness in the transition beyond what
V_deploy_t and V_deploy_{t+1} already integrate over separately). Backward
induction over the reference-date grid already computed by
compute_strategic_window.py (120/90/60/45/30/21/14/7 days out) gives
V_t = max(V_deploy_t, V_{t+1}), with V at the final date forced to
V_deploy (matching strategic_window.py's own finding that the final week
is a mechanical 100%-retention floor, not a substantive continuation
target).

Theta_t = V_t - V_deploy_t decomposes, by holding one state variable fixed
at a time (the counterfactual-simulation method the project's next-phase
review recommended, rather than describing the two components as summable
by construction):

  - Theta_flex_only: freeze information (epsilon=0 always, i.e. the
    committee always picks the TRUE V_uni-best race) -- isolates the value
    of waiting for the opponent's capital to lock up, with no information
    channel at all.
  - Theta_info_only: freeze opponent commitments (retention_i held at its
    t=120-days-out, ~full-flexibility value for every later date, so PSV
    itself does not improve with delay) -- isolates the value of waiting
    for the committee's OWN uncertainty about which race is best to
    resolve, with no strategic-flexibility channel at all.
  - Theta_full: both channels active -- the actual quantity a real
    decision-maker faces.

Theta_full is NOT asserted to equal Theta_flex_only + Theta_info_only (the
two counterfactuals interact: which race looks best under noise also
depends on how much retention has grown), and the compute script reports
the gap explicitly rather than papering over it.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from . import gradients, payoff
from .persistent_value import _finance_delta


def noisy_v_uni_all(races, coef, sigma_model, cand_r_total,
                     party_d_obs: np.ndarray, party_r_obs: np.ndarray,
                     candidate_indices: list[int], delta: float, cap: float,
                     epsilon: float, side: str) -> dict[int, float]:
    """V_uni for EVERY candidate race under a shared generic-ballot shock
    epsilon (closed form, no best-response solve) -- generalizes
    information_value.py's `_noisy_best_pick` (which only returns the
    argmax) so a caller can combine each candidate's noisy V_uni with a
    separate, date-varying retention multiplier instead of picking under
    V_uni alone."""
    perturbed = [dataclasses.replace(r, generic_ballot=r.generic_ballot - epsilon) for r in races]
    arrays = payoff.baseline_arrays(perturbed, coef, sigma_model, cand_r_total)
    baseline_d = float(payoff.p_win_shared(party_d_obs, party_r_obs, arrays).sum())
    n = len(races)
    out: dict[int, float] = {}
    if side == "D":
        msg = gradients.msg_d(party_d_obs, party_r_obs, arrays)
        for i in candidate_indices:
            dev = _finance_delta(party_d_obs, msg, i, delta, cap)
            out[i] = float(payoff.p_win_shared(dev, party_r_obs, arrays).sum()) - baseline_d
    else:
        msg = gradients.msg_r(party_d_obs, party_r_obs, arrays)
        for i in candidate_indices:
            dev = _finance_delta(party_r_obs, msg, i, delta, cap)
            e_d = float(payoff.p_win_shared(party_d_obs, dev, arrays).sum())
            out[i] = (float(n) - e_d) - (float(n) - baseline_d)
    return out


def deploy_value(candidate_indices: list[int], v_uni_noisy: dict[int, float],
                  retention: dict[int, float], psv_true: dict[int, float]) -> tuple[int, float]:
    """The ONE decision rule this module uses everywhere: pick the race
    with the highest estimated PSV under the noisy signal
    (v_uni_noisy[i] * retention[i]), then realize that race's TRUE PSV
    (psv_true, no noise) as the payoff. NaN/missing retention (
    persistent_value.RETENTION_MATERIALITY_THRESHOLD -- no material
    unilateral opportunity to begin with) is treated as 0, not skipped, so
    such a race is simply never picked rather than crashing the max()."""
    best_idx, best_est = None, -np.inf
    for i in candidate_indices:
        r = retention.get(i, 0.0)
        r = r if np.isfinite(r) else 0.0
        est = v_uni_noisy[i] * max(r, 0.0)
        if est > best_est:
            best_est, best_idx = est, i
    return best_idx, psv_true[best_idx]


def expected_deploy_value(candidate_indices: list[int], psv_true: dict[int, float],
                           retention: dict[int, float], rng: np.random.Generator,
                           sigma_eps: float, v_uni_sampler, n_draws: int) -> float:
    """E[realized TRUE PSV of the noisy-signal pick] at one reference
    date. v_uni_sampler(epsilon) -> {idx: V_uni}, closed form (see
    noisy_v_uni_all) -- no best-response solve inside the Monte Carlo
    loop, matching information_value.py's cost profile."""
    if sigma_eps <= 0.0:
        _, realized = deploy_value(candidate_indices, v_uni_sampler(0.0), retention, psv_true)
        return realized
    total = 0.0
    for eps in rng.normal(0.0, sigma_eps, size=n_draws):
        v_uni_noisy = v_uni_sampler(float(eps))
        _, realized = deploy_value(candidate_indices, v_uni_noisy, retention, psv_true)
        total += realized
    return total / n_draws


def solve_bellman(dates_chronological: list[str], deploy_value_by_date: dict[str, float]) -> dict:
    """Backward induction, dates_chronological ordered earliest (most
    opponent flexibility) -> latest (closest to Election Day).
    V[last] = deploy_value[last] (no further waiting option once at the
    final reference date). V[k] = max(deploy_value[k], V[k+1]) for
    earlier k. Theta_t at the FIRST date = V[0] - deploy_value[0], the
    unified value of waiting from the initial decision point."""
    n = len(dates_chronological)
    v = [0.0] * n
    v[-1] = deploy_value_by_date[dates_chronological[-1]]
    for k in range(n - 2, -1, -1):
        d = dates_chronological[k]
        v[k] = max(deploy_value_by_date[d], v[k + 1])
    deploy_seq = [deploy_value_by_date[d] for d in dates_chronological]
    theta_t = v[0] - deploy_seq[0]
    return dict(V=v, deploy_value=deploy_seq, theta_t0=theta_t)
