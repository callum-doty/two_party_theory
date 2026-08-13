"""
Information option value: the cost of committing to your best-LOOKING
race under noisy date-t information about the national environment,
relative to committing to the TRULY best race (only fully knowable once
uncertainty resolves). Second half of Theta = information option value +
strategic flexibility option value -- game/strategic_window.py and
compute_value_of_waiting.py built the strategic-flexibility term (the
opponent's shrinking ability to respond); this module builds the term that
was explicitly left unmodeled there: the mover's OWN uncertainty about
which race is actually the best target, resolving as Election Day nears.

Deliberately does NOT reuse or reimplement Paper III's solve_bellman_lsm.py
(Longstaff-Schwartz optimal-stopping dynamic program) -- that machinery is
single-player, built for a specific 2026-live "close one discretionary
reserve at the optimal moment" decision, and carries its own considerable
validated-but-fragile history (the module's own docstring records an
allocator-degeneracy bug that silently flipped Theta's SIGN). Reusing it
here would mean adapting a large, delicate, single-player tool to a
two-player retrospective question it wasn't built for. Instead, this is a
much narrower, closed-form Monte Carlo that only needs the ALREADY-VALIDATED
historical generic-ballot volatility calibration (estimation.
gb_uncertainty, itself reusing Paper III's own realized-volatility
estimation, not refitting it) plus quantities this project's payoff model
already computes for free (V_uni is closed-form -- no best-response solve).

Mechanism: a committee standing at reference date t has a noisy read on the
national environment. The EVENTUAL (Election Day) generic ballot can still
differ from whatever a date-t committee would have inferred, by an amount
whose historical standard deviation is estimation.gb_uncertainty.
residual_gb_std(cycle, days_before). Simulating that noise as a shared,
race-invariant shock to every race's generic_ballot term (the margin model
enters it as `coef.alpha3 * generic_ballot`, additively and identically
across races -- a genuinely national-level shifter, not a race-specific
one) lets a committee's date-t belief about which of its candidate races is
BEST diverge from the true (final-data) ranking. Whichever race looks best
under the noisy belief is the one a real committee would have picked; this
module looks up that pick's TRUE, full-information value (computed once,
by the caller, from already-existing strategic_window.py/value_of_
waiting.py results) as the REALIZED outcome of a decision made under
imperfect information.

info_option_value = best_true_immediate - E[realized value of the noisy
pick] -- provably >= 0 in expectation IF the decision rule used under noise
is the SAME rule that defines "best_true_immediate" at zero noise. That
requirement is why best_true_immediate is defined as the PSV of whichever
race V_uni (the SAME decision proxy the noisy MC uses) ranks highest at
EXACTLY ZERO noise -- not the globally highest-PSV race. The first version
of this module used the latter and failed its own zero-noise sanity check:
V_uni (raw unilateral appeal) and PSV (value after the opponent's best
response) can rank candidate races DIFFERENTLY -- confirmed directly on
2024 D-side, where CT-02 has the highest V_uni but WI-01 has the highest
PSV, because CT-02's opponent response erodes much more of its raw appeal
(retention ~62%) than WI-01's does (~99%). That is itself a real, separate
finding (a V_uni-only targeting heuristic and a full game-theoretic PSV
calculation can disagree about which race is best) -- but it is NOT what
this module is trying to measure, and conflating it with the information
question would contaminate info_option_value with a fixed, noise-independent
offset. Anchoring to the zero-noise V_uni pick isolates the noise's own
marginal cost cleanly: info_option_value now measures "what does NOT
knowing the environment precisely cost, given a committee that uses this
same V_uni-based rule to decide either way" -- arguably the more realistic
framing regardless, since real committees making real-time decisions use
fast heuristics like MSG/persuadability more often than a full best-
response simulation against every candidate race.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from . import gradients, payoff
from .persistent_value import _finance_delta
from estimation.gb_uncertainty import residual_gb_std


def _noisy_best_pick(races, coef, sigma_model, cand_r_total,
                      party_d_obs: np.ndarray, party_r_obs: np.ndarray,
                      candidate_indices: list[int], delta: float, cap_d: float,
                      epsilon: float, side: str) -> int:
    """Which candidate race looks best (highest V_uni, closed-form, no
    best-response solve) under a shared generic-ballot shock epsilon.
    side='D': Democratic deviation, R held fixed. side='R': mirror."""
    perturbed = [dataclasses.replace(r, generic_ballot=r.generic_ballot - epsilon) for r in races]
    arrays = payoff.baseline_arrays(perturbed, coef, sigma_model, cand_r_total)
    baseline_d = float(payoff.p_win_shared(party_d_obs, party_r_obs, arrays).sum())
    n = len(races)

    best_idx, best_v = None, -np.inf
    if side == "D":
        msg_d = gradients.msg_d(party_d_obs, party_r_obs, arrays)
        for i in candidate_indices:
            dev = _finance_delta(party_d_obs, msg_d, i, delta, cap_d)
            v_uni = float(payoff.p_win_shared(dev, party_r_obs, arrays).sum()) - baseline_d
            if v_uni > best_v:
                best_v, best_idx = v_uni, i
    else:
        msg_r = gradients.msg_r(party_d_obs, party_r_obs, arrays)
        for i in candidate_indices:
            dev = _finance_delta(party_r_obs, msg_r, i, delta, cap_d)
            e_d_uni = float(payoff.p_win_shared(party_d_obs, dev, arrays).sum())
            v_uni = (float(n) - e_d_uni) - (float(n) - baseline_d)
            if v_uni > best_v:
                best_v, best_idx = v_uni, i
    return best_idx


def information_option_value(races, coef, sigma_model, cand_r_total,
                              party_d_obs: np.ndarray, party_r_obs: np.ndarray,
                              candidate_indices: list[int], delta: float,
                              cap_fraction: float, budget: float,
                              cycle: int, days_before: int, side: str,
                              psv_true_at_t: dict[int, float],
                              n_draws: int = 3000, seed: int = 0) -> dict:
    """Monte Carlo estimate of the information-option-value term at
    reference date `days_before` days before Election Day, for `side`'s
    delta among `candidate_indices`.

    psv_true_at_t: {race_idx: TRUE (unperturbed) PSV at this same
    reference date}, computed once by the caller from already-existing
    strategic_window.py/value_of_waiting.py results -- NOT recomputed
    here (no best-response solves happen in this module at all)."""
    cap = cap_fraction * budget
    sigma_eps = residual_gb_std(cycle, days_before)
    rng = np.random.default_rng(seed)

    # Anchor "best_true_immediate" to the ZERO-NOISE V_uni pick (the same
    # decision rule the noisy MC uses), not the globally highest-PSV race
    # -- see module docstring for why: this makes the zero-noise case an
    # exact fixed point (info_value=0 by construction) and isolates the
    # noise's own marginal cost from the separate V_uni-vs-PSV targeting
    # disagreement.
    zero_noise_pick = _noisy_best_pick(races, coef, sigma_model, cand_r_total,
                                        party_d_obs, party_r_obs, candidate_indices,
                                        delta, cap, epsilon=0.0, side=side)
    best_true_idx = zero_noise_pick
    best_true_immediate = psv_true_at_t[best_true_idx]

    picks = np.empty(n_draws, dtype=int)
    epsilons = rng.normal(0.0, sigma_eps, size=n_draws)
    for k, eps in enumerate(epsilons):
        picks[k] = _noisy_best_pick(races, coef, sigma_model, cand_r_total,
                                     party_d_obs, party_r_obs, candidate_indices,
                                     delta, cap, epsilon=float(eps), side=side)

    realized_values = np.array([psv_true_at_t[i] for i in picks])
    e_realized = float(realized_values.mean())
    info_value = best_true_immediate - e_realized

    pick_district_counts = {int(i): int((picks == i).sum()) for i in candidate_indices}

    # Reported separately from info_option_value, not folded into it: does
    # the V_uni-based decision rule even agree with the globally-best-PSV
    # race when there's no noise at all? A real, distinct finding (see
    # module docstring) -- a targeting heuristic based on raw unilateral
    # appeal can systematically favor a different race than the one that
    # actually survives the opponent's response best.
    global_best_psv_idx = max(candidate_indices, key=lambda i: psv_true_at_t[i])

    return dict(
        cycle=cycle, side=side, days_before=days_before, sigma_eps=sigma_eps, n_draws=n_draws,
        best_true_idx=int(best_true_idx), best_true_immediate=best_true_immediate,
        v_uni_rule_disagrees_with_psv_best=bool(zero_noise_pick != global_best_psv_idx),
        global_best_psv_idx=int(global_best_psv_idx), global_best_psv_value=psv_true_at_t[global_best_psv_idx],
        e_realized_value_under_noise=e_realized,
        info_option_value=info_value,
        pick_frequency=pick_district_counts,
    )
