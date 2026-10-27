"""
Combine the two jobs and spend a budget.

    p_i(x) = Phi((mu_i + f_i(x_i) - f_i(x_obs_i)) / sigma)

mu_i is the baseline margin forecast, which already reflects the money that
was actually spent (x_obs), so only DEVIATIONS from the observed allocation
move a race. At x = x_obs every race is exactly at its baseline forecast.

The opponent's allocation is held fixed. allocate() is greedy: each step
funds whichever race buys the most expected seats per dollar from its next
chunk. Several chunk sizes are tried per step so races whose payoff only
appears after a larger buy (the convex part of Phi, below 50%) are not
starved by looking one small step ahead.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .spending import SpendingEffect

CHUNK_MULTIPLES = (1, 2, 5, 10, 20, 50)


def win_prob(mu, sigma, effect: SpendingEffect, voters, party, party_obs) -> np.ndarray:
    shift = effect.shift(party, voters) - effect.shift(party_obs, voters)
    return norm.cdf((np.asarray(mu) + shift) / sigma)


def allocate(mu, sigma, effect: SpendingEffect, voters, party_obs,
             budget: float, cap: float, step: float = 100_000, start=None) -> np.ndarray:
    """Greedy allocation of `budget` on top of `start` (default: nothing),
    adding at most `cap` per race, in multiples of `step`. Returns the total
    allocation (start + added)."""
    mu, voters, party_obs = (np.asarray(a, dtype=float) for a in (mu, voters, party_obs))
    start = np.zeros_like(mu) if start is None else np.asarray(start, dtype=float)
    x = start.copy()
    remaining = float(budget)
    p = win_prob(mu, sigma, effect, voters, x, party_obs)
    while remaining >= step:
        best_rate, best_i, best_amt = 0.0, -1, 0.0
        room = cap - (x - start)
        for k in CHUNK_MULTIPLES:
            amt = k * step
            if amt > remaining:
                break
            ok = room >= amt
            if not ok.any():
                continue
            gain = win_prob(mu, sigma, effect, voters, x + amt, party_obs) - p
            rate = np.where(ok, gain / amt, -np.inf)
            i = int(np.argmax(rate))
            if rate[i] > best_rate:
                best_rate, best_i, best_amt = rate[i], i, amt
        if best_i < 0:
            break
        x[best_i] += best_amt
        remaining -= best_amt
        p[best_i] = win_prob(mu[best_i:best_i + 1], sigma, effect, voters[best_i:best_i + 1],
                             x[best_i:best_i + 1], party_obs[best_i:best_i + 1])[0]
    return x
