"""
Job 2: what party money does to a race's margin.

    shift_i(x) = scale_pts * log(1 + x / (saturation_per_voter * N_i))

x is party dollars in race i and N_i its citizen voting-age population. The
effect depends on dollars per eligible voter, with diminishing returns, and
not on the race's share of total spending. That is the property the old
log-share term lacked: there, a dollar's effect scaled with 1 / (money
already in the race), so the first $100K in a quiet long-shot race bought
~10x the margin of $100K in a saturated toss-up.

This is an ASSUMPTION, not an estimate. The only causal evidence in this
repo (the repeat-challenger design) is identified almost entirely by
challengers spending under $250K, and loses significance as soon as those
tiny budgets are cushioned. Report results across a range of both
parameters, and replace the defaults with measured effects (e.g. randomized
mail holdouts) when available.

Defaults: scale_pts=1.5, saturation_per_voter=$10. In a typical district
(~550K CVAP) the first $1M moves the margin ~0.25 points, close to the old
model's marginal effect at a typical toss-up (beta_RC=5.475 with D = R =
$10M gives 0.27); the marginal effect halves by ~$5.5M, and $15M buys ~2
points in total.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpendingEffect:
    scale_pts: float = 1.5
    saturation_per_voter: float = 10.0

    def shift(self, dollars, voters) -> np.ndarray:
        """Margin points (D-positive) bought by `dollars` of party money."""
        dollars = np.maximum(np.asarray(dollars, dtype=float), 0.0)
        voters = np.maximum(np.asarray(voters, dtype=float), 1.0)
        return self.scale_pts * np.log1p(dollars / (self.saturation_per_voter * voters))
