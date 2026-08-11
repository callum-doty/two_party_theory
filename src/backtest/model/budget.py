"""
Derive BUDGET_2026 (Paper II Section 7.1's live-cycle party budget input)
from real data plus a documented BLS CPI-U projection, instead of a
hand-typed literal.

Previously this entire derivation existed only as prose in
docs/paper2_draft.md Section 7.1, with the result ($394.3M) hardcoded as an
independent literal in scripts/plot_2026_live_allocation.py -- no code path
connected the two. See config.yaml's `budget_2026_projection:` block for the
raw CPI index values (genuine external BLS inputs this repo does not itself
fetch, and therefore the one part of this derivation that must be
hand-updated when a newer CPI release changes them).
"""

from __future__ import annotations
import numpy as np

from .. import config
from ..data.universe import build_universe


def party_controlled_budget(cycle: int) -> float:
    """Total DCCC-controllable spend for a cycle's universe: total D spend
    minus each race's non-discretionary candidate-committee floor. Same
    formula scripts/run_backtest.py uses for its headline per-cycle figure
    (e.g. 2024's $465.0M) -- reused here so the 2018/2022 base budgets below
    are computed the same way, not re-typed from a paper's stated result."""
    races = build_universe(cycle=cycle)
    return float(sum(r.d_total - r.cand_d_total for r in races))


def estimate_budget_2026() -> float:
    """Average the party-controlled budgets of config.yaml's
    `budget_2026_projection.base_cycles` (2018, 2022), each inflated from its
    own November CPI-U reading to a projected November-2026 reading (trailing
    year-over-year rate applied to the most recent available CPI print)."""
    cfg = config.budget_2026_projection_cfg()
    cpi_by_cycle = {2018: cfg["cpi_nov_2018"], 2022: cfg["cpi_nov_2022"]}
    cpi_nov_2026 = cfg["cpi_nov_2025"] * (1.0 + cfg["trailing_yoy_rate"])

    inflated = []
    for cycle in cfg["base_cycles"]:
        base_budget = party_controlled_budget(cycle)
        inflated.append(base_budget * (cpi_nov_2026 / cpi_by_cycle[cycle]))
    return float(np.mean(inflated))
