"""
Level D benchmark strategies (project_spec.md Section 20): "Are observed
allocations closer to the estimated Nash equilibrium than to reasonable
alternative strategies? Compare observed spending with: equal allocation;
Cook-category heuristic; one-sided optimizer; Nash equilibrium; random
feasible portfolios."

Each function here builds ONE side's feasible allocation vector (respects
the same per-race cap and total-budget constraint as `game/best_response.py`
et al.) for one of the non-optimized benchmark strategies -- "Nash
equilibrium"/"one-sided optimizer" are built directly from existing solvers
(`game/best_response.py`, `game/double_oracle.py`) in
`scripts/level_d_benchmark.py`, not duplicated here.
"""

from __future__ import annotations

import numpy as np

_COOK_CATEGORY_WEIGHT = {
    "Toss-Up": 5.0,
    "Lean": 3.0,
    "Likely": 1.0,
    "Safe": 0.0,
}


def cap_and_redistribute(weights: np.ndarray, budget: float, cap: float, max_iter: int = 100) -> np.ndarray:
    """Scale `weights` to sum to `budget`, then iteratively clip any race
    exceeding the per-race `cap` and redistribute the excess proportionally
    among still-uncapped races -- a fixed-proportions analogue of
    water-filling for benchmark strategies that aren't themselves the
    output of an optimizer (equal/Cook/random all route through this)."""
    w = np.asarray(weights, dtype=float)
    n = len(w)
    if budget <= 0 or n == 0:
        return np.zeros(n)
    if w.sum() <= 0:
        return np.zeros(n)
    alloc = w / w.sum() * budget
    fixed = np.zeros(n, dtype=bool)
    for _ in range(max_iter):
        over = (~fixed) & (alloc > cap + 1e-9)
        if not over.any():
            break
        alloc[over] = cap
        fixed |= over
        remaining_budget = budget - alloc[fixed].sum()
        free = ~fixed
        if not free.any() or remaining_budget <= 0:
            alloc[free] = 0.0
            break
        free_w = w[free]
        if free_w.sum() <= 0:
            alloc[free] = 0.0
            break
        alloc[free] = free_w / free_w.sum() * remaining_budget
    return np.clip(alloc, 0.0, cap)


def equal_allocation(n: int, budget: float) -> np.ndarray:
    """Uniform allocation across all n races -- spec's "equal allocation"
    benchmark. No cap needed in practice (budget/n is far below any
    realistic cap_fraction*budget for this project's race counts), but
    included for consistency with the other benchmark builders."""
    return np.full(n, budget / n) if n else np.zeros(0)


def cook_category_weight(cook_rating: str) -> float:
    """Strip the D/R suffix (e.g. "Lean D" -> "Lean") and look up a fixed
    competitiveness weight -- SAME weight regardless of which party the
    race currently favors, matching a stylized "spend where it's
    competitive" committee heuristic (spec's "Cook-category heuristic"),
    applied identically to both D and R side allocators (see
    cook_heuristic_allocation)."""
    category = cook_rating.rsplit(" ", 1)[0] if " " in cook_rating else cook_rating
    return _COOK_CATEGORY_WEIGHT.get(category, 0.0)


def cook_heuristic_allocation(races, budget: float, cap_fraction: float) -> np.ndarray:
    """Allocate proportional to each race's Cook-category weight (Toss-Up
    highest, Safe zero), capped per race at cap_fraction*budget and
    redistributed -- a real committee doesn't spend on safe seats and
    concentrates on toss-ups, without needing to know which party a race
    currently favors (spec's "Cook-category heuristic," Section 20)."""
    weights = np.array([cook_category_weight(r.cook_rating) for r in races])
    cap = cap_fraction * budget
    return cap_and_redistribute(weights, budget, cap)


def random_feasible_allocation(n: int, budget: float, cap_fraction: float,
                                rng: np.random.Generator) -> np.ndarray:
    """One random feasible portfolio: Dirichlet(1,...,1) shares of the
    budget, capped per race and redistributed -- spec's "random feasible
    portfolios" benchmark. Callers should average over several draws
    (`scripts/level_d_benchmark.py` uses K=20) rather than reading a single
    draw as representative."""
    weights = rng.dirichlet(np.ones(n)) if n else np.zeros(0)
    cap = cap_fraction * budget
    return cap_and_redistribute(weights, budget, cap)
