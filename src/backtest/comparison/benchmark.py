"""
Benchmark comparisons: Brier score and four-way allocator comparison.

Compares model-implied win probabilities against:
  1. Cook Political Report converted probabilities
  2. FiveThirtyEight final pre-election probabilities (where available)

Also implements the null (equal-weight) allocator benchmark.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from ..types import RaceRecord, ModelOutputs, SigmaModel
from ..model.margin import MarginModelCoefficients
from ..optimizer.allocator import nonlinear_expected_seats_at_party_dollars
from .. import config

logger = logging.getLogger(__name__)


# ─── Brier score ──────────────────────────────────────────────────────────────

def brier_score(p_win: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Brier score = mean((p − outcome)²).
    outcomes: 1 if Democrat won, 0 if Republican won.
    """
    return float(np.mean((p_win - outcomes) ** 2))


def compute_brier_comparison(
    races: list[RaceRecord],
    model_outputs: list[ModelOutputs],
    fte_probs: dict[str, float] | None = None,
) -> dict:
    """
    Compute Brier scores for model, Cook, and (optionally) FiveThirtyEight.

    Parameters
    ----------
    fte_probs : district_id → D win probability, from FTE (where available)

    Returns dict: {model, cook, fte (if available), n_races}
    """
    cook_map = config.cook_win_probs()
    races_with_outcomes = [r for r in races if r.outcome is not None]
    outputs_with_outcomes = [o for r, o in zip(races, model_outputs) if r.outcome is not None]

    if not races_with_outcomes:
        logger.warning("No outcome data available — skipping Brier score computation")
        return {}

    outcomes = np.array([1.0 if r.outcome == "D" else 0.0 for r in races_with_outcomes])
    model_p = np.array([o.p_win for o in outputs_with_outcomes])
    cook_p = np.array([cook_map.get(r.cook_rating, 0.5) for r in races_with_outcomes])

    result = {
        "model": brier_score(model_p, outcomes),
        "cook": brier_score(cook_p, outcomes),
        "n_races": len(races_with_outcomes),
    }

    if fte_probs:
        fte_p = np.array([fte_probs.get(r.district_id, 0.5) for r in races_with_outcomes])
        result["fte"] = brier_score(fte_p, outcomes)

    for source, score in result.items():
        if source != "n_races":
            logger.info(f"Brier score ({source}): {score:.4f}")

    return result


# ─── Null allocator ───────────────────────────────────────────────────────────

def null_equal_weight_shares(races: list[RaceRecord]) -> np.ndarray:
    """
    Null benchmark: uniform allocation across competitive races, zero elsewhere.

    Returns array of length len(races) with shares summing to 1.
    """
    competitive = set(config.competitive_ratings())
    n_comp = sum(1 for r in races if r.cook_rating in competitive)
    if n_comp == 0:
        raise ValueError("No competitive races for null allocator")

    shares = np.array([
        1.0 / n_comp if r.cook_rating in competitive else 0.0
        for r in races
    ])
    return shares


def cook_proportional_shares(races: list[RaceRecord]) -> np.ndarray:
    """
    Cook-implied allocation: proportional to Cook win probability.
    Only allocates to competitive races.

    Returns array of length len(races) with shares summing to ≤ 1.
    """
    cook_map = config.cook_win_probs()
    competitive = set(config.competitive_ratings())

    raw = np.array([
        cook_map.get(r.cook_rating, 0.0) if r.cook_rating in competitive else 0.0
        for r in races
    ])
    total = raw.sum()
    return raw / total if total > 0 else raw


def expected_seats(p_win: np.ndarray) -> float:
    """E[Seats] = Σ P_win_i."""
    return float(np.sum(p_win))


def compare_allocators(
    races: list[RaceRecord],
    model_outputs: list[ModelOutputs],
    coef: MarginModelCoefficients,
    sigma_model: SigmaModel,
    model_shares: np.ndarray,
    null_shares: np.ndarray,
    cook_shares: np.ndarray,
    budget: float,
    party_budget: float,
    model_label: str = "Model optimizer",
    eta: float = 0.0,
) -> pd.DataFrame:
    """
    Four-way E[Seats] comparison table.

    Every row uses the true nonlinear Φ(μ/σ) evaluation
    (optimizer.allocator.nonlinear_expected_seats_at_party_dollars()), and
    every hypothetical row (Null, Cook-implied, Model) redistributes only
    the DCCC-controllable party_budget, holding every race's own
    candidate-committee floor fixed -- the only fair basis for comparing
    allocators against each other, since that is the actual budget any of
    them could really control. DCCC observed uses its real historical party
    spend instead (floors fixed, same as the others, but not a
    reallocation).

    This function went through two rounds of correction on 2026-07-22, both
    prompted by an anomalous 2022 OOS result where Null appeared to edge out
    the model optimizer (scripts/investigate_null_benchmark_bias.py):
      1. Originally used a linearized P_win⁰+MSG·Δspend approximation for
         Null/Cook while the Model row was patched post-hoc with the
         optimizer's own nonlinear result -- an inconsistency large enough
         to flip which allocator looked better in 2022.
      2. After fixing (1), Null/Cook were still scaled against the *entire*
         two-party spending pool across all 433 races -- including every
         candidate's own committee money in every safe seat -- while the
         Model optimizer only ever redistributed the DCCC-controllable party
         budget. This is the fix in this version: Null and Cook-implied now
         compete over the same $party_budget as the Model optimizer, never
         money DCCC does not control.

    Returns DataFrame with columns: allocator, expected_seats, description
    """
    floors = np.array([r.cand_d_total for r in races])
    observed_d = np.array([r.d_total for r in races])

    def true_seats(party_dollars: np.ndarray) -> float:
        return nonlinear_expected_seats_at_party_dollars(races, coef, sigma_model, party_dollars, eta=eta)

    dccc_party = np.maximum(observed_d - floors, 0.0)
    null_party = null_shares * party_budget
    cook_party = cook_shares * party_budget
    model_party = np.maximum(model_shares * budget - floors, 0.0)

    rows = [
        {"allocator": "DCCC observed",     "expected_seats": true_seats(dccc_party),
         "description": "Actual DCCC party+IE spend (floors fixed), true nonlinear P_win"},
        {"allocator": "Null (equal-weight)", "expected_seats": true_seats(null_party),
         "description": "DCCC party budget spread uniformly across competitive races (floors fixed)"},
        {"allocator": "Cook-implied",        "expected_seats": true_seats(cook_party),
         "description": "DCCC party budget proportional to Cook win probability (floors fixed)"},
        {"allocator": model_label,           "expected_seats": true_seats(model_party),
         "description": "DCCC party budget via marginal seat gain optimization (floors fixed)"},
    ]

    return pd.DataFrame(rows)


def _expected_seats_at_shares(
    races: list[RaceRecord],
    outputs: list[ModelOutputs],
    shares: np.ndarray,
) -> float:
    """
    Approximate E[Seats] by re-evaluating P_win at the scaled spending implied
    by the new shares.  Uses the linearized MSG approximation:
        P_win_i(new) ≈ P_win_i⁰ + MSG_i · (new_spend_i − observed_spend_i)

    No longer used by compare_allocators() or the permutation tests (both
    switched to optimizer.allocator.nonlinear_expected_seats_at_party_dollars()
    on 2026-07-22 after this approximation was found to bias several reported
    comparisons -- see that function's docstring for the full correction
    history). Retained only for scripts/investigate_null_benchmark_bias.py,
    which uses it deliberately to quantify the size of the bias by comparing
    against the true nonlinear evaluation side by side.
    """
    p_win0 = np.array([o.p_win for o in outputs])
    msg = np.array([o.msg_i for o in outputs])
    budget = sum(r.d_total for r in races)
    observed = np.array([r.d_total for r in races])
    new_spend = shares * budget
    delta = new_spend - observed
    p_win_new = np.clip(p_win0 + msg * delta, 0.0, 1.0)
    return float(p_win_new.sum())


def permutation_test_allocation_efficiency(
    races: list[RaceRecord],
    coef: MarginModelCoefficients,
    sigma_model: SigmaModel,
    model_shares: np.ndarray,
    n_permutations: int = 2000,
    rng: np.random.Generator | None = None,
    eta: float = 0.0,
) -> dict:
    """
    Permutation null distribution for DCCC's spending-to-race assignment.

    Randomly reassigns DCCC's observed per-race PARTY-DOLLAR amounts (its
    own coordinated + IE spend, not the candidate committee's own money)
    across competitive races, holding every race's own candidate-committee
    floor fixed. This is a direct robustness check on two distinct claims:

      1. Is DCCC's *actual* choice of which race gets which dollar amount
         worse than a random shuffle of its own dollars? This is a stronger,
         assumption-lighter claim than the rank-correlation test — it
         doesn't require interpreting a Spearman ρ, just comparing one real
         number against an empirical null.
      2. Is the model optimizer's seat gain over DCCC bigger than what a
         random reshuffle alone would produce? If most of the "+X seats"
         headline is achievable by literally any reshuffle of the same
         dollars (a consequence of the win-probability curve's concavity,
         not of MSG-based targeting), that is a real problem for the
         targeting claim, not just a footnote.

    Uses optimizer.allocator.nonlinear_expected_seats_at_party_dollars() --
    the true Φ(μ/σ) evaluation over the DCCC-controllable budget only, not
    the linearized P_win⁰+MSG·Δspend approximation in _expected_seats_at_shares()
    below. Two corrections were made to this function on 2026-07-22, both
    triggered by an anomalous 2022 OOS result in the sibling comparison
    (compare_allocators()):
      1. An earlier version used the linearized approximation for DCCC, the
         model, and every null draw; checked against the true nonlinear
         evaluation, P(random reshuffle ≥ DCCC) dropped from a reported 100%
         to 35.6% (2024) / 85.4% (2022) -- the model-side finding (0%) was
         unaffected in both cycles.
      2. That fix still reshuffled full observed d_total amounts, which
         include each race's own candidate-committee money -- not something
         DCCC actually decided. This version reshuffles only the
         party-controllable increment (d_total − cand_d_total), holding
         floors fixed, matching compare_allocators()'s same-day fix.
      See scripts/investigate_null_benchmark_bias.py for the diagnostics
      that surfaced both.

    Returns dict with: dccc_expected_seats, model_expected_seats,
    null_mean_expected_seats, null_ci_95, n_permutations, n_competitive,
    p_value_dccc_below_null (fraction of null ≥ DCCC's actual E[Seats]),
    p_value_model_exceeds_null (fraction of null ≥ the optimizer's E[Seats]),
    null_seats (the raw null distribution, for plotting -- not written to
    the JSON summary by callers)
    """
    rng = rng or np.random.default_rng(42)
    competitive = set(config.competitive_ratings())
    comp_idx = np.array([i for i, r in enumerate(races) if r.cook_rating in competitive])
    if len(comp_idx) == 0:
        raise ValueError("No competitive races for permutation test")

    floors = np.array([r.cand_d_total for r in races])
    observed_d = np.array([r.d_total for r in races])
    observed_party = np.maximum(observed_d - floors, 0.0)
    total_budget = observed_d.sum()

    def true_seats(party_dollars: np.ndarray) -> float:
        return nonlinear_expected_seats_at_party_dollars(races, coef, sigma_model, party_dollars, eta=eta)

    model_party = np.maximum(model_shares * total_budget - floors, 0.0)

    dccc_expected_seats = true_seats(observed_party)
    model_expected_seats = true_seats(model_party)

    null_seats = np.empty(n_permutations)
    permuted_party = observed_party.copy()
    for i in range(n_permutations):
        permuted_party[comp_idx] = rng.permutation(observed_party[comp_idx])
        null_seats[i] = true_seats(permuted_party)

    p_dccc_below_null = float(np.mean(null_seats >= dccc_expected_seats))
    p_model_exceeds_null = float(np.mean(null_seats >= model_expected_seats))

    logger.info(
        f"Allocation permutation test (n={n_permutations}, {len(comp_idx)} competitive races): "
        f"DCCC E[Seats]={dccc_expected_seats:.2f}, model E[Seats]={model_expected_seats:.2f}, "
        f"null mean={null_seats.mean():.2f}; "
        f"P(random reshuffle ≥ DCCC)={p_dccc_below_null:.3f}, "
        f"P(random reshuffle ≥ model)={p_model_exceeds_null:.3f}"
    )

    return {
        "dccc_expected_seats": dccc_expected_seats,
        "model_expected_seats": model_expected_seats,
        "null_mean_expected_seats": float(null_seats.mean()),
        "null_ci_95": [float(np.percentile(null_seats, 2.5)), float(np.percentile(null_seats, 97.5))],
        "n_permutations": n_permutations,
        "n_competitive": int(len(comp_idx)),
        "p_value_dccc_below_null": p_dccc_below_null,
        "p_value_model_exceeds_null": p_model_exceeds_null,
        "null_seats": null_seats,
    }
