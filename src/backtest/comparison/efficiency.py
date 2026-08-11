"""
Efficiency tests: rank correlation and misallocation characterization.

Primary test: Spearman ρ between MSG_i and observed spending rank across
competitive races. Under efficient DCCC allocation this should be strongly
positive. A weak or negative ρ is evidence of systematic misallocation.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy import stats
from ..types import ModelOutputs, RaceRecord, SigmaModel
from ..model.margin import MarginModelCoefficients
from ..model.win_prob import compute_floor_msg
from .. import config

logger = logging.getLogger(__name__)


def spearman_efficiency_test(
    races: list[RaceRecord],
    outputs: list[ModelOutputs],
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Compute Spearman ρ between observed spending rank and MSG_i rank,
    restricted to competitive races.

    Returns dict with: rho, p_value, ci_low, ci_high, n_competitive
    """
    rng = rng or np.random.default_rng(42)
    competitive = set(config.competitive_ratings())

    pairs = [
        (r, o) for r, o in zip(races, outputs)
        if r.cook_rating in competitive
    ]

    if not pairs:
        raise ValueError("No competitive races found for efficiency test")

    comp_races, comp_outputs = zip(*pairs)
    observed_spend = np.array([r.d_total for r in comp_races])
    msg_vals = np.array([o.msg_i for o in comp_outputs])

    rho, p_value = stats.spearmanr(observed_spend, msg_vals)
    logger.info(f"Spearman ρ (competitive, n={len(comp_races)}): {rho:.3f} (p={p_value:.4f})")

    # Bootstrap CI
    bootstrap_rhos = []
    n = len(comp_races)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        r_boot, _ = stats.spearmanr(observed_spend[idx], msg_vals[idx])
        bootstrap_rhos.append(r_boot)

    ci_low = float(np.percentile(bootstrap_rhos, 2.5))
    ci_high = float(np.percentile(bootstrap_rhos, 97.5))

    return {
        "rho": float(rho),
        "p_value": float(p_value),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_competitive": len(comp_races),
    }


def permutation_test_spearman_efficiency(
    races: list[RaceRecord],
    outputs: list[ModelOutputs],
    n_permutations: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Permutation test for the primary Spearman efficiency correlation.

    scipy.stats.spearmanr's p-value relies on an asymptotic approximation
    that is untested at the small-n categories this project already reports
    (e.g. Lean R, n=7, in spearman_by_cook_category()). This instead builds
    an exact empirical null: randomly reassign DCCC's observed per-race
    spending amounts across competitive races — breaking any link between
    spending and MSG while holding the multiset of dollar amounts and MSG
    values fixed — and recompute ρ, n_permutations times. The permutation
    p-value is the fraction of null |ρ| at least as extreme as the observed
    |ρ|, with no distributional assumption.

    Returns dict with: rho, p_value_asymptotic, p_value_permutation,
    n_permutations, n_competitive, null_rhos (the raw null distribution,
    for plotting -- not written to the JSON summary by callers)
    """
    rng = rng or np.random.default_rng(42)
    competitive = set(config.competitive_ratings())

    pairs = [
        (r, o) for r, o in zip(races, outputs)
        if r.cook_rating in competitive
    ]
    if not pairs:
        raise ValueError("No competitive races found for efficiency test")

    comp_races, comp_outputs = zip(*pairs)
    observed_spend = np.array([r.d_total for r in comp_races])
    msg_vals = np.array([o.msg_i for o in comp_outputs])

    rho_obs, p_asymptotic = stats.spearmanr(observed_spend, msg_vals)

    null_rhos = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled_spend = rng.permutation(observed_spend)
        null_rhos[i], _ = stats.spearmanr(shuffled_spend, msg_vals)

    p_permutation = float(np.mean(np.abs(null_rhos) >= abs(rho_obs)))

    logger.info(
        f"Permutation test (n={n_permutations}): ρ={rho_obs:.3f}, "
        f"asymptotic p={p_asymptotic:.4g}, permutation p={p_permutation:.4g}"
    )

    return {
        "rho": float(rho_obs),
        "p_value_asymptotic": float(p_asymptotic),
        "p_value_permutation": p_permutation,
        "n_permutations": n_permutations,
        "n_competitive": len(comp_races),
        "null_rhos": null_rhos,
    }


def spearman_by_cook_category(
    races: list[RaceRecord],
    outputs: list[ModelOutputs],
    categories: tuple[str, ...] = ("Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R"),
) -> pd.DataFrame:
    """
    Spearman ρ between observed spending and MSG_i, computed separately
    within each Cook rating category (Paper I Table 2).

    Unlike spearman_efficiency_test(), this is not restricted to
    config.competitive_ratings() — it reports every category in `categories`
    so that Likely D/Likely R (outside the primary n=53/61 competitive set)
    are included alongside Lean D/Toss-Up/Lean R.
    """
    rows = []
    for cat in categories:
        pairs = [(r, o) for r, o in zip(races, outputs) if r.cook_rating == cat]
        if len(pairs) < 3:
            continue
        cat_races, cat_outputs = zip(*pairs)
        spend = np.array([r.d_total for r in cat_races])
        msg_vals = np.array([o.msg_i for o in cat_outputs])
        rho, p_value = stats.spearmanr(spend, msg_vals)
        rows.append({"cook_category": cat, "n": len(pairs), "rho": float(rho), "p_value": float(p_value)})
    return pd.DataFrame(rows)


def matched_group_efficiency_test(
    races: list[RaceRecord],
    outputs: list[ModelOutputs],
    categories: tuple[str, ...] = ("Lean D", "Toss-Up"),
    max_abs_pvi: float = 5.0,
) -> dict:
    """
    Spearman ρ restricted to races matched on partisan lean and Cook
    category (Paper I §9, "matched-group test") — the risk-tolerance-robust
    efficiency test of §3.3: within races with similar factor loadings,
    γ·∂Var[Seats]/∂sᵢ is approximately constant, so equalization of raw MSG
    is the relevant efficiency condition.
    """
    pairs = [
        (r, o) for r, o in zip(races, outputs)
        if r.cook_rating in categories and abs(r.pvi) <= max_abs_pvi
    ]
    if not pairs:
        raise ValueError("No races found for matched-group test")
    m_races, m_outputs = zip(*pairs)
    spend = np.array([r.d_total for r in m_races])
    msg_vals = np.array([o.msg_i for o in m_outputs])
    rho, p_value = stats.spearmanr(spend, msg_vals)
    return {"rho": float(rho), "p_value": float(p_value), "n": len(pairs)}


def floor_baseline_efficiency_test(
    races: list[RaceRecord],
    coef: MarginModelCoefficients,
    sigma_model: SigmaModel,
    beta1_override: float | None = None,
) -> dict:
    """
    Efficiency test #2 (replacement design, see FINDINGS.md): Spearman
    correlation between observed DCCC party spending and MSG evaluated at
    each race's own candidate-only floor (model.win_prob.compute_floor_msg),
    restricted to competitive races.

    This is the direct fix for the diminishing-returns confound in
    spearman_efficiency_test(): that function correlates observed spending
    against MSG evaluated *at that same observed spending level*, which is
    mechanically biased toward a negative reading regardless of targeting
    quality, because more spending always depresses a race's own current-MSG
    reading under a concave response curve. Floor-baseline MSG is fixed
    before any party dollar is committed, so it cannot be contaminated by
    the spending decision under test -- a positive correlation here means
    the party actually sent more money toward races that had higher
    pre-allocation marginal return; a negative correlation is genuine
    evidence of misallocation, not an artifact of the test's own mechanics.
    """
    competitive = set(config.competitive_ratings())
    comp = [r for r in races if r.cook_rating in competitive]
    if not comp:
        raise ValueError("No competitive races found for floor-baseline efficiency test")

    party_spend = np.array([r.d_total - r.cand_d_total for r in comp])
    msg_floor = np.array([
        compute_floor_msg(r, coef, sigma_model, beta1_override) for r in comp
    ])
    rho, p_value = stats.spearmanr(party_spend, msg_floor)
    logger.info(
        f"Floor-baseline efficiency test (n={len(comp)}): ρ={rho:.3f} (p={p_value:.4f})"
    )
    return {"rho": float(rho), "p_value": float(p_value), "n": len(comp)}


def kkt_dispersion_test(
    races: list[RaceRecord],
    outputs: list[ModelOutputs],
    optimal_party_allocations: np.ndarray,
    optimal_msg: np.ndarray,
    party_budget: float,
    cap_fraction: float,
) -> dict:
    """
    Efficiency test #1 (replacement design, see FINDINGS.md and Appendix
    C.1's KKT derivation): direct test of the paper's own stationarity
    condition rather than a correlation-sign proxy for it.

    At a risk-neutral interior optimum, MSG_i = lambda (constant) for every
    race with strictly-interior party funding (Appendix C.1) -- efficient
    allocation *equalizes* marginal seat gain among funded races, it does
    not produce any particular correlation between spending and MSG. This
    test computes the dispersion (coefficient of variation) of MSG among
    interior-funded races under (a) DCCC's observed allocation and (b) the
    model-optimal allocation, which by construction should show near-zero
    dispersion (bounded only by SLSQP's convergence tolerance). Large
    dispersion in (a) relative to (b) is direct evidence that DCCC's
    allocation violates the necessary first-order condition for efficiency
    -- no assumption about correlation sign is required.

    Parameters
    ----------
    outputs                     : ModelOutputs at DCCC's *observed* spending
                                   (compute_outputs_batch's default behavior)
    optimal_party_allocations   : (n_races,) model-optimal party $ per race
                                   (OptimizerResult.allocations - floors)
    optimal_msg                 : (n_races,) MSG at the model-optimal
                                   allocation (e.g. allocator._msg_vec(...)
                                   evaluated at optimal_party_allocations)
    party_budget, cap_fraction  : same constraint parameters the optimizer
                                   was run with, used to define "interior"
                                   consistently for both allocations
    """
    cap = cap_fraction * party_budget
    tol = 1e-3 * party_budget

    party_obs = np.array([r.d_total - r.cand_d_total for r in races])
    msg_obs = np.array([o.msg_i for o in outputs])
    interior_obs = (party_obs > tol) & (party_obs < cap - tol)

    interior_opt = (optimal_party_allocations > tol) & (optimal_party_allocations < cap - tol)

    def _dispersion(msg_vals: np.ndarray, tol_frac: float = 0.25) -> dict:
        """
        Dispersion of MSG around an *implied* shadow price lambda, estimated
        empirically as the median MSG within the group (no internal ledger
        or Lagrange multiplier is observable for DCCC's real allocation, so
        the same empirical estimator is used for both DCCC and the model-
        optimal allocation for comparability). Reports several statistics
        beyond CV, since CV alone can be unstable when the mean is small:
        MAD from lambda, IQR, the p90/p10 ratio, and the fraction of races
        outside a +/-tol_frac band around lambda.
        """
        n = int(len(msg_vals))
        if n < 2:
            return {"n": n, "mean": float(msg_vals[0]) if n else float("nan"),
                    "std": float("nan"), "cv": float("nan"), "lambda_implied": float("nan"),
                    "mad_from_lambda": float("nan"), "iqr": float("nan"),
                    "p90_p10_ratio": float("nan"), "frac_outside_tolerance": float("nan")}
        mean = float(np.mean(msg_vals))
        std = float(np.std(msg_vals))
        cv = float(std / mean) if mean != 0 else float("nan")
        lam = float(np.median(msg_vals))
        mad = float(np.median(np.abs(msg_vals - lam)))
        q25, q75 = np.percentile(msg_vals, [25, 75])
        iqr = float(q75 - q25)
        p10, p90 = np.percentile(msg_vals, [10, 90])
        p_ratio = float(p90 / p10) if p10 > 0 else float("inf")
        band = tol_frac * lam
        frac_outside = float(np.mean(np.abs(msg_vals - lam) > band)) if lam != 0 else float("nan")
        return {
            "n": n, "mean": mean, "std": std, "cv": cv,
            "lambda_implied": lam, "mad_from_lambda": mad, "iqr": iqr,
            "p90_p10_ratio": p_ratio, "frac_outside_tolerance": frac_outside,
            "tolerance_frac": tol_frac,
        }

    result = {
        "dccc_observed": _dispersion(msg_obs[interior_obs]),
        "model_optimal": _dispersion(optimal_msg[interior_opt]),
        "cap_fraction": cap_fraction,
        "party_budget": party_budget,
    }
    logger.info(
        f"KKT dispersion test: DCCC interior n={result['dccc_observed']['n']}, "
        f"CV={result['dccc_observed']['cv']:.3f}  |  "
        f"Model interior n={result['model_optimal']['n']}, "
        f"CV={result['model_optimal']['cv']:.3f}"
    )
    return result


def boundary_kkt_test(
    races: list[RaceRecord],
    outputs: list[ModelOutputs],
    party_budget: float,
    cap_fraction: float,
    lambda_implied: float,
    tol_frac: float = 0.25,
) -> dict:
    """
    Complementary-slackness (boundary) counterpart to kkt_dispersion_test()'s
    interior-race check, applied to DCCC's observed allocation.

    At a true KKT optimum: a race pinned at its lower bound (zero party
    funding) must have MSG_i <= lambda (no incentive to fund it further); a
    race pinned at its upper bound (the cap) must have MSG_i >= lambda (it
    would be funded even more if the cap were relaxed). Violations of either
    inequality in DCCC's *observed* allocation are races the model says are
    priced wrong at the boundary, not just among the interior-funded set --
    "should have gotten some money but got none" (zero-funded, high MSG) or
    "already maxed out but its marginal value is still low" (cap-funded,
    low MSG; rare in practice since DCCC has never observably concentrated
    that heavily in a single race).

    lambda_implied: the empirical shadow-price estimate from
    kkt_dispersion_test()'s DCCC-observed interior group (its
    "lambda_implied" field), reused here for a single consistent lambda
    across the interior and boundary checks.
    """
    cap = cap_fraction * party_budget
    tol = 1e-3 * party_budget
    band = tol_frac * lambda_implied

    party_obs = np.array([r.d_total - r.cand_d_total for r in races])
    msg_obs = np.array([o.msg_i for o in outputs])

    at_zero = party_obs <= tol
    at_cap = party_obs >= cap - tol

    zero_violations = at_zero & (msg_obs > lambda_implied + band)
    cap_violations = at_cap & (msg_obs < lambda_implied - band)

    result = {
        "lambda_implied": float(lambda_implied),
        "tolerance_frac": tol_frac,
        "n_at_zero": int(at_zero.sum()),
        "n_zero_violations": int(zero_violations.sum()),
        "zero_violation_districts": [r.district_id for r, v in zip(races, zero_violations) if v],
        "n_at_cap": int(at_cap.sum()),
        "n_cap_violations": int(cap_violations.sum()),
        "cap_violation_districts": [r.district_id for r, v in zip(races, cap_violations) if v],
    }
    logger.info(
        f"Boundary KKT test: {result['n_zero_violations']}/{result['n_at_zero']} "
        f"zero-funded races exceed λ (should have received funding); "
        f"{result['n_cap_violations']}/{result['n_at_cap']} cap-level races fall below λ"
    )
    return result


def pairwise_transfer_gain(
    races: list[RaceRecord],
    coef: MarginModelCoefficients,
    sigma_model: SigmaModel,
    outputs: list[ModelOutputs],
    party_budget: float,
    cap_fraction: float,
    transfer_amount: float = 100_000.0,
    eta: float = 0.0,
) -> dict:
    """
    Concrete, single-swap illustration of the KKT dispersion finding: move
    `transfer_amount` dollars from DCCC's lowest-MSG interior-funded race to
    its highest-MSG race with remaining capacity under the cap, and report
    the resulting change in true nonlinear E[Seats]. Turns an abstract "MSG
    is not equalized" statistic into a single, immediately interpretable
    reallocation and its seat consequence.

    Uses the same true nonlinear Phi(mu(D)/sigma) evaluation (with the
    persuasion ceiling applied) as every other headline figure in this
    project, via optimizer.allocator's internal per-race arrays -- not a
    linear MSG approximation, so the reported gain is exact for this single
    swap, not a first-order estimate.
    """
    from ..optimizer import allocator as _allocator  # local import: avoid module-level cycle

    cap = cap_fraction * party_budget
    tol = 1e-3 * party_budget

    party_obs = np.array([r.d_total - r.cand_d_total for r in races])
    msg_obs = np.array([o.msg_i for o in outputs])
    interior = (party_obs > tol) & (party_obs < cap - tol)
    has_capacity = party_obs < cap - transfer_amount

    if not interior.any() or not has_capacity.any():
        raise ValueError("No eligible donor/recipient races for a pairwise transfer")

    donor_idx = int(np.where(interior, msg_obs, np.inf).argmin())
    # Recipient: highest-MSG race with room to receive the transfer, excluding the donor itself.
    recipient_pool = has_capacity.copy()
    recipient_pool[donor_idx] = False
    recipient_idx = int(np.where(recipient_pool, msg_obs, -np.inf).argmax())

    arrays = _allocator._precompute_race_arrays(races, coef, sigma_model, eta=eta)
    party_before = party_obs.copy()
    e_seats_before = float(_allocator._p_win_vec(party_before, arrays).sum())

    party_after = party_before.copy()
    party_after[donor_idx] = max(party_after[donor_idx] - transfer_amount, 0.0)
    party_after[recipient_idx] = party_after[recipient_idx] + transfer_amount
    e_seats_after = float(_allocator._p_win_vec(party_after, arrays).sum())

    result = {
        "transfer_amount": transfer_amount,
        "donor_district": races[donor_idx].district_id,
        "donor_msg": float(msg_obs[donor_idx]),
        "recipient_district": races[recipient_idx].district_id,
        "recipient_msg": float(msg_obs[recipient_idx]),
        "expected_seats_before": e_seats_before,
        "expected_seats_after": e_seats_after,
        "delta_expected_seats": e_seats_after - e_seats_before,
    }
    logger.info(
        f"Pairwise transfer: moving ${transfer_amount:,.0f} from "
        f"{result['donor_district']} (MSG={result['donor_msg']:.2e}) to "
        f"{result['recipient_district']} (MSG={result['recipient_msg']:.2e}) "
        f"changes E[Seats] by {result['delta_expected_seats']:+.5f}"
    )
    return result


def characterize_misallocation(
    races: list[RaceRecord],
    outputs: list[ModelOutputs],
    allocation_diffs: list[float],
    budget: float,
) -> dict:
    """
    For races with material allocation divergence (|diff| > 1% of budget),
    characterize the direction by cook_rating, PVI, and incumbency status.

    Returns dict with "overfunded" and "underfunded" summaries.
    """
    threshold = config.outputs_cfg()["material_divergence_threshold"]
    material = threshold * budget

    over, under = [], []
    for race, out, diff in zip(races, outputs, allocation_diffs):
        if diff < -material:
            over.append({"race": race, "output": out, "diff": diff})
        elif diff > material:
            under.append({"race": race, "output": out, "diff": diff})

    def _summarize(items: list) -> dict:
        if not items:
            return {"count": 0, "by_rating": {}, "by_incumb": {}, "by_pvi_bin": {}}

        by_rating = pd.Series([i["race"].cook_rating for i in items]).value_counts().to_dict()
        by_incumb = pd.Series([i["race"].incumb_status for i in items]).value_counts().to_dict()

        pvi_vals = np.array([abs(i["race"].pvi) for i in items])
        bins = [0, 5, 10, 20, 100]
        labels = ["0-5", "5-10", "10-20", "20+"]
        bin_counts = pd.cut(pvi_vals, bins=bins, labels=labels).value_counts().to_dict()

        return {
            "count": len(items),
            "total_diff_pp": sum(abs(i["diff"]) for i in items),
            "by_rating": by_rating,
            "by_incumb": by_incumb,
            "by_pvi_bin": bin_counts,
        }

    return {
        "overfunded": _summarize(over),
        "underfunded": _summarize(under),
    }
