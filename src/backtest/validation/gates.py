"""
Validation gates — all 6 checks must pass before interpreting allocation results.

If any gate fails, raise ValidationError with a diagnosis.
Validation gates are checks on model integrity, not on outcomes.
"""

from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass
from ..types import RaceRecord, ModelOutputs, SigmaModel
from .. import config

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when a validation gate fails."""


@dataclass
class GateResult:
    name: str
    passed: bool
    value: float | str
    threshold: str
    notes: str = ""


def run_all_gates(
    races: list[RaceRecord],
    outputs: list[ModelOutputs],
    sigma_model: SigmaModel,
    margin_r2_competitive: float,
    optimizer_status: str,
    n_corner_solutions: int,
    brier_model: float,
    brier_cook: float,
    budget: float,
) -> list[GateResult]:
    """
    Run all 6 validation gates and return results.

    Logs each gate outcome. Raises ValidationError on first failure
    (caller can catch and decide whether to continue).
    """
    vcfg = config.validation_cfg()
    results = []

    # Gate 1: Spending data completeness
    n_total = len(races)
    n_complete = sum(
        1 for r in races if r.d_total > 0 and r.r_total > 0
    )
    frac_complete = n_complete / n_total if n_total > 0 else 0.0
    g1 = GateResult(
        name="Spending data completeness",
        passed=frac_complete >= vcfg["spending_completeness_min"],
        value=frac_complete,
        threshold=f">= {vcfg['spending_completeness_min']:.0%}",
        notes=f"{n_complete}/{n_total} races with both D and R spend",
    )
    results.append(g1)

    # Gate 2: Margin model R²
    r2_pass = vcfg["margin_model_r2_pass"]
    r2_stretch = vcfg["margin_model_r2_stretch"]
    r2_note = "stretch goal met" if margin_r2_competitive >= r2_stretch else (
        "pass threshold met" if margin_r2_competitive >= r2_pass else "FAILED"
    )
    g2 = GateResult(
        name="Margin model R² (competitive)",
        passed=margin_r2_competitive >= r2_pass,
        value=margin_r2_competitive,
        threshold=f">= {r2_pass:.2f} (stretch: {r2_stretch:.2f})",
        notes=r2_note,
    )
    results.append(g2)

    # Gate 3: σᵢ ordering (open > challenger > incumbent)
    pvi_bins = np.arange(0, 25, 5)
    ok_count = 0
    for pvi in pvi_bins:
        s_open = sigma_model.predict(float(pvi), "Open", generic_ballot=0.0)
        s_chall = sigma_model.predict(float(pvi), "Challenger", generic_ballot=0.0)
        s_incumb = sigma_model.predict(float(pvi), "Incumbent", generic_ballot=0.0)
        if s_open > s_chall > s_incumb:
            ok_count += 1
    frac_ordered = ok_count / len(pvi_bins)
    g3 = GateResult(
        name="σᵢ ordering (open > chall > incumb)",
        passed=frac_ordered >= vcfg["sigma_ordering_frac_min"],
        value=frac_ordered,
        threshold=f">= {vcfg['sigma_ordering_frac_min']:.0%} of PVI bins",
        notes=f"{ok_count}/{len(pvi_bins)} bins ordered correctly",
    )
    results.append(g3)

    # Gate 4: MSG sign (all competitive races should have MSG > 0)
    competitive = set(config.competitive_ratings())
    comp_outputs = [o for r, o in zip(races, outputs) if r.cook_rating in competitive]
    n_positive_msg = sum(1 for o in comp_outputs if o.msg_i > 0)
    all_positive = n_positive_msg == len(comp_outputs)
    g4 = GateResult(
        name="MSG sign (competitive races)",
        passed=all_positive,
        value=f"{n_positive_msg}/{len(comp_outputs)}",
        threshold="MSG_i > 0 for all competitive races",
        notes="" if all_positive else f"{len(comp_outputs) - n_positive_msg} races with MSG ≤ 0",
    )
    results.append(g4)

    # Gate 5: Optimizer convergence
    # Corner solutions (races at 0 or cap) are expected under risk-neutral (γ=0)
    # LP — the optimal LP solution IS a corner. Only check corner fraction for γ>0.
    corner_frac = n_corner_solutions / max(len(races), 1)
    # Note: allocator.py's optimize_nonlinear() only ever prefixes a status with
    # "slsqp:" when result.success is False (status = "optimal" if result.success
    # else f"slsqp:{result.message}") — any "slsqp:"-prefixed status is a real
    # failure and must not pass this gate, regardless of what result.message says.
    _ok_statuses = ("optimal", "optimal_inaccurate", "Optimization terminated successfully.")
    g5 = GateResult(
        name="Optimizer convergence",
        passed=optimizer_status in _ok_statuses,
        value=optimizer_status,
        threshold="status=optimal or optimal_inaccurate",
        notes=f"{n_corner_solutions} corner solutions ({corner_frac:.0%})",
    )
    results.append(g5)

    # Gate 6: Brier score
    brier_tol = vcfg["brier_tolerance"]
    brier_ok = brier_model <= brier_cook + brier_tol
    g6 = GateResult(
        name="Brier score",
        passed=brier_ok,
        value=brier_model,
        threshold=f"≤ Cook Brier + {brier_tol} (Cook Brier = {brier_cook:.4f})",
        notes=f"model={brier_model:.4f}, threshold={brier_cook + brier_tol:.4f}",
    )
    results.append(g6)

    # ── Report ────────────────────────────────────────────────────────────────
    for g in results:
        status = "PASS" if g.passed else "FAIL"
        logger.info(f"[{status}] {g.name}: {g.value} (threshold: {g.threshold}) — {g.notes}")

    failed = [g for g in results if not g.passed]
    if failed:
        names = ", ".join(g.name for g in failed)
        raise ValidationError(
            f"{len(failed)} validation gate(s) failed: {names}. "
            "Resolve before interpreting allocation results."
        )

    logger.info("All validation gates passed.")
    return results
