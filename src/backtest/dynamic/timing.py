"""
Deployment-timing diagnostic (paper §5.4.1, §6.3, docs/paper2_draft.md).

Section 4's receding-horizon optimizer has no term rewarding retained
flexibility (paper §5.4's option-value/Θ-decay account), so it is expected
to recommend deploying capital earlier than DCCC's actual, more patient,
behavior. This module turns that predicted gap into a diagnostic — a
revealed-preference estimate of the option value DCCC implicitly assigns to
retained flexibility — rather than treating front-loading as a defect to
patch (paper §5.4.1).
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date

import pandas as pd

from ..types import RaceRecord, SigmaModel
from .horizon import PeriodResult


@dataclass
class TimingComparison:
    """One race's deployment-timing comparison at one reporting period."""

    district_id: str
    period: int
    period_date: date
    model_recommended_party_spend: float        # this period's model recommendation, if F_t were deployed now
    dccc_actual_cumulative_party_spend: float    # real, reconstructed cumulative party spend as of this period
    dccc_actual_incremental_party_spend: float   # vs. the previous period
    gap: float                                   # model recommended − DCCC actual incremental (paper predicts > 0)


def build_timing_table(results: list[PeriodResult]) -> list[TimingComparison]:
    """
    Per-period, per-race deployment-timing comparison (paper §6.3's second
    question). `results` must be in chronological period order — the
    direct output of `simulate.one_step_ahead` or
    `horizon.run_receding_horizon`.

    `dccc_actual_cumulative_party_spend` is read directly off each period's
    `PeriodResult.state` (real reconstructed `d_total − cand_d_total`), not
    recomputed from any separate "actual" argument: the one-step-ahead
    harness (paper §6.2) never lets model output enter that state, so it is
    safe to treat as ground truth here.

    A district's first appearance in `results` produces no row: coordinated
    expenditures (RealizedSpendCommitmentSource) are held at their
    cycle-final total at every period rather than genuinely date-bucketed,
    so that static component cancels correctly in every period-over-period
    difference from the second observation onward, but would appear as one
    giant fabricated "incremental" spend if diffed against an assumed-zero
    baseline at the first observation. Rather than report a number known to
    be an artifact, the first period only seeds `prev_cumulative`.
    """
    comparisons: list[TimingComparison] = []
    prev_cumulative: dict[str, float] = {}

    for res in results:
        races = res.state.to_race_records()
        floor = res.ledger.deployable_floor_for(races)
        allocations = res.optimizer_result.allocations

        for i, race in enumerate(races):
            cumulative_actual = race.d_total - race.cand_d_total
            if race.district_id not in prev_cumulative:
                prev_cumulative[race.district_id] = cumulative_actual
                continue
            incremental_actual = cumulative_actual - prev_cumulative[race.district_id]
            model_recommended = float(allocations[i] - floor[i])
            comparisons.append(TimingComparison(
                district_id=race.district_id,
                period=res.period,
                period_date=res.period_date,
                model_recommended_party_spend=model_recommended,
                dccc_actual_cumulative_party_spend=cumulative_actual,
                dccc_actual_incremental_party_spend=incremental_actual,
                gap=model_recommended - incremental_actual,
            ))
            prev_cumulative[race.district_id] = cumulative_actual

    return comparisons


def timing_gap_vs_volatility(
    timing: list[TimingComparison],
    sigma_model: SigmaModel,
    races_by_district: dict[str, RaceRecord],
) -> pd.DataFrame:
    """
    Correlate each race's cumulative timing gap (summed across periods — an
    AUC-style measure of front-loading) against its estimated σᵢ
    (volatility) — paper §6.3's third question: does gap size correlate
    with race volatility, as the option-value account (paper §5.2–5.4)
    predicts.

    Returns a DataFrame with columns: district_id, total_gap, sigma_i. The
    Pearson correlation between the two columns (when at least 2 races are
    present) is stashed in `df.attrs["correlation"]`.
    """
    totals: dict[str, float] = {}
    for tc in timing:
        totals[tc.district_id] = totals.get(tc.district_id, 0.0) + tc.gap

    rows = []
    for district_id, total_gap in totals.items():
        race = races_by_district.get(district_id)
        if race is None:
            continue
        sigma_i = sigma_model.predict(abs(race.pvi), race.incumb_status, race.generic_ballot)
        rows.append({"district_id": district_id, "total_gap": total_gap, "sigma_i": sigma_i})

    df = pd.DataFrame(rows, columns=["district_id", "total_gap", "sigma_i"])
    if len(df) >= 2:
        df.attrs["correlation"] = float(df["total_gap"].corr(df["sigma_i"]))
    return df
