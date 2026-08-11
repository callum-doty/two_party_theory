"""
Reporting-period calendar abstraction (paper §3, §6, docs/paper2_draft.md).

Genuinely new: `scripts/fetch_live_ies.py` has no discrete-period concept,
only a rolling `lookback_hours` window. This module is a small, self
contained calendar utility — it does not fetch or transform any data.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class ReportingPeriod:
    index: int
    period_date: date   # the "as-of" date this period's snapshot reflects
    label: str


def biweekly_periods(start: date, end: date, period_days: int = 14) -> list[ReportingPeriod]:
    """Return biweekly reporting periods from `start` through `end`
    (inclusive of any period landing on or before `end`).

    `period_days` defaults to 14 to keep this a self-contained, dependency-
    free calendar utility (no import of backtest.config). The default is the
    single canonical value -- also exposed as config.yaml's
    `dynamic.period_days` / `backtest.config.period_days()` for callers that
    need to keep their own period-length constants in sync with this one."""
    if end < start:
        raise ValueError("end must be >= start")
    periods: list[ReportingPeriod] = []
    current = start
    i = 0
    while current <= end:
        periods.append(ReportingPeriod(index=i, period_date=current, label=f"P{i}"))
        current = current + timedelta(days=period_days)
        i += 1
    return periods


def fec_quarterly_periods(cycle: int) -> list[ReportingPeriod]:
    """Return approximate FEC quarterly reporting dates for a House
    election cycle: Q1/Q2/Q3 of the odd year before the cycle, then
    Q1/Q2/Q3 plus pre-general and year-end filings of the cycle year.

    Dates are approximate FEC quarterly filing deadlines (15th of the month
    after quarter-end). Phase 3's point-in-time historical reconstruction
    should substitute exact historical filing deadlines where precision
    matters; this calendar is meant for scaffolding the loop, not as a
    source of truth for filing-date accuracy.
    """
    quarters = [
        (cycle - 1, 4, 15), (cycle - 1, 7, 15), (cycle - 1, 10, 15),
        (cycle, 4, 15), (cycle, 7, 15), (cycle, 10, 15),
        (cycle, 11, 25), (cycle, 12, 31),
    ]
    return [
        ReportingPeriod(
            index=i,
            period_date=date(y, m, d),
            label=f"{y}-Q{((m - 1) // 3) + 1}",
        )
        for i, (y, m, d) in enumerate(quarters)
    ]
