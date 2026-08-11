"""
Capital account: B_t = L_t + F_t (paper §3.2, docs/paper2_draft.md).

The sequential optimizer (dynamic/horizon.py) solves only over deployable
capital F_t; committed (irreversible) capital L_t enters as an addition to
each race's existing candidate-committee spend floor — the exact mechanism
Paper I already uses for `RaceRecord.cand_d_total`, not a new constraint
type.

This module also implements the paper's research-mode/operational-mode
distinction for where L_t comes from: `CommitmentSource` is a Protocol with
three implementations, so the optimization architecture never depends on a
specific data source having been solved.
"""

from __future__ import annotations
import dataclasses
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ..types import RaceRecord

logger = logging.getLogger(__name__)


class CommitmentSource(Protocol):
    """Supplies L_t — committed, irreversible capital — per race, per period."""

    def committed_capital(
        self, period: int, period_date: date, races: list[RaceRecord],
    ) -> dict[str, float]:
        """Return {district_id: L_i,t}. Districts absent from the returned
        dict are treated as L_i,t = 0."""
        ...


class ZeroCommitmentSource:
    """L_t = 0 for every race, every period.

    The honest default when no commitment data is configured — this makes
    "no commitment data available" an explicit, named, testable state
    (paper §3.2) rather than an implicit zero buried inside a stub.
    """

    def committed_capital(
        self, period: int, period_date: date, races: list[RaceRecord],
    ) -> dict[str, float]:
        return {r.district_id: 0.0 for r in races}


class OperationalLedgerSource:
    """Operational mode (paper §3.2): L_t supplied directly by the
    committee's own internal accounting ledger.

    Reads a CSV with columns `period`, `district_id`, `committed`. Fully
    implementable and testable now — this is a plain file reader with a
    defined schema; no real committee ledger is required to exercise it
    (tests construct a small synthetic ledger CSV).
    """

    def __init__(self, ledger_path: Path | str):
        self.ledger_path = Path(ledger_path)
        self._df = pd.read_csv(self.ledger_path)
        required = {"period", "district_id", "committed"}
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(f"Ledger file {self.ledger_path} missing columns: {missing}")

    def committed_capital(
        self, period: int, period_date: date, races: list[RaceRecord],
    ) -> dict[str, float]:
        sub = self._df[self._df["period"] == period]
        return dict(zip(sub["district_id"], sub["committed"].astype(float)))


class AdReservationProxySource:
    """Research mode (paper §3.2): L_t approximated from publicly
    observable commitment proxies — e.g. booked-but-unaired television
    reservation data published by commercial ad-tracking services such as
    AdImpact or Medium Buying/CMAG, which report reservations at booking
    time rather than at eventual FEC disbursement filing.

    STUB. No such data source is available, licensed, or fetched anywhere
    in this repository. This class exists purely so `CommitmentSource` has
    a "research mode" implementation that is swappable in later without
    changing any other code — it does not fabricate a synthetic dataset
    dressed up to look like real ad-reservation data, which would
    misrepresent data provenance (see docs/paper2_draft.md §3.2).

    `strict=True` raises `NotImplementedError`. `strict=False` (default)
    logs a warning and returns zero commitments for every race, so the rest
    of the pipeline remains exercisable end-to-end while this data source
    is unavailable.
    """

    def __init__(self, data_path: Path | str | None = None, strict: bool = False):
        self.data_path = Path(data_path) if data_path is not None else None
        self.strict = strict

    def committed_capital(
        self, period: int, period_date: date, races: list[RaceRecord],
    ) -> dict[str, float]:
        msg = (
            "AdReservationProxySource has no real ad-reservation data source "
            "configured — see docs/paper2_draft.md §3.2 (research mode). This "
            "is a placeholder interface for a future AdImpact/Medium Buying/"
            "CMAG-style feed, not an implemented estimator."
        )
        if self.strict:
            raise NotImplementedError(msg)
        logger.warning(msg + " Falling back to L_t=0 for all races (strict=False).")
        return {r.district_id: 0.0 for r in races}


class RealizedSpendCommitmentSource:
    """Live/historical mode: L_t = money already disbursed by DCCC-aligned
    committees as of `period_date` — a lower-bound, but fully real,
    alternative to the unimplementable `AdReservationProxySource`.

    Two components, both already-spent and therefore unambiguously
    irreversible:
      - Coordinated expenditures (FEC Schedule F, via
        `fec.load_coordinated_expenditures`) — cycle-cumulative only; this
        repo has no per-filing date source for this component (the same
        documented gap as `dynamic/simulate.py`'s `_static_floor_totals`),
        so it is held at its latest fetched total regardless of
        `period_date`.
      - Independent expenditures (FEC Schedule E, via
        `fec.cumulative_ie_as_of`) — genuinely date-bucketed from
        per-transaction data, so this component does respect `period_date`.

    Candidate-committee spend is intentionally NOT a third component here —
    committed capital (L_t) is party-committee money (coordinated + IE);
    candidate-committee spend is the party's own remaining-budget floor,
    not something the party "commits" to itself. That said, the underlying
    per-filing-date gap this docstring used to cite for candidate spend as
    well is resolved as of this session: `data_catalog.md` §2.7's dated
    periodic-reports panel gives candidate-committee spend genuine
    per-filing dating, now used in `dynamic/simulate.py`'s
    `_reconstruct_races_at` for the floor itself. Coordinated expenditures
    remain the one genuinely undated component in this ledger.

    This does NOT capture booked-but-unaired reservations (the concept
    `AdReservationProxySource` targets and cannot obtain affordably — see
    the implementation plan's Phase 4 scoping). Money already disbursed is
    a strictly smaller, conservative estimate of true committed capital,
    but it is real, requires no new data acquisition, and is a meaningful
    improvement over `ZeroCommitmentSource` for a live, in-progress cycle.
    """

    def __init__(self, cycle: int, party: str = "D"):
        self.cycle = cycle
        self.party = party

    def committed_capital(
        self, period: int, period_date: date, races: list[RaceRecord],
    ) -> dict[str, float]:
        from ..data import fec  # local import: dynamic/ depends on data/, not vice versa

        coord = fec.load_coordinated_expenditures(self.cycle)
        coord = coord[coord["party"] == self.party]
        coord_by_district = dict(zip(coord["district_id"], coord["coordinated_expenditures"]))

        ie_asof = fec.cumulative_ie_as_of(self.cycle, period_date)
        ie_asof = ie_asof[ie_asof["party"] == self.party]
        ie_by_district = dict(zip(ie_asof["district_id"], ie_asof["ie_net"]))

        return {
            r.district_id: (
                coord_by_district.get(r.district_id, 0.0)
                + ie_by_district.get(r.district_id, 0.0)
            )
            for r in races
        }


@dataclass
class CapitalLedger:
    """Party-level and per-race capital account at period t.

    B_t = L_t + F_t. `dynamic/horizon.py`'s optimizer call solves only over
    `deployable_total` (F_t); `committed_by_race` (L_t, per race) is baked
    into the optimizer's floor via `apply_to_races`.
    """

    period: int
    total_budget: float                    # B_t
    committed_by_race: dict[str, float]    # L_i,t
    committed_total: float                 # L_t = sum(committed_by_race.values())
    deployable_total: float                # F_t = B_t - L_t

    @classmethod
    def build(
        cls,
        period: int,
        total_budget: float,
        commitment_source: CommitmentSource,
        period_date: date,
        races: list[RaceRecord],
    ) -> "CapitalLedger":
        committed_by_race = commitment_source.committed_capital(period, period_date, races)
        committed_total = float(sum(committed_by_race.values()))
        deployable_total = total_budget - committed_total
        if deployable_total < 0:
            raise ValueError(
                f"Period {period}: committed capital (${committed_total:,.0f}) "
                f"exceeds total budget (${total_budget:,.0f})."
            )
        return cls(
            period=period,
            total_budget=total_budget,
            committed_by_race=committed_by_race,
            committed_total=committed_total,
            deployable_total=deployable_total,
        )

    def deployable_floor_for(self, races: list[RaceRecord]) -> np.ndarray:
        """Return, aligned to `races`, cand_d_total_i + L_i,t — the exact
        per-race floor the sequential optimizer should respect."""
        return np.array([
            r.cand_d_total + self.committed_by_race.get(r.district_id, 0.0)
            for r in races
        ])

    def apply_to_races(self, races: list[RaceRecord]) -> list[RaceRecord]:
        """Return new RaceRecords with `cand_d_total` increased by L_i,t.

        `optimize_nonlinear()` reads its per-race floor directly from
        `RaceRecord.cand_d_total` (it takes no separate floor argument), so
        committed capital must be baked into that field before calling it —
        this is that step. No new constraint type is introduced: L_t uses
        exactly the mechanism Paper I already uses for the
        candidate-committee spend floor.
        """
        floor = self.deployable_floor_for(races)
        return [
            dataclasses.replace(r, cand_d_total=float(floor[i]))
            for i, r in enumerate(races)
        ]
