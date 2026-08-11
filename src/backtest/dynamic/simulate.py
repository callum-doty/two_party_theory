"""
One-step-ahead historical simulation harness (paper §6, docs/paper2_draft.md).

**Never a closed-loop autoregressive rollout** — this is paper §6.2's
explicit methodological constraint, and the reason this module exists
separately from `dynamic/horizon.py`'s live receding-horizon loop. At each
historical reporting period, the campaign state is reconstructed from real
historical data only — never from a prior period's model recommendation —
and the single-period receding-horizon solve is compared against DCCC's
actual behavior. The model's own recommendation for period t is discarded
before moving to period t+1; nothing about it is ever fed into period
t+1's reconstructed state.

Why this matters: historical polling, fundraising, and spending data
available as of any date t+1 are themselves a function of what DCCC
*actually* spent through date t — not of what this architecture would have
recommended spending. Feeding a hypothetical model-driven spending path
back into a reconstruction of "what the world looked like next" would
optimize against a state contaminated by a counterfactual that never
happened. `_reconstruct_races_at`'s signature is the structural safeguard
against this: it takes no `PeriodResult`/`prior_results` argument, so there
is no way — even by mistake — to pass a past period's model output into a
later period's reconstructed state.
"""

from __future__ import annotations
import dataclasses
import logging
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from ..types import RaceRecord, SigmaModel
from ..model.margin import MarginModelCoefficients
from ..data import fec
from .ledger import CapitalLedger, CommitmentSource
from .updates import StateUpdater, compute_raw_snapshot
from .horizon import PeriodResult, _solve_one_period
from .periods import ReportingPeriod
from .state import CampaignState

logger = logging.getLogger(__name__)


def _static_floor_totals(cycle: int) -> pd.DataFrame:
    """Coordinated-expenditure spend per (district_id, party), held fixed
    across every period of the historical harness.

    This repo has no per-filing date source for coordinated expenditures
    (Phase 3 documented gap — see docs/paper2_draft.md §6.2's
    data-availability table): `coordinated_expenditures_{cycle}.csv` is a
    cycle-cumulative FEC (Schedule F) snapshot with no filing-date
    granularity, and Schedule F's periodic dating has not been investigated
    (a separate, unexplored question from the candidate-committee gap this
    docstring used to also claim).

    Candidate-committee spend is NOT included here as of this session —
    unlike coordinated expenditures, it now HAS a per-filing date source
    (`candidate_periodic_reports_{cycle}.csv`, FEC API's
    `/committee/{id}/reports/` endpoint, confirmed live this session; see
    data_catalog.md §2.7) and is applied dynamically in
    `_reconstruct_races_at` via `fec.cumulative_candidate_spend_as_of`
    instead of being folded into this fixed total.

    Returns DataFrame with columns: district_id, party, coord_static.
    """
    coord = fec.load_coordinated_expenditures(cycle)[
        ["district_id", "party", "coordinated_expenditures"]
    ]
    coord = coord.rename(columns={"coordinated_expenditures": "coord_static"})
    coord["coord_static"] = coord["coord_static"].fillna(0.0)
    return coord[["district_id", "party", "coord_static"]]


def _has_dated_candidate_panel(cycle: int) -> bool:
    """Whether candidate_periodic_reports_{cycle}.csv has been fetched
    (data_catalog.md §2.7) — checked once per cycle by one_step_ahead, not
    per period, so a missing panel doesn't retry a filesystem check (or log
    a repeated warning) at every one of N reporting periods."""
    from .. import config
    return (config.raw_path("fec") / f"candidate_periodic_reports_{cycle}.csv").exists()


def _candidate_fallback_totals(cycle: int) -> pd.DataFrame:
    """Cycle-final candidate-committee disbursements, held fixed across
    every period — the pre-dated-panel behavior, used only when
    `candidate_periodic_reports_{cycle}.csv` has not been fetched for this
    cycle yet (requires a registered FEC API key; not every cycle may be
    fetched at any given time). Mirrors this project's established honest-
    fallback pattern (e.g. dynamic/ledger.py's AdReservationProxySource)
    rather than crashing the historical harness because one cycle's dated
    data isn't present.

    Returns DataFrame with columns: district_id, party, cand_static.
    """
    cand = fec.load_candidate_disbursements(cycle)[
        ["district_id", "party", "candidate_disbursements"]
    ]
    return cand.rename(columns={"candidate_disbursements": "cand_static"})


def _reconstruct_races_at(
    period_index: int,
    period_date: date,
    cycle: int,
    base_races: list[RaceRecord],
    static_totals: pd.DataFrame,
    use_dated_candidate_spend: bool,
    candidate_fallback_totals: pd.DataFrame | None = None,
) -> list[RaceRecord]:
    """
    Reconstruct one historical period's RaceRecord snapshot from real data
    only.

    Structural no-lookahead safeguard: none of this function's parameters
    is a `PeriodResult`/`prior_results` — a prior period's *model
    recommendation* cannot be passed into this call even by mistake.
    `d_total`/`r_total` are built entirely from DCCC's real, actual
    historical spend as of `period_date`.

    Held fixed for every period (documented Phase 3 gaps, narrowed this
    session — see data_catalog.md §2.7): coordinated-expenditure spend
    (`static_totals` — see `_static_floor_totals`; Schedule F has no
    investigated per-filing date source), Cook rating (carried unchanged
    from `base_races` — no historical revision time series exists here),
    and cash on hand (no data source at all; stays unset downstream).
    Candidate-committee spend is now genuinely period-varying when
    `use_dated_candidate_spend` is True (the common case once
    `candidate_periodic_reports_{cycle}.csv` has been fetched for `cycle`
    — `fec.cumulative_candidate_spend_as_of`); when False, it falls back to
    `candidate_fallback_totals` (cycle-final, held fixed) — the same
    behavior this function had before this session, for cycles the dated
    panel hasn't been fetched for. Independent-expenditure spend
    (`fec.cumulative_ie_as_of`) has always had real per-transaction dates.

    Generic ballot is held fixed too, but for a different and more
    fundamental reason than "no data source exists" — see the
    implementation plan's Phase 4A note. `coef.alpha3` (this margin
    model's generic-ballot coefficient) is estimated in
    `model.margin.estimate_from_panel()` from a design matrix with exactly
    one GB value per election cycle (`generic_ballot_by_cycle`), identical
    across every race in that cycle. Its identifying variation is
    therefore entirely *between* cycles (six cycles, six GB values), never
    *within* a cycle. Even where a genuine within-cycle historical GB time
    series is available, applying alpha3 to day-to-day movement in that
    series would be substituting an estimand the model was never fit
    against — a modeling decision, not a data-acquisition one — and this
    function does not make that substitution.
    """
    ie_asof = fec.cumulative_ie_as_of(cycle, period_date)
    ie_by_key = {(r.district_id, r.party): r.ie_net for r in ie_asof.itertuples()}
    coord_by_key = {(r.district_id, r.party): r.coord_static for r in static_totals.itertuples()}

    if use_dated_candidate_spend:
        cand_dated = fec.cumulative_candidate_spend_as_of(cycle, period_date)
        cand_by_key = {(r.district_id, r.party): r.disb_cum for r in cand_dated.itertuples()}
    else:
        assert candidate_fallback_totals is not None
        cand_by_key = {
            (r.district_id, r.party): r.cand_static for r in candidate_fallback_totals.itertuples()
        }

    snapshot: list[RaceRecord] = []
    for race in base_races:
        d_cand = cand_by_key.get((race.district_id, "D"), 0.0)
        r_cand = cand_by_key.get((race.district_id, "R"), 0.0)
        d_coord = coord_by_key.get((race.district_id, "D"), 0.0)
        r_coord = coord_by_key.get((race.district_id, "R"), 0.0)
        d_ie = ie_by_key.get((race.district_id, "D"), 0.0)
        r_ie = ie_by_key.get((race.district_id, "R"), 0.0)
        snapshot.append(dataclasses.replace(
            race,
            d_total=d_cand + d_coord + d_ie,
            r_total=r_cand + r_coord + r_ie,
            # Bug fix (found while building the portfolio-level ceiling
            # check): cand_d_total -- the field every optimizer/ceiling call
            # actually reads as each race's floor -- was never updated here,
            # only d_total/r_total. This module's own docstring already
            # claimed "applied dynamically ... instead of being folded into
            # this fixed total" for candidate spend; that claim was true for
            # d_total but not for cand_d_total, so every downstream floor
            # computation (predict_floor_margin, the persuasion ceiling,
            # optimize_nonlinear's floor constraint) was silently using the
            # cycle-*final* candidate floor at every historical checkpoint,
            # not the period-specific one. No existing test caught this --
            # TestDatedCandidateSpendReconstruction only asserts d_total.
            cand_d_total=d_cand,
        ))
    return snapshot


def one_step_ahead(
    periods: list[ReportingPeriod],
    cycle: int,
    base_races: list[RaceRecord],
    coef: MarginModelCoefficients,
    sigma_model: SigmaModel,
    commitment_source: CommitmentSource,
    state_updater: StateUpdater,
    cov_matrix_fn: Callable[[list[RaceRecord]], np.ndarray],
    gamma: float,
    cap_fraction: float,
    total_budget_fn: Callable[[int], float],
    generic_ballot_national: float,
) -> list[PeriodResult]:
    """
    Evaluate the architecture one historical reporting period at a time
    (paper §6.2) — never a closed-loop rollout.

    For each period t: reconstruct RaceRecords from real historical data
    only (`_reconstruct_races_at`, which cannot receive any prior period's
    model output by construction); re-run Paper I's pipeline on that real
    snapshot (`compute_raw_snapshot`); apply the state-update operator
    (`state_updater`) — carrying forward only the EMA's `mu_hat`/`sigma_hat`
    summary, itself computed purely from real past data, never from a past
    model *recommendation* (paper §5.3 draws this distinction explicitly);
    solve exactly one period via `horizon._solve_one_period`, the identical
    single-period logic the live receding-horizon loop uses (paper §6.1).
    Each period's `PeriodResult` is appended and the loop moves on — no
    `PeriodResult` is ever read back by a later iteration.

    Compare the returned results against DCCC's actual behavior and
    against Paper I's static recommendation using `dynamic/timing.py`
    (paper §6.3): `PeriodResult.state` already carries the real
    reconstructed `d_total_t`/`cand_d_total_t` at each period, so the
    "actual" comparison series does not need to be threaded through
    separately.
    """
    static_totals = _static_floor_totals(cycle)
    use_dated_candidate_spend = _has_dated_candidate_panel(cycle)
    candidate_fallback_totals = None if use_dated_candidate_spend else _candidate_fallback_totals(cycle)
    if not use_dated_candidate_spend:
        logger.warning(
            f"one_step_ahead(cycle={cycle}): no dated candidate periodic-reports panel found "
            "(data_catalog.md §2.7) — falling back to cycle-final candidate spend held fixed "
            "across every period. Run `python scripts/fetch_data.py --only fec-periodic "
            f"--cycles {cycle} --fec-api-key YOUR_KEY` to enable genuinely dated candidate spend."
        )
    results: list[PeriodResult] = []
    prev_state: CampaignState | None = None

    for rp in periods:
        races_t = _reconstruct_races_at(
            rp.index, rp.period_date, cycle, base_races, static_totals,
            use_dated_candidate_spend, candidate_fallback_totals,
        )

        cash_on_hand_by_district = None
        if use_dated_candidate_spend:
            coh = fec.cash_on_hand_as_of(cycle, rp.period_date)
            coh_d = coh[coh["party"] == "D"]
            cash_on_hand_by_district = dict(zip(coh_d["district_id"], coh_d["cash_on_hand"]))

        raw_snapshot = compute_raw_snapshot(
            races_t, coef, sigma_model, rp.index, rp.period_date, generic_ballot_national,
            cash_on_hand_by_district=cash_on_hand_by_district,
        )
        state_t = state_updater.update(prev_state, raw_snapshot)
        prev_state = state_t

        total_budget_t = total_budget_fn(rp.index)
        ledger_t = CapitalLedger.build(
            rp.index, total_budget_t, commitment_source, rp.period_date, races_t,
        )
        # committed_t is a diagnostic mirror of ledger_t.committed_by_race (state.py's
        # RaceState docstring) -- populated here, the first point in the loop where
        # both state_t and ledger_t exist, rather than left at its 0.0 default.
        state_t.races.update({
            did: dataclasses.replace(rs, committed_t=ledger_t.committed_by_race.get(did, 0.0))
            for did, rs in state_t.races.items()
        })

        cov_matrix = cov_matrix_fn(races_t)
        optimizer_result, allocation = _solve_one_period(
            races_t, coef, sigma_model, ledger_t, cov_matrix, gamma, cap_fraction,
        )

        results.append(PeriodResult(
            period=rp.index,
            period_date=rp.period_date,
            state=state_t,
            ledger=ledger_t,
            optimizer_result=optimizer_result,
            recommended_allocation=allocation,
        ))

    return results
