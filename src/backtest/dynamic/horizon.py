"""
Receding-horizon (MPC-style) sequential optimization loop (paper §4,
docs/paper2_draft.md).

This module is deliberately thin glue: every numerically meaningful step
(state re-estimation, portfolio optimization) delegates to Paper I's
existing, unmodified code (`compute_raw_snapshot` -> `compute_outputs_batch`,
`optimize_nonlinear`) or to the small dynamic/ helpers in state.py,
ledger.py, updates.py. No new solver logic lives here — see paper §4: "This
is, mechanically, the identical SLSQP problem solved in Paper I, with two
substitutions: the budget constraint uses deployable capital F_t rather
than the full-cycle budget, and each race's already-committed capital L_i,t
is added to the spending floor."

`_solve_one_period()` is shared with `dynamic/simulate.py`'s one-step-ahead
historical harness, so the live loop and the historical replay call
identical single-period logic.
"""

from __future__ import annotations
import dataclasses
import logging
from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np

from ..types import RaceRecord, SigmaModel, AllocationResult
from ..model.margin import MarginModelCoefficients
from ..model.win_prob import compute_outputs_batch
from ..optimizer.allocator import optimize_nonlinear, build_allocation_results, OptimizerResult
from .state import CampaignState
from .ledger import CapitalLedger, CommitmentSource
from .updates import StateUpdater, compute_raw_snapshot
from .periods import ReportingPeriod

logger = logging.getLogger(__name__)


@dataclass
class PeriodResult:
    """One reporting period's state, capital account, and optimizer
    recommendation (paper §4's pseudocode output)."""

    period: int
    period_date: date
    state: CampaignState
    ledger: CapitalLedger
    optimizer_result: OptimizerResult
    recommended_allocation: list[AllocationResult]


def _solve_one_period(
    races: list[RaceRecord],
    coef: MarginModelCoefficients,
    sigma_model: SigmaModel,
    ledger: CapitalLedger,
    cov_matrix: np.ndarray,
    gamma: float,
    cap_fraction: float,
) -> tuple[OptimizerResult, list[AllocationResult]]:
    """Solve exactly one period's allocation problem: Paper I's
    `optimize_nonlinear()`, called with that period's deployable capital
    F_t as `party_budget` and committed capital L_t baked into the floor
    via `ledger.apply_to_races`. Shared by `run_receding_horizon` (live
    loop, this module) and `simulate.one_step_ahead` (historical harness),
    so both call identical single-period logic — see paper §6.1.

    `optimize_nonlinear`'s `budget` argument is purely a `shares` reporting
    denominator (it does not affect `allocations`/`budget_used`/
    `expected_seats`). `ledger.total_budget` (B_t) is only the DCCC's own
    controllable capital — mirroring Paper I's `run_backtest.py`, where
    `budget` is the *total* D spend (candidate-committee floor + party
    budget) rather than the party budget alone, we report shares against
    the candidate/committed floor plus B_t, not B_t in isolation.

    `deployable_floor_for(races)` already returns `cand_d_total_i + L_i,t`
    per race (ledger.py), so summing it and adding `ledger.total_budget`
    (= L_t + F_t) would double-count L_t. Subtract `ledger.committed_total`
    once to cancel that: total = cand_total + L_t + B_t - L_t = cand_total + B_t.
    """
    races_with_floor = ledger.apply_to_races(races)
    outputs = compute_outputs_batch(races_with_floor, coef, sigma_model)
    total_d_spend = (
        ledger.total_budget
        + float(ledger.deployable_floor_for(races).sum())
        - ledger.committed_total
    )
    result = optimize_nonlinear(
        races_with_floor,
        coef,
        sigma_model,
        budget=total_d_spend,
        cov_matrix=cov_matrix,
        gamma=gamma,
        cap_fraction=cap_fraction,
        party_budget=ledger.deployable_total,
    )
    allocation = build_allocation_results(races_with_floor, outputs, result, total_d_spend)
    return result, allocation


def run_receding_horizon(
    periods: list[ReportingPeriod],
    initial_races: list[RaceRecord],
    coef: MarginModelCoefficients,
    sigma_model: SigmaModel,
    commitment_source: CommitmentSource,
    state_updater: StateUpdater,
    cov_matrix_fn: Callable[[list[RaceRecord]], np.ndarray],
    gamma: float,
    cap_fraction: float,
    total_budget_fn: Callable[[int], float],
    generic_ballot_national: float,
    period_races_fn: Callable[[int, date], list[RaceRecord]] | None = None,
) -> list[PeriodResult]:
    """
    Run the sequential allocation problem across `periods` (paper §4's
    pseudocode):

        for each reporting period t:
            observe new information -> build this period's RaceRecord snapshot
            update X_t -> X_{t+1}                          (state_updater)
            update capital account: F_t = B_t - L_t         (commitment_source)
            recompute mu, sigma, MSG from updated state      (compute_raw_snapshot)
            solve the optimizer over F_t, with L_t as a fixed floor
            output: recommended allocation of F_t

    `period_races_fn(period_index, period_date) -> list[RaceRecord]` supplies
    each period's spend snapshot; if omitted, `initial_races` is held fixed
    across all periods (useful for validating the loop against real
    coefficients before point-in-time reconstruction exists — see the
    implementation plan's Phase 2). `total_budget_fn(period_index) ->
    float` supplies B_t (the fundraising path); a constant function is a
    valid input for testing.

    This function only sequences existing pieces — see `_solve_one_period`
    for the one call that actually touches the optimizer.
    """
    results: list[PeriodResult] = []
    prev_state: CampaignState | None = None

    for rp in periods:
        races_t = period_races_fn(rp.index, rp.period_date) if period_races_fn else initial_races

        raw_snapshot = compute_raw_snapshot(
            races_t, coef, sigma_model, rp.index, rp.period_date, generic_ballot_national,
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
