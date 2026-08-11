# Methodology notes

Supplement to `docs/project_spec.md` -- implementation-level decisions the
spec leaves open, and why each was resolved the way it was.

## Shared probability model, not two calibrated formulas

`game/payoff.py` and `game/gradients.py` use ONE p_i(D, R) model for both
sides' utilities and both sides' gradients: the D-anchored margin/ceiling
construction from `backtest.optimizer.allocator`, which IS empirically
calibrated against real DCCC spending-response data. R's spending enters
only through the shared `total_r` argument.

This is a deliberate departure from `backtest.optimizer.nash.py`, which
scores R's own best-response SEARCH through a separate, uncalibrated
mirrored-ceiling formula (documented there as a stated assumption, not a
validated one) and then RE-SCORES the result through the D-side formula for
reporting. This project's `game/best_response.py` still calls that same
`nash.best_response()` under the hood for `BR_R` (no reason to re-derive a
working, already-debugged SLSQP solve) -- but every OTHER quantity
(`MSG^R`, `U_R`, exploitability, PSV) is computed through the single shared
`payoff.p_win`, so nothing in this project's own reporting layer mixes two
different implicit beliefs about the same allocation. See
`game/gradients.py`'s module docstring for the derivation of `d p / d R`
this required (no counterpart existed in the old codebase).

## The PSV baseline choice (spec Section 14)

Read literally, `PSV_i = U_D(D', BR_R(D')) - U_D(D, R)` measures against the
OBSERVED R baseline. Empirically (2024 universe, see
`tests/test_exploitability_real_universe.py`), observed R is far enough from
R's own unilateral optimum (RegretR ~= +4.6 seats) that this literal
baseline makes PSV nearly race-invariant -- every race's `BR_R(D')` recovers
most of that same aggregate swing, swamping the race-specific signal Section
14's own worked example is illustrating.

`game/persistent_value.py` defaults to an ISOLATED baseline instead --
`U_D(D, BR_R(D_observed))`, i.e. R already at its own best response,
computed ONCE and reused across every candidate race -- which cancels the
shared RegretR term and leaves each race's own erosion. Pass
`baseline_e_seats=None` (or `--baseline observed` on
`scripts/compute_persistent_value.py`) to get the literal spec formula
instead. Report BOTH when writing this up; the difference between them IS a
finding (how much of the aggregate regret any single deviation forces R to
"absorb" incidentally).

## What "financing the delta" means, concretely

Spec Section 13 says a race's `D_i' = D_i + delta` deviation is "financed by
removing delta from their portfolio's current marginal use." Operationalized
in `game/persistent_value._finance_delta()` as: cut from the CURRENTLY
FUNDED race(s) with the lowest `MSG^D` at the observed point, cascading to
the next-lowest if one race can't cover delta alone. This is literally "the
portfolio's own optimizer would cut here first," not a proportional or
arbitrary trim.

## Validation levels in this codebase

| Level | Spec Section 20 question | Where |
|---|---|---|
| A | Response model predicts out of sample? | Reused from old project (`backtest.estimation.*`); not re-run here. |
| B | Nonlinear vs. surrogate agree? | D-side: validated in the old project (`theta_concave_surrogate.py`). R-side surrogate (`src/optimizer/concave_surrogate.py::surrogate_allocate_r`) is NOT yet validated -- explicit TODO. |
| C | Algorithm recovers known synthetic equilibria? | `src/validation/synthetic_games.py`, `tests/test_synthetic_game.py`. Standalone logistic-contest game, decoupled from the estimated election model on purpose (isolates the ITERATION algorithm from the SUBSTANTIVE model). |
| D | Observed allocations closer to Nash than alternatives? | `src/validation/historical_backtest.py::run_cycle` computes the L1 distance; the full comparison against equal-allocation / Cook-heuristic / one-sided-optimizer / random-feasible benchmarks (spec Section 20's five-way comparison) is not yet implemented -- next step after the 2022/2024 MVP (spec Section 26). |

## Known open items (see spec Section 19, addressed only partially)

- **D/R elasticity symmetry test**: not implemented (`scripts/estimate_response_model.py::test_d_r_symmetry` raises `NotImplementedError` with what it would take). Until run, treat `MSG^R` as "D's mirrored elasticity," an assumption, not a tested symmetric response curve.
- **R-side concave surrogate validation**: not benchmarked against `backtest.optimizer.nash.best_response("R", ...)` the way the D-side surrogate was.
- **Level D five-way benchmark comparison**: only the Nash-vs-observed L1 distance is wired up so far.
