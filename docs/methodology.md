# Methodology notes

Supplement to `docs/project_spec.md` -- implementation-level decisions the
spec leaves open, and why each was resolved the way it was.

## PSV retention >100% anomaly: investigated and fixed (2026-08-11)

The first historical-backtest run (`docs/results_2022_2024.md`) showed
several D-side races with PSV retention above 100% -- the opponent's
optimal response making the deviation race MORE valuable, the opposite of
what PSV is designed to detect. Traced end to end (GA-07, 2024) before
accepting or dismissing it. Two distinct, now-fixed causes were found; a
smaller amount of genuine >100% variation remains and is NOT a bug.

**Cause 1 (the big one): candidate selection picked $0-spend races.**
`historical_backtest.run_cycle` ranked ALL 433 races by `|Z_D|` (a
normalized, dimensionless surplus measure) to pick PSV candidates. Once
x_D/x_R were correctly scoped to control-only money (see the section
below), DCCC places $0 of party money in 379 of 433 races (NRCC in 394 of
433) -- and MSG evaluated AT exactly $0 sits at the steepest, most unstable
point of the persuasion-ceiling curve, a known artifact
`scripts/game_theory/race_level_exploitability.py`'s own scatter-plot code
already documents ("low-spend MSG artifact dominates"). Verified directly:
every single flagged race had `party_d_obs == $0`. Because these races'
TRUE unilateral value (evaluated at a real $100K-$1M injection, not the
instantaneous derivative) saturates almost immediately and stays tiny --
confirmed by re-running GA-07 at 10x the delta and finding V_uni virtually
unchanged (+0.0395 -> +0.0394) while PSV got WORSE (retention 142.5% ->
195.7%) -- while R's full 433-race reoptimization produces a comparably-
sized second-order reshuffling effect under the shared budget constraint,
PSV/V_uni divides two numbers of similar magnitude and the ratio becomes
unstable. **Fix**: restrict the candidate pool to races with real current
party spend (`> $10,000`) before ranking by `|Z|` -- both
`historical_backtest.py` and `compute_persistent_value.py`.

**Cause 2 (smaller): near-zero V_uni even among funded races.** After Cause
1's fix, 2024's R-side top-|Z_R| picks were all funded races but still had
tiny |V_uni| (~0.0001-0.0002 expected seats) -- R's allocation among
races it actually funds was already close to locally optimal, so there was
barely any unilateral opportunity to begin with, and PSV/V_uni was again
dividing near-noise. **Fix**: `game/persistent_value.py`'s
`RETENTION_MATERIALITY_THRESHOLD` (0.001 expected seats) -- below it,
`retention_rate` is reported as `NaN` ("no material opportunity here," not
a percentage) rather than a technically-defined but practically-misleading
ratio.

**What's genuine and NOT a bug**: even after both fixes, a handful of
well-funded races with real V_uni (NV-01 2024: V_uni=+0.0042) still show
retention above 100% (209.5%). Traced analytically: PSV - V_uni =
[U_D(D', BR_R(D')) - U_D(D', R_obs)] - (-RegretR). The bracketed term is
the aggregate effect of R's FULL reoptimization specifically at the
perturbed allocation D' (not the generic -RegretR baseline effect at
D_obs) -- moving delta from D's lowest-value funded race INTO the target
race changes what R's whole-portfolio optimum looks like under the shared
budget constraint, and that reshuffling can happen to land marginally more
favorably for D than the -RegretR baseline would suggest. This is a real
second-order equilibrium interaction, not solver noise -- confirmed by
inspecting where R's money actually moved between BR_R(D_obs) and BR_R(D')
in the GA-07 trace (R poured MORE money into GA-07 itself under BR_R(D'),
which should hurt D there specifically, yet the AGGREGATE effect across all
433 races was still slightly more D-favorable than -RegretR). Report
retention as a real but occasionally counterintuitive statistic, not one
that's always boundable to [0%, 100%].

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

## x_D / x_R must be CONTROLLED money, not just non-candidate money (fixed 2026-08-11)

A design critique caught a real conceptual error before any historical
backtest ran: `party_r = r_total - cand_r_total` (and the D-side mirror)
were being fed directly into `BR_D`/`BR_R`/`MSG^D`/`MSG^R` as the two-player
game's decision variables. But `r_total - cand_r_total` is "every dollar
not raised by the R candidate committee" -- it includes super-PAC and other
outside-group independent expenditures the NRCC has no legal authority over
(FEC's own definition: an independent expenditure is, by definition, NOT
coordinated with any party committee). Feeding that into an optimizer as
"NRCC's action space" lets the model reallocate money the NRCC could never
actually move.

Fixed via `src/estimation/control_provenance.py`, which decomposes every
race's d_total/r_total into four sources with a checked accounting identity
(`cand + party_natl + party_state + outside == total`, verified to float
precision on both 2022 and 2024):

- `cand` -- candidate committee disbursements (unchanged).
- `party_natl` -- the NATIONAL committee's OWN money: its coordinated
  expenditures PLUS its own independent expenditures (a "hybrid" IE
  strategy party committees may legally use -- still the committee's own
  spending decision, unlike a super PAC's IE, even though both show up as
  "R-aligned IE" in the raw comprehensive file). **This is x_D / x_R.**
- `party_state` -- state party 24K coordinated spending (see the
  state-party section below): real, coordinated, party money, but
  controlled by STATE parties, not DCCC/NRCC -- floor, not action, for a
  DCCC-vs-NRCC game.
- `outside` -- every other IE (super PACs, 527s): floor money that still
  compresses the persuasion ceiling like candidate spending does, but isn't
  a lever either optimizer can pull.

This was NOT a small correction. On the 433-race 2024 universe:

| | Old ("party" = total − candidate) | Corrected (x = national-committee-controlled) |
|---|---|---|
| DCCC budget | $465.2M | **$102.1M** |
| NRCC budget | $132.1M | **$47.2M** |
| D/R budget ratio | 3.5x | 2.2x |

NRCC's own independent expenditures ($48.4M more IE than only $3.1M of
coordinated spend, more than 16x the old NRCC "party" bucket that everyone
assumed was NRCC's decision) were previously indistinguishable from
Congressional Leadership Fund's, Club for Growth Action's, or any other
outside group's IE spending in the same district. 2022's corrected budgets
are DCCC $99.5M vs. NRCC $91.4M -- nearly even, a materially different
picture from what any "everything non-candidate" measure would show.
`compute_exploitability.py --cycle 2024` under the corrected budgets:
RegretD 2.36 (was 2.85), RegretR 3.23 (was 4.61) -- smaller absolute regret
with a smaller, correctly-scoped action space, as expected.

`src/game/*.py` needed ZERO code changes for this: `apply_control_floor()`
overwrites `RaceRecord.cand_d_total` (and returns a corrected
`cand_r_total` array) with the new floor BEFORE anything in `game/` ever
sees the race universe, so `party_d = d_total - cand_d_total` recovers
x_D automatically everywhere it's already computed that way.
`d_total`/`r_total` themselves are untouched -- the estimated margin model
still correctly sees TOTAL two-party spending (its persuasive effect
doesn't depend on who controls the money); only the game layer's notion of
the controllable action changed.

**Confirmed already correct, no fix needed**: IE party-alignment is
race/candidate-outcome-oriented (`(D candidate AND support) OR (R candidate
AND oppose)` per transaction, via `sup_opp`), not "spender is generally
R-aligned" -- `build_comprehensive_ie()` already did this right.

**Known open item, not fixed here** (flagged by the same critique): candidate
"disbursements" (`load_candidate_disbursements`) are FEC's broader
disbursement category, which can include refunds, transfers, and loan
repayments alongside genuine campaign-relevant expenditures -- FEC
distinguishes "disbursement" from "expenditure" and this project currently
uses the former uncleaned. Auditing this against the response model's own
estimation sample (so `cand_d_total`/`cand_r_total` and whatever the
persuasion-elasticity coefficients were actually fit on match) would need
FEC Schedule B purpose-code filtering -- a real, separate data-quality task,
not attempted in this pass.

## R-side state-party coordinated spending (closed 2026-08-11)

Before continuing to any historical backtest, we audited how R's spending
environment (`r_total`, `cand_r_total`, `budget_r`) is actually assembled
(`backtest.data.fec.build_total_spend`): candidate committee disbursements +
NRCC coordinated expenditures (FEC API) + comprehensive R-aligned
independent expenditures. The IE and candidate-committee channels are
symmetric methodology already (same alignment rule, same top-spender
selection, applied identically to both parties) -- the ~3.5x D/R gap in
observed 2024 independent expenditures ($479M vs. $136M) reads as a real
empirical fact about that cycle, not a coverage artifact.

One channel WASN'T symmetric: state party committees' own 24K coordinated
expenditures were scanned from the raw FEC bulk transaction files
(`all_committee_transactions/itoth*.txt`) for Democratic state parties only
(`identify_state_dem_party_committees`, inherited from the old project's
`FINDINGS.md` Section 10.7 Gap 3, never closed for R). Fixed by adding
`identify_state_rep_party_committees()` in `scripts/fetch_data.py`, verified
state-by-state against `committee_master` (43/50 states matched by the same
structural name pattern already used for D; the remaining 7 -- CO, NM, NV,
NY, OK, PA, VT -- added as individually-verified manual entries, mirroring
exactly how GA/LA/WV/WY were added on the D side). `parse_state_party_
coordinated_24k` is now `party`-parametrized instead of hardcoded to "D".

Ran for both 2022 and 2024 (the spec's initial two historical cycles):
$153,650 (2024) and $131,425 (2022) in previously-uncounted R state-party
coordinated spending, re-consolidated into `coordinated_expenditures_
{cycle}.csv`. NRCC's 2024 discretionary budget moved from $131.95M to
$132.11M -- real, but small enough that it does not change the qualitative
regret/exploitability picture from the earlier smoke test. `data/catalog/
data_catalog.md` has the full before/after table.

## Known open items (see spec Section 19, addressed only partially)

- **D/R elasticity symmetry test**: not implemented (`scripts/estimate_response_model.py::test_d_r_symmetry` raises `NotImplementedError` with what it would take). Until run, treat `MSG^R` as "D's mirrored elasticity," an assumption, not a tested symmetric response curve.
- **R-side concave surrogate validation**: not benchmarked against `backtest.optimizer.nash.best_response("R", ...)` the way the D-side surrogate was.
- **Level D five-way benchmark comparison**: only the Nash-vs-observed L1 distance is wired up so far.
