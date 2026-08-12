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

## Shared probability model: fixed baseline, signed saturation (rewritten 2026-08-12; supersedes the "D-anchored formula" design below)

**Status as of 2026-08-11 (superseded the same day):** the paragraph
originally here described `game/payoff.py` reusing the D-anchored
margin/ceiling formula for both sides, while `BR_R` still searched through
`backtest.optimizer.nash.best_response()`'s separate, uncalibrated mirrored
ceiling and only re-scored the result via the D-side formula afterward. That
split was flagged as the highest-priority open issue: a Nash equilibrium
requires each side to actually maximize the SAME utility function it's
reported against, and `BR_R` never did.

**The "obvious" fix broke worse.** Rewriting `BR_R` to search directly
against the literal D-anchored formula (`payoff.p_win`, `total_r` passed in
raw) -- "no old mirrored objective under the hood," exactly what the
one-formula requirement calls for -- was implemented, tested (gradient
correctness, local-optimum checks), and run against the real 2024 universe.
`regret_R` did not converge toward the old ~9.7-seat figure; it INCREASED,
to 16.06 (2022) and 9.66→16-range instability depending on run. Inspecting
the actual allocation explained why: R's optimizer dumped ~$3M each into
races like **HI-02** (R candidate-committee spend on record: **$10**),
TX-20, and NC-04 -- safe D seats with essentially no Republican presence --
dragging modeled D win probability from ~99% down to ~30-40%. Compare the
OLD mirrored-ceiling search: it spread R's budget across 134 races,
concentrated where R already had real committee money (FL-27, FL-28, NC-06,
six-to-seven-figure existing spend). The literal D-anchored formula was
never a symmetric two-player payoff -- it only regularizes `mu_raw`'s
excursion ABOVE `mu_floor` (D's own diminishing-returns cap, calibrated
against DCCC data); nothing bounds how far `mu_floor` itself falls as R's
total grows, because `mu_floor` is re-derived at whatever R total is
plugged in. The old mirrored ceiling, despite being uncalibrated, was doing
real regularization work the "exact" fix silently discarded. This was
caught before landing: the change was reverted (`git checkout`) rather than
shipped with a worse number than what it replaced.

**The real fix: one payoff, fixed baseline, signed saturation.**
`game/payoff.py::baseline_arrays/p_win_shared/grad_shared` replace the
moving D-side floor with a FIXED, party-neutral one:

```
mu_0_i    = mu_raw(F^D_i, F^R_i)                    # both sides' UNCONTROLLED floor
C_i       = c_max * 4*Phi(mu_0_i/sigma_i)*(1-Phi(mu_0_i/sigma_i))   # same shape as the old ceiling
Δmu_raw_i = mu_raw(F^D_i + x_D_i, F^R_i + x_R_i) - mu_0_i
Δmu_cap_i = C_i * tanh(Δmu_raw_i / C_i)              # SIGNED, symmetric saturation
mu_i      = mu_0_i + Δmu_cap_i                       # so mu_0_i - C_i < mu_i < mu_0_i + C_i, always
p_i       = Phi(mu_i / sigma_i)
```

`F^D`/`F^R` are `estimation.control_provenance.apply_control_floor`'s
existing uncontrolled floors (candidate + state party + outside --
`RaceRecord.cand_d_total` and the separately-returned `cand_r_total`
array), already threaded through this codebase; no new data dependency.
`C_i` keeps the original ceiling's exact functional form (`c_max *
4*p(1-p)`, already party-symmetric since `p(1-p) = (1-p)p`) -- only its
ANCHOR changes, from a moving D-side floor to the fixed two-sided baseline.
`tanh` (not the original one-sided `exp` form) is used because it is smooth
and antisymmetric by construction (`g(-z) = -g(z)`): there is exactly one
`p_i(x_D, x_R)`, used as both the search objective and the reported score
for both sides, with analytic gradients (`grad_shared`) differentiable
everywhere. `c_max` is still the single DCCC-calibrated value from
`config.yaml`, applied identically to R -- an explicit, stated assumption
(same status the old mirrored ceiling's `c_max` carried), not an
independently-validated NRCC figure.

Validated in stages before touching the live pipeline (all in
`tests/test_shared_payoff.py` and `tests/test_best_response_shared.py`):
bounds hold (`mu_0 - C < mu < mu_0 + C`) under arbitrarily large one-sided
spending by either side; HI-02/TX-20/NC-04-style near-zero-floor races no
longer collapse (a $3M unilateral R dump now saturates immediately, leaving
the seat at ~99%, not ~33%); the saturation step itself is sign-symmetric
(a full race-level D<->R mirror test does NOT hold, but that's the
underlying margin regression's `log(D/(D+R))` market-share form -- not
antisymmetric under swap the way `log(D/R)` log-odds would be -- a
pre-existing, separately-tracked property (Section 19), not something this
fix changes or should paper over); on the real 2022/2024 universes, the
ceiling binds on only 2-3% of races (the highest-spend, most competitive
ones), median saturation `q=0` -- it regularizes genuine extrapolation, not
ordinary observed data.

`game/best_response.py`, `gradients.py`, `exploitability.py`,
`persistent_value.py`, and `equilibrium.py` were all rewired onto
`p_win_shared`/`grad_shared`; `game/equilibrium.py::solve_nash` no longer
delegates to `backtest.optimizer.nash` at all -- both sides' best-response
solves and the Gauss-Seidel/Jacobi dynamics are self-contained in `game/`
now. `payoff.p_win`/`race_arrays_at` (the old D-anchored functions) and
`expected_seats_d`/`expected_seats_r` were deleted once nothing referenced
them; callers now use `p_win_shared` directly.

**A real divide-by-zero bug was caught in the rewire, not before.**
`grad_shared`'s first version computed `1/total_d` and `1/(total_d+total_r)`
using the UNCLAMPED totals, while `p_win_shared`'s own `_mu_raw` correctly
clamps both to `>= $1` (matching `predict_floor_margin`'s
`floor_dollars=1.0` convention). Any race with `F^D + x_D` exactly `$0`
(zero floor, zero party spend) hit a real division by zero, producing NaN
gradients that corrupted the SLSQP search silently enough to produce a
logically impossible `regret_D < 0` on the real universe -- caught by
`tests/test_exploitability_real_universe.py`'s existing ballpark assertion,
exactly the kind of bug class `docs/methodology.md`'s R-side rescoring fix
(2026-08-10) already warned this project is prone to. Fixed by clamping `d`,
`r`, `t` the same way inside `grad_shared` before dividing.

## Corrected headline exploitability (2026-08-12; supersedes the 2026-08-11 control-floor numbers above)

One-shot unilateral exploitability under the fixed, symmetric payoff,
replicating cleanly across both historical cycles:

| Cycle | RegretD | RegretR | E (total) | E as % of E[D seats] |
|---|---|---|---|---|
| 2022 | 3.03 | 2.41 | 5.44 | 2.53% |
| 2024 | 2.84 | 2.30 | 5.14 | 2.37% |

This REVERSES the asymmetric narrative every earlier version of this
project reported (RegretR > RegretD, "Republicans have more to gain than
Democrats"): RegretD is the larger term in both cycles now. That earlier
asymmetry was itself partly an artifact -- the OLD mirrored ceiling, while
better-behaved than the literal D-anchored formula, was still looser than
the fixed, two-sided ceiling both sides now share. `RegretD` barely moved
from the control-floor-fix numbers above (2.36/2.85 -> 2.84 now for 2024;
D's side was never the one exploiting an unregularized extrapolation).
`RegretR` roughly halved (2024: 3.23 under the mirrored ceiling -> 2.30
under the fixed ceiling; other historical versions of this project reported
figures as high as 9.66 for the same cycle under an even looser rescoring
convention). `docs/results_2022_2024.md` has been refreshed against this
payoff (2026-08-12) -- exploitability/taxonomy/PSV via exact SLSQP, Nash via
the validated surrogate below.

## Nash equilibrium: a real limit cycle, not a solver tuning issue (2026-08-12)

`docs/results_2022_2024.md` previously reported clean Nash convergence
(`converged: Yes`, 3-start agreement within ~$5-7K) under the OLD payoff.
Re-run under the fixed, symmetric payoff, `game/equilibrium.py`'s damped
Gauss-Seidel best-response dynamics do NOT converge to `RegretD(D*,R*) ~=
RegretR(D*,R*) ~= 0` -- they settle into a small, bounded, non-shrinking
oscillation instead, confirmed under two independent damping regimes:

| Cycle | theta | rounds | RegretD at star | RegretR at star | converged | last-20-round allocation delta |
|---|---|---|---|---|---|---|
| 2024 | 0.5 | 150 | 0.150 | 0.514 | No | oscillating $1.9M-5.6M, no shrinking trend |
| 2024 | 0.2 | 300 | 0.138 | 0.595 | No | oscillating $530K-1.4M, no shrinking trend |
| 2022 | 0.5 | 150 | 0.064 | 0.592 | No | oscillating $1.5M-7.2M, no shrinking trend |
| 2022 | 0.2 | 300 | 0.092 | 0.678 | No | oscillating $857K-2.4M, no shrinking trend |

Halving the damping step and doubling the round budget shrank the
oscillation's AMPLITUDE (roughly proportionally) but did not remove it --
`RegretR` at the fixed point was, if anything, slightly worse at the more
conservative setting. Two different damping regimes agreeing on "cycles,
doesn't converge" across two different historical cycles is strong evidence
this is a genuine property of the game's best-response dynamics, not
numerical noise or an under-tuned solver: the residual regret band
(~0.06-0.15 seats D-side, ~0.51-0.68 seats R-side) is small relative to the
observed-allocation regrets above (3.0/2.8 and 2.4/2.3), so the dynamics ARE
finding a genuine near-equilibrium region -- they just don't settle inside
it. Plausible explanations, none yet distinguished: a mixed-strategy
equilibrium (no pure-strategy fixed point exists for Gauss-Seidel to find);
multiple pure equilibria the dynamics orbit between; or a structural
near-tie in marginal race values that flips which races look most
attractive from round to round. `project_spec.md` Section 12 explicitly
asks for a "presence of cycles" diagnostic -- this is that diagnostic
firing, not a bug to chase further with more rounds at these damping
settings. `iterate_best_response`'s `cycle_detected` heuristic (checks only
whether the last 4 rounds' combined delta shrank monotonically) did NOT
flag this -- the oscillation period is longer than 4 rounds -- and should be
revisited if this becomes a tracked diagnostic rather than a one-off finding.

Raw run data: `results/nash_check_full.json` (theta=0.5), `results/
nash_check_lowdamp.json` (theta=0.2); per-cycle equilibrium allocations
saved as `results/nash_{full,lowdamp}_party_{d,r}_{cycle}.npy`.

## Concave-envelope surrogate, validated for BOTH sides (2026-08-12)

The SLSQP solves above are slow enough (~20-30s/call) that a thorough Nash
search (many rounds, several starts) costs hours. `src/optimizer/
concave_surrogate.py` already had a validated fast alternative for D
(`surrogate_allocate`, ~2,000-2,700x faster, within 0.11-0.19 expected seats
of the true optimum -- `scripts/theta_concave_surrogate.py`), but its R-side
mirror (`surrogate_allocate_r`) was built on the old mirrored-ceiling
formula and explicitly flagged as never validated -- `project_spec.md`
Section 12's "symmetrical validation for both players" was still
outstanding.

Since both sides now search the SAME `payoff.p_win_shared`, one surrogate
serves both: `game/best_response_surrogate.py::br_d_surrogate/
br_r_surrogate` reuse `build_concave_segments`/`greedy_allocate` UNCHANGED
(that machinery only needs a `payoff_fn(party, arrays) -> array` closure,
regardless of which formula it wraps), evaluating each race's own-side
payoff curve on a grid, taking its piecewise-linear concave upper envelope,
and solving the envelope relaxation exactly via greedy water-filling
(sort every race-segment by slope, fill highest-slope-first until budget is
exhausted) -- no SLSQP iteration at all.

Validated against exact SLSQP on both real cycles before being trusted for
anything: objective value agrees within **0.03-0.10 expected seats**
(BR_D: +0.061/2024, +0.096/2022; BR_R: +0.028/2024, +0.081/2022) at
**~500-1,000x speedup** (32.5s -> 0.033s for BR_D on 2024; 16.3s -> 0.034s
for BR_R). Allocation-level L1 differences are larger ($4-14M) than the
objective-value agreement would suggest -- consistent with this project's
own repeated finding that the objective has a wide, flat region near its
optimum (many different allocations achieve nearly the same aggregate
value), not evidence the surrogate is picking a meaningfully worse point.

Used to re-run the Nash search far more thoroughly than SLSQP could
feasibly manage: 3 starts x 2,000 rounds (10x the round budget of the
exact-SLSQP checks above) completed in ~7 minutes per cycle, confirming the
limit-cycle finding independently and more decisively -- see
`docs/results_2022_2024.md`'s Nash equilibrium section for the numbers.
Raw allocations: `results/nash_surrogate_party_{d,r}_{cycle}.npy`.

**A verification pitfall worth recording**: the first attempt to check
`RegretD(D*, R*)` at the surrogate's 2,000-round fixed point via exact
SLSQP came back NEGATIVE (`-0.17` for 2024) -- logically impossible on its
face, since `br_d`'s own argmax can never score worse than the specific
point `(D*, R*)` it's being compared against. Cause, found immediately: the
check called `br.br_d(..., x0=None)`, which defaults to starting SLSQP from
**zero** D spending -- a cold start that let SLSQP's local search settle
into a genuinely worse local optimum than `D*` (itself the product of 2,000
rounds of iterative improvement, i.e. already a good starting point). Not a
bug in `br_d`, `p_win_shared`, or the surrogate -- a bug in the CHECK.
Re-run with `x0=D*` (warm-started, so SLSQP can only find something at
least as good), both regrets came back cleanly positive
(`docs/results_2022_2024.md`'s numbers). Any future spot-check of a
best-response solver against a point it did NOT itself produce should warm-
start from that point, not from the solver's own default -- a cold start
can manufacture a spurious negative regret purely from which local optimum
SLSQP happens to land in, independent of whether the formula or the search
is actually correct.

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

- **D/R elasticity symmetry test**: not implemented (`scripts/estimate_response_model.py::test_d_r_symmetry` raises `NotImplementedError` with what it would take). Until run, treat `MSG^R` as "D's mirrored elasticity," an assumption, not a tested symmetric response curve. The 2026-08-12 payoff fix makes both sides share one formula, but `c_max` applied identically to R is still unvalidated for R specifically -- this test would also bear on whether a common `c_max` is defensible, not just `beta_D = beta_R`.
- ~~**R-side concave surrogate validation**~~: closed 2026-08-12 -- see "Concave-envelope surrogate, validated for BOTH sides" above. `game/best_response_surrogate.py` replaces the old, never-validated `surrogate_allocate_r`; benchmarked against the current `game/best_response.py::br_d/br_r` directly.
- **Level D five-way benchmark comparison**: only the Nash-vs-observed L1 distance is wired up so far.
- **Nash equilibrium does not converge under best-response dynamics** (2026-08-12): confirmed under two damping regimes on both cycles -- see the dedicated section above. Open questions: does a mixed-strategy equilibrium exist and is it computable here; do multiple pure equilibria exist and is the game's structure characterizable well enough to enumerate them; would a genuinely different dynamic (e.g. simultaneous/Jacobi rather than Gauss-Seidel, or a fictitious-play average-strategy tracker) behave differently. `docs/results_2022_2024.md` needs a full re-run against the current payoff before any of its headline numbers (exploitability OR Nash) are cited again.
