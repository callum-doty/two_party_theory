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

## Direct pure-strategy exploitability minimum: confirms the floor, doesn't explain it (2026-08-12)

The Nash section above establishes that damped Gauss-Seidel best-response
dynamics cycle rather than converge, settling into a bounded residual-regret
band (roughly 0.06-0.68 seats per side depending on damping/cycle) instead
of zero. That is evidence the DYNAMICS don't reach a zero-regret point --
it is NOT evidence that no such point exists nearby for the dynamics to
have missed. `scripts/minimize_pure_exploitability.py` asks the sharper
question directly: searching jointly over (D, R), not just wherever
alternating best-response happens to land, what is the lowest combined
regret `E(D,R) = RegretD + RegretR` actually reachable?

Method: (1) a 400-round x 3-start surrogate-driven best-response trajectory
scan that scores EVERY round's (D,R) pair, not just the final one; (2) 300
rounds of surrogate-scored local stochastic search (basin hopping) around
the best point found; (3) exact-SLSQP refinement (warm-started from the
candidate itself -- a cold zero-start SLSQP check on a point it didn't
produce can manufacture a spurious negative regret, as the surrogate
verification note above already found once) of the single best candidate.

| Cycle | Observed E | Best-on-trajectory (surrogate, round found) | After basin-hop | E_min (exact, warm-started) |
|---|---|---|---|---|
| 2024 | 5.14 | 0.374 (uniform start, round 3/400) | 0.373 | **0.484** (RegretD=0.449, RegretR=0.035) |
| 2022 | 5.44 | 0.767 (uniform start, round 5/400) | 0.767 (no improvement) | **0.862** (RegretD=0.730, RegretR=0.132) |

Three findings, read together:

1. **The trajectory's own minimum occurs almost immediately** (round 3-7
   of 400), not after the dynamics "settle" -- the oscillating band the
   dynamics spend most of their time in is not obviously better or worse
   than where they started converging toward it. There is no deep-round
   improvement being missed by only checking the endpoint.
2. **300 rounds of local stochastic search barely moves the needle**
   (2024: 0.374 -> 0.373; 2022: no improvement at all) -- a real, if
   cheap, local search around the best trajectory point does not find a
   materially lower-regret pure allocation pair nearby.
3. **The final exact E_min (0.48 / 0.86) lands within the same range this
   project's ORIGINAL, much shorter Nash-dynamics runs already reported**
   (`docs/results_2022_2024.md`: ~0.44 / ~0.76). A much more thorough,
   independently-designed search landing back in the same band is the
   right kind of confirmation: it says that band is a property of the
   payoff surface in this region, not an artifact of how many rounds or
   which damping schedule the original dynamics used.

This is evidence AGAINST a low-regret pure point existing in the
neighborhood the dynamics and local search actually cover -- not a proof
of global non-existence (433x2 dimensions is not exhaustively searchable),
and not yet a distinction between "no pure equilibrium exists" and "one
exists somewhere far from this neighborhood." It does shift the practical
question toward the mixed-strategy hypothesis rather than "try harder to
converge" -- see the fictitious-play and double-oracle sections below.

Raw output: `results/pure_exploitability_min_{cycle}.json`, allocations at
`results/pure_exploitability_min_party_{d,r}_{cycle}.npy`.

## Fictitious play: the time-average beats the best pure point found (2026-08-12)

The classic finite-zero-sum-game algorithm (Robinson 1951): each round,
both sides best-respond to the OTHER side's TIME-AVERAGE allocation so far
(not its most recent play, which is what the Gauss-Seidel/Jacobi dynamics
above do), and the realized best responses are folded into a running
average on each side. It is the pair of AVERAGES, not the last round's
pure best responses, that fictitious play's convergence guarantee is
about. That guarantee is a theorem for FINITE matrix games; applied here
to a continuous allocation space it is an empirical question, not a
theorem being invoked -- `payoff.p_win_shared`'s tanh saturation is not
globally linear in the opponent's spending, so best-responding to the
opponent's average allocation is not exactly the same as best-responding
to its distribution (`game/double_oracle.py`'s mixture best response,
below, does solve the exact version of that problem).

`game/equilibrium.py::fictitious_play` implements this (surrogate-scored
during the run, exact-SLSQP-checked once at the end, same division of
labor as the E_min search above). 400 rounds, two starts each cycle:

| Cycle | Start | Final round E(avg), surrogate | Last-20-round range | Exact E(avg) at best start |
|---|---|---|---|---|
| 2024 | observed | 0.237 | [0.195, 0.265] | -- |
| 2024 | uniform | 0.194 | [0.194, 0.211] | **0.359** (RegretD=0.106, RegretR=0.254) |
| 2022 | observed | 0.460 | [0.460, 0.501] | **0.666** (RegretD=0.167, RegretR=0.500) |
| 2022 | uniform | 0.487 | [0.425, 0.493] | -- |

The result worth noting: fictitious play's exact-checked average-pair
regret (0.359 for 2024) is LOWER than both the best pure point the direct
E_min search found (0.484) and the raw best-response orbit's residual
regret (~0.44) -- for 2022 it's in the same range as the orbit (0.666 vs.
~0.76) rather than clearly better, so this isn't a universal win, but 2024
shows a real gap in the direction the mixed-strategy hypothesis predicts:
a TIME-AVERAGED (i.e., mixed) strategy pair can out-perform anything found
by searching over PURE strategy pairs directly. That is the signature this
project's "Revised order of work" flagged fictitious play as a cheap first
diagnostic for, and it is consistent enough with a mixed equilibrium to
justify the full double-oracle solve below rather than treating the
E_min section's pure-strategy floor as the last word.

Raw output: `results/fictitious_play_{cycle}.json`, average allocations at
`results/fictitious_play_avg_party_{d,r}_{cycle}.npy`.

## Double-oracle mixed equilibrium: converges for both cycles (2026-08-12)

The full mixed-strategy solve (`game/double_oracle.py`, driven by
`scripts/double_oracle.py`): treat an entire 433-race allocation vector as
one pure strategy ("portfolio"), build a payoff matrix over a small,
growing pool of D and R portfolios, solve the finite zero-sum matrix game
EXACTLY via LP (`solve_zero_sum_matrix_game`, von Neumann minimax --
row-player and column-player LPs agreed to within `1e-12` seats on every
round run so far, i.e. no numerical daylight between the two sides' LPs),
then compute each side's EXACT best response to the opponent's mixture
(`br_d_to_mixture`/`br_r_to_mixture` -- not the fictitious-play shortcut of
best-responding to the opponent's average allocation, but the true
race-separable expectation over the discrete mixture) and add it to the
pool if it improves by more than `eps=0.02` seats. Classic double-oracle
(McMahan, Gordon & Blum 2003): the strategy space grows with the search
instead of being fixed in advance.

Both cycles seeded with the same 6 portfolios per side: observed, uniform,
zero, the one-shot unilateral best response, the E_min search's best
candidate, and the fictitious-play time-average -- i.e., every
strategically-motivated allocation already computed for that cycle.

**2024: converged after 13 rounds.** Final pools 15 (D) / 13 (R); support
shrinks to 5 portfolios per side (`d_gain`/`r_gain` both under 0.02 seats
at the stopping round). Value (E[D seats] at the mixed equilibrium) =
**218.60** -- matching the E_min search's 218.31 and the raw BR-dynamics
orbit's ~218.3-218.6 (`docs/results_2022_2024.md`) to within the same
noise band all three independent methods have been landing in. All 5 of
D's support portfolios were discovered DURING the oracle search (none of
the 6 seeds survive in D's final mixture); R's mixture keeps the
fictitious-play average (weight 0.22) alongside 4 oracle-discovered
portfolios.

**2022: did not converge in the first 25 rounds** -- pools grew to 31 (D) /
30 (R), support grew to 10/10, and `d_gain`/`r_gain` were still bouncing in
the 0.02-0.5 range with no visible shrinking trend by round 24 (contrast
2024's clean monotonic approach to `eps` in 13 rounds). Resumed from the
grown 31/30 pool (not restarted) for up to 40 more rounds --
**converged after 27 additional rounds** (52 combined), final pools 53 (D)
/ 48 (R), support 11/11. Value = **216.375** -- matching the E_min search's
216.07 and the raw BR-dynamics orbit's ~215.2-216.5 (`docs/
results_2022_2024.md`). So 2022 DOES have a mixed equilibrium the
double-oracle process finds -- it just needed roughly 4x the rounds and
resulted in roughly double the support size (11 vs. 5) that 2024 needed.
Raw output: `results/double_oracle_2022_resumed.json`.

**Reading this together with the pure-strategy sections above**: both
cycles now have three independent methods (E_min direct search, fictitious
play, double oracle) landing on compatible values -- 2024 in the
218.3-218.6 band, 2022 in the 215.2-216.5 band -- with double oracle
additionally showing that a small-support mixture (5 portfolios for 2024,
11 for 2022) achieves a value neither the best pure point found (E_min:
218.31 / 216.07) nor the raw orbit fully reaches without residual regret --
direct evidence for "no stable deterministic portfolio, but a stable
distribution over a HANDFUL of near-optimal targeting portfolios," the
working thesis this project's roadmap flagged as the more original result
if it held up. The real remaining asymmetry is in DIFFICULTY, not
existence: 2022's equilibrium support is roughly double 2024's and took
roughly 4x the double-oracle rounds to find -- worth understanding (a
genuinely flatter/more contested landscape in 2022? more near-tied
marginal races?) rather than treating as a solved footnote now that both
converged.

Raw output: `results/double_oracle_{cycle}.json`; per-portfolio allocations
at `results/double_oracle_{d,r}_portfolio_{i}_{cycle}.npy`.

## Equilibrium support composition: core, swing, and irrelevant races (2026-08-12)

A mixed equilibrium over 5-11 full 433-race portfolios per side is correct
but not directly readable -- "here is a probability distribution over
433-dimensional vectors" doesn't say anything a person can act on.
`scripts/equilibrium_support_composition.py` collapses it to a per-race
summary: for each race i and side, across that side's support portfolios
`{portfolio_j}` with mixture weights `{p_j}`,

    E[w_i]   = sum_j p_j * portfolio_j[i]
    Var[w_i] = sum_j p_j * (portfolio_j[i] - E[w_i])^2
    CV[w_i]  = sqrt(Var[w_i]) / E[w_i]

then buckets races into three categories, thresholds stated explicitly
(not tuned): **irrelevant** (E[w_i] below 1% of that side's per-race cap --
essentially never funded across the support); among the remaining
materially-funded races, split by the MEDIAN CV into **core** (below-median
CV -- nearly every support portfolio funds this race about the same
amount) and **swing** (above-median CV -- funding depends heavily on which
equilibrium portfolio gets drawn).

| Cycle | Side | Core | Swing | Irrelevant | Cap | Median CV (funded) |
|---|---|---|---|---|---|---|
| 2024 | D | 30 | 29 | 374 | $15.3M/race | 0.034 |
| 2024 | R | 26 | 26 | 381 | $7.1M/race | 0.169 |
| 2022 | D | 38 | 37 | 358 | $14.9M/race | 0.045 |
| 2022 | R | 36 | 35 | 362 | $13.7M/race | 0.195 |

**2022's richer equilibrium (11-portfolio support, "Double-oracle mixed
equilibrium" section above) touches more races, not just more portfolios**:
75 D-side races funded at all (38+37) vs. 59 in 2024; 71 R-side vs. 52 --
consistent with 2022 having a genuinely flatter or more contested strategic
landscape (more near-tied marginal races generating more substitutable
portfolio configurations) rather than 2022's larger support size being an
artifact of the double-oracle search just running longer.

**A concrete illustration of what "swing" means**: several of 2024's
top-CV R-side races (NC-14, NC-06, NV-01, FL-22, NC-13) are funded by
EXACTLY ONE of the five support portfolios and zero in the other four --
literally "included on the target list" in ~22% of draws (that portfolio's
mixture weight) and off it otherwise, not a race that gets a little more
or less money depending on the draw. (Mechanically, a race funded by only
one portfolio has `CV = sqrt((1-w)/w)`, a function of that portfolio's
weight `w` ALONE, independent of the dollar amount -- several races tied at
CV=1.89 in 2024's R-side output for exactly this reason, before the
mean-based tiebreak below was added to the ranking.)

Top races by category (full table: `results/
equilibrium_support_composition_{cycle}.csv`):

- 2024 D-side core (always funded): VA-10, WI-01, CA-40, NJ-02, NY-02
- 2024 D-side swing (funding depends on the draw): AZ-09, CT-05, AZ-04, CT-02, CA-31
- 2024 R-side swing: NC-14, NC-06, NV-01, FL-22, NC-13
- 2022 D-side swing: CT-02, NC-06, NY-20, MD-03, CA-06
- 2022 R-side swing: OK-05, FL-07, PA-04, TX-32, TN-09

This is the operational form of the "small distribution over near-optimal
portfolios" finding: a committee reading this wouldn't see "here is your
one target list," but "these ~30 races are core to every version of the
strategy; these ~29 are where staying unpredictable actually matters;
everything else isn't part of the equilibrium's support at all."

Raw output: `results/equilibrium_support_composition_{cycle}.json` (summary
+ top-N lists per category) and `.csv` (full per-race table).

## Level D five-way benchmark: H3 REJECTED, both cycles (2026-08-12)

Spec Section 20's fourth validation level, previously unbuilt (`docs/
results_2022_2024.md` flagged it as the clear next step): compare the
OBSERVED allocation against equal allocation, a Cook-category heuristic,
the one-sided optimizer, the (mixed) equilibrium, and random feasible
portfolios, by L1 distance and E[D seats]. This is the test spec's H3
("observed allocations are substantially closer to Nash equilibrium than
to unilateral optima") actually needs.

`game/benchmarks.py` builds the three non-solver strategies (equal, Cook
heuristic, random feasible), each defined for BOTH sides so "E[D seats]"
means D's-version-of-the-strategy against R's-version-of-the-strategy, not
one side's benchmark against the other's OBSERVED allocation:

- **Equal**: uniform across all 433 races.
- **Cook heuristic**: proportional to a fixed competitiveness weight by
  Cook category (Toss-Up=5, Lean=3, Likely=1, Safe=0 -- same weights
  regardless of which party currently favors the race), capped per race
  and redistributed (`cap_and_redistribute`, a fixed-proportions water-
  filling analogue for benchmarks that aren't themselves optimizer output).
- **One-sided optimizer**: `BR_D(R_observed)` vs. `BR_R(D_observed)`,
  played against EACH OTHER (not against the observed opponent that
  produced them).
- **Mixed equilibrium**: the double-oracle LP value (the game-theoretically
  correct E[D seats] under the mixture -- NOT recomputed from the
  mixture's average portfolio, since `p_win_shared` is nonlinear in the
  opponent's spending); L1 distance uses the mixture's EXPECTED portfolio
  (`sum_j p_j * portfolio_j`) as a descriptive summary only, reported
  separately from the value for exactly that reason. 2022 uses the
  converged extended double-oracle run (`double_oracle_2022_resumed.json`,
  53/48-portfolio pools), not the original 25-round run that hadn't
  converged.
- **Random feasible**: mean over 20 independent Dirichlet-weighted random
  portfolios per side.

`scripts/level_d_benchmark.py`, run on both cycles:

| Cycle | Closest to observed (L1) | 2nd | 3rd | 4th | Farthest |
|---|---|---|---|---|---|
| 2024 | **Cook heuristic** ($192.1M) | equal ($271.4M) | random ($273.2M) | one-sided ($292.6M) | mixed equilibrium ($293.1M) |
| 2022 | **Cook heuristic** ($203.5M) | equal ($335.0M) | random ($338.3M) | one-sided ($344.7M) | mixed equilibrium ($349.2M) |

**Both cycles replicate the same ranking, and it is the opposite of H3.**
Observed DCCC/NRCC spending is closest, by a wide margin, to a simple
Cook-category competitiveness heuristic -- NOT to either the one-sided
optimizer or the mixed equilibrium, which are (within noise) tied for
FARTHEST from observed on both cycles. Equal allocation and random
feasible portfolios both land closer to observed than either
optimization-derived benchmark does, in both cycles.

E[D seats] tells a consistent story: Cook heuristic (218.14 / 215.82) is
close to observed (217.17 / 215.17), while the optimization-derived
strategies score higher (one-sided: 218.50 / 216.40; mixed equilibrium:
218.60 / 216.38) -- the strategies that are FARTHEST in allocation space
also perform BEST, exactly the pattern you'd expect if committees are
optimizing much less aggressively than either benchmark and instead
following something closer to a competitiveness-proportional rule of
thumb.

**Per spec Section 27's decision gate: H3 is REJECTED, replicated across
two independent cycles.** This closes out this project's last open Level D
question with a clean, if not the hoped-for, answer: real committee
behavior in this dataset does not resemble either a unilateral optimum or
a game-theoretic (mixed) equilibrium nearly as much as it resembles a
much simpler heuristic. That is itself informative for how to read every
other result in this project -- the exploitability/regret numbers
(observed E~5.1-5.4 seats) and the mixed-equilibrium characterization are
descriptions of the STRATEGIC GAME'S structure, not evidence that real
committees are approximately playing it.

Raw output: `results/level_d_benchmark_{cycle}.json`.

## D/R elasticity symmetry test: rejected under a linear specification, cannot be rejected under a flexible one (2026-08-12, revised twice same day)

Spec Section 19: "Do not simply mirror the Democratic response curve onto
Republicans without testing it." `game/payoff.py`'s shared formula uses ONE
coefficient, `coef.beta1` (== `beta_rc.estimate`), for the log-spending-
share-ratio term `c_spend * log(x_D/(x_D+x_R))` applied identically to both
sides' dollars -- so the real question was never "does the formula treat D
and R differently" (it doesn't, by construction), but "was that one
coefficient ever estimated on a sample that could reveal an asymmetry if
one existed." Until now, `beta_rc.identify_repeat_pairs` only ever selected
the sample where the DEMOCRAT is the repeat challenger (`incumb_status ==
"Challenger"`, R holds the seat) -- beta1 had only ever been tested on "D
attacking an R-held seat," never the mirror image.

`identify_repeat_pairs` now takes a `challenger_party` parameter
(`scripts/estimate_response_model.py::test_d_r_symmetry`): `"D"` is the
original sample; `"R"` selects `incumb_status == "Incumbent"` (D holds the
seat, R is the repeat challenger), using the SAME margin_pp/D-spending-
share units, so the two fits are directly comparable. (Fixing this also
surfaced a latent bug: `_normalize_name` crashed on `NaN` challenger names,
which the "D" sample happened to never contain but the "R" sample does --
now filtered out before matching, for both branches.)

| Sample | n pairs | beta estimate | SE | 95% CI |
|---|---|---|---|---|
| D-challenger (original beta_rc) | 118 | 5.47 | 1.59 | [2.36, 8.59] |
| R-challenger (mirror-image) | 143 | 24.17 | 7.60 | [9.28, 39.06] |

The confidence intervals do not overlap. A pooled OLS with a challenger-
party interaction term (equivalent to a Chow test for equal slopes, same
HC3 robust covariance `estimate_beta_rc` already uses) puts the difference
at 18.69 (SE 7.76, t=2.41, **p=0.016) -- symmetry is REJECTED at
alpha=0.05.**

**Important caveat before over-reading the 4.4x magnitude**: the two
samples' identifying variation is wildly different in scale.
`delta_log_ratio` (the regressor) has std=1.88 (range -6.77 to 2.64) in the
D-challenger sample but std=0.177 (range -0.67 to 0.78) in the R-challenger
sample -- roughly 10x narrower. That thinner identifying variation is
exactly why beta_R's SE (7.60) is ~5x beta_D's (1.59). But there is a
sharper, more consequential version of this same observation, below.

### Common-support re-test: the two samples occupy almost disjoint regions of the spending-share axis

`identify_repeat_pairs` now also returns `ratio_prev`/`ratio_curr` (D's
share of combined D+R spend at each endpoint of a pair) so this can be
checked directly, not just inferred from the regressor's variance.
`ratio_mid = (ratio_prev + ratio_curr) / 2` summarizes where on the
spending-share axis each pair sits:

| Sample | mean D-share | median D-share |
|---|---|---|
| D-challenger | 0.222 | 0.112 |
| R-challenger | 0.899 | 0.960 |

These are not overlapping ranges -- they are close to OPPOSITE extremes.
D-challenger pairs are races where D (the challenger) is chronically
outspent by R; R-challenger pairs are races where R (the challenger) is
outspent even more severely by the D incumbent (median R gets only ~4% of
combined spending). This matters because the whole reason this project's
payoff formula saturates with a tanh ceiling (`payoff.p_win_shared`'s
module docstring, and the HI-02 incident it documents: a $10-on-record R
candidate producing an absurd extrapolation) is that the raw log-ratio
relationship is not linear everywhere -- it is steepest near the extremes
and flattens elsewhere. Measuring "beta" in a sample that sits almost
entirely in the extreme, barely-funded-challenger region (R-challenger,
median R-share ~4%) versus a sample sitting in a more moderate
underfunded-challenger region (D-challenger, median D-share ~11%) risks
comparing the LOCAL SLOPE at two different points on a concave curve, not
a clean party-vs-party difference.

`scripts/estimate_response_model.py::common_support_symmetry_test` re-runs
the full test after trimming BOTH samples to a shared `ratio_mid` band:

| Band | n (D, R) | beta_D | beta_R | diff | p-value | Verdict |
|---|---|---|---|---|---|---|
| Untrimmed | 118, 143 | 5.47 | 24.17 | 18.69 | **0.016** | Reject |
| [0.10, 0.90] | 63, 46 | 10.04 | 19.07 | 9.03 | **0.188** | Cannot reject |
| [0.20, 0.80] | 44, 24 | 8.48 | 17.15 | 8.67 | **0.188** | Cannot reject (R below min-pairs threshold, SE unreliable) |

Restricting to the [0.10, 0.90] band -- the widest window where BOTH
trimmed samples still clear `min_repeat_pairs` (40) -- the point estimates
move much closer together (10.04 vs. 19.07, versus 5.47 vs. 24.17
untrimmed) and the interaction test's p-value rises from 0.016 to 0.188:
**symmetry can no longer be rejected at conventional significance once
both samples are compared on overlapping ground.** The direction persists
(beta_R still numerically larger in every band) but is no longer
statistically distinguishable from sampling noise at this sample size.

**Reading the two results together**: the untrimmed rejection was real but
substantially confounded by near-disjoint support -- a large part of what
looked like "R's dollars are more persuasive than D's" is better explained
by "the two samples measure the log-ratio relationship at very different,
extreme points on a concave curve, and a linear-in-log-ratio specification
does not extrapolate cleanly between them." This does not resupport strict
symmetry either (the direction and rough magnitude of the gap survive
trimming, just without significance at n=46-63) -- the honest state of
evidence is "a real but small-sample-uncertain asymmetry, most of the
apparent 4.4x untrimmed gap driven by comparing different regions of a
nonlinear curve rather than a clean party effect."

**What this means for the current payoff model**: `game/payoff.py`'s
single shared `c_spend` (and the identically-applied `c_max` persuasion
ceiling) remains an approximation, but the case for urgently rebuilding it
around party-specific elasticities is weaker than the untrimmed result
alone suggested -- the common-support estimates (10.04 vs. 19.07) are
closer together and their difference is no longer significant. Rebuilding
the payoff around party-specific elasticities is still real estimation and
re-derivation work (a new `c_spend_R` distinct from `c_spend_D`, threaded
through `baseline_arrays`/`p_win_shared`/`grad_shared`, and every
downstream BR/exploitability/equilibrium result re-run against it) -- out
of scope to fold into this fix, and now a lower priority than it looked
from the untrimmed test alone. Any claim that this
project's results are symmetric OR asymmetric between the two parties
needs this section (not just the untrimmed table above it) cited.

Raw output: `results/d_r_symmetry_test.json` (untrimmed),
`results/d_r_symmetry_common_support.json` (banded re-test).

### Nonlinear g(s) re-test: symmetry holds under the better-supported specification

The common-support re-test above is still built on the SAME linear-in-log-
ratio functional form as `beta_rc` -- trimming the sample to overlapping
support sidesteps the disjoint-region problem, but doesn't test whether
"linear in log-spending-share" is even the right shape to begin with. This
final check replaces the single elasticity with a genuinely flexible
common response function and asks whether the symmetry conclusion survives
a specification that isn't tied to one particular transform.

`scripts/estimate_response_model.py::nonlinear_common_curve_test` fits
`g(s) = sum_k beta_k * (s - 0.5)^k` (s = D's spend share, `s - 0.5`
centered to cut collinearity between powers), via
`delta_margin = g(s_curr) - g(s_prev) + eps` -- g's own intercept cancels
exactly under first-differencing (same district, same challenger), so this
is still one OLS fit, just against a differenced polynomial design matrix
instead of a single `delta_log_ratio` column. Pooling the D- and
R-challenger samples with a full interaction (every power term x
`is_r_challenger`) and F-testing the interaction block jointly is the
nonlinear generalization of the single-coefficient Chow test used above.

| Degree | Untrimmed p-value | [0.10, 0.90] p-value |
|---|---|---|
| 2 (quadratic) | **0.0011 — reject** | **0.0169 — reject** |
| 3 (cubic) | **0.799 — cannot reject** | **0.438 — cannot reject** |

The conclusion FLIPS between degree 2 and degree 3 -- not a result to
quietly pick the answer you prefer from. Resolved by testing which
specification the data actually supports: a nested F-test on whether the
cubic terms improve the fit over the quadratic (`common_u3` and
`interact_u3` jointly zero) rejects at **p=0.013** -- the cubic curvature
is not overfitting noise, it is a real feature of the data the quadratic
model is too rigid to capture. Once that curvature is allowed for, the
apparent D/R difference the quadratic (and the original linear-in-log-
ratio) specification found is no longer distinguishable from sampling
noise. **The best-supported specification available says: cannot reject
symmetry.**

Caveat on reading the coefficients themselves (not the F-test, which is
specification-appropriate regardless): the D-challenger sample's mass sits
around s~0.22 and the R-challenger sample's around s~0.90 (the same
disjoint-support fact from the common-support section above) -- almost no
data sits near s=0.5, where `(s-0.5)^k` is centered. The fitted
`common_coefficients`/`interaction_coefficients` describe the curve's
local behavior at a point with little identifying data, so they should not
be read as "the elasticity at 50/50 spending" -- the F-test on the joint
interaction block is the reliable object here, not the individual
coefficient values.

Three independent lines of evidence now say the same thing: (1) common-
support trimming under the original linear specification, (2) a flexible
nonlinear common curve fit to the FULL untrimmed sample (no trimming
needed), and (3) that nonlinear fit's own internal model-selection check
confirming the flexible specification is the right one to trust. All three
point to "symmetry cannot be rejected once the functional form is not
forced to be linear in log-spending-share" -- this is now the strongest
evidence in the project for treating `game/payoff.py`'s single shared
`c_spend` as a reasonable approximation, not just an untested one.

Raw output: `results/d_r_symmetry_nonlinear.json`.

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
| B | Nonlinear vs. surrogate agree? | Closed 2026-08-12 -- `game/best_response_surrogate.py` validated for BOTH sides against exact SLSQP (see "Concave-envelope surrogate, validated for BOTH sides" above): objective agrees within 0.03-0.10 expected seats at ~500-1,000x speedup. Supersedes the old project's D-only `theta_concave_surrogate.py` validation and the never-validated R-side `surrogate_allocate_r`. |
| C | Algorithm recovers known synthetic equilibria? | `src/validation/synthetic_games.py`, `tests/test_synthetic_game.py`. Standalone logistic-contest game, decoupled from the estimated election model on purpose (isolates the ITERATION algorithm from the SUBSTANTIVE model). |
| D | Observed allocations closer to Nash than alternatives? | Closed 2026-08-12 -- see "Level D five-way benchmark: H3 REJECTED, both cycles" above. `scripts/level_d_benchmark.py` runs the full five-way comparison (equal / Cook heuristic / one-sided optimizer / mixed equilibrium / random feasible); observed spending is closest to the Cook heuristic and farthest from both optimization-derived benchmarks, on both cycles. |

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

## Candidate "disbursements" audit: real bias, confirmed null effect on every headline result (2026-08-12)

The open item above flagged that `load_candidate_disbursements` uses FEC's
TTL_DISB (weball col 7, "total disbursements") uncleaned -- a broader
category than genuine campaign expenditure, since FEC's own totals API
separately reports `operating_expenditures` alongside `other_disbursements`,
`loan_repayments`, `transfers_to_other_authorized_committee`, and
`contribution_refunds`. This audit checks (a) whether the gap is real and
how large, and (b) whether it actually moves anything this project reports.

**(a) The gap is real and can be large, concentrated in a specific,
identifiable archetype.** Pulled two known House leadership figures'
`/candidate/{id}/totals/` from the FEC API (`api.open.fec.gov`, confirmed
against the live data, not estimated):

| Candidate | Role | District (safety) | TTL_DISB (`cand_d`) | `operating_expenditures` | Gap |
|---|---|---|---|---|---|
| Nancy Pelosi | Speaker (2022) | CA-11 (Safe D) | $28.28M | $10.81M | **62% is `other_disbursements` ($16.92M)** |
| Hakeem Jeffries | Minority Leader (2024) | NY-08 (Safe D) | $20.24M | $14.61M | **28% is `other_disbursements` ($5.48M)** |

`other_disbursements` for a party leader is dominated by redistribution to
other campaigns via their fundraising apparatus -- real money, just not
this district's own campaign spending. This is a genuine, confirmed
overstatement of `cand_d_total` for these specific candidates -- not a
hypothetical.

**(b) But it provably cannot move any strategic result this project
reports**, for two independent reasons:

1. **The decision variable is mathematically insulated from `cand_d`'s
   magnitude.** `apply_control_floor` sets `floor_d = cand_d + party_state_d
   + outside_d` ("x_D / x_R must be CONTROLLED money" section above), and
   `d_total` is untouched. So `party_d_obs = d_total - floor_d =
   party_natl_d` EXACTLY -- the DCCC's own coordinated spend + national IEs,
   an entirely SEPARATE data source (FEC coordinated-expenditure and IE
   filings, not the candidate committee's disbursement total). However
   inflated `cand_d` is, it cancels out of this subtraction identically;
   `budget_d`, every `BR_D`/`BR_R` call, every exploitability/equilibrium
   number in this project is computed from `party_d_obs`/`party_r_obs`, not
   from `cand_d_total` directly.

2. **The one channel that DOES depend on `cand_d_total`'s magnitude (the
   baseline `mu_0` and persuasion ceiling `C_i` -- see `payoff.py`'s
   module docstring) is a non-issue for exactly these races.** Checked
   directly against `results/race_surplus_2024.csv`: EVERY 2022/2024 House
   leadership figure or DCCC/NRCC chair with a district in this universe
   (Pelosi CA-11, Jeffries NY-08, McCarthy/Johnson-era leadership seats,
   Emmer MN-06, Hudson NC-09, DelBene WA-01, Clark MA-05, Clyburn SC-06,
   Scalise LA-01/LA-04) sits in a Safe D or Safe R seat with `MSG_D` between
   `1e-8` and `1e-17` -- numerically saturated. The sigmoid is already flat
   at these races' true margins regardless of whether the floor is off by
   0%, 30%, or 60%; no realistic correction changes `p_win_obs`, `MSG_D`,
   or the "possible over-capitalization" classification these races already
   get.

**Verdict**: confirmed, quantified, real data-quality issue -- and a
confirmed NULL effect on this project's exploitability numbers, best-
response dynamics, mixed equilibrium, Level D benchmark, and PSV/race
taxonomy, because of (1)'s exact cancellation and (2)'s saturation, not
because the bias is merely "probably small." A full fix (re-fetching
`operating_expenditures` for every candidate via the FEC's per-candidate
totals endpoint instead of the bulk weball file) would still be worth doing
before citing any SPECIFIC leadership district's individual numbers in a
race-level table -- those two entries in `race_surplus_2024.csv` are
individually still using the inflated floor -- but is not a prerequisite
for anything already reported. Not attempted here beyond the two confirmed
spot-checks: `api.open.fec.gov`'s `DEMO_KEY` is rate-limited to 10
requests per window, exhausted during this audit; a full fix needs a free
registered key (https://api.open.fec.gov/developers).

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

- ~~**D/R elasticity symmetry test**~~: closed 2026-08-12 -- see "D/R elasticity symmetry test: rejected under a linear specification, cannot be rejected under a flexible one" above. Untrimmed linear-in-log-ratio: `beta_D`=5.47 vs. `beta_R`=24.17 reject symmetry at p=0.016 -- but the two samples sit at nearly disjoint D-spending-share regions (mean 0.22 vs. 0.90). Common-support trimming ([0.10,0.90]) narrows the gap (10.04 vs. 19.07, p=0.188, cannot reject). A genuinely flexible common curve `g(s)` (cubic in spending share, fit to the FULL untrimmed sample, no trimming needed) also cannot reject symmetry (p=0.80) -- and a nested F-test confirms the cubic terms are themselves statistically justified (p=0.013), so this is the better-supported specification, not a convenient pick (a quadratic version of the same flexible approach DOES still reject, p=0.001 -- degree matters, and the data prefers the degree under which symmetry holds). Net verdict: `game/payoff.py`'s single shared `c_spend` is a reasonable approximation under the best-supported specification tried; a party-specific rebuild is not motivated by this evidence.
- ~~**R-side concave surrogate validation**~~: closed 2026-08-12 -- see "Concave-envelope surrogate, validated for BOTH sides" above. `game/best_response_surrogate.py` replaces the old, never-validated `surrogate_allocate_r`; benchmarked against the current `game/best_response.py::br_d/br_r` directly.
- ~~**Level D five-way benchmark comparison**~~: closed 2026-08-12 -- see "Level D five-way benchmark: H3 REJECTED" above. Observed spending is closest to a Cook-category heuristic and roughly tied for farthest from both the one-sided optimizer and the mixed equilibrium, on both cycles. H3 does not hold in this data.
- ~~**Candidate "disbursements" include non-campaign spending (transfers/loan repayments/refunds)**~~: closed 2026-08-12 -- see "Candidate 'disbursements' audit" above. Confirmed real via the FEC API for two known House leadership figures (Pelosi CA-11: 62% of TTL_DISB is `other_disbursements`; Jeffries NY-08: 28%) -- but PROVEN to have zero effect on every result this project reports: `party_d_obs`/`party_r_obs` (the actual decision variables) are mathematically insulated from `cand_d_total`'s magnitude by the floor/party accounting identity, and the only channel that does depend on it (`mu_0`/`C_i`) only matters for races that are already numerically saturated regardless (every affected candidate is a known leadership figure in a Safe D/R seat, `MSG_D` between 1e-8 and 1e-17). A full fix (real FEC API key, re-fetch `operating_expenditures` for every candidate) would still be worth doing before citing any specific leadership district's individual numbers, but isn't a prerequisite for anything already reported.
- ~~**Nash equilibrium does not converge under best-response dynamics**~~: addressed 2026-08-12 -- "Direct pure-strategy exploitability minimum," "Fictitious play," "Double-oracle mixed equilibrium," and "Equilibrium support composition" sections above. Direct search confirms the residual-regret floor is not a search-thoroughness artifact (E_min lands back in the same 0.48-0.86 band the original dynamics found); fictitious play's time-average beats the best pure point found for 2024; double oracle finds a small-support mixed equilibrium on BOTH cycles (converged, not just 2024); the support decomposes into interpretable core/swing/irrelevant races. Still genuinely open, not chased further on purpose (per this project's own roadmap discussion): whether multiple PURE equilibria exist and the game's structure is characterizable well enough to enumerate them -- the evidence collected supports "no low-regret pure point found under extensive search, mixed equilibrium computed successfully," not the stronger, unproven claim "no pure equilibrium exists." `docs/results_2022_2024.md` is already the post-refresh version (its own intro states this) -- no outstanding re-run needed.
