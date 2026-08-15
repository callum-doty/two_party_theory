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

## Strategic leverage and response displacement (2026-08-13)

Follow-on to persistent_value.py, prompted by a research-direction discussion about whether a durable edge could come from forcing the opponent into a *costly* response rather than searching for another static allocation it can't counter (the double-oracle/Level-D results above already show that search is close to exhausted: the mixed equilibrium is found, and observed committees don't even play close to it). The discussion's proposal was a "counter-response cost" statistic -- how much does a Democratic move cost Republicans, measured by the opportunity value they sacrifice responding to it.

**Algebraic finding before writing any code**: in this project's constant-sum payoff (`U_R = n - U_D`, true of every `p_win_shared` caller), "how much does D's move cost R" -- `U_R(D_obs, BR_R(D_obs)) - U_R(D', BR_R(D'))` -- reduces, after substituting `U_R = n - U_D` and cancelling `n`, to `U_D(D', BR_R(D')) - U_D(D_obs, BR_R(D_obs))`. That is exactly `persistent_value.py`'s `PSV_i^D` under the isolated baseline. **"Republican opportunity cost" is not a new number sitting on top of PSV -- it IS PSV**, read off the other side of the same zero-sum ledger. `src/game/strategic_leverage.py` does not recompute it under a new name; its module docstring has the full derivation.

What's actually new, implemented in `src/game/strategic_leverage.py` + `scripts/compute_strategic_leverage.py`:

1. **The response-displacement map** -- `BR_R(D') - BR_R(D_obs)`, race by race: which specific races the opponent's best response pulls money out of, or into, to answer a move elsewhere. Nothing in this codebase exposed this before; `persistent_value.py` computes the full opponent allocation internally and discards everything but the aggregate PSV scalar.
2. **`reshuffle_l1` / `reshuffle_per_million`** -- total opponent dollars disrupted per $1M committed. Distinct from PSV/leverage: reshuffling can be large even when it nets to a small aggregate seat swing (self-cancelling reallocation in the zero-sum accounting), so PSV alone cannot tell a "decoy" race (opponent forced to reshuffle a lot for little of its own aggregate seat change) from a race the opponent can simply shrug off.
3. **A leverage curve** across delta = $250K/$500K/$1M/$2M for the top few candidates per side, to see whether PSV-per-dollar is roughly constant or diminishing as commitment size grows.

**A real surrogate failure, caught by the first run, not before.** The first version of this script used `best_response_surrogate.py` (validated elsewhere to within 0.03-0.10 expected seats of exact SLSQP on AGGREGATE objective value) for speed, since a multi-delta multi-race sweep at exact-SLSQP cost seemed impractical. First real result: 2024 D-side, WI-01 at $1M, surrogate reported `reshuffle_l1 = $4,124,880`. The exact-SLSQP recheck of the exact same point found `reshuffle_l1 = $16` -- Republicans' true best response barely moves at all (`retention = 98.7%`); the "$4.1M reshuffle" was pure surrogate artifact. This is a sharper reading of a caveat `docs/methodology.md`'s own "Concave-envelope surrogate" section already recorded ("allocation-level L1 differences are larger ($4-14M) than the objective-value agreement would suggest") -- that note was about the surrogate's allocation being a worse descriptive object, not about whether it could be trusted for a *difference* of two allocations, which is exactly what a displacement map is. It can't be: aggregate-objective accuracy does not bound allocation-vector accuracy well enough to difference against another allocation vector. Fixed by dropping the surrogate entirely and paying exact-SLSQP cost throughout, trimming the candidate pool (top-4-swing + top-4-|Z| funded per side, from `equilibrium_support_composition_{cycle}.json` and `exploitability.race_level_surplus`) and using a two-phase design (every candidate gets one exact solve at $1M; only the top 3 by leverage get the full 4-delta curve) to keep total runtime bounded (~15-20 min/cycle).

**Headline numbers, both cycles, exact SLSQP, delta=$1M** (`results/strategic_leverage_{cycle}.json`):

| Cycle | Side | Highest-leverage race | V_uni | PSV | retention | leverage (seats/$M) | reshuffle |
|---|---|---|---|---|---|---|---|
| 2024 | D | WI-01 | +0.036 | +0.036 | 98.7% | +0.036 | $16 (~none) |
| 2024 | R | NV-01 | +0.058 | +0.023 | 39.7% | +0.023 | $3.44M |
| 2022 | D | MD-03 | +0.029 | +0.058 | 197.8% | +0.058 | $0.55M |
| 2022 | R | NY-20 | +0.035 | +0.055 | 155.3% | +0.055 | $1.52M |

Two distinct race types show up clearly in both cycles, matching the direct-value/forcing distinction the research discussion anticipated: races like 2024's NC-06/NC-14 (R-side, V_uni ~0.16 but retention ~10%) have large *apparent* unilateral value that Democratic response erodes almost entirely -- exactly the pattern `persistent_value.py` was built to detect. Races like WI-01 (2024) or FL-27/WI-03 (2022 D-side, retention 96-98%, reshuffle near zero) are close to *unanswered*: the opponent's true best response barely reacts to the perturbation at all.

**A robust, cross-cycle-replicating structural finding, not anticipated going in**: the response-displacement map shows each side draws on a small, STABLE set of races to finance almost any response, largely independent of which race the pressure targets. 2022 D-side pressure on CT-02, NC-06, NY-20, and AZ-02 (four different target races) all draws from the *identical* 8-race Republican set (TX-15, NC-14, CO-07, NY-04, CA-13, NM-01, AZ-06, CA-03); 2022 R-side pressure on FL-07, PA-04, TX-32, and AZ-02 all draws from the identical 8-race Democratic set (CA-27, NE-02, CA-45, CA-40, IA-02, TX-15, AZ-01, +1). 2024 shows the same pattern with a different (cycle-specific) race set on each side. The likely mechanism: `_finance_delta`'s water-filling logic cuts from whichever currently-funded races sit at the *lowest* marginal value, and for small perturbations relative to a ~$47-100M budget, that low-marginal-value set is locally stable regardless of where the pressure originates -- it is a property of the opponent's OWN allocation's shape at the observed point, not of the specific race being pressured. Checked at larger delta (2022 R-side AZ-02: $250K/$500K keep the identical 8 races scaling roughly linearly; at $2M, OH-13 drops out of the pool and AZ-06 enters) -- the pool has finite depth and does start to shift as commitment size grows, consistent with `leverage_seats_per_million` declining with delta in most rows (diminishing returns as bigger moves force digging into progressively less-marginal financing sources).

**Reading this against the research discussion's original "decoy race" hypothesis**: this cuts against decoys being a powerful lever in this data, at least at the $250K-$2M scale tested. A decoy race is supposed to work by forcing the opponent to sacrifice a *valuable* race to respond. What the displacement map actually shows is that both sides have real slack -- a small reserve of low-priority races -- that absorbs small-to-moderate pressure without ever touching the opponent's actual priority targets. The "cost" of responding shows up as PSV/retention (already known via `persistent_value.py`), not as a forced sacrifice of specifically valuable races. Whether that changes at commitment sizes large enough to exhaust the reserve (the $2M row's early sign of that) is the natural next question, not yet run here.

Raw output: `results/strategic_leverage_{cycle}.json` (`leverage_D_primary`/`leverage_R_primary`: every candidate at $1M with full displacement lists; `leverage_D_curve`/`leverage_R_curve`: top-3-by-leverage candidates per side across all 4 deltas).

### Pushing delta larger: does the financing pool actually exhaust? (2026-08-13)

The section above ends by flagging the natural next question: the "shared financing pool" result only covers delta up to $2M against per-race caps of $7-15M -- does pushing further actually force either side to start sacrificing valuable races, the way the original "decoy race" hypothesis predicted? `scripts/compute_strategic_leverage_large_delta.py` extends the same 12 already-identified top candidates (6 per cycle) to delta = $3M/$5M/$8M/$12M, exact SLSQP throughout, deliberately past what any single real race would plausibly absorb, specifically to find where the pool runs out.

**Capping matters at this scale and is now tracked explicitly.** `_finance_delta` clips the amount actually placed at the target race once `delta` approaches `cap_fraction * budget` (2024: D-side cap $15.3M, R-side cap only $7.1M; 2022: D $14.9M, R $13.7M -- 2024's unusually low R cap is a real, previously-reported feature of that cycle, not new). At $8M-$12M requested, several 2024 R-side races (FL-22, NC-06, NV-01) hit their cap and can only deploy ~$7.1M. `src/game/strategic_leverage.py` was extended to report `delta_requested`/`delta_deployed`/`capped` per row and normalize leverage by `delta_deployed`, not the nominal request -- otherwise a capped row's leverage would be silently understated (dividing by a bigger number than what was actually spent).

**Headline pattern: leverage mostly decays smoothly, consistent with the earlier finding, but with one sharp exception verified before being trusted.** Most of the 12 curves flatten out in the expected direction -- e.g. 2022 R-side NY-20 collapses from 0.016 seats/$M at $3M to 0.0003 at $12M (Democrats' response nearly fully neutralizes it at that scale), and several D-side races (2024 WI-01, 2022 FL-27/WI-03) stay essentially unanswered (`reshuffle` in the tens of dollars) all the way to $12M -- the "unanswered opportunity" pattern from the $1M analysis persists at 12x the commitment size, not just a small-delta artifact.

One point broke that pattern sharply enough to need checking before reporting it: 2024 R-side NC-06 jumps from leverage +0.013 ($3M) to +0.061 ($5M), then flattens once capped at $7.1M. Given this project's own prior history of cold-start SLSQP producing spurious results when checked against a point it didn't itself produce (docs/methodology.md's "Verification pitfall" note, and the "obvious fix broke worse" HI-02 incident), this was checked directly rather than taken at face value: BR_D at NC-06's $5M deviation was re-solved from a warm start (`x0=party_d_obs`) alongside the default cold start (`x0=None`, what the actual run used). Both converged to the identical objective value (219.7027 expected D seats, agreeing to 4 decimal places) -- strong evidence this is a real, sharp feature of the payoff surface at that point, not a solver artifact. (At $3M, by contrast, cold-start found a BETTER point than warm-start, 219.9666 vs. 219.6859 -- a reminder that this payoff surface is not globally concave and different starting points CAN land in different local optima; it just happens not to explain the $5M jump.) The likely mechanism: NC-06 is one of the redistricting-flagged districts (`backtest.data.universe`'s own warning list) with a correspondingly less certain baseline, and `payoff.p_win_shared`'s tanh saturation means a race can cross from "worth contesting" to "saturated, not worth D's marginal dollar" at a fairly sharp threshold rather than gradually -- once R's spending pushes NC-06 far enough, D's optimizer abruptly stops defending it and reallocates that capital elsewhere, producing a real discontinuity in the best-response value function rather than a smooth curve. Flagged here as a verified-but-sharp finding, not smoothed over.

**Answering the original question**: at the $250K-$2M scale, neither side was forced to touch its priority races. Pushed to $3M-$12M, the picture is more mixed than a clean "pool exhausts, decoys start working" story -- leverage generally DECLINES with scale (diminishing returns dominate for most races), a few races remain essentially cost-free to press even at $12M, and the one sharp exception found (NC-06) reflects a threshold effect in a specific, lower-confidence district rather than a general pattern of decoys becoming more effective at scale. The practical reading: this data does not support "commit more to force a bigger sacrifice" as a reliable strategy in the $3M-$12M range -- if anything, the better opportunities found here are the ones that stay cheap to press rather than the ones that get more expensive for the opponent to answer.

Raw output: `results/strategic_leverage_large_delta_{cycle}.json`.

## Locked capital and response delay: the two deferred pieces, completed (2026-08-13)

The research-discussion follow-up work above was scoped down twice: first to a static opportunity-cost pass (skipping the transcript's irreversible-capital and response-timing ideas), then further after the surrogate-noise and NC-06 findings. Both deferred pieces are now built.

### Piece 1: locked capital in the two-player solver itself

Every result in this project through the sections above gives the opponent a "godlike" best response: `BR_R`/`BR_D` always re-solve over the FULL budget from scratch, as if every dollar could be costlessly reallocated regardless of what's already been spent. The old single-player trilogy (Papers I-III) already solved this for the single-player case -- `src/backtest/dynamic/ledger.py`'s `B_t = L_t + F_t` capital account, `horizon.py`'s receding-horizon optimizer over `F_t` only -- but it was never connected to this project's two-player `game/` module, which is why `project_spec.md` Section 23 explicitly said "do not begin with Theta" for the new game (deliberate sequencing, not an oversight).

`game/best_response.py::_solve` now accepts an optional `committed_own` (L_t) array: dollars per race already spent, entering as an addition to the race's spend floor exactly as `ledger.py`'s docstring specifies. The search variable becomes the FLEXIBLE portion only; each race's remaining room is `cap_fraction*budget - committed[i]`, and the flexible budget constraint is `budget - sum(committed)`, not the full `budget`. `committed_own=None` (every pre-existing caller, unchanged) is mathematically identical to an all-zero commitment array -- verified directly (`none vs zero committed match: True`, `e_seats diff 0.0`) before trusting anything built on top of it. Four regression tests added to `tests/test_best_response_shared.py`: the none/zero equivalence; locked capital as a hard floor the solver cannot reduce; the flexible-budget constraint being `budget - sum(committed)` rather than the full budget; and a fully-committed/zero-flexible-budget case collapsing to exactly the committed allocation with the correspondingly lower objective value a free solve would beat.

### Piece 2: response delay, using real dated data rather than a synthetic schedule

`src/estimation/commitment_timing.py` computes REAL locked-capital fractions from `backtest.data.fec.load_ie_transactions_dated`, extended with a new `national_committee_only` filter (mirrors `scripts/fetch_data.py::extract_national_committee_ies`'s `spe_id == DCCC_COMMITTEE_ID/NRCC_COMMITTEE_ID` filter, applied to the DATED transaction stream instead of the cycle-aggregated one) -- this is the dated version of `party_natl` / x_D / x_R, this project's actual game-theoretic decision variables, not the broader outside-group IE universe `load_ie_transactions_dated`'s original (undated-filter) behavior returns.

**Calibration check, done before trusting the curve shape**: the dated subset's own total should approximate the full control-floor budget (`build_cycle_state.py`'s `budget_d`/`budget_r`, which also includes undated transactions). Three of four cycle/party combinations match almost exactly (2022 D: $93.25M dated vs. $93.27M full; 2022 R and 2024 R: exact matches) -- but 2024 D captures only $70.84M of $98.48M (72%, a real 28% gap). Because of that unevenness, this module does NOT use the dated subset's raw dollars directly; it computes a FRACTION curve (cumulative dated $ / that party's own dated total, reaching exactly 1.0 at the last transaction by construction) and applies the fraction to the TRUE full budget -- correct regardless of how much of the total the dated subset captures, under the stated assumption that undated transactions are paced similarly to dated ones over time. Flagged as least trustworthy for 2024 D specifically, where more of the curve's shape is invisible to it.

Second simplifying assumption, stated rather than hidden: ONE aggregate fraction curve per (cycle, party), applied uniformly to every race's own `party_obs`, not a separate per-race date curve -- necessary because most funded races have only a handful of dated national-committee transactions each (the same sparse-data problem this project's eta reaction estimation pools across Cook-rating tiers to avoid, here pooled across races instead of districts).

`game/response_delay.py` reuses `strategic_leverage.py`'s `_finance_delta`/isolated-baseline machinery but swaps the frictionless `BR_R(D')` for the locked-capital version: one side moves `delta` at a reference date `t0`, the opponent observes it immediately (this tests RESPONSE delay, not information delay) but can only react with whatever fraction of its budget is still flexible `tau` days later. `t0` = September 1 was chosen by checking the actual commitment curves first, not arbitrarily: by that date only 1-6% of either committee's eventual national-committee spending has happened in every cycle/party combination, rising to a meaningful-but-not-saturated 17-58% by `t0+28`. The baseline for each `tau` is the opponent's best response to the UNCHANGED allocation under the SAME commitment constraint being tested (parameterizing persistent_value.py's isolated-baseline principle by `tau` instead of it being a single fixed quantity).

**Result, run on the same 12 already-identified top candidates (delta=$1M, exact SLSQP, ~60 solves/cycle) both cycles**: the predicted effect -- retention rising as the opponent's flexible budget shrinks -- holds clearly for 5 of 12 races, all races that were PARTIALLY answered under full flexibility (retention < 100% at tau=0): NV-01 2024 (41%→71%), FL-22 2024 (34%→60%), CT-02 2024 (63%→75%), AZ-02 2022 (65%→86%), and more weakly NC-06 2024 (10%→18%, still heavily eroded even at tau=28 -- consistent with NC-06's already-flagged redistricting-related unreliability). Races already near 100% retention (WI-01 2024, FL-27/WI-03 2022, PA-12 2022) stay flat -- a ceiling effect, nothing to gain since the opponent barely responds even with full flexibility.

**The counter-example, reported rather than smoothed over**: the three races with retention ABOVE 100% at tau=0 (MD-03 2022: 198%→195%, NY-20 2022: 154%→138%, CT-05 2024: 179%→113%) all show retention FALLING as tau grows, the opposite of the naive prediction. This has a coherent mechanism, not just noise: the corrected-payoff methodology section above ("PSV retention >100% anomaly") already traced the >100%-retention pattern to the opponent's FULL-PORTFOLIO reoptimization landing somewhere more favorable to the mover than the simple `-Regret` baseline predicts -- a second-order effect that specifically depends on the opponent having a LOT of flexibility to reshuffle its whole budget. Locking part of that capital removes the exact mechanism that produced the >100% bonus, so constraining the opponent's response pulls these three races BACK toward (not above) 100% rather than pushing them further above it. Response delay helps a mover when the opponent was going to partially neutralize the move; it doesn't help, and can mildly hurt, when the "opportunity" was itself an artifact of the opponent's own unconstrained flexibility.

Figure: `figures/static/response_delay_summary.png` (`scripts/plot_response_delay.py`). Raw output: `results/response_delay_{cycle}.json`.

**What this changes about the original research-discussion agenda**: both deferred pieces confirm the transcript's core intuition -- a durable edge is more plausible from timing/friction than from a smarter static allocation -- but with an important correction the static analysis alone couldn't surface: friction doesn't uniformly help the mover. It helps races that were genuinely being contested away, and actively erodes the (rarer, previously-flagged-as-counterintuitive) races whose apparent value came from the opponent's own reshuffling flexibility rather than from being under-defended.

## Strategic window: partial pooling by competitiveness tier, and V_i(t) (2026-08-13)

Follow-on to a second research-discussion round after the locked-capital/response-delay results above. Two things were flagged as the natural next step: (1) the commitment-timing curves are pooled at the coarsest possible level (one curve per party per cycle) when a genuine competitiveness gradient might exist underneath, and (2) the response-delay sweep only ever varied tau at one fixed reference date (Sept 1) -- the more direct descriptive question is how retention evolves as the ELECTION approaches, i.e. varying the reference date t itself.

**Checked before building anything**: does partial pooling by Cook-rating tier have enough data to be worth it, and is there a real timing difference to capture? The full 7-way Cook breakdown is too thin to trust on its own -- most non-competitive ratings have only 1-4 funded districts and a few dozen transactions each in the dated national-committee-only IE data. Collapsing to three tiers (competitive = Toss-Up; lean = Lean D/Lean R; safe_likely = the four Safe/Likely categories) gives workable counts (10-22 districts / 200-450 transactions in competitive and lean; 3-5 districts / 29-104 in safe_likely, thin but usable pooled where it was unusable split four ways). And there IS a real gradient: in 3 of 4 cycle/party combinations, competitive races are funded measurably earlier than lean or safe/likely ones (2022 R: competitive median transaction date Oct 11 vs. lean Oct 18 vs. safe_likely Oct 25, a 2-week spread). The one exception (2024 R, safe_likely earliest) rests on only 3 districts/35 transactions and is treated as noise. `estimation.commitment_timing.py` gained `commitment_fraction_curve_tiered`/`committed_capital_per_race_tiered`/`build_tiered_curves`: same fraction-of-eventual-total logic as the aggregate curve, just computed separately per (cycle, party, tier) and applied to each race via its own Cook rating.

**Scoping the (t, tau, delta) surface down to a tractable first pass**: a full grid over the discussion's proposed 8 reference dates x 5 delay values x an expanded ~30-race candidate set x both directions x both cycles would run to many hours of exact-SLSQP solves (estimated ~9,600 solves at that scale, against an observed ~20-30s/solve). Collapsed to a 1D sweep instead: fix tau=0 (the opponent's response uses whatever it has ACTUALLY committed as of the SAME date the mover commits -- not an additional artificial delay stacked on top) and delta=$1M, and vary only the reference date t across 120/90/60/45/30/21/14/7 days before Election Day (true election dates: 2022-11-08, 2024-11-05), on the same 12 already-identified candidates. `game/strategic_window.py` implements this (`retention_by_date_d`/`_r`), with a `baseline_cache` shared across every candidate race at a given cycle -- the opponent's baseline response to the UNCHANGED allocation at date t doesn't depend on which race is being tested, so caching it cut the solve count from a naive 192/cycle to 64/cycle (~27 min/cycle instead of ~80).

**A structural sanity check confirmed before trusting the results**: as the reference date approaches Election Day, every committee's flexible budget shrinks toward zero (by construction -- almost all its eventual spending has already happened), and once flexible budget hits ~0 the opponent literally cannot respond to ANYTHING, so retention must converge to EXACTLY 100% regardless of the specific race. This shows up cleanly in both cycles: every one of the 12 curves lands at 99.8-100.0% retention at t=7 days, whether it started at 10% or 198%. This is a mechanical floor, not a race-specific finding, and the writeup below treats it accordingly -- the informative part of the surface is the MIDDLE of the season, not the final week.

**Result, both cycles, 12 candidates x 8 dates**: races that were already near 100% retention under full flexibility (WI-01, FL-27, WI-03 -- the "already unanswered" races from earlier sections) stay flat throughout, trivially. Races that started ABOVE 100% (CT-05, MD-03, NY-20 -- the previously-flagged reshuffling-bonus races) decline SMOOTHLY and monotonically toward exactly 100% as the election approaches, replicating the response-delay section's finding with a full trajectory instead of two endpoints: CT-05 179%->157%->90%->...->99.9%; MD-03 198%->192%->...->100%; NY-20 155%->147%->129%->75%(!)->106%->...->100% (NY-20 actually dips BELOW 100% around 3 weeks out before an uneven final approach -- a real non-monotonicity, not smoothed over).

The genuinely new finding is in the races that started BELOW 100%: three of four (CT-02, FL-22, NV-01 in 2024; AZ-02 in 2022) cross the 80%-retention threshold around **30 days before the election**, while the opponent still has $25-60M in flexible budget remaining -- a meaningful, actionable lead time, not just "wait until the opponent runs out of money." The fourth (NC-06) does not cross 80% until the final week (39% at 14 days, then a sharp jump to 111% at 7 days) -- consistent with NC-06's already-documented unreliability (redistricting-flagged district, the same race whose $5M threshold jump in the large-delta section required a separate cold/warm-start verification). `strategic_window.py::strategic_opening_date` computes T_i^80 requiring retention to STAY above 80% for every later date too, not just cross it once, given the NY-20-style non-monotonicity found here.

**Reading this against the discussion's proposed agenda**: this replicates the core "strategic window" idea in both cycles for the races that had real erosion to begin with (3-4 of 6 candidates per direction), with a consistent ~30-day lead time before the terminal squeeze -- a genuinely new, actionable statistic (`T_i^80`) this project didn't have before. It does NOT yet address the discussion's "value of waiting has costs" caveat (own-side capital becoming committed elsewhere, the opponent moving first, uncertainty resolving unfavorably) -- this sweep only asks whether a LATER move survives better, not whether waiting to make it was worth what else could have been done with that capital in the meantime. That remains the natural next layer if this project continues into the full sequential timing game the discussion outlined, not attempted here.

Figure: `figures/static/strategic_window_summary.png` (`scripts/plot_strategic_window.py`). Raw output: `results/strategic_window_{cycle}.json` (includes each side's `T80_D`/`T80_R` dict).

## Value of waiting: durability gained vs. the best immediate alternative (2026-08-13)

The strategic-window section above ends by flagging exactly what it doesn't answer: it shows retention improves with delay, but never compares that against what the SAME capital could have done if deployed immediately elsewhere. "Waiting helps this race" and "waiting was the right call" are different claims -- the second requires an opportunity-cost comparison, not just a before/after on one race. This section builds that comparison.

**Pure post-processing, no new solves**: `results/strategic_window_{cycle}.json` already has PSV_i(t) for every one of the 12 candidates at all 8 reference dates, both sides, both cycles -- everything needed was already computed. `scripts/compute_value_of_waiting.py` defines, for a delta at race i:

```
V_now(i)          = PSV_i at the earliest date (120 days out, ~full flexibility)
V_alt(t_early)    = max PSV_j at 120 days out, over the OTHER 5 same-side candidates j != i
best_immediate(i) = max(V_now(i), V_alt(t_early))
V_wait(i)         = PSV_i at race i's own T_i^80 (NOT the literal end of the cycle --
                    strategic_window.py already flagged the final week as a mechanical
                    convergence floor common to every race, uninformative as a wait target)
net_waiting_value(i) = V_wait(i) - best_immediate(i)
```

Positive means holding delta in reserve specifically for race i, until it becomes durable, beats deploying it to the best currently-known alternative right away. Negative means the reverse.

**Two limitations stated up front, not modeled away**: (1) `V_alt` only searches the other 5 pre-screened candidates on the same side, not the full 433-race universe -- since these were screened for already being attractive (top swing / top-|Z|), the true best immediate alternative is probably at least this good and could be better, making `net_waiting_value` a plausible UPPER bound on the true value of waiting, not a tight estimate. (2) This is retrospective, on realized data -- it compares two CERTAIN outcomes, not the uncertain ones a real-time decision-maker actually faces (the discussion's own list: information resolving unfavorably, the opponent locking up the SAME race first, the race's own fundamentals shifting). It answers "would waiting have paid off, given what happened," not "should you wait, given what you could know in advance."

**A bookkeeping distinction that matters for reading the result**: 7 of the 12 candidates already had `T_i^80` = day one (the earliest date tested) -- these races were already durable under full opponent flexibility, so no delay was actually being tested; `net_waiting_value` for them degenerates to "was this race the best immediate choice, or was there a better one already available" rather than "did waiting help." Only 5 candidates (CT-02, FL-22, NC-06, NV-01 in 2024; AZ-02 in 2022) involved genuine waiting -- T_i^80 fell strictly between the first and last reference dates.

**Result**: net waiting value is large and clearly positive ONLY for 2024's R-side pool -- NC-06 (+0.155 expected seats), FL-22 (+0.043), NV-01 (+0.038) -- because every candidate in that specific pool was mediocre immediately (best immediate alternative there tops out at 0.023), so nothing was foregone by waiting for any one of them to mature. One D-side 2024 race (CT-02) shows a modest genuine gain (+0.008). Everywhere else, waiting is roughly neutral-to-negative: WI-03 2022 (-0.025), AZ-02 2022 (-0.015), PA-12 2022 (-0.012), FL-27 2022 (-0.006), CT-05 2024 (-0.004) all show the best immediate alternative beating whatever that race's own delayed value reached.

**AZ-02 is the cleanest illustration of the discussion's caveat**, because it's the one genuine-wait case that comes out negative: its own retention rose substantially with delay (63%->86%, strategic_window.py's earlier finding), but NY-20 was simply a better immediate choice throughout the whole season (0.055 vs. AZ-02's matured 0.040) -- confirming that "this race gets more durable if you wait" and "you should wait for this race" are genuinely different claims, and conflating them is exactly the mistake the discussion warned against. Waiting for a specific opportunity to ripen has a real cost when a better one is already sitting there unexploited.

**Reading this against the discussion's original framing**: `Theta = information option value + strategic flexibility option value`. This section (with strategic_window.py) now has a real, data-grounded estimate of the SECOND term's net effect after accounting for its most obvious cost (foregone immediate deployment) -- and finds it is conditionally, not universally, positive: valuable specifically when the currently-visible options are all weak (2024 R), close to a wash when a strong option already exists (most of 2022, most of D-side 2024). The FIRST term (information option value -- race-state uncertainty resolving over time) remains unmodeled; this project's static, retrospective payoff has no mechanism for "the race turns out to be more/less winnable than currently believed," which is a structurally different kind of value than anything computed here.

Figure: `figures/static/value_of_waiting_summary.png` (`scripts/plot_value_of_waiting.py`). Raw output: `results/value_of_waiting.json`.

## Information option value: completing Theta (2026-08-13)

The value-of-waiting section above ends by naming exactly what's missing: `Theta = information option value + strategic flexibility option value`, and only the second term had been built. This section builds the first, completing the decomposition -- without touching Paper III's existing single-player `solve_bellman_lsm.py` machinery, which is built for a different question (a live, forward-looking, one-shot "close the reserve now vs. later" decision) and carries its own considerable validated-but-fragile history (its own docstring records an allocator-degeneracy bug that used to silently flip Theta's sign).

**What's reused vs. built fresh**: the historical generic-ballot volatility calibration (`scripts/estimate_gb_volatility.py`'s pooled 2018-2024 538 series) is genuinely cycle-general, not 2026-specific -- reimplemented in `estimation/gb_uncertainty.py` (not imported cross-layer from `scripts/`, matching this project's src/-doesn't-depend-on-scripts/ convention) rather than refit. Checked directly rather than assumed: per-sqrt-day volatility is NOT perfectly flat within a cycle (2022 actually rises with horizon, 0.21/day at 30 days out to 0.26 at 180), so `residual_gb_std(cycle, days_before)` looks up the DIRECTLY REALIZED std at the exact requested horizon from that cycle's own series, rather than fitting one constant and extrapolating.

**The mechanism, and why it needed no new best-response solves**: `coef.alpha3 * generic_ballot` enters the margin model additively and identically across every race -- a genuinely national-level shifter. Simulating a shared shock `epsilon ~ N(0, residual_gb_std(cycle, t)^2)` to every race's `generic_ballot` and recomputing `V_uni` (closed-form -- no SLSQP) for each of the 6 same-side candidates lets a committee's date-t belief about which race looks best diverge from the truth. `game/information_value.py` runs 5,000 such draws per side per cycle in well under a minute total (contrast the response-delay/strategic-window sections' tens of minutes of exact SLSQP), since nothing in the loop touches an optimizer.

**A sanity check caught a real conceptual issue on the first run, not a bug**: the first version defined "best_true_immediate" as the globally highest-PSV candidate and checked that zero noise recovers it. It didn't -- on 2024 D-side, CT-02 has the highest V_uni (0.053) but WI-01 has the highest PSV (0.036), because CT-02's opponent response erodes far more of its raw appeal (retention ~62%) than WI-01's does (~99%). This is a real, separate, and itself-interesting finding -- **a raw-persuadability targeting heuristic and a full game-theoretic PSV ranking can disagree about which race is best** -- but conflating it with the information question would contaminate the information estimate with a fixed, noise-independent offset having nothing to do with uncertainty. Fixed by anchoring `best_true_immediate` to the ZERO-NOISE V_uni pick (the same decision rule the noisy Monte Carlo uses) rather than the globally-best-PSV race, making the zero-noise case an exact fixed point and isolating the noise's own marginal cost. Arguably the more realistic framing regardless: real committees making real-time decisions plausibly use something closer to a raw-persuadability signal than a full best-response simulation against every candidate race.

**Result: the V_uni/PSV disagreement itself replicates in all 4 of 4 (cycle, side) combinations tested** -- the cheap heuristic never once agrees with the game-theoretic ranking in this candidate pool. That is a real, generalizable-flavored finding on its own, reported separately from the information-value number itself (`v_uni_rule_disagrees_with_psv_best` in the output).

**The information-value result itself is small in 3 of 4 cases, and understandably so**: 2024 D-side, 2024 R-side, and 2022 D-side all show `info_option_value` within Monte Carlo noise of exactly zero (pick frequency 4,998-5,000 of 5,000 draws for the SAME race) -- the V_uni gap between the top candidate and the runner-up in these pools is simply too large for a historically-realistic generic-ballot swing to overturn. **2022 R-side is the one real exception**: AZ-02, NY-20, and PA-12 have closely-matched V_uni scores, so the SAME noise magnitude genuinely splits the pick (832 / 680 / 3,488 of 5,000 draws), producing a small but real `info_option_value = +0.0013` -- positive as required (more information cannot make an optimal decision worse in expectation), small in absolute terms, but non-trivial relative to how tightly bunched the pick frequencies are.

**Reading the two Theta components side by side** (figure below): strategic flexibility (net_waiting_value, the section above) reaches +0.155 expected seats on its largest case; information option value tops out at +0.0013 -- two orders of magnitude smaller, in every case tested. For this specific dataset (12 pre-screened candidates, both cycles, 120-days-out reference point), **the strategic-flexibility channel is the one actually carrying Theta**; the information channel is real (correct sign, occasionally non-trivial) but small enough here that treating Theta as approximately equal to the strategic-flexibility term alone would not be a bad first-order approximation for these races. This should not be over-generalized past the tested setup -- a different candidate pool with more closely-bunched V_uni scores, or a later/earlier reference date, could show a larger information term; this section establishes the METHOD and one honest data point, not a universal ratio.

Figure: `figures/static/theta_decomposition_summary.png` (`scripts/plot_theta_decomposition.py`). Raw output: `results/information_value.json`.

## Unified sequential decision value: completing Theta as one Bellman value (2026-08-14)

The section above closed with a claim -- "the strategic-flexibility channel is the one actually carrying Theta" -- that a research-direction review flagged as premature. The reason: `value_of_waiting.py`'s strategic-flexibility number and `information_value.py`'s information number are not two components of one objective. They rank candidate races by two DIFFERENT decision rules. `value_of_waiting.py`'s `best_immediate`/`V_wait` compare races by PSV (the game-theoretic, opponent-best-response-adjusted value). `information_value.py`'s noisy pick ranks races by V_uni (a cheap, closed-form proxy), specifically BECAUSE its own zero-noise sanity check found V_uni and PSV disagree about which race is best (2024 D-side: V_uni picks CT-02, PSV picks WI-01). Describing `Theta = Theta_strategic + Theta_info` when the two terms are the output of two different maximization problems is not yet a decomposition of a common Bellman value -- it is two separate diagnostics that happen to get added together.

**The fix**: one decision rule, used at every reference date and in every counterfactual. `game/unified_theta.py::deploy_value` estimates each candidate race's PSV under a noisy generic-ballot signal as `V_uni_noisy(epsilon) * retention(t)` -- V_uni's existing cheap noise machinery (`information_value.py::_noisy_best_pick`, generalized to return every candidate's value, not just the argmax) combined with retention_i(t), the TRUE (zero-noise) PSV/V_uni ratio strategic_window.py already computed at each date. The committee picks the race with the highest ESTIMATED PSV under noise; the realized payoff is that race's TRUE PSV. This is a real approximation (retention itself is estimated at zero noise and held fixed while V_uni is perturbed, rather than re-solving BR_R under the noise-perturbed race arrays -- which would cost a full SLSQP solve per Monte Carlo draw, the same compute-cost tradeoff `strategic_window.py`'s own docstring already flagged as impractical at this scale: a full grid would run to thousands of solves). It is now the SAME approximation applied identically everywhere, which is the property the two-module version lacked.

State `X_t = (opponent's true committed capital at t, generic-ballot noise at t)`. `game/unified_theta.py::solve_bellman` backward-induces `V_t = max(V_deploy_t, V_{t+1})` across strategic_window.py's existing 8-date grid (120/90/60/45/30/21/14/7 days before Election Day), with `V` at the final date forced to `V_deploy` (the final week is already established as a mechanical 100%-retention floor, not a substantive continuation target). `scripts/compute_theta_unified.py` runs this three ways per (cycle, side), reusing every quantity strategic_window.py already computed -- no new best-response solves:

- **full**: both channels active (the actual quantity a real decision-maker faces).
- **flex_only**: information frozen at zero noise (the committee always picks the TRUE V_uni-best race) -- isolates the value of waiting for the opponent's capital to lock up, with no information channel.
- **info_only**: opponent commitments frozen at the t=120-days-out (~full-flexibility) level for every later date, so PSV itself does not improve with delay -- isolates the value of waiting for the committee's OWN uncertainty to resolve, with no strategic-flexibility channel.

`Theta_full` is reported alongside `Theta_flex_only + Theta_info_only` WITHOUT asserting they are equal; `interaction = Theta_full - (Theta_flex_only + Theta_info_only)` is computed and reported explicitly.

**Headline result, same 12 candidates (top-3-by-leverage per side, both cycles) already used throughout this project's dynamic-extension work** (`results/theta_unified.json`):

| Cycle | Side | Districts | Theta_full | Theta_flex_only | Theta_info_only | Sum of parts | Interaction |
|---|---|---|---|---|---|---|---|
| 2024 | D | CT-02, CT-05, WI-01 | +0.0620 | +0.0612 | +0.0006 | +0.0619 | +0.0001 |
| 2024 | R | FL-22, NC-06, NV-01 | +0.1553 | +0.1553 | +0.0000 | +0.1553 | −0.0000 |
| 2022 | D | FL-27, MD-03, WI-03 | +0.0002 | +0.0000 | +0.0016 | +0.0016 | −0.0014 |
| 2022 | R | AZ-02, NY-20, PA-12 | +0.0002 | +0.0000 | +0.0002 | +0.0002 | +0.0000 |

**The earlier claim survives unification, on firmer ground than before.** In 3 of 4 (cycle, side) pools, `Theta_flex_only` accounts for effectively all of `Theta_full` and `Theta_info_only` is at or near zero -- the same qualitative pattern the two-module version found, now demonstrated under one consistent objective rather than an artifact of comparing a PSV-ranked calculation against a V_uni-ranked one. The one exception, 2022 D-side, is where information contributes MORE than strategic flexibility (+0.0016 vs +0.0000) and the interaction term is large in RELATIVE terms (sum of parts overstates the unified value by roughly 8x) -- but every number in that row is under 0.002 expected seats, i.e. this is a real qualitative flip with no practical significance: there is essentially nothing to gain from waiting in that specific 3-race pool either way.

**A second, more consequential correction surfaced along the way, not anticipated going in.** 2024 D-side's `Theta_full` (+0.0620) is roughly 8x the earlier `value_of_waiting.py` estimate for CT-02 specifically (+0.008, docs/methodology.md's "Value of waiting" section above). Tracing it down: `value_of_waiting.py` set `V_wait(i) = PSV_i` at race i's `T_i^80` -- the FIRST reference date at which retention crosses 80% and stays there. But CT-02's full PSV trajectory (`results/strategic_window_2024.json`) is not monotonic past that crossing:

| Date | Days out | PSV | Retention |
|---|---|---|---|
| 2024-10-06 | 30 (= T_i^80) | +0.0440 | 82.9% |
| 2024-10-15 | 21 | **+0.0970** | 182.8% |
| 2024-10-22 | 14 | +0.0885 | 166.9% |
| 2024-10-29 | 7 | +0.0530 | 100.0% (mechanical floor) |

CT-02's PSV actually PEAKS at 21 days out (+0.0970, retention 182.8%) -- a genuine second-order equilibrium effect this project's methodology already documented and explained as real, not a bug ("PSV retention >100% anomaly" section above: R's full-portfolio reoptimization at the perturbed allocation can land more favorably for D in aggregate than the simple baseline predicts) -- then declines back toward the mechanical 100% floor by the final week. `value_of_waiting.py`'s "value at first crossing" convention captured only the +0.0440 point on this curve, understating CT-02's true best continuation value by more than half. The unified Bellman recursion does not have this blind spot BY CONSTRUCTION: `V_t = max(deploy_t, V_{t+1})` takes the max over the ENTIRE remaining horizon at every step, not just the first date clearing a threshold, so it automatically finds and propagates back the +0.0970 peak. This is a real, generalizable lesson beyond this specific race: **defining "value of waiting" as "value at the first date retention clears some bar" understates the true value of waiting whenever retention is non-monotonic past that bar** (already known to happen -- NY-20 2022's documented dip below 100% partway through its approach is the same phenomenon in the opposite direction). The Bellman formulation fixes this for free; it was not a motivation for building it, but is a direct consequence of doing so correctly.

**2024 R-side's Theta_full (+0.1553) closely reproduces `value_of_waiting.py`'s NC-06 estimate (+0.155) despite being computed by an entirely different method** (portfolio-level backward induction over a noisy decision rule, vs. single-race first-vs-best-immediate comparison) -- independent-method agreement of this kind is the right kind of confirmation that NC-06's large, late-arriving PSV spike (retention only 10-39% until the very last week, then +0.1782 at 7 days out, `results/strategic_window_2024.json`) is a robust feature of the data, not an artifact of either specific calculation. **Caveat added after the K-expansion work below surfaced it**: this number is realized AT the final (7-days-out) reference date -- the same date `strategic_window.py`'s own methodology already flagged as a mechanical 100%-retention floor common to every race, not a race-specific finding. `scripts/theta_final_week_sensitivity.py` (below) reruns the SAME recursion excluding that date: 2024 R's Theta_full drops from +0.1553 to +0.0430 once the mechanical floor is excluded -- still positive and still the largest "genuine mid-season timing" value found, but roughly a quarter of the headline number. Both figures are now reported together rather than only the larger one.

**What this does and doesn't change**: the practical strategic-flexibility-dominates-information conclusion from the two-module version of Theta is now demonstrated, not assumed -- worth stating with more confidence than before, not less. What changes is the CT-02 finding specifically (larger and more interesting than previously reported) and the methodological standard going forward: any future addition to Theta's decomposition should enter this same `game/unified_theta.py::solve_bellman` recursion as a fourth counterfactual regime, not a separately-defined diagnostic added on top by hand.

Figure: `figures/static/theta_unified_summary.png` (`scripts/plot_theta_unified.py`). Raw output: `results/theta_unified.json`. Regression tests for the decision rule and recursion: `tests/test_unified_theta.py`.

## Widening the action space to K~8, and a mechanical-floor artifact it exposed (2026-08-14)

The section above's Bellman recursion ran on a K=3 candidate pool per side (`strategic_leverage.py`'s "top-3-by-leverage" curve subset) -- a restricted action space in the spirit of the next-phase review's suggestion to bound the sequential game to `{hold} u {deploy to one of K races}` rather than the full 433-race universe, but narrower than the K=10-20 the review proposed. This section widens it.

**K chosen from already-validated data, not an arbitrary new cutoff.** `strategic_leverage.py`'s own candidate screen (docs/methodology.md's "Strategic leverage" section above) already solves a broader "primary" pool once at $1M -- the top-4-swing + top-4-|Z| funded races per side (7-8 distinct districts after overlap) -- before narrowing to the top-3 that get the full delta-curve treatment. `scripts/compute_strategic_window.py` gained a `--pool {curve,primary}` flag to sweep that FULL primary pool across the same 8-date grid instead of just the top-3 (`strategic_window_expanded_{cycle}.json`, kept separate from the original `strategic_window_{cycle}.json` so neither run clobbers the other); `compute_theta_unified.py` gained a matching `--pool` flag (`theta_unified_expanded.json`). Real exact-SLSQP cost, no shortcuts: ~136-144 new solves per cycle (`(K+1) dates x 8`, baseline-cached same as strategic_window.py), ~35-40 minutes per cycle rather than the ~2 hours a naive per-solve estimate suggested. **Weekly time steps (the review's other proposed expansion, 17 dates instead of 8) were deliberately NOT added in the same pass** -- combining both expansions would run to an estimated 600+ solves per cycle, several more hours of compute for a single step. Widening K first and keeping the already-validated 8-date grid follows this project's own repeated practice (project_spec.md Section 26, and nearly every dynamic-extension section above): prove a coarser version works before refining a second dimension on top of it.

**Headline: three of four pools are UNCHANGED by widening K from 3 to ~8.** 2024 D, 2024 R, and 2022 D all produce IDENTICAL `Theta_full` at K~8 as at K=3, to four decimal places -- the additional 4-5 candidates per pool are never the argmax at any of the 8 reference dates under either the zero-noise or noisy decision rule, so the original top-3-by-leverage screen already contained the portfolio-optimal choice for those three pools. That is itself informative: `strategic_leverage.py`'s screen (current-leverage-based) is a reasonable proxy for "which races could matter for a timing decision" in most of this candidate universe, not just a computational convenience.

**The one exception, 2022 R-side, is where the real finding is -- and it needed the sensitivity check above to interpret correctly.** Widening the pool to include FL-07 and PA-04 (both absent from the original top-3, both with V_uni ~0.096-0.098 -- roughly 2.5-3x any candidate in the original K=3 pool) raised `Theta_full` from +0.0002 to +0.0622. FL-07's own trajectory (`results/strategic_window_expanded_2022.json`) explains why it was screened out originally: retention stays between 16-49% for the ENTIRE season (heavily, persistently answered by Democrats) and only crosses 1.0 in the final week (PSV +0.1168, retention 121.1%, at 7 days out) -- `strategic_leverage.py`'s current-point screen correctly saw a heavily-eroded opportunity and ranked it low; only a FULL-SEASON trajectory reveals the late spike. Applying `theta_final_week_sensitivity.py`'s excl-final-week check (below) to this specific number is what separates a genuine finding from an artifact: **2022 R's Theta_full collapses from +0.0622 to +0.0037 once the mechanical final-week date is excluded** -- almost the entire apparent gain was FL-07 cashing in at the same universal 100%-retention floor every race hits at 7 days out, not a genuine mid-season strategic window the wider pool discovered. Widening K did surface a real, previously-invisible candidate (FL-07's raw persuadability was real and large) -- but it did not surface a new genuine TIMING opportunity, once measured correctly.

### Final-week sensitivity: separating genuine mid-season timing value from the mechanical floor

`strategic_window.py`'s own methodology (above) already established that every race's retention converges to ~100% at 7 days out BY CONSTRUCTION -- opponent flexible budget is nearly exhausted for everyone at that point, independent of whether that specific race was ever genuinely contested. `game/unified_theta.py::solve_bellman` had no equivalent safeguard: `V_t = max(deploy_t, ..., deploy_last)` includes that mechanical date in the max like any other, so a pool member with the single largest raw V_uni will always look "worth waiting for" at t=7 regardless of its actual mid-season trajectory -- and the risk of that happening mechanically grows with K, simply because a wider pool is more likely to contain SOME large-V_uni race.

`scripts/theta_final_week_sensitivity.py` (pure post-processing, reruns `solve_bellman` on `dates[:-1]` using the SAME already-computed `deploy_value_full/flex_only/info_only` dicts -- no new solves, no new Monte Carlo) makes this visible directly instead of leaving it implicit:

| Pool | Cycle/side | K | Theta_full | Realized at | Theta_full excl. final week | Realized at | Reading |
|---|---|---|---|---|---|---|---|
| curve | 2024 D | 3 | +0.0620 | 21d out | +0.0620 | 21d out | genuine mid-season timing (unchanged) |
| curve | 2024 R | 3 | +0.1553 | 7d out | +0.0430 | 30d out | **mostly mechanical floor** |
| curve | 2022 D | 3 | +0.0002 | 90d out | +0.0002 | 90d out | genuine (negligible magnitude either way) |
| curve | 2022 R | 3 | +0.0002 | 90d out | +0.0002 | 90d out | genuine (negligible magnitude either way) |
| primary | 2024 D | 7 | +0.0620 | 21d out | +0.0620 | 21d out | genuine mid-season timing (unchanged) |
| primary | 2024 R | 8 | +0.1553 | 7d out | +0.0430 | 30d out | **mostly mechanical floor** |
| primary | 2022 D | 8 | +0.0002 | 90d out | +0.0002 | 90d out | genuine (negligible magnitude either way) |
| primary | 2022 R | 8 | +0.0622 | 7d out | +0.0037 | 25d out | **almost entirely mechanical floor** (this is the FL-07 discovery above) |

**Reading this against the review's original framing**: the practical conclusion does not reverse, but it sharpens in an important way. Of the two pools with a materially positive Theta_full, only 2024 D (CT-02's genuine mid-season equilibrium-reshuffling peak, already traced in the section above) survives excluding the mechanical floor at close to its full headline size. 2024 R's NC-06 retains a real, positive, but much smaller (+0.0430, not +0.1553) genuine-timing value. 2022 R's apparent K-expansion discovery (FL-07) is almost entirely a final-week artifact once isolated (+0.0037). **Going forward, `theta_full_excl_final_week` (not `theta_full`) should be treated as this project's headline "value of waiting" statistic** -- `theta_full` remains useful for showing the TOTAL gap including the trivial end-of-cycle convergence effect, but reporting it alone risks presenting "you were the last party with money left" as if it were a discovered strategic-timing edge.

Figure: `figures/static/theta_final_week_sensitivity.png` (`scripts/plot_theta_final_week_sensitivity.py`). Raw output: `results/strategic_window_expanded_{cycle}.json`, `results/theta_unified_expanded.json`, `results/theta_final_week_sensitivity.json`.

## The K=15-20 stress test: a redistricting-baseline confound found, and the optimal-stopping principle sharpened (2026-08-14)

The two sections above established an operational optimal-stopping principle: hold flexible capital when the expected improvement in the future strategic opportunity set exceeds today's best deployment value; deploy now when a strong, durable opportunity already exists. That principle was validated at K=3 and K~8 -- both small, hand-picked pools. This section stress-tests it the way it should be stress-tested before being treated as mature: widen the action space with a smarter, principled screen (not just a bigger arbitrary cutoff), and check whether the "genuine mid-season wait" stories found so far survive.

**The pre-screen: `scripts/rank_candidate_races.py`.** A K=10-20 action space needs a candidate-selection rule broader than "top current leverage" -- `strategic_leverage.py`'s own screen already showed its blind spot (FL-07, 2022 R, had poor CURRENT leverage but a large late-arriving opportunity). Built as `union(top |Z|, top V_uni, equilibrium-swing races, revealed-contested races)`:

- **top |Z|** and **top V_uni** are genuinely cheap (closed-form, no best-response solve) -- V_uni in particular is computed FRESH across the full 433-race universe here for the first time, rather than reused from a pool that only ever evaluated its own small candidate set.
- **equilibrium swing** reuses the double-oracle mixed-equilibrium support decomposition (`equilibrium_support_composition_{cycle}.json`'s top-CV races).
- **revealed-contested** is an explicitly-labeled APPROXIMATION to "largest strategic-window slope" -- the true slope needs the very per-race date-sweep this pre-screen exists to avoid running on all 433 races, a circularity flagged rather than solved around. Approximated instead as: races with large opponent OBSERVED spend (revealed priority) among races with material V_uni, on the logic that a race the opponent already spends heavily on is exactly the kind a full best response would defend hardest early, producing the large-initial-erosion / late-recovery pattern this project keeps finding.

Run on both cycles (`scripts/rank_candidate_races.py --cycle {2024,2022}`, seconds each, no solves): union sizes landed at K=20/16 (2024 D/R) and K=20/17 (2022 D/R) -- squarely in the K=15-20 range targeted, without manual tuning. FL-07 (2022 R) reappeared in the union via top-V_uni independently of the equilibrium-swing criterion that first surfaced it -- cross-validation from a second, differently-constructed screen that it is a real, reproducible feature of the data, not a one-off artifact of one method.

**Real cost, run for real**: `compute_strategic_window.py --pool union` swept the full union pool across the same 8-date grid, ~616 exact-SLSQP solves total, ~85 minutes wall-clock running both cycles in parallel. `compute_theta_unified.py --pool union` then ran the same closed-form Bellman recursion as the K=3/K~8 sections, ~35-45 minutes (no solves, Monte Carlo only, cost scales with pool size).

**Headline, before any further correction**: 2024 D `Theta_full` ROSE to +0.0796 (from +0.0620 at K=3/8); 2024 R FELL to +0.1033 (from +0.1553); 2022 D stayed ~0; 2022 R rose to +0.0186 (from +0.0002 at K=3, matching K~8's +0.0622... a different number again). Tracing what drove the new 2024 numbers immediately surfaced a second confound, on top of the mechanical-final-week one from the section above:

**A redistricting-baseline confound.** The top V_uni entrants pulling the biggest new candidates into both 2024 pools were NC-06, NC-13, and NC-14 (D-side) and NC-01/06/13/14 (R-side) -- and all four are among `project_spec`'s 13 `redistricting_flagged` districts (`RaceRecord.redistricting_flagged`, `backtest.data.universe`), already documented elsewhere in this project as having a less certain baseline (the "$5M threshold jump" verification in the large-delta strategic-leverage section above traced a real, sharp discontinuity in NC-06's response surface to exactly this). A "top V_uni" screen has no way to distinguish a genuinely large, reliable opportunity from a large NUMBER produced by an unstable baseline -- it will pick up both. `compute_theta_unified.py` gained an `--exclude-redistricting` flag (filters on `RaceRecord.redistricting_flagged` directly, not a hardcoded district list, so it stays in sync with `config.yaml`) to check how much of the headline number this confound was responsible for.

**Doubly-corrected results (excl. redistricting-flagged AND excl. mechanical final week, `theta_final_week_sensitivity.py` extended to cover this pool too):**

| Cycle/side | K (clean) | Theta_full | Realized at | Driving race |
|---|---|---|---|---|
| 2024 D | 17 | **+0.0032** | 30d out | FL-27 (already durable day one -- nothing to wait for) |
| 2024 R | 12 | **+0.0490** | 30d out | AZ-09 (genuine: 60% -> >100% retention by 30d) |
| 2022 D | 19 | **+0.0000** | 120d out | FL-02 (already durable day one) |
| 2022 R | 17 | **+0.0000** | (was 7d/mechanical before correction) | -- |

**2024 D's genuine mid-season value collapsed from +0.0796 to +0.0032 -- and CT-02, the section above's headline example, turns out not to be the reason why.** The K=3/K~8 pools never included FL-27 as a D-side candidate at all (it wasn't in `strategic_leverage.py`'s narrower top-current-leverage screen); once the union pool adds it, FL-27's OWN trajectory (`results/strategic_window_union_2024.json`) shows it already at 112% retention on day one (V_uni=0.124, more than double CT-02's) and RISING slightly through the season before settling back to 100% -- a race that needed no waiting at all because it was already the best available option from day one. Once FL-27 is in the choice set, waiting specifically for CT-02 (net -0.043 against FL-27's `best_immediate`) is strictly dominated. This is the OTHER half of the optimal-stopping principle demonstrated concretely, not just stated: "deploy now when a strong, durable opportunity already exists" is exactly what FL-27 represents, and its mere presence in the pool is what makes waiting for anything else in that pool not worth it.

**2024 R's surviving case is AZ-09, not NC-06.** NC-06's `Theta_full` contribution is now understood as two confounds stacked on each other: the section above already showed +0.1553 was mostly the mechanical final-week floor (+0.0430 genuine); this section adds that NC-06 itself is also a redistricting-flagged district. AZ-09 replaces it as the pool's actual best mid-season-wait case -- smoothly rising from 60.5% retention at 120 days out to >100% by 30 days out (`results/strategic_window_union_2024.json`), not redistricting-flagged, and beating its own `best_immediate` alternative (NM-01, itself individually a smaller but still-positive genuine-wait case at +0.0220) by a clear margin (net +0.0490).

**2022 R's FL-07 "discovery" from the K~8 section is now fully closed out, not just downgraded.** Already shown to be ~94% mechanical-floor-driven there; at K=17 with the full union pool, FL-07 is ALSO dominated by TN-09 (already durable from day one, V_now=0.109) -- net -0.061 against the best immediate alternative. FL-07 was never a genuine strategic-timing opportunity at any pool size or correction level tested. Its repeated reappearance across THREE independent screens (equilibrium-swing at K~8, top-V_uni at the union screen) is a real, reproducible fact about the RAW size of its opportunity (V_uni ~0.10, genuinely large) -- just not, once properly measured, a genuine WAITING opportunity, because a better option (TN-09) was sitting there the entire season.

**Reading this against the optimal-stopping principle stated above**: it survives the stress test, but in a narrower and more precise form than the pre-stress-test evidence suggested. Of five previously-reported "genuine mid-season wait" races (CT-02, FL-22, NV-01, NC-06, FL-07), **four are dominated or artifactual once the search is widened and the redistricting confound is controlled for; only AZ-09 (plus, more weakly, NM-01) survives.** That is not a failure of the principle -- it is the principle doing its job: "deploy when a strong, durable opportunity already exists" correctly predicts that FL-27/FL-02/TN-09 (all found only once the pool widened) should dominate their respective pools, and they do. The practical lesson for anyone using this framework operationally: a candidate list assembled from ANY single screening heuristic (current leverage, raw persuadability, equilibrium support) is not sufficient to identify genuine waiting opportunities -- it must be checked against (1) a broader, independently-constructed candidate pool, (2) the mechanical final-week floor, and (3) known data-reliability flags, in that order, before a "wait for this race" recommendation should be trusted.

**What was deliberately not done in this pass, per the review's own sequencing**: weekly time steps (17 dates instead of 8). The K-widening alone was the first stress test; a genuine mid-season timing result should survive BOTH a wider action space and finer time resolution before being called mature. AZ-09's own trajectory is smooth and monotonic (60% -> 100%+ without the sharp jumps NC-06 and CT-02 showed at the 8-date resolution), which is a good sign for how it would likely behave under a weekly grid, but that is a prediction, not yet a checked result.

Figure: `figures/static/conditional_waiting_value_final.png` (`scripts/plot_conditional_waiting_value_final.py`, consuming `scripts/analyze_conditional_waiting_value_union.py`'s output). Raw output: `results/candidate_union_{cycle}.json`, `results/strategic_window_union_{cycle}.json`, `results/theta_unified_union.json`, `results/theta_unified_union_excl_redistricting.json`, `results/conditional_waiting_value_union_clean.json`.

## Second stress-test check: weekly time resolution, and the verdict (2026-08-14)

The section above closed with an open prediction: AZ-09's trajectory was smooth and monotonic at 8-date resolution, "a good sign for how it would likely behave under a weekly grid, but that is a prediction, not yet a checked result." This section runs that check -- the second half of the stress test the review specified, done second and separately from the K-widening rather than combined with it, per the review's own sequencing ("I would not expand both dimensions simultaneously").

**Method**: `compute_strategic_window.py` gained `--exclude-redistricting` (filters `RaceRecord.redistricting_flagged` districts OUT of the candidate pool BEFORE sweeping, so no solves are wasted on candidates already known to be unreliable -- reordered from the section above's after-the-fact filter) and `--weekly` (an 18-point grid, 120 down to 7 days out in 7-day steps, keeping the same 7-day mechanical-floor endpoint so `theta_final_week_sensitivity.py`'s logic still applies unchanged). Run on the SAME stable candidate set the K-widening section validated (K=17/12 for 2024 D/R, K=19/17 for 2022 D/R, redistricting-flagged districts already excluded) -- ~1,242 exact-SLSQP solves total, ~2.9-3.9 hours wall-clock per cycle running both cycles in parallel.

**Headline, weekly grid, clean pool** (`results/theta_unified_union_weekly_clean.json`):

| Cycle/side | Theta_full (weekly) | Theta_full (8-date, for comparison) | Realized at | Verdict |
|---|---|---|---|---|
| 2024 D | +0.0032 | +0.0032 | 29d out | Unchanged -- FL-27 dominant, negligible either way |
| 2024 R | **+0.0492** | +0.0490 | 29d out | **Unchanged to 3 decimal places** |
| 2022 D | +0.0003 | +0.0000 | 64d out | Unchanged -- negligible either way |
| 2022 R | +0.0319 | +0.0000 | 8d out | **New at weekly resolution -- investigated below, NOT counted as genuine** |

**AZ-09 (2024 R) is confirmed the mature, validated result.** Its weekly trajectory (`results/strategic_window_union_2024_excl_redistricting_weekly.json`) is completely smooth and monotonic -- 60.5% retention flat through early August, climbing steadily through September (62% -> 87%), crossing 100% cleanly at 29 days out, flat afterward. No jumps, no discontinuities, matching its 8-date profile almost exactly (Theta_full +0.0490 -> +0.0492, a difference smaller than Monte-Carlo noise). This is the strongest evidence this project has produced for a genuine, real, actionable mid-season timing signal: it survives a wider candidate pool, a redistricting-reliability filter, AND a finer time grid, all independently.

**2022 R's new weekly-only number needed the same scrutiny CT-02 and NC-06 got, and fails it.** `theta_final_week_sensitivity.py`'s existing "exclude only the literal final date" rule does not flag it (realized at 8 days out, one point before the excluded 7-day mechanical floor, so it technically passes). But the underlying trajectory (`results/strategic_window_union_2022_excl_redistricting_weekly.json`, district FL-02) tells a different story: retention sits FLAT at 13.6% from 120 days out all the way through 24 days out -- ten consecutive weekly observations showing essentially zero recovery -- then jumps sharply to 160.1% in the space of one week (24 days out to 8 days out) before settling to 145% at the literal final date. That is not a gradual mid-season conversion; it is the SAME sharp, late, threshold-like discontinuity pattern this project already found and explained for NC-06 (the "$5M threshold jump" verification) and, at coarser resolution, implicitly for the mechanical floor itself -- concentrated in the single week before the excluded final date rather than exactly on it. A one-date exclusion rule cannot catch this; the right diagnostic is the SHAPE of the trajectory (smooth and gradual vs. flat-then-a-late-jump), not just which single date the peak happens to fall on. **This project does not currently automate that shape check** -- it was applied here by direct inspection, the same way the earlier NC-06 threshold jump was verified (warm-start vs. cold-start SLSQP agreement) rather than taken at face value. Flagging FL-02 as NOT a genuine timing result, and recording the general lesson: any future date-grid refinement that finds a "new" positive Theta contribution needs its full trajectory inspected for this pattern before being trusted, not just checked against the single-date exclusion rule.

**Answering the review's original question directly**: "If the genuine mid-season Theta result survives K~20 and weekly timing, then I would consider the descriptive timing result mature enough to write up." One result clears that bar. AZ-09 (2024 R-side): a Republican deviation of $1M into AZ-09, held until approximately 29 days before Election Day (while Democrats still had substantial flexible budget remaining), converts an initially 40%-eroded opportunity into a fully durable one, worth +0.049 expected seats net of the best immediately-available alternative -- and this finding is now validated against a K~20 union-screened candidate pool, a redistricting-reliability filter, and 18-point weekly time resolution, independently and simultaneously. Every other candidate "genuine mid-season wait" story tested across this project's several rounds of stress-testing (CT-02, FL-22, NV-01, NC-06, FL-07, and now FL-02) turned out to be either dominated by a better immediately-available option, a redistricting-baseline artifact, a mechanical-final-week artifact, or a near-mechanical late discontinuity indistinguishable in character from one. The operational form of the finding: **a genuine strategic-timing wait signal is rare, identifiable by a smooth multi-week retention climb (not a late jump), and even when real, is only worth acting on when it beats what is already achievable today** -- exactly the optimal-stopping principle this stress test was designed to test, now demonstrated rather than assumed.

Figure: `figures/static/weekly_stress_test_verdict.png` (`scripts/plot_weekly_stress_test_verdict.py`). Raw output: `results/strategic_window_union_{cycle}_excl_redistricting_weekly.json`, `results/theta_unified_union_weekly_clean.json`.
