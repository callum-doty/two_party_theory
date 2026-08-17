# The sequential timing result: process and outcome

Standalone writeup of one self-contained research arc: unifying this
project's two separately-computed "value of waiting" diagnostics into a
single sequential decision model, then stress-testing the result until it
either broke or held up. Full blow-by-blow entries are in
`docs/methodology.md` (dated sections, 2026-08-14); this document is the
narrative version -- what was done, in what order, and what survived.

## 1. Starting point: two diagnostics, not one decomposition

Before this arc, the project had two separately-built pieces:

- **`value_of_waiting.py`**: compared a race's value if deployed today
  against its value if held until it became durable (`T_i^80`, the date
  retention first crosses 80% and stays there). Ranked candidate races by
  **PSV** (the game-theoretic, opponent-best-response-adjusted value).
- **`information_value.py`**: simulated a committee deciding under
  generic-ballot uncertainty. Ranked candidate races by **V_uni** (a cheap,
  closed-form proxy), specifically because its own zero-noise sanity check
  had found that V_uni and PSV rank races *differently* -- 2024 D-side,
  V_uni picks CT-02, PSV picks WI-01.

`Theta = value_of_waiting + information_value` was being reported as a
decomposition, but the two terms were the output of two different
maximization problems. A research-direction review flagged this directly:
not yet a decomposition of one Bellman value, and the "strategic
flexibility is carrying Theta" conclusion built on top of it was
premature.

## 2. Building one Bellman value

**The fix** (`src/game/unified_theta.py`): a single decision rule, used at
every reference date and in every counterfactual. A committee's estimate
of race *i*'s PSV under a noisy generic-ballot signal is
`V_uni_noisy(epsilon) x retention(t)` -- V_uni's existing cheap
noise-simulation machinery, combined with `retention(t)`, the TRUE
(zero-noise) PSV/V_uni ratio `strategic_window.py` already computes at
each date. The committee picks the race with the highest *estimated* PSV
under noise; the realized payoff is that race's *true* PSV.
`game/unified_theta.py::solve_bellman` backward-induces
`V_t = max(V_deploy_t, V_{t+1})` across the reference-date grid, with the
final date forced to `V_deploy` (a mechanical floor, not a real
continuation target -- see Section 4).

Three counterfactual regimes, computed on the *same* recursion:

- **full**: both information noise and opponent-commitment maturation active.
- **flex_only**: information frozen at zero noise -- isolates the value of
  waiting for the opponent's capital to lock up.
- **info_only**: opponent commitments frozen at the earliest date's level
  -- isolates the value of the committee's own uncertainty resolving.

`Theta_full` is reported alongside `Theta_flex_only + Theta_info_only`
without asserting they're equal; the gap (`interaction`) is reported
explicitly.

**First result, K=3 candidates per side** (the pre-existing "top-3-by-leverage"
pool), same 8-date grid (120/90/60/45/30/21/14/7 days before Election Day):

| Cycle/side | Theta_full | Theta_flex_only | Theta_info_only |
|---|---|---|---|
| 2024 D | +0.0620 | +0.0612 | +0.0006 |
| 2024 R | +0.1553 | +0.1553 | +0.0000 |
| 2022 D | +0.0002 | +0.0000 | +0.0002 |
| 2022 R | +0.0002 | +0.0000 | +0.0002 |

The two-module conclusion survived unification in 3 of 4 pools: strategic
flexibility, not information, carries Theta. One genuine correction fell
out along the way: CT-02's true value of waiting (+0.062) is roughly 8x
the old `value_of_waiting.py` estimate (+0.008), because that script
measured value at the *first* date retention crossed 80%, while CT-02's
PSV actually peaks later (183% retention at 21 days out, a real
second-order equilibrium effect this project had already documented
elsewhere) before settling back down. The Bellman recursion catches this
automatically -- it takes the max over the *entire* remaining horizon, not
just the first date clearing a threshold.

## 3. First expansion: K~8, and a discovery

Widened to `strategic_leverage.py`'s full "primary" candidate pool
(top-4-swing + top-4-|Z|, 7-8 districts/side) instead of just the top-3.
Real exact-SLSQP cost: ~136-144 new solves per cycle, ~35-40 minutes/cycle.

Three of four pools were unchanged to four decimal places -- the top-3
screen already contained the portfolio-optimal choice. The exception,
2022 R-side, gained two new candidates (FL-07, PA-04) with V_uni roughly
2.5-3x anything in the original pool, raising `Theta_full` from +0.0002 to
+0.0622.

## 4. The mechanical-floor trap

Every race's retention converges to ~100% at the final reference date (7
days out) *by construction* -- both committees' flexible budgets are
nearly exhausted for everyone at that point, independent of whether that
specific race was ever genuinely contested. `solve_bellman`'s
`max(..., deploy_last)` had no safeguard against this: a pool member with
the single largest raw V_uni will always look "worth waiting for" at the
final date, regardless of its actual trajectory -- and the risk grows with
K, since a wider pool is more likely to contain some large-V_uni race.

`scripts/theta_final_week_sensitivity.py` reruns the same recursion
excluding the final date (pure post-processing, no new solves):

| Pool | Cycle/side | Theta_full | Realized at | Excl. final week | Reading |
|---|---|---|---|---|---|
| K=3 | 2024 D | +0.0620 | 21d out | +0.0620 | genuine |
| K=3 | 2024 R | +0.1553 | 7d out | **+0.0430** | mostly mechanical |
| K~8 | 2022 R | +0.0622 | 7d out | **+0.0037** | almost entirely mechanical |

The 2022 R "discovery" from Section 3 (FL-07) was ~94% a final-week
artifact, not a genuine strategic-timing find.

## 5. K=15-20: a principled wider screen, and a second confound

**The pre-screen** (`scripts/rank_candidate_races.py`): a K=10-20 action
space needs a broader candidate rule than "top current leverage" -- FL-07
had poor *current* leverage but a large late-arriving opportunity, exactly
the blind spot a leverage-only screen has. Built as

```
union(top |Z|, top V_uni, equilibrium-swing races, revealed-contested races)
```

Three of the four criteria are cheap and closed-form (no solves): top |Z|
reuses `race_surplus_{cycle}.csv`; top V_uni is computed FRESH across the
full 433-race universe for the first time; equilibrium-swing reuses the
double-oracle mixed-equilibrium support decomposition. The fourth
("largest strategic-window slope") can't be computed directly without the
very date-sweep this pre-screen exists to avoid running on all 433 races
-- approximated instead as "revealed contested-ness" (races the opponent
already spends heavily on, a proxy for where a full best response would
defend hardest). Run on both cycles in seconds: union sizes landed at
K=20/16 (2024 D/R) and K=20/17 (2022 D/R), in the target range without
manual tuning. FL-07 reappeared independently via top-V_uni -- cross-validation
that it's a real, reproducible feature of the data, just not (per Section 4)
a genuine timing opportunity.

**Real cost**: `compute_strategic_window.py --pool union` swept the full
union pool across the 8-date grid, ~616 exact-SLSQP solves, ~85 minutes
wall-clock running both cycles in parallel.

**Headline, before further correction**: 2024 D rose to +0.0796; 2024 R
fell to +0.1033; 2022 D stayed ~0; 2022 R rose to +0.0186. Tracing what
drove the new 2024 numbers surfaced a confound distinct from the
mechanical floor: the top-V_uni entrants pulling in the biggest new
candidates were **NC-06, NC-13, NC-14** (2024 D) and **NC-01/06/13/14**
(2024 R) -- all four among this project's 13 `redistricting_flagged`
districts, already documented elsewhere as having a less certain baseline
(a prior "$5M threshold jump" investigation had traced a real, sharp
discontinuity in NC-06's response surface to exactly this). A top-V_uni
screen has no way to distinguish a genuinely large, reliable opportunity
from a large number produced by an unstable baseline.

`compute_theta_unified.py` gained `--exclude-redistricting` (filters on
`RaceRecord.redistricting_flagged` directly, staying in sync with
`config.yaml` rather than a hardcoded list). **Doubly-corrected results**
(redistricting-flagged excluded, final week excluded):

| Cycle/side | K (clean) | Theta_full | Realized at | Driving race |
|---|---|---|---|---|
| 2024 D | 17 | **+0.0032** | 30d out | FL-27 -- already durable day one |
| 2024 R | 12 | **+0.0490** | 30d out | **AZ-09** -- genuine, 60%→100%+ retention |
| 2022 D | 19 | **+0.0000** | 120d out | FL-02 -- already durable day one |
| 2022 R | 17 | **+0.0000** | -- | (was mechanical before correction) |

2024 D's genuine value collapsed from +0.0796 to +0.0032 -- and CT-02
(Section 2's headline example) turns out not to be why. The union pool
added **FL-27**, a race the K=3/K~8 pools never included, already at 112%
retention on day one with V_uni more than double CT-02's. Once FL-27 is in
the choice set, waiting specifically for CT-02 is strictly dominated
(net -0.043 against FL-27's `best_immediate`). This is the *other* half of
the optimal-stopping principle demonstrated concretely: "deploy now when a
strong, durable opportunity already exists" is exactly what FL-27
represents.

2024 R's surviving case is **AZ-09**, not NC-06 -- smoothly rising from
60.5% retention at 120 days out to >100% by 30 days out, not
redistricting-flagged, beating its own best alternative (NM-01) by a clear
margin. FL-07's 2022 R "discovery" is closed out completely at this stage:
dominated by TN-09 (durable from day one) on top of being ~94% mechanical.

Of five previously-reported "genuine mid-season wait" races (CT-02, FL-22,
NV-01, NC-06, FL-07), four were dominated or artifactual once the search
widened and the redistricting confound was controlled for. Only AZ-09
(plus, more weakly, NM-01) survived.

## 6. Second stress test: weekly time resolution

The last open question: does AZ-09's smooth trajectory hold up at finer
time resolution, or was its 8-date profile hiding a sharp jump the way
NC-06's did? Per the review's own sequencing, this was run *after* the
K-widening was validated, not combined with it.

**Method**: `compute_strategic_window.py` gained `--weekly` (an 18-point
grid, 120 down to 7 days out in 7-day steps, keeping the same 7-day
mechanical-floor endpoint) and `--exclude-redistricting` was reordered to
filter candidates *before* sweeping, so no solves were wasted on
already-known-unreliable districts. Run on the same stable candidate set
Section 5 validated: ~1,242 exact-SLSQP solves, ~2.9-3.9 hours wall-clock
per cycle running both cycles in parallel.

**Headline, weekly grid, clean pool**:

| Cycle/side | Theta_full (weekly) | Theta_full (8-date) | Realized at | Verdict |
|---|---|---|---|---|
| 2024 D | +0.0032 | +0.0032 | 29d out | unchanged, negligible |
| 2024 R | **+0.0492** | +0.0490 | 29d out | **unchanged to 3 decimals** |
| 2022 D | +0.0003 | +0.0000 | 64d out | unchanged, negligible |
| 2022 R | +0.0319 | +0.0000 | 8d out | **new -- investigated, rejected** |

**AZ-09 is confirmed.** Its weekly trajectory is completely smooth and
monotonic: 60.5% retention flat through early August, climbing steadily
through September (62% -> 87%), crossing 100% cleanly at 29 days out, flat
afterward. `Theta_full` moved from +0.0490 to +0.0492 -- a difference
smaller than Monte Carlo noise. This is the strongest evidence in the
project for a genuine, actionable mid-season timing signal: it survives a
wider pool, a redistricting filter, and finer time resolution,
independently and simultaneously.

**2022 R's new number needed the same scrutiny CT-02 and NC-06 got, and
failed it.** The single-date exclusion rule doesn't flag it (realized at 8
days out, one point before the excluded date). But the underlying
trajectory (district FL-02) is flat at 13.6% retention for ten
consecutive weekly readings (120 days out through 24 days out), then jumps
to 160% in the single week before the excluded final date. That is the
same sharp, late, threshold-like pattern already documented for NC-06 --
concentrated one week earlier than the mechanical floor rather than
exactly on it. A single-date exclusion rule cannot catch this; the right
diagnostic is the *shape* of the trajectory (smooth and gradual vs.
flat-then-a-late-jump), which this project does not yet automate --
applied here by direct inspection, the same way NC-06's threshold jump was
originally verified.

## 7. Outcome

**One result cleared the review's bar** ("if the genuine mid-season Theta
result survives K~20 and weekly timing, then I would consider the
descriptive timing result mature enough to write up"):

> A Republican deviation of $1M into **AZ-09**, held until approximately
> **30 days before Election Day** (while Democrats still had substantial
> flexible budget remaining), converts an initially ~40%-eroded
> opportunity into a fully durable one -- worth **+0.049 expected seats**
> net of the best immediately-available alternative (NM-01). Validated
> against a K~20 union-screened candidate pool, a redistricting-reliability
> filter, and 18-point weekly time resolution, independently and
> simultaneously.

Every other candidate "genuine mid-season wait" story tested across this
arc (CT-02, FL-22, NV-01, NC-06, FL-07, FL-02) turned out to be either
dominated by a better immediately-available option, a redistricting-baseline
artifact, a mechanical-final-week artifact, or a near-mechanical late
discontinuity indistinguishable in character from one.

**The operational principle**, stated at the start of this arc and now
demonstrated rather than assumed:

> Hold flexible capital when the expected improvement in the future
> strategic opportunity set exceeds today's best deployment value. Deploy
> now when a strong, durable opportunity already exists. Wait when today's
> options are weak and opponent commitments are expected to create a
> materially better opportunity before the terminal spending squeeze.

A genuine strategic-timing wait signal, per everything found in this arc,
is **rare**, identifiable by a **smooth multi-week retention climb** (not a
late jump), and even when real, **only worth acting on when it beats what
is already achievable today**.

## 8. What this doesn't establish, and the natural next steps

- **No automated shape-check for trajectories.** FL-02 and NC-06 were both
  caught by direct inspection, not a script. A future pass could formalize
  "flat for N periods, then a jump of magnitude M within the last K
  periods" as an explicit filter alongside the single-date exclusion rule.
- **The union pre-screen's fourth criterion is still an approximation.**
  "Revealed contested-ness" stands in for "largest strategic-window slope"
  because the true quantity needs the sweep the screen exists to avoid.
  AZ-09 was found via top-V_uni, not this criterion -- worth checking
  whether a genuinely slope-aware screen (impossible without the sweep
  itself, but perhaps approximable via a cheap two-point difference) would
  have found it faster or found something else.
- **Single-race, single-delta.** Every number here is "one race, one $1M
  deviation, one reference decision." The full sequential game -- a
  committee allocating its ENTIRE flexible budget across many races over
  time, not one race in isolation -- was explicitly out of scope for this
  arc (per the review's own sequencing: prove the single-race timing logic
  first).
- **2022 is a near-total null result**, both sides, at every pool size and
  time resolution tested. Worth understanding on its own terms rather than
  just noting it: is 2022's flatter, more contested strategic landscape
  (documented elsewhere in this project -- the double-oracle equilibrium's
  larger support size that cycle) the reason waiting has so little value
  when the underlying game itself has fewer clearly-dominant options in
  either direction?

## Reproducibility

```
python scripts/rank_candidate_races.py --cycle {2024,2022}
python scripts/compute_strategic_window.py --cycle {2024,2022} --pool union --exclude-redistricting --weekly
python scripts/compute_theta_unified.py --pool union_weekly_clean
python scripts/theta_final_week_sensitivity.py
python scripts/plot_weekly_stress_test_verdict.py
```

Key source: `src/game/unified_theta.py` (decision rule + Bellman
recursion). Key scripts: `scripts/rank_candidate_races.py`,
`scripts/compute_strategic_window.py`, `scripts/compute_theta_unified.py`,
`scripts/theta_final_week_sensitivity.py`,
`scripts/analyze_conditional_waiting_value_union.py`. Figures:
`figures/static/theta_unified_summary.png`,
`figures/static/theta_final_week_sensitivity.png`,
`figures/static/conditional_waiting_value_final.png`,
`figures/static/weekly_stress_test_verdict.png`. Full dated log entries:
`docs/methodology.md` (search "Unified sequential decision value" through
"Second stress-test check"). Regression tests:
`tests/test_unified_theta.py`.
