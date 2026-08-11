# Strategic Campaign Allocation as a Two-Player Resource-Allocation Game

Project specification, as written 2026-08-11. Reference document for this
project's scope, formalism, and validation hierarchy -- implementation
should stay traceable back to the section numbers below.

## 1. Project objective

Build and empirically evaluate a two-player model of U.S. House campaign
spending in which Democratic and Republican committees simultaneously
allocate finite budgets across the same set of races.

The project will estimate:

- The marginal value of another Democratic dollar in each race.
- The marginal value of another Republican dollar in each race.
- Each party's best response to the other party's allocation.
- The strategic equilibrium allocation of both parties.
- How far observed spending is from that equilibrium.
- Which individual races, if any, contain persistent exploitable value after
  optimal opponent response.

The primary scientific question is no longer:

> Is DCCC spending optimally?

It is:

> Are observed Democratic and Republican spending portfolios approximately a
> strategic equilibrium, and where do deviations from that equilibrium
> remain?

## 2. Core hypothesis

The working hypothesis is: apparent campaign-spending inefficiencies are
substantially larger in a one-sided optimization problem than in a two-sided
strategic game because the opposing committee can respond to the same
marginal opportunities.

The existing work provides the motivating observation. The descriptive
marginal-return pattern survives the model's corrections, while the
estimated monetary/seat value of exploiting that pattern repeatedly
contracts as progressively stronger opponent-response assumptions are
imposed.

The new project should test this directly rather than treating it as a
concluding observation.

## 3. Unit of analysis

The fundamental decision unit is `(i, t)` where:

- `i` = House race/district.
- `t` = decision date or reporting period.

For the initial static project, `t` can be fixed at the final-cycle or
selected historical information date.

Each race has state:

```
X_i = (partisan baseline, candidate spending, Democratic party spending,
       Republican party spending, incumbency, race competitiveness,
       uncertainty, ...)
```

The existing research already represents race state using expected margin,
uncertainty, Democratic and Republican spending, committed capital, and
broader political-state variables.

## 4. Two-player payoff model

Let `D_i` denote Democratic discretionary spending in race `i`, and `R_i`
Republican discretionary spending. For each race estimate:

```
p_i(D_i, R_i, X_i) = P(Democrat wins race i)
```

Then Democratic expected seats are:

```
U_D(D, R) = sum_i p_i(D_i, R_i, X_i)
```

If the game is modeled as constant-sum, Republican utility is:

```
U_R(D, R) = N - U_D(D, R)
          = sum_i [1 - p_i(D_i, R_i, X_i)]
```

That constant-sum formulation should be the primary model unless empirical
considerations require party-specific utility functions.

## 5. Budget constraints

Each player has a finite spending budget:

```
sum_i D_i <= B_D
sum_i R_i <= B_R
```

Race-level constraints are:

```
0 <= D_i <= D_i_bar
0 <= R_i <= R_i_bar
```

The upper bounds should incorporate the existing persuasion/extrapolation
ceiling rather than allowing either optimizer to exploit unsupported regions
of the spending-response function.

The ceiling therefore remains useful, but its role changes. It is no longer
a safeguard around a Democratic optimizer. It becomes a symmetric
feasible-action constraint on both players.

## 6. Race-level marginal values

For Democrats:

```
MSG_i^D = d p_i / d D_i
```

For Republicans:

```
MSG_i^R = - d p_i / d R_i
```

Both should be expressed in a common unit such as expected seats per $1
million.

The existing opponent-reaction work should not be mechanically inserted
into these derivatives if the opponent is now an endogenous player. In the
new game, Republican spending is a decision variable rather than a
deterministic function `R(D)`.

This is an important conceptual separation from Paper III, where opponent
reaction is estimated as a reactive transition component.

## 7. Portfolio shadow prices

A raw race-level marginal return is not sufficient to establish
inefficiency because each party faces its own opportunity cost of capital.

At an interior optimum, Democrats satisfy `MSG_i^D = lambda_D`, while
Republicans satisfy `MSG_i^R = lambda_R`, where `lambda_D, lambda_R` are the
respective budget shadow prices.

Define Democratic marginal surplus `S_i^D = MSG_i^D - lambda_D`, and
Republican marginal surplus `S_i^R = MSG_i^R - lambda_R`.

For comparability across budgets, also calculate normalized surplus:

```
Z_i^D = MSG_i^D / lambda_D - 1
Z_i^R = MSG_i^R / lambda_R - 1
```

These become the key race-level strategic diagnostics.

## 8. Race taxonomy

Each district can then be classified according to `(Z_i^D, Z_i^R)`. The
initial interpretation should be:

| Z^D | Z^R | Interpretation |
|---|---|---|
| near 0 | near 0 | Locally equilibrated |
| positive | near 0/negative | Democratic opportunity |
| near 0/negative | positive | Republican opportunity |
| positive | positive | Under-contested / escalation pressure |
| negative | negative | Possible over-capitalization |
| opposite large signs | | Strong strategic asymmetry |

This taxonomy should be treated as descriptive until verified by explicit
best-response calculations.

## 9. Best-response functions

For observed Republican spending `R_obs`, Democratic best response is:

```
BR_D(R_obs) = argmax_D U_D(D, R_obs)
```

subject to the Democratic budget and race constraints. Likewise:

```
BR_R(D_obs) = argmax_R U_R(D_obs, R)
```

These produce the first two key quantities:

```
Regret_D = U_D(BR_D(R_obs), R_obs) - U_D(D_obs, R_obs)
Regret_R = U_R(D_obs, BR_R(D_obs)) - U_R(D_obs, R_obs)
```

Call these unilateral exploitability. They answer: how much can either side
improve if the opponent is held fixed?

## 10. Strategic exploitability

Define total observed-profile exploitability as `E = Regret_D + Regret_R`.

An exact Nash equilibrium has `E = 0`. Observed allocations with small `E`
are approximately strategically efficient even if they fail a one-sided KKT
test.

This should probably become the main empirical statistic of the project.
Report it in:

- Expected seats.
- Percentage of one seat.
- Percentage of total expected competitive seats.
- Equivalent spending value where meaningful.

## 11. Equilibrium model

The primary equilibrium object is `(D*, R*)` such that `D* = BR_D(R*)` and
`R* = BR_R(D*)`.

For the constant-sum formulation, also investigate the saddle-point
formulation: `max_D min_R U_D(D, R)`, subject to each party's feasible set.

The observed portfolio can then be compared against:

- Democratic unilateral optimum.
- Republican unilateral optimum.
- Strategic equilibrium.
- Observed joint allocation.

## 12. Computational solution

Start with iterative best response:

```
R_0 = R_obs
D_1 = BR_D(R_0)
R_1 = BR_R(D_1)
D_2 = BR_D(R_1)
...
```

and continue until `||D_{k+1} - D_k||_1 + ||R_{k+1} - R_k||_1 < epsilon`.

Track both allocations and utilities at every iteration. Do not assume
convergence implies a unique Nash equilibrium.

Required diagnostics:

- Multiple starting allocations.
- Democratic-first versus Republican-first updating.
- Simultaneous versus sequential best response.
- Convergence tolerance.
- Utility convergence.
- Allocation convergence.
- Presence of cycles.
- Multiple-equilibrium detection.

The existing nonlinear optimizer should be the benchmark solver. The
concave-envelope/water-filling work from Paper III may provide a fast
alternative after symmetrical validation for both players; the existing
surrogate is already validated to very small aggregate objective-value error
against the nonlinear optimizer.

## 13. Persistent strategic opportunity

This should be a central new object.

For race `i`, begin with a small Democratic local deviation:
`D_i' = D_i + delta`. Allow Democrats to finance that change by removing
`delta` from their portfolio's current marginal use. Then let Republicans
optimally respond.

Define persistent strategic value:

```
PSV_i^D = U_D(D', BR_R(D')) - U_D(D, R)
```

Equivalent Republican statistic: `PSV_i^R`.

This asks something more useful than MSG: does a race remain attractive
after the opponent is allowed to respond rationally?

A high unilateral MSG combined with near-zero PSV means the apparent
opportunity is competed away. A high positive PSV is a genuine candidate for
persistent strategic mispricing.

## 14. Strategic response decomposition

For every candidate opportunity, decompose:

```
Unilateral value = retained strategic value + opponent-response erosion
```

Specifically:

```
V_i^uni        = U_D(D', R) - U_D(D, R)
V_i^strategic  = U_D(D', BR_R(D')) - U_D(D, R)
Erosion_i      = V_i^uni - V_i^strategic
Retention_i    = V_i^strategic / V_i^uni
```

This could become one of the project's most interpretable results.

Example: a race appears worth +0.12 expected seats under unilateral
optimization, but only +0.01 after optimal Republican response; 92% of the
apparent edge is competed away.

## 15. Historical empirical design

Initial cycles: 2022, 2024. Then extend backward as data quality permits.
Run each cycle independently. For each cycle calculate:

- Observed Democratic allocation.
- Observed Republican allocation.
- Democratic unilateral best response.
- Republican unilateral best response.
- Iterated strategic equilibrium.
- Democratic regret.
- Republican regret.
- Total exploitability.
- Race-level `Z_D, Z_R`.
- Race-level persistent strategic value.
- Distance from observed allocation to equilibrium.

A result replicated in both 2022 and 2024 is substantially stronger than a
live-cycle-only finding.

## 16. Prospective 2026 analysis

The 2026 analysis should be explicitly secondary until the historical game
is validated. At each live information date `X_t`, construct `(D_t*, R_t*)`.
Then report:

- Current equilibrium.
- Observed allocations.
- Best responses.
- Exploitability.
- Strategic-surplus map.
- Persistent strategic opportunities.
- Changes from previous reporting period.

Do not initially make a statement such as "Democrats can gain X seats."
Instead say: "Under the estimated two-player model, the current spending
profile has X expected-seat exploitability, with Y attributable to
Democratic deviation opportunities and Z to Republican deviation
opportunities."

## 17. Public data

This should reuse the existing project's public-data architecture rather
than creating a separate data universe.

Confirmed reusable sources from the existing project include:

- FEC independent-expenditure data / Schedule E.
- Dated candidate-committee reports through the FEC API.
- Historical generic-ballot data.
- Existing race fundamentals and political-state inputs from Papers I-II.
- Existing candidate and party spending reconstructions.

The project should maintain the same public-data-only constraint. A formal
data catalog should distinguish:

- Candidate spending.
- Party committee spending.
- Independent expenditures.
- Democratic-aligned outside spending.
- Republican-aligned outside spending.
- Race fundamentals.
- District boundaries.
- Candidate identity/status.
- Competitiveness ratings.
- Election outcomes.

## 18. Major identification problem

The model must distinguish strategic response from common response to race
competitiveness. If both parties spend more in a race because it suddenly
becomes competitive, naively regressing Republican spending on prior
Democratic spending will overstate strategic reaction.

The new project should therefore treat the existing eta estimates as
descriptive/supporting evidence, not as the definition of strategic
behavior. The game itself should endogenize response through `BR_D, BR_R`.
This is conceptually cleaner.

## 19. Symmetry tests

A two-player project requires explicitly checking whether the
spending-response function is actually symmetric. Estimate `dp/dD` and
`-dp/dR` separately if the data permit. Test whether `beta_D = beta_R`.

If symmetry cannot be rejected, the simpler symmetric game is justified. If
it can, party-specific persuasion efficiencies must be retained. Do not
simply mirror the Democratic response curve onto Republicans without testing
it.

## 20. Validation hierarchy

The project should have four validation levels.

- **Level A -- Response model.** Does relative spending predict
  margins/outcomes out of sample?
- **Level B -- Optimization.** Do nonlinear and surrogate solutions agree in
  objective value and KKT conditions?
- **Level C -- Game solver.** On synthetic games with known equilibria, does
  the algorithm recover them?
- **Level D -- Historical behavior.** Are observed allocations closer to the
  estimated Nash equilibrium than to reasonable alternative strategies?

That final test is particularly interesting. Compare observed spending
with: equal allocation; Cook-category heuristic; one-sided optimizer; Nash
equilibrium; random feasible portfolios. If real committees repeatedly land
unusually close to Nash, that would be powerful evidence that the strategic
model captures something real.

## 21. Primary hypotheses

Preregistered roughly as follows.

- **H1 -- Unilateral exploitability exists.** Observed party allocations are
  not exactly optimal holding opponent spending fixed.
- **H2 -- Strategic response substantially reduces exploitability.** The
  expected-seat gain from unilateral reallocation exceeds the gain remaining
  after optimal opponent response.
- **H3 -- Observed allocations are substantially closer to Nash equilibrium
  than to unilateral optima.**
- **H4 -- Most high-MSG opportunities have low persistent strategic value.**
- **H5 -- A small subset of races retains positive strategic value even
  after best response.**
- **H6 -- Strategic exploitability declines as Election Day approaches.**

H5 is the one that could produce the most operationally interesting result.

## 22. Primary outputs

The project should produce at least five headline outputs.

1. **Strategic marginal-value map.** Scatter `(Z_D, Z_R)` for every district.
2. **Best-response trajectory.** Show how allocation and expected seats
   evolve from observed spending toward equilibrium.
3. **Exploitability table.**

   | Cycle | Dem regret | GOP regret | Total exploitability |
   |---|---|---|---|
   | 2022 | | | |
   | 2024 | | | |
   | 2026 live | | | |

4. **Opportunity erosion table.**

   | Race | Unilateral value | After GOP response | Retention |
   |---|---|---|---|

5. **Observed versus Nash allocation comparison.** Measure: L1 allocation
   distance; race-ranking overlap; expected-seat difference; shadow-price
   dispersion.

## 23. What not to carry forward

The new project should deliberately avoid several pieces of legacy framing.

- Do not use the negative spending-versus-current-MSG Spearman correlation
  as evidence of inefficiency.
- Do not treat opponent spending as purely exogenous.
- Do not make the DCCC the only optimizing actor.
- Do not equate unilateral best response with exploitable real-world value.
- Do not begin with `Theta`.
- Do not make "Democrats should move money toward Republican districts" the
  motivating claim.

Those can all appear later if the strategic game supports them.

## 24. Relationship to the previous research

The old trilogy supplies: spending-response estimation; race uncertainty;
persuasion ceiling; nonlinear optimizer; fast concave surrogate; spending
reconstruction; opponent-response evidence; dynamic transition machinery;
`Theta`.

The new project asks a different scientific question: what happens when
both organizations optimize simultaneously? Paper III already demonstrates
why this matters: control specification and opponent-response calibration
can materially change -- and even reverse -- the implied decision.

The new project makes that strategic interaction the starting point rather
than an after-the-fact robustness check.

## 25. Suggested project structure

```
strategic-campaign-allocation/
    data/
        raw/
        processed/
        catalog/
    src/
        estimation/
            spending_response.py
            uncertainty.py
        game/
            payoff.py
            gradients.py
            best_response.py
            equilibrium.py
            exploitability.py
            persistent_value.py
        optimizer/
            nonlinear.py
            concave_surrogate.py
        validation/
            synthetic_games.py
            historical_backtest.py
    scripts/
        build_cycle_state.py
        estimate_response_model.py
        solve_best_responses.py
        solve_nash.py
        compute_exploitability.py
        compute_race_surplus.py
        compute_persistent_value.py
        run_historical_backtest.py
        run_live_2026.py
    tests/
    figures/
    results/
    docs/
        project_spec.md
        data_dictionary.md
        methodology.md
```

## 26. Minimum viable research project

Do not begin by rebuilding the entire dynamic system. The first decisive
version needs only:

1. Freeze 2022 final state.
2. Estimate `p_i(D_i, R_i)`.
3. Validate gradients.
4. Compute both parties' unilateral best responses.
5. Solve iterative Nash.
6. Compute exploitability.
7. Repeat for 2024.
8. Compute race-level persistent strategic value.
9. Determine whether the near-zero aggregate Nash result replicates.

If it does, you already have a paper. Only after that should you add 2026
and time.

## 27. Decision gate

The project should continue to the full dynamic version if either of these
is true:

- **A.** Observed allocations are consistently close to Nash in multiple
  historical cycles. That would support a paper about strategic efficiency
  despite apparent unilateral inefficiency.
- **B.** Aggregate exploitability is low, but a stable subset of races has
  persistent strategic value. That would support a paper about localized
  strategic mispricing within an otherwise efficient political capital
  market.

If neither occurs -- if Nash is unstable, historically implausible, and no
persistent race-level structure replicates -- that is also informative and
should stop the project before adding unnecessary complexity.

---

## Addendum: Manim visualization layer

Manim (ManimGL, `3b1b/manim` -- distinct from Manim Community Edition;
their install instructions are not interchangeable) is a strong fit for this
project because the subject matter is fundamentally about interaction,
response, convergence, and equilibrium -- dynamic concepts, not just static
tables.

Six scene families map onto the theoretical objects above:

1. **Two-player marginal-value surface** -- `p_i(D_i, R_i)` as a surface over
   (D-spend, R-spend) axes for a single race.
2. **Best-response dynamics** -- the `(D_0, R_0) -> (D_1, R_1) -> ... ->
   (D*, R*)` convergence path (Section 12).
3. **Race-level strategic-surplus map** -- animated `(Z_D, Z_R)` scatter as
   budgets/allocations change (Section 8).
4. **How the apparent edge disappears** -- the old trilogy's
   13.24 -> 2.83 -> 1.36 -> 0.10 progression as successively stronger
   opponent-response assumptions are added.
5. **Budget allocation as water filling** -- races as containers, money
   flowing in until marginal returns equalize at `lambda_D`/`lambda_R`.
6. **Unilateral value vs. persistent strategic value** -- a race's apparent
   `V_uni` collapsing to its `V_strategic` once the opponent best-responds
   (Section 14).

Design rule: Manim never calculates the scientific result. It consumes
frozen result files already written by the research pipeline (`results/*.json`,
`results/*.csv`, `results/*.parquet`) -- `data -> estimation -> optimization
-> frozen results -> Manim`, never `Manim scene -> hidden calculation`. See
`visuals/README.md` for the scene-to-object mapping and current install
status.
