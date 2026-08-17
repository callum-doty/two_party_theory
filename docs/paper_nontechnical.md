# When Should Campaigns Spend? {-}

### What a Strategic Model of House Campaign Finance Actually Finds {-}

*A nontechnical account of reciprocal strategy, timing, and the search for genuine opportunities in the 2022 and 2024 U.S. House elections*

# Abstract {-}

Campaign committees face a deceptively simple question: where and when should they spend their money?

At first glance, the answer seems straightforward. Find the races where another dollar is most valuable and spend there. But the opposing party is making the same calculation. A strategy that looks attractive when the opponent does nothing may become much less attractive once the opponent responds.

This project builds a two-player model of Democratic and Republican House campaign spending and asks a simple question:

*Can one party find a spending opportunity that the other party cannot simply respond to?*

The answer is mostly no — but with an important exception.

When both parties are allowed to adjust their spending, most apparent opportunities disappear. The model instead finds a small set of near-equivalent strategies rather than one unbeatable spending plan. Actual committee spending also resembles a simple competitiveness-based rule more closely than either a fully optimized strategy or the model's equilibrium.

Time changes the problem. Money spent today is no longer available tomorrow, so an early investment can eventually become difficult for the opponent to counter. This creates a potential timing advantage.

The project then subjects that timing effect to increasingly demanding tests. Apparent opportunities disappear because of mechanical end-of-cycle effects, better alternatives, unstable district data, and insufficient time resolution.

After all of these tests, one 2024 Republican-side case, AZ-09, survives. Its estimated value is about +0.049 expected seats.

That number should not be interpreted as a typical return from waiting. It is an existence proof: evidence that the strategic timing mechanism can genuinely occur, not evidence that campaigns generally benefit from waiting.

The broader conclusion is that campaign spending appears to behave less like an inefficient market waiting to be exploited and more like an already-adapted strategic environment in which both sides have learned broadly similar conventions about when and where to spend.

# 1. The Question

Imagine two campaign committees, one Democratic and one Republican.

Each has a limited amount of money and hundreds of House races in which that money could be spent.

The obvious strategy is:

Find the races where spending produces the largest improvement in expected electoral outcomes and put money there.

But there is a problem.

The other committee can do the same thing.

Suppose Democrats discover that spending another $1 million in a particular race is unusually valuable. Republicans notice the spending and respond. Democrats then have to decide whether the opportunity is still attractive after the Republican response.

This creates a fundamentally different question:

*Is there an opportunity that remains valuable even after the opponent reacts?*

And there is a second question.

Even if such an opportunity exists today, perhaps the committee should wait.

Why?

Because money that has already been spent is effectively locked into the battlefield. As Election Day approaches, both committees have less flexible money remaining. A spending move that Republicans could easily counter early in the campaign might become much harder to counter later.

That produces the central puzzle of this project:

**Can campaign committees create a strategic advantage by choosing not only where to spend, but when to spend?**

# 2. First, Get the Game Right

Before asking who has an advantage, the model has to define what each player actually controls.

An early version of the project treated essentially all non-candidate spending in a district as money that the national committees could move around.

That was too broad.

State parties and outside groups make independent spending decisions. The DCCC and NRCC cannot simply take those dollars out of one district and put them into another.

So the model was rebuilt around money the two national committees can actually control.

That reduced the modeled 2024 budget from roughly $465 million to $102 million for Democrats and from $132 million to $47 million for Republicans.

This correction mattered.

A strategic game only makes sense if the players are allowed to move the pieces they are actually responsible for moving.

The project also uncovered several data and modeling problems at this stage, including missing state-party spending and a bug that could produce an impossible negative measure of regret.

Those problems were fixed before the main analysis proceeded.

That became an important principle for the rest of the project:

*If a result looks interesting, try to break the machinery that produced it before believing the result.*

# 3. What Happens If Only One Side Changes Its Strategy?

The first experiment was deliberately simple.

Take what each committee actually spent.

Now imagine that one committee is allowed to rearrange its own money while the opponent keeps its spending exactly where it was.

This measures one-sided opportunity.

The model found roughly:

- 2022: 5.44 expected seats of combined opportunity
- 2024: 5.14 expected seats of combined opportunity

That sounds substantial.

If the opponent really did nothing, there would be several expected seats available through better allocation.

But this is not yet a realistic strategic result.

It assumes the opponent sits still.

# 4. The Opponent Fights Back

The next experiment lets both sides respond.

Democrats optimize against Republicans. Republicans respond to the Democrats. Democrats respond again. And so on.

The important result is that the process does not settle into one simple, permanent spending plan.

Instead, it cycles among several nearly equivalent portfolios.

This is useful information.

It tells us that there is not necessarily one magic collection of races that beats everything else.

Instead, the strategic solution looks more like a small menu of reasonable portfolios.

The model finds:

- a 2024 equilibrium supported by five portfolios per side;
- a 2022 equilibrium supported by eleven portfolios per side.

The 2022 solution also touches substantially more races.

The practical interpretation is simple:

*Once both parties are allowed to respond intelligently, the obvious spending opportunities become much harder to exploit.*

A strategy that looks brilliant against a passive opponent may simply cause the opponent to move somewhere else.

# 5. What Do Real Campaigns Actually Do?

This raises another interesting question.

If the mathematically optimized strategies are so clever, do real committees appear to be using them?

The answer is surprisingly clear.

The model compared actual spending with five alternatives:

1. equal spending,
2. a simple competitiveness-based strategy,
3. a one-sided optimizer,
4. the mixed equilibrium,
5. random feasible spending portfolios.

Actual spending was much closer to the Cook-rating competitiveness heuristic than to either of the optimization-derived strategies.

And although the optimization strategies produced slightly higher expected Democratic seat totals, the improvement was small.

For example, in 2024:

- observed allocation: 217.17 expected Democratic seats
- Cook-style heuristic: 218.14
- one-sided optimizer: 218.50
- mixed equilibrium: 218.60

The optimized strategies were also much farther away from what committees actually spent.

This creates an interesting picture:

*Campaign committees do not appear to be solving the mathematical game explicitly, but their simple rules of thumb are not obviously terrible.*

That is consistent with an environment in which organizations repeatedly compete against one another and learn from previous elections.

# 6. The Missing Ingredient: Time

So far, the model has treated spending as if it could be moved around whenever necessary.

Real money does not work that way.

If a committee spends $5 million in September, that $5 million cannot be recovered in November and moved somewhere else.

This creates two kinds of money:

**Flexible money** — still available to spend.

**Committed money** — already spent and therefore locked into particular races.

Early in the campaign, the opponent has a large amount of flexible money. Late in the campaign, much less remains.

This changes the strategic problem.

Suppose Democrats spend money in a race early. Republicans might respond immediately.

But suppose Democrats make the same move later, after Republicans have already committed most of their money elsewhere.

The Republican response may now be much harder.

This produces a potential timing mechanism:

*Waiting can make an otherwise counterable move more durable because the opponent gradually loses the flexibility needed to respond.*

# 7. But Becoming Durable Is Not the Same as Being Worth Waiting For

This distinction became one of the most important findings of the project.

Imagine a race where your spending becomes increasingly difficult for the opponent to counter as Election Day approaches.

That sounds like a reason to wait.

But there is another possibility:

There may be a better race to spend the money on right now.

Therefore, the real question is not:

"Does waiting make this race more valuable?"

It is:

"Does waiting for this race eventually become better than spending the money somewhere else today?"

This distinction eliminates several apparent opportunities.

One example is AZ-02 in 2022. Its strategic durability increased substantially as the election approached. But another race, NY-20, remained a better immediate opportunity.

So:

**greater durability does not mean greater value of waiting.**

That distinction prevents a potentially serious mistake in interpreting the timing mechanism.

# 8. The First Timing Opportunities

The first version of the timing analysis looked promising.

In an initial three-race pool, the model found a waiting value as large as +0.155 expected seats for the 2024 Republican side.

That was exciting.

It was also exactly the moment when the analysis needed to become more skeptical.

A positive result discovered after searching a handful of races can easily be an artifact of how the search was constructed.

So the project began trying to eliminate it.

# 9. Stress Test #1: The End-of-Campaign Trap

The first problem was mechanical.

Near Election Day, both parties have spent almost all of their flexible money.

Therefore, almost any race becomes difficult to counter.

If we measure the value of waiting at the very end of the campaign, nearly every race looks durable.

That does not mean the race contained a genuine strategic opportunity.

It simply means: there is almost no flexible money left for anyone to respond with.

Removing this terminal period reduced the largest initial timing estimate from:

**+0.155 → +0.043 expected seats.**

A large part of the original result was therefore a mechanical end-of-cycle effect.

It was discarded.

# 10. Stress Test #2: Search More Races

The first analysis looked at only a small, hand-selected group of races.

That is dangerous.

Perhaps the apparent opportunity exists only because the best alternative was not included.

So the candidate pool was widened from roughly three races per side to approximately 15–20 races using several independent screening rules.

This produced several new candidates.

But it also changed the interpretation of the earlier results.

Several previously interesting candidates were shown to be dominated by better alternatives.

In other words:

*A race can look like a good reason to wait until you discover an even better race that is already worth funding.*

Several earlier timing stories disappeared at this stage.

# 11. Stress Test #3: Remove Unstable Districts

The expanded search uncovered another problem.

Some of the races generating large timing values were districts affected by redistricting and unstable baseline relationships.

Those districts can produce unusual-looking changes in modeled win probability that are not necessarily evidence of a genuine strategic timing mechanism.

After excluding the flagged districts, the 2024 Democratic timing estimate fell dramatically:

**+0.0796 → +0.0032.**

The earlier Democratic timing story essentially disappeared.

The Republican side also changed. AZ-09 emerged as the strongest remaining candidate.

This was important because AZ-09 had a different pattern. Its value did not depend on a sudden jump or an unstable district boundary. Its retention increased smoothly as the election approached.

# 12. Stress Test #4: Look Every Week

The previous analysis used a relatively coarse set of dates.

A smooth-looking curve at eight dates might hide a sudden jump between two observations.

So the surviving candidates were tested again using an 18-point weekly grid, covering roughly 120 days through one week before Election Day.

This test did two things.

First, it confirmed that AZ-09's trajectory remained smooth. Its estimated waiting value was approximately:

**+0.0492 expected seats.**

The estimate was essentially unchanged from the coarser analysis.

Second, the finer grid uncovered another 2022 candidate whose apparent value appeared only very late in the campaign. Its trajectory was flat for many weeks and then suddenly jumped.

That looked like the same mechanical end-of-cycle behavior that had already been rejected.

It was discarded.

# 13. What Survives?

After progressively widening the search and applying increasingly demanding tests, one candidate remained:

**AZ-09, 2024 Republican side.**

Its estimated value of waiting is approximately +0.049 expected seats.

More importantly, its trajectory has the characteristics we were looking for:

- the opportunity develops gradually;
- it does not depend on the final-week floor;
- it survives the wider candidate search;
- it survives the reliability screen;
- it remains smooth when examined weekly;
- and it beats the best immediate alternative available in the comparison.

That makes AZ-09 useful evidence that the proposed mechanism is real.

But there is an important qualification.

# 14. AZ-09 Is Not "The Optimal Strategy"

It would be tempting to write:

"The model discovers that Republicans should wait 30 days before spending in AZ-09."

That would be too strong.

The model searched across:

- two election cycles,
- both parties,
- multiple candidate pools,
- multiple dates,
- multiple screening rules,
- and several different stress tests.

If you search enough possibilities, something will eventually look unusually good simply by chance.

Therefore, the +0.049 estimate should not be interpreted as the typical return to waiting. Instead:

**AZ-09 is an existence proof.**

It demonstrates that the mechanism we theorized can actually occur. It does not tell us how frequently it occurs or how large its average effect is.

That distinction is critical.

# 15. What About 2022?

Perhaps the most interesting part of the result is what did not happen.

The 2022 election produced substantial strategic structure in the model. The equilibrium involved more portfolios and touched more races than the 2024 equilibrium.

Yet after the same corrections and stress tests, essentially no genuine mid-season timing opportunity survived in 2022.

This is important.

If timing were a universal advantage, we would expect to see it consistently. Instead, the model finds:

*Strategic opportunity and strategic timing are different things.*

A campaign can face meaningful strategic choices without having a meaningful reason to wait.

The absence of a 2022 timing result is therefore part of the finding, not a failed experiment.

# 16. The Answer to the Original Question

The project began with a simple question:

*Can one party identify spending strategies that the other party cannot simply optimize away?*

The answer has two parts.

**Static spending.** Generally, no. A unilateral optimizer can find several expected seats of opportunity when the opponent is held fixed. But once the opponent is allowed to respond, most of that advantage disappears. Neither party appears to possess a simple, unbeatable portfolio.

**Timing.** Here the answer is more interesting. Sometimes, potentially. As money becomes committed, the opponent's ability to respond decreases. This can turn an ordinary spending opportunity into a durable one. But durability alone is not enough. The future opportunity must become better than the best opportunity available today. Across the full search, only one case clearly survives that test: AZ-09 in 2024.

So the strongest conclusion is:

**Genuine strategic timing opportunities can exist, but they appear rare, conditional, and small.**

# 17. What Does This Say About Campaign Strategy?

The results suggest a picture that is somewhat different from the usual idea of an inefficient political market.

If campaigns repeatedly left enormous, easy-to-find opportunities on the table, we would expect optimization to uncover large systematic gains.

Instead, the pattern is:

1. unilateral optimization finds opportunities;
2. reciprocal optimization removes most of them;
3. actual spending resembles a simple competitiveness heuristic;
4. timing creates a second strategic channel;
5. increasingly aggressive tests eliminate almost all of the apparent timing opportunities;
6. one small, conditional opportunity remains.

That looks less like an untouched market inefficiency and more like an already-adapted strategic environment.

Campaign committees have fought these battles repeatedly. They raise money, watch their opponents, monitor races, learn from previous elections, and adjust their behavior.

A convention can therefore become self-reinforcing. If both sides expect serious spending to occur late, both sides preserve money for later. Because both sides preserve money, early spending remains easier to counter. And because early spending is easy to counter, waiting becomes rational.

*The convention helps create the environment that makes the convention rational.*

# 18. What This Project Does Not Claim

There are several things this analysis does not establish.

**It is not a causal estimate.** The model is built from historical spending, electoral outcomes, and estimated response relationships. It does not prove exactly how many votes a real dollar of campaign spending caused.

**It is not a prospective prediction.** The AZ-09 result was discovered retrospectively after a large search. A committee operating in real time would not know in advance that AZ-09 would survive every test.

**It is not a complete sequential allocation model.** The timing analysis studies individual $1 million deviations into individual races. It does not solve the full problem of allocating an entire flexible budget across hundreds of races over an entire campaign.

Those are important limitations.

They also define the natural next step.

# 19. What the Model Is Actually Useful For

The most promising future application is therefore not:

"Here is the perfect spending formula."

It is:

"Here is a framework for detecting when the strategic environment appears to be changing."

A future campaign could monitor questions such as:

- Is spending occurring earlier or later than historical norms?
- How much flexible money does each side still have?
- Which races are emerging as priorities?
- Where is marginal spending becoming unusually valuable?
- Is the opponent responding differently than in previous cycles?
- Is a race becoming strategically durable?
- And, most importantly, is that future opportunity actually better than the best alternative available today?

The model is better suited to answering these questions than to claiming that it has discovered a universal winning allocation.

# 20. The Broader Lesson

The most important result of this project may not be AZ-09.

It may be the fact that most of the exciting results disappeared.

The project began with several apparent opportunities. Each layer of realism made the conclusion harder to sustain:

passive opponent → responsive opponent

reversible money → committed money

small candidate pool → broad search

coarse dates → weekly dates

stable-looking races → reliability screening

durability → comparison against real alternatives

After all of that, only one small timing opportunity remained.

That is not a disappointing result.

It is exactly what we should expect from a strategic environment in which sophisticated organizations repeatedly compete against one another.

The lesson is not that campaign spending is perfectly efficient. It is that large, systematic, easily exploitable opportunities are difficult to sustain when the opponent is playing the same game.

The remaining opportunities are more likely to be conditional, temporary, and difficult to distinguish from noise or modeling artifacts.

In that sense, the project does not uncover a magic formula for campaign spending. It does something more modest — and potentially more useful:

*It shows how to distinguish an opportunity that merely looks good from one that remains good after the opponent responds, time passes, alternatives are considered, and the model is subjected to increasingly hostile tests.*

That is the central result.

# Reproducibility {-}

The complete technical project contains the underlying optimization routines, equilibrium searches, response-model validation, timing analysis, regression tests, data documentation, and intermediate results.

This reader-facing version intentionally omits most mathematical and implementation detail. The technical paper and codebase remain the appropriate sources for reproducing the numerical results.
