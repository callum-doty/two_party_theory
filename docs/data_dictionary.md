# Data dictionary

Field-level reference for the objects `src/game/*.py` and `scripts/*.py`
pass around. For data SOURCES (which FEC/Census/Cook files back which
field), see `data/catalog/data_catalog.md`.

## RaceRecord (`backtest.types.RaceRecord`, reused unchanged)

One row per `(i, t)` (spec Section 3), `t` fixed at the cycle's final
information date for the static project.

| Field | Meaning |
|---|---|
| `district_id` | e.g. `"NC-13"`. |
| `cook_rating` | Safe D / Likely D / Lean D / Toss-Up / Lean R / Likely R / Safe R. |
| `pvi` | Partisan Voting Index, signed (positive = more Democratic). |
| `incumb_status` | `"Incumbent"`, `"Challenger"`, or `"Open"`. |
| `generic_ballot` | National generic-ballot D-R margin at the information date. |
| `cvap` | Citizen voting-age population (turnout-model denominator). |
| `indiv_share` | Individual-donor share of total receipts. |
| `cand_d_total` | D candidate-committee disbursements ($) -- D's own spending FLOOR (not reallocatable). |
| `d_total` | D TOTAL spend ($): `cand_d_total` + D party/coordinated money. |
| `r_total` | R TOTAL spend ($): R's candidate-committee floor + R party/coordinated money. |
| `redistricting_flagged` | True if this district's lines changed since the prior cycle (excluded from "competitive-only" cuts). |

`cand_r_total` (R's OWN candidate-committee floor) is **not** a RaceRecord
field -- loaded separately via `build_cycle_state.load_cand_r_total()`, same
convention `scripts/solve_nash_equilibrium.py` established.

## Decision variables

| Symbol | Code | Meaning |
|---|---|---|
| `D_i` (party money only) | `party_d` | D's discretionary allocation, race `i`. Total D spend = `cand_d_total + party_d`. |
| `R_i` (party money only) | `party_r` | R's discretionary allocation, race `i`. Total R spend = `cand_r_total + party_r`. |
| `total_r` | `total_r` | Full R $ total (floor + party) -- what `game/payoff.py` actually conditions on. |

## Core game-theoretic quantities (spec Sections 4-14)

| Symbol | Code (`src/game/`) |
|---|---|
| `p_i(D,R)` | `payoff.p_win` |
| `U_D`, `U_R` | `payoff.expected_seats_d`, `payoff.expected_seats_r` |
| `MSG_i^D`, `MSG_i^R` | `gradients.msg_d`, `gradients.msg_r` |
| `BR_D`, `BR_R` | `best_response.br_d`, `best_response.br_r` |
| `(D*, R*)` | `equilibrium.solve_nash` |
| `lambda_D`, `lambda_R`, `S_i`, `Z_i` | `exploitability.race_level_surplus` |
| `Regret_D`, `Regret_R`, `E` | `exploitability.exploitability` |
| `PSV_i^D`, `PSV_i^R`, erosion, retention | `persistent_value.persistent_strategic_value_d` / `_r` |
