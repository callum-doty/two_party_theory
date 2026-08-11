# Data catalog

project_spec.md Section 17: distinguishes data by TYPE (not by source file
layout) so it's clear what each series can and can't answer. All sources are
public. Reused unchanged from the old project (`Political Portfolio`) --
this project adds no new data universe.

| Category (spec Section 17) | Source(s) in `data/raw/` | Notes |
|---|---|---|
| Candidate spending | `fec/`, `candidate_master/`, `candidate_committee_linkage/` | Per-candidate committee disbursements (D and R), via `backtest.data.fec.load_candidate_disbursements`. This project's `cand_r_total` (R's own floor) is filtered from the same source as D's `cand_d_total`. |
| Party committee spending | `fec/`, `committee_master/` | DCCC/NRCC coordinated spending (FEC API, per-committee) PLUS state party committees' own 24K coordinated expenditures (`scripts/fetch_data.py::parse_state_party_coordinated_24k`, scanned from `all_committee_transactions/itoth*.txt`) -- feeds `RaceRecord.d_total` / `r_total`. Symmetric as of 2026-08-11: `coordinated_state_party_{dem,rep}_{cycle}.csv` both exist for 2022 and 2024 (see catalog note below; was D-only through the old project). |
| Independent expenditures | `independent_expenditure/`, `schedule_b-2026-07-07T15_54_35.csv` | FEC Schedule E-derived IE data. |
| Democratic-aligned outside spending | `fec/`, `pac_summary/` | Non-party D-aligned spending, where identifiable from FEC filer data. |
| Republican-aligned outside spending | `fec/`, `pac_summary/` | Mirror of the above for R-aligned spenders. |
| Race fundamentals | `cook_pvi/`, `census/`, `generic_ballot/`, `generic_ballot_averages.csv` | PVI, CVAP, generic-ballot series feeding `mu_const`/`sigma` in the margin model. |
| District boundaries | `census/` | Used for CVAP/demographic joins, not spatial analysis in this project. |
| Candidate identity/status | `candidate_master/`, `house_senate_current_campaigns/`, `mit_elections/` | Incumbency status, open-seat flags, ballot membership. |
| Competitiveness ratings | `cook_pvi/` | Cook rating tiers used throughout (Safe/Likely/Lean/Toss-Up). |
| Election outcomes | `mit_elections/`, `bulk_all/` | Historical results for out-of-sample validation (Level A). |
| Bulk FEC universe | `bulk_all/`, `all_committee_transactions/` | Multi-GB raw bulk files; NOT committed to git (see `.gitignore`) -- re-fetch via `scripts/fetch_data.py`. |
| Presidential-year context | `presidential/`, `rcp/` | Presidential-race polling/results context variables. |

## Processed artifacts (`data/processed/`)

Estimation outputs already fit against the sources above, reused unchanged:
`margin_model_coef.json` (D-side elasticity `beta_D`), `sigma_model.json`
(race-level uncertainty), `beta_rc.json`/`beta_rc_bootstrap.json`
(response-curve elasticity + bootstrap), `open_seat_calibration.json`,
`eta_uncertainty.json` (old project's reactive-response estimate -- kept as
descriptive evidence only, per spec Section 18, not used as this project's
definition of strategic response), `candidate_spend_trickle.json`,
`gb_dynamics.json`, `dccc_forecast_model.json`.

**No R-side elasticity artifact exists yet** -- `game/gradients.py`'s MSG^R
and `backtest/optimizer/nash.py`'s R best response both use D's own
calibrated ceiling/elasticity, mirrored, as a stated assumption (spec
Section 19's symmetry test is not yet run -- see
`scripts/estimate_response_model.py`).

## State-party coordinated spending gap (closed 2026-08-11)

The old project's state-party 24K coordinated-expenditure scan
(`FINDINGS.md` Section 10.7, Gap 3) only ever covered Democratic state
parties -- `identify_state_dem_party_committees()` had no Republican
counterpart, so `r_total`/`cand_r_total` were a documented, if small,
undercount relative to D's side. Closed here by adding
`identify_state_rep_party_committees()` (full 50-state coverage, verified
against `committee_master`) and generalizing
`parse_state_party_coordinated_24k(cycle, party=...)`. Actual dollar
impact, both cycles now scanned:

| Cycle | D state-party coordinated | R state-party coordinated |
|---|---|---|
| 2022 | $116,973 (37 districts) | $131,425 (30 districts) |
| 2024 | $153,106 (39 districts) | $153,650 (25 districts) |

Small relative to total budgets (NRCC 2024 budget moved $131.95M ->
$132.11M) -- not a driver of the project's headline exploitability
numbers, but a real, now-closed asymmetry rather than an assumption. See
`docs/methodology.md` for the verification process (all 7 manually-added
Republican state committees cross-checked directly against
`committee_master`, same standard the original Dem-side manual list used).

## Live data (`data/live/`)

`house_district_polls*.csv/json`, `generic_ballot_*.csv/json`,
`msg_live.csv`, `spending_live.json` -- feeds the secondary 2026 live
analysis (spec Section 16) only, via `scripts/run_live_2026.py`.
