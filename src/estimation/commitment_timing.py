"""
Real, dated commitment fractions -- how much of a committee's eventual
national-committee-own IE spending was ALREADY locked in as of a given
date, for the two-player game's response-delay analysis (2026-08-13
research-discussion follow-up: "the opportunity doesn't survive a fully
informed instantaneous response, but does survive long enough to matter
because response is not instantaneous").

Source: backtest.data.fec.load_ie_transactions_dated(cycle,
national_committee_only=True) -- the DATED subset of DCCC's/NRCC's OWN IE
spending (party_natl / x_D / x_R, this project's actual game-theoretic
decision variables, not the broader outside-group IE universe).

CALIBRATION CHECK, done before trusting this for anything: the dated
subset's own total should approximate the full, control-floor-derived
budget (build_cycle_state.py's budget_d/budget_r, which also includes
transactions with a blank exp_date -- see load_ie_transactions_dated's
docstring). Checked directly: 2022 D $93.25M dated vs. $93.27M full (99.97%
captured), 2022 R $83.21M vs. $83.21M (exact), 2024 R $45.48M vs. $45.48M
(exact) -- but 2024 D is only $70.84M dated vs. $98.48M full (72% captured,
a real 28% gap, not a rounding difference). Because that gap is uneven
across cycle/party rather than a small, ignorable residual, this module
does NOT use the dated subset's raw cumulative dollars directly. Instead it
computes a FRACTION curve (cumulative dated $ / that party's own total
dated $, which reaches exactly 1.0 at the last transaction date by
construction) and applies that fraction to the TRUE full observed budget
(party_obs, passed in by the caller from build_cycle_state.py) --
correct regardless of how much of the total the dated subset captures,
under the stated assumption that undated transactions are paced similarly
in time to dated ones. That assumption is least trustworthy for 2024 D
specifically (28% of the total is invisible to the curve's shape); flagged
here rather than silently treated as equally reliable across all four
cycle/party combinations.

SECOND SIMPLIFYING ASSUMPTION, PARTIALLY RELAXED 2026-08-13: the functions
above use one aggregate fraction curve per (cycle, party), applied
uniformly to every race's own party_obs, rather than a separate per-race
date curve -- necessary because most funded races have only a handful of
dated national-committee transactions each, and a per-race curve would be
dominated by small-sample noise (the same sparse-data problem this
project's eta reaction estimation pools across Cook-rating tiers to avoid,
here pooled across races instead of districts). A race with party_obs=0
correctly gets committed capital = 0 at every date regardless.

PARTIAL POOLING BY COMPETITIVENESS TIER (functions below, suffixed
`_tiered`): checked the district/transaction counts by full 7-way Cook
rating before building this -- most non-competitive ratings have only 1-4
funded districts and a few dozen transactions each, too thin to trust
individually. Collapsing to three tiers (competitive = Toss-Up; lean = Lean
D/Lean R; safe_likely = Safe D/Likely D/Safe R/Likely R) gives workable
counts: 10-22 districts / 200-450 transactions in competitive and lean,
3-5 districts / 29-104 transactions in safe_likely (thin but usable pooled,
where it was unusable split 4 ways). This is not just noise reduction for
its own sake -- there is a real timing gradient to capture: in 3 of 4
cycle/party combinations checked, competitive races are funded measurably
earlier than lean or safe/likely ones (2022 R: competitive median txn date
Oct 11 -> lean Oct 18 -> safe_likely Oct 25, a 2-week spread). The one
exception (2024 R, safe_likely funded earliest) is built on only 3
districts/35 transactions and is treated as noise, not a real reversal.
Still one curve per (cycle, party, TIER), not per race -- the tier is the
unit of pooling, chosen to be the coarsest split that still shows a real
effect, not the finest split the data could nominally support.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from backtest.data.fec import load_ie_transactions_dated

TIER_MAP = {
    "Toss-Up": "competitive",
    "Lean D": "lean", "Lean R": "lean",
    "Safe D": "safe_likely", "Likely D": "safe_likely",
    "Safe R": "safe_likely", "Likely R": "safe_likely",
}
TIERS = ("competitive", "lean", "safe_likely")


def race_tier(cook_rating: str) -> str:
    """Map a RaceRecord's cook_rating to one of TIERS. Unrecognized ratings
    fall back to safe_likely (the pool least sensitive to a handful of
    misbucketed districts, since it already spans four Cook categories)."""
    return TIER_MAP.get(cook_rating, "safe_likely")


def commitment_fraction_curve(cycle: int, party: str) -> pd.DataFrame:
    """Cumulative national-committee-own IE spend as a fraction of that
    party's own dated-transaction total, at every distinct transaction
    date (monotonic 0 -> 1). Columns: exp_date, fraction_committed."""
    txns = load_ie_transactions_dated(cycle, national_committee_only=True)
    txns = txns[txns["party"] == party].sort_values("exp_date")
    total = float(txns["amount"].sum())
    if total <= 0:
        raise ValueError(f"No dated national-committee-own IE spend found for {cycle} {party}")
    cum = txns.groupby("exp_date")["amount"].sum().cumsum()
    frac = (cum / total).rename("fraction_committed")
    return frac.reset_index()


def commitment_fraction_as_of(cycle: int, party: str, as_of_date: date,
                               curve: pd.DataFrame | None = None) -> float:
    """Fraction of the party's dated national-committee IE total committed
    by as_of_date: 0.0 before the first transaction, 1.0 on/after the last."""
    if curve is None:
        curve = commitment_fraction_curve(cycle, party)
    as_of_ts = pd.Timestamp(as_of_date)
    prior = curve[curve["exp_date"] <= as_of_ts]
    if prior.empty:
        return 0.0
    return float(prior["fraction_committed"].iloc[-1])


def committed_capital_per_race(cycle: int, party: str, as_of_date: date,
                                party_obs: np.ndarray,
                                curve: pd.DataFrame | None = None) -> np.ndarray:
    """L_t per race: party_obs (the TRUE full-cycle observed allocation --
    build_cycle_state.py's party_d_obs/party_r_obs) scaled by the aggregate
    commitment fraction as of as_of_date. See module docstring for why the
    fraction (not the dated subset's raw dollars) is what gets applied, and
    why it is one aggregate curve rather than a per-race one."""
    frac = commitment_fraction_as_of(cycle, party, as_of_date, curve)
    return frac * np.asarray(party_obs, dtype=float)


def commitment_fraction_curve_tiered(cycle: int, party: str, tier: str, races) -> pd.DataFrame:
    """Same as commitment_fraction_curve, restricted to districts whose
    Cook rating maps to `tier` (see race_tier/TIER_MAP). `races` is the
    RaceRecord list for this cycle (build_cycle_state.py's `races`) --
    passed in rather than refetched, since every caller already has it."""
    cook = {r.district_id: r.cook_rating for r in races}
    txns = load_ie_transactions_dated(cycle, national_committee_only=True)
    txns = txns[txns["party"] == party].copy()
    txns["tier"] = txns["district_id"].map(lambda d: race_tier(cook.get(d, "")))
    txns = txns[txns["tier"] == tier].sort_values("exp_date")
    total = float(txns["amount"].sum())
    if total <= 0:
        raise ValueError(f"No dated national-committee-own IE spend found for {cycle} {party} tier={tier}")
    cum = txns.groupby("exp_date")["amount"].sum().cumsum()
    frac = (cum / total).rename("fraction_committed")
    return frac.reset_index()


def build_tiered_curves(cycle: int, party: str, races) -> dict[str, pd.DataFrame]:
    """{tier: commitment_fraction_curve_tiered(...)} for all of TIERS --
    compute ONCE per (cycle, party) and reuse across every race/date in a
    sweep, the same reuse pattern the rest of this module and
    persistent_value.py's isolated baseline already rely on."""
    return {tier: commitment_fraction_curve_tiered(cycle, party, tier, races) for tier in TIERS}


def committed_capital_per_race_tiered(cycle: int, party: str, as_of_date: date,
                                       races, party_obs: np.ndarray,
                                       curves: dict[str, pd.DataFrame] | None = None) -> np.ndarray:
    """L_t per race using the TIER-APPROPRIATE commitment fraction for each
    race's own Cook rating, instead of one blended fraction for every race
    (committed_capital_per_race above). curves: build_tiered_curves(...)'s
    output, computed once by the caller and reused across a sweep -- each
    call here would otherwise reload and refilter the raw transaction file
    three times (once per tier)."""
    if curves is None:
        curves = build_tiered_curves(cycle, party, races)
    fracs = np.array([
        commitment_fraction_as_of(cycle, party, as_of_date, curve=curves[race_tier(r.cook_rating)])
        for r in races
    ])
    return fracs * np.asarray(party_obs, dtype=float)
