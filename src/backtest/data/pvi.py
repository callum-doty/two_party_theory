"""
Partisan Voting Index — computed from presidential results by congressional district.

Data contract
─────────────
Place presidential result CSVs under data/raw/presidential/:

  pres_{year}.csv
    Columns: district_id (str), d_votes (int), r_votes (int)
    One row per congressional district, general election only, 2-party votes.

Cook's formula (public):
  PVI_i = avg(D_2party_i over 2 elections) − avg(D_national over 2 elections)
  Expressed in percentage points: positive = D-lean, negative = R-lean.
  Result is continuous float; "D+3" ≈ +3.0, "R+8" ≈ -8.0.

Data sources (free, no API key):
  118th Congress districts (2022–2024):
    Daily Kos Elections — 2016 and 2020 presidential results by 118th Congress map.
    Export the Google Sheet or download from their website.
    → data/raw/presidential/pres_2016.csv
    → data/raw/presidential/pres_2020.csv

  Historical congresses (pre-2021 districts):
    Daily Kos Elections archives (previous Congress editions of same dataset).
    Required only for historical panel β_RC estimation (2012–2022 cycles).
    Cycles 2022 and 2024 both use the post-2021 redistricting map.
"""

from __future__ import annotations
import pandas as pd
from .. import config


# Known national 2-party Democratic share from certified final results.
# Used instead of data-derived national shares so that partial state coverage
# (e.g. 25/50 states for 2016) does not bias the national baseline.
# Source: FEC certified results.
NATIONAL_D2PARTY: dict[int, float] = {
    2000: 0.4979,  # Gore
    2004: 0.4886,  # Kerry
    2008: 0.5382,  # Obama
    2012: 0.5195,  # Obama
    2016: 0.5111,  # Clinton  (65,853,514 / 128,838,342)
    2020: 0.5226,  # Biden    (81,268,924 / 155,485,078)
    2024: 0.4978,  # Harris
}


# Which two presidential elections to use for PVI per House cycle.
#
# Ideal mapping (correct district boundaries per cycle):
#   2012: (2004, 2008) on 113th Congress boundaries
#   2014: (2008, 2012) on 114th Congress boundaries
#   2016: (2008, 2012) on 115th Congress boundaries
#   2018: (2012, 2016) on 115th Congress boundaries
#   2020: (2012, 2016) on 116th Congress boundaries
#   2022: (2016, 2020) on 118th Congress boundaries  ← only cycle fully correct
#   2024: (2016, 2020) on 118th Congress boundaries  ← correct
#
# We only have presidential results allocated to 118th Congress (post-2021)
# boundaries (pres_2016.csv and pres_2020.csv from DKE).  For panel cycles
# 2012–2020 we use (2016, 2020) as a consistent proxy.  This adds noise to
# the margin model R² but does NOT affect β_RC (which differences away
# district-level lean entirely).  PVI for 2022 and 2024 is exact.
CYCLE_TO_PRES_YEARS: dict[int, tuple[int, int]] = {
    2012: (2016, 2020),  # proxy — correct years would be (2004, 2008) on 113th map
    2014: (2016, 2020),  # proxy — correct years would be (2008, 2012) on 114th map
    2016: (2016, 2020),  # proxy — correct years would be (2008, 2012) on 115th map
    2018: (2016, 2020),  # proxy — correct years would be (2012, 2016) on 115th map
    2020: (2016, 2020),  # proxy — correct years would be (2012, 2016) on 116th map
    2022: (2016, 2020),  # exact — 118th Congress boundaries
    2024: (2016, 2020),  # exact — 118th Congress boundaries
    2026: (2016, 2020),  # proxy — no pres_2024.csv in this repo yet (only 2016/2020
                         # exist under data/raw/presidential/), so this reuses the
                         # same pair as 2022/2024 rather than the more standard
                         # (2020, 2024) mapping. Also note: several states have
                         # mid-decade 2026 redistricting in progress (e.g. Texas) —
                         # this mapping additionally assumes unchanged district
                         # lines, which will not hold everywhere. Upgrade to
                         # (2020, 2024) once district-level 2024 presidential
                         # results are added to this repo.
}


def load_presidential(year: int) -> pd.DataFrame:
    """
    Load presidential results for a given election year.
    Returns DataFrame: district_id, d_votes (int), r_votes (int).
    """
    path = config.raw_path("presidential") / f"pres_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Presidential results file not found: {path}\n"
            "Download 2-party presidential results by congressional district from:\n"
            "  Daily Kos Elections — presidential results by congressional district\n"
            "Save as data/raw/presidential/pres_{year}.csv with columns:\n"
            "  district_id, d_votes, r_votes"
        )
    df = pd.read_csv(path, dtype={"district_id": str})
    df["d_votes"] = pd.to_numeric(df["d_votes"], errors="coerce").fillna(0)
    df["r_votes"] = pd.to_numeric(df["r_votes"], errors="coerce").fillna(0)
    return df[["district_id", "d_votes", "r_votes"]]


def _d2party_share(df: pd.DataFrame) -> pd.Series:
    """2-party Democratic share per district (indexed by district_id)."""
    total = df["d_votes"] + df["r_votes"]
    # Uncontested districts: treat as 0.5 so they don't bias PVI
    return (df["d_votes"] / total.where(total > 0)).fillna(0.5)


def compute_pvi(year1: int, year2: int) -> pd.DataFrame:
    """
    Compute PVI (percentage points, D-positive) from two presidential elections.

    PVI_i = avg over available elections of (district_D2party − national_D2party).

    National D-share comes from NATIONAL_D2PARTY (certified results), not from
    the data files.  This ensures correct baselines even when data is partial
    (e.g. only 25/50 states have year1 results).

    For districts present in both elections: average of both lean values.
    For districts present in only one election: single-election lean value.
    Returns DataFrame: district_id, pvi (float).
    """
    n1 = NATIONAL_D2PARTY.get(year1)
    n2 = NATIONAL_D2PARTY.get(year2)
    if n1 is None or n2 is None:
        missing = [y for y, n in [(year1, n1), (year2, n2)] if n is None]
        raise ValueError(
            f"No known national D2party share for year(s) {missing}. "
            "Add entries to NATIONAL_D2PARTY in data/pvi.py."
        )

    p1 = load_presidential(year1).set_index("district_id")
    p2 = load_presidential(year2).set_index("district_id")

    s1 = _d2party_share(p1)   # district D2party share for year1
    s2 = _d2party_share(p2)   # district D2party share for year2

    lean1 = (s1 - n1) * 100   # district lean relative to national for year1
    lean2 = (s2 - n2) * 100   # district lean relative to national for year2

    # Union index; NaN where a district lacks data for that election
    all_ids = lean1.index.union(lean2.index)
    lean1 = lean1.reindex(all_ids)
    lean2 = lean2.reindex(all_ids)

    # Per-district average over available elections
    n_available = lean1.notna().astype(int) + lean2.notna().astype(int)
    pvi = lean1.fillna(0).add(lean2.fillna(0)).div(n_available)

    return pvi.rename("pvi").reset_index()


def load_pvi(cycle: int) -> pd.DataFrame:
    """
    Return computed PVI per district for the given House election cycle.
    Uses the two most recent presidential elections preceding the cycle.
    Returns DataFrame: district_id, pvi (float, percentage points, D-positive).
    """
    if cycle not in CYCLE_TO_PRES_YEARS:
        raise ValueError(
            f"No presidential year mapping defined for cycle {cycle}. "
            f"Add an entry to CYCLE_TO_PRES_YEARS in data/pvi.py."
        )
    year1, year2 = CYCLE_TO_PRES_YEARS[cycle]
    return compute_pvi(year1, year2)


def derive_rating(pvi: float, incumb_status: str) -> str:
    """
    Derive an approximate Cook-style competitive rating from PVI + incumbency.

    Applies a ±2-point incumbency adjustment before thresholding:
      D incumbent → +2 (district effectively safer for D)
      R incumbent → −2 (district effectively tougher for D)
      Open seat   →  0 (no adjustment)

    Thresholds (on effective PVI after adjustment):
      ≥ +10 → Safe D
      +5 to +10 → Likely D
      +1 to  +5 → Lean D
      −3 to  +1 → Toss-Up
      −5 to  −3 → Lean R
      −10 to −5 → Likely R
      ≤ −10 → Safe R
    """
    bonus = {"Incumbent": 2.0, "Challenger": -2.0, "Open": 0.0}
    eff = pvi + bonus.get(str(incumb_status), 0.0)

    if eff >= 10:   return "Safe D"
    if eff >= 5:    return "Likely D"
    if eff >= 1:    return "Lean D"
    if eff >= -3:   return "Toss-Up"
    if eff >= -5:   return "Lean R"
    if eff >= -10:  return "Likely R"
    return "Safe R"
