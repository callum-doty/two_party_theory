"""
PVI on the district lines each House election actually used.

    PVI_i = mean over the two most recent presidential elections BEFORE the
            House election of (district D two-party share - national share)

from data/processed/pres_by_lines.csv (scripts/build_pres_by_lines.py, The
Downballot's calculations). backtest/data/pvi.py instead computes every
cycle from 2016/2020 results on 118th-Congress lines, which is the wrong
map for 2012-2020; the old pipeline is left as it was.

Older presidential years are only partly back-computed onto later maps
(e.g. 2016 results exist on 2022 lines but not 2024 lines). A missing year
is borrowed from the previous map only for districts whose results in a
year both maps cover are identical, i.e. districts that were not redrawn.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

PRES_BY_LINES = Path(__file__).resolve().parents[2] / "data" / "processed" / "pres_by_lines.csv"

# Certified national two-party D share (D / (D + R)). backtest/data/pvi.py's
# NATIONAL_D2PARTY has 2008 = 0.5382 and 2024 = 0.4978; the certified totals
# give 0.5369 (69.50M / 129.45M) and 0.4925 (75.02M / 152.32M).
NATIONAL_D2PARTY = {2008: 0.5369, 2012: 0.5196, 2016: 0.5111, 2020: 0.5227, 2024: 0.4925}
# The sheets round to 0.1 point and differ slightly in method between
# releases, so unchanged districts differ by up to ~0.005; at this tolerance
# the 2022->2024 "redrawn" set is exactly the 2024 redraws (NC, NY, GA, AL, LA).
UNCHANGED_TOL = 0.005

# (election cycle, state) -> the lines_cycle whose map that state actually
# uses. The Downballot's 2026 sheet (July 2026) has Missouri on its 2025 map;
# the U.S. Supreme Court ruled on 2026-09-25 that the 2022 map governs 2026.
MAP_OVERRIDES = {(2026, "MO"): 2024}


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    return pd.read_csv(PRES_BY_LINES, dtype={"district_id": str})


def _shares(lines_cycle: int) -> pd.DataFrame:
    """district x pres_year D shares on one map, with missing years borrowed
    from earlier maps for districts that were not redrawn."""
    t = _table()
    cycles = sorted(c for c in t["lines_cycle"].unique() if c <= lines_cycle)
    if lines_cycle not in cycles:
        raise ValueError(f"no presidential-by-district data for {lines_cycle} lines")
    wide = t[t["lines_cycle"] == lines_cycle].pivot(index="district_id", columns="pres_year", values="d_share")
    for prev in reversed(cycles[:-1]):
        old = t[t["lines_cycle"] == prev].pivot(index="district_id", columns="pres_year", values="d_share")
        common = [y for y in wide.columns if y in old.columns]
        if not common:
            continue
        diff = (wide[common] - old.reindex(wide.index)[common]).abs()
        unchanged = (diff.max(axis=1) < UNCHANGED_TOL) & diff.notna().any(axis=1)
        for y in old.columns:
            fill = old.reindex(wide.index)[y].where(unchanged)
            wide[y] = wide[y].combine_first(fill) if y in wide.columns else fill
    return wide


def load_pvi(cycle: int) -> pd.Series:
    """district_id -> PVI (percentage points, D-positive) for the House
    election in `cycle`. Districts with no prior presidential result on
    their lines are omitted."""
    wide = _shares(cycle)
    for (c, state), lines in MAP_OVERRIDES.items():
        if c == cycle:
            other = _shares(lines)
            wide = pd.concat([wide[~wide.index.str.startswith(f"{state}-")],
                              other[other.index.str.startswith(f"{state}-")]])
    prior = sorted((y for y in wide.columns if y < cycle), reverse=True)[:2]
    lean = pd.DataFrame({y: (wide[y] - NATIONAL_D2PARTY[y]) * 100 for y in prior})
    return lean.mean(axis=1, skipna=True).dropna().rename("pvi")
