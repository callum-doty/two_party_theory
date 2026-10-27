#!/usr/bin/env python3
"""
Parse The Downballot's presidential-results-by-district sheets (one per set
of district lines, CSV exports in data/raw/presidential/by_lines/) plus the
repo's existing 118th-Congress files into one tidy table:

    data/processed/pres_by_lines.csv
        lines_cycle, pres_year, district_id, d_share   (D two-party share)

lines_cycle is the House election whose district lines the row describes.
Source: https://www.the-downballot.com/p/the-downballots-calculations-of-presidential

Usage:
    python scripts/build_pres_by_lines.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
RAW = REPO_ROOT / "data" / "raw" / "presidential"
OUT = REPO_ROOT / "data" / "processed" / "pres_by_lines.csv"

# raw file -> House cycles that used those lines
SHEETS = {
    "lines2012_pres2008_2012.csv": (2012, 2014),
    "lines2016_pres2008_2016.csv": (2016,),
    "lines2018_pres2008_2016.csv": (2018,),
    "lines2020_pres2008_2020.csv": (2020,),
    "lines2024_pres2020_2024.csv": (2024,),
    "lines2026_pres2024.csv": (2026,),
}
YEAR = re.compile(r"^(20\d\d)$")
DISTRICT = re.compile(r"^([A-Z]{2})-(\d{1,2}|AL)$")


def _district_id(raw: str) -> str | None:
    m = DISTRICT.match(raw.strip())
    if not m:
        return None
    state, num = m.groups()
    return f"{state}-{'00' if num == 'AL' else int(num):0>2}"


def parse_sheet(path: Path) -> pd.DataFrame:
    """Each sheet has a row of election years over (D, R[, margin]) column
    groups, then one row per district. The D column sits under the year
    label and R immediately to its right."""
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    year_row = next(i for i, r in enumerate(rows) if sum(bool(YEAR.match(c.strip())) for c in r) >= 1
                    and any(c.strip() in ("CD", "District") for c in rows[i] + rows[i + 1]))
    year_cols = {int(c.strip()): j for j, c in enumerate(rows[year_row]) if YEAR.match(c.strip())}
    out = []
    for r in rows[year_row + 1:]:
        did = _district_id(r[0]) if r else None
        if not did:
            continue
        for year, j in year_cols.items():
            try:
                d, rep = float(r[j]), float(r[j + 1])
            except (ValueError, IndexError):
                continue
            out.append(dict(pres_year=year, district_id=did, d_share=d / (d + rep)))
    return pd.DataFrame(out)


def main() -> None:
    frames = []
    for name, cycles in SHEETS.items():
        df = parse_sheet(RAW / "by_lines" / name)
        for c in cycles:
            frames.append(df.assign(lines_cycle=c))
        print(f"{name}: {df.district_id.nunique()} districts, years {sorted(df.pres_year.unique())}")
    # 2022 lines: the repo's existing 118th-Congress vote files
    for year in (2016, 2020):
        p = pd.read_csv(RAW / f"pres_{year}.csv", dtype={"district_id": str})
        frames.append(pd.DataFrame(dict(lines_cycle=2022, pres_year=year, district_id=p.district_id,
                                        d_share=p.d_votes / (p.d_votes + p.r_votes))))
    out = pd.concat(frames, ignore_index=True)[["lines_cycle", "pres_year", "district_id", "d_share"]]
    if out.duplicated(["lines_cycle", "pres_year", "district_id"]).any():
        sys.exit("duplicate district rows")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(["lines_cycle", "pres_year", "district_id"]).to_csv(OUT, index=False, float_format="%.5f")
    print(f"{len(out)} rows -> {OUT.relative_to(REPO_ROOT)}")
    # older presidential years are only partly back-computed onto later maps
    print("districts covered (lines_cycle x pres_year):")
    print(out.groupby(["lines_cycle", "pres_year"]).size().unstack().fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
