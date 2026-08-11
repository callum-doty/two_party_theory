#!/usr/bin/env python3
"""
Parse the Daily Kos Elections 118th Congress Members Guide CSV and extract
2016 and 2020 presidential results by congressional district (118th Congress
boundaries) into pres_2016.csv and pres_2020.csv.

Input:
    data/118th Congress Members Guide with Election Results and Demographic
    Data by District - House.csv

Output:
    data/raw/presidential/pres_2020.csv  (435 rows: district_id, d_votes, r_votes)
    data/raw/presidential/pres_2016.csv  (435 rows: district_id, d_votes, r_votes)

Column map (0-indexed, established by auditing the raw file):
    col 1  = district code  (e.g. "TX-07", "AK-AL")
    col 51 = Biden 2020 votes
    col 52 = Trump 2020 votes
    col 54 = Clinton 2016 votes
    col 55 = Trump 2016 votes

Row layout:
    row 0 = main column headers (used as default pandas header — discarded here)
    row 1 = sub-headers (candidate names etc.)
    row 2 = nationwide totals
    rows 3-437 = 435 congressional districts
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from backtest import config

_GUIDE_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "118th Congress Members Guide with Election Results and Demographic Data by District - House.csv"
)

_COL_CODE   = 1
_COL_BIDEN  = 51
_COL_TRUMP20 = 52
_COL_CLINTON = 54
_COL_TRUMP16 = 55


def _parse_int(val: str) -> int:
    return int(str(val).replace(",", "").strip())


def _normalize_code(code: str) -> str:
    """Convert DKE code to district_id: 'AK-AL' → 'AK-00', 'TX-07' → 'TX-07'."""
    state, dist = code.strip().split("-", 1)
    if dist.upper() == "AL":
        dist = "00"
    return f"{state}-{dist}"


def parse(dest: Path) -> None:
    if not _GUIDE_PATH.exists():
        print(f"ERROR: DKE guide not found at:\n  {_GUIDE_PATH}", file=sys.stderr)
        sys.exit(1)

    raw = pd.read_csv(_GUIDE_PATH, dtype=str, header=None)
    # rows 3+ are district rows (row 0 = headers, row 1 = sub-headers, row 2 = nationwide)
    districts = raw.iloc[3:].reset_index(drop=True)

    rows_2020: list[dict] = []
    rows_2016: list[dict] = []

    for _, row in districts.iterrows():
        code = str(row[_COL_CODE]).strip()
        if not code or code.upper() in ("NAN", ""):
            continue
        try:
            district_id = _normalize_code(code)
            d20 = _parse_int(row[_COL_BIDEN])
            r20 = _parse_int(row[_COL_TRUMP20])
            d16 = _parse_int(row[_COL_CLINTON])
            r16 = _parse_int(row[_COL_TRUMP16])
        except (ValueError, AttributeError) as e:
            print(f"  WARNING: skipping {code!r}: {e}")
            continue

        rows_2020.append({"district_id": district_id, "d_votes": d20, "r_votes": r20})
        rows_2016.append({"district_id": district_id, "d_votes": d16, "r_votes": r16})

    for year, rows in [(2020, rows_2020), (2016, rows_2016)]:
        df = pd.DataFrame(rows).sort_values("district_id").reset_index(drop=True)
        out = dest / f"pres_{year}.csv"
        df.to_csv(out, index=False)
        d_total = df["d_votes"].sum()
        r_total = df["r_votes"].sum()
        national_d2 = d_total / (d_total + r_total)
        print(
            f"pres_{year}.csv: {len(df)} districts, "
            f"D 2-party share {national_d2:.3%}  "
            f"(D {d_total:,} / R {r_total:,})  → {out}"
        )


def main() -> None:
    dest = config.raw_path("presidential")
    dest.mkdir(parents=True, exist_ok=True)
    parse(dest)


if __name__ == "__main__":
    main()
