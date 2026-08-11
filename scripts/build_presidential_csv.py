#!/usr/bin/env python3
"""
Consolidate per-state presidential result CSVs into a single pres_{year}.csv.

Input: data/raw/presidential/{year}/*.csv
  Each file is a state-level spreadsheet with county-level rows and district
  total rows (e.g. "1 Total", "AL Total") and a "Grand Total" footer.
  The "{N} Total" rows are used — county rows are discarded.

Output: data/raw/presidential/pres_{year}.csv
  Columns: district_id (e.g. "TX-01"), d_votes (int), r_votes (int)

Note on district boundaries
────────────────────────────
The 2020 files in data/raw/presidential/2020/ use 116th Congress district
boundaries (the districts in effect for the 2020 election, before the 2021
redistricting). For the 118th Congress PVI computation (2022–2024 cycles),
these are an approximation. States with major 2021 redistricting (TX, NC, OH,
FL, GA) may have PVI errors of several points in redrawn districts.

For maximum accuracy: replace pres_2020.csv with the Daily Kos Elections
version reallocated to 118th Congress boundaries after this script runs.

Usage
─────
    python scripts/build_presidential_csv.py --year 2020
    python scripts/build_presidential_csv.py --year 2016
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from backtest import config

STATE_ABBR: dict[str, str] = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX",
    "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
}


def _state_from_filename(stem: str) -> str:
    """
    Extract state name from a file stem like:
      '2020 Montana pres-by-CD - Summary'
      'alaska 2020 pres-by-CD - Summary'
      'North Carolina 2020 pres-by-2020-CD - Summary'
    """
    # Drop everything from " pres" onward
    name_part = re.split(r"\s+pres", stem, flags=re.IGNORECASE)[0]
    # Drop any 4-digit year token
    name_part = re.sub(r"\b\d{4}\b", "", name_part).strip()
    return name_part.strip()


def _parse_district(cd: str) -> str | None:
    """
    Convert a CD 'Total' label to a zero-padded district number.
      '1 Total'  → '01'
      '27 Total' → '27'
      'AL Total' → '00'   (at-large)
      'Grand Total' → None (skip)
    """
    cd = str(cd).strip()
    if "grand" in cd.lower():
        return None
    num = cd.replace("Total", "").strip()
    if num.upper() == "AL":
        return "00"
    try:
        return str(int(num)).zfill(2)
    except ValueError:
        return None


def _parse_votes(val: str) -> int:
    return int(str(val).replace(",", "").strip())


def consolidate(year: int) -> None:
    import pandas as pd

    state_dir = config.raw_path("presidential") / str(year)
    if not state_dir.exists():
        print(f"ERROR: Directory not found: {state_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(state_dir.glob("*.csv"))
    if not files:
        print(f"ERROR: No CSV files found in {state_dir}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    missing_states: list[str] = []

    for f in files:
        df = pd.read_csv(f, dtype=str)

        # Identify state abbreviation from filename
        raw_name = _state_from_filename(f.stem).upper()
        state_abbr = STATE_ABBR.get(raw_name)
        if state_abbr is None:
            missing_states.append(f"{f.name!r} → {raw_name!r}")
            continue

        # Detect D-candidate column: 2020 = "Biden", 2016 = "Clinton", etc.
        d_col = next((c for c in df.columns if c in ("Biden", "Clinton", "Obama", "Kerry", "Gore")), None)
        if d_col is None:
            print(f"  WARNING: No recognized D-candidate column in {f.name}; columns={list(df.columns)}")
            continue

        if "CD" not in df.columns:
            # No CD column: at-large state (whole state = one district).
            # Use the Grand Total row as district "00".
            grand = df[df.iloc[:, 0].astype(str).str.upper().str.contains("GRAND TOTAL", na=False)]
            if grand.empty:
                # Fall back to the last row
                grand = df.iloc[[-1]]
            try:
                d_votes = _parse_votes(grand[d_col].iloc[0])
                r_votes = _parse_votes(grand["Trump"].iloc[0])
            except (ValueError, KeyError) as e:
                print(f"  WARNING: Could not parse votes in {f.name}: {e}")
                continue
            rows.append({"district_id": f"{state_abbr}-00", "d_votes": d_votes, "r_votes": r_votes})
            continue

        # Normal case: CD column present, use "{N} Total" rows
        total_rows = df[df["CD"].notna() & df["CD"].str.contains("Total", na=False)]
        for _, row in total_rows.iterrows():
            dist = _parse_district(row["CD"])
            if dist is None:
                continue
            try:
                d_votes = _parse_votes(row[d_col])
                r_votes = _parse_votes(row["Trump"])
            except (ValueError, KeyError) as e:
                print(f"  WARNING: Could not parse votes in {f.name} row {row['CD']!r}: {e}")
                continue
            rows.append({"district_id": f"{state_abbr}-{dist}", "d_votes": d_votes, "r_votes": r_votes})

    if missing_states:
        print(f"WARNING: Could not map state for {len(missing_states)} file(s):")
        for m in missing_states:
            print(f"  {m}")

    out = pd.DataFrame(rows).sort_values("district_id").reset_index(drop=True)
    out_path = config.raw_path("presidential") / f"pres_{year}.csv"
    out.to_csv(out_path, index=False)
    n_states = out["district_id"].str[:2].nunique()
    print(f"Saved {len(out)} districts from {n_states} states → {out_path}")
    if len(out) < 435:
        print(f"  NOTE: {435 - len(out)} districts missing (partial data). "
              f"PVI will use single-election fallback for missing districts.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate per-state presidential CSVs")
    parser.add_argument("--year", type=int, required=True,
                        help="Presidential election year (e.g. 2020)")
    args = parser.parse_args()
    consolidate(args.year)


if __name__ == "__main__":
    main()
