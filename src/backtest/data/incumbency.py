"""
Incumbency status loader.

Raw data contract (file produced by scripts/fetch_data.py --only incumbency)
─────────────────
  data/raw/fec/incumbency_{cycle}.csv
    Columns: district_id, cycle, incumb_status, incumbent_name, challenger_name
    incumb_status values: "Incumbent" | "Challenger" | "Open"

incumb_status is from the Democratic candidate's perspective:
  - "Incumbent"   — Democrat is the sitting member
  - "Challenger"  — Republican is the sitting member, Democrat is challenging
  - "Open"        — No incumbent in the race (retirement, redistricting, etc.)

incumbent_name / challenger_name are used by estimation/beta_rc.py for
repeat-challenger pair identification across consecutive cycles.
"""

from __future__ import annotations
import pandas as pd
from .. import config


def load_incumbency(cycle: int) -> pd.DataFrame:
    """
    Return incumbency status per district.

    Returns DataFrame with columns: district_id, cycle, incumb_status
    """
    path = config.raw_path("fec") / f"incumbency_{cycle}.csv"
    df = pd.read_csv(path, dtype={"district_id": str, "incumb_status": str})
    df["cycle"] = cycle

    valid = {"Incumbent", "Challenger", "Open"}
    bad = set(df["incumb_status"].dropna().unique()) - valid
    if bad:
        raise ValueError(f"Unexpected incumbency values in cycle {cycle}: {bad}")

    # Ensure optional name columns exist (may not be present in hand-curated files)
    for col in ("incumbent_name", "challenger_name"):
        if col not in df.columns:
            df[col] = ""

    return df[["district_id", "cycle", "incumb_status", "incumbent_name", "challenger_name"]]
