"""
Cook Political Report data — PVI and race ratings.

Primary path (computed from presidential results):
  PVI is computed from daily presidential results by congressional district.
  Ratings are derived from PVI + incumbency using standard thresholds.
  See data/pvi.py for the formula and data sources.

Override path (if you have proprietary Cook files):
  Place these CSVs under data/raw/cook_pvi/ and they will take precedence:

  cook_pvi_{cycle}.csv
    Columns: district_id, pvi_raw (e.g. "D+3", "R+8", "EVEN")

  cook_ratings_2024.csv
    Columns: district_id, rating
    Values:  "Safe D", "Likely D", "Lean D", "Toss-Up", "Lean R", "Likely R", "Safe R"
"""

from __future__ import annotations
import re
import pandas as pd
from .. import config


def _parse_pvi(raw: str) -> float:
    """
    Convert a Cook PVI string to a signed float.
    "D+3" → +3.0, "R+8" → -8.0, "EVEN" → 0.0
    """
    raw = str(raw).strip().upper()
    if raw in ("EVEN", "0", ""):
        return 0.0
    m = re.match(r"([DR])\+(\d+(?:\.\d+)?)", raw)
    if not m:
        raise ValueError(f"Cannot parse PVI: {raw!r}")
    sign = 1.0 if m.group(1) == "D" else -1.0
    return sign * float(m.group(2))


def load_pvi(cycle: int) -> pd.DataFrame:
    """
    Return signed PVI per district for a given cycle.

    Uses proprietary Cook PVI file if present; otherwise computes from
    presidential results via data/pvi.py.

    Returns DataFrame: district_id, pvi (float, D-positive)
    """
    cook_path = config.raw_path("cook") / f"cook_pvi_{cycle}.csv"
    if cook_path.exists():
        df = pd.read_csv(cook_path, dtype={"district_id": str})
        df["pvi"] = df["pvi_raw"].apply(_parse_pvi)
        return df[["district_id", "pvi"]]

    from . import pvi as pvi_module
    return pvi_module.load_pvi(cycle)


def load_ratings(cycle: int) -> pd.DataFrame:
    """
    Return race ratings for a given cycle.

    Uses proprietary Cook ratings file (cook_ratings_{cycle}.csv) if present;
    otherwise derives ratings from PVI + incumbency thresholds.

    Returns DataFrame: district_id, cook_rating (str)
    """
    cook_path = config.raw_path("cook") / f"cook_ratings_{cycle}.csv"
    if cook_path.exists():
        df = pd.read_csv(cook_path, dtype={"district_id": str, "rating": str})
        return df.rename(columns={"rating": "cook_rating"})[["district_id", "cook_rating"]]

    from . import pvi as pvi_module
    from . import incumbency as incumb_module

    pvi_df = pvi_module.load_pvi(cycle)
    incumb_df = incumb_module.load_incumbency(cycle)
    merged = pvi_df.merge(
        incumb_df[["district_id", "incumb_status"]], on="district_id", how="left"
    )
    merged["incumb_status"] = merged["incumb_status"].fillna("Open")
    merged["cook_rating"] = merged.apply(
        lambda row: pvi_module.derive_rating(row["pvi"], row["incumb_status"]), axis=1
    )
    return merged[["district_id", "cook_rating"]]
