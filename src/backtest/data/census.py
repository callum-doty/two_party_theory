"""
Census ACS 2022 5-year CVAP estimates loader.

Raw data contract
─────────────────
Place under data/raw/census/:
  cvap_2022_acs5.csv
    Columns: district_id, cvap (integer)

Source: Census Bureau CVAP Special Tabulation
  https://www.census.gov/programs-surveys/decennial-census/about/voting-rights/cvap.html
"""

from __future__ import annotations
import pandas as pd
from .. import config


def load_cvap() -> pd.DataFrame:
    """
    Return citizen voting age population per congressional district.

    Returns DataFrame with columns: district_id, cvap (int)
    """
    path = config.raw_path("census") / "cvap_2022_acs5.csv"
    df = pd.read_csv(path, dtype={"district_id": str})
    df["cvap"] = df["cvap"].astype(int)
    return df[["district_id", "cvap"]]
