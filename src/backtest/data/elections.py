"""
MIT Election Data and Science Lab — House results loader.

Raw data contract
─────────────────
Place the MIT MEDSL House tab file under data/raw/mit_elections/:
  1976-2024-house.tab   (comma-delimited despite the .tab extension)

Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2
"""

from __future__ import annotations
import pandas as pd
from .. import config


_FILENAME = "1976-2024-house.tab"

# MEDSL records state-affiliate party labels literally, not normalized to the
# national party a candidate caucuses with. Minnesota's Democratic-Farmer-
# Labor party and North Dakota's Democratic-NPL fusion label are the two
# variants confirmed present in this repo's panel cycles (2012-2024) via
# direct inspection of 1976-2024-house.tab -- an exact "DEMOCRAT" match alone
# silently coded every MN/ND Democratic candidate as 0 votes (100%-R
# landslide), corrupting the historical panel used to fit beta_RC and the
# margin model for both states, every cycle.
_DEMOCRAT_LABELS = {"DEMOCRAT", "DEMOCRATIC", "DEMOCRATIC-FARMER-LABOR",
                     "DEMOCRATIC-NPL", "DEMOCRATIC-NONPARTISAN LEAGUE"}
_REPUBLICAN_LABELS = {"REPUBLICAN", "INDEPENDENT-REPUBLICAN"}


def _load_raw() -> pd.DataFrame:
    path = config.raw_path("mit") / _FILENAME
    # File uses comma as delimiter despite the .tab extension.
    # Force district to str to preserve leading-zero formatting (e.g. "01").
    # low_memory=False suppresses mixed-type warnings on runoff/special/writein columns.
    return pd.read_csv(path, sep=",", dtype={"district": str}, low_memory=False)


def load_results(cycle: int) -> pd.DataFrame:
    """
    Return two-party vote shares and winner per district for a given cycle.

    Filters to: general election (GEN), TOTAL mode (not absentee/early),
    non-special, non-runoff, non-write-in.

    Sums candidatevotes per party per district to handle multi-candidate cases
    (e.g. fusion tickets, third-party candidates sharing a party line).

    Returns DataFrame with columns:
        district_id, cycle, d_votes, r_votes, d_share, r_share,
        margin_pp (D − R in percentage points), winner ("D" | "R")
    """
    raw = _load_raw()
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")

    # Use "not TRUE" rather than "== FALSE" so that NaN rows (which appear in
    # older cycles where the field was not populated) are kept, not dropped.
    def _not_true(col: pd.Series) -> pd.Series:
        return ~col.astype(str).str.upper().isin(["TRUE", "1"])

    keep = (
        (raw["year"] == cycle)
        & (raw["stage"].str.upper() == "GEN")
        & (raw["mode"].str.upper() == "TOTAL")
        & _not_true(raw["runoff"])
        & _not_true(raw["special"])
        & _not_true(raw["writein"])
    )
    yr = raw[keep].copy()
    yr["candidatevotes"] = pd.to_numeric(yr["candidatevotes"], errors="coerce").fillna(0)
    yr["district_id"] = yr["state_po"] + "-" + yr["district"].str.zfill(2)

    # Sum votes per party × district (handles fusion tickets and multi-candidate primaries
    # where the same party has more than one general-election row)
    d = (
        yr[yr["party"].str.upper().isin(_DEMOCRAT_LABELS)]
        .groupby("district_id")["candidatevotes"].sum()
        .rename("d_votes")
    )
    r = (
        yr[yr["party"].str.upper().isin(_REPUBLICAN_LABELS)]
        .groupby("district_id")["candidatevotes"].sum()
        .rename("r_votes")
    )

    merged = pd.concat([d, r], axis=1).dropna(how="all").fillna(0).reset_index()
    merged["cycle"] = cycle
    merged["total_2p"] = merged["d_votes"] + merged["r_votes"]

    # Avoid division by zero for uncontested races that slipped through
    valid = merged["total_2p"] > 0
    merged.loc[valid, "d_share"] = merged.loc[valid, "d_votes"] / merged.loc[valid, "total_2p"]
    merged.loc[valid, "r_share"] = merged.loc[valid, "r_votes"] / merged.loc[valid, "total_2p"]
    merged.loc[~valid, ["d_share", "r_share"]] = 0.5

    merged["margin_pp"] = (merged["d_share"] - merged["r_share"]) * 100
    merged["winner"] = (merged["d_votes"] > merged["r_votes"]).map({True: "D", False: "R"})

    return merged[[
        "district_id", "cycle", "d_votes", "r_votes",
        "d_share", "r_share", "margin_pp", "winner",
    ]]


def load_panel() -> pd.DataFrame:
    """Return results for all historical panel cycles (2012–2022)."""
    return pd.concat(
        [load_results(c) for c in config.panel_cycles()],
        ignore_index=True,
    )
