#!/usr/bin/env python3
"""
Fetch a multi-vintage CVAP panel (FINDINGS.md Section 10.7 Gap 1) -- Census
publishes CVAP special tabulations annually as overlapping 5-year ACS
windows; only a single static 2018-2022 cross-section
(data/raw/census/cvap_2022_acs5.csv) currently exists in this project, which
is why alpha4 (spending-intensity, log((D+R)/CVAP)) is estimated by naive
OLS and found endogenous (FINDINGS.md Gap 1: high-spending races are
structurally more competitive, so OLS picks up selection bias, not a causal
effect) with no within-district variation available to separate the two.

IMPORTANT BOUNDARY, stated explicitly rather than glossed over: vintages
before 2019-2023 use PRE-2022-redistricting (113th-117th Congress)
district boundaries; every state redrew its congressional map after the
2020 census (not just the ~13 districts config.yaml's
universe.redistricting_flag_districts tracks -- that list is a separate,
narrower set of districts redrawn AGAIN after 2022, for the 2024 cycle
specifically). This means district_id is NOT a stable panel key across the
2022-redistricting boundary without a real geographic (GIS) crosswalk,
which this script does NOT attempt to build -- see
src/backtest/estimation/cvap_iv.py's module docstring for how the
fixed-effects estimation this panel feeds handles (and limits itself
around) that gap.

Usage:
    python scripts/fetch_cvap_panel.py
    python scripts/fetch_cvap_panel.py --vintages 2018 2019 2020 2022 2024
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import config
import fetch_data as fd

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("fetch_cvap_panel")

# End years of each rolling 5-year ACS window Census has published a CVAP
# special tabulation for, covering the pre- and post-2022-redistricting
# boundary: 2014-2018 through 2018-2022 use OLD (113th-117th Congress)
# lines; 2019-2023 and 2020-2024 use NEW (118th Congress, current) lines.
DEFAULT_VINTAGES = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Verified EMPIRICALLY (2026-08, not assumed): fetched the 2018/2020/2022/
# 2023/2024 vintages and diffed their district_id sets directly. 2018->2020
# is IDENTICAL (0 districts added/removed). 2020->2022 shows exactly the
# real 2020-census apportionment shift (TX 36->38, MT 1->2 districts, CA/IL/
# MI/NY/OH/PA/WV each -1) -- i.e. Census re-tabulates CVAP under whatever
# congressional boundaries are CURRENT at publication time, not the
# boundaries in effect during the ACS collection years. 2022->2023->2024 are
# each IDENTICAL to the prior vintage too. Practical consequence, stated
# plainly: there is NO within-district_id boundary variation to exploit
# across the "post" vintages (2022 onward) at all -- they are the same map,
# re-tabulated. The real, usable pre/post split is 2018 & 2020 (old
# boundaries) vs. 2022/2023/2024 (current boundaries, all identical to each
# other) -- see src/backtest/estimation/cvap_iv.py for what this means for
# the fixed-effects/instrument design.
POST_REDISTRICTING_MIN_END_YEAR = 2022


def fetch_cvap_vintage(end_year: int, force: bool = False) -> "pd.DataFrame":
    """Download+parse one CVAP vintage. Cached to
    data/raw/census/cvap_panel_{end_year}.csv (skip if present, matching
    fetch_census_cvap()'s idempotency)."""
    import pandas as pd

    out_path = config.raw_path("census") / f"cvap_panel_{end_year}.csv"
    if out_path.exists() and not force:
        logger.info(f"CVAP vintage {end_year}: already present, skipping")
        return pd.read_csv(out_path, dtype={"district_id": str})

    url = fd._cvap_vintage_url(end_year)
    content = fd._download_cvap_zip(url, f"{end_year - 4}-{end_year}")
    out = fd._parse_cvap_cd_csv(content)
    out["vintage_end_year"] = end_year
    out["is_post_redistricting"] = end_year >= POST_REDISTRICTING_MIN_END_YEAR
    out.to_csv(out_path, index=False)
    logger.info(f"CVAP vintage {end_year}: saved {len(out)} districts → {out_path}")
    return out


def build_cvap_panel(vintages: list[int] = DEFAULT_VINTAGES, force: bool = False) -> "pd.DataFrame":
    """Concatenate all vintages into one panel -> data/raw/census/cvap_panel_all_vintages.csv
    (district_id, cvap, vintage_end_year, is_post_redistricting).

    Does NOT resolve the pre/post-redistricting district_id discontinuity --
    flags it via is_post_redistricting so downstream estimation code
    (src/backtest/estimation/cvap_iv.py) can restrict to a defensible
    subset rather than silently treating district_id as continuous across
    the boundary."""
    import pandas as pd

    frames = [fetch_cvap_vintage(y, force=force) for y in vintages]
    panel = pd.concat(frames, ignore_index=True)

    out_path = config.raw_path("census") / "cvap_panel_all_vintages.csv"
    panel.to_csv(out_path, index=False)
    logger.info(f"CVAP panel: {len(panel)} district-vintage rows across {len(vintages)} "
                f"vintages → {out_path}")

    # Diagnostic, not a resolution: which district_ids appear in BOTH a
    # pre- and a post-redistricting vintage but plausibly refer to
    # different underlying geography (flagged, not silently trusted).
    pre_ids = set(panel.loc[~panel["is_post_redistricting"], "district_id"])
    post_ids = set(panel.loc[panel["is_post_redistricting"], "district_id"])
    overlap = pre_ids & post_ids
    logger.info(f"CVAP panel: {len(overlap)} district_ids appear in both pre- and "
                f"post-2022-redistricting vintages (same ID does NOT guarantee same "
                f"geography -- see this script's module docstring).")
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a multi-vintage Census CVAP panel")
    parser.add_argument("--vintages", nargs="+", type=int, default=DEFAULT_VINTAGES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_cvap_panel(args.vintages, force=args.force)


if __name__ == "__main__":
    main()
