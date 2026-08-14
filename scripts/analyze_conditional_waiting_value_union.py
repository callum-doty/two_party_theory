#!/usr/bin/env python3
"""
Per-race conditional value of waiting, union pool with redistricting-
flagged districts excluded (2026-08-14, K=15-20 stress-test follow-up).

Pure post-processing of results/strategic_window_union_{cycle}.json --
no new solves. Mirrors compute_value_of_waiting.py's per-race
V_now/best_immediate/V_wait/net_waiting_value design, but: (1) runs on
the much larger union candidate pool instead of K=3, (2) excludes
RaceRecord.redistricting_flagged districts (NC-06/13/14/etc. --
documented elsewhere in this project as having a less certain baseline),
and (3) defines V_wait using the genuine (excl. mechanical-final-week)
peak PSV, per theta_final_week_sensitivity.py's finding that including
the final reference date manufactures spurious "waiting value" for any
pool member with a large raw V_uni, independent of its actual trajectory.

Usage:
    python scripts/analyze_conditional_waiting_value_union.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS = REPO_ROOT / "results"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("analyze_conditional_waiting_value_union")

REDISTRICTING_FLAGGED = {
    "AL-02", "LA-06", "NC-01", "NC-06", "NC-07", "NC-10", "NC-13", "NC-14",
    "NY-03", "NY-04", "NY-17", "NY-18", "NY-22",
}


def analyze(window: dict, side: str) -> list[dict]:
    key = "strategic_window_D" if side == "D" else "strategic_window_R"
    by_district: dict[str, list[dict]] = {}
    for row in window[key]:
        if row["district_id"] in REDISTRICTING_FLAGGED:
            continue
        by_district.setdefault(row["district_id"], []).append(row)
    days = window["days_before"]

    v_now = {d: rows[0]["PSV"] for d, rows in by_district.items()}
    out = []
    for d, rows in by_district.items():
        excl_final = rows[:-1]
        best_excl = max(excl_final, key=lambda r: r["PSV"])
        alt_vals = [v for dd, v in v_now.items() if dd != d]
        best_immediate = max([v_now[d]] + alt_vals)
        best_alt = max((dd for dd in v_now if dd != d), key=lambda dd: v_now[dd], default=None)
        already_durable = rows[0]["retention_rate"] >= 0.80
        out.append(dict(
            district=d, V_now=v_now[d], best_immediate=best_immediate, best_alt=best_alt,
            V_wait_genuine=best_excl["PSV"], genuine_days_out=days[rows.index(best_excl)],
            genuine_retention=best_excl["retention_rate"], already_durable=already_durable,
            net_genuine=best_excl["PSV"] - best_immediate,
        ))
    return sorted(out, key=lambda r: -r["net_genuine"])


def main() -> None:
    all_out = {}
    for cycle in (2024, 2022):
        window = json.load(open(RESULTS / f"strategic_window_union_{cycle}.json"))
        for side in ("D", "R"):
            key = f"{cycle}_{side}"
            all_out[key] = analyze(window, side)
            logger.info(f"=== {key} (K={len(all_out[key])}, redistricting-flagged excluded) ===")
            for r in all_out[key]:
                logger.info(
                    f"  {r['district']:6s} V_now={r['V_now']:+.4f} best_immediate={r['best_immediate']:+.4f}({r['best_alt']}) "
                    f"V_wait_gen={r['V_wait_genuine']:+.4f}@{r['genuine_days_out']}d(ret={r['genuine_retention']:.2f}) "
                    f"net={r['net_genuine']:+.4f} durable_day1={r['already_durable']}"
                )

    out_path = RESULTS / "conditional_waiting_value_union_clean.json"
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
