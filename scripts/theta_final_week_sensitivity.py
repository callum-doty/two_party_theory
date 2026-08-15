#!/usr/bin/env python3
"""
Final-week sensitivity check for unified Theta (2026-08-14 follow-up,
surfaced by the K~8 action-space expansion): strategic_window.py's own
methodology already established that EVERY race converges to ~100%
retention at the last reference date (7 days out) by construction --
"a mechanical floor, not a race-specific finding," specifically because
opponent flexible budget is nearly exhausted for everyone at that point,
regardless of whether that specific race was ever genuinely contested.
game/unified_theta.py's Bellman recursion has no equivalent safeguard: it
takes max(deploy_t, ..., deploy_last) over ALL 8 dates including the
mechanical final one, so a race with a large raw V_uni that happens to
still be in the K-pool will always be "worth waiting for" at t=7 even if
it was never a genuine mid-season strategic window -- V_uni doesn't vary
by date, and every race's retention approaches 1.0 there regardless of its
OWN trajectory. Expanding the candidate pool from K=3 to K~8 exposed this:
2024 R (NC-06) and 2022 R (FL-07) both have Theta_full realized AT the
final date, while 2024 D (CT-02) and 2022 D are realized at earlier dates
-- a real, substantive difference this script makes visible rather than
papering over by reporting Theta_full alone.

Pure post-processing of results/theta_unified*.json -- reuses the
deploy_value_full/flex_only/info_only dicts already computed there
(no new Monte Carlo, no new solves) and re-runs game/unified_theta.py's
OWN solve_bellman on the date grid with the final (7-days-out) date
dropped, so "value of waiting" is measured against the last date that
still has a materially non-mechanical opponent-flexibility gap.

Usage:
    python scripts/theta_final_week_sensitivity.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from game import unified_theta as ut  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("theta_final_week_sensitivity")


def _trimmed(dates: list[str], deploy_by_date: dict[str, float]) -> dict:
    trimmed_dates = dates[:-1]
    out = ut.solve_bellman(trimmed_dates, deploy_by_date)
    best_date = max(trimmed_dates, key=lambda d: deploy_by_date[d])
    return dict(theta=out["theta_t0"], realized_at=best_date)


def main() -> None:
    all_out = {}
    for pool, fname in (("curve_K3", "theta_unified.json"), ("primary_K8", "theta_unified_expanded.json"),
                        ("union_K15to20", "theta_unified_union.json"),
                        ("union_K15to20_excl_redistricting", "theta_unified_union_excl_redistricting.json"),
                        ("union_weekly_clean", "theta_unified_union_weekly_clean.json")):
        path = REPO_ROOT / "results" / fname
        if not path.exists():
            logger.warning(f"{path} not found, skipping {pool}")
            continue
        data = json.load(open(path))
        pool_out = {}
        for cycle_str, cycle_data in data.items():
            for side in ("D", "R"):
                r = cycle_data[side]
                dates = r["dates"]
                full_last_date = max(dates, key=lambda d: r["deploy_value_full"][d])
                full_last_days = r["days_before"][dates.index(full_last_date)]

                trimmed_full = _trimmed(dates, r["deploy_value_full"])
                trimmed_flex = _trimmed(dates, r["deploy_value_flex_only"])
                trimmed_info = _trimmed(dates, r["deploy_value_info_only"])

                key = f"{cycle_str}_{side}"
                pool_out[key] = dict(
                    districts=r["districts"], K=len(r["districts"]),
                    theta_full=r["theta_full"], theta_full_realized_at_days_before=full_last_days,
                    theta_full_excl_final_week=trimmed_full["theta"],
                    theta_full_excl_final_week_realized_at=trimmed_full["realized_at"],
                    theta_flex_only_excl_final_week=trimmed_flex["theta"],
                    theta_info_only_excl_final_week=trimmed_info["theta"],
                )
                logger.info(
                    f"{pool} {key} (K={pool_out[key]['K']}): "
                    f"Theta_full={r['theta_full']:+.4f} (realized {full_last_days}d out)  "
                    f"-> excl-final-week={trimmed_full['theta']:+.4f} (realized {trimmed_full['realized_at']})  "
                    f"{'[MECHANICAL-FLOOR-DRIVEN]' if full_last_days == 7 else '[genuine mid-season timing]'}"
                )
        all_out[pool] = pool_out

    out_path = REPO_ROOT / "results" / "theta_final_week_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
