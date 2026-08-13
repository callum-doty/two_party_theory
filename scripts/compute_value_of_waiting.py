#!/usr/bin/env python3
"""
Value-of-waiting: is it actually better to hold delta in reserve and
deploy it later (once the opponent's flexibility has shrunk), or would
deploying it NOW to the best currently-available alternative dominate?
(2026-08-13 follow-up -- strategic_window.py showed retention improves
with delay, but never compared that against the foregone value of
deploying the SAME capital elsewhere immediately. That comparison is the
actual "value of waiting," not just "does this race's retention rise.")

Pure post-processing of results/strategic_window_{cycle}.json -- no new
solves. Every quantity needed (PSV_i(t) for every candidate race at every
reference date, both sides, both cycles) was already computed there.

Definitions, for a Democratic delta at race i (mirror for R):
  V_now(i)           = PSV_i at the EARLIEST reference date (120 days out,
                        ~full flexibility) -- deploy delta to race i right away.
  V_alt(t_early)      = max over the OTHER D-side candidate races j != i of
                        PSV_j at 120 days out -- the best available immediate
                        alternative use of the SAME delta.
  best_immediate(i)   = max(V_now(i), V_alt(t_early)) -- the best thing you
                        could do RIGHT NOW with this delta, race i or not.
  V_wait(i)           = PSV_i at race i's OWN T_i^80 (its strategic opening
                        date from strategic_window.py -- the first date the
                        improvement becomes durable, NOT the literal end of
                        the cycle, which that module already flagged as a
                        mechanical convergence floor common to every race
                        and therefore uninformative as a "wait target").
  net_waiting_value(i) = V_wait(i) - best_immediate(i)

Positive net_waiting_value: holding delta in reserve specifically for race
i, until it becomes durable, beats deploying it to the best currently-known
alternative right away. Negative: the reverse -- act now, on whichever
target is actually best today.

TWO LIMITATIONS, stated rather than modeled away:
  1. V_alt only searches the OTHER 5 pre-selected candidates on the same
     side, not the full 433-race universe -- these were screened for being
     already-attractive (top swing / top-|Z|), so the TRUE best immediate
     alternative is probably at least this good and could be better.
     net_waiting_value is therefore a plausible UPPER bound on the true
     value of waiting, not a tight estimate.
  2. This is RETROSPECTIVE, on realized data -- it compares two certain
     outcomes, not the uncertain ones a real-time decision-maker actually
     faces (information resolving unfavorably, the opponent locking up the
     SAME race first, the race's own fundamentals shifting). It answers
     "would waiting have paid off, given what actually happened," not
     "should you wait, given what you could know in advance."

Usage:
    python scripts/compute_value_of_waiting.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS = REPO_ROOT / "results"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("compute_value_of_waiting")


def load(cycle: int) -> dict:
    return json.load(open(RESULTS / f"strategic_window_{cycle}.json"))


def analyze_side(rows: list[dict], t80_map: dict, earliest_date: str) -> list[dict]:
    districts = sorted({r["district_id"] for r in rows})
    by_district = {d: [r for r in rows if r["district_id"] == d] for d in districts}
    psv_at_earliest = {d: next(r["PSV"] for r in by_district[d] if r["ref_date"] == earliest_date) for d in districts}

    out = []
    for d in districts:
        v_now = psv_at_earliest[d]
        alt_vals = [psv_at_earliest[j] for j in districts if j != d]
        v_alt = max(alt_vals) if alt_vals else float("-inf")
        alt_district = max((j for j in districts if j != d), key=lambda j: psv_at_earliest[j]) if alt_vals else None
        best_immediate = max(v_now, v_alt)

        t80 = t80_map.get(d)
        v_wait = next(r["PSV"] for r in by_district[d] if r["ref_date"] == t80) if t80 else None
        net = (v_wait - best_immediate) if v_wait is not None else None

        out.append(dict(
            district_id=d, V_now=v_now, best_alt_district=alt_district, V_alt_immediate=v_alt,
            best_immediate=best_immediate, T80=t80, V_wait=v_wait, net_waiting_value=net,
        ))
    return out


def main() -> None:
    all_results = {}
    for cycle in (2024, 2022):
        data = load(cycle)
        earliest_date = data["strategic_window_D"][0]["ref_date"]  # first row = first (farthest-out) date processed
        d_rows = analyze_side(data["strategic_window_D"], data["T80_D"], earliest_date)
        r_rows = analyze_side(data["strategic_window_R"], data["T80_R"], earliest_date)
        all_results[cycle] = dict(D=d_rows, R=r_rows)

        logger.info(f"=== {cycle} ===")
        for side, rows in (("D", d_rows), ("R", r_rows)):
            logger.info(f"--- {side}-side ---")
            for r in rows:
                net_str = f"{r['net_waiting_value']:+.4f}" if r['net_waiting_value'] is not None else "n/a (never reached 80%)"
                logger.info(
                    f"  {r['district_id']:8s} V_now={r['V_now']:+.4f}  "
                    f"best_alt={r['best_alt_district']}({r['V_alt_immediate']:+.4f})  "
                    f"best_immediate={r['best_immediate']:+.4f}  T80={r['T80']}  "
                    f"V_wait={r['V_wait']:+.4f}  net_waiting_value={net_str}"
                )

    out_path = RESULTS / "value_of_waiting.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
