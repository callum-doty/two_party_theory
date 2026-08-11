#!/usr/bin/env python3
"""
Paper II — one-step-ahead historical simulation (docs/paper2_draft.md §6).

Requires run_estimation.py to have completed successfully first (same
assumption run_backtest.py already makes).

Replays a historical cycle's reporting periods one at a time, reconstructing
each period's state from real historical data only (never from the model's
own prior recommendation — see dynamic/simulate.py's module docstring for
why), and compares the model's period-by-period recommendation against
DCCC's actual behavior.

This is NOT the same computation as run_backtest.py: run_backtest.py asks
"was the full-cycle observed allocation efficient" (Paper I, retrospective,
one-shot); this script asks "at each point in the cycle, does the model's
recommendation for the *next* dollar diverge from what DCCC actually did"
(Paper II, sequential, one-step-ahead).

Usage:
    python scripts/run_dynamic_backtest.py                      # 2024, biweekly
    python scripts/run_dynamic_backtest.py --cycle 2022
    python scripts/run_dynamic_backtest.py --cadence fec_quarterly
"""

from __future__ import annotations
import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from backtest import config
from backtest.data.universe import build_universe
from backtest.dynamic.ledger import ZeroCommitmentSource, OperationalLedgerSource
from backtest.dynamic.updates import EMAStateUpdater
from backtest.dynamic.periods import biweekly_periods, fec_quarterly_periods
from backtest.dynamic.simulate import one_step_ahead
from backtest.dynamic.timing import build_timing_table, timing_gap_vs_volatility

from run_backtest import load_processed_artifacts, build_dummy_factor_model  # Paper I loaders, reused unmodified

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_dynamic_backtest")


def _commitment_source(mode: str):
    if mode == "zero":
        return ZeroCommitmentSource()
    if mode == "operational":
        ledger_path = config.processed_path().parent / "dynamic_ledger.csv"
        return OperationalLedgerSource(ledger_path)
    if mode == "research_stub":
        from backtest.dynamic.ledger import AdReservationProxySource
        return AdReservationProxySource(strict=False)
    raise ValueError(f"Unknown dynamic.commitment_mode: {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper II one-step-ahead historical simulation")
    parser.add_argument("--cycle", type=int, default=2024,
                        help="Election cycle to replay (default: 2024)")
    parser.add_argument("--processed-dir", type=str, default=None,
                        help="Path to estimation artifacts directory "
                             "(default: data/processed). Use with --cycle for OOS cycles.")
    parser.add_argument("--cadence", type=str, default=None, choices=["biweekly", "fec_quarterly"],
                        help="Reporting-period cadence (default: config.yaml's dynamic.reporting_cadence)")
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="Risk-aversion coefficient (default: 0.0, risk-neutral)")
    parser.add_argument("--cap-fraction", type=float, default=0.15,
                        help="Per-race concentration cap as a fraction of deployable capital")
    parser.add_argument("--n-competitive-races", type=int, default=None,
                        help="Limit to the top N competitive races (faster runs). Default: all.")
    args = parser.parse_args()

    dyn_cfg = config.dynamic_cfg()
    cadence = args.cadence or dyn_cfg["reporting_cadence"]

    processed = Path(args.processed_dir) if args.processed_dir else config.processed_path()
    _, coef, sigma_model = load_processed_artifacts(processed)

    logger.info(f"Building race universe for cycle {args.cycle}…")
    all_races = build_universe(cycle=args.cycle)
    races = [r for r in all_races if r.cook_rating in ("Toss-Up", "Lean D", "Lean R")]
    if args.n_competitive_races:
        races = races[: args.n_competitive_races]
    logger.info(f"Historical harness universe: {len(races)} competitive races")

    gb = config.generic_ballot_for_cycle(args.cycle)
    factor_model = build_dummy_factor_model(races, gb)
    cov_matrix = factor_model.race_covariance()

    if cadence == "biweekly":
        periods = biweekly_periods(date(args.cycle - 1, 6, 1), date(args.cycle, 11, 5))
    else:
        periods = fec_quarterly_periods(args.cycle)
    logger.info(f"Reporting-period cadence: {cadence} ({len(periods)} periods)")

    # DCCC's own controllable budget, held fixed across all periods — the
    # historical harness does not attempt to reconstruct a period-by-period
    # fundraising path (a further, explicitly stated approximation on top
    # of the Phase 3 data gaps documented in dynamic/simulate.py).
    party_budget = sum(max(r.d_total - r.cand_d_total, 0.0) for r in races)

    commitment_source = _commitment_source(dyn_cfg["commitment_mode"])
    state_updater = EMAStateUpdater(lam=dyn_cfg["ema_lambda"])

    logger.info("Running one-step-ahead historical simulation…")
    results = one_step_ahead(
        periods, args.cycle, races, coef, sigma_model,
        commitment_source, state_updater,
        cov_matrix_fn=lambda rs: cov_matrix,
        gamma=args.gamma, cap_fraction=args.cap_fraction,
        total_budget_fn=lambda t: party_budget,
        generic_ballot_national=gb,
    )
    logger.info(f"Completed {len(results)} one-step-ahead periods")

    timing = build_timing_table(results)
    races_by_district = {r.district_id: r for r in races}
    gap_vs_vol = timing_gap_vs_volatility(timing, sigma_model, races_by_district)

    out_dir = config.outputs_path()
    out_dir.mkdir(parents=True, exist_ok=True)

    timing_df = pd.DataFrame([vars(tc) for tc in timing])
    timing_path = out_dir / f"dynamic_timing_{args.cycle}.csv"
    timing_df.to_csv(timing_path, index=False)
    logger.info(f"Per-period, per-race timing table → {timing_path}")

    gap_path = out_dir / f"dynamic_timing_gap_vs_volatility_{args.cycle}.csv"
    gap_vs_vol.to_csv(gap_path, index=False)
    correlation = gap_vs_vol.attrs.get("correlation")
    logger.info(f"Timing-gap-vs-volatility table → {gap_path}"
                + (f" (correlation={correlation:.3f})" if correlation is not None else ""))

    total_gap = sum(tc.gap for tc in timing)
    logger.info(
        f"Summary: {len(races)} races × {len(periods)} periods, "
        f"total model-vs-actual deployment gap = ${total_gap:,.0f} "
        "(positive = model front-loads relative to DCCC's actual pacing, "
        "per paper §5.4.1)."
    )


if __name__ == "__main__":
    main()
