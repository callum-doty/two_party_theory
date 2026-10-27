#!/usr/bin/env python3
"""
Frozen pre-election forecast of the 2026 House under the two-job model, and
optionally where the next chunk of DCCC money would buy the most expected
seats.

  * baseline fit on every completed cycle (2012-2024), PVI on 2026 lines
    (10 states redrew; Missouri stays on its 2022 map, simple_model/pvi.py);
  * the generic ballot is an input (--gb), with +/-3 point scenarios for
    national polling error, which hits every race at once;
  * money outside the national committees' control comes from data/raw/fec
    (refresh it first with scripts/fetch_data.py --cycles 2026 --only fec).

Writes forecasts/house_2026_<as_of>.csv (one row per race) and .json
(metadata, coefficients, scenarios). Meant to be committed before Election
Day and scored against results afterwards.

Usage:
    python scripts/forecast_2026.py --gb 8.4
    python scripts/forecast_2026.py --gb 8.4 --extra 5e6   # once race-level DCCC data is complete
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from simple_model import baseline as bl  # noqa: E402
from simple_model.allocate import allocate, win_prob  # noqa: E402
from simple_model.pvi import MAP_OVERRIDES, _shares  # noqa: E402
from simple_model.spending import SpendingEffect  # noqa: E402

CYCLE = 2026
GB_SCENARIO_PTS = 3.0
SNAPSHOT_FILES = (
    "data/raw/fec/candidate_disbursements_2026.csv",
    "data/raw/fec/independent_expenditures_2026.csv",
    "data/raw/fec/coordinated_expenditures_2026.csv",
    "data/raw/fec/incumbency_2026.csv",
    "data/raw/independent_expenditure/independent_expenditure_2026.csv",
    "data/processed/pres_by_lines.csv",
)


def redrawn_states() -> dict[str, int]:
    """States whose 2026 districts differ from 2024 lines (2024 results changed)."""
    new, old = _shares(CYCLE)[2024], _shares(2024)[2024]
    changed = (new - old.reindex(new.index)).abs() > 0.005
    counts = changed[changed].index.str[:2].value_counts().to_dict()
    for (c, state), _ in MAP_OVERRIDES.items():
        if c == CYCLE:
            counts.pop(state, None)
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen 2026 House forecast")
    parser.add_argument("--gb", type=float, required=True, help="generic ballot, D minus R, points")
    parser.add_argument("--extra", type=float, default=0.0,
                        help="additional DCCC dollars to place on top of spending to date (0 = skip). "
                             "Only meaningful once race-level DCCC spending is complete; see the notes.")
    parser.add_argument("--step", type=float, default=100_000)
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    as_of = dt.date.today().isoformat()

    base = bl.fit(bl.TRAINING_CYCLES)
    races, df = bl.load_cycle(CYCLE)
    df["gb"] = args.gb
    mu = base.margin(df)
    p = norm.cdf(mu / base.sigma)

    scenarios = {f"{g:+.1f}": float(norm.cdf((mu + (g - args.gb)) / base.sigma).sum())
                 for g in (args.gb - GB_SCENARIO_PTS, args.gb, args.gb + GB_SCENARIO_PTS)}

    out = df[["district_id", "cook_rating", "status", "pvi"]].rename(columns={"cook_rating": "tier"}).assign(
        dem_margin=mu.round(2), p_dem=p.round(4))

    effect, gain = SpendingEffect(), None
    if args.extra > 0:
        party_d = np.array([max(r.d_total - r.cand_d_total, 0.0) for r in races])
        voters = df["cvap"].to_numpy(dtype=float)
        x = allocate(mu, base.sigma, effect, voters, party_d, args.extra, cap=args.extra,
                     step=args.step, start=party_d)
        gain = float(win_prob(mu, base.sigma, effect, voters, x, party_d).sum() - p.sum())
        out = out.assign(dccc_to_date=party_d.round(0), extra_recommended=(x - party_d).round(0))
    out_dir = REPO_ROOT / "forecasts"
    out_dir.mkdir(exist_ok=True)
    stem = out_dir / f"house_2026_{as_of}"
    out.to_csv(f"{stem}.csv", index=False)

    meta = dict(
        as_of=as_of, election_day="2026-11-03", generic_ballot=args.gb,
        model=dict(train_cycles=list(base.train_cycles), n_train=base.n_train, sigma=base.sigma,
                   coef=dict(zip(["intercept", "pvi", "d_incumbent", "open_seat", "floor_share"],
                                 base.coef.round(4).tolist())),
                   spending_effect=dict(scale_pts=effect.scale_pts,
                                        saturation_per_voter=effect.saturation_per_voter)),
        n_races_modeled=len(out), expected_dem_seats_modeled=float(p.sum()),
        expected_dem_seats_by_gb=scenarios,
        extra_dollars=args.extra, extra_expected_seat_gain=gain,
        redrawn_states_2026=redrawn_states(),
        map_overrides={f"{c}-{s}": f"uses {lines} lines" for (c, s), lines in MAP_OVERRIDES.items()},
        data_snapshot={f: dt.datetime.fromtimestamp((REPO_ROOT / f).stat().st_mtime).isoformat(timespec="minutes")
                       for f in SNAPSHOT_FILES if (REPO_ROOT / f).exists()},
        notes=[
            "Tiers are derived from PVI and incumbency (backtest/data/pvi.py derive_rating), not published ratings.",
            "Only races with >= $100K total spending are modeled (backtest/data/universe.py); Alaska is excluded.",
            "floor_share uses money outside the national committees' control as reported to the FEC by the snapshot.",
            "The spending effect is an assumption (simple_model/spending.py), not an estimate.",
            "NRSC v. FEC (2026-06-30) struck down party coordinated-spending limits. For 2026 the FEC reports DCCC "
            "with $0 in independent expenditures and $60.9M in coordinated party expenditures through 2026-10-14, "
            "but only ~$5.2M of that is itemized by race in the API data, so race-level DCCC spending to date is "
            "incomplete. The win probabilities do not depend on it; any --extra allocation does.",
            "The D-incumbent coefficient (~9-11 margin points) is estimated mostly from incumbents in their own "
            "districts; for incumbents on heavily redrawn 2026 maps (e.g. TX-28, TX-34, NC-01, OH-09) it is "
            "probably overstated.",
            "Map status: Texas (new map, SCOTUS Dec 2025) and Missouri (2022 map, SCOTUS 2026-09-25) were checked "
            "after July 2026; other redrawn states follow The Downballot's July 2026 data.",
        ],
    )
    Path(f"{stem}.json").write_text(json.dumps(meta, indent=2))

    print(f"2026 House forecast as of {as_of}, GB D{args.gb:+.1f}; {len(out)} races modeled")
    print(f"  baseline: {base.n_train} contested races 2012-2024, sigma {base.sigma:.1f} pts")
    print("  expected D seats among modeled races by generic ballot: "
          + ", ".join(f"D{k}: {v:.1f}" for k, v in scenarios.items()))
    print(f"  redrawn states (districts changed): {meta['redrawn_states_2026']}")
    close = out[(out.p_dem > 0.2) & (out.p_dem < 0.8)].sort_values("p_dem")
    print(f"\n  {len(close)} races between 20% and 80%:")
    print(close[["district_id", "tier", "status", "pvi", "dem_margin", "p_dem"]]
          .to_string(index=False, formatters={"p_dem": "{:.0%}".format}))
    if gain is not None:
        money = lambda v: f"${v / 1e6:.1f}M"  # noqa: E731
        top = out[out.extra_recommended > 0].sort_values("extra_recommended", ascending=False)
        print(f"\n  Next ${args.extra / 1e6:.1f}M of DCCC money (expected gain {gain:+.2f} seats):")
        print(top[["district_id", "tier", "p_dem", "dccc_to_date", "extra_recommended"]].to_string(
            index=False, formatters={"p_dem": "{:.0%}".format, "dccc_to_date": money, "extra_recommended": money}))
    print(f"\nSaved -> {stem.relative_to(REPO_ROOT)}.csv / .json")


if __name__ == "__main__":
    main()
