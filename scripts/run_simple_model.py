#!/usr/bin/env python3
"""
The two-job model end to end, for one completed backtest cycle:

  1. fit the baseline on every EARLIER cycle and check its calibration on
     this one;
  2. reallocate DCCC's own controllable budget greedily under the default
     spending assumption, compared with what DCCC actually did;
  3. repeat step 2 across a grid of spending assumptions.

The live 2026 cycle has its own script, scripts/forecast_2026.py.

Usage:
    python scripts/run_simple_model.py --cycle 2024
    python scripts/run_simple_model.py --cycle 2022 --train 2014 2016 2018 2020
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from simple_model import baseline as bl  # noqa: E402
from simple_model.allocate import allocate, win_prob  # noqa: E402
from simple_model.spending import SpendingEffect  # noqa: E402

COMPETITIVE = ("Lean D", "Toss-Up", "Lean R")
SCALES = (0.5, 1.5, 4.5)
SATURATIONS = (3.0, 10.0, 30.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-job baseline + spending-effect model")
    parser.add_argument("--cycle", type=int, default=2024)
    parser.add_argument("--train", type=int, nargs="+", default=None,
                        help="training cycles (default: every earlier cycle)")
    parser.add_argument("--cap-fraction", type=float, default=0.15)
    parser.add_argument("--step", type=float, default=100_000)
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    train = tuple(args.train or [c for c in bl.TRAINING_CYCLES if c < args.cycle])
    base = bl.fit(train)
    a, b, inc, opn, fs = base.coef
    print(f"Baseline, trained on {train} (n={base.n_train} contested races):")
    print(f"  margin = {a:+.1f} + {b:.2f}*PVI + {inc:+.1f}*[D incumbent] + {opn:+.1f}*[open]"
          f" + {fs:.2f}*floor_share + GB;  sigma = {base.sigma:.1f} pts")

    races, df = bl.load_cycle(args.cycle)
    mu = base.margin(df)
    cal = bl.calibration_by_tier(base.win_prob(df), df)
    print(f"\nCalibration on {args.cycle} (not used in fitting), Brier {cal.attrs['brier']:.4f}:")
    print(cal.to_string(index=False, formatters={"predicted": "{:.1%}".format, "actual": "{:.1%}".format}))

    party_obs = np.array([max(r.d_total - r.cand_d_total, 0.0) for r in races])
    budget = float(party_obs.sum())
    cap = args.cap_fraction * budget
    voters = df["cvap"].to_numpy(dtype=float)
    tier = df["cook_rating"].to_numpy()

    def run(effect: SpendingEffect) -> dict:
        x = allocate(mu, base.sigma, effect, voters, party_obs, budget, cap, step=args.step)
        e_obs = float(win_prob(mu, base.sigma, effect, voters, party_obs, party_obs).sum())
        e_new = float(win_prob(mu, base.sigma, effect, voters, x, party_obs).sum())
        return dict(x=x, e_obs=e_obs, e_new=e_new, gain=e_new - e_obs,
                    competitive_share=float(x[np.isin(tier, COMPETITIVE)].sum() / max(x.sum(), 1.0)),
                    newly_funded=int(((party_obs < 1e4) & (x > 1e4)).sum()))

    default = SpendingEffect()
    r = run(default)
    obs_share = party_obs[np.isin(tier, COMPETITIVE)].sum() / budget
    print(f"\nReallocation of DCCC's ${budget / 1e6:.1f}M (cap {args.cap_fraction:.0%}/race), "
          f"default spending effect (scale {default.scale_pts} pts, saturation ${default.saturation_per_voter:g}/voter):")
    print("  tier        DCCC $M   model $M")
    for t in bl.TIERS:
        k = tier == t
        print(f"  {t:9s}  {party_obs[k].sum() / 1e6:8.1f}   {r['x'][k].sum() / 1e6:8.1f}")
    print(f"  Lean/Toss-Up share: DCCC {obs_share:.0%}, model {r['competitive_share']:.0%};  "
          f"races newly funded: {r['newly_funded']}")
    print(f"  Expected D seats: DCCC {r['e_obs']:.2f}, model {r['e_new']:.2f}, gain {r['gain']:+.2f}")

    print("\nSensitivity (gain in seats / Lean+Toss-Up share of model money):")
    print("  scale \\ saturation " + "".join(f"  ${s:>4g}/voter " for s in SATURATIONS))
    grid = []
    for sc in SCALES:
        cells = []
        for sat in SATURATIONS:
            g = run(SpendingEffect(scale_pts=sc, saturation_per_voter=sat))
            grid.append(dict(scale_pts=sc, saturation_per_voter=sat, gain=g["gain"],
                             competitive_share=g["competitive_share"]))
            cells.append(f"  {g['gain']:+5.2f} / {g['competitive_share']:3.0%} ")
        print(f"  {sc:>4g} pts          " + "".join(cells))

    out = dict(cycle=args.cycle, train_cycles=list(train), baseline_coef=base.coef.tolist(),
               sigma=base.sigma, brier=cal.attrs["brier"], calibration=cal.to_dict("records"),
               budget=budget, default=dict(gain=r["gain"], competitive_share=r["competitive_share"],
                                           newly_funded=r["newly_funded"]),
               allocation={d: float(v) for d, v in zip(df["district_id"], r["x"]) if v > 0},
               sensitivity=grid)
    out_path = REPO_ROOT / "results" / f"simple_model_{args.cycle}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
