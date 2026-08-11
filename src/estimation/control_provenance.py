"""
Per-race spending-CONTROL provenance table.

Added 2026-08-11 in response to a design critique of the original party_r =
r_total - cand_r_total construction: that quantity is "R-side money not
raised by the R candidate," which conflates money the NRCC actually
DECIDES how to spend with money spent independently by super PACs, 527s,
and other outside groups the NRCC has no authority over. A dollar the
NRCC cannot move is not a valid action in BR_R's action space -- feeding it
in as decision-variable dollars would let the "player" reallocate money it
was never able to touch.

Every dollar in d_total / r_total is exactly ONE of:
    cand         candidate committee's own disbursements
                 (backtest.data.fec.load_candidate_disbursements)
    party_natl   the NATIONAL committee's own money: its coordinated
                 expenditures (capped by FEC coordinated-expenditure limits)
                 PLUS its own independent expenditures (a "hybrid" IE
                 strategy party committees legally may use, distinct from a
                 super PAC's IE -- see extract_national_committee_ies()).
                 THIS is x_D / x_R: the two-player game's actual decision
                 variable for Model A (DCCC vs. NRCC).
    party_state  state party committees' own 24K coordinated expenditures.
                 Real, coordinated, party money -- but controlled by STATE
                 parties, not DCCC/NRCC. Floor money for a DCCC-vs-NRCC game,
                 not decision-variable money, even though it's legally
                 "coordinated" the same way party_natl is.
    outside      every other independent expenditure: super PACs, 527s, any
                 group that is NOT the national committee itself. Floor
                 money -- it still affects a race's marginal returns (more
                 outside spending compresses the persuasion ceiling exactly
                 like more candidate spending does), it's just not a lever
                 either player's optimizer is allowed to pull.

Identity checked, not assumed: cand + party_natl + party_state + outside ==
d_total (resp. r_total) for every race, since these four sources are a
strict re-partition of the exact same underlying candidate/coordinated/IE
data build_total_spend() already sums -- see build_provenance_table()'s
assertion. A mismatch means a source was missed or double-counted, not a
modeling choice.

This decomposition is "Model A" (project critique's term): candidate and
outside-group spending held as exogenous floors, only the two national
committees' own money treated as strategic. Model B (state parties/major
outside groups as additional strategic agents) and Model C (a full
multi-player game) are explicitly out of scope here.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from backtest.data import fec  # noqa: E402
from backtest import config  # noqa: E402

import fetch_data  # noqa: E402 -- extract_national_committee_ies, DCCC_/NRCC_COMMITTEE_ID


def _load_coord_source(cycle: int, filename: str) -> pd.Series:
    path = config.raw_path("fec") / filename
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, dtype={"district_id": str})
    return df.set_index("district_id")["coordinated_expenditures"]


def build_provenance_table(cycle: int) -> pd.DataFrame:
    """One row per (district_id, party): cand, party_natl, party_state,
    outside, and their sum, checked against build_total_spend()'s d_total/
    r_total for the same race."""
    cand = fec.load_candidate_disbursements(cycle)
    national_ie = fetch_data.extract_national_committee_ies(cycle)
    all_ie = fec.load_independent_expenditures(cycle)
    totals = fec.build_total_spend(cycle).set_index("district_id")

    dccc_coord = _load_coord_source(cycle, f"coordinated_dccc_{cycle}.csv")
    nrcc_coord = _load_coord_source(cycle, f"coordinated_nrcc_{cycle}.csv")
    state_dem_coord = _load_coord_source(cycle, f"coordinated_state_party_dem_{cycle}.csv")
    state_rep_coord = _load_coord_source(cycle, f"coordinated_state_party_rep_{cycle}.csv")

    rows = []
    all_districts = sorted(set(cand["district_id"]) | set(totals.index))
    for district_id in all_districts:
        d_total = float(totals["d_total"].get(district_id, 0.0))
        r_total = float(totals["r_total"].get(district_id, 0.0))

        cand_d = float(cand.loc[(cand.district_id == district_id) & (cand.party == "D"),
                                 "candidate_disbursements"].sum())
        cand_r = float(cand.loc[(cand.district_id == district_id) & (cand.party == "R"),
                                 "candidate_disbursements"].sum())

        natl_ie_d = float(national_ie.loc[(national_ie.district_id == district_id)
                                           & (national_ie.party == "D"), "national_committee_ie"].sum())
        natl_ie_r = float(national_ie.loc[(national_ie.district_id == district_id)
                                           & (national_ie.party == "R"), "national_committee_ie"].sum())

        all_ie_d = float(all_ie.loc[(all_ie.district_id == district_id) & (all_ie.party == "D"), "ie_net"].sum())
        all_ie_r = float(all_ie.loc[(all_ie.district_id == district_id) & (all_ie.party == "R"), "ie_net"].sum())

        party_natl_d = dccc_coord.get(district_id, 0.0) + natl_ie_d
        party_natl_r = nrcc_coord.get(district_id, 0.0) + natl_ie_r
        party_state_d = state_dem_coord.get(district_id, 0.0)
        party_state_r = state_rep_coord.get(district_id, 0.0)
        outside_d = all_ie_d - natl_ie_d
        outside_r = all_ie_r - natl_ie_r

        rows.append(dict(
            district_id=district_id, cycle=cycle,
            cand_d=cand_d, party_natl_d=party_natl_d, party_state_d=party_state_d, outside_d=outside_d,
            cand_r=cand_r, party_natl_r=party_natl_r, party_state_r=party_state_r, outside_r=outside_r,
            d_total_check=cand_d + party_natl_d + party_state_d + outside_d, d_total=d_total,
            r_total_check=cand_r + party_natl_r + party_state_r + outside_r, r_total=r_total,
        ))

    df = pd.DataFrame(rows)
    max_d_err = (df["d_total_check"] - df["d_total"]).abs().max()
    max_r_err = (df["r_total_check"] - df["r_total"]).abs().max()
    assert max_d_err < 1.0, f"D-side provenance identity broken: max error ${max_d_err:,.2f}"
    assert max_r_err < 1.0, f"R-side provenance identity broken: max error ${max_r_err:,.2f}"
    return df


def apply_control_floor(races: list, cycle: int) -> tuple[list, np.ndarray]:
    """Redefine each race's D floor (RaceRecord.cand_d_total) and R floor
    (returned array, mirroring load_cand_r_total's existing convention) to
    be the FULL non-national-committee-controlled money (cand + party_state
    + outside), not just candidate money -- so that, everywhere downstream
    in game/, `party_d = d_total - cand_d_total` and `party_r = r_total -
    cand_r_total` recover x_D / x_R (national-committee-controlled money
    only) automatically, with no changes needed to game/payoff.py,
    best_response.py, exploitability.py, or persistent_value.py: they all
    already treat "floor" and "total minus floor" generically.

    d_total / r_total themselves are NOT touched -- the margin/response
    model still correctly sees TOTAL two-party spending (its effect on vote
    share doesn't depend on who controls the money), only the game layer's
    notion of the controllable action changes."""
    prov = build_provenance_table(cycle).set_index("district_id")

    new_races = []
    for r in races:
        row = prov.loc[r.district_id] if r.district_id in prov.index else None
        floor_d = (row["cand_d"] + row["party_state_d"] + row["outside_d"]) if row is not None else r.cand_d_total
        new_races.append(dataclasses.replace(r, cand_d_total=float(floor_d)))

    floor_r = np.array([
        float(prov.loc[r.district_id, "cand_r"] + prov.loc[r.district_id, "party_state_r"]
              + prov.loc[r.district_id, "outside_r"]) if r.district_id in prov.index else 0.0
        for r in races
    ])
    return new_races, floor_r
