#!/usr/bin/env python3
"""
Fetch all raw data required by the backtest pipeline.

FEC data strategy (two tiers)
──────────────────────────────
TIER 1 — Bulk downloads (no API key, no rate limit):
  Candidate committee totals:
    https://www.fec.gov/files/bulk-downloads/{year}/weball{yy}.zip
    Pipe-delimited, ~170 KB per cycle — downloads in <1 second.

TIER 2 — FEC API (requires registered key for multi-cycle runs):
  DCCC/NRCC independent expenditures:
    /schedules/schedule_e/?committee_id=C00000935   (DCCC)
    /schedules/schedule_e/?committee_id=C00075820   (NRCC)
    ~1,000–3,000 rows per committee per cycle (~10–30 pages each).
  DCCC/NRCC coordinated party expenditures:
    /schedules/schedule_f/?committee_id=...
    Fewer records than IEs.

  *** DEMO_KEY has 30 req/hr — it exhausts after ~3 pages of IE data. ***
  *** Use --skip-party-spend to run on candidate committee data only,  ***
  *** or register a free key: https://api.open.fec.gov/developers      ***

  Registered key: 1,000 req/hr — handles all cycles without throttling.

Usage
─────
    # Candidate committee totals + Census CVAP (no API key needed):
    python scripts/fetch_data.py --skip-party-spend

    # Full run with registered FEC API key (party IEs + coordinated):
    python scripts/fetch_data.py --fec-api-key YOUR_KEY

    python scripts/fetch_data.py --only fec --cycles 2024
    python scripts/fetch_data.py --only census
    python scripts/fetch_data.py --only incumbency --cycles 2024

Manual data required (not available via API)
────────────────────────────────────────────
  MIT MEDSL House results:
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IG0UN2
    → data/raw/mit_elections/house_results_2012_2024.csv

  Cook PVI + ratings (proprietary):
    → data/raw/cook_pvi/cook_pvi_{cycle}.csv  (columns: district_id, pvi_raw)
    → data/raw/cook_pvi/cook_ratings_2024.csv (columns: district_id, rating)
"""

from __future__ import annotations
import argparse
import io
import logging
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("fetch_data")

FEC_API_BASE = "https://api.open.fec.gov/v1"
FEC_BULK_BASE = "https://www.fec.gov/files/bulk-downloads"

DCCC_COMMITTEE_ID = "C00000935"
# Corrected 2026-07-23 (codebase audit): the previous value "C00075473" is not
# the NRCC -- it's "CMS ENERGY CORPORATION EMPLOYEES FOR BETTER GOVERNMENT",
# an unrelated corporate PAC (verified against fec.gov/data/committee/C00075473/).
# The NRCC's actual FEC ID, per fec.gov/data/committee/C00075820/, is below.
# Any independent_expenditures_*.csv / coordinated_expenditures_*.csv fetched
# under the old ID must be re-fetched -- see FINDINGS.md for the affected cycles.
NRCC_COMMITTEE_ID = "C00075820"
DNC_COMMITTEE_ID = "C00010603"
DSCC_COMMITTEE_ID = "C00042366"
# Verified 2026-08-11 directly against data/raw/committee_master/cm*.txt
# (same standard as the NRCC correction above) while building the R-side
# mirror of identify_state_dem_party_committees() -- RNC's name is literally
# "REPUBLICAN NATIONAL COMMITTEE" (cmte_pty_affiliation=REP); NRSC's is
# literally "NRSC" (not spelled out, matching DSCC's own convention), and is
# NOT the same committee as the several unrelated Senate-candidate/PAC rows
# that also contain "NATIONAL REPUBLICAN SENATORIAL" as free text.
RNC_COMMITTEE_ID = "C00003418"
NRSC_COMMITTEE_ID = "C00027466"


FIPS_TO_STATE = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}


# ─── Bulk download helpers ────────────────────────────────────────────────────

def _download_zip(url: str, member_name: str) -> bytes:
    """Download a ZIP from FEC and return the bytes of a named member."""
    import requests
    logger.info(f"Downloading {url}…")
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    content = b"".join(resp.iter_content(chunk_size=1 << 20))
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = zf.namelist()
        # The zip may contain just one file, or a file with the same stem
        match = next((n for n in names if member_name.lower() in n.lower()), names[0])
        logger.info(f"  Extracting {match} ({len(content) // 1024:,} KB zip)")
        return zf.read(match)


def _cycle_to_yy(cycle: int) -> str:
    return str(cycle)[-2:]


# ─── Tier 1: Bulk candidate committee totals ──────────────────────────────────

def _parse_weball_bytes(wb_bytes: bytes) -> "pd.DataFrame":
    """
    Parse raw weball pipe-delimited bytes into a DataFrame of House candidates.

    weball column layout (0-indexed, verified against FEC bulk spec):
      col  0: CAND_ID              H… = House, S… = Senate, P… = President
      col  1: CAND_NAME
      col  2: CAND_ICI             I=Incumbent, C=Challenger, O=Open
      col  3: PTY_CD               numeric party code
      col  4: CAND_PTY_AFFILIATION DEM, REP, …
      col  5: TTL_RECEIPTS         total receipts
      col  7: TTL_DISB             ← total disbursements (spend)
      col  9: COH_BOP              cash on hand beginning of period
      col 10: COH_COP              cash on hand close of period
      col 17: TTL_INDIV_CONTRIB    total individual contributions received
      col 18: CAND_OFFICE_ST       state abbreviation
      col 19: CAND_OFFICE_DISTRICT two-digit district number
    """
    import pandas as pd
    import io as _io
    df = pd.read_csv(
        _io.BytesIO(wb_bytes), sep="|", header=None,
        names=list(range(31)), dtype=str, on_bad_lines="skip",
    )
    return df[df[0].str.startswith("H", na=False)].copy()


def _weball_to_disbursements(house: "pd.DataFrame", cycle: int) -> "pd.DataFrame":
    """Convert parsed weball House rows to the candidate_disbursements schema."""
    import pandas as pd
    ici_map = {"I": "Incumbent", "C": "Challenger", "O": "Open seat"}
    # Map FEC party affiliations to D/R.
    # DFL = Democratic-Farmer-Labor (Minnesota's Democratic party).
    # WFP = Working Families Party (NY/CT/etc., nominates Democratic candidates).
    party_map = {
        "DEM": "D", "DFL": "D", "WFP": "D",   # Democratic-aligned
        "REP": "R", "CON": "R",                 # Republican-aligned (CON = NY Conservative)
    }

    raw_party = house[4].str.strip()
    mapped_party = raw_party.map(party_map)
    # Any unmapped code stays as-is (e.g., IND, LIB, GRE) — filtered out later
    mapped_party = mapped_party.fillna(raw_party)

    ttl_receipts = pd.to_numeric(house[5], errors="coerce").fillna(0)
    ttl_indiv = pd.to_numeric(house[17], errors="coerce").fillna(0)

    out = pd.DataFrame({
        "fec_candidate_id":        house[0].str.strip(),
        "candidate_name":          house[1].str.strip(),
        "incumbent_challenge_full": house[2].str.strip().map(ici_map).fillna("Open seat"),
        "party":                   mapped_party,
        "state":                   house[18].str.strip(),
        "district_num":            house[19].str.strip().str.zfill(2),
        "candidate_disbursements": pd.to_numeric(house[7], errors="coerce").fillna(0),
        "ttl_receipts":            ttl_receipts,
        "ttl_indiv_contrib":       ttl_indiv,
        "cycle":                   cycle,
    })
    out["district_id"] = out["state"] + "-" + out["district_num"]
    # indiv_share: fraction of receipts from individual donors (0–1)
    out["indiv_share"] = (
        (ttl_indiv / ttl_receipts.replace(0, float("nan")))
        .clip(0.0, 1.0)
        .fillna(0.0)
    )
    return out.drop(columns=["state", "district_num"])


def fetch_candidate_totals_local(cycle: int, force: bool = False) -> bool:
    """
    Build candidate_disbursements_{cycle}.csv from a locally cached weball file.

    Reads from data/raw/bulk_all/weball{yy}.txt or
    data/raw/house_senate_current_campaigns/webl{yy}.txt, whichever exists.
    Returns True if the output was written, False if skipped.
    """
    import pandas as pd

    out_path = config.raw_path("fec") / f"candidate_disbursements_{cycle}.csv"
    if out_path.exists() and not force:
        logger.info(f"Candidate totals {cycle}: already present, skipping")
        return False

    yy = _cycle_to_yy(cycle)
    local_paths = [
        Path(__file__).parent.parent / "data" / "raw" / "bulk_all" / f"weball{yy}.txt",
        Path(__file__).parent.parent / "data" / "raw" / "house_senate_current_campaigns" / f"webl{yy}.txt",
    ]
    # Resolve: use the first existing local file
    local_file = None
    for p in local_paths:
        if p.exists():
            local_file = p
            break

    if local_file is None:
        logger.info(f"No local bulk file for {cycle}; will download")
        return False

    logger.info(f"Reading candidate totals for {cycle} from local file: {local_file.name}")
    with open(local_file, "rb") as f:
        wb_bytes = f.read()

    house = _parse_weball_bytes(wb_bytes)
    logger.info(f"  {len(house)} House candidate rows in {local_file.name}")
    out = _weball_to_disbursements(house, cycle)

    out[[
        "district_id", "fec_candidate_id", "candidate_name", "party",
        "cycle", "candidate_disbursements", "incumbent_challenge_full",
        "ttl_receipts", "ttl_indiv_contrib", "indiv_share",
    ]].to_csv(out_path, index=False)
    logger.info(f"Saved {len(out)} House candidates → {out_path}")
    return True


def fetch_candidate_totals_bulk(cycle: int, force: bool = False) -> None:
    """
    Download and parse FEC bulk weball file to produce
    candidate_disbursements_{cycle}.csv with no API key required.

    Prefers local weball file (data/raw/bulk_all/ or
    data/raw/house_senate_current_campaigns/) over download when available.

    Output schema:
        district_id, fec_candidate_id, candidate_name, party, cycle,
        candidate_disbursements, incumbent_challenge_full,
        ttl_receipts, ttl_indiv_contrib, indiv_share
    """
    import pandas as pd

    out_path = config.raw_path("fec") / f"candidate_disbursements_{cycle}.csv"
    if out_path.exists() and not force:
        logger.info(f"Candidate totals {cycle}: already present, skipping")
        return

    # Prefer local file over download
    if fetch_candidate_totals_local(cycle, force=force):
        return

    yy = _cycle_to_yy(cycle)
    wb_bytes = _download_zip(f"{FEC_BULK_BASE}/{cycle}/weball{yy}.zip", f"weball{yy}.txt")
    house = _parse_weball_bytes(wb_bytes)
    logger.info(f"  {len(house)} House candidate rows in weball{yy}")
    out = _weball_to_disbursements(house, cycle)

    out[[
        "district_id", "fec_candidate_id", "candidate_name", "party",
        "cycle", "candidate_disbursements", "incumbent_challenge_full",
        "ttl_receipts", "ttl_indiv_contrib", "indiv_share",
    ]].to_csv(out_path, index=False)
    logger.info(f"Saved {len(out)} House candidates → {out_path}")


# ─── FEC API helpers ──────────────────────────────────────────────────────────

def _fec_get(session, endpoint: str, params: dict, timeout: int = 60) -> dict:
    """Single FEC API GET with retry on rate-limit (429) and server errors."""
    url = f"{FEC_API_BASE}/{endpoint.lstrip('/')}"
    for attempt in range(5):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                if attempt >= 2:
                    raise RuntimeError(
                        "FEC API rate limit exceeded after 3 attempts. "
                        "Options: (a) register a free key at https://api.open.fec.gov/developers "
                        "and pass --fec-api-key YOUR_KEY, "
                        "or (b) add --skip-party-spend to run on candidate committee data only."
                    )
                wait = 60
                logger.warning(f"Rate limited. Sleeping {wait}s…")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = 10 * (attempt + 1)
                logger.warning(f"Server error {resp.status_code}. Retrying in {wait}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(1.5)  # polite pause; stays within DEMO_KEY limit for small pulls
            return resp.json()
        except RuntimeError:
            raise  # don't retry our own bailout errors (rate limit, etc.)
        except Exception as e:
            if attempt == 4:
                raise
            wait = 10 * (attempt + 1)
            logger.warning(f"Request error ({e}). Retrying in {wait}s…")
            time.sleep(wait)
    raise RuntimeError("FEC API request failed after 5 attempts")


def _fec_paginate(session, endpoint: str, params: dict) -> list[dict]:
    """Paginate through all pages of a FEC API endpoint."""
    all_results: list[dict] = []
    page = 1
    while True:
        data = _fec_get(session, endpoint, {**params, "page": page, "per_page": 100})
        results = data.get("results", [])
        all_results.extend(results)
        pagination = data.get("pagination", {})
        total_pages = pagination.get("pages", 1)
        logger.info(f"  Page {page}/{total_pages} ({len(all_results)} records)")
        if page >= total_pages:
            break
        page += 1
    return all_results


# ─── Tier 2: DCCC / NRCC IEs via API (filtered → small result set) ───────────

def fetch_ie_by_committee(cycle: int, api_key: str, committee_id: str, party: str) -> None:
    """
    Fetch independent expenditures made by DCCC or NRCC for a cycle.

    Filtered to a single committee, so ~1,000–3,000 rows (10–30 pages).
    This is manageable even with DEMO_KEY.

    Output schema: district_id, party, cycle, support_oppose, amount
    """
    import requests
    import pandas as pd

    label = "DCCC" if committee_id == DCCC_COMMITTEE_ID else "NRCC"
    out_path = config.raw_path("fec") / f"ie_{label.lower()}_{cycle}.csv"
    if out_path.exists():
        logger.info(f"IE {label} {cycle}: already present, skipping")
        return

    logger.info(f"Fetching {label} IEs for {cycle} via API…")
    with requests.Session() as session:
        records = _fec_paginate(session, "/schedules/schedule_e/", {
            "api_key":      api_key,
            "committee_id": committee_id,
            "cycle":        cycle,
            "sort":         "-expenditure_date",
        })

    if not records:
        logger.warning(f"No IE records for {label} {cycle}")
        pd.DataFrame(columns=["district_id", "party", "cycle", "support_oppose", "amount"]
                     ).to_csv(out_path, index=False)
        return

    df = pd.DataFrame(records)
    df["state"]       = df.get("candidate_office_state", "").fillna("")
    df["district_num"]= df.get("candidate_office_district", "00").fillna("00").astype(str).str.zfill(2)
    df["district_id"] = df["state"] + "-" + df["district_num"]
    df["support_oppose"] = df.get("support_oppose_indicator", "S").fillna("S")
    df["amount"]      = pd.to_numeric(df.get("expenditure_amount", 0), errors="coerce").fillna(0)
    df["party"]       = party
    df["cycle"]       = cycle

    df[["district_id", "party", "cycle", "support_oppose", "amount"]].to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} IE transactions → {out_path}")


def fetch_coordinated_by_committee(cycle: int, api_key: str, committee_id: str, party: str) -> None:
    """
    Fetch Schedule F coordinated party expenditures by DCCC or NRCC.
    Output schema: district_id, party, cycle, coordinated_expenditures
    """
    import requests
    import pandas as pd

    label = "DCCC" if committee_id == DCCC_COMMITTEE_ID else "NRCC"
    out_path = config.raw_path("fec") / f"coordinated_{label.lower()}_{cycle}.csv"
    if out_path.exists():
        logger.info(f"Coordinated {label} {cycle}: already present, skipping")
        return

    logger.info(f"Fetching {label} coordinated expenditures for {cycle} via API…")
    with requests.Session() as session:
        records = _fec_paginate(session, "/schedules/schedule_f/", {
            "api_key":          api_key,
            "committee_id":     committee_id,
            "cycle":            cycle,
            "candidate_office": "H",
        })

    if not records:
        logger.warning(f"No coordinated expenditure records for {label} {cycle}")
        pd.DataFrame(columns=["district_id", "party", "cycle", "coordinated_expenditures"]
                     ).to_csv(out_path, index=False)
        return

    df = pd.DataFrame(records)
    df["state"]       = df.get("candidate_office_state", "").fillna("")
    df["district_num"]= df.get("candidate_office_district", "00").fillna("00").astype(str).str.zfill(2)
    df["district_id"] = df["state"] + "-" + df["district_num"]
    df["amount"]      = pd.to_numeric(df.get("expenditure_amount", 0), errors="coerce").fillna(0)
    df["party"]       = party
    df["cycle"]       = cycle

    out = (
        df.groupby(["district_id", "party", "cycle"])["amount"]
        .sum().reset_index()
        .rename(columns={"amount": "coordinated_expenditures"})
    )
    out.to_csv(out_path, index=False)
    logger.info(f"Saved {len(out)} districts → {out_path}")


# ─── Tier 2: dated candidate-committee periodic reports (Paper III, new) ─────
# Distinct from candidate_disbursements_{cycle}.csv (weball bulk file,
# cycle-cumulative-final TTL_DISB only): this is the genuinely dated
# per-filing-period panel that docs/theta_followup_plan.md Section 0.1.1
# documented as "no per-filing date field anywhere in this repository" --
# that claim was true of the bulk files this project used, but not of the
# FEC API's /committee/{id}/reports/ endpoint, confirmed against live data
# this session, not assumed from documentation.

def _load_candidate_committee_crosswalk(cycle: int) -> dict[str, str]:
    """
    Map House CAND_ID -> principal campaign committee ID for a cycle, from
    the raw candidate-committee-linkage bulk file(s)
    (data/raw/candidate_committee_linkage/ccl*.txt -- downloaded per cycle
    with inconsistent filenames, e.g. "ccl.txt", "ccl 2.txt"; this reads
    every ccl*.txt present and filters by the FEC_ELECTION_YR column rather
    than trusting the filename).

    Raw schema (pipe-delimited, no header): CAND_ID, CAND_ELECTION_YR,
    FEC_ELECTION_YR, CMTE_ID, CMTE_TSPE, CMTE_DSGN, LINKAGE_ID.

    Filters to CMTE_DSGN == "P" (principal campaign committee -- a
    candidate can have multiple linked committees, e.g. joint fundraisers;
    only the principal committee files the Form 3 periodic reports needed
    here) and CAND_ID starting with "H" (House). A candidate can appear
    more than once if their principal-committee linkage was amended
    mid-cycle; kept the highest LINKAGE_ID (most recently created linkage
    row) as an approximation of "most current," not a resolved amendment
    chain like load_ie_transactions_dated's file_num/prev_file_num logic.
    """
    import pandas as pd

    ccl_dir = config.raw_path("candidate_committee_linkage")
    cols = ["cand_id", "cand_election_yr", "fec_election_yr", "cmte_id", "cmte_tspe", "cmte_dsgn", "linkage_id"]
    paths = sorted(ccl_dir.glob("ccl*.txt")) if ccl_dir.exists() else []
    if not paths:
        raise FileNotFoundError(f"No ccl*.txt files found in {ccl_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path, sep="|", header=None, names=cols, dtype=str)
        frames.append(df[df["fec_election_yr"] == str(cycle)])
    all_df = pd.concat(frames, ignore_index=True)
    principal = all_df[
        (all_df["cmte_dsgn"] == "P") & (all_df["cand_id"].str.startswith("H", na=False))
    ].copy()
    principal["linkage_id"] = pd.to_numeric(principal["linkage_id"], errors="coerce")
    principal = principal.sort_values("linkage_id").drop_duplicates("cand_id", keep="last")
    return dict(zip(principal["cand_id"], principal["cmte_id"]))


# ─── State-party 24K coordinated expenditures (FINDINGS.md Section 10.7, Gap 3) ──
#
# FINDINGS.md previously documented the raw bulk file at
# data/raw/bulk_all/itoth.txt -- that path is wrong (bulk_all/ only holds
# weball*.txt candidate-totals files). The real file lives at
# data/raw/all_committee_transactions/itoth.txt (167MB, ~1.01M rows), with
# 7 additional undocumented sibling files (itoth 2.txt ... itoth 8.txt,
# 90MB-3.3GB) covering other, OVERLAPPING date ranges -- verified directly
# against the files (a sampled per-file year scan), not assumed from
# filenames, which give no indication of coverage (the same numbered-
# sibling unreliability _load_candidate_committee_crosswalk already works
# around for ccl*.txt).

_ITOTH_COLUMNS = [
    "cmte_id", "amndt_ind", "rpt_tp", "transaction_pgi", "image_num", "transaction_tp",
    "entity_tp", "name", "city", "state", "zip", "employer", "occupation", "transaction_dt",
    "transaction_amt", "other_id", "tran_id", "file_num", "memo_cd", "memo_text", "sub_id",
]

_CN_COLUMNS = [
    "cand_id", "cand_name", "cand_pty_affiliation", "cand_election_yr", "cand_office_st",
    "cand_office", "cand_office_district", "cand_ici", "cand_status", "cand_pcc",
    "cand_st1", "cand_st2", "cand_city", "cand_state", "cand_zip4",
]

# The CMTE_TP/CMTE_DSGN/CMTE_PTY_AFFILIATION filter alone is far too loose --
# verified directly against the real committee_master data (2026-08): it
# also passes 235 committees including town/county Democratic committees
# ("KENNEBUNKPORT DEMOCRATIC COMMITTEE", ME), party caucuses ("AFRICAN
# AMERICAN CAUCUS OF THE NORTH CAROLINA DEMOCRATIC PARTY"), GOTV/PAC-style
# committees ("RICO DEMOCRATIC GOTV"), and legislative-district clubs
# ("70TH ASSEMBLY DISTRICT DEMOCRATIC COALITION TASK FORCE", CA) -- none of
# which are the actual state party. A blocklist-only approach (excluding
# sub-state name patterns) is insufficient on its own, since it still
# passed e.g. "IDP5 FEDERAL" and "ORLEANS DEMOCRATIC EXECUTIVE COMMITTEE
# FEDERAL" (Orleans Parish, LA -- a county-equivalent with no "PARISH" in
# its own name). The precise, whitelist-style check below instead requires
# a committee to (a) spell out its OWN registered state's full name inside
# its own committee name, cross-checked against CMTE_ST rather than just
# grepped -- real state parties consistently do this
# ("NEBRASKA DEMOCRATIC PARTY", "DEMOCRATIC PARTY OF SOUTH CAROLINA"),
# sub-state committees almost never do ("KENNEBUNKPORT..." never spells
# "MAINE"; "IDP5 FEDERAL" never spells "IOWA") -- AND (b) use one of the
# small set of structural suffixes real state parties actually register
# under. Both conditions were spot-checked to correctly admit every real
# state party in the sample above and correctly reject every false
# positive found, before being adopted here.
_FULL_STATE_NAMES = {
    "AL": "ALABAMA", "AK": "ALASKA", "AZ": "ARIZONA", "AR": "ARKANSAS", "CA": "CALIFORNIA",
    "CO": "COLORADO", "CT": "CONNECTICUT", "DE": "DELAWARE", "DC": "DISTRICT OF COLUMBIA",
    "FL": "FLORIDA", "GA": "GEORGIA", "HI": "HAWAII", "ID": "IDAHO", "IL": "ILLINOIS",
    "IN": "INDIANA", "IA": "IOWA", "KS": "KANSAS", "KY": "KENTUCKY", "LA": "LOUISIANA",
    "ME": "MAINE", "MD": "MARYLAND", "MA": "MASSACHUSETTS", "MI": "MICHIGAN", "MN": "MINNESOTA",
    "MS": "MISSISSIPPI", "MO": "MISSOURI", "MT": "MONTANA", "NE": "NEBRASKA", "NV": "NEVADA",
    "NH": "NEW HAMPSHIRE", "NJ": "NEW JERSEY", "NM": "NEW MEXICO", "NY": "NEW YORK",
    "NC": "NORTH CAROLINA", "ND": "NORTH DAKOTA", "OH": "OHIO", "OK": "OKLAHOMA", "OR": "OREGON",
    "PA": "PENNSYLVANIA", "RI": "RHODE ISLAND", "SC": "SOUTH CAROLINA", "SD": "SOUTH DAKOTA",
    "TN": "TENNESSEE", "TX": "TEXAS", "UT": "UTAH", "VT": "VERMONT", "VA": "VIRGINIA",
    "WA": "WASHINGTON", "WV": "WEST VIRGINIA", "WI": "WISCONSIN", "WY": "WYOMING",
}

_STATE_PARTY_SUFFIX_PATTERN = (
    r"DEMOCRATIC[- ](?:PARTY|STATE CENTRAL COMMITTEE|EXECUTIVE COMMITTEE|"
    r"CENTRAL EXECUTIVE COMMITTEE|FARMER-LABOR PARTY|NONPARTISAN LEAGUE PARTY|"
    r"STATE COMMITTEE|STATE CMTE)|"
    r"STATE DEMOCRATIC (?:CENTRAL |EXECUTIVE )?COMMITTEE|"
    r"(?:STATE CENTRAL|STATE EXECUTIVE) COMMITTEE.*DEMOCRAT"
)

# A handful of real state party committees, individually verified directly
# against data/raw/committee_master/cm*.txt (2026-08), register under names
# no DEMOCRATIC/PARTY text pattern could plausibly catch (Georgia's federal
# party committee is literally named "GEORGIA FEDERAL ELECTIONS COMMITTEE",
# with no "DEMOCRATIC" or "PARTY" anywhere in it) or spell their state only
# as a two-letter abbreviation rather than the full name ("DEMOCRATIC STATE
# CENTRAL COMMITTEE OF LA"; "WV STATE DEMOCRATIC EXECUTIVE COMMITTEE") --
# whole-word abbreviation matching was deliberately NOT added to
# has_own_state_name below to catch these generically, since several state
# abbreviations (IN, OR, ME, HI, PA, ...) are also common English words and
# would produce real false positives elsewhere in the committee master.
# Same manual, individually-verified-exception pattern this project already
# uses for live_cycle_ballot_exclusions (config.yaml) -- add here only,
# never inferred.
_MANUAL_STATE_PARTY_COMMITTEE_IDS = {
    "GA": "C00041269",  # GEORGIA FEDERAL ELECTIONS COMMITTEE
    "LA": "C00071365",  # DEMOCRATIC STATE CENTRAL COMMITTEE OF LA
    "WV": "C00162578",  # WV STATE DEMOCRATIC EXECUTIVE COMMITTEE / W VA STATE DEMOCRATIC EX COM
    "WY": "C00001917",  # WY DEMOCRATIC STATE CENTRAL COMMITTEE
}
# Indiana has no committee this heuristic (structural or manual) covers with
# confidence -- its closest candidate, "INDIANA DEMOCRATIC CONGRESSIONAL
# VICTORY COMMITTEE" (C00108613), is a joint-fundraising-style "victory"
# committee, not unambiguously the state party itself (unlike the VICTORY
# FUND entries that appear only as OTHER state parties' connected_org_nm,
# i.e. a fund THEY control, not their own registration). Left out rather
# than guessed -- Indiana's state-party 24K coordinated spend (if any) is
# a known, documented gap in this function's coverage, not silently assumed
# zero or silently included on a weak match.

# ─── R-side mirror, added 2026-08-11 ──────────────────────────────────────────
# Until this point, the state-party 24K scan (parse_state_party_coordinated_24k
# below) only ever covered Democratic state parties -- flagged in FINDINGS.md
# Section 10.7 (Gap 3) as future work and never closed for the R side. Left
# unaddressed, R's spending total (r_total, and therefore cand_r_total /
# party_r / budget_r throughout src/game/) was a documented undercount
# relative to D's -- a real asymmetry in a project whose whole premise is
# treating both sides symmetrically, not a fixed benchmark like the mirrored
# persuasion ceiling.
_STATE_PARTY_SUFFIX_PATTERN_REP = (
    r"REPUBLICAN[- ](?:PARTY|STATE CENTRAL COMMITTEE|EXECUTIVE COMMITTEE|"
    r"CENTRAL EXECUTIVE COMMITTEE|STATE COMMITTEE|STATE CMTE)|"
    r"STATE REPUBLICAN (?:CENTRAL |EXECUTIVE )?COMMITTEE|"
    r"(?:STATE CENTRAL|STATE EXECUTIVE) COMMITTEE.*REPUBLICAN"
)
# Structural pattern alone (same has_own_state_name + suffix + ~is_substate
# check as the Dem side) matches 43 of 50 states cleanly. The remaining 7,
# each individually verified directly against committee_master (same
# standard as the Dem-side manual list): all are cmte_dsgn="U",
# cmte_tp="Y" (qualified party committee, matching every automatic match's
# own designation) with no other Y-type REP committee registered in that
# state, but a name shape the suffix regex doesn't (and, per the Dem-side
# comment above, deliberately doesn't try to) catch generically:
_MANUAL_STATE_PARTY_COMMITTEE_IDS_REP = {
    "CO": "C00033134",  # COLORADO REPUBLICAN COMMITTEE (no PARTY/STATE COMMITTEE suffix)
    "NM": "C00020818",  # REPUBLICAN CAMPAIGN COMMITTEE OF NEW MEXICO
    "NV": "C00082925",  # NEVADA REPUBLICAN CENTRAL COMMITTEE (missing "STATE")
    "NY": "C00055582",  # NY REPUBLICAN FEDERAL CAMPAIGN COMMITTEE (abbreviated state, mirrors WV on the Dem side)
    "OK": "C00167213",  # OKLAHOMA LEADERSHIP COUNCIL (Oklahoma GOP's actual FEC-registered name)
    "PA": "C00044842",  # REPUBLICAN FEDERAL COMMITTEE OF PENNSYLVANIA (mirrors GA's "FEDERAL ELECTIONS COMMITTEE" shape on the Dem side)
    "VT": "C00035618",  # VERMONT REPUBLICAN FEDERAL ELECTIONS COMMITTEE
}
# Unlike the Dem side, Indiana IS covered automatically here ("INDIANA
# REPUBLICAN STATE COMMITTEE, INC." matches the structural pattern) -- no
# manual entry needed, and no equivalent gap on this side.

# Belt-and-suspenders blocklist -- catches the rare case where a genuinely
# sub-state committee happens to also spell out its state's full name (e.g.
# a county committee named "X COUNTY CALIFORNIA DEMOCRATIC PARTY"), plus two
# patterns found only after broadening the suffix regex above: "DISTRICT"
# (bare, not just "CONGRESSIONAL DISTRICT" -- state legislative/party
# sub-districts like "SIXTH DISTRICT DEMOCRATIC PARTY OF WISCONSIN" also
# spell out the full state name and would otherwise pass) and "TRUST" (joint
# fundraising vehicles like "CALIFORNIA STATE OF THE UNION DEMOCRATIC PARTY
# TRUST" are not the state party committee itself).
_SUBSTATE_NAME_PATTERNS = ("COUNTY", "DISTRICT", "TOWNSHIP", "PRECINCT", "WARD ", "PARISH",
                           "CLUB", "CAUCUS", "COALITION", "TASK FORCE", " CITY ", "TRUST")


def _load_committee_master() -> pd.DataFrame:
    """Map every committee to its type/designation/party, from the raw
    committee-master bulk file(s) (data/raw/committee_master/cm*.txt --
    same inconsistent-numbered-filename situation as ccl*.txt, so every
    cm*.txt present is read, not just cm.txt).

    Raw schema (pipe-delimited, no header): CMTE_ID, CMTE_NM, TRES_NM,
    CMTE_ST1, CMTE_ST2, CMTE_CITY, CMTE_ST, CMTE_ZIP, CMTE_DSGN, CMTE_TP,
    CMTE_PTY_AFFILIATION, CMTE_FILING_FREQ, ORG_TP, CONNECTED_ORG_NM, CAND_ID."""
    import pandas as pd

    cm_dir = config.raw_path("committee_master")
    cols = ["cmte_id", "cmte_nm", "tres_nm", "cmte_st1", "cmte_st2", "cmte_city", "cmte_st",
            "cmte_zip", "cmte_dsgn", "cmte_tp", "cmte_pty_affiliation", "cmte_filing_freq",
            "org_tp", "connected_org_nm", "cand_id"]
    paths = sorted(cm_dir.glob("cm*.txt")) if cm_dir.exists() else []
    if not paths:
        raise FileNotFoundError(f"No cm*.txt files found in {cm_dir}")
    frames = [pd.read_csv(p, sep="|", header=None, names=cols, dtype=str) for p in paths]
    all_df = pd.concat(frames, ignore_index=True)
    return all_df.drop_duplicates(subset=["cmte_id"], keep="last")


def identify_state_dem_party_committees(exclude_national: set[str] | None = None) -> "pd.DataFrame":
    """State-level Democratic party committees: CMTE_TP in {X (non-qualified
    party), Y (qualified party)}, CMTE_DSGN == "U" (unauthorized -- the
    designation party committees themselves use, distinct from a candidate's
    own authorized/principal committee), CMTE_PTY_AFFILIATION == "DEM",
    excluding known national committees and sub-state (county/congressional-
    district-level) committees by name pattern.

    Not guaranteed complete or precise -- verified directly against the real
    committee_master data (2026-08): correctly identifies 48 of 50 states'
    Democratic party committees (49 including the manual-override list's WY/
    GA/LA/WV entries above, which the structural pattern alone cannot catch)
    and correctly excludes every sub-state false positive found during
    development (town/county committees, congressional- and state-legislative-
    district party organizations, caucuses, clubs, joint-fundraising trusts --
    an earlier, looser version of this filter incorrectly admitted 235
    committees before the whitelist-style state-name + suffix-pattern check
    and the manual-override list were added). Indiana has no committee this
    function covers with confidence (see _MANUAL_STATE_PARTY_COMMITTEE_IDS's
    comment) and DC is out of scope for this project's House-race universe
    regardless. Errs toward excluding an ambiguous committee rather than
    including one, matching this project's standing preference for an honest
    undercount over a fabricated-precision overcount (see e.g. Gap 2's
    committee-ID correction in this same section of FINDINGS.md)."""
    import pandas as pd

    if exclude_national is None:
        exclude_national = {DCCC_COMMITTEE_ID, DNC_COMMITTEE_ID, DSCC_COMMITTEE_ID}
    cm = _load_committee_master()
    is_party_committee = cm["cmte_tp"].isin(["X", "Y"]) & (cm["cmte_dsgn"] == "U")
    is_dem = cm["cmte_pty_affiliation"] == "DEM"
    not_national = ~cm["cmte_id"].isin(exclude_national)

    name_upper = cm["cmte_nm"].fillna("").str.upper()
    state_full = cm["cmte_st"].map(_FULL_STATE_NAMES).fillna("")
    has_own_state_name = pd.Series(
        [bool(s) and s in n for s, n in zip(state_full, name_upper)], index=cm.index,
    )
    has_party_suffix = name_upper.str.contains(_STATE_PARTY_SUFFIX_PATTERN, na=False, regex=True)
    is_substate = name_upper.str.contains("|".join(_SUBSTATE_NAME_PATTERNS), na=False, regex=True)

    candidates = cm[
        is_party_committee & is_dem & not_national
        & has_own_state_name & has_party_suffix & ~is_substate
    ].copy()
    manual = cm[cm["cmte_id"].isin(_MANUAL_STATE_PARTY_COMMITTEE_IDS.values())].copy()
    candidates = pd.concat([candidates, manual], ignore_index=True).drop_duplicates(subset=["cmte_id"])
    return candidates[["cmte_id", "cmte_nm", "cmte_st"]].rename(
        columns={"cmte_id": "committee_id", "cmte_nm": "committee_name", "cmte_st": "state"}
    )


def identify_state_rep_party_committees(exclude_national: set[str] | None = None) -> "pd.DataFrame":
    """State-level Republican party committees -- mirror of
    identify_state_dem_party_committees() above (same CMTE_TP/CMTE_DSGN
    filter, same name-shape heuristic), added 2026-08-11 to close the R-side
    gap in the state-party 24K coordinated-spending scan (see that
    function's own docstring and _MANUAL_STATE_PARTY_COMMITTEE_IDS_REP's
    comment for the verification history).

    Structural pattern + manual list together identify all 50 states'
    Republican party committees (unlike the Dem side, no state is left
    uncovered here -- Indiana, the Dem side's one gap, has an unambiguous
    structural match on this side)."""
    import pandas as pd

    if exclude_national is None:
        exclude_national = {NRCC_COMMITTEE_ID, RNC_COMMITTEE_ID, NRSC_COMMITTEE_ID}
    cm = _load_committee_master()
    is_party_committee = cm["cmte_tp"].isin(["X", "Y"]) & (cm["cmte_dsgn"] == "U")
    is_rep = cm["cmte_pty_affiliation"] == "REP"
    not_national = ~cm["cmte_id"].isin(exclude_national)

    name_upper = cm["cmte_nm"].fillna("").str.upper()
    state_full = cm["cmte_st"].map(_FULL_STATE_NAMES).fillna("")
    has_own_state_name = pd.Series(
        [bool(s) and s in n for s, n in zip(state_full, name_upper)], index=cm.index,
    )
    has_party_suffix = name_upper.str.contains(_STATE_PARTY_SUFFIX_PATTERN_REP, na=False, regex=True)
    is_substate = name_upper.str.contains("|".join(_SUBSTATE_NAME_PATTERNS), na=False, regex=True)

    candidates = cm[
        is_party_committee & is_rep & not_national
        & has_own_state_name & has_party_suffix & ~is_substate
    ].copy()
    manual = cm[cm["cmte_id"].isin(_MANUAL_STATE_PARTY_COMMITTEE_IDS_REP.values())].copy()
    candidates = pd.concat([candidates, manual], ignore_index=True).drop_duplicates(subset=["cmte_id"])
    return candidates[["cmte_id", "cmte_nm", "cmte_st"]].rename(
        columns={"cmte_id": "committee_id", "cmte_nm": "committee_name", "cmte_st": "state"}
    )


def _itoth_file_year_range(path: Path) -> tuple[int, int]:
    """Cheap, O(1)-I/O sampled scan (5 seek points, not exhaustive) of one
    itoth*.txt bulk file's approximate TRANSACTION_DT year coverage, used
    only to decide which of the (up to 8, up to 3.3GB each) files are worth
    a full scan for a given cycle. Assumes each file's rows are roughly
    chronologically grouped (verified directly: a denser every-2000th-row
    sample of all 8 files at implementation time showed each file spanning
    a narrow, contiguous 2-6 year window, not scattered across decades) --
    callers should still treat the returned range as approximate (see
    parse_state_party_coordinated_24k's +/-1 year padding when deciding
    file relevance), not an exact bound."""
    size = path.stat().st_size
    if size == 0:
        return (0, 0)
    years: set[int] = set()
    with open(path, "rb") as f:
        for frac in (0.0, 0.25, 0.5, 0.75, 0.99):
            f.seek(int(size * frac))
            if frac > 0.0:
                f.readline()  # discard partial line from an arbitrary seek offset
            line = f.readline().decode("latin-1", errors="replace")
            fields = line.split("|")
            if len(fields) > 13:
                dt = fields[13].strip()
                if len(dt) == 8 and dt.isdigit():
                    years.add(int(dt[4:8]))
    return (min(years), max(years)) if years else (0, 0)


def _scan_itoth_file_for_24k(path: Path, cycle_years: set[int], chunksize: int = 200_000) -> "pd.DataFrame":
    """Chunked scan of one itoth*.txt bulk file (files up to 3.3GB, so this
    is never read into memory at once), filtered to TRANSACTION_TP == "24K"
    (FEC Schedule B line 23, "Coordinated Party Expenditures") and
    ENTITY_TP == "CCM" (recipient is a candidate's principal campaign
    committee, not another committee/PAC -- excludes the large majority of
    24K rows, which are committee-to-committee transfers with no single
    House race to attribute to), restricted to cycle_years via
    TRANSACTION_DT."""
    import pandas as pd

    keep = []
    for chunk in pd.read_csv(
        path, sep="|", header=None, names=_ITOTH_COLUMNS, dtype=str,
        chunksize=chunksize, encoding="latin-1", on_bad_lines="skip",
    ):
        mask = (chunk["transaction_tp"] == "24K") & (chunk["entity_tp"] == "CCM")
        sub = chunk[mask]
        if len(sub):
            years = pd.to_numeric(sub["transaction_dt"].str.strip().str[4:8], errors="coerce")
            sub = sub[years.isin(cycle_years)]
        if len(sub):
            keep.append(sub)
    if not keep:
        return pd.DataFrame(columns=_ITOTH_COLUMNS)
    return pd.concat(keep, ignore_index=True)


def parse_state_party_coordinated_24k(cycle: int, party: str = "D") -> "pd.DataFrame":
    """State party committees' 24K coordinated expenditures into House
    candidate committees, aggregated per district -- the data gap
    FINDINGS.md Section 10.7 (Gap 3) documented as "out of scope for this
    audit" and left as future work FOR THE D SIDE ONLY. Additive to, NOT a
    duplicate of, the existing DCCC/NRCC-only coordinated_{dccc,nrcc}_{cycle}.csv
    path (fetched via the FEC API against each committee's own ID) -- that
    committee's own 24K rows are explicitly excluded here to avoid
    double-counting.

    party: "D" or "R" (added 2026-08-11, generalizing the original D-only
    implementation so the SAME scan logic covers both sides -- see
    identify_state_rep_party_committees()'s docstring for why this closes a
    real, quantified asymmetry rather than being purely a code-cleanliness
    change).

    District attribution: 24K rows carry CMTE_ID (the filing party
    committee) and OTHER_ID (the recipient candidate committee) but no
    district directly, unlike the API-based coordinated_{dccc,nrcc} fetch
    (which gets candidate_office_state/district for free in the API
    response). Resolved here via the same CAND_ID <-> principal-committee
    crosswalk _load_candidate_committee_crosswalk() already builds from
    ccl*.txt for the candidate-periodic-reports fetch, inverted (committee
    -> candidate) and joined to candidate_master/cn*.txt for
    CAND_OFFICE_ST/CAND_OFFICE_DISTRICT.

    Output columns match coordinated_{dccc,nrcc}_{cycle}.csv's existing
    schema so consolidate_fec_files() can merge this in as a third source
    with no changes to src/backtest/data/fec.py's consumption path:
        district_id, party, cycle, coordinated_expenditures
    """
    import pandas as pd

    if party not in ("D", "R"):
        raise ValueError(f"party must be 'D' or 'R', got {party!r}")
    national_committee_id = DCCC_COMMITTEE_ID if party == "D" else NRCC_COMMITTEE_ID
    identify_fn = identify_state_dem_party_committees if party == "D" else identify_state_rep_party_committees
    label = "Dem" if party == "D" else "Rep"

    all_dir = config.raw_path("all_committee_transactions")
    paths = sorted(all_dir.glob("itoth*.txt")) if all_dir.exists() else []
    if not paths:
        raise FileNotFoundError(f"No itoth*.txt files found in {all_dir}")

    cycle_years = {cycle - 1, cycle}
    relevant_paths = []
    for p in paths:
        lo, hi = _itoth_file_year_range(p)
        if lo == 0 and hi == 0:
            continue
        if (lo - 1) <= max(cycle_years) and (hi + 1) >= min(cycle_years):
            relevant_paths.append(p)
    if not relevant_paths:
        logger.warning(
            f"24K state-party scan {cycle} ({label}): no itoth*.txt file's sampled date range "
            f"overlaps {sorted(cycle_years)} -- returning an empty result."
        )
        return pd.DataFrame(columns=["district_id", "party", "cycle", "coordinated_expenditures"])

    logger.info(f"24K state-party scan {cycle} ({label}): scanning {[p.name for p in relevant_paths]} "
                f"(cycle_years={sorted(cycle_years)})")
    frames = [_scan_itoth_file_for_24k(p, cycle_years) for p in relevant_paths]
    raw = pd.concat(frames, ignore_index=True)
    n_before_dedup = len(raw)
    raw = raw.drop_duplicates(subset=["sub_id"])   # dedup across overlapping-vintage files
    logger.info(f"24K state-party scan {cycle} ({label}): {n_before_dedup} raw 24K/CCM rows, "
                f"{len(raw)} after cross-file dedup on sub_id")

    state_committees = identify_fn()
    raw = raw[raw["cmte_id"] != national_committee_id]   # never double-count DCCC/NRCC's own coordinated spend
    raw = raw.merge(state_committees[["committee_id"]], left_on="cmte_id", right_on="committee_id", how="inner")
    logger.info(f"24K state-party scan {cycle} ({label}): {len(raw)} rows from an identified state {label} party committee")

    crosswalk = _load_candidate_committee_crosswalk(cycle)   # CAND_ID -> principal CMTE_ID
    committee_to_cand = {v: k for k, v in crosswalk.items()}   # inverted: CMTE_ID -> CAND_ID
    raw["cand_id"] = raw["other_id"].map(committee_to_cand)
    n_unmatched = int(raw["cand_id"].isna().sum())
    if n_unmatched:
        logger.warning(f"24K state-party scan {cycle} ({label}): {n_unmatched} row(s) had no matching House "
                        f"principal-committee crosswalk entry for OTHER_ID and were dropped.")
    raw = raw[raw["cand_id"].notna()]

    cn_dir = config.raw_path("candidate_master")
    cn_paths = sorted(cn_dir.glob("cn*.txt")) if cn_dir.exists() else []
    if not cn_paths:
        raise FileNotFoundError(f"No cn*.txt files found in {cn_dir}")
    cn = pd.concat(
        [pd.read_csv(p, sep="|", header=None, names=_CN_COLUMNS, dtype=str) for p in cn_paths],
        ignore_index=True,
    ).drop_duplicates(subset=["cand_id"], keep="last")
    cn["district_id"] = cn["cand_office_st"] + "-" + cn["cand_office_district"].fillna("00").str.zfill(2)
    raw = raw.merge(cn[["cand_id", "district_id"]], on="cand_id", how="left")
    n_no_district = int(raw["district_id"].isna().sum())
    if n_no_district:
        logger.warning(f"24K state-party scan {cycle} ({label}): {n_no_district} row(s) had a candidate_id "
                        f"with no candidate_master match and were dropped.")
    raw = raw[raw["district_id"].notna()]

    raw["transaction_amt"] = pd.to_numeric(raw["transaction_amt"], errors="coerce").fillna(0.0)
    out = (
        raw.groupby("district_id")["transaction_amt"].sum().reset_index()
        .rename(columns={"transaction_amt": "coordinated_expenditures"})
    )
    out["party"] = party
    out["cycle"] = cycle
    logger.info(f"24K state-party scan {cycle} ({label}): ${out['coordinated_expenditures'].sum():,.0f} "
                f"across {len(out)} districts")
    return out[["district_id", "party", "cycle", "coordinated_expenditures"]]


_PERIODIC_REPORT_COLUMNS = [
    "district_id", "party", "cycle", "fec_candidate_id", "committee_id",
    "coverage_start_date", "coverage_end_date", "receipts_period",
    "disbursements_period", "cash_on_hand_end_period", "report_type_full",
    "beginning_image_number",
]


def fetch_candidate_periodic_reports(cycle: int, api_key: str, force: bool = False) -> None:
    """
    Fetch dated, per-period Form 3 financial reports (quarterly + pre/post-
    general) for every House candidate's principal campaign committee in a
    cycle, via FEC API's /committee/{committee_id}/reports/ endpoint.

    One API call (paginated) per committee -- the endpoint returns every
    period for a committee/cycle in one call, not one call per report -- so
    this is roughly (n nominees from fec.load_candidate_disbursements(cycle))
    calls, NOT one per row of the raw candidate_disbursements_{cycle}.csv
    file. This distinction matters for correctness, not just speed: that raw
    file has one row per candidate INCLUDING primary losers (multiple D or R
    candidates can share a district), while
    src.backtest.data.fec.cumulative_candidate_spend_as_of groups by
    (district_id, party) and sums -- fetching every primary loser's
    committee too would silently double- or triple-count a district's real
    candidate spend. fec.load_candidate_disbursements(cycle) already applies
    the exact same top-spender-per-party-per-district nominee selection
    (with MIT-ballot cross-referencing) this project's static pipeline uses
    everywhere else, so fetching exactly that roster keeps the dated panel
    consistent with it and cuts the fetch volume roughly 3-4x for free
    (~1,750 nominees across 2022+2024 vs. ~6,400 raw candidate rows).
    Requires a registered key (1,000 req/hr); DEMO_KEY (40 req/hr) exhausts
    almost immediately at this volume.

    Checkpointed and resumable: results are appended to a `.partial.csv`
    file after every committee (not accumulated in memory and written once
    at the end) and renamed to the final path only once every candidate has
    been processed. A run that gets killed partway (observed in practice --
    a ~3,300-candidate cycle at this endpoint's pace takes over an hour, well
    past what a single long-running background process can be relied on to
    complete in one sitting) can simply be re-invoked: committees already
    present in the `.partial.csv` are skipped, not re-fetched.

    Amendments are NOT resolved here -- multiple report rows can share the
    same (coverage_start_date, coverage_end_date) if a filing was amended.
    Raw data is written as-is; src.backtest.data.fec.load_candidate_periodic_reports
    resolves amendments at load time (same division of responsibility as
    load_ie_transactions_dated's file_num/prev_file_num resolution).

    Output: candidate_periodic_reports_{cycle}.csv, columns:
        district_id, party, cycle, fec_candidate_id, committee_id,
        coverage_start_date, coverage_end_date, receipts_period,
        disbursements_period, cash_on_hand_end_period, report_type_full,
        beginning_image_number
    """
    import requests
    import pandas as pd

    out_path = config.raw_path("fec") / f"candidate_periodic_reports_{cycle}.csv"
    if out_path.exists() and not force:
        logger.info(f"Candidate periodic reports {cycle}: already present, skipping")
        return

    cand_path = config.raw_path("fec") / f"candidate_disbursements_{cycle}.csv"
    if not cand_path.exists():
        raise FileNotFoundError(
            f"{cand_path} not found -- run fetch_candidate_totals_bulk({cycle}) first; "
            "the periodic-reports fetch needs the candidate roster it selects nominees from."
        )
    from backtest.data import fec as _fec_loader   # local import: avoids a module-level
    # dependency from this data-fetch script on the estimation-layer package for its
    # every other function, matching this file's existing "import inside the function
    # that needs it" convention (see fetch_ie_by_committee, etc.)
    nominees = _fec_loader.load_candidate_disbursements(cycle)   # top spender per (district, party)
    raw_cand_df = pd.read_csv(cand_path, dtype=str)
    # nominees (from load_candidate_disbursements) has no fec_candidate_id column --
    # it's dropped during nominee selection -- so recover it by re-joining on
    # (district_id, party, candidate_disbursements), the same key that selection was
    # performed on. This keeps this function using load_candidate_disbursements as the
    # single source of truth for "who is the nominee" rather than re-deriving it.
    raw_cand_df["candidate_disbursements"] = pd.to_numeric(
        raw_cand_df["candidate_disbursements"], errors="coerce"
    ).fillna(0)
    merged = nominees.merge(
        raw_cand_df[["district_id", "party", "fec_candidate_id", "candidate_disbursements"]],
        on=["district_id", "party", "candidate_disbursements"], how="left",
    )
    # Collision case, checked directly against real 2022/2024 data (not hypothetical):
    # a small number of (district_id, party) groups (~1% -- 5/875 in 2022, 11/872 in
    # 2024) have MORE THAN ONE raw candidate row with the identical disbursement
    # amount as the selected nominee -- e.g. WI-03 2024 D, "COOKE, REBECCA" appears
    # under two different fec_candidate_id values (H2WI03130, H4WI03169) with the
    # exact same $6,347,919.13 total, an FEC re-registration quirk that predates this
    # fetch (load_candidate_disbursements's own groupby(...).first() already resolves
    # this identically arbitrarily, since both raw rows tie on disbursement amount --
    # this is not a new ambiguity introduced here). Logged and kept (first match
    # taken) rather than silently resolved, since a wrong choice here fetches a real
    # but possibly-mismatched committee's periodic reports for ~1% of districts --
    # acceptable for a trickle-rate calibration that aggregates hundreds of
    # district-periods to a tier median, not acceptable to leave undocumented.
    collision_groups = merged.groupby(["district_id", "party"]).size()
    n_collisions = int((collision_groups > 1).sum())
    if n_collisions:
        logger.warning(
            f"Candidate periodic reports {cycle}: {n_collisions} (district_id, party) "
            "nominee(s) matched more than one raw candidate row on tied disbursement "
            "amounts (see fetch_candidate_periodic_reports docstring) -- keeping the "
            "first match for each, not guaranteed to be the intended committee."
        )
    cand_df = merged.drop_duplicates(subset=["district_id", "party"])
    n_unmatched = int(cand_df["fec_candidate_id"].isna().sum())
    if n_unmatched:
        logger.warning(
            f"Candidate periodic reports {cycle}: {n_unmatched} nominee(s) could not be "
            "re-matched to a fec_candidate_id (no raw row with a matching disbursement "
            "amount) and will be skipped."
        )
        cand_df = cand_df[cand_df["fec_candidate_id"].notna()].copy()
    crosswalk = _load_candidate_committee_crosswalk(cycle)

    partial_path = config.raw_path("fec") / f"candidate_periodic_reports_{cycle}.partial.csv"
    done_committees: set[str] = set()
    if partial_path.exists() and not force:
        existing = pd.read_csv(partial_path, dtype=str)
        if len(existing):
            done_committees = set(existing["committee_id"].dropna().unique())
        logger.info(
            f"Resuming {cycle}: {len(done_committees)} committees already fetched in "
            f"{partial_path.name}, will be skipped."
        )
    else:
        pd.DataFrame(columns=_PERIODIC_REPORT_COLUMNS).to_csv(partial_path, index=False)

    n_no_committee = 0
    n_resumed_skip = 0
    n_fetched_this_run = 0
    n_report_rows_this_run = 0
    with requests.Session() as session:
        for i, row in cand_df.iterrows():
            cand_id = row.get("fec_candidate_id")
            committee_id = crosswalk.get(cand_id)
            if not committee_id:
                n_no_committee += 1
                continue
            if committee_id in done_committees:
                n_resumed_skip += 1
                continue
            try:
                records = _fec_paginate(session, f"/committee/{committee_id}/reports/", {
                    "api_key": api_key,
                    "cycle": cycle,
                })
            except Exception as e:
                logger.warning(f"  Failed to fetch reports for {committee_id} ({cand_id}): {e}")
                continue
            rows = [{
                "district_id": row.get("district_id"),
                "party": row.get("party"),
                "cycle": cycle,
                "fec_candidate_id": cand_id,
                "committee_id": committee_id,
                "coverage_start_date": r.get("coverage_start_date"),
                "coverage_end_date": r.get("coverage_end_date"),
                "receipts_period": r.get("total_receipts_period"),
                "disbursements_period": r.get("total_disbursements_period"),
                "cash_on_hand_end_period": r.get("cash_on_hand_end_period"),
                "report_type_full": r.get("report_type_full"),
                "beginning_image_number": r.get("beginning_image_number"),
            } for r in records]
            # Always append a row even for a committee with zero reports, so a
            # resumed run's `done_committees` set (built from committee_id
            # values actually present in the file) still recognizes this
            # committee as already handled -- otherwise a zero-report
            # committee would be re-fetched on every resume, forever.
            if not rows:
                rows = [{col: None for col in _PERIODIC_REPORT_COLUMNS}]
                rows[0].update({"district_id": row.get("district_id"), "party": row.get("party"),
                                 "cycle": cycle, "fec_candidate_id": cand_id, "committee_id": committee_id})
            pd.DataFrame(rows, columns=_PERIODIC_REPORT_COLUMNS).to_csv(
                partial_path, mode="a", header=False, index=False
            )
            n_fetched_this_run += 1
            n_report_rows_this_run += len(records)
            if n_fetched_this_run % 50 == 0:
                logger.info(
                    f"  {cycle}: {n_fetched_this_run} committees fetched this run "
                    f"({n_resumed_skip} resumed/skipped, {i + 1}/{len(cand_df)} candidates scanned), "
                    f"{n_report_rows_this_run} report-rows this run"
                )

    if n_no_committee:
        logger.warning(
            f"Candidate periodic reports {cycle}: {n_no_committee}/{len(cand_df)} candidates "
            "had no principal-committee match in the ccl crosswalk and were skipped."
        )

    partial_path.rename(out_path)
    final_df = pd.read_csv(out_path, dtype=str)
    n_committees = final_df["committee_id"].nunique() if len(final_df) else 0
    logger.info(
        f"Complete: {len(final_df)} periodic report-rows across {n_committees} committees "
        f"({n_fetched_this_run} fetched this run, {n_resumed_skip} resumed from a prior run) → {out_path}"
    )


def consolidate_fec_files(cycle: int, force: bool = False, kinds: list[str] | None = None) -> None:
    """Merge per-committee IE and coordinated files into single canonical files.

    force=True re-consolidates even if the output already exists -- needed
    when a new source label (e.g. "state_party_dem") gains data after the
    canonical file was first built; without it, this function's normal
    if-not-exists guard would silently keep the older, incomplete file.

    kinds: restrict which canonical file(s) to touch (subset of
    {"coordinated", "ie"}; default both, the original behavior). Added
    2026-08 after force=True's blast radius caused a real, damaging
    regression: adding the state-party coordinated source and calling
    consolidate_fec_files(cycle, force=True) also silently re-ran the "ie"
    branch, overwriting the (existing, richer) build_comprehensive_ie()
    output -- multi-outside-group independent-expenditure data -- with a
    much narrower DCCC/NRCC-API-only concatenation, corrupting $350M+ of
    real spending data on the first live run. Discovered immediately via a
    before/after d_total sanity diff (a $358M drop was not remotely
    plausible from a $153K new source) and restored from git; kinds lets a
    caller that only has a new SOURCE for one canonical file force-refresh
    just that one, without touching the other."""
    import pandas as pd

    fec_dir = config.raw_path("fec")
    coordinated_labels = ["dccc", "nrcc", "state_party_dem", "state_party_rep"]
    all_kinds = [("coordinated", coordinated_labels), ("ie", ["dccc", "nrcc"])]
    selected = all_kinds if kinds is None else [(k, labels) for k, labels in all_kinds if k in kinds]
    for kind, labels in selected:
        out = fec_dir / f"{'coordinated_expenditures' if kind == 'coordinated' else 'independent_expenditures'}_{cycle}.csv"
        if not out.exists() or force:
            frames = [
                pd.read_csv(fec_dir / f"{kind}_{label}_{cycle}.csv")
                for label in labels
                if (fec_dir / f"{kind}_{label}_{cycle}.csv").exists()
            ]
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(out, index=False)
                logger.info(f"Consolidated → {out}")


# ─── Skip-mode: empty party spend placeholders ───────────────────────────────

def generate_empty_party_spend_files(cycle: int) -> None:
    """
    Write zero-row canonical party spend files so the pipeline can run with
    candidate committee data only (no API key required).

    To add real party spending later: delete these files and re-run without
    --skip-party-spend.
    """
    import pandas as pd

    fec_dir = config.raw_path("fec")

    ie_path = fec_dir / f"independent_expenditures_{cycle}.csv"
    if not ie_path.exists():
        pd.DataFrame(columns=["district_id", "party", "cycle", "support_oppose", "amount"]
                     ).to_csv(ie_path, index=False)
        logger.info(f"Empty placeholder → {ie_path.name}")

    coord_path = fec_dir / f"coordinated_expenditures_{cycle}.csv"
    if not coord_path.exists():
        pd.DataFrame(columns=["district_id", "party", "cycle", "coordinated_expenditures"]
                     ).to_csv(coord_path, index=False)
        logger.info(f"Empty placeholder → {coord_path.name}")


# ─── Incumbency (derived from candidate totals, no API needed) ────────────────

def derive_incumbency(cycle: int) -> None:
    """
    Build incumbency_{cycle}.csv from candidate_disbursements_{cycle}.csv.

    incumb_status is from the Democratic candidate's perspective:
      D candidate "Incumbent"  → "Incumbent"
      D candidate "Challenger" → "Challenger"
      "Open seat"              → "Open"

    Also captures incumbent_name / challenger_name for repeat-challenger
    pair identification in estimation/beta_rc.py.
    """
    import pandas as pd

    out_path = config.raw_path("fec") / f"incumbency_{cycle}.csv"
    if out_path.exists():
        logger.info(f"Incumbency {cycle}: already present, skipping")
        return

    cand_path = config.raw_path("fec") / f"candidate_disbursements_{cycle}.csv"
    if not cand_path.exists():
        logger.warning(f"candidate_disbursements_{cycle}.csv not found — run --only fec first")
        return

    logger.info(f"Deriving incumbency {cycle}…")
    df = pd.read_csv(cand_path, dtype={"district_id": str, "party": str})
    df = df[df["party"].isin(["D", "R"])]

    # Top spender per party × district as nominee proxy
    df = (
        df.sort_values("candidate_disbursements", ascending=False)
        .groupby(["district_id", "party"], sort=False).first().reset_index()
    )

    rows = []
    for dist_id, grp in df.groupby("district_id"):
        d_row = grp[grp["party"] == "D"]
        r_row = grp[grp["party"] == "R"]

        d_ic = str(d_row["incumbent_challenge_full"].iloc[0]) if not d_row.empty else "Open seat"

        if d_ic == "Incumbent":
            incumb_status   = "Incumbent"
            incumbent_name  = d_row["candidate_name"].iloc[0] if not d_row.empty else ""
            challenger_name = r_row["candidate_name"].iloc[0] if not r_row.empty else ""
        elif d_ic == "Challenger":
            incumb_status   = "Challenger"
            incumbent_name  = r_row["candidate_name"].iloc[0] if not r_row.empty else ""
            challenger_name = d_row["candidate_name"].iloc[0] if not d_row.empty else ""
        else:
            incumb_status   = "Open"
            incumbent_name  = ""
            challenger_name = d_row["candidate_name"].iloc[0] if not d_row.empty else ""

        rows.append({
            "district_id":    dist_id,
            "cycle":          cycle,
            "incumb_status":  incumb_status,
            "incumbent_name": incumbent_name,
            "challenger_name": challenger_name,
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"Saved {len(rows)} districts → {out_path}")


# ─── Comprehensive independent expenditures ──────────────────────────────────

def build_comprehensive_ie(cycle: int, force: bool = False) -> None:
    """
    Build independent_expenditures_{cycle}.csv from the comprehensive FEC IE file
    (data/raw/independent_expenditure/independent_expenditure_{cycle}.csv).

    This replaces the DCCC/NRCC-only IE approach with ALL outside group spending,
    which is especially important for capturing R-aligned spending (super PACs and
    other Republican-aligned groups that often outspend the NRCC in competitive races).

    Party alignment is computed from candidate party × support/oppose indicator:
      D-aligned: (candidate is D AND support) OR (candidate is R AND oppose)
      R-aligned: (candidate is R AND support) OR (candidate is D AND oppose)

    Output schema: district_id, party [D or R aligned], cycle, amount
    (same schema as the DCCC/NRCC-only file it replaces)
    """
    import pandas as pd

    out_path = config.raw_path("fec") / f"independent_expenditures_{cycle}.csv"
    if out_path.exists() and not force:
        logger.info(f"Independent expenditures {cycle}: already present, skipping")
        return

    src_dir = Path(__file__).parent.parent / "data" / "raw" / "independent_expenditure"
    src_path = src_dir / f"independent_expenditure_{cycle}.csv"
    if not src_path.exists():
        logger.warning(
            f"Comprehensive IE file not found: {src_path}. "
            "Falling back to DCCC/NRCC-only approach."
        )
        return

    logger.info(f"Building comprehensive IEs for {cycle} from {src_path.name}…")
    df = pd.read_csv(src_path, dtype=str, low_memory=False)

    # Filter to House general election races
    df = df[(df["can_office"] == "H") & (df["ele_type"] == "G")].copy()
    logger.info(f"  {len(df)} House general IE rows")

    df["exp_amo"] = pd.to_numeric(df["exp_amo"], errors="coerce").fillna(0).abs()

    # Build district_id
    df["state"] = df["can_office_state"].str.strip().str.upper()
    df["dist"]  = df["can_office_dis"].str.strip().str.zfill(2)
    df["district_id"] = df["state"] + "-" + df["dist"]

    # Determine party alignment
    is_dem_cand = df["cand_pty_aff"].str.upper().str.contains("DEMOCRAT", na=False)
    is_rep_cand = df["cand_pty_aff"].str.upper().str.contains("REPUBLICAN", na=False)
    is_support  = df["sup_opp"].str.upper() == "S"
    is_oppose   = df["sup_opp"].str.upper() == "O"

    d_aligned = (is_dem_cand & is_support) | (is_rep_cand & is_oppose)
    r_aligned = (is_rep_cand & is_support) | (is_dem_cand & is_oppose)

    d_ie = (
        df[d_aligned].groupby("district_id")["exp_amo"].sum()
        .reset_index().rename(columns={"exp_amo": "amount"})
    )
    d_ie["party"] = "D"

    r_ie = (
        df[r_aligned].groupby("district_id")["exp_amo"].sum()
        .reset_index().rename(columns={"exp_amo": "amount"})
    )
    r_ie["party"] = "R"

    out = pd.concat([d_ie, r_ie], ignore_index=True)
    out["cycle"] = cycle

    out[["district_id", "party", "cycle", "amount"]].to_csv(out_path, index=False)
    logger.info(
        f"Comprehensive IEs saved: {len(d_ie)} D-aligned districts, "
        f"{len(r_ie)} R-aligned districts → {out_path}"
    )


def rebuild_all_from_local(cycles: list[int]) -> None:
    """
    Regenerate candidate_disbursements, independent_expenditures, and incumbency CSVs
    for all cycles from local bulk files. Use this after adding new data or fixing bugs.
    """
    import os
    for cycle in cycles:
        logger.info(f"─── Rebuilding cycle {cycle} ───")
        fetch_candidate_totals_local(cycle, force=True)
        build_comprehensive_ie(cycle, force=True)
        # Force-rebuild incumbency by deleting existing file first
        inc_path = config.raw_path("fec") / f"incumbency_{cycle}.csv"
        if inc_path.exists():
            os.remove(inc_path)
        derive_incumbency(cycle)
    logger.info("Rebuild complete.")


# ─── Census CVAP ─────────────────────────────────────────────────────────────

CVAP_BULK_URL = (
    "https://www2.census.gov/programs-surveys/decennial/rdo/datasets"
    "/2022/2022-cvap/CVAP_2018-2022_ACS_csv_files.zip"
)


def _cvap_vintage_url(end_year: int) -> str:
    """URL pattern verified live for end_year in {2014,2018,2019,2020,2024}
    (2026-08) -- Census publishes CVAP special tabulations annually, named
    by the LAST year of the rolling 5-year ACS span:
    .../datasets/{end_year}/{end_year}-cvap/CVAP_{end_year-4}-{end_year}_ACS_csv_files.zip"""
    return (
        f"https://www2.census.gov/programs-surveys/decennial/rdo/datasets"
        f"/{end_year}/{end_year}-cvap/CVAP_{end_year - 4}-{end_year}_ACS_csv_files.zip"
    )


def _parse_cvap_cd_csv(content: bytes) -> "pd.DataFrame":
    """Shared extraction logic for a CVAP special-tabulation ZIP's CD.csv:
    keep the "Total" row per district (lntitle has 13 race/ethnicity rows),
    parse geoid ("5001800US{STATE_FIPS:02d}{DISTRICT:02d}") into
    district_id. Refactored out of fetch_census_cvap() (2026-08) so
    fetch_cvap_vintage()/build_cvap_panel() below can reuse it instead of
    duplicating the parsing logic for each of up to 11 vintages.

    Returns district_id, cvap (int) -- caller adds any vintage/cycle label."""
    import pandas as pd

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        cd_raw = zf.read("CD.csv")
    df = pd.read_csv(io.BytesIO(cd_raw), dtype=str)
    # Older vintages (verified: 2016, 2017) use UPPERCASE column names
    # (GEOID, LNTITLE, CVAP_EST); 2018+ use lowercase. Normalize so one
    # parser handles both rather than branching on vintage year.
    df.columns = df.columns.str.lower()

    total = df[df["lntitle"] == "Total"].copy()
    suffix = total["geoid"].str.split("US").str[-1]
    total["fips"] = suffix.str[:2]
    total["dist"] = suffix.str[2:].str.zfill(2)
    total["state"] = total["fips"].map(FIPS_TO_STATE)
    total["district_id"] = total["state"] + "-" + total["dist"]
    total["cvap"] = pd.to_numeric(total["cvap_est"], errors="coerce")

    return (
        total.dropna(subset=["district_id", "state", "cvap"])
        [["district_id", "cvap"]].copy()
        .assign(cvap=lambda d: d["cvap"].astype(int))
    )


def _download_cvap_zip(url: str, label: str) -> bytes:
    import requests

    logger.info(f"Downloading Census CVAP bulk file ({label})…")
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == 2:
                raise
            logger.warning(f"Census download error: {e}. Retrying…")
            time.sleep(10)
    content = b"".join(resp.iter_content(chunk_size=1 << 20))
    logger.info(f"  Downloaded {len(content) // 1024 // 1024} MB")
    return content


def fetch_census_cvap(census_api_key: str = "") -> None:
    """
    Download CVAP per congressional district from the Census CVAP Special
    Tabulation (2018–2022 ACS 5-year).  No API key required — uses the
    bulk ZIP published at www2.census.gov.

    The census_api_key argument is accepted for backward compatibility but
    is no longer used (the Census API now requires a key even for public
    tables, and the bulk file is the preferred source).

    Output schema: district_id, cvap
    """
    out_path = config.raw_path("census") / "cvap_2022_acs5.csv"
    if out_path.exists():
        logger.info("Census CVAP: already present, skipping")
        return

    content = _download_cvap_zip(CVAP_BULK_URL, "2018-2022, ~54 MB")
    out = _parse_cvap_cd_csv(content)
    out.to_csv(out_path, index=False)
    logger.info(f"Saved {len(out)} congressional districts → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch backtest raw data")
    parser.add_argument(
        "--fec-api-key", default="DEMO_KEY",
        help="FEC API key for IE/coordinated pulls (Tier 2). "
             "DEMO_KEY has 30 req/hr and will fail on multi-cycle runs. "
             "Register free at https://api.open.fec.gov/developers (1,000 req/hr).",
    )
    parser.add_argument(
        "--skip-party-spend", action="store_true",
        help="Skip DCCC/NRCC IE and coordinated expenditure API calls; "
             "write empty placeholder files instead. The pipeline runs on "
             "candidate committee data only. Use with DEMO_KEY or when no "
             "registered API key is available. "
             "To fill in party spending later: delete the placeholder files "
             "and re-run without this flag.",
    )
    parser.add_argument(
        "--census-api-key", default="",
        help="Census API key — register free at https://api.census.gov/data/key_signup.html",
    )
    parser.add_argument(
        "--cycles", nargs="+", type=int,
        default=config.panel_cycles() + [2024],
    )
    parser.add_argument(
        "--only", choices=["fec", "fec-periodic", "incumbency", "census",
                            "state-party-coordinated", "all"],
        default="all",
    )
    parser.add_argument(
        "--rebuild-local", action="store_true",
        help="Force rebuild of candidate_disbursements and independent_expenditures CSVs "
             "from locally cached bulk files (data/raw/bulk_all/ and "
             "data/raw/independent_expenditure/). Use this to apply the corrected "
             "TTL_DISB column mapping and switch to comprehensive IE data.",
    )
    args = parser.parse_args()

    # Fast path: rebuild everything from local files
    if args.rebuild_local:
        logger.info("Rebuilding all spending CSVs from local bulk files…")
        rebuild_all_from_local(args.cycles)
        return

    if args.fec_api_key == "DEMO_KEY" and not args.skip_party_spend:
        logger.warning(
            "DEMO_KEY detected. Tier 2 (DCCC/NRCC IEs + coordinated) "
            "exhausts the 30 req/hr quota after ~3 pages and will fail. "
            "Add --skip-party-spend to use candidate committee data only, "
            "or register a free key at https://api.open.fec.gov/developers "
            "and pass --fec-api-key YOUR_KEY."
        )

    if args.only in ("fec", "all"):
        for cycle in args.cycles:
            logger.info(f"─── Cycle {cycle} ───")
            fetch_candidate_totals_bulk(cycle)
            # Prefer comprehensive IEs from local file; fall back to DCCC/NRCC-only API
            build_comprehensive_ie(cycle)
            if not (config.raw_path("fec") / f"independent_expenditures_{cycle}.csv").exists():
                if args.skip_party_spend:
                    generate_empty_party_spend_files(cycle)
                else:
                    fetch_ie_by_committee(cycle, args.fec_api_key, DCCC_COMMITTEE_ID, "D")
                    fetch_ie_by_committee(cycle, args.fec_api_key, NRCC_COMMITTEE_ID, "R")
                    consolidate_fec_files(cycle)
            if args.skip_party_spend:
                coord_path = config.raw_path("fec") / f"coordinated_expenditures_{cycle}.csv"
                if not coord_path.exists():
                    generate_empty_party_spend_files(cycle)
            else:
                fetch_coordinated_by_committee(cycle, args.fec_api_key, DCCC_COMMITTEE_ID, "D")
                fetch_coordinated_by_committee(cycle, args.fec_api_key, NRCC_COMMITTEE_ID, "R")
                coord_path = config.raw_path("fec") / f"coordinated_expenditures_{cycle}.csv"
                if not coord_path.exists():
                    consolidate_fec_files(cycle)

    if args.only in ("fec", "incumbency", "all"):
        for cycle in args.cycles:
            derive_incumbency(cycle)

    if args.only == "fec-periodic":
        if args.fec_api_key == "DEMO_KEY":
            logger.warning(
                "DEMO_KEY detected (40 req/hr). This fetch makes one call per "
                "House candidate committee (~400-800 per cycle) -- register a "
                "free key at https://api.open.fec.gov/developers and pass "
                "--fec-api-key YOUR_KEY, or this will be extremely slow."
            )
        for cycle in args.cycles:
            logger.info(f"─── Candidate periodic reports: cycle {cycle} ───")
            fetch_candidate_periodic_reports(cycle, args.fec_api_key)

    if args.only in ("census", "all"):
        fetch_census_cvap(args.census_api_key)

    if args.only == "state-party-coordinated":
        # Deliberately NOT part of "all" -- a heavy scan of local multi-GB
        # bulk files (up to 3.3GB each), not a routine API fetch, and it
        # re-consolidates coordinated_expenditures_{cycle}.csv with force=True
        # (see consolidate_fec_files), overwriting any already-fetched
        # DCCC/NRCC-only version -- an explicit, deliberate step per
        # FINDINGS.md Section 10.7 Gap 3, not something a --only all run
        # should trigger silently.
        for cycle in args.cycles:
            logger.info(f"─── State-party 24K coordinated expenditures: cycle {cycle} ───")
            for party, label in [("D", "dem"), ("R", "rep")]:
                out = parse_state_party_coordinated_24k(cycle, party=party)
                out_path = config.raw_path("fec") / f"coordinated_state_party_{label}_{cycle}.csv"
                out.to_csv(out_path, index=False)
                logger.info(f"Saved → {out_path}")
            consolidate_fec_files(cycle, force=True, kinds=["coordinated"])

    logger.info("\nFetch complete.")
    if getattr(args, "skip_party_spend", False):
        logger.info(
            "Party spend (IEs + coordinated) was skipped — placeholder files written. "
            "D_total / R_total will reflect candidate committee disbursements only. "
            "To add party spending: register at https://api.open.fec.gov/developers, "
            "delete the placeholder files, and re-run with --fec-api-key YOUR_KEY."
        )
    logger.info(
        "\nManual steps still required:\n"
        "\n"
        "  1. MIT MEDSL House results (Harvard Dataverse) — already done if you see this:\n"
        "       https://dataverse.harvard.edu/dataset.xhtml"
        "?persistentId=doi:10.7910/DVN/IG0UN2\n"
        "     → data/raw/mit_elections/1976-2024-house.tab\n"
        "\n"
        "  2. Presidential results by congressional district (for PVI computation):\n"
        "     Source: Daily Kos Elections — 2016 and 2020 presidential results\n"
        "     by 118th Congress districts (post-2021 redistricting maps).\n"
        "     Export their Google Sheet or download from dailykos.com/elections.\n"
        "     Format: district_id, d_votes, r_votes (one row per district).\n"
        "     → data/raw/presidential/pres_2016.csv\n"
        "     → data/raw/presidential/pres_2020.csv\n"
        "     For historical panel cycles 2012–2020, also acquire pre-2021 map\n"
        "     editions (113th–116th Congress) from Daily Kos Elections archives.\n"
        "\n"
        "  3. Cook ratings 2024 (optional — derived from PVI if not present):\n"
        "     → data/raw/cook_pvi/cook_ratings_2024.csv\n"
    )


if __name__ == "__main__":
    main()
