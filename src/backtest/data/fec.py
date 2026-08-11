"""
FEC disbursement and expenditure loader.

Raw data contract (files produced by scripts/fetch_data.py)
───────────────────────────────────────────────────────────
  candidate_disbursements_{cycle}.csv
    Columns: district_id, fec_candidate_id, candidate_name, party, cycle,
             candidate_disbursements, incumbent_challenge_full,
             ttl_receipts, ttl_indiv_contrib, indiv_share
    One row per candidate. Multiple candidates per party per district are
    possible (primary losers); load_candidate_disbursements selects the
    top spender per party per district as the general-election nominee proxy.

    IMPORTANT: candidate_disbursements = TTL_DISB (total disbursements, weball
    col 7). Earlier pipeline versions incorrectly used col 17 (TTL_INDIV_CONTRIB).
    Run `scripts/fetch_data.py --rebuild-local` to regenerate with the fix.

    indiv_share = TTL_INDIV_CONTRIB (col 17) / TTL_RECEIPTS (col 5), clipped
    to [0, 1]. Zero when TTL_RECEIPTS = 0. Proxy for candidate quality /
    grassroots fundraising strength, orthogonal to total spend.

  coordinated_expenditures_{cycle}.csv
    Columns: district_id, party, cycle, coordinated_expenditures
    One row per (district, party) — pre-aggregated by fetch_data.py.

  independent_expenditures_{cycle}.csv
    Two formats (auto-detected):
      Comprehensive (from data/raw/independent_expenditure/):
        Columns: district_id, party [aligned: D or R], cycle, amount
        Covers ALL outside groups — super PACs, party committees, 527s.
      Legacy DCCC/NRCC-only:
        Columns: district_id, party [spender party], cycle, support_oppose, amount
    load_independent_expenditures handles both formats.
"""

from __future__ import annotations
import logging
import pandas as pd
from .. import config

logger = logging.getLogger(__name__)


def _ballot_last_names(cycle: int) -> set[tuple[str, str, str]]:
    """
    Return the set of (district_id, party_initial, last_name) tuples for
    candidates who actually appeared on the general-election ballot, sourced
    from the MIT MEDSL elections file.

    Used to filter out weball entries for members who ran for a different
    federal office (e.g., Senate) while their House committee was still active.
    MIT name format is "FIRSTNAME LASTNAME"; FEC format is "LASTNAME, FIRSTNAME".
    We match on upper-cased last name token only.
    """
    mit_path = config.raw_path("mit") / "1976-2024-house.tab"
    if not mit_path.exists():
        logger.warning(
            f"_ballot_last_names({cycle}): {mit_path} not found — the Senate/"
            "President-runner exclusion filter in load_candidate_disbursements() "
            "will be SKIPPED for this cycle, so committees of House members who "
            "ran for a different office will NOT be excluded."
        )
        return set()
    raw = pd.read_csv(mit_path, sep=",", dtype={"district": str}, low_memory=False)
    gen = raw[
        (raw["year"] == cycle)
        & (raw["stage"].str.upper() == "GEN")
        & (~raw["candidate"].isna())
    ]
    if gen.empty:
        logger.warning(
            f"_ballot_last_names({cycle}): {mit_path} has no GEN-stage rows for "
            f"{cycle} (MIT MEDSL data lags the live cycle — its latest covered "
            "year is typically the prior completed election). The Senate/"
            "President-runner exclusion filter in load_candidate_disbursements() "
            "will be SKIPPED for this cycle: any House member's committee that "
            "disbursed >$10M while running for a different office will NOT be "
            "excluded from district totals until MIT's data catches up."
        )
    result: set[tuple[str, str, str]] = set()
    for _, row in gen.iterrows():
        dist = str(row["state_po"]) + "-" + str(row["district"]).zfill(2)
        party = str(row["party"]).upper()
        party_initial = "D" if "DEMOCRAT" in party else ("R" if "REPUBLICAN" in party else "X")
        # MIT name: "FIRSTNAME LASTNAME" or "FIRSTNAME MIDDLE LASTNAME"
        name_parts = str(row["candidate"]).upper().split()
        last = name_parts[-1] if name_parts else ""
        result.add((dist, party_initial, last))
    return result


def load_candidate_disbursements(cycle: int) -> pd.DataFrame:
    """
    Return total principal-committee disbursements per district per party.

    Selects the top-spending candidate per party per district as the
    general-election nominee proxy.  Candidates are filtered against the MIT
    MEDSL ballot list so that House members who ran for a different federal
    office in the same cycle (e.g., Gallego/AZ-03, Schiff/CA-30 running for
    Senate in 2024) are excluded — their committee disbursements would otherwise
    inflate the district's apparent candidate spending by $40–60M.

    Returns DataFrame with columns:
        district_id, party, cycle, candidate_disbursements (float),
        indiv_share (float, 0–1)
    """
    path = config.raw_path("fec") / f"candidate_disbursements_{cycle}.csv"
    df = pd.read_csv(path, dtype={"district_id": str, "party": str})
    df["cycle"] = cycle
    df["candidate_disbursements"] = pd.to_numeric(
        df["candidate_disbursements"], errors="coerce"
    ).fillna(0)
    if "indiv_share" not in df.columns:
        df["indiv_share"] = 0.0

    df = df[df["party"].isin(["D", "R"])].copy()

    # Cross-reference against actual ballot candidates from MIT elections data.
    # Only exclude HIGH-SPENDING candidates (>$10M) who do NOT appear on the
    # House general-election ballot.  This targets the specific artifact where a
    # House member running for Senate (e.g., Gallego/AZ-03 $59M, Schiff/CA-30 $45M)
    # has their committee spending attributed to their old House district.
    # Low-spending candidates (<$10M) pass through regardless of name matching,
    # because name-format mismatches (JR/SR suffixes, compound names) cause false
    # exclusions and drop legitimate R candidates from safe-D districts.
    _LARGE_SPEND_THRESHOLD = 10_000_000
    ballot = _ballot_last_names(cycle)
    # Manual, dated, individually-verified substitute for cycles MIT can't
    # cover yet (config.yaml's `live_cycle_ballot_exclusions`) -- applied even
    # when `ballot` is empty, since that's exactly the situation it exists for.
    manual_exclusions = {
        (e["district_id"], e["party"], e["last_name"])
        for e in config.live_cycle_ballot_exclusions(cycle)
    }

    if ballot or manual_exclusions:
        def _on_ballot(row: pd.Series) -> bool:
            # Last WORD only, matching _ballot_last_names()'s MIT-side extraction
            # (name_parts[-1]) -- taking the full pre-comma segment instead breaks
            # on compound surnames: FEC's "GLUESENKAMP PEREZ, MARIE" gives
            # "GLUESENKAMP PEREZ" here vs MIT's "PEREZ", so the real WA-03 2024
            # incumbent (an $11.9M-spending candidate, over _LARGE_SPEND_THRESHOLD)
            # was silently excluded as "not on ballot" and the next-highest D
            # candidate ($7.6k) was used for cand_d_total instead -- caught via
            # the persuasion-ceiling MSG-sign validation gate (FINDINGS.md).
            last = str(row["candidate_name"]).split(",")[0].strip().upper().split()[-1]
            key = (row["district_id"], row["party"], last)
            # Manual exclusions apply BEFORE and regardless of
            # _LARGE_SPEND_THRESHOLD -- unlike the automated MIT name-match
            # below, each manual entry is individually hand-verified against
            # Ballotpedia/FEC.gov (config.yaml's own documented standard for
            # this list), so there's no name-format-mismatch false-positive
            # risk to guard against by gating it on dollar amount. Found
            # necessary 2026-08-10: Kyrsten Sinema's stale House candidate ID
            # (H2AZ09019, from her 2012-2018 AZ-09 tenure) still carries her
            # Senate committee's disbursements ($2.33M in 2022, $9.27M in
            # 2024) -- both under $10M, so the threshold-gated check below
            # never evaluated her, even though MIT's ballot data already
            # correctly has no "SINEMA" entry for AZ-09 in either cycle
            # (confirmed directly via _ballot_last_names()).
            if key in manual_exclusions:
                return False
            if row["candidate_disbursements"] < _LARGE_SPEND_THRESHOLD:
                return True
            if ballot:
                return key in ballot
            return True  # no MIT data and not manually excluded: honest pass-through

        before = len(df)
        df = df[df.apply(_on_ballot, axis=1)]
        excluded = before - len(df)
        if excluded:
            logger.info(
                f"Candidate disbursements {cycle}: excluded {excluded} large-spending "
                "candidates not on the House ballot (likely ran for Senate/President, "
                "or lost a primary) via MIT data and/or live_cycle_ballot_exclusions"
            )

    df["indiv_share"] = pd.to_numeric(df["indiv_share"], errors="coerce").fillna(0.0)

    top = (
        df.sort_values("candidate_disbursements", ascending=False)
        .groupby(["district_id", "party"], sort=False)
        .first()
        .reset_index()
    )
    return top[["district_id", "party", "cycle", "candidate_disbursements", "indiv_share"]]


def load_coordinated_expenditures(cycle: int) -> pd.DataFrame:
    """
    Return DCCC / NRCC coordinated expenditures per race.

    Returns DataFrame with columns:
        district_id, party, cycle, coordinated_expenditures (float)
    """
    path = config.raw_path("fec") / f"coordinated_expenditures_{cycle}.csv"
    df = pd.read_csv(path, dtype={"district_id": str, "party": str})
    df["cycle"] = cycle
    df["coordinated_expenditures"] = pd.to_numeric(
        df["coordinated_expenditures"], errors="coerce"
    ).fillna(0)
    return df[["district_id", "party", "cycle", "coordinated_expenditures"]]


def load_independent_expenditures(cycle: int) -> pd.DataFrame:
    """
    Return aligned independent expenditures per race (all outside groups).

    Handles two file formats:
      Comprehensive (produced by build_comprehensive_ie):
        district_id, party [D=D-aligned, R=R-aligned], cycle, amount
      Legacy DCCC/NRCC-only:
        district_id, party, cycle, support_oppose, amount

    Returns DataFrame with columns:
        district_id, party, cycle, ie_net (float)
    """
    path = config.raw_path("fec") / f"independent_expenditures_{cycle}.csv"
    df = pd.read_csv(path, dtype={"district_id": str, "party": str}, low_memory=False)
    df["cycle"] = cycle
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # Both the legacy (DCCC/NRCC-only) and comprehensive formats store `amount`
    # unsigned with party alignment already reflected in the `party` column.
    df["signed_amount"] = df["amount"].abs()

    ie = (
        df.groupby(["district_id", "party", "cycle"])["signed_amount"]
        .sum()
        .reset_index()
        .rename(columns={"signed_amount": "ie_net"})
    )
    return ie


def build_total_spend(cycle: int) -> pd.DataFrame:
    """
    Construct D_total and R_total per race for a given cycle.

    D_total_i = candidate_D_i + DCCC_coordinated_i + DCCC_IE_i
    R_total_i = candidate_R_i + NRCC_coordinated_i + NRCC_IE_i

    Returns DataFrame with columns:
        district_id, cycle, d_total, r_total
    """
    cand = load_candidate_disbursements(cycle)
    coord = load_coordinated_expenditures(cycle)
    ie = load_independent_expenditures(cycle)

    def _party_total(party: str) -> pd.Series:
        c = (
            cand[cand["party"] == party]
            .set_index("district_id")["candidate_disbursements"]
        )
        # groupby-sum, not set_index: coord can legitimately carry more than
        # one row per (district_id, party) once more than one coordinating
        # source exists for the same party -- e.g. DCCC's own coordinated
        # spend (coordinated_dccc_{cycle}.csv) AND a state party's 24K
        # coordinated spend into the same district
        # (coordinated_state_party_dem_{cycle}.csv) are both party="D".
        # set_index() alone would leave duplicate district_id index labels,
        # which crashes the pd.concat([d, r], axis=1) below with
        # "cannot reindex on an axis with duplicate labels" -- found via a
        # real 2024 run once the state-party source was added (Gap 3).
        # cand/ie don't need this: load_candidate_disbursements() already
        # collapses to one row per (district_id, party) via its own
        # groupby(...).first(), and load_independent_expenditures() via its
        # own groupby(...).sum().
        co = (
            coord[coord["party"] == party]
            .groupby("district_id")["coordinated_expenditures"].sum()
        )
        i = (
            ie[ie["party"] == party]
            .set_index("district_id")["ie_net"]
        )
        return c.add(co, fill_value=0).add(i, fill_value=0).fillna(0)

    d = _party_total("D").rename("d_total")
    r = _party_total("R").rename("r_total")
    combined = pd.concat([d, r], axis=1).fillna(0).reset_index()
    combined["cycle"] = cycle
    return combined[["district_id", "cycle", "d_total", "r_total"]]


# ─── Paper II: point-in-time (dated) IE reconstruction ───────────────────────
# Everything above this line is Paper I and is unmodified. The two functions
# below serve dynamic/simulate.py's one-step-ahead historical harness
# (docs/paper2_draft.md §6.2) and are not used anywhere in Paper I's pipeline.


def load_ie_transactions_dated(cycle: int) -> pd.DataFrame:
    """
    Return transaction-level, dated independent expenditures for a cycle.

    Derived from the raw comprehensive Schedule E file
    (data/raw/independent_expenditure/independent_expenditure_{cycle}.csv) —
    NOT the pre-aggregated independent_expenditures_{cycle}.csv that
    load_independent_expenditures() reads, which collapses to a cycle total
    and drops the transaction date. This is the one component of total race
    spend that can genuinely be reconstructed point-in-time from data
    already in this repo (see the Paper II implementation plan's Phase 3
    data-gap table); candidate-committee and coordinated-expenditure spend
    have no per-filing date source here and must be held fixed by the caller.

    Applies the identical House-general-election filter and D/R alignment
    logic as scripts/fetch_data.py::build_comprehensive_ie() (candidate
    party × support/oppose), but retains each transaction's `exp_date`
    instead of collapsing to a cycle total.

    exp_date in the raw file is in DD-MON-YY form (e.g. "23-SEP-22") and is
    blank for a substantial minority of rows in both cycles checked (2022:
    ~33%, 2024: ~28%) — this is missing data in the source filing, not a
    parsing failure once the correct format string is used (confirmed: with
    format="%d-%b-%y", every non-blank value parses). Rows with a blank
    exp_date are dropped; the dropped count and fraction are logged, not
    silently absorbed.

    Two further data-quality issues (Paper III §4.2) are resolved here:

    FEC amendment chains. `tran_id` is NOT a unique per-transaction key in
    this raw format, despite the name — the same tran_id recurs across many
    rows with unrelated payees, amounts, and dates (e.g. tran_id "SE.4228"
    appears for four unrelated 2018 committees). The real duplication
    mechanism is amendments: `amndt_ind` in {A1, A2, ...} marks a re-filing
    of an earlier transaction, with `prev_file_num` pointing back at the
    filing it supersedes. Any row whose `file_num` is referenced as another
    row's `prev_file_num` has been superseded and is dropped, keeping only
    the terminal version of each amendment chain — otherwise the same
    underlying expenditure is counted once per amendment (observed at
    15-48% of rows across the 2012-2024 panel).

    Implausible single-transaction amounts. The 2022 file contains one row
    (tran_id "F57.000001"-style batch id notwithstanding) with
    exp_amo=$9,999,999,999 and agg_amo=2024.00 — the agg_amo value looks
    like a year that leaked into the wrong field, i.e. a parsing/upstream
    data bug, not real spending. Every other cycle's legitimate maximum is
    under $4.6M. Rows with exp_amo > $20M are dropped as implausible for a
    single IE transaction; the threshold is set well above every other
    cycle's genuine maximum specifically so it cannot silently discard real
    (if unusually large) spending.

    Returns DataFrame with columns: district_id, party [D/R-aligned],
    exp_date (datetime64), amount (float).
    """
    src_path = config.raw_path("ie_comprehensive") / f"independent_expenditure_{cycle}.csv"
    if not src_path.exists():
        raise FileNotFoundError(
            f"Raw comprehensive IE file not found: {src_path}. This is the "
            "transaction-dated source required for point-in-time "
            f"reconstruction — the aggregated independent_expenditures_{cycle}.csv "
            "does not retain dates."
        )
    df = pd.read_csv(src_path, dtype=str, low_memory=False)
    df = df[(df["can_office"] == "H") & (df["ele_type"] == "G")].copy()

    superseded = set(df["prev_file_num"].dropna().astype(str))
    n_before_amend = len(df)
    df = df[~df["file_num"].astype(str).isin(superseded)].copy()
    n_superseded = n_before_amend - len(df)
    if n_superseded:
        logger.info(
            f"load_ie_transactions_dated({cycle}): dropped {n_superseded}/{n_before_amend} "
            f"({n_superseded / n_before_amend:.1%}) rows superseded by a later FEC "
            "amendment (resolved via file_num/prev_file_num, not tran_id)."
        )

    n_total = len(df)
    df["exp_date_parsed"] = pd.to_datetime(df["exp_date"], errors="coerce", format="%d-%b-%y")
    n_dropped = int(df["exp_date_parsed"].isna().sum())
    if n_dropped:
        logger.warning(
            f"load_ie_transactions_dated({cycle}): dropping {n_dropped}/{n_total} "
            f"({n_dropped / n_total:.1%}) House-general IE rows with a blank "
            "exp_date. These transactions are excluded from point-in-time "
            "reconstruction entirely (not attributed to any period)."
        )
    df = df[df["exp_date_parsed"].notna()].copy()

    df["exp_amo"] = pd.to_numeric(df["exp_amo"], errors="coerce").fillna(0).abs()
    n_implausible = int((df["exp_amo"] > 20_000_000).sum())
    if n_implausible:
        logger.warning(
            f"load_ie_transactions_dated({cycle}): dropping {n_implausible} row(s) "
            "with exp_amo > $20M (implausible for a single IE transaction; see "
            "docstring)."
        )
        df = df[df["exp_amo"] <= 20_000_000].copy()

    df["state"] = df["can_office_state"].str.strip().str.upper()
    df["dist"] = df["can_office_dis"].str.strip().str.zfill(2)
    df["district_id"] = df["state"] + "-" + df["dist"]

    is_dem_cand = df["cand_pty_aff"].str.upper().str.contains("DEMOCRAT", na=False)
    is_rep_cand = df["cand_pty_aff"].str.upper().str.contains("REPUBLICAN", na=False)
    is_support = df["sup_opp"].str.upper() == "S"
    is_oppose = df["sup_opp"].str.upper() == "O"

    d_aligned = (is_dem_cand & is_support) | (is_rep_cand & is_oppose)
    r_aligned = (is_rep_cand & is_support) | (is_dem_cand & is_oppose)

    d = df.loc[d_aligned, ["district_id", "exp_date_parsed", "exp_amo"]].copy()
    d["party"] = "D"
    r = df.loc[r_aligned, ["district_id", "exp_date_parsed", "exp_amo"]].copy()
    r["party"] = "R"

    out = pd.concat([d, r], ignore_index=True)
    out = out.rename(columns={"exp_date_parsed": "exp_date", "exp_amo": "amount"})
    return out[["district_id", "party", "exp_date", "amount"]]


def cumulative_ie_as_of(cycle: int, as_of_date) -> pd.DataFrame:
    """
    Return cumulative D/R-aligned independent-expenditure spend per
    district, summing every transaction with exp_date <= as_of_date.

    Returns DataFrame with columns: district_id, party, cycle, ie_net —
    the same schema as load_independent_expenditures(), so callers can
    substitute a point-in-time snapshot wherever the full-cycle aggregate
    would otherwise be used.
    """
    as_of = pd.Timestamp(as_of_date)
    txns = load_ie_transactions_dated(cycle)
    cum = txns[txns["exp_date"] <= as_of]
    ie = (
        cum.groupby(["district_id", "party"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "ie_net"})
    )
    ie["cycle"] = cycle
    return ie[["district_id", "party", "cycle", "ie_net"]]


# ─── Paper III: dated candidate-committee financial panel (new) ─────────────
# candidate_periodic_reports_{cycle}.csv (scripts/fetch_data.py's
# fetch_candidate_periodic_reports) is genuinely dated, per-filing-period
# candidate-committee financial data from the FEC API's
# /committee/{id}/reports/ endpoint — resolving the gap
# docs/theta_followup_plan.md Section 0.1.1 documented as "no per-filing
# date field anywhere in this repository" (true of the bulk weball files
# load_candidate_disbursements reads; not true of this API endpoint, which
# nobody had checked until this session). Distinct from
# load_candidate_disbursements() above, which stays cycle-cumulative-final
# (TTL_DISB) and is Paper I's frozen input — unmodified by anything below.


def load_candidate_periodic_reports(cycle: int) -> pd.DataFrame:
    """
    Return dated, amendment-resolved candidate-committee periodic reports
    for a cycle.

    Amendment resolution: a filing that was later amended can appear more
    than once with the identical (committee_id, coverage_start_date,
    coverage_end_date) — the raw fetch does not resolve this (see
    fetch_candidate_periodic_reports's docstring). Kept: the row with the
    lexicographically largest `beginning_image_number` per
    (committee_id, coverage_start_date, coverage_end_date) group —
    FEC image numbers are date-prefixed and monotonically increasing with
    filing time, so the largest value is the most recently filed version of
    that period's report. This is an approximation (unlike
    load_ie_transactions_dated's exact file_num/prev_file_num chain
    resolution), stated as such rather than assumed exact.

    Returns DataFrame with columns: district_id, party, cycle,
    fec_candidate_id, committee_id, coverage_start_date (datetime64),
    coverage_end_date (datetime64), receipts_period (float),
    disbursements_period (float), cash_on_hand_end_period (float).
    """
    path = config.raw_path("fec") / f"candidate_periodic_reports_{cycle}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run "
            f"`python scripts/fetch_data.py --only fec-periodic --cycles {cycle} "
            "--fec-api-key YOUR_KEY` first (requires a registered FEC API key)."
        )
    df = pd.read_csv(path, dtype={"district_id": str, "party": str,
                                   "fec_candidate_id": str, "committee_id": str,
                                   "beginning_image_number": str})
    df["coverage_start_date"] = pd.to_datetime(df["coverage_start_date"], errors="coerce")
    df["coverage_end_date"] = pd.to_datetime(df["coverage_end_date"], errors="coerce")
    for col in ["receipts_period", "disbursements_period", "cash_on_hand_end_period"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    n_before = len(df)
    df = df[df["coverage_end_date"].notna()].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(
            f"load_candidate_periodic_reports({cycle}): dropped {n_dropped}/{n_before} "
            "rows with an unparseable coverage_end_date."
        )

    # beginning_image_number must stay a string end-to-end: read with dtype=str
    # above (not inferred) because NaNs elsewhere in the column otherwise force
    # float64 on the whole column, silently truncating these 18-digit FEC image
    # numbers to ~17 significant digits and rendering them in scientific
    # notation -- which corrupts the "most recent amendment" sort below.
    df["_img_sort_key"] = df["beginning_image_number"].fillna("")
    df = (
        df.sort_values("_img_sort_key")
        .drop_duplicates(
            subset=["committee_id", "coverage_start_date", "coverage_end_date"], keep="last"
        )
        .drop(columns=["_img_sort_key"])
    )
    return df[["district_id", "party", "cycle", "fec_candidate_id", "committee_id",
               "coverage_start_date", "coverage_end_date", "receipts_period",
               "disbursements_period", "cash_on_hand_end_period"]]


def cumulative_candidate_spend_as_of(cycle: int, as_of_date) -> pd.DataFrame:
    """
    Return cumulative candidate-committee disbursements per (district,
    party), summing every periodic report with coverage_end_date <=
    as_of_date — the dated analog of load_candidate_disbursements() for
    point-in-time reconstruction (dynamic/simulate.py), replacing what was
    previously held fixed at the cycle-final total for every period.

    Returns DataFrame with columns: district_id, party, cycle, disb_cum.
    """
    as_of = pd.Timestamp(as_of_date)
    reports = load_candidate_periodic_reports(cycle)
    cum = reports[reports["coverage_end_date"] <= as_of]
    out = (
        cum.groupby(["district_id", "party"])["disbursements_period"]
        .sum()
        .reset_index()
        .rename(columns={"disbursements_period": "disb_cum"})
    )
    out["cycle"] = cycle
    return out[["district_id", "party", "cycle", "disb_cum"]]


def cumulative_candidate_receipts_as_of(cycle: int, as_of_date) -> pd.DataFrame:
    """
    Return cumulative candidate-committee receipts per (district, party) as
    of as_of_date — same shape as cumulative_candidate_spend_as_of, for
    fundraising-velocity features.

    Returns DataFrame with columns: district_id, party, cycle, receipts_cum.
    """
    as_of = pd.Timestamp(as_of_date)
    reports = load_candidate_periodic_reports(cycle)
    cum = reports[reports["coverage_end_date"] <= as_of]
    out = (
        cum.groupby(["district_id", "party"])["receipts_period"]
        .sum()
        .reset_index()
        .rename(columns={"receipts_period": "receipts_cum"})
    )
    out["cycle"] = cycle
    return out[["district_id", "party", "cycle", "receipts_cum"]]


def cash_on_hand_as_of(cycle: int, as_of_date) -> pd.DataFrame:
    """
    Return each (district, party) candidate committee's most recently
    reported cash_on_hand_end_period at or before as_of_date — populates
    dynamic/state.py's long-stubbed RaceState.cash_on_hand_d field, which
    previously had no data source anywhere in this repo.

    Districts/parties with no report on or before as_of_date are absent
    from the result (not zero-filled) — a candidate with no filing yet as
    of a given date has genuinely unknown cash on hand, not $0.

    Returns DataFrame with columns: district_id, party, cycle, cash_on_hand.
    """
    as_of = pd.Timestamp(as_of_date)
    reports = load_candidate_periodic_reports(cycle)
    cum = reports[reports["coverage_end_date"] <= as_of]
    if not len(cum):
        return pd.DataFrame(columns=["district_id", "party", "cycle", "cash_on_hand"])
    latest = (
        cum.sort_values("coverage_end_date")
        .groupby(["district_id", "party"])
        .last()
        .reset_index()
        .rename(columns={"cash_on_hand_end_period": "cash_on_hand"})
    )
    latest["cycle"] = cycle
    return latest[["district_id", "party", "cycle", "cash_on_hand"]]


def spend_velocity(cycle: int, as_of_date, window_days: int = 30) -> pd.DataFrame:
    """
    Return each (district, party) candidate committee's average daily
    disbursement rate over the trailing window_days ending at as_of_date —
    "candidate spending velocity" (paper3_draft.md's feedback catalog).

    A report whose coverage window only partially overlaps
    [as_of_date - window_days, as_of_date] contributes its full
    disbursements_period (no intra-report proration — FEC periodic reports
    don't have finer-grained dating than their own coverage window, so this
    is the finest resolution available), divided by window_days to express
    a $/day rate.

    Returns DataFrame with columns: district_id, party, cycle,
    disb_velocity_per_day.
    """
    as_of = pd.Timestamp(as_of_date)
    window_start = as_of - pd.Timedelta(days=window_days)
    reports = load_candidate_periodic_reports(cycle)
    windowed = reports[
        (reports["coverage_end_date"] > window_start) & (reports["coverage_end_date"] <= as_of)
    ]
    out = (
        windowed.groupby(["district_id", "party"])["disbursements_period"]
        .sum()
        .reset_index()
    )
    out["disb_velocity_per_day"] = out["disbursements_period"] / window_days
    out["cycle"] = cycle
    return out[["district_id", "party", "cycle", "disb_velocity_per_day"]]


def receipts_velocity(cycle: int, as_of_date, window_days: int = 30) -> pd.DataFrame:
    """
    Fundraising-velocity analog of spend_velocity — average daily receipts
    rate over the trailing window_days ending at as_of_date.

    Returns DataFrame with columns: district_id, party, cycle,
    receipts_velocity_per_day.
    """
    as_of = pd.Timestamp(as_of_date)
    window_start = as_of - pd.Timedelta(days=window_days)
    reports = load_candidate_periodic_reports(cycle)
    windowed = reports[
        (reports["coverage_end_date"] > window_start) & (reports["coverage_end_date"] <= as_of)
    ]
    out = (
        windowed.groupby(["district_id", "party"])["receipts_period"]
        .sum()
        .reset_index()
    )
    out["receipts_velocity_per_day"] = out["receipts_period"] / window_days
    out["cycle"] = cycle
    return out[["district_id", "party", "cycle", "receipts_velocity_per_day"]]
