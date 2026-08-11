"""Real-data check that the control-vs-alignment decomposition
(src/estimation/control_provenance.py) is a strict re-partition of
d_total/r_total, not an approximation -- and that x_D/x_R (party_natl) is
meaningfully smaller than "everything non-candidate" once outside-group IEs
are excluded. Not marked slow: build_provenance_table() only reads already-
fetched CSVs (no SLSQP solve), ~1-2s."""

import pytest

from estimation.control_provenance import build_provenance_table


@pytest.fixture(scope="module")
def prov_2024():
    return build_provenance_table(2024)


def test_accounting_identity_holds(prov_2024):
    df = prov_2024
    assert (df["d_total_check"] - df["d_total"]).abs().max() < 1.0
    assert (df["r_total_check"] - df["r_total"]).abs().max() < 1.0


def test_national_committee_money_is_a_minority_of_non_candidate_spend(prov_2024):
    df = prov_2024
    party_natl_r = df["party_natl_r"].sum()
    non_candidate_r = df["party_natl_r"].sum() + df["party_state_r"].sum() + df["outside_r"].sum()
    assert 0 < party_natl_r < non_candidate_r
    assert party_natl_r / non_candidate_r < 0.6  # outside spending dominates the non-candidate total
