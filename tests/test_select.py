"""Tests for the select module — fully offline via fixtures."""

import pytest

from vitaldb_audit import select


def test_build_case_track_sets_groups_by_case(trks_df):
    sets = select.build_case_track_sets(trks_df)
    assert sets[6] == {"Solar8000/HR", "Solar8000/PLETH_SPO2"}
    assert "SNUADC/ART" in sets[1]


def test_score_cases_marks_full_panel(trks_df, cases_df):
    scored = select.score_cases(select.build_case_track_sets(trks_df), cases_df)
    by_case = scored.set_index("caseid")
    assert bool(by_case.loc[1, "has_full_panel"]) is True
    assert bool(by_case.loc[6, "has_full_panel"]) is False


def test_score_cases_computes_duration_band(trks_df, cases_df):
    scored = select.score_cases(select.build_case_track_sets(trks_df), cases_df)
    by_case = scored.set_index("caseid")
    assert by_case.loc[1, "case_minutes"] == 100.0
    assert bool(by_case.loc[1, "in_duration_band"]) is True
    # caseid 4 is 30 min → too short
    assert bool(by_case.loc[4, "in_duration_band"]) is False
    # caseid 5 is 700 min → too long
    assert bool(by_case.loc[5, "in_duration_band"]) is False


def test_score_cases_marks_qualifying(trks_df, cases_df):
    scored = select.score_cases(select.build_case_track_sets(trks_df), cases_df)
    by_case = scored.set_index("caseid")
    # caseid 1-3: full panel AND in band → qualifies
    assert bool(by_case.loc[1, "qualifies"]) is True
    assert bool(by_case.loc[2, "qualifies"]) is True
    assert bool(by_case.loc[3, "qualifies"]) is True
    # caseid 4: full panel but too short → does not qualify
    assert bool(by_case.loc[4, "qualifies"]) is False
    # caseid 6: no full panel → does not qualify
    assert bool(by_case.loc[6, "qualifies"]) is False


def test_select_candidates_picks_top_n(trks_df, cases_df):
    scored = select.score_cases(select.build_case_track_sets(trks_df), cases_df)
    selected = select.select_candidates(scored, n=2)
    assert len(selected) == 2
    # caseid 1 has the most tracks (10), then caseid 2 (9)
    assert selected == [1, 2]


def test_select_candidates_is_deterministic(trks_df, cases_df):
    scored = select.score_cases(select.build_case_track_sets(trks_df), cases_df)
    a = select.select_candidates(scored, n=3)
    b = select.select_candidates(scored, n=3)
    assert a == b == [1, 2, 3]


def test_select_candidates_warns_when_fewer_than_requested(trks_df, cases_df):
    scored = select.score_cases(select.build_case_track_sets(trks_df), cases_df)
    # Only 3 qualifying cases in the fixture; requesting 10
    selected = select.select_candidates(scored, n=10)
    assert len(selected) == 3


def test_select_candidates_raises_on_no_qualifying(trks_df, cases_df):
    scored = select.score_cases(select.build_case_track_sets(trks_df), cases_df)
    # Remove all qualifying cases
    no_qualifying = scored[~scored["qualifies"]].copy()
    no_qualifying = no_qualifying.copy()
    no_qualifying["qualifies"] = False
    with pytest.raises(select.NoCandidatesError):
        select.select_candidates(no_qualifying, n=3)
