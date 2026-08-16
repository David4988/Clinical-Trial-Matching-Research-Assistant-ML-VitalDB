"""Tests for the profile module — fully offline via fixtures."""

import pandas as pd

from vitaldb_audit import profile


def test_track_inventory_splits_device_and_signal(trks_df):
    inv = profile.build_track_inventory(trks_df)
    row = inv[inv["tname"] == "Solar8000/HR"].iloc[0]
    assert row["device"] == "Solar8000"
    assert row["signal"] == "HR"


def test_track_inventory_counts_distinct_cases(trks_df):
    inv = profile.build_track_inventory(trks_df)
    # Solar8000/HR appears in cases 1-6; SNUADC/ART only in 1-5.
    assert inv.set_index("tname").loc["Solar8000/HR", "n_cases"] == 6
    assert inv.set_index("tname").loc["SNUADC/ART", "n_cases"] == 5


def test_track_inventory_is_sorted_by_coverage_descending(trks_df):
    inv = profile.build_track_inventory(trks_df)
    assert inv["n_cases"].is_monotonic_decreasing


def test_track_inventory_pct_is_relative_to_cases_with_tracks(trks_df):
    inv = profile.build_track_inventory(trks_df)
    row = inv[inv["tname"] == "Solar8000/HR"].iloc[0]
    assert row["pct_cases"] == 100.0


def test_clinical_missingness_reports_fully_missing_column(cases_df):
    miss = profile.profile_clinical_missingness(cases_df)
    row = miss[miss["column"] == "cline2"].iloc[0]
    assert row["n_missing"] == 5
    assert row["pct_missing"] == 100.0


def test_clinical_missingness_covers_every_column(cases_df):
    miss = profile.profile_clinical_missingness(cases_df)
    assert len(miss) == cases_df.shape[1]
    assert set(miss.columns) == {
        "column", "dtype", "n_missing", "pct_missing", "n_unique",
    }


def test_duration_stats_are_in_minutes(cases_df):
    stats = profile.compute_duration_stats(cases_df)
    # caseid 1 runs 6000 s = 100 min; caseid 5 runs 42000 s = 700 min.
    assert stats["case_minutes"]["min"] == 30.0
    assert stats["case_minutes"]["max"] == 700.0
    assert stats["case_minutes"]["count"] == 5


def test_duration_stats_include_anesthesia_and_operation(cases_df):
    stats = profile.compute_duration_stats(cases_df)
    assert set(stats) == {"case_minutes", "anesthesia_minutes", "operation_minutes"}


def test_lab_inventory_counts_rows_and_cases(labs_df):
    inv = profile.build_lab_inventory(labs_df)
    alb = inv[inv["name"] == "alb"].iloc[0]
    assert alb["n_rows"] == 2
    assert alb["n_cases"] == 1
    hb = inv[inv["name"] == "hb"].iloc[0]
    assert hb["n_rows"] == 2
    assert hb["n_cases"] == 2


def test_tracks_per_case_stats_reports_count(trks_df):
    stats = profile.tracks_per_case_stats(trks_df)
    assert stats["count"] == 6  # 6 distinct cases in the fixture
