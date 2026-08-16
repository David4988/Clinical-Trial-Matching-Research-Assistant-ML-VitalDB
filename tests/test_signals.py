"""Tests for the signals (classification) module — fully offline."""

import pandas as pd

from vitaldb_audit import config, profile, signals


def test_classify_real_track_names():
    assert signals.classify_track("Solar8000/HR") == "heart_rate"
    assert signals.classify_track("SNUADC/ART") == "arterial_bp_invasive"
    assert signals.classify_track("Solar8000/NIBP_MBP") == "blood_pressure_noninvasive"
    assert signals.classify_track("Solar8000/PLETH_SPO2") == "spo2_oximetry"
    assert signals.classify_track("Solar8000/RR_CO2") == "respiratory_rate"
    assert signals.classify_track("Solar8000/BT") == "temperature"
    assert signals.classify_track("SNUADC/ECG_II") == "ecg_derived"


def test_classify_returns_none_for_non_physiological_tracks():
    assert signals.classify_track("Orchestra/PPF20_RATE") is None
    assert signals.classify_track("Primus/SET_AGE") is None
    assert signals.classify_track("BIS/SQI") is None


def test_classify_handles_name_without_slash():
    assert signals.classify_track("HR") == "heart_rate"


def test_candidates_only_contain_names_present_in_the_inventory(trks_df):
    inv = profile.build_track_inventory(trks_df)
    cand = signals.build_physiological_candidates(inv)
    assert set(cand["tname"]).issubset(set(inv["tname"]))


def test_candidates_exclude_unclassified_tracks(trks_df):
    inv = profile.build_track_inventory(trks_df)
    cand = signals.build_physiological_candidates(inv)
    assert cand["category"].notna().all()


def test_fluid_warmer_temperature_is_flagged_as_not_a_patient_signal():
    inv = pd.DataFrame([
        {"tname": "FMS/INPUT_TEMP", "device": "FMS", "signal": "INPUT_TEMP",
         "n_cases": 15, "pct_cases": 0.23},
        {"tname": "Solar8000/BT", "device": "Solar8000", "signal": "BT",
         "n_cases": 5917, "pct_cases": 92.63},
    ])
    cand = signals.build_physiological_candidates(inv)
    flags = cand.set_index("tname")["is_patient_signal"]
    assert flags["FMS/INPUT_TEMP"] == False
    assert flags["Solar8000/BT"] == True


def test_unmatched_suffixes_flags_a_stale_map():
    # Only one suffix present → many configured suffixes will be "unmatched"
    inv = pd.DataFrame([
        {"tname": "Solar8000/HR", "device": "Solar8000", "signal": "HR",
         "n_cases": 10, "pct_cases": 100.0},
    ])
    unmatched = signals.unmatched_suffixes(inv)
    # HR is present, so "heart_rate" should still report HR_AVG and PLETH_HR missing
    assert "PLETH_HR" in unmatched.get("heart_rate", [])


def test_unmatched_is_empty_when_all_suffixes_present():
    # Build a fake inventory with EVERY configured suffix
    rows = []
    for category, suffixes in config.SIGNAL_SUFFIXES.items():
        for suffix in suffixes:
            rows.append({
                "tname": f"Device/{suffix}", "device": "Device",
                "signal": suffix, "n_cases": 1, "pct_cases": 1.0,
            })
    unmatched = signals.unmatched_suffixes(pd.DataFrame(rows))
    assert unmatched == {}


def test_category_coverage_reports_best_track_per_category(trks_df):
    inv = profile.build_track_inventory(trks_df)
    cov = signals.category_coverage(
        signals.build_physiological_candidates(inv)
    )
    row = cov.set_index("category").loc["heart_rate"]
    assert row["best_tname"] == "Solar8000/HR"
    assert row["best_n_cases"] == 6
