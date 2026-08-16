"""Tests for config module."""

from pathlib import Path

from vitaldb_audit import config


def test_endpoints_are_the_three_documented_metadata_urls():
    assert config.ENDPOINTS == {
        "cases": "https://api.vitaldb.net/cases",
        "trks": "https://api.vitaldb.net/trks",
        "labs": "https://api.vitaldb.net/labs",
    }


def test_required_panel_is_the_eight_verified_tracks():
    assert config.REQUIRED_PANEL == [
        "Solar8000/HR",
        "Solar8000/PLETH_SPO2",
        "Solar8000/ART_MBP",
        "Solar8000/RR_CO2",
        "Solar8000/BT",
        "SNUADC/ECG_II",
        "SNUADC/ART",
        "SNUADC/PLETH",
    ]


def test_output_dirs_are_three_way_separated():
    assert config.RAW_DIR.name == "raw"
    assert config.RAW_DIR.parent.name == "data"
    assert config.PROCESSED_DIR.name == "processed"
    assert config.PROCESSED_DIR.parent.name == "data"
    assert config.RESULTS_DIR.name == "results"


def test_ensure_dirs_creates_all_three(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    config.ensure_dirs()
    assert (tmp_path / "data" / "raw").is_dir()
    assert (tmp_path / "data" / "processed").is_dir()
    assert (tmp_path / "results").is_dir()


def test_signal_suffix_map_covers_every_requested_physiological_family():
    assert set(config.SIGNAL_SUFFIXES) == {
        "heart_rate",
        "arterial_bp_invasive",
        "blood_pressure_noninvasive",
        "spo2_oximetry",
        "respiratory_rate",
        "temperature",
        "ecg_derived",
    }
