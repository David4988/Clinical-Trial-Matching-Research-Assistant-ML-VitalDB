"""Paths, endpoints, panel definition, and signal suffix map.

Every track name and suffix below was confirmed present in the live /trks
endpoint during planning (2026-08-16). Nothing here is invented.
"""

from pathlib import Path

# ── API ──────────────────────────────────────────────────────────────────────
API_URL = "https://api.vitaldb.net"
DATASET_VERSION = "1.0.1"

ENDPOINTS = {
    "cases": f"{API_URL}/cases",
    "trks": f"{API_URL}/trks",
    "labs": f"{API_URL}/labs",
}

# ── Directories ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── 8-track required panel ───────────────────────────────────────────────────
# A case must carry all eight of these to qualify for deeper inspection.
# Verified against /trks: 3322 of 6388 cases satisfy this panel.
REQUIRED_PANEL = [
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
    "Solar8000/ART_MBP",
    "Solar8000/RR_CO2",
    "Solar8000/BT",
    "SNUADC/ECG_II",
    "SNUADC/ART",
    "SNUADC/PLETH",
]

# ── Signal suffix map ────────────────────────────────────────────────────────
# Maps a physiological family to the set of REAL track-name suffixes that
# belong to it.  Suffix = the segment after "DEVICE/".  Every entry below was
# confirmed present in /trks during planning.  Nothing here is invented; if a
# suffix stops matching, signals.unmatched_suffixes() will surface it.
SIGNAL_SUFFIXES: dict[str, set[str]] = {
    "heart_rate": {"HR", "PLETH_HR", "HR_AVG"},
    "arterial_bp_invasive": {
        "ART", "ART_SBP", "ART_DBP", "ART_MBP",
        "FEM", "FEM_SBP", "FEM_DBP", "FEM_MBP",
    },
    "blood_pressure_noninvasive": {"NIBP_SBP", "NIBP_DBP", "NIBP_MBP"},
    "spo2_oximetry": {"PLETH", "PLETH_SPO2"},
    "respiratory_rate": {"RR", "RR_CO2", "VENT_RR"},
    # FMS/* temps are fluid-warmer device temperatures, NOT patient body
    # temperature.  They are classified here but flagged in the audit report.
    "temperature": {
        "BT", "BT_PA",
        "INPUT_TEMP", "OUTPUT_TEMP", "INPUT_AMB_TEMP", "OUTPUT_AMB_TEMP",
    },
    "ecg_derived": {
        "ECG_II", "ECG_V5",
        "ST_I", "ST_II", "ST_III", "ST_AVR", "ST_AVL", "ST_AVF", "ST_V5",
    },
}

# Device prefixes whose temperature tracks measure equipment, not the patient.
NON_PATIENT_TEMPERATURE_DEVICES = {"FMS"}

# ── Selection parameters ─────────────────────────────────────────────────────
N_CANDIDATE_CASES = 8
RANDOM_SEED = 20260816

# Candidate cases are restricted to a sane duration band so that a pathological
# 27-minute or 17-hour case does not win on track count alone.
# IMPORTANT: This is a hackathon case-selection heuristic, NOT a VitalDB rule.
MIN_CASE_MINUTES = 60.0
MAX_CASE_MINUTES = 600.0

# ── Probe parameters ─────────────────────────────────────────────────────────
# Bytes read per track when --probe-signals is active.  200 KB is enough for
# ~20k waveform rows, far more than needed to establish a sampling interval.
PROBE_MAX_BYTES = 200_000


def ensure_dirs() -> None:
    """Create the raw/processed/results tree if it does not already exist."""
    for directory in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
