"""Profiling: shapes, missingness, durations, track inventory.

Track names are preserved EXACTLY as returned by /trks.  No renaming,
no fabrication.
"""

import logging

import pandas as pd

from vitaldb_audit import config

logger = logging.getLogger("vitaldb_audit.profile")

_DURATION_SPANS = {
    "case_minutes": ("casestart", "caseend"),
    "anesthesia_minutes": ("anestart", "aneend"),
    "operation_minutes": ("opstart", "opend"),
}

_DESCRIBE_KEYS = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]


def build_track_inventory(trks: pd.DataFrame) -> pd.DataFrame:
    """One row per REAL track name, with device/signal split and coverage.

    Track names are never rewritten: ``tname`` is carried through verbatim and
    ``device``/``signal`` are additive columns derived from the single "/".
    """
    total_cases = trks["caseid"].nunique()
    grouped = (
        trks.groupby("tname")["caseid"]
        .nunique()
        .reset_index(name="n_cases")
    )
    split = grouped["tname"].str.split("/", n=1, expand=True)
    grouped["device"] = split[0]
    grouped["signal"] = split[1]
    grouped["pct_cases"] = (grouped["n_cases"] / total_cases * 100).round(2)

    inventory = grouped[["tname", "device", "signal", "n_cases", "pct_cases"]]
    inventory = inventory.sort_values(
        ["n_cases", "tname"], ascending=[False, True]
    ).reset_index(drop=True)
    logger.info(
        "track inventory: %d distinct names across %d cases",
        len(inventory), total_cases,
    )
    return inventory


def profile_clinical_missingness(cases: pd.DataFrame) -> pd.DataFrame:
    """Per-column missingness and cardinality for the clinical table."""
    n_rows = len(cases)
    rows = []
    for column in cases.columns:
        series = cases[column]
        n_missing = int(series.isna().sum())
        rows.append({
            "column": column,
            "dtype": str(series.dtype),
            "n_missing": n_missing,
            "pct_missing": round(n_missing / n_rows * 100, 2) if n_rows else 0.0,
            "n_unique": int(series.nunique(dropna=True)),
        })
    frame = pd.DataFrame(
        rows, columns=["column", "dtype", "n_missing", "pct_missing", "n_unique"]
    )
    logger.info(
        "clinical missingness: %d columns, %d fully populated",
        len(frame), int((frame["n_missing"] == 0).sum()),
    )
    return frame


def compute_duration_stats(cases: pd.DataFrame) -> dict:
    """Case / anaesthesia / operation duration statistics, in minutes.

    VitalDB timestamps are seconds relative to the case start.  Spans whose
    endpoints are missing are dropped rather than imputed.
    """
    stats = {}
    for label, (start_col, end_col) in _DURATION_SPANS.items():
        if start_col not in cases.columns or end_col not in cases.columns:
            logger.warning("missing columns for %s; recording as unavailable", label)
            stats[label] = {"available": False}
            continue
        minutes = ((cases[end_col] - cases[start_col]) / 60.0).dropna()
        described = minutes.describe()
        stats[label] = {
            key: round(float(described[key]), 2)
            for key in _DESCRIBE_KEYS
            if key in described
        }
    return stats


def build_lab_inventory(labs: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct lab test name, with row and case coverage."""
    total_cases = labs["caseid"].nunique()
    inventory = (
        labs.groupby("name")
        .agg(n_rows=("name", "size"), n_cases=("caseid", "nunique"))
        .reset_index()
    )
    inventory["pct_cases"] = (
        inventory["n_cases"] / total_cases * 100
    ).round(2) if total_cases else 0.0
    inventory = inventory.sort_values(
        ["n_rows", "name"], ascending=[False, True]
    ).reset_index(drop=True)
    logger.info("lab inventory: %d distinct tests", len(inventory))
    return inventory


def tracks_per_case_stats(trks: pd.DataFrame) -> dict:
    """Distribution of how many distinct tracks each case carries."""
    per_case = trks.groupby("caseid")["tname"].nunique()
    described = per_case.describe()
    return {
        key: round(float(described[key]), 2)
        for key in _DESCRIBE_KEYS
        if key in described
    }
