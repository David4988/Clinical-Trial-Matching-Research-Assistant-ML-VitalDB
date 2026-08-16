"""Validation checks for synthetic clinical-trial data.

Every check returns a dict with keys:
    name:    short identifier
    passed:  bool
    detail:  human-readable explanation

No clinical validity is claimed.  These checks enforce structural integrity
and internal consistency of the synthetic dataset.
"""

import numpy as np
import pandas as pd

from generator import (
    SIGNAL_BOUNDS, SIGNAL_NAMES, SCENARIOS,
    WINDOW_MINUTES, data_quality_label,
)


def _check(name, passed, detail=""):
    return {"name": name, "passed": bool(passed), "detail": detail}


# ── Individual checks ────────────────────────────────────────────────────────


def check_no_duplicate_patient_ids(tables):
    df = tables["patients"]
    n_unique = df["patient_id"].nunique()
    ok = n_unique == len(df)
    return _check("no_duplicate_patient_ids", ok,
                  f"{n_unique} unique of {len(df)} rows")


def check_no_null_critical_ids(tables):
    issues = []
    for name, df in tables.items():
        if "patient_id" in df.columns and df["patient_id"].isna().any():
            issues.append(f"{name}: null patient_id")
        if "trial_id" in df.columns and df["trial_id"].isna().any():
            issues.append(f"{name}: null trial_id")
    ok = len(issues) == 0
    return _check("no_null_critical_ids", ok,
                  "clean" if ok else "; ".join(issues))


def check_no_duplicate_timestamps(tables):
    df = tables["observations"]
    dupes = df.groupby("patient_id")["timestamp"].apply(
        lambda s: s.duplicated().any()
    )
    n_bad = int(dupes.sum())
    return _check("no_duplicate_timestamps_per_patient",
                  n_bad == 0,
                  f"{n_bad} patients with duplicate timestamps")


def check_dose_numbers_increase(tables):
    df = tables["observations"]
    issues = 0
    for pid, group in df.groupby("patient_id"):
        doses = group.sort_values("timestamp")["dose_number"].values
        if not np.all(np.diff(doses) >= 0):
            issues += 1
    return _check("dose_numbers_increase", issues == 0,
                  f"{issues} patients with non-monotonic dose numbers")


def check_no_broken_timelines(tables):
    df = tables["observations"]
    issues = 0
    for pid, group in df.groupby("patient_id"):
        timestamps = group.sort_values("timestamp")["timestamp"].values
        diffs = np.diff(timestamps)
        if not np.all(diffs == WINDOW_MINUTES):
            issues += 1
    return _check("no_broken_timelines", issues == 0,
                  f"{issues} patients with timeline gaps")


def check_physiological_ranges(tables):
    df = tables["observations"]
    col_map = {
        "heart_rate":       "heart_rate",
        "spo2":             "spo2",
        "respiratory_rate": "respiratory_rate",
        "systolic_bp":      "systolic_bp",
        "diastolic_bp":     "diastolic_bp",
        "temperature":      "temperature",
    }
    violations = []
    for signal, col in col_map.items():
        lo, hi = SIGNAL_BOUNDS[signal]
        below = int((df[col] < lo).sum())
        above = int((df[col] > hi).sum())
        if below or above:
            violations.append(f"{col}: {below} below {lo}, {above} above {hi}")
    ok = len(violations) == 0
    return _check("physiological_ranges", ok,
                  "all within bounds" if ok else "; ".join(violations))


def check_scenario_labels_valid(tables):
    obs = tables["observations"]
    patients = tables["patients"]
    invalid_obs = set(obs["scenario"].unique()) - set(SCENARIOS)
    invalid_pat = set(patients["scenario"].unique()) - set(SCENARIOS)
    ok = len(invalid_obs) == 0 and len(invalid_pat) == 0
    return _check("scenario_labels_valid", ok,
                  "all valid" if ok else
                  f"invalid obs: {invalid_obs}, patients: {invalid_pat}")


def check_data_quality_failure_coverage(tables):
    df = tables["observations"]
    dq = df[df["scenario"] == "DATA_QUALITY_FAILURE"]
    if dq.empty:
        return _check("dq_failure_coverage", True,
                      "no DATA_QUALITY_FAILURE patients (skipped)")
    # Each DQ patient should have at least some low-coverage windows
    issues = 0
    for pid, group in dq.groupby("patient_id"):
        min_cov = group["coverage_percent"].min()
        if min_cov > 50:
            issues += 1
    ok = issues == 0
    return _check("dq_failure_coverage", ok,
                  f"{issues} DQ patients without any low-coverage windows"
                  if not ok else
                  f"all {dq['patient_id'].nunique()} DQ patients have "
                  f"low-coverage windows")


def check_events_reference_existing(tables):
    patients = set(tables["patients"]["patient_id"])
    trials = set(tables["trial_assignments"]["trial_id"])
    events = tables["events"]
    if events.empty:
        return _check("events_reference_existing", True, "no events to check")
    bad_patients = set(events["patient_id"]) - patients
    bad_trials = set(events["trial_id"]) - trials
    ok = len(bad_patients) == 0 and len(bad_trials) == 0
    return _check("events_reference_existing", ok,
                  "all valid" if ok else
                  f"orphan patients: {bad_patients}, trials: {bad_trials}")


def check_scenario_consistency(tables):
    """Verify that patient scenario matches their observations' scenario."""
    patients = tables["patients"].set_index("patient_id")["scenario"]
    obs = tables["observations"][["patient_id", "scenario"]].drop_duplicates()
    mismatches = 0
    for _, row in obs.iterrows():
        if row["patient_id"] in patients.index:
            if patients[row["patient_id"]] != row["scenario"]:
                mismatches += 1
    return _check("scenario_consistency", mismatches == 0,
                  f"{mismatches} mismatches between patient and obs scenarios")


def check_deterministic_regeneration(tables, generator_class, seed, n_patients):
    """Regenerate with the same seed and verify identical output."""
    from generator import SyntheticTrialGenerator
    gen2 = SyntheticTrialGenerator(n_patients=n_patients, seed=seed)
    tables2 = gen2.generate()

    mismatches = []
    for name in ["patients", "trial_assignments", "observations", "events"]:
        df1 = tables[name].reset_index(drop=True)
        df2 = tables2[name].reset_index(drop=True)
        if df1.shape != df2.shape:
            mismatches.append(f"{name}: shape {df1.shape} vs {df2.shape}")
        elif not df1.equals(df2):
            mismatches.append(f"{name}: content differs")

    ok = len(mismatches) == 0
    return _check("deterministic_regeneration", ok,
                  "identical" if ok else "; ".join(mismatches))


# ── Runner ───────────────────────────────────────────────────────────────────


ALL_CHECKS = [
    check_no_duplicate_patient_ids,
    check_no_null_critical_ids,
    check_no_duplicate_timestamps,
    check_dose_numbers_increase,
    check_no_broken_timelines,
    check_physiological_ranges,
    check_scenario_labels_valid,
    check_data_quality_failure_coverage,
    check_events_reference_existing,
    check_scenario_consistency,
]


def validate_dataset(tables, seed=None, n_patients=None):
    """Run all validation checks.

    Returns a list of check-result dicts.  If *seed* and *n_patients* are
    provided, also runs the deterministic-regeneration check.
    """
    results = [check(tables) for check in ALL_CHECKS]

    if seed is not None and n_patients is not None:
        from generator import SyntheticTrialGenerator
        results.append(check_deterministic_regeneration(
            tables, SyntheticTrialGenerator, seed, n_patients))

    return results
