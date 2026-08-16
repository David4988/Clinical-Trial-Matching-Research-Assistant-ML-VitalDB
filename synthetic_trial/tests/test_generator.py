"""Tests for the synthetic clinical-trial generator.

Uses a small patient count (50) for speed.  Verifies determinism, schema,
scenario behaviour, and structural integrity.
"""

import numpy as np
import pytest

from generator import (
    SCENARIOS, SIGNAL_BOUNDS, SIGNAL_NAMES,
    WINDOW_MINUTES, WINDOWS_PER_DOSE, DOSES_PER_PATIENT,
    SyntheticTrialGenerator, data_quality_label,
)
from validate import validate_dataset

SEED = 20260817
N_PATIENTS = 50
TOTAL_WINDOWS = DOSES_PER_PATIENT * WINDOWS_PER_DOSE  # 144


@pytest.fixture(scope="module")
def tables():
    gen = SyntheticTrialGenerator(n_patients=N_PATIENTS, seed=SEED)
    return gen.generate()


# ── Determinism ──────────────────────────────────────────────────────────────


class TestDeterminism:

    def test_same_seed_same_output(self):
        gen1 = SyntheticTrialGenerator(n_patients=20, seed=42)
        gen2 = SyntheticTrialGenerator(n_patients=20, seed=42)
        t1 = gen1.generate()
        t2 = gen2.generate()
        for name in ["patients", "trial_assignments", "observations", "events"]:
            assert t1[name].equals(t2[name]), f"{name} differs"

    def test_different_seed_different_output(self):
        gen1 = SyntheticTrialGenerator(n_patients=20, seed=1)
        gen2 = SyntheticTrialGenerator(n_patients=20, seed=2)
        t1 = gen1.generate()
        t2 = gen2.generate()
        assert not t1["observations"]["heart_rate"].equals(
            t2["observations"]["heart_rate"])


# ── Schema / shape ───────────────────────────────────────────────────────────


class TestSchema:

    def test_patients_shape(self, tables):
        assert len(tables["patients"]) == N_PATIENTS

    def test_patients_columns(self, tables):
        required = {"patient_id", "age", "sex", "baseline_hr", "baseline_spo2",
                     "baseline_rr", "baseline_sbp", "baseline_dbp",
                     "baseline_temperature", "scenario", "bmi",
                     "condition_category"}
        assert required.issubset(set(tables["patients"].columns))

    def test_assignments_shape(self, tables):
        expected = N_PATIENTS * DOSES_PER_PATIENT
        assert len(tables["trial_assignments"]) == expected

    def test_assignments_columns(self, tables):
        required = {"patient_id", "trial_id", "treatment_arm", "drug_id",
                     "dose_number", "dose_amount", "enrollment_time", "scenario"}
        assert required.issubset(set(tables["trial_assignments"].columns))

    def test_observations_shape(self, tables):
        expected = N_PATIENTS * TOTAL_WINDOWS
        assert len(tables["observations"]) == expected

    def test_observations_columns(self, tables):
        required = {"patient_id", "trial_id", "timestamp", "dose_number",
                     "heart_rate", "spo2", "respiratory_rate",
                     "systolic_bp", "diastolic_bp", "temperature",
                     "scenario", "coverage_percent", "data_quality",
                     "ground_truth_state"}
        assert required.issubset(set(tables["observations"].columns))

    def test_events_columns(self, tables):
        required = {"patient_id", "trial_id", "timestamp", "event_type",
                     "severity", "scenario", "description"}
        assert required.issubset(set(tables["events"].columns))


# ── Scenario distribution ────────────────────────────────────────────────────


class TestScenarios:

    def test_all_scenarios_present(self, tables):
        present = set(tables["patients"]["scenario"].unique())
        # With 50 patients and default weights, most scenarios should appear
        assert len(present) >= 5, f"only {present} present"

    def test_scenario_labels_valid(self, tables):
        invalid = set(tables["patients"]["scenario"]) - set(SCENARIOS)
        assert invalid == set()

    def test_scenario_consistent_across_tables(self, tables):
        patient_scenarios = tables["patients"].set_index("patient_id")["scenario"]
        obs_scenarios = tables["observations"][
            ["patient_id", "scenario"]
        ].drop_duplicates()
        for _, row in obs_scenarios.iterrows():
            assert patient_scenarios[row["patient_id"]] == row["scenario"]


# ── Observations integrity ───────────────────────────────────────────────────


class TestObservations:

    def test_no_null_patient_ids(self, tables):
        assert not tables["observations"]["patient_id"].isna().any()

    def test_no_duplicate_timestamps_per_patient(self, tables):
        df = tables["observations"]
        dupes = df.groupby("patient_id")["timestamp"].apply(
            lambda s: s.duplicated().any())
        assert not dupes.any()

    def test_dose_numbers_monotonic(self, tables):
        df = tables["observations"]
        for pid, group in df.groupby("patient_id"):
            doses = group.sort_values("timestamp")["dose_number"].values
            assert np.all(np.diff(doses) >= 0), f"{pid} non-monotonic doses"

    def test_timestamps_are_spaced_correctly(self, tables):
        df = tables["observations"]
        for pid, group in list(df.groupby("patient_id"))[:10]:  # sample
            ts = group.sort_values("timestamp")["timestamp"].values
            diffs = np.diff(ts)
            assert np.all(diffs == WINDOW_MINUTES)

    def test_physiological_ranges(self, tables):
        df = tables["observations"]
        for signal in SIGNAL_NAMES:
            lo, hi = SIGNAL_BOUNDS[signal]
            col = signal
            assert df[col].min() >= lo, f"{col} below {lo}"
            assert df[col].max() <= hi, f"{col} above {hi}"


# ── Scenario-specific behaviour ──────────────────────────────────────────────


class TestScenarioBehaviour:

    def _patients_for_scenario(self, tables, scenario):
        pids = tables["patients"][
            tables["patients"]["scenario"] == scenario
        ]["patient_id"]
        return tables["observations"][
            tables["observations"]["patient_id"].isin(pids)
        ]

    def test_stable_low_variance(self, tables):
        obs = self._patients_for_scenario(tables, "STABLE")
        if obs.empty:
            pytest.skip("no STABLE patients in sample")
        # For each STABLE patient, HR std across all windows should be modest
        for pid, grp in obs.groupby("patient_id"):
            hr_std = grp["heart_rate"].std()
            assert hr_std < 15, f"STABLE {pid} HR std={hr_std:.1f}"

    def test_gradual_deterioration_trend(self, tables):
        obs = self._patients_for_scenario(tables, "GRADUAL_DETERIORATION")
        if obs.empty:
            pytest.skip("no GRADUAL_DETERIORATION patients")
        pid = obs["patient_id"].iloc[0]
        grp = obs[obs["patient_id"] == pid].sort_values("timestamp")
        first_quarter = grp["heart_rate"].iloc[:TOTAL_WINDOWS // 4].mean()
        last_quarter = grp["heart_rate"].iloc[-TOTAL_WINDOWS // 4:].mean()
        # HR should increase over time
        assert last_quarter > first_quarter + 5

    def test_sudden_deterioration_has_change_point(self, tables):
        obs = self._patients_for_scenario(tables, "SUDDEN_DETERIORATION")
        if obs.empty:
            pytest.skip("no SUDDEN_DETERIORATION patients")
        pid = obs["patient_id"].iloc[0]
        grp = obs[obs["patient_id"] == pid].sort_values("timestamp")
        states = grp["ground_truth_state"].tolist()
        assert "normal" in states
        assert "acute_change" in states

    def test_data_quality_failure_has_low_coverage(self, tables):
        obs = self._patients_for_scenario(tables, "DATA_QUALITY_FAILURE")
        if obs.empty:
            pytest.skip("no DATA_QUALITY_FAILURE patients")
        for pid, grp in obs.groupby("patient_id"):
            assert grp["coverage_percent"].min() < 30, \
                f"DQ patient {pid} never drops below 30% coverage"

    def test_recovery_has_three_phases(self, tables):
        obs = self._patients_for_scenario(tables, "RECOVERY")
        if obs.empty:
            pytest.skip("no RECOVERY patients")
        pid = obs["patient_id"].iloc[0]
        grp = obs[obs["patient_id"] == pid]
        states = set(grp["ground_truth_state"])
        assert "deteriorating" in states
        assert "recovering" in states
        assert "recovered" in states

    def test_adverse_event_has_event_record(self, tables):
        ae_patients = set(tables["patients"][
            tables["patients"]["scenario"] == "ADVERSE_EVENT"
        ]["patient_id"])
        if not ae_patients:
            pytest.skip("no ADVERSE_EVENT patients")
        ae_events = tables["events"][
            (tables["events"]["patient_id"].isin(ae_patients))
            & (tables["events"]["event_type"] == "ADVERSE_EVENT")
        ]
        # Every AE patient should have at least one event
        assert set(ae_events["patient_id"]) == ae_patients

    def test_improving_shows_trend(self, tables):
        obs = self._patients_for_scenario(tables, "IMPROVING")
        if obs.empty:
            pytest.skip("no IMPROVING patients")
        pid = obs["patient_id"].iloc[0]
        grp = obs[obs["patient_id"] == pid].sort_values("timestamp")
        states = grp["ground_truth_state"].tolist()
        assert "improving" in states
        assert "improved" in states


# ── Events ───────────────────────────────────────────────────────────────────


class TestEvents:

    def test_events_reference_valid_patients(self, tables):
        if tables["events"].empty:
            pytest.skip("no events")
        valid = set(tables["patients"]["patient_id"])
        event_pids = set(tables["events"]["patient_id"])
        assert event_pids.issubset(valid)

    def test_events_reference_valid_trials(self, tables):
        if tables["events"].empty:
            pytest.skip("no events")
        valid = set(tables["trial_assignments"]["trial_id"])
        event_tids = set(tables["events"]["trial_id"])
        assert event_tids.issubset(valid)

    def test_event_types_valid(self, tables):
        if tables["events"].empty:
            pytest.skip("no events")
        valid = {"ADVERSE_EVENT", "RECOVERY", "DOSE_RESPONSE",
                 "DATA_QUALITY_EVENT"}
        found = set(tables["events"]["event_type"])
        assert found.issubset(valid), f"unexpected types: {found - valid}"


# ── Data quality label ───────────────────────────────────────────────────────


class TestDataQualityLabel:

    def test_good(self):
        assert data_quality_label(100) == "GOOD"
        assert data_quality_label(95) == "GOOD"

    def test_partial(self):
        assert data_quality_label(80) == "PARTIAL"
        assert data_quality_label(70) == "PARTIAL"

    def test_poor(self):
        assert data_quality_label(50) == "POOR"
        assert data_quality_label(30) == "POOR"

    def test_missing(self):
        assert data_quality_label(10) == "MISSING"
        assert data_quality_label(0) == "MISSING"


# ── Validation module integration ────────────────────────────────────────────


class TestValidation:

    def test_all_checks_pass(self, tables):
        results = validate_dataset(tables, seed=SEED, n_patients=N_PATIENTS)
        failures = [r for r in results if not r["passed"]]
        assert failures == [], \
            f"failed checks: {[f['name'] for f in failures]}"
