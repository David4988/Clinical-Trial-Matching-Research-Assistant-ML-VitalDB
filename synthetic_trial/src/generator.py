"""Deterministic synthetic clinical-trial data generator.

Produces four linked tables — patients, trial_assignments, observations,
events — with known ground-truth scenario labels.  Every patient follows one
of seven temporal scenarios, and every observation window retains the scenario
label and a per-window ``ground_truth_state``.

This is a hackathon prototype.  The trajectories are rule-based synthetic
constructions, not pharmacokinetic simulations.  No clinical validity is
claimed.

Determinism
-----------
Same seed + same configuration = identical output.  The generator uses
``numpy.random.Generator`` seeded once; per-patient sub-seeds are drawn
sequentially so every patient's trajectory is reproducible.
"""

import numpy as np
import pandas as pd

# ── Scenarios ────────────────────────────────────────────────────────────────

SCENARIOS = [
    "STABLE",
    "IMPROVING",
    "GRADUAL_DETERIORATION",
    "SUDDEN_DETERIORATION",
    "RECOVERY",
    "ADVERSE_EVENT",
    "DATA_QUALITY_FAILURE",
]

DEFAULT_SCENARIO_WEIGHTS = {
    "STABLE":                  0.20,
    "IMPROVING":               0.15,
    "GRADUAL_DETERIORATION":   0.15,
    "SUDDEN_DETERIORATION":    0.10,
    "RECOVERY":                0.15,
    "ADVERSE_EVENT":           0.15,
    "DATA_QUALITY_FAILURE":    0.10,
}

# ── Signal configuration ────────────────────────────────────────────────────

SIGNAL_NAMES = [
    "heart_rate", "spo2", "respiratory_rate",
    "systolic_bp", "diastolic_bp", "temperature",
]

BASELINE_PARAMS = {                      # (mean, std) for patient generation
    "heart_rate":        (72.0,  8.0),
    "spo2":              (97.0,  1.0),
    "respiratory_rate":  (16.0,  2.0),
    "systolic_bp":       (120.0, 12.0),
    "diastolic_bp":      (80.0,  8.0),
    "temperature":       (36.8,  0.3),
}

NOISE_SCALE = {                          # per-window noise σ
    "heart_rate":        3.0,
    "spo2":              0.5,
    "respiratory_rate":  1.5,
    "systolic_bp":       5.0,
    "diastolic_bp":      3.0,
    "temperature":       0.15,
}

SIGNAL_BOUNDS = {                        # broad synthetic bounds
    "heart_rate":        (40,   180),
    "spo2":              (70,   100),
    "respiratory_rate":  (6,    40),
    "systolic_bp":       (60,   200),
    "diastolic_bp":      (40,   120),
    "temperature":       (35.0, 41.0),
}

IDEAL_VALUES = {                         # "healthy" targets for IMPROVING
    "heart_rate":        70.0,
    "spo2":              98.0,
    "respiratory_rate":  14.0,
    "systolic_bp":       118.0,
    "diastolic_bp":      78.0,
    "temperature":       36.7,
}

# Gradual deterioration: change per window (sign encodes direction)
GRADUAL_RATE = {
    "heart_rate":        +0.15,
    "spo2":              -0.03,
    "respiratory_rate":  +0.08,
    "systolic_bp":       -0.12,
    "diastolic_bp":      -0.08,
    "temperature":       +0.005,
}

# Sudden shifts (applied as a step change)
SUDDEN_SHIFT = {
    "heart_rate":        +20,
    "spo2":              -5,
    "respiratory_rate":  +8,
    "systolic_bp":       -25,
    "diastolic_bp":      -12,
    "temperature":       +0.8,
}

# Adverse-event shift (more severe)
ADVERSE_SHIFT = {
    "heart_rate":        +30,
    "spo2":              -8,
    "respiratory_rate":  +10,
    "systolic_bp":       -30,
    "diastolic_bp":      -15,
    "temperature":       +1.2,
}

# Condition offset for IMPROVING patients (starts slightly abnormal)
CONDITION_OFFSET = {
    "heart_rate":        +10,
    "spo2":              -2,
    "respiratory_rate":  +4,
    "systolic_bp":       -10,
    "diastolic_bp":      -5,
    "temperature":       +0.4,
}

# ── Trial definitions ────────────────────────────────────────────────────────

TRIALS = [
    {"trial_id": "TRIAL-001", "drug_id": "DRUG-A",
     "dose_amount": 100, "indication": "cardiovascular"},
    {"trial_id": "TRIAL-002", "drug_id": "DRUG-B",
     "dose_amount": 50,  "indication": "respiratory"},
    {"trial_id": "TRIAL-003", "drug_id": "DRUG-C",
     "dose_amount": 200, "indication": "metabolic"},
]

# ── Defaults ─────────────────────────────────────────────────────────────────

DOSES_PER_PATIENT = 3
WINDOWS_PER_DOSE  = 48      # 48 × 5 min = 4 hours monitoring per dose cycle
WINDOW_MINUTES    = 5


# ── Helpers ──────────────────────────────────────────────────────────────────


def _clamp(values, signal):
    lo, hi = SIGNAL_BOUNDS[signal]
    return np.clip(values, lo, hi)


def data_quality_label(coverage_pct):
    """Classify a single coverage percentage."""
    if coverage_pct >= 95:
        return "GOOD"
    if coverage_pct >= 70:
        return "PARTIAL"
    if coverage_pct >= 30:
        return "POOR"
    return "MISSING"


# ── Generator ────────────────────────────────────────────────────────────────


class SyntheticTrialGenerator:
    """Deterministic synthetic clinical-trial data generator.

    Parameters
    ----------
    n_patients : int
        Number of patients to generate.
    seed : int
        Random seed for deterministic output.
    scenario_weights : dict or None
        Per-scenario relative weights.  Need not sum to 1.
    doses_per_patient : int
        Number of dose cycles per patient.
    windows_per_dose : int
        Number of 5-minute monitoring windows per dose cycle.
    """

    def __init__(self, n_patients=500, seed=20260817,
                 scenario_weights=None,
                 doses_per_patient=DOSES_PER_PATIENT,
                 windows_per_dose=WINDOWS_PER_DOSE):
        self.n_patients = n_patients
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.scenario_weights = dict(scenario_weights or DEFAULT_SCENARIO_WEIGHTS)
        self.doses_per_patient = doses_per_patient
        self.windows_per_dose = windows_per_dose
        self.total_windows = doses_per_patient * windows_per_dose

    # ── Public ───────────────────────────────────────────────────────────────

    def generate(self):
        """Generate all four tables.

        Returns dict with keys: patients, trial_assignments, observations,
        events — each a ``pandas.DataFrame``.
        """
        scenarios = self._assign_scenarios()
        patients = self._generate_patients(scenarios)
        assignments = self._generate_trial_assignments(patients)
        observations, events = self._generate_monitoring(patients, assignments)

        result = {
            "patients": pd.DataFrame(patients),
            "trial_assignments": pd.DataFrame(assignments),
            "observations": pd.DataFrame(observations),
            "events": pd.DataFrame(
                events,
                columns=["patient_id", "trial_id", "timestamp",
                         "event_type", "severity", "scenario", "description"],
            ) if events else pd.DataFrame(
                columns=["patient_id", "trial_id", "timestamp",
                         "event_type", "severity", "scenario", "description"]),
        }
        return result

    # ── Scenarios ────────────────────────────────────────────────────────────

    def _assign_scenarios(self):
        weights = np.array([self.scenario_weights[s] for s in SCENARIOS])
        probs = weights / weights.sum()
        return list(self.rng.choice(SCENARIOS, size=self.n_patients, p=probs))

    # ── Patients ─────────────────────────────────────────────────────────────

    def _generate_patients(self, scenarios):
        patients = []
        for i in range(self.n_patients):
            pid = f"P{i + 1:04d}"
            age = int(self.rng.integers(25, 81))
            sex = str(self.rng.choice(["M", "F"]))

            baselines = {}
            for signal, (mean, std) in BASELINE_PARAMS.items():
                lo, hi = SIGNAL_BOUNDS[signal]
                val = float(np.clip(self.rng.normal(mean, std), lo + 5, hi - 5))
                baselines[signal] = val

            hypertension   = int(self.rng.random() < 0.30)
            diabetes       = int(self.rng.random() < 0.15)
            cardiovascular = int(self.rng.random() < 0.10)
            respiratory    = int(self.rng.random() < 0.10)
            smoker         = int(self.rng.random() < 0.20)
            bmi = float(np.clip(self.rng.normal(26, 5), 16, 45))

            if cardiovascular:
                category = "cardiovascular"
            elif respiratory:
                category = "respiratory"
            elif diabetes:
                category = "metabolic"
            else:
                category = "general"

            patients.append({
                "patient_id": pid,
                "age": age,
                "sex": sex,
                "baseline_hr": round(baselines["heart_rate"], 1),
                "baseline_spo2": round(baselines["spo2"], 1),
                "baseline_rr": round(baselines["respiratory_rate"], 1),
                "baseline_sbp": round(baselines["systolic_bp"], 1),
                "baseline_dbp": round(baselines["diastolic_bp"], 1),
                "baseline_temperature": round(baselines["temperature"], 2),
                "hypertension": hypertension,
                "diabetes": diabetes,
                "cardiovascular_disease": cardiovascular,
                "respiratory_condition": respiratory,
                "smoker": smoker,
                "bmi": round(bmi, 1),
                "condition_category": category,
                "scenario": scenarios[i],
            })
        return patients

    # ── Trial assignments ────────────────────────────────────────────────────

    def _generate_trial_assignments(self, patients):
        rows = []
        n_trials = len(TRIALS)
        for patient in patients:
            trial = TRIALS[int(self.rng.integers(0, n_trials))]
            arm = "TREATMENT" if self.rng.random() < 0.67 else "CONTROL"
            drug = trial["drug_id"] if arm == "TREATMENT" else "PLACEBO"
            dose_amt = trial["dose_amount"] if arm == "TREATMENT" else 0
            enrollment = int(self.rng.integers(1, 91))

            for dose_num in range(1, self.doses_per_patient + 1):
                rows.append({
                    "patient_id":    patient["patient_id"],
                    "trial_id":      trial["trial_id"],
                    "treatment_arm": arm,
                    "drug_id":       drug,
                    "dose_number":   dose_num,
                    "dose_amount":   dose_amt,
                    "enrollment_time": enrollment,
                    "scenario":      patient["scenario"],
                })
        return rows

    # ── Monitoring (observations + events) ───────────────────────────────────

    def _generate_monitoring(self, patients, assignments):
        all_obs = []
        all_events = []

        # Build patient_id → trial_id lookup
        trial_lookup = {}
        for a in assignments:
            if a["patient_id"] not in trial_lookup:
                trial_lookup[a["patient_id"]] = a["trial_id"]

        for patient in patients:
            pid = patient["patient_id"]
            scenario = patient["scenario"]
            trial_id = trial_lookup[pid]

            # Per-patient RNG (deterministic sub-seed)
            pseed = int(self.rng.integers(0, 2**31))
            prng = np.random.default_rng(pseed)

            baselines = {
                "heart_rate":       patient["baseline_hr"],
                "spo2":             patient["baseline_spo2"],
                "respiratory_rate": patient["baseline_rr"],
                "systolic_bp":      patient["baseline_sbp"],
                "diastolic_bp":     patient["baseline_dbp"],
                "temperature":      patient["baseline_temperature"],
            }

            values, coverage, gt_states, event_info = \
                self._generate_trajectory(scenario, baselines, prng)

            for w in range(self.total_windows):
                cov = float(coverage[w])
                all_obs.append({
                    "patient_id":        pid,
                    "trial_id":          trial_id,
                    "timestamp":         w * WINDOW_MINUTES,
                    "window_index":      w,
                    "dose_number":       (w // self.windows_per_dose) + 1,
                    "heart_rate":        round(float(values["heart_rate"][w]), 1),
                    "spo2":              round(float(values["spo2"][w]), 1),
                    "respiratory_rate":  round(float(values["respiratory_rate"][w]), 1),
                    "systolic_bp":       round(float(values["systolic_bp"][w]), 1),
                    "diastolic_bp":      round(float(values["diastolic_bp"][w]), 1),
                    "temperature":       round(float(values["temperature"][w]), 2),
                    "scenario":          scenario,
                    "coverage_percent":  round(cov, 1),
                    "data_quality":      data_quality_label(cov),
                    "ground_truth_state": gt_states[w],
                })

            events = self._generate_events(pid, trial_id, scenario,
                                           event_info, prng)
            all_events.extend(events)

        return all_obs, all_events

    # ── Trajectory dispatch ──────────────────────────────────────────────────

    def _generate_trajectory(self, scenario, baselines, rng):
        """Dispatch to the scenario-specific trajectory generator.

        Returns
        -------
        values : dict[str, np.ndarray]   — one array per signal
        coverage : np.ndarray            — coverage percent per window
        ground_truth : list[str]         — per-window state label
        event_info : dict or None        — timing for event generation
        """
        dispatch = {
            "STABLE":                  self._traj_stable,
            "IMPROVING":               self._traj_improving,
            "GRADUAL_DETERIORATION":   self._traj_gradual_deterioration,
            "SUDDEN_DETERIORATION":    self._traj_sudden_deterioration,
            "RECOVERY":                self._traj_recovery,
            "ADVERSE_EVENT":           self._traj_adverse_event,
            "DATA_QUALITY_FAILURE":    self._traj_data_quality_failure,
        }
        return dispatch[scenario](baselines, self.total_windows, rng)

    # ── Scenario trajectories ────────────────────────────────────────────────

    def _traj_stable(self, baselines, n, rng):
        values = {}
        for sig in SIGNAL_NAMES:
            noise = rng.normal(0, NOISE_SCALE[sig], n)
            values[sig] = _clamp(baselines[sig] + noise, sig)
        coverage = np.clip(rng.normal(98, 1.5, n), 90, 100)
        return values, coverage, ["normal"] * n, None

    def _traj_improving(self, baselines, n, rng):
        values = {}
        for sig in SIGNAL_NAMES:
            start = baselines[sig] + CONDITION_OFFSET[sig]
            ideal = IDEAL_VALUES[sig]
            improvement = np.linspace(0, (ideal - start) * 0.6, n)
            # Noise decreases over time (stabilisation)
            noise_scale = np.linspace(NOISE_SCALE[sig], NOISE_SCALE[sig] * 0.5, n)
            noise = rng.normal(0, 1, n) * noise_scale
            values[sig] = _clamp(start + improvement + noise, sig)
        coverage = np.clip(rng.normal(98, 1.5, n), 90, 100)
        split = 2 * n // 3
        gt = ["improving"] * split + ["improved"] * (n - split)
        return values, coverage, gt, None

    def _traj_gradual_deterioration(self, baselines, n, rng):
        values = {}
        for sig in SIGNAL_NAMES:
            trend = np.arange(n, dtype=float) * GRADUAL_RATE[sig]
            noise = rng.normal(0, NOISE_SCALE[sig], n)
            values[sig] = _clamp(baselines[sig] + trend + noise, sig)
        coverage = np.clip(rng.normal(98, 1.5, n), 90, 100)
        return values, coverage, ["deteriorating"] * n, None

    def _traj_sudden_deterioration(self, baselines, n, rng):
        cp = int(rng.integers(int(n * 0.6), int(n * 0.8)))
        values = {}
        for sig in SIGNAL_NAMES:
            trajectory = np.full(n, baselines[sig], dtype=float)
            trajectory[cp:] += SUDDEN_SHIFT[sig]
            noise = rng.normal(0, NOISE_SCALE[sig], n)
            values[sig] = _clamp(trajectory + noise, sig)
        coverage = np.clip(rng.normal(98, 1.5, n), 90, 100)
        gt = ["normal"] * cp + ["acute_change"] * (n - cp)
        return values, coverage, gt, {"change_point": cp}

    def _traj_recovery(self, baselines, n, rng):
        nadir = n // 3
        recovered_start = 3 * n // 4
        values = {}
        for sig in SIGNAL_NAMES:
            shift = SUDDEN_SHIFT[sig]
            trajectory = np.zeros(n, dtype=float)
            # Deterioration ramp
            trajectory[:nadir] = np.linspace(0, shift, nadir)
            # Recovery ramp
            trajectory[nadir:] = np.linspace(shift, shift * 0.1, n - nadir)
            noise = rng.normal(0, NOISE_SCALE[sig], n)
            values[sig] = _clamp(baselines[sig] + trajectory + noise, sig)
        coverage = np.clip(rng.normal(98, 1.5, n), 90, 100)
        gt = (["deteriorating"] * nadir
              + ["recovering"] * (recovered_start - nadir)
              + ["recovered"] * (n - recovered_start))
        return values, coverage, gt, {"nadir_window": nadir}

    def _traj_adverse_event(self, baselines, n, rng):
        event_start = int(rng.integers(n // 3, 2 * n // 3))
        event_duration = int(rng.integers(8, 16))
        event_end = min(event_start + event_duration, n)
        values = {}
        for sig in SIGNAL_NAMES:
            trajectory = np.full(n, baselines[sig], dtype=float)
            trajectory[event_start:event_end] += ADVERSE_SHIFT[sig]
            # Partial lingering effect after event
            if event_end < n:
                residual = np.linspace(
                    ADVERSE_SHIFT[sig] * 0.4, ADVERSE_SHIFT[sig] * 0.05,
                    n - event_end)
                trajectory[event_end:] += residual
            noise = rng.normal(0, NOISE_SCALE[sig], n)
            values[sig] = _clamp(trajectory + noise, sig)
        coverage = np.clip(rng.normal(98, 1.5, n), 90, 100)
        gt = (["normal"] * event_start
              + ["adverse_event"] * (event_end - event_start)
              + ["post_event"] * (n - event_end))
        info = {"event_start": event_start, "event_end": event_end}
        return values, coverage, gt, info

    def _traj_data_quality_failure(self, baselines, n, rng):
        # Normal physiology
        values = {}
        for sig in SIGNAL_NAMES:
            noise = rng.normal(0, NOISE_SCALE[sig], n)
            values[sig] = _clamp(baselines[sig] + noise, sig)
        # Normal coverage with a large gap
        coverage = np.clip(rng.normal(98, 1.5, n), 90, 100)
        gap_start = int(rng.integers(n // 4, n // 2))
        gap_length = int(rng.integers(12, 24))
        gap_end = min(gap_start + gap_length, n)
        coverage[gap_start:gap_end] = rng.uniform(0, 30, gap_end - gap_start)
        gt = ["normal"] * n
        for i in range(gap_start, gap_end):
            gt[i] = "data_gap"
        info = {"gap_start": gap_start, "gap_end": gap_end}
        return values, coverage, gt, info

    # ── Event generation ─────────────────────────────────────────────────────

    def _generate_events(self, patient_id, trial_id, scenario,
                         event_info, rng):
        events = []

        if scenario == "ADVERSE_EVENT" and event_info:
            ts = event_info["event_start"] * WINDOW_MINUTES
            events.append({
                "patient_id":  patient_id,
                "trial_id":    trial_id,
                "timestamp":   ts,
                "event_type":  "ADVERSE_EVENT",
                "severity":    str(rng.choice(["MODERATE", "SEVERE"])),
                "scenario":    scenario,
                "description": ("Acute physiological change detected "
                                "during monitoring window"),
            })

        if scenario == "RECOVERY" and event_info:
            ts = event_info["nadir_window"] * WINDOW_MINUTES
            events.append({
                "patient_id":  patient_id,
                "trial_id":    trial_id,
                "timestamp":   ts,
                "event_type":  "RECOVERY",
                "severity":    "MILD",
                "scenario":    scenario,
                "description": ("Physiological parameters beginning to "
                                "normalise after deterioration"),
            })

        if scenario == "IMPROVING":
            events.append({
                "patient_id":  patient_id,
                "trial_id":    trial_id,
                "timestamp":   WINDOW_MINUTES * 5,
                "event_type":  "DOSE_RESPONSE",
                "severity":    "MILD",
                "scenario":    scenario,
                "description": "Progressive improvement in monitored parameters",
            })

        if scenario == "SUDDEN_DETERIORATION" and event_info:
            ts = event_info["change_point"] * WINDOW_MINUTES
            events.append({
                "patient_id":  patient_id,
                "trial_id":    trial_id,
                "timestamp":   ts,
                "event_type":  "ADVERSE_EVENT",
                "severity":    "SEVERE",
                "scenario":    scenario,
                "description": "Sudden physiological shift during monitoring",
            })

        if scenario == "DATA_QUALITY_FAILURE" and event_info:
            ts = event_info["gap_start"] * WINDOW_MINUTES
            events.append({
                "patient_id":  patient_id,
                "trial_id":    trial_id,
                "timestamp":   ts,
                "event_type":  "DATA_QUALITY_EVENT",
                "severity":    "MODERATE",
                "scenario":    scenario,
                "description": ("Significant reduction in monitoring "
                                "data coverage"),
            })

        return events
