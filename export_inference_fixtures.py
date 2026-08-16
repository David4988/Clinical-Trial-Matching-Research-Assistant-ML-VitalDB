"""Export canonical windows and their research-pipeline scores for parity testing.

    .venv/bin/python export_inference_fixtures.py

The application cannot import this repository, and the two run under different
interpreters, so parity between them is checked by exporting what the research
pipeline computes and asserting against it on the other side. This script
produces that fixture; `backend/tests/test_research_parity.py` in the
application consumes it.

Every exported window is drawn from the **evaluation half** of the patient
split, so no row here was seen during fitting. For each one the fixture records
the raw physiology of the window and its predecessor, the six features the
research pipeline built, and the score and label the artifact produced — enough
for the application to rebuild the vector from raw observations and compare.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from synthetic_trial.src.artifact import FEATURE_ORDER, load_artifact
from synthetic_trial.src.features import CORE_SIGNALS, build_features
from synthetic_trial.src.model import MODEL_FEATURES, score_observations, split_patients

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("export_inference_fixtures")

OBSERVATIONS = Path("synthetic_trial/data/observations.csv")
FIXTURE_DIR = Path(
    "../Clinical-Trial-Matching---Research-Assistant/backend/tests/fixtures"
)
DEFAULT_OUT = FIXTURE_DIR / "research_inference_parity.json"
TRAJECTORY_OUT = FIXTURE_DIR / "research_trajectory.json"

#: Windows per (scenario, flagged) pair. Small enough to read, wide enough that
#: a sign error or an off-by-one in the delta cannot hide.
PER_GROUP = 4

#: The patient replayed end to end through the application. Chosen from the
#: evaluation half for: a SUDDEN_DETERIORATION scenario with a late change
#: point, every window at GOOD data quality, a near-silent pre-event stretch and
#: a sustained post-event one. Nothing about the trajectory is edited — it is
#: exported exactly as the generator produced it.
TRAJECTORY_PATIENT = "P0014"


def main() -> int:
    if not OBSERVATIONS.exists():
        logger.error("Missing %s", OBSERVATIONS)
        return 1

    artifact = load_artifact()
    features = build_features(pd.read_csv(OBSERVATIONS))

    _, eval_patients = split_patients(features, train_ratio=0.5, seed=42)
    eval_df = features[features["patient_id"].isin(eval_patients)].copy()

    scored = score_observations(artifact.model, eval_df)
    scored = scored[scored["anomaly_score"].notna()]

    # Pair each scored window with the row before it, which is where its deltas
    # came from. `build_features` sorted by (patient_id, window_index), so the
    # predecessor is the previous row within the patient.
    previous = scored.sort_values(["patient_id", "window_index"]).groupby("patient_id")
    for signal in CORE_SIGNALS:
        scored[f"prev_{signal}"] = previous[signal].shift(1)
    scored = scored.dropna(subset=[f"prev_{s}" for s in CORE_SIGNALS])

    selected = (
        scored.sort_values(["scenario", "predicted_anomaly", "patient_id", "window_index"])
        .groupby(["scenario", "predicted_anomaly"], group_keys=False)
        .head(PER_GROUP)
    )

    cases = [
        {
            "patient_id": row["patient_id"],
            "scenario": row["scenario"],
            "window_index": int(row["window_index"]),
            "ground_truth_state": row["ground_truth_state"],
            "previous": {
                "heart_rate": float(row["prev_heart_rate"]),
                "spo2": float(row["prev_spo2"]),
                "respiratory_rate": float(row["prev_respiratory_rate"]),
            },
            "current": {
                "heart_rate": float(row["heart_rate"]),
                "spo2": float(row["spo2"]),
                "respiratory_rate": float(row["respiratory_rate"]),
            },
            "expected_features": {
                name: float(row[name]) for name in MODEL_FEATURES
            },
            "expected_anomaly_score": float(row["anomaly_score"]),
            "expected_predicted_anomaly": int(row["predicted_anomaly"]),
        }
        for _, row in selected.iterrows()
    ]

    payload = {
        "generated_by": "export_inference_fixtures.py",
        "source": str(OBSERVATIONS),
        "note": (
            "Windows from the evaluation half of the patient split — none was "
            "seen during fitting. Scores come from the serialised artifact."
        ),
        "model_version": artifact.metadata["model_version"],
        "feature_version": artifact.metadata["feature_version"],
        "model_sha256": artifact.metadata["model_sha256"],
        "feature_order": list(FEATURE_ORDER),
        "scoring": artifact.metadata["scoring"],
        "window_minutes": artifact.metadata["features"]["window_minutes"],
        "environment": artifact.metadata["environment"],
        "cases": cases,
    }

    out = DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    flagged = sum(c["expected_predicted_anomaly"] for c in cases)
    logger.info(
        "Wrote %d cases (%d flagged, %d normal) to %s",
        len(cases), flagged, len(cases) - flagged, out.resolve(),
    )
    logger.info("Scenarios: %s", sorted({c["scenario"] for c in cases}))

    _export_trajectory(artifact, features, eval_patients)
    return 0


def _export_trajectory(artifact, features: pd.DataFrame, eval_patients) -> None:
    """One complete held-out patient, window by window, with expected scores.

    The application replays this through its own ingestion and compares every
    window. Exporting the whole trajectory rather than sampled windows is the
    point: a delta computed against the wrong predecessor, or temporal state
    that leaks across a boundary, shows up as a run of disagreements rather
    than a single one.
    """
    if TRAJECTORY_PATIENT not in set(eval_patients):
        raise SystemExit(
            f"{TRAJECTORY_PATIENT} is not in the evaluation half; replaying a "
            "patient the model was fitted on would not prove anything."
        )

    patient = features[features["patient_id"] == TRAJECTORY_PATIENT].copy()
    patient = patient.sort_values("window_index")
    scored = score_observations(artifact.model, patient)

    windows = []
    for _, row in scored.iterrows():
        scoreable = pd.notna(row["anomaly_score"])
        windows.append(
            {
                "window_index": int(row["window_index"]),
                "timestamp_minutes": int(row["timestamp"]),
                "dose_number": int(row["dose_number"]),
                "ground_truth_state": row["ground_truth_state"],
                "observed": {
                    # Systolic BP is carried so the application's data-quality
                    # gate sees a complete record. The model never reads it.
                    "heart_rate": float(row["heart_rate"]),
                    "spo2": float(row["spo2"]),
                    "respiratory_rate": float(row["respiratory_rate"]),
                    "systolic_bp": float(row["systolic_bp"]),
                },
                "coverage_percent": float(row["coverage_percent"]),
                "data_quality": row["data_quality"],
                "expected_features": (
                    {name: float(row[name]) for name in MODEL_FEATURES}
                    if scoreable
                    else None
                ),
                "expected_anomaly_score": (
                    float(row["anomaly_score"]) if scoreable else None
                ),
                "expected_predicted_anomaly": (
                    int(row["predicted_anomaly"]) if scoreable else None
                ),
            }
        )

    acute = [w for w in windows if w["ground_truth_state"] == "acute_change"]
    payload = {
        "generated_by": "export_inference_fixtures.py",
        "source": str(OBSERVATIONS),
        "note": (
            "One complete patient trajectory from the evaluation half, exported "
            "unmodified. The application replays it through its own ingestion "
            "and compares every scoreable window."
        ),
        "patient_id": TRAJECTORY_PATIENT,
        "scenario": scored["scenario"].iloc[0],
        "change_point_window": acute[0]["window_index"] if acute else None,
        "window_minutes": artifact.metadata["features"]["window_minutes"],
        "model_version": artifact.metadata["model_version"],
        "feature_version": artifact.metadata["feature_version"],
        "model_sha256": artifact.metadata["model_sha256"],
        "feature_order": list(FEATURE_ORDER),
        "environment": artifact.metadata["environment"],
        "windows": windows,
    }

    TRAJECTORY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAJECTORY_OUT, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    scoreable = [w for w in windows if w["expected_anomaly_score"] is not None]
    flagged = sum(w["expected_predicted_anomaly"] for w in scoreable)
    logger.info(
        "Wrote trajectory %s (%s): %d windows, %d scoreable, %d flagged, "
        "change point at window %s -> %s",
        TRAJECTORY_PATIENT, payload["scenario"], len(windows), len(scoreable),
        flagged, payload["change_point_window"], TRAJECTORY_OUT.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
