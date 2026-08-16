"""Freeze the synthetic Isolation Forest baseline into a deployable artifact.

    .venv/bin/python train_model_artifact.py

This is the *only* supported way to produce a model the application may load.
It calls the same `synthetic_trial.src.model` functions the evaluation pipeline
calls — it does not reimplement training — so the serialised estimator is by
construction the one the reports describe:

    observations.csv
      -> build_features                 (per-patient deltas, causal)
      -> split_patients(0.5, seed=42)   (deterministic patient-level split)
      -> select_training_cohort         (STABLE + IMPROVING patients only)
      -> prepare_model_matrix           (rows with all six features present)
      -> IsolationForest(contamination=0.10, random_state=42)
      -> artifacts/synthetic_isolation_forest.{joblib,json}

The evaluation half of the split is never touched here. Nothing in the
application retrains; it loads what this script wrote.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from synthetic_trial.src.artifact import (
    FEATURE_ORDER,
    build_metadata,
    load_artifact,
    save_artifact,
)
from synthetic_trial.src.features import build_features
from synthetic_trial.src.model import (
    MODEL_FEATURES,
    prepare_model_matrix,
    select_training_cohort,
    split_patients,
    train_anomaly_model,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_model_artifact")

OBSERVATIONS = Path("synthetic_trial/data/observations.csv")
ARTIFACT_DIR = Path("synthetic_trial/artifacts")

#: The frozen baseline. These are not tunable knobs — they are the numbers the
#: published evaluation reports were produced under.
TRAIN_RATIO = 0.5
CONTAMINATION = 0.10
SEED = 42
COHORT_SCENARIOS = ("STABLE", "IMPROVING")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observations", type=Path, default=OBSERVATIONS,
        help="Synthetic observation table to train from.",
    )
    parser.add_argument(
        "--out", type=Path, default=ARTIFACT_DIR,
        help="Directory to write the .joblib / .json pair into.",
    )
    args = parser.parse_args()

    if not args.observations.exists():
        logger.error(
            "Missing %s. Run `python generate_synthetic_trial.py` first.",
            args.observations,
        )
        return 1

    logger.info("Loading %s", args.observations)
    observations = pd.read_csv(args.observations)

    logger.info("Building features")
    features = build_features(observations)

    logger.info("Splitting patients (train_ratio=%s, seed=%s)", TRAIN_RATIO, SEED)
    train_patients, eval_patients = split_patients(
        features, train_ratio=TRAIN_RATIO, seed=SEED
    )
    train_df = features[features["patient_id"].isin(train_patients)].copy()

    # Recomputed here purely to describe the cohort in the metadata. The fit
    # below runs the same two calls internally — this does not change it.
    cohort_df = select_training_cohort(train_df)
    matrix = prepare_model_matrix(cohort_df)[MODEL_FEATURES].to_numpy(dtype=float)

    logger.info(
        "Training cohort: %d patients, %d windows, %d scoreable rows",
        cohort_df["patient_id"].nunique(),
        len(cohort_df),
        len(matrix),
    )

    logger.info("Fitting Isolation Forest (contamination=%s, seed=%s)", CONTAMINATION, SEED)
    model = train_anomaly_model(train_df, contamination=CONTAMINATION, seed=SEED)

    cohort = {
        "source": str(args.observations),
        "selection": (
            "Patients in the deterministic training half of the split whose "
            "scenario label is STABLE or IMPROVING. Rows missing any of the "
            "six features (the first window of every patient) are dropped."
        ),
        "scenarios": list(COHORT_SCENARIOS),
        "split": {
            "strategy": "patient-level, numpy default_rng shuffle",
            "train_ratio": TRAIN_RATIO,
            "seed": SEED,
        },
        "random_seed": SEED,
        "contamination": CONTAMINATION,
        "total_patients": int(features["patient_id"].nunique()),
        "train_patients": len(train_patients),
        "eval_patients": len(eval_patients),
        "cohort_patients": int(cohort_df["patient_id"].nunique()),
        "cohort_windows": int(len(cohort_df)),
        "fitted_rows": int(len(matrix)),
        "evaluation_half_used_in_fit": False,
    }

    metadata = build_metadata(model, matrix, cohort)
    model_path, metadata_path = save_artifact(model, metadata, args.out)

    logger.info("Wrote %s", model_path)
    logger.info("Wrote %s", metadata_path)

    _verify(model, matrix, args.out)
    return 0


def _verify(model, matrix, directory: Path) -> None:
    """Reload from disk and require the reloaded model to score identically.

    A serialisation step that silently changes predictions is the failure mode
    worth spending twenty lines to rule out.
    """
    reloaded = load_artifact(directory)

    original_scores = -model.decision_function(matrix)
    reloaded_scores = -reloaded.model.decision_function(matrix)
    max_drift = float(abs(original_scores - reloaded_scores).max())
    labels_match = bool(
        (model.predict(matrix) == reloaded.model.predict(matrix)).all()
    )

    print()
    print("=" * 62)
    print("MODEL ARTIFACT VERIFICATION")
    print("=" * 62)
    print(f"model_version      {reloaded.model_version}")
    print(f"feature order      {' | '.join(FEATURE_ORDER)}")
    print(f"contamination      {reloaded.metadata['estimator']['contamination']}")
    print(f"random_state       {reloaded.metadata['estimator']['random_state']}")
    print(f"offset_            {reloaded.metadata['estimator']['offset_']:.9f}")
    print(f"fitted rows        {reloaded.metadata['training_matrix']['rows']}")
    print(f"sha256             {reloaded.metadata['model_sha256'][:16]}…")
    print("-" * 62)
    print(f"round-trip score drift   {max_drift:.3e}")
    print(f"round-trip labels match  {labels_match}")
    print("=" * 62)

    if max_drift > 1e-12 or not labels_match:
        raise SystemExit(
            "Reloaded model does not reproduce the fitted model's output. "
            "The artifact is not safe to deploy."
        )


if __name__ == "__main__":
    raise SystemExit(main())
