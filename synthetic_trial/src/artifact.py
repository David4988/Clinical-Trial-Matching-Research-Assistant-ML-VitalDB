"""The model deployment contract.

This module is the boundary between the research pipeline and anything that
performs inference outside it. Everything a consumer needs in order to score a
window correctly is written down here and serialised alongside the estimator —
so a deployed model can never quietly disagree with the pipeline that produced
it.

Two files make up an artifact:

    synthetic_isolation_forest.joblib   the fitted sklearn estimator
    synthetic_isolation_forest.json     the contract it must be used under

The JSON is deliberately the *authoritative* half. A consumer reads it first,
checks the feature order against its own hard-coded expectation, and refuses to
run if they differ. That is why `FEATURE_ORDER` is a tuple: an ordered,
immutable sequence, never a dict view and never a DataFrame's accidental
column order.

The estimator is stored with no preprocessing wrapper because there is none —
`model.py` fits on raw physiological units. If scaling is ever introduced it
must be fitted inside a Pipeline and serialised as one object, or the contract
below becomes a lie.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest

from .model import MODEL_FEATURES

#: The deployment feature vector, in the exact order the estimator was fitted
#: on. Immutable and explicit — never derived from a dict, a set, or a
#: DataFrame's column order.
FEATURE_ORDER: tuple[str, ...] = (
    "heart_rate",
    "spo2",
    "respiratory_rate",
    "heart_rate_delta",
    "spo2_delta",
    "respiratory_rate_delta",
)

# The training code and the deployment contract must agree, and this is the
# only place that can be checked once rather than hoped for everywhere.
assert tuple(MODEL_FEATURES) == FEATURE_ORDER, (
    "model.MODEL_FEATURES has drifted from artifact.FEATURE_ORDER; the "
    "serialised model would be scored on a different vector than it was fitted "
    "on."
)

#: Bumped when the estimator is refitted in a way that changes its outputs.
MODEL_VERSION = "synthetic_if_v1"

#: Bumped only when the *meaning* of the six features changes. A consumer that
#: computes deltas differently must not load an artifact from another version.
FEATURE_VERSION = "synthetic_features_v1"

#: How to read `anomaly_score`. Recorded rather than assumed, because the sign
#: convention is the single easiest thing to get backwards downstream.
SCORE_ORIENTATION = "negated_decision_function"
SCORE_DECIMALS = 6

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACT_STEM = "synthetic_isolation_forest"


@dataclass(frozen=True)
class LoadedArtifact:
    """A deserialised estimator together with the contract it travels under."""

    model: IsolationForest
    metadata: dict[str, Any]

    @property
    def feature_order(self) -> tuple[str, ...]:
        return tuple(self.metadata["features"]["names"])

    @property
    def model_version(self) -> str:
        return self.metadata["model_version"]


def artifact_paths(directory: Path | None = None) -> tuple[Path, Path]:
    """The (.joblib, .json) pair for an artifact directory."""
    base = Path(directory) if directory is not None else DEFAULT_ARTIFACT_DIR
    return base / f"{ARTIFACT_STEM}.joblib", base / f"{ARTIFACT_STEM}.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata(
    model: IsolationForest,
    matrix: np.ndarray,
    cohort: dict[str, Any],
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Describe a fitted model completely enough to redeploy or reproduce it."""
    return {
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
        "estimator": {
            "class": "sklearn.ensemble.IsolationForest",
            "n_estimators": model.n_estimators,
            "contamination": model.contamination,
            "random_state": model.random_state,
            "max_samples": model.max_samples,
            "bootstrap": model.bootstrap,
            "n_jobs": model.n_jobs,
            # The threshold sklearn derived from `contamination` at fit time.
            # Recorded so a reviewer can see the decision boundary without
            # unpickling the estimator.
            "offset_": float(model.offset_),
            "max_samples_": int(model.max_samples_),
        },
        "features": {
            "names": list(FEATURE_ORDER),
            "count": len(FEATURE_ORDER),
            "dtype": "float64",
            "preprocessing": "none",
            "delta_definition": (
                "delta[t] = value[t] - value[t-1] for consecutive monitoring "
                "windows of the same patient, rounded to 4 decimals. A window "
                "with no valid predecessor has no delta and MUST NOT be scored."
            ),
            "window_minutes": 5,
        },
        "scoring": {
            "anomaly_score": SCORE_ORIENTATION,
            "anomaly_score_note": (
                "anomaly_score = round(-model.decision_function(X), 6); higher "
                "means more unusual."
            ),
            "decimals": SCORE_DECIMALS,
            "predicted_anomaly": "1 when model.predict(X) == -1, else 0",
        },
        "training_cohort": cohort,
        "training_matrix": {
            "rows": int(matrix.shape[0]),
            "columns": int(matrix.shape[1]),
            "feature_means": [round(float(v), 6) for v in matrix.mean(axis=0)],
            "feature_stds": [round(float(v), 6) for v in matrix.std(axis=0)],
            "feature_mins": [round(float(v), 6) for v in matrix.min(axis=0)],
            "feature_maxes": [round(float(v), 6) for v in matrix.max(axis=0)],
        },
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }


def save_artifact(
    model: IsolationForest,
    metadata: dict[str, Any],
    directory: Path | None = None,
) -> tuple[Path, Path]:
    """Write the estimator and its contract. Returns the two paths written."""
    model_path, metadata_path = artifact_paths(directory)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    # The checksum covers the file that was just written, so a metadata JSON
    # can never describe a different binary than the one beside it.
    metadata = dict(metadata)
    metadata["model_file"] = model_path.name
    metadata["model_sha256"] = _sha256(model_path)

    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    return model_path, metadata_path


def load_artifact(
    directory: Path | None = None, verify_checksum: bool = True
) -> LoadedArtifact:
    """Load an artifact and refuse anything that does not match the contract.

    `joblib.load` unpickles, so it executes whatever the file says. That is
    acceptable here and only here: the file is a first-party build product of
    `train_model_artifact.py`, it ships inside the repository, and its SHA-256
    is checked against the metadata before it is opened. An artifact from any
    other source must not be loaded with this function.
    """
    model_path, metadata_path = artifact_paths(directory)

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Model artifact incomplete. Expected both {model_path} and "
            f"{metadata_path}. Run `python train_model_artifact.py`."
        )

    with open(metadata_path) as handle:
        metadata = json.load(handle)

    names = tuple(metadata["features"]["names"])
    if names != FEATURE_ORDER:
        raise ValueError(
            f"Artifact feature order {names} does not match this code's "
            f"{FEATURE_ORDER}. Refusing to score on a different vector than "
            "the model was fitted on."
        )

    if verify_checksum:
        actual = _sha256(model_path)
        if actual != metadata.get("model_sha256"):
            raise ValueError(
                f"{model_path.name} checksum {actual} does not match the "
                f"metadata's {metadata.get('model_sha256')}. The artifact pair "
                "is inconsistent."
            )

    return LoadedArtifact(model=joblib.load(model_path), metadata=metadata)
