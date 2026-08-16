"""The serialised model artifact and its deployment contract."""

import json
import shutil

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from synthetic_trial.src.artifact import (
    ARTIFACT_STEM,
    FEATURE_ORDER,
    MODEL_VERSION,
    artifact_paths,
    build_metadata,
    load_artifact,
    save_artifact,
)
from synthetic_trial.src.model import MODEL_FEATURES

MODEL_PATH, METADATA_PATH = artifact_paths()

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="no artifact on disk; run `python train_model_artifact.py`",
)


@pytest.fixture(scope="module")
def artifact():
    return load_artifact()


@pytest.fixture(scope="module")
def metadata():
    return json.loads(METADATA_PATH.read_text())


# -- the contract itself ---------------------------------------------------


def test_feature_order_matches_the_training_code():
    # The one assertion that keeps deployment honest: a permuted vector would
    # score without raising and would be wrong everywhere.
    assert tuple(MODEL_FEATURES) == FEATURE_ORDER


def test_feature_order_is_immutable():
    assert isinstance(FEATURE_ORDER, tuple)
    with pytest.raises((TypeError, AttributeError)):
        FEATURE_ORDER[0] = "something_else"


# -- training produced both halves ----------------------------------------


def test_training_produces_a_model_file():
    assert MODEL_PATH.exists()
    assert MODEL_PATH.name == f"{ARTIFACT_STEM}.joblib"
    assert MODEL_PATH.stat().st_size > 0


def test_metadata_file_exists():
    assert METADATA_PATH.exists()
    assert METADATA_PATH.name == f"{ARTIFACT_STEM}.json"


def test_metadata_matches_the_model_configuration(artifact, metadata):
    model = artifact.model

    assert isinstance(model, IsolationForest)
    assert model.contamination == metadata["estimator"]["contamination"] == 0.10
    assert model.random_state == metadata["estimator"]["random_state"] == 42
    assert model.n_estimators == metadata["estimator"]["n_estimators"]
    assert model.bootstrap == metadata["estimator"]["bootstrap"]
    assert model.offset_ == pytest.approx(metadata["estimator"]["offset_"])
    assert model.n_features_in_ == len(FEATURE_ORDER)


def test_metadata_records_the_frozen_baseline(metadata):
    assert metadata["model_version"] == MODEL_VERSION
    assert sorted(metadata["training_cohort"]["scenarios"]) == ["IMPROVING", "STABLE"]
    assert metadata["training_cohort"]["random_seed"] == 42
    assert metadata["training_cohort"]["split"]["train_ratio"] == 0.5
    assert metadata["training_cohort"]["evaluation_half_used_in_fit"] is False
    assert metadata["features"]["preprocessing"] == "none"
    assert metadata["scoring"]["anomaly_score"] == "negated_decision_function"


def test_metadata_records_the_environment_it_was_built_in(metadata):
    assert metadata["environment"]["sklearn"]
    assert metadata["environment"]["numpy"]
    assert metadata["created_at"]


# -- reloading -------------------------------------------------------------


def test_model_reloads_successfully(artifact):
    assert hasattr(artifact.model, "decision_function")
    assert artifact.feature_order == FEATURE_ORDER
    assert artifact.model_version == MODEL_VERSION


def test_reloaded_model_scores_deterministically(artifact):
    matrix = np.array([[88.0, 94.0, 21.0, 18.0, -4.0, 6.0]])
    scores = {
        float(load_artifact().model.decision_function(matrix)[0]) for _ in range(3)
    }
    assert len(scores) == 1


def test_reloading_does_not_refit(artifact):
    before = artifact.model.offset_
    artifact.model.decision_function(np.array([[140.0, 80.0, 35.0, 60.0, -18.0, 20.0]]))
    assert artifact.model.offset_ == before


# -- refusing an inconsistent artifact ------------------------------------


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_artifact(tmp_path)


def test_reordered_features_are_refused(tmp_path):
    shutil.copy(MODEL_PATH, tmp_path / MODEL_PATH.name)
    payload = json.loads(METADATA_PATH.read_text())
    payload["features"]["names"] = list(reversed(FEATURE_ORDER))
    (tmp_path / METADATA_PATH.name).write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="feature order"):
        load_artifact(tmp_path)


def test_checksum_mismatch_is_refused(tmp_path):
    shutil.copy(MODEL_PATH, tmp_path / MODEL_PATH.name)
    payload = json.loads(METADATA_PATH.read_text())
    payload["model_sha256"] = "0" * 64
    (tmp_path / METADATA_PATH.name).write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="checksum"):
        load_artifact(tmp_path)


# -- round trip ------------------------------------------------------------


def test_save_then_load_reproduces_the_estimator_exactly(tmp_path):
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(200, len(FEATURE_ORDER)))
    model = IsolationForest(n_estimators=25, contamination=0.10, random_state=42)
    model.fit(matrix)

    save_artifact(model, build_metadata(model, matrix, {"selection": "test"}), tmp_path)
    reloaded = load_artifact(tmp_path)

    assert np.array_equal(
        model.decision_function(matrix), reloaded.model.decision_function(matrix)
    )
    assert np.array_equal(model.predict(matrix), reloaded.model.predict(matrix))


def test_saved_checksum_describes_the_file_written(tmp_path):
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(50, len(FEATURE_ORDER)))
    model = IsolationForest(n_estimators=10, random_state=42).fit(matrix)

    save_artifact(model, build_metadata(model, matrix, {}), tmp_path)

    # load_artifact verifies the checksum; a clean load is the assertion.
    assert load_artifact(tmp_path).model.n_estimators == 10
