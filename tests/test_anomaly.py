"""Tests for the Isolation Forest baseline.

Offline and synthetic.  The emphasis is on the properties that would make the
result meaningless without being obvious: identifiers leaking into the matrix,
nulls being imputed, the score orientation being inverted, and the run not
being reproducible.
"""

import numpy as np
import pandas as pd
import pytest

from vitaldb_audit import anomaly, features


# ── Helpers ──────────────────────────────────────────────────────────────────


def _row(caseid, index, hr=70.0, spo2=99.0, rr=12.0, usable=True, rr_delta=0.5):
    """One feature-table row with every v1 column present."""
    record = {
        "caseid": caseid,
        "window_index": index,
        "window_start_s": index * 300.0,
        "window_end_s": (index + 1) * 300.0,
        "n_core_usable": 3 if usable else 1,
        "window_usable": usable,
        "consecutive_usable_windows": index + 1 if usable else 0,
    }
    values = {"hr": (hr, 0.4), "spo2": (spo2, -0.1), "rr": (rr, rr_delta)}
    for short, (mean, delta) in values.items():
        record[f"{short}_mean"] = mean
        record[f"{short}_std"] = 1.5
        record[f"{short}_min"] = mean - 4
        record[f"{short}_max"] = mean + 4
        record[f"{short}_delta"] = delta
        record[f"{short}_coverage_pct"] = 98.0 if usable else 40.0
        record[f"{short}_usable"] = usable
    return record


def _table(n=40, caseid=1):
    """A calm baseline population with mild variation."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        rows.append(_row(
            caseid, i,
            hr=70.0 + rng.normal(0, 2.0),
            spo2=99.0 + rng.normal(0, 0.3),
            rr=12.0 + rng.normal(0, 0.5),
        ))
    return pd.DataFrame(rows)


# ── Feature set ──────────────────────────────────────────────────────────────


def test_model_uses_exactly_the_18_requested_features():
    assert len(anomaly.MODEL_FEATURES) == 18
    for signal in ("hr", "spo2", "rr"):
        for stat in ("mean", "std", "min", "max", "delta", "coverage_pct"):
            assert f"{signal}_{stat}" in anomaly.MODEL_FEATURES


def test_identifiers_and_flags_are_never_model_features():
    """A leak here would let the model learn 'case 4' or 'late window'."""
    banned = [
        "caseid", "window_index", "window_start_s", "window_end_s",
        "hr_usable", "spo2_usable", "rr_usable",
        "n_core_usable", "window_usable", "consecutive_usable_windows",
    ]
    for column in banned:
        assert column not in anomaly.MODEL_FEATURES


def test_no_art_mbp_feature_reaches_the_model():
    assert not any("mbp" in c or c.startswith("art") for c in anomaly.MODEL_FEATURES)


def test_model_features_all_exist_in_the_v1_feature_schema():
    """The baseline cannot ask for a column the feature stage does not emit."""
    for column in anomaly.MODEL_FEATURES:
        assert column in features.FEATURE_COLUMNS


# ── Row selection ────────────────────────────────────────────────────────────


def test_only_usable_windows_are_analyzed():
    table = pd.concat([
        _table(20),
        pd.DataFrame([_row(1, 20, usable=False), _row(1, 21, usable=False)]),
    ], ignore_index=True)
    selected, report = anomaly.select_model_rows(table)
    assert len(selected) == 20
    assert report["rows_excluded_unusable_window"] == 2


def test_rows_with_null_features_are_excluded_not_imputed():
    table = _table(20)
    table.loc[3, "rr_delta"] = np.nan
    table.loc[7, "hr_std"] = np.nan
    selected, report = anomaly.select_model_rows(table)

    assert report["rows_excluded_null_feature"] == 2
    assert report["rows_analyzed"] == 18
    assert report["imputation"].startswith("none")
    assert not selected[anomaly.MODEL_FEATURES].isna().any().any()
    # The excluded windows are named, not silently dropped.
    excluded = {r["window_index"] for r in report["excluded_null_windows"]}
    assert excluded == {3, 7}


def test_null_rows_are_never_zero_filled():
    """The specific failure mode: a null delta becoming a real-looking 0.0."""
    table = _table(12)
    table.loc[5, "rr_delta"] = np.nan
    selected, _ = anomaly.select_model_rows(table)
    assert 5 not in selected["window_index"].tolist()
    assert len(selected) == 11


def test_selection_counts_reconcile():
    table = pd.concat([_table(15), pd.DataFrame([_row(1, 15, usable=False)])],
                      ignore_index=True)
    table.loc[2, "hr_delta"] = np.nan
    _, report = anomaly.select_model_rows(table)
    assert (report["rows_analyzed"]
            + report["rows_excluded_unusable_window"]
            + report["rows_excluded_null_feature"]) == report["rows_in_feature_table"]


def test_missing_column_raises():
    table = _table(10).drop(columns=["spo2_std"])
    with pytest.raises(ValueError, match="spo2_std"):
        anomaly.select_model_rows(table)


# ── Fit and score ────────────────────────────────────────────────────────────


def test_fit_and_score_returns_one_row_per_analyzed_window():
    table = _table(40)
    selected, _ = anomaly.select_model_rows(table)
    model = anomaly.fit_isolation_forest(selected)
    results = anomaly.score_windows(model, selected)

    assert len(results) == len(selected)
    for column in ("caseid", "window_index", "window_start_s", "window_end_s",
                   "anomaly_score", "anomaly_label"):
        assert column in results.columns


def test_model_is_fitted_on_the_18_features_only():
    table = _table(30)
    selected, _ = anomaly.select_model_rows(table)
    model = anomaly.fit_isolation_forest(selected)
    assert model.n_features_in_ == 18


def test_higher_score_means_more_unusual():
    """An obviously extreme window must rank above the calm population."""
    table = _table(60)
    table.loc[59, ["hr_mean", "hr_min", "hr_max"]] = [190.0, 180.0, 205.0]
    table.loc[59, "hr_delta"] = 115.0
    table.loc[59, "spo2_mean"] = 61.0

    selected, _ = anomaly.select_model_rows(table)
    model = anomaly.fit_isolation_forest(selected, contamination=0.05)
    results = anomaly.score_windows(model, selected)

    assert int(results.iloc[0]["window_index"]) == 59
    assert results.iloc[0]["anomaly_label"] == 1
    assert results["anomaly_score"].iloc[0] > results["anomaly_score"].iloc[-1]


def test_score_orientation_matches_the_sklearn_decision_function():
    """anomaly_score is the NEGATED decision function; inverting it would
    silently invert every conclusion drawn from the experiment."""
    table = _table(30)
    selected, _ = anomaly.select_model_rows(table)
    model = anomaly.fit_isolation_forest(selected)
    results = anomaly.score_windows(model, selected)
    assert np.allclose(
        results["anomaly_score"].to_numpy(),
        -results["iforest_decision_function"].to_numpy(),
        atol=1e-6,
    )


def test_label_agrees_with_the_decision_boundary():
    table = _table(50)
    selected, _ = anomaly.select_model_rows(table)
    model = anomaly.fit_isolation_forest(selected)
    results = anomaly.score_windows(model, selected)
    flagged = results[results["anomaly_label"] == 1]
    not_flagged = results[results["anomaly_label"] == 0]
    assert (flagged["iforest_decision_function"] < 0).all()
    assert (not_flagged["iforest_decision_function"] >= 0).all()


def test_results_are_ranked_most_unusual_first():
    table = _table(40)
    selected, _ = anomaly.select_model_rows(table)
    model = anomaly.fit_isolation_forest(selected)
    results = anomaly.score_windows(model, selected)
    assert results["anomaly_score"].is_monotonic_decreasing
    assert results["anomaly_rank"].tolist() == list(range(1, len(results) + 1))


def test_fixed_seed_makes_the_run_reproducible():
    table = _table(40)
    selected, _ = anomaly.select_model_rows(table)
    first = anomaly.score_windows(anomaly.fit_isolation_forest(selected), selected)
    second = anomaly.score_windows(anomaly.fit_isolation_forest(selected), selected)
    pd.testing.assert_frame_equal(first, second)


def test_contamination_is_configurable_and_controls_the_flag_rate():
    table = _table(100)
    selected, _ = anomaly.select_model_rows(table)
    counts = {}
    for contamination in (0.05, 0.20):
        model = anomaly.fit_isolation_forest(selected, contamination=contamination)
        results = anomaly.score_windows(model, selected)
        counts[contamination] = int(results["anomaly_label"].sum())
    assert counts[0.05] < counts[0.20]
    assert counts[0.20] == pytest.approx(20, abs=3)


def test_window_identity_survives_scoring():
    """Every analyzed window must still be traceable to its case and time."""
    table = _table(25)
    selected, _ = anomaly.select_model_rows(table)
    model = anomaly.fit_isolation_forest(selected)
    results = anomaly.score_windows(model, selected)

    merged = results.merge(table, on=["caseid", "window_index"], suffixes=("", "_t"))
    assert len(merged) == len(results)
    assert (merged["window_start_s"] == merged["window_start_s_t"]).all()
    assert set(results["window_index"]) == set(selected["window_index"])


def test_scoring_refuses_a_matrix_with_nulls():
    table = _table(20)
    selected, _ = anomaly.select_model_rows(table)
    selected.loc[2, "hr_mean"] = np.nan
    with pytest.raises(ValueError, match="nulls"):
        anomaly.fit_isolation_forest(selected)


# ── Reporting ────────────────────────────────────────────────────────────────


SUPERVISED_METRICS = (
    "accuracy", "precision", "recall", "f1", "auroc", "auprc", "roc_auc",
)


def _all_keys(node):
    """Every dict key anywhere in a nested structure."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.append(str(key).lower())
            found += _all_keys(value)
    elif isinstance(node, list):
        for item in node:
            found += _all_keys(item)
    return found


def test_summary_reports_no_supervised_metric_values():
    """No supervised metric may be REPORTED. The ground_truth field is allowed
    to name them, because its whole job is to say they are undefined here."""
    table = _table(40)
    selected, report = anomaly.select_model_rows(table)
    model = anomaly.fit_isolation_forest(selected)
    results = anomaly.score_windows(model, selected)
    summary = anomaly.build_summary(report, results, 0.1, anomaly.RANDOM_SEED, 200)

    for key in _all_keys(summary):
        assert not any(metric in key for metric in SUPERVISED_METRICS), key

    disclaimer = summary.pop("ground_truth").lower()
    assert "undefined" in disclaimer and "not reported" in disclaimer

    remaining = str(summary).lower()
    for metric in SUPERVISED_METRICS:
        assert metric not in remaining


def test_summary_carries_the_non_clinical_framing():
    table = _table(30)
    selected, report = anomaly.select_model_rows(table)
    model = anomaly.fit_isolation_forest(selected)
    results = anomaly.score_windows(model, selected)
    summary = anomaly.build_summary(report, results, 0.1, anomaly.RANDOM_SEED, 200)

    interpretation = summary["interpretation"].lower()
    assert "statistically unusual" in interpretation
    assert "not an adverse event" in interpretation
    assert summary["ground_truth"].startswith("none")


def test_extremes_report_returns_both_ends():
    table = _table(40)
    selected, _ = anomaly.select_model_rows(table)
    model = anomaly.fit_isolation_forest(selected)
    results = anomaly.score_windows(model, selected)
    extremes = anomaly.extreme_windows(results, n=5)

    assert len(extremes["most_unusual"]) == 5
    assert len(extremes["least_unusual"]) == 5
    assert (extremes["most_unusual"][0]["anomaly_score"]
            > extremes["least_unusual"][0]["anomaly_score"])
    for entry in extremes["most_unusual"]:
        assert entry["window_end_min"] > entry["window_start_min"]


def test_flagged_summary_counts_per_case():
    table = pd.concat([_table(20, caseid=1), _table(20, caseid=2)], ignore_index=True)
    selected, _ = anomaly.select_model_rows(table)
    model = anomaly.fit_isolation_forest(selected)
    results = anomaly.score_windows(model, selected)
    summary = anomaly.flagged_summary(results)

    assert summary["windows_analyzed"] == 40
    assert {r["caseid"] for r in summary["per_case"]} == {1, 2}
    assert sum(r["windows_flagged"] for r in summary["per_case"]) == summary["windows_flagged"]
