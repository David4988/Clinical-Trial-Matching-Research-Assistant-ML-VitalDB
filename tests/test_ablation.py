"""Tests for the coverage-feature ablation.

The experiment is only meaningful if the two variants differ in EXACTLY one
respect.  Most of these tests exist to prove that, because a difference that
crept in through the row selection or the seed would produce a confident and
completely wrong conclusion.
"""

import json

import numpy as np
import pandas as pd
import pytest

from vitaldb_audit import ablation, anomaly


def _row(caseid, index, hr=70.0, spo2=99.0, rr=12.0, coverage=100.0, usable=True):
    record = {
        "caseid": caseid,
        "window_index": index,
        "window_start_s": index * 300.0,
        "window_end_s": (index + 1) * 300.0,
        "n_core_usable": 3 if usable else 1,
        "window_usable": usable,
        "consecutive_usable_windows": index + 1 if usable else 0,
    }
    for short, mean in (("hr", hr), ("spo2", spo2), ("rr", rr)):
        record[f"{short}_mean"] = mean
        record[f"{short}_std"] = 1.5
        record[f"{short}_min"] = mean - 4
        record[f"{short}_max"] = mean + 4
        record[f"{short}_delta"] = 0.3
        record[f"{short}_coverage_pct"] = coverage
        record[f"{short}_usable"] = usable
    return record


def _table(n=60, caseid=1):
    rng = np.random.default_rng(11)
    return pd.DataFrame([
        _row(caseid, i,
             hr=70.0 + rng.normal(0, 2.0),
             spo2=99.0 + rng.normal(0, 0.3),
             rr=12.0 + rng.normal(0, 0.5))
        for i in range(n)
    ])


# ── The single controlled difference ─────────────────────────────────────────


def test_model_b_removes_exactly_the_three_coverage_features():
    assert set(ablation.MODEL_A_FEATURES) - set(ablation.MODEL_B_FEATURES) == {
        "hr_coverage_pct", "spo2_coverage_pct", "rr_coverage_pct",
    }
    assert len(ablation.MODEL_A_FEATURES) == 18
    assert len(ablation.MODEL_B_FEATURES) == 15


def test_model_b_keeps_every_other_feature_in_the_same_order():
    kept = [c for c in ablation.MODEL_A_FEATURES
            if c not in ablation.COVERAGE_FEATURES]
    assert ablation.MODEL_B_FEATURES == kept


def test_model_a_is_identical_to_the_current_baseline_feature_set():
    assert ablation.MODEL_A_FEATURES == anomaly.MODEL_FEATURES


def test_both_variants_score_the_same_windows():
    bundle = ablation.run_ablation(_table(50))
    a, b = bundle["results_a"], bundle["results_b"]
    assert len(a) == len(b) == len(bundle["selected"])
    assert (set(map(tuple, a[["caseid", "window_index"]].to_numpy()))
            == set(map(tuple, b[["caseid", "window_index"]].to_numpy())))


def test_row_selection_is_computed_once_and_shared():
    """Both variants must train on the same population, not two selections."""
    table = _table(40)
    table.loc[5, "rr_delta"] = np.nan          # excluded from BOTH
    bundle = ablation.run_ablation(table)
    assert bundle["selection"]["rows_analyzed"] == 39
    assert len(bundle["results_a"]) == 39
    assert len(bundle["results_b"]) == 39
    for results in (bundle["results_a"], bundle["results_b"]):
        assert 5 not in results["window_index"].tolist()


def test_variants_are_fitted_on_18_and_15_features():
    bundle = ablation.run_ablation(_table(40))
    assert bundle["model_a"].n_features_in_ == 18
    assert bundle["model_b"].n_features_in_ == 15


def test_seed_and_contamination_are_shared():
    bundle = ablation.run_ablation(_table(40), contamination=0.15, random_state=99)
    assert bundle["model_a"].random_state == bundle["model_b"].random_state == 99
    assert bundle["model_a"].contamination == bundle["model_b"].contamination == 0.15


def test_ablation_is_reproducible():
    first = ablation.run_ablation(_table(40))
    second = ablation.run_ablation(_table(40))
    pd.testing.assert_frame_equal(first["results_a"], second["results_a"])
    pd.testing.assert_frame_equal(first["results_b"], second["results_b"])


def test_model_a_reproduces_the_standalone_baseline_exactly():
    """The ablation harness must not perturb the existing baseline result."""
    table = _table(45)
    selected, _ = anomaly.select_model_rows(table)
    baseline = anomaly.score_windows(anomaly.fit_isolation_forest(selected), selected)
    bundle = ablation.run_ablation(table)
    pd.testing.assert_frame_equal(baseline, bundle["results_a"])


# ── Comparison mechanics ─────────────────────────────────────────────────────


def test_compare_joins_every_window_once():
    bundle = ablation.run_ablation(_table(50))
    comparison = ablation.compare(bundle["results_a"], bundle["results_b"])
    assert len(comparison) == len(bundle["results_a"])
    assert not comparison.duplicated(["caseid", "window_index"]).any()


def test_transition_labels_are_exhaustive_and_correct():
    bundle = ablation.run_ablation(_table(60))
    comparison = ablation.compare(bundle["results_a"], bundle["results_b"])
    counts = ablation.transition_counts(comparison)
    assert sum(counts.values()) == len(comparison)

    for _, row in comparison.iterrows():
        was, now = bool(row["anomaly_label_a"]), bool(row["anomaly_label_b"])
        expected = {
            (True, True): "flagged in both",
            (True, False): "flagged -> not flagged",
            (False, True): "not flagged -> flagged",
            (False, False): "flagged in neither",
        }[(was, now)]
        assert row["transition"] == expected


def test_flag_counts_match_the_transition_table():
    bundle = ablation.run_ablation(_table(60))
    comparison = ablation.compare(bundle["results_a"], bundle["results_b"])
    counts = ablation.transition_counts(comparison)
    assert (counts["flagged in both"] + counts["flagged -> not flagged"]
            == int(bundle["results_a"]["anomaly_label"].sum()))
    assert (counts["flagged in both"] + counts["not flagged -> flagged"]
            == int(bundle["results_b"]["anomaly_label"].sum()))


def test_removing_coverage_changes_a_coverage_driven_flag():
    """A window that is normal except for uniquely low coverage should stop
    standing out once the coverage columns are gone."""
    table = _table(60)
    table.loc[59, ["hr_coverage_pct", "spo2_coverage_pct", "rr_coverage_pct"]] = 72.0

    bundle = ablation.run_ablation(table, contamination=0.05)
    comparison = ablation.compare(bundle["results_a"], bundle["results_b"])
    row = comparison[comparison["window_index"] == 59].iloc[0]

    assert row["anomaly_score_a"] > row["anomaly_score_b"]
    assert row["anomaly_rank_b"] > row["anomaly_rank_a"]


def test_a_physiology_driven_outlier_survives_the_ablation():
    """The control: an outlier with perfect coverage must not be an artifact
    of the coverage features."""
    table = _table(60)
    table.loc[59, ["hr_mean", "hr_min", "hr_max"]] = [185.0, 172.0, 199.0]
    table.loc[59, "hr_delta"] = 108.0

    bundle = ablation.run_ablation(table, contamination=0.05)
    comparison = ablation.compare(bundle["results_a"], bundle["results_b"])
    row = comparison[comparison["window_index"] == 59].iloc[0]

    assert bool(row["anomaly_label_a"]) and bool(row["anomaly_label_b"])
    assert row["transition"] == "flagged in both"


def test_rank_agreement_is_descriptive_only():
    bundle = ablation.run_ablation(_table(50))
    comparison = ablation.compare(bundle["results_a"], bundle["results_b"])
    agreement = ablation.rank_agreement(comparison)
    assert -1.0 <= agreement["spearman_rank_correlation"] <= 1.0
    assert 0 <= agreement["top10_overlap"] <= 10


def test_per_case_counts_sum_to_the_totals():
    table = pd.concat([_table(30, caseid=1), _table(30, caseid=2)], ignore_index=True)
    bundle = ablation.run_ablation(table)
    counts = ablation.per_case_counts(bundle["results_a"], bundle["results_b"])
    assert counts["flagged_a"].sum() == int(bundle["results_a"]["anomaly_label"].sum())
    assert counts["flagged_b"].sum() == int(bundle["results_b"]["anomaly_label"].sum())
    assert counts["analyzed"].sum() == len(bundle["results_a"])


# ── Reporting ────────────────────────────────────────────────────────────────


def test_coverage_profile_counts_flags_by_data_quality():
    table = _table(60)
    table.loc[59, ["hr_coverage_pct", "spo2_coverage_pct", "rr_coverage_pct"]] = 72.0
    bundle = ablation.run_ablation(table)
    comparison = ablation.compare(bundle["results_a"], bundle["results_b"])
    profile = ablation.coverage_profile(comparison, table)

    assert profile["analyzed_population"]["below_100pct_coverage"] == 1
    for variant in ("model_a", "model_b"):
        entry = profile[variant]
        assert (entry["flagged_with_full_coverage"]
                + entry["flagged_below_100pct_coverage"] == entry["n_flagged"])


def test_report_contains_no_supervised_metrics():
    bundle = ablation.run_ablation(_table(40))
    report = ablation.build_report(bundle, _table(40))
    text = json.dumps(report, default=str).lower()
    for metric in ("accuracy", "precision", "recall", "f1_", "auroc", "auprc"):
        assert metric not in text


def test_report_answers_the_landmark_questions():
    bundle = ablation.run_ablation(_table(40))
    report = ablation.build_report(bundle, _table(40))
    assert "case4_cluster" in report
    assert "case8_window16" in report
    assert "transitions" in report
    assert report["model_a"]["n_features"] == 18
    assert report["model_b"]["n_features"] == 15


def test_report_declares_the_held_identical_conditions():
    bundle = ablation.run_ablation(_table(30))
    held = bundle["config"]["held_identical"]
    assert "selected rows" in held and "seed" in held and "contamination" in held
    assert bundle["config"]["removed_features"] == ablation.COVERAGE_FEATURES
