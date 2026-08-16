"""Tests for the descriptive class profile.

The point of this report is that it is honest about having no ground truth, so
the tests that matter most are the ones that keep supervised metrics out and
keep the class arithmetic reconciling.
"""

import json

import numpy as np
import pandas as pd
import pytest

from vitaldb_audit import ablation, class_profile


# ── Helpers ──────────────────────────────────────────────────────────────────


def _table(n=40, caseid=1, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        record = {
            "caseid": caseid, "window_index": i,
            "window_start_s": i * 300.0, "window_end_s": (i + 1) * 300.0,
            "n_core_usable": 3, "window_usable": True,
            "consecutive_usable_windows": i + 1,
        }
        for short, base in (("hr", 70.0), ("spo2", 99.0), ("rr", 12.0)):
            mean = base + rng.normal(0, 1.5)
            record[f"{short}_mean"] = mean
            record[f"{short}_std"] = abs(rng.normal(1.5, 0.4))
            record[f"{short}_min"] = mean - 4
            record[f"{short}_max"] = mean + 4
            record[f"{short}_delta"] = rng.normal(0, 1.0)
            record[f"{short}_coverage_pct"] = 100.0
            record[f"{short}_usable"] = True
        rows.append(record)
    return pd.DataFrame(rows)


def _results(table, flagged_indices):
    return pd.DataFrame([
        {
            "caseid": int(r.caseid), "window_index": int(r.window_index),
            "window_start_s": r.window_start_s, "window_end_s": r.window_end_s,
            "anomaly_rank": i + 1,
            "anomaly_score": 0.1 if r.window_index in flagged_indices else -0.1,
            "anomaly_label": 1 if r.window_index in flagged_indices else 0,
        }
        for i, r in enumerate(table.itertuples())
    ])


def _selection(table, results):
    return {
        "rows_in_feature_table": len(table),
        "rows_window_usable": len(results),
        "rows_excluded_unusable_window": len(table) - len(results),
        "rows_excluded_null_feature": 0,
        "rows_analyzed": len(results),
    }


def _profile(n=40, flagged=(0, 1, 2, 3)):
    table = _table(n)
    results = _results(table, set(flagged))
    return class_profile.build_profile(results, table, _selection(table, results))


# ── No supervised metrics ────────────────────────────────────────────────────


SUPERVISED = ("accuracy", "precision", "recall", "f1", "auroc", "auprc",
              "roc_auc", "confusion", "true_positive", "false_positive",
              "sensitivity", "specificity")


def _all_keys(node):
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.append(str(key).lower())
            found += _all_keys(value)
    elif isinstance(node, list):
        for item in node:
            found += _all_keys(item)
    return found


def test_profile_reports_no_supervised_metric_keys():
    profile = _profile()
    for key in _all_keys(profile):
        assert not any(metric in key for metric in SUPERVISED), key


def test_profile_reports_no_supervised_metric_values():
    """The ground_truth field is allowed to name them, since its job is to say
    they are undefined here."""
    profile = _profile()
    disclaimer = profile.pop("ground_truth").lower()
    assert "no supervised performance measure" in disclaimer

    remaining = json.dumps(profile).lower()
    for metric in SUPERVISED:
        assert metric not in remaining, metric


def test_profile_states_there_is_no_ground_truth():
    profile = _profile()
    assert profile["ground_truth"].startswith("none available")
    assert "not" in profile["interpretation"].lower()


# ── Support arithmetic ───────────────────────────────────────────────────────


def test_class_supports_sum_to_the_analyzed_total():
    profile = _profile(n=40, flagged=(0, 1, 2, 3, 4))
    support = profile["support"]
    total = sum(entry["support"] for entry in support["classes"])
    assert total == support["analyzed_total"] == 40


def test_class_percentages_sum_to_100():
    profile = _profile(n=40, flagged=(0, 1, 2, 3))
    percentages = sum(e["pct_of_analyzed"] for e in profile["support"]["classes"])
    assert percentages == pytest.approx(100.0, abs=0.05)


def test_unclassified_windows_are_accounted_for_not_hidden():
    table = _table(40)
    results = _results(table, {0, 1}).iloc[:35]
    selection = {
        "rows_in_feature_table": 40, "rows_window_usable": 37,
        "rows_excluded_unusable_window": 3, "rows_excluded_null_feature": 2,
        "rows_analyzed": 35,
    }
    profile = class_profile.build_profile(results, table, selection)
    unclassified = profile["support"]["not_classified"]
    assert unclassified["excluded_window_not_usable"] == 3
    assert unclassified["excluded_null_model_feature"] == 2
    assert "not a third class" in unclassified["note"]


def test_class_balance_is_declared_a_parameter_not_a_finding():
    profile = _profile()
    assert "contamination" in profile["support"]["class_balance_note"]


def test_per_case_composition_reconciles():
    table = pd.concat([_table(20, caseid=1), _table(20, caseid=2, seed=9)],
                      ignore_index=True)
    results = _results(table, {0, 1, 2})
    profile = class_profile.build_profile(results, table, _selection(table, results))
    for entry in profile["per_case_composition"]:
        assert entry["flagged"] + entry["not_flagged"] == entry["analyzed"]
    assert sum(e["analyzed"] for e in profile["per_case_composition"]) == len(results)


# ── Score profile ────────────────────────────────────────────────────────────


def test_score_profile_covers_both_classes():
    profile = _profile()
    names = {entry["class"] for entry in profile["score_distribution_by_class"]}
    assert names == {"flagged", "not_flagged"}


def test_score_profile_quantiles_are_ordered():
    profile = _profile()
    for entry in profile["score_distribution_by_class"]:
        if entry["support"] < 2:
            continue
        assert entry["min"] <= entry["q25"] <= entry["median"] <= entry["q75"] <= entry["max"]


def test_empty_class_does_not_crash_the_profile():
    table = _table(20)
    results = _results(table, set())          # nothing flagged
    profile = class_profile.build_profile(results, table, _selection(table, results))
    flagged = [e for e in profile["score_distribution_by_class"]
               if e["class"] == "flagged"][0]
    assert flagged["support"] == 0


# ── Cliff's delta ────────────────────────────────────────────────────────────


def test_cliffs_delta_is_one_for_complete_separation():
    assert class_profile.cliffs_delta([10, 11, 12], [1, 2, 3]) == pytest.approx(1.0)


def test_cliffs_delta_is_minus_one_when_reversed():
    assert class_profile.cliffs_delta([1, 2, 3], [10, 11, 12]) == pytest.approx(-1.0)


def test_cliffs_delta_is_zero_for_identical_samples():
    assert class_profile.cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_cliffs_delta_handles_partial_overlap():
    # 2 of 3 values in a exceed the median of b
    delta = class_profile.cliffs_delta([1, 5, 6], [2, 3, 4])
    assert -1.0 < delta < 1.0


def test_cliffs_delta_is_none_for_an_empty_sample():
    assert class_profile.cliffs_delta([], [1, 2, 3]) is None


def test_cliffs_delta_ignores_nulls():
    assert class_profile.cliffs_delta(
        [10, 11, np.nan], [1, 2]) == pytest.approx(1.0)


def test_magnitude_bands():
    assert class_profile.magnitude(0.05) == "negligible"
    assert class_profile.magnitude(0.25) == "small"
    assert class_profile.magnitude(0.40) == "medium"
    assert class_profile.magnitude(0.90) == "large"
    assert class_profile.magnitude(None) == "not computable"


def test_magnitude_uses_absolute_value():
    assert class_profile.magnitude(-0.9) == "large"


# ── Feature separation ───────────────────────────────────────────────────────


def test_feature_separation_covers_every_model_feature():
    profile = _profile()
    reported = {r["feature"] for r in profile["feature_separation"]["model_features"]}
    assert reported == set(ablation.MODEL_B_FEATURES)


def test_feature_separation_is_sorted_by_absolute_effect():
    profile = _profile()
    deltas = [abs(r["cliffs_delta"] or 0)
              for r in profile["feature_separation"]["model_features"]]
    assert deltas == sorted(deltas, reverse=True)


def test_feature_separation_detects_a_planted_difference():
    table = _table(40)
    table.loc[[0, 1, 2, 3], "hr_std"] = 50.0      # flagged windows get huge spread
    results = _results(table, {0, 1, 2, 3})
    profile = class_profile.build_profile(results, table, _selection(table, results))
    hr_std = [r for r in profile["feature_separation"]["model_features"]
              if r["feature"] == "hr_std"][0]
    assert hr_std["cliffs_delta"] == pytest.approx(1.0)
    assert hr_std["separation"] == "large"


def test_coverage_features_are_profiled_separately_as_non_inputs():
    profile = _profile()
    context = profile["feature_separation"]["context_features_not_model_inputs"]
    assert {r["feature"] for r in context} == set(ablation.COVERAGE_FEATURES)
    for feature in ablation.COVERAGE_FEATURES:
        assert feature not in ablation.MODEL_B_FEATURES


def test_separation_measure_is_labelled_as_not_importance():
    profile = _profile()
    measure = profile["feature_separation"]["measure"].lower()
    assert "not a performance measure" in measure
    assert "not feature importance" in measure


# ── Serialisation ────────────────────────────────────────────────────────────


def test_profile_serialises_without_a_custom_encoder():
    profile = _profile()
    reloaded = json.loads(json.dumps(profile))
    assert reloaded["report"] == "descriptive_class_profile"
