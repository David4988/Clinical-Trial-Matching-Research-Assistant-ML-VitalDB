"""Tests for the Model B benchmarks.

These benchmarks exist to make claims about stability, agreement and structure,
so the tests concentrate on the arithmetic that backs those claims — run
detection, budget matching, degenerate scale handling — and on keeping
supervised metrics out of a report that has no ground truth.
"""

import json

import numpy as np
import pandas as pd
import pytest

from vitaldb_audit import ablation, anomaly, benchmark


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


def _results(flagged_keys, all_keys):
    """Canonical-shaped results frame from explicit (case, window) keys."""
    rows = []
    for rank, (caseid, window) in enumerate(all_keys, start=1):
        rows.append({
            "caseid": caseid, "window_index": window,
            "window_start_s": window * 300.0, "window_end_s": (window + 1) * 300.0,
            "anomaly_rank": rank,
            "anomaly_score": 0.1 if (caseid, window) in flagged_keys else -0.1,
            "anomaly_label": 1 if (caseid, window) in flagged_keys else 0,
        })
    return pd.DataFrame(rows)


# ── Benchmark 1 — seed stability ─────────────────────────────────────────────


def test_canonical_seed_row_agrees_perfectly_with_itself():
    table = _table(60)
    selected, _ = anomaly.select_model_rows(table)
    _, canonical = ablation.run_variant(selected, ablation.MODEL_B_FEATURES)

    frame = benchmark.seed_stability(selected, canonical, seeds=[1, 7])
    row = frame[frame["is_canonical"]].iloc[0]
    assert row["spearman_rank_correlation"] == pytest.approx(1.0)
    assert row["top10_overlap_with_canonical"] == 10
    assert row["pct_labels_unchanged"] == pytest.approx(100.0)


def test_every_seed_scores_the_same_windows():
    table = _table(60)
    selected, _ = anomaly.select_model_rows(table)
    _, canonical = ablation.run_variant(selected, ablation.MODEL_B_FEATURES)
    frame = benchmark.seed_stability(selected, canonical, seeds=[1, 7, 42])
    assert (frame["windows_analyzed"] == len(selected)).all()


def test_contamination_fixes_the_flag_count_across_seeds():
    """Flag count cannot vary with the seed; it is a budget, not a discovery."""
    table = _table(60)
    selected, _ = anomaly.select_model_rows(table)
    _, canonical = ablation.run_variant(selected, ablation.MODEL_B_FEATURES)
    frame = benchmark.seed_stability(selected, canonical, seeds=[1, 7, 42])
    assert frame["windows_flagged"].nunique() == 1


def test_seed_stability_includes_the_canonical_row_first():
    table = _table(40)
    selected, _ = anomaly.select_model_rows(table)
    _, canonical = ablation.run_variant(selected, ablation.MODEL_B_FEATURES)
    frame = benchmark.seed_stability(selected, canonical, seeds=[1])
    assert frame.iloc[0]["seed"] == benchmark.CANONICAL_SEED
    assert int(frame["is_canonical"].sum()) == 1


# ── Benchmark 2 — robust baseline ────────────────────────────────────────────


def test_robust_scale_prefers_mad():
    scale, estimator = benchmark.robust_scale(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert estimator == "MAD"
    assert scale > 0


def test_robust_scale_falls_back_to_iqr_when_mad_is_zero():
    """Real case: SpO2 std is zero in more than half the windows."""
    values = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    scale, estimator = benchmark.robust_scale(values)
    assert estimator == "IQR"
    assert scale > 0


def test_robust_scale_reports_degenerate_instead_of_dividing_by_zero():
    scale, estimator = benchmark.robust_scale(pd.Series([5.0] * 10))
    assert estimator == "degenerate"
    assert scale == 0.0


def test_baseline_excludes_degenerate_features_rather_than_producing_infinities():
    table = _table(40)
    table["spo2_delta"] = 0.0            # perfectly constant
    selected, _ = anomaly.select_model_rows(table)
    scored, meta = benchmark.robust_zscore_baseline(selected, k_flagged=4)

    assert "spo2_delta" in meta["features_excluded_degenerate"]
    assert meta["features_used"] < meta["features_available"]
    assert np.isfinite(scored["baseline_score"]).all()


def test_baseline_flags_exactly_the_matched_budget():
    table = _table(40)
    selected, _ = anomaly.select_model_rows(table)
    for k in (5, 14):
        scored, _ = benchmark.robust_zscore_baseline(selected, k_flagged=k)
        assert int(scored["baseline_label"].sum()) == k


def test_baseline_is_deterministic():
    """No fitting, no randomness — two runs must be identical."""
    table = _table(40)
    selected, _ = anomaly.select_model_rows(table)
    first, _ = benchmark.robust_zscore_baseline(selected, k_flagged=4)
    second, _ = benchmark.robust_zscore_baseline(selected, k_flagged=4)
    pd.testing.assert_frame_equal(first, second)


def test_baseline_ranks_a_planted_outlier_first_and_names_the_feature():
    table = _table(40)
    table.loc[39, "hr_std"] = 500.0
    selected, _ = anomaly.select_model_rows(table)
    scored, _ = benchmark.robust_zscore_baseline(selected, k_flagged=4)

    top = scored.iloc[0]
    assert int(top["window_index"]) == 39
    assert top["driving_feature"] == "hr_std"
    assert top["driving_z"] > 0


def test_baseline_ranks_are_dense_and_ordered():
    table = _table(30)
    selected, _ = anomaly.select_model_rows(table)
    scored, _ = benchmark.robust_zscore_baseline(selected, k_flagged=3)
    assert scored["baseline_rank"].tolist() == list(range(1, len(scored) + 1))
    assert scored["baseline_score"].is_monotonic_decreasing


def test_comparison_agreement_categories_are_exhaustive_and_reconcile():
    table = _table(40)
    selected, _ = anomaly.select_model_rows(table)
    _, canonical = ablation.run_variant(selected, ablation.MODEL_B_FEATURES)
    k = int(canonical["anomaly_label"].sum())
    baseline, _ = benchmark.robust_zscore_baseline(selected, k_flagged=k)

    merged, summary = benchmark.compare_to_isolation_forest(baseline, canonical)
    assert len(merged) == len(canonical)
    assert (summary["flagged_by_both"] + summary["isolation_forest_only"]
            + summary["baseline_only"] + summary["flagged_by_neither"]) == len(merged)
    assert (summary["flagged_by_both"] + summary["isolation_forest_only"]
            == summary["isolation_forest_flagged"])
    assert (summary["flagged_by_both"] + summary["baseline_only"]
            == summary["baseline_flagged"])


def test_episode_detection_reports_windows_in_range():
    keys = [(4, i) for i in range(70)]
    results = _results({(4, 60), (4, 61)}, keys)
    episode = benchmark.episode_detected(results, "anomaly_label",
                                         caseid=4, start_min=285.0, end_min=340.0)
    assert episode["window_indices"] == [60, 61]
    assert episode["windows_in_range"] == 11


# ── Benchmark 3 — temporal coherence ─────────────────────────────────────────


def test_runs_detects_one_contiguous_block():
    keys = [(1, i) for i in range(10)]
    results = _results({(1, 3), (1, 4), (1, 5)}, keys)
    runs = benchmark.flagged_runs(results)
    assert runs == [{"caseid": 1, "start_window": 3, "end_window": 5, "length": 3}]


def test_runs_splits_on_a_gap():
    keys = [(1, i) for i in range(10)]
    results = _results({(1, 1), (1, 2), (1, 6)}, keys)
    runs = benchmark.flagged_runs(results)
    assert [r["length"] for r in runs] == [2, 1]


def test_runs_never_span_two_cases():
    keys = [(1, 5), (2, 6)]
    results = _results({(1, 5), (2, 6)}, keys)
    runs = benchmark.flagged_runs(results)
    assert len(runs) == 2
    assert all(r["length"] == 1 for r in runs)


def test_coherence_counts_reconcile():
    keys = [(1, i) for i in range(12)]
    results = _results({(1, 2), (1, 3), (1, 4), (1, 9)}, keys)
    coherence = benchmark.temporal_coherence(results)
    overall = coherence["overall"]

    assert overall["flagged"] == 4
    assert overall["with_adjacent_flagged_neighbour"] == 3
    assert overall["longest_run"] == 3
    assert overall["isolated_flags"] == 1
    assert overall["pct_in_contiguous_runs"] == pytest.approx(75.0)
    assert sum(overall["run_lengths"]) == overall["flagged"]


def test_coherence_handles_a_case_with_no_flags():
    keys = [(1, i) for i in range(5)]
    results = _results(set(), keys)
    overall = benchmark.temporal_coherence(results)["overall"]
    assert overall["flagged"] == 0
    assert overall["n_runs"] == 0
    assert overall["longest_run"] == 0
    assert overall["pct_in_contiguous_runs"] == 0.0


def test_per_case_coherence_sums_to_the_overall_flag_count():
    keys = [(1, i) for i in range(8)] + [(2, i) for i in range(8)]
    results = _results({(1, 1), (1, 2), (2, 5)}, keys)
    coherence = benchmark.temporal_coherence(results)
    assert (sum(c["flagged"] for c in coherence["per_case"])
            == coherence["overall"]["flagged"])


# ── Benchmark 4 — evidence review ────────────────────────────────────────────


def _evidence_entry(caseid, window, changed=0, dispersion=0, rank=1):
    observations = {
        "n_signals_changed": changed,
        "n_signals_dispersion_unusual": dispersion,
        "hr_changed": changed > 0, "spo2_changed": False, "rr_changed": False,
        "hr_dispersion_unusual": dispersion > 0,
        "spo2_dispersion_unusual": False, "rr_dispersion_unusual": False,
    }
    return {
        "case_id": caseid, "window_index": window, "anomaly_rank": rank,
        "anomaly_score": 0.1,
        "time_range": {"label": f"{window * 5}.0-{(window + 1) * 5}.0 min from case start"},
        "observations": observations,
    }


def _review_for(coverage, changed, dispersion):
    table = _table(10)
    for column in ablation.COVERAGE_FEATURES:
        table.loc[table["window_index"] == 3, column] = coverage
    document = {"evidence": [_evidence_entry(1, 3, changed, dispersion)]}
    return benchmark.review_flagged_windows(document, table).iloc[0]["review"]


def test_full_coverage_with_evidence_is_physiologically_supported():
    assert _review_for(100.0, changed=1, dispersion=0) == "physiologically_supported"
    assert _review_for(100.0, changed=0, dispersion=1) == "physiologically_supported"


def test_low_coverage_without_evidence_is_data_quality():
    assert _review_for(72.0, changed=0, dispersion=0) == "mainly_data_quality"


def test_low_coverage_with_evidence_is_ambiguous():
    assert _review_for(72.0, changed=1, dispersion=0) == "ambiguous"


def test_full_coverage_without_evidence_is_ambiguous():
    assert _review_for(100.0, changed=0, dispersion=0) == "ambiguous"


def test_review_summary_counts_reconcile():
    table = _table(10)
    document = {"evidence": [
        _evidence_entry(1, 1, changed=1, rank=1),
        _evidence_entry(1, 2, dispersion=1, rank=2),
        _evidence_entry(1, 3, rank=3),
    ]}
    review = benchmark.review_flagged_windows(document, table)
    summary = benchmark.review_summary(review)
    assert (summary["physiologically_supported"] + summary["mainly_data_quality"]
            + summary["ambiguous"]) == summary["windows_reviewed"] == 3


def test_review_refuses_the_clinical_framing():
    table = _table(10)
    document = {"evidence": [_evidence_entry(1, 1, changed=1)]}
    review = benchmark.review_flagged_windows(document, table)
    disclaimer = benchmark.review_summary(review)["not_clinical_validation"].lower()
    assert "no clinical review has taken place" in disclaimer
    assert "not a claim" in disclaimer
    assert "medically accurate" in disclaimer


# ── No supervised metrics ────────────────────────────────────────────────────


SUPERVISED = ("accuracy", "precision", "recall", "f1_", "auroc", "auprc",
              "roc_auc", "true_positive", "false_positive", "sensitivity",
              "specificity")


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


def test_no_supervised_metric_keys_in_any_benchmark_output():
    keys = [(1, i) for i in range(12)]
    results = _results({(1, 2), (1, 3)}, keys)
    coherence = benchmark.temporal_coherence(results)

    table = _table(10)
    document = {"evidence": [_evidence_entry(1, 1, changed=1)]}
    reviewed = benchmark.review_summary(
        benchmark.review_flagged_windows(document, table))

    for payload in (coherence, reviewed):
        for key in _all_keys(payload):
            assert not any(metric in key for metric in SUPERVISED), key


def test_generated_report_separates_observation_from_interpretation():
    report_path = benchmark.BENCHMARK_DIR / "benchmark_report.md"
    if not report_path.exists():
        pytest.skip("benchmark_checkpoint.py has not been run")
    text = report_path.read_text(encoding="utf-8")
    assert text.count("### OBSERVED RESULTS") == 4
    assert text.count("### INTERPRETATION") == 4
    assert "no ground-truth anomaly label" in text.lower()
