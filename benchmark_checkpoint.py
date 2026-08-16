"""Run the four Model B benchmarks and write results/benchmark/.

Analysis only. Reads the canonical model artifacts and the feature table; writes
nothing outside results/benchmark/.

    python benchmark_checkpoint.py
"""

import json
import sys

import pandas as pd

sys.path.insert(0, "src")

from vitaldb_audit import anomaly, benchmark, evidence, features  # noqa: E402

pd.set_option("display.width", 220)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    table = pd.read_csv(features.FEATURES_DIR / "feature_table_5min.csv")
    canonical = pd.read_csv(anomaly.MODEL_DIR / "anomaly_results.csv")
    selected, selection = anomaly.select_model_rows(table)
    evidence_document = json.loads(
        (evidence.XAI_DIR / "evidence_cases.json").read_text(encoding="utf-8"))

    n_flagged = int(canonical["anomaly_label"].sum())
    benchmark.BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    rule("MODEL B BENCHMARK — ANALYSIS ONLY")
    print(f"  analyzed windows : {len(canonical)}")
    print(f"  flagged          : {n_flagged}")
    print(f"  canonical seed   : {benchmark.CANONICAL_SEED}")
    print(f"  NO GROUND TRUTH  : nothing here measures correctness")

    # ── Benchmark 1 ──────────────────────────────────────────────────────────
    rule("BENCHMARK 1 — SEED STABILITY")
    seed_frame = benchmark.seed_stability(selected, canonical)
    seed_frame.to_csv(benchmark.BENCHMARK_DIR / "seed_stability.csv", index=False)
    print(seed_frame[["seed", "is_canonical", "windows_flagged",
                      "top10_overlap_with_canonical", "spearman_rank_correlation",
                      "pct_labels_unchanged"]].to_string(index=False))

    alternates = seed_frame[~seed_frame["is_canonical"]]
    seed_summary = {
        "seeds_tested": seed_frame["seed"].tolist(),
        "canonical_seed": benchmark.CANONICAL_SEED,
        "flagged_min": int(seed_frame["windows_flagged"].min()),
        "flagged_max": int(seed_frame["windows_flagged"].max()),
        "top10_overlap_min": int(alternates["top10_overlap_with_canonical"].min()),
        "top10_overlap_max": int(alternates["top10_overlap_with_canonical"].max()),
        "top10_overlap_mean": round(
            float(alternates["top10_overlap_with_canonical"].mean()), 2),
        "spearman_min": round(float(alternates["spearman_rank_correlation"].min()), 4),
        "spearman_max": round(float(alternates["spearman_rank_correlation"].max()), 4),
        "spearman_mean": round(float(alternates["spearman_rank_correlation"].mean()), 4),
        "pct_labels_unchanged_min": round(float(alternates["pct_labels_unchanged"].min()), 2),
        "pct_labels_unchanged_max": round(float(alternates["pct_labels_unchanged"].max()), 2),
        "pct_labels_unchanged_mean": round(float(alternates["pct_labels_unchanged"].mean()), 2),
    }

    # ── Benchmark 2 ──────────────────────────────────────────────────────────
    rule("BENCHMARK 2 — SIMPLE STATISTICAL BASELINE")
    baseline, baseline_meta = benchmark.robust_zscore_baseline(selected, n_flagged)
    comparison, comparison_summary = benchmark.compare_to_isolation_forest(
        baseline, canonical)
    comparison.to_csv(
        benchmark.BENCHMARK_DIR / "baseline_comparison.csv", index=False)

    print(f"  method  : {baseline_meta['method']}")
    print(f"  features: {baseline_meta['features_used']} of "
          f"{baseline_meta['features_available']} used; excluded as degenerate: "
          f"{baseline_meta['features_excluded_degenerate']}")
    print(f"\n  flagged by both        : {comparison_summary['flagged_by_both']}")
    print(f"  isolation forest only  : {comparison_summary['isolation_forest_only']}")
    print(f"  baseline only          : {comparison_summary['baseline_only']}")
    print(f"  top-10 overlap         : {comparison_summary['top10_overlap']} of 10")
    print(f"  Spearman               : {comparison_summary['spearman_rank_correlation']}")

    episode_iso = benchmark.episode_detected(comparison, "anomaly_label")
    episode_base = benchmark.episode_detected(comparison, "baseline_label")
    print(f"\n  case 4 episode (285-340 min):")
    print(f"    isolation forest : {episode_iso['flagged']}/"
          f"{episode_iso['windows_in_range']} -> {episode_iso['window_indices']}")
    print(f"    baseline         : {episode_base['flagged']}/"
          f"{episode_base['windows_in_range']} -> {episode_base['window_indices']}")

    print(f"\n  top 10 by baseline:")
    print(comparison.nsmallest(10, "baseline_rank")[
        ["caseid", "window_index", "baseline_score", "driving_feature",
         "driving_z", "anomaly_rank", "agreement"]].to_string(index=False))

    baseline_summary = {
        **baseline_meta, **comparison_summary,
        "episode_isolation_forest": episode_iso,
        "episode_baseline": episode_base,
    }

    # ── Benchmark 3 ──────────────────────────────────────────────────────────
    rule("BENCHMARK 3 — TEMPORAL COHERENCE")
    coherence = benchmark.temporal_coherence(canonical)
    (benchmark.BENCHMARK_DIR / "temporal_coherence.json").write_text(
        json.dumps(coherence, indent=2), encoding="utf-8")
    overall = coherence["overall"]
    print(f"  flagged                        : {overall['flagged']}")
    print(f"  with adjacent flagged neighbour: {overall['with_adjacent_flagged_neighbour']}")
    print(f"  pct in contiguous runs         : {overall['pct_in_contiguous_runs']}%")
    print(f"  runs                           : {overall['n_runs']}  "
          f"lengths {overall['run_lengths']}")
    print(f"  longest run                    : {overall['longest_run']} windows "
          f"({overall['longest_run'] * 5} min)")
    print(f"  isolated single flags          : {overall['isolated_flags']}")
    print()
    print(pd.DataFrame(coherence["per_case"]).to_string(index=False))

    # ── Benchmark 4 ──────────────────────────────────────────────────────────
    rule("BENCHMARK 4 — EVIDENCE REVIEW")
    review = benchmark.review_flagged_windows(evidence_document, table)
    review.to_csv(
        benchmark.BENCHMARK_DIR / "flagged_window_review.csv", index=False)
    reviewed = benchmark.review_summary(review)
    print(review[["caseid", "window_index", "anomaly_rank", "min_coverage_pct",
                  "evidence_drivers", "review"]].to_string(index=False))
    print(f"\n  physiologically supported : {reviewed['physiologically_supported']}")
    print(f"  mainly data quality       : {reviewed['mainly_data_quality']}")
    print(f"  ambiguous                 : {reviewed['ambiguous']}")

    # ── Interpretations ──────────────────────────────────────────────────────
    seed_summary["interpretation"] = (
        f"The ranking is highly reproducible across seeds (Spearman "
        f"{seed_summary['spearman_min']}–{seed_summary['spearman_max']}), and the "
        f"flag count is fixed at {n_flagged} by the contamination parameter rather "
        f"than by anything the model discovered, so it cannot vary. "
        f"{seed_summary['pct_labels_unchanged_mean']}% of labels are unchanged on "
        f"average, meaning the disagreement is confined to a small number of "
        f"windows near the decision boundary — exactly where a fixed budget forces "
        f"an arbitrary cut. This says the PROCEDURE is stable. It says nothing "
        f"about whether the flagged windows are meaningful, and a seed sweep "
        f"cannot address that."
    )
    baseline_summary["interpretation"] = (
        f"A deterministic robust z-score with no model recovers "
        f"{comparison_summary['flagged_by_both']} of the "
        f"{comparison_summary['isolation_forest_flagged']} Isolation Forest flags "
        f"and {comparison_summary['top10_overlap']} of its top 10, with Spearman "
        f"{comparison_summary['spearman_rank_correlation']} across all 135 windows. "
        f"Both methods identify the case 4 episode. Two unsupervised methods "
        f"agreeing is not evidence that either is correct — they read the same 15 "
        f"features, so shared blind spots are expected rather than surprising. "
        f"What it does establish is that the flags are not an artifact of tree "
        f"ensembling: a transparent rule reaches largely the same conclusion. "
        f"Where they differ is informative, because the baseline scores on a "
        f"single most-extreme feature while the Isolation Forest can combine "
        f"several moderately unusual ones. Neither is declared superior here; the "
        f"evidence does not support that claim in either direction."
    )
    coherence_interpretation = (
        f"{overall['pct_in_contiguous_runs']}% of flags sit adjacent to another "
        f"flag, in {overall['n_runs']} runs, the longest spanning "
        f"{overall['longest_run']} consecutive windows "
        f"({overall['longest_run'] * 5} minutes). Independent noise over "
        f"{overall['flagged']} flags in {overall['windows_analyzed']} windows would "
        f"rarely produce a run that long, so the flags are picking up something "
        f"with temporal extent rather than firing at random. Two cautions: the "
        f"model scores each window independently with no temporal features, so "
        f"clustering is a property of the underlying signal and not of the "
        f"detector; and a sustained physiological state and a sustained artifact "
        f"both produce runs, so coherence alone does not distinguish them."
    )

    reviewed["interpretation"] = (
        f"{reviewed['physiologically_supported']} of {reviewed['windows_reviewed']} "
        f"flags rest on signal behaviour at full coverage, "
        f"{reviewed['mainly_data_quality']} on data quality alone, and "
        f"{reviewed['ambiguous']} are ambiguous. This is partly by construction: "
        f"the coverage ablation removed the features that previously drove "
        f"data-quality flags, so a low count here reflects that earlier decision "
        f"working as intended rather than an independent discovery. The category "
        f"names describe what the evidence points at, not whether the physiology "
        f"is abnormal."
    )

    # ── Summary and report ───────────────────────────────────────────────────
    summary = {
        "benchmark": "model_b_baseline",
        "ground_truth": (
            "none available; no accuracy, precision, recall, F1, AUROC or AUPRC "
            "is defined for this experiment and none is reported"
        ),
        "model": {
            "algorithm": "sklearn.ensemble.IsolationForest",
            "n_features": len(anomaly.MODEL_FEATURES) - 3,
            "n_estimators": anomaly.DEFAULT_N_ESTIMATORS,
            "contamination": anomaly.DEFAULT_CONTAMINATION,
            "canonical_seed": benchmark.CANONICAL_SEED,
        },
        "windows_analyzed": int(len(canonical)),
        "windows_flagged": n_flagged,
        "data_selection": selection,
        "benchmark_1_seed_stability": seed_summary,
        "benchmark_2_simple_baseline": baseline_summary,
        "benchmark_3_temporal_coherence": {
            **coherence, "interpretation": coherence_interpretation},
        "benchmark_4_evidence_review": reviewed,
        "limitations": [
            "No ground truth exists, so none of these benchmarks measures "
            "correctness. They measure reproducibility, agreement, structure and "
            "composition.",
            "Seed stability and baseline agreement both rest on the same 15 "
            "features over the same 135 windows. A blind spot shared by both "
            "methods is invisible to this benchmark by construction.",
            "The flagged count is fixed by contamination = 0.10 in every run, so "
            "no benchmark here can tell whether 14 is the right number.",
            "Four cases, and case 4 supplies nearly half the analyzed windows, so "
            "'unusual' is defined largely by one case.",
            "Temporal coherence does not distinguish a sustained physiological "
            "state from a sustained artifact.",
            "The evidence review applies a mechanical rule to existing evidence "
            "objects. It is not clinical review and involved no clinician.",
        ],
    }
    (benchmark.BENCHMARK_DIR / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    report = benchmark.render_report(summary, seed_frame, comparison, coherence, review)
    (benchmark.BENCHMARK_DIR / "benchmark_report.md").write_text(
        report, encoding="utf-8")

    rule("FILES")
    for path in sorted(benchmark.BENCHMARK_DIR.glob("*")):
        print(f"  {path.name:<30}{path.stat().st_size / 1024:>8.1f} KB")
    print(f"\n  Existing model artifacts untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
