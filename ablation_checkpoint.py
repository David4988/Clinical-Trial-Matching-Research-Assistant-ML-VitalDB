"""Coverage-feature ablation: MODEL A (18 features) vs MODEL B (15 features).

MODEL B removes only hr_coverage_pct, spo2_coverage_pct and rr_coverage_pct.
Every other input, parameter, seed and training row is held identical.

No ground truth exists, so no supervised performance measure is reported and
no winner is selected by the code.

    python ablation_checkpoint.py [--contamination 0.10] [--seed 20260816]
"""

import argparse
import sys

import pandas as pd

sys.path.insert(0, "src")

from vitaldb_audit import ablation, anomaly, features  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def print_top(results, table, label):
    top = ablation.top_windows(results, n=10)
    detail = anomaly.attach_features(results.head(10), table)
    print(f"\n  {label}")
    print(f"  {'rank':>4}  {'case':>4}  {'win':>4}  {'time range':>16}  {'score':>9}  "
          f"{'min cov %':>9}")
    print(f"  {'-' * 60}")
    for rank, ((_, row), (_, det)) in enumerate(
            zip(top.iterrows(), detail.iterrows()), start=1):
        min_cov = min(det[c] for c in ablation.COVERAGE_FEATURES)
        print(f"  {rank:>4}  {int(row['caseid']):>4}  {int(row['window_index']):>4}  "
              f"{row['time_range']:>16}  {row['anomaly_score']:>+9.4f}  {min_cov:>9.2f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contamination", type=float,
                        default=anomaly.DEFAULT_CONTAMINATION)
    parser.add_argument("--seed", type=int, default=anomaly.RANDOM_SEED)
    parser.add_argument("--n-estimators", type=int,
                        default=anomaly.DEFAULT_N_ESTIMATORS)
    args = parser.parse_args(argv)

    table = pd.read_csv(features.FEATURES_DIR / "feature_table_5min.csv")

    rule("CONTROLLED ABLATION — COVERAGE FEATURES")
    print(f"  MODEL A : {len(ablation.MODEL_A_FEATURES)} features (current baseline)")
    print(f"  MODEL B : {len(ablation.MODEL_B_FEATURES)} features "
          f"(minus {', '.join(ablation.COVERAGE_FEATURES)})")
    print(f"\n  held identical: contamination={args.contamination}, seed={args.seed}, "
          f"n_estimators={args.n_estimators},")
    print(f"                  same selected rows, same feature values, no "
          f"standardization, no imputation")

    bundle = ablation.run_ablation(
        table,
        contamination=args.contamination,
        random_state=args.seed,
        n_estimators=args.n_estimators,
    )
    comparison = ablation.compare(bundle["results_a"], bundle["results_b"])
    report = ablation.build_report(bundle, table)

    ablation.ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(ablation.ABLATION_DIR / "ablation_comparison.csv", index=False)
    bundle["results_b"].to_csv(
        ablation.ABLATION_DIR / "anomaly_results_model_b.csv", index=False)
    report_path = ablation.write_report(report)

    # ── 1 & 2 ────────────────────────────────────────────────────────────────
    rule("1-2. WINDOWS ANALYZED AND FLAGGED")
    print(f"  {'':<10}{'features':>9}{'analyzed':>10}{'flagged':>9}{'pct':>8}")
    for name, key in (("MODEL A", "model_a"), ("MODEL B", "model_b")):
        entry = report[key]
        pct = entry["windows_flagged"] / entry["windows_analyzed"] * 100
        print(f"  {name:<10}{entry['n_features']:>9}{entry['windows_analyzed']:>10}"
              f"{entry['windows_flagged']:>9}{pct:>7.1f}%")
    print(f"\n  Identical analyzed population: "
          f"{report['data_selection']['rows_analyzed']} rows "
          f"({report['data_selection']['rows_excluded_unusable_window']} unusable, "
          f"{report['data_selection']['rows_excluded_null_feature']} null-feature, "
          f"both excluded from BOTH variants)")

    # ── 3 ────────────────────────────────────────────────────────────────────
    rule("3. TOP 10 MOST ANOMALOUS WINDOWS")
    print_top(bundle["results_a"], table, "MODEL A (with coverage features)")
    print_top(bundle["results_b"], table, "MODEL B (coverage features removed)")

    agreement = report["rank_agreement"]
    print(f"\n  top-10 overlap between variants : {agreement['top10_overlap']} of 10")
    print(f"  Spearman rank correlation       : "
          f"{agreement['spearman_rank_correlation']:+.4f}   "
          f"(descriptive agreement between two unsupervised rankings)")
    print(f"  mean |rank change|              : {agreement['mean_abs_rank_change']}   "
          f"max {agreement['max_abs_rank_change']}")

    # ── 4 ────────────────────────────────────────────────────────────────────
    rule("4. WINDOWS THAT CHANGED")
    for key, value in report["transitions"].items():
        print(f"  {key:<26} {value:>4}")

    changed = comparison[
        comparison["transition"].isin(
            ["flagged -> not flagged", "not flagged -> flagged"])
    ].merge(table[["caseid", "window_index"] + ablation.COVERAGE_FEATURES],
            on=["caseid", "window_index"], how="left")

    if changed.empty:
        print("\n  No window changed flag status.")
    else:
        changed["min_cov"] = changed[ablation.COVERAGE_FEATURES].min(axis=1)
        print(f"\n  {'case':>4}  {'win':>4}  {'time':>16}  {'score A':>9}  "
              f"{'score B':>9}  {'rank A':>7}  {'rank B':>7}  {'min cov':>8}  transition")
        for _, r in changed.sort_values("anomaly_score_a", ascending=False).iterrows():
            span = f"{r['window_start_s'] / 60:.0f}-{r['window_end_s'] / 60:.0f} min"
            print(f"  {int(r['caseid']):>4}  {int(r['window_index']):>4}  {span:>16}  "
                  f"{r['anomaly_score_a']:>+9.4f}  {r['anomaly_score_b']:>+9.4f}  "
                  f"{int(r['anomaly_rank_a']):>7}  {int(r['anomaly_rank_b']):>7}  "
                  f"{r['min_cov']:>8.2f}  {r['transition']}")

    # ── 5 ────────────────────────────────────────────────────────────────────
    rule("5. PER-CASE FLAGGED COUNTS")
    per_case = pd.DataFrame(report["per_case"])
    print(per_case.to_string(index=False))

    # ── 6 ────────────────────────────────────────────────────────────────────
    rule("6. CASE 4 CLUSTER (285-340 MIN)")
    cluster = report["case4_cluster"]
    print(f"  windows in range : {cluster['windows_in_range']}")
    print(f"  MODEL A flagged  : {cluster['flagged_model_a']}  "
          f"-> windows {cluster['window_indices_a']}")
    print(f"  MODEL B flagged  : {cluster['flagged_model_b']}  "
          f"-> windows {cluster['window_indices_b']}")
    survived = (set(cluster["window_indices_a"]) & set(cluster["window_indices_b"]))
    print(f"  survived in both : {len(survived)} -> {sorted(survived)}")

    # ── 7 ────────────────────────────────────────────────────────────────────
    rule("7. CASE 8 WINDOW 16")
    w16 = report["case8_window16"]
    if w16.get("present"):
        print(f"  MODEL A : score {w16['score_a']:+.4f}  rank {w16['rank_a']}  "
              f"flagged={w16['flagged_a']}")
        print(f"  MODEL B : score {w16['score_b']:+.4f}  rank {w16['rank_b']}  "
              f"flagged={w16['flagged_b']}")
        print(f"  outcome : {w16['transition']}")
    else:
        print("  window not present in the analyzed population")

    # ── 8 ────────────────────────────────────────────────────────────────────
    rule("8. SCORE COMPARISON FOR THE SAME WINDOWS")
    cols = ["caseid", "window_index", "anomaly_score_a", "anomaly_score_b",
            "score_change", "anomaly_rank_a", "anomaly_rank_b"]
    decreased = comparison[comparison["score_change"] < 0].nsmallest(
        20, "score_change")
    print(f"  Windows that became LESS unusual without the coverage features "
          f"(all {len(decreased)}):\n")
    print(decreased[cols].round(4).to_string(index=False))
    print(f"\n  Largest increases (of {int((comparison['score_change'] > 0).sum())} "
          f"windows that rose):\n")
    print(comparison.nlargest(8, "score_change")[cols].round(4).to_string(index=False))
    print(f"\n  mean score change {comparison['score_change'].mean():+.4f}   "
          f"sd {comparison['score_change'].std():.4f}   "
          f"range {comparison['score_change'].min():+.4f} to "
          f"{comparison['score_change'].max():+.4f}")
    print("  NOTE: absolute scores from two different feature spaces are not")
    print("  directly comparable; the ordering and the flag transitions are.")

    # ── 9 ────────────────────────────────────────────────────────────────────
    rule("9. EVIDENCE ON PHYSIOLOGICAL INTERPRETABILITY")
    profile = report["coverage_profile"]
    population = profile["analyzed_population"]
    print(f"  analyzed population : {population['n_analyzed']} windows, "
          f"{population['at_full_coverage']} at 100% coverage, "
          f"{population['below_100pct_coverage']} below")
    print(f"\n  {'':<10}{'flagged':>9}{'full cov':>10}{'<100% cov':>11}{'<90% cov':>10}")
    for name, key in (("MODEL A", "model_a"), ("MODEL B", "model_b")):
        entry = profile[key]
        print(f"  {name:<10}{entry['n_flagged']:>9}"
              f"{entry['flagged_with_full_coverage']:>10}"
              f"{entry['flagged_below_100pct_coverage']:>11}"
              f"{entry['flagged_below_90pct_coverage']:>10}")
    print("\n  Counted, not judged: a flag on a low-coverage window is not")
    print("  automatically wrong. It is a flag the coverage features could have")
    print("  produced on their own, which is exactly what this ablation isolates.")

    # ── Plot and files ───────────────────────────────────────────────────────
    rule("PLOT AND FILES")
    plot_path = ablation.plot_comparison(comparison, table)
    print(f"  {plot_path}")
    print(f"  {ablation.ABLATION_DIR / 'ablation_comparison.csv'}")
    print(f"  {ablation.ABLATION_DIR / 'anomaly_results_model_b.csv'}")
    print(f"  {report_path}")
    print("\n  MODEL A artifacts in results/model/ are untouched.")

    rule("NOTE")
    print("  No winner is selected here. Both variants are reported side by side;")
    print("  the choice is a judgement call about what the flags should mean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
