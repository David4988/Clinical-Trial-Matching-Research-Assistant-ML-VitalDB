"""Run the Isolation Forest baseline on the v1 feature table.

Fits the model, saves it with its predictions and scores, generates the
trajectory plots with flagged windows highlighted, and prints the experiment
summary.

A flagged window is a STATISTICALLY UNUSUAL MONITORING WINDOW.  It is not an
adverse event and not a clinical finding; there is no ground truth here.

    python model_checkpoint.py [--contamination 0.10] [--seed 20260816]
"""

import argparse
import sys

import joblib
import pandas as pd

sys.path.insert(0, "src")

from vitaldb_audit import anomaly, features  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contamination", type=float,
                        default=anomaly.DEFAULT_CONTAMINATION,
                        help="assumed proportion of unusual windows")
    parser.add_argument("--seed", type=int, default=anomaly.RANDOM_SEED)
    parser.add_argument("--n-estimators", type=int,
                        default=anomaly.DEFAULT_N_ESTIMATORS)
    args = parser.parse_args(argv)

    # ── Load ─────────────────────────────────────────────────────────────────
    rule("ISOLATION FOREST BASELINE — DATA SELECTION")

    table_path = features.FEATURES_DIR / "feature_table_5min.csv"
    table = pd.read_csv(table_path)
    print(f"feature table : {table_path.name}  ({len(table)} rows, "
          f"{len(table.columns)} columns) — left unmodified")

    selected, selection = anomaly.select_model_rows(table)
    print(f"\n  rows in feature table          : {selection['rows_in_feature_table']}")
    print(f"  window_usable == true          : {selection['rows_window_usable']}")
    print(f"  excluded, window not usable    : {selection['rows_excluded_unusable_window']}")
    print(f"  excluded, null model feature   : {selection['rows_excluded_null_feature']}")
    print(f"  ROWS ANALYZED                  : {selection['rows_analyzed']}")
    print(f"\n  imputation: {selection['imputation']}")
    if selection["excluded_null_windows"]:
        print("\n  windows excluded for a null feature "
              f"(nulls by column: {selection['null_counts_among_usable_rows']}):")
        for entry in selection["excluded_null_windows"]:
            print(f"    case {entry['caseid']:>2}  window {entry['window_index']:>2}"
                  f"  — first usable window for that signal, so no delta exists")

    # ── Fit ──────────────────────────────────────────────────────────────────
    rule("MODEL")
    print(f"  algorithm       : IsolationForest")
    print(f"  features        : {len(anomaly.MODEL_FEATURES)} "
          f"(HR / SpO2 / RR x mean, std, min, max, delta, coverage_pct)")
    print(f"  n_estimators    : {args.n_estimators}")
    print(f"  contamination   : {args.contamination}  (assumed flag rate, configurable)")
    print(f"  random_state    : {args.seed}  (fixed)")
    print(f"  standardization : none — Isolation Forest is invariant to")
    print(f"                    per-feature monotone rescaling")
    print(f"  excluded inputs : caseid, window_index, timestamps, usability")
    print(f"                    flags, ART_MBP, anything future-looking")

    model = anomaly.fit_isolation_forest(
        selected,
        contamination=args.contamination,
        random_state=args.seed,
        n_estimators=args.n_estimators,
    )
    results = anomaly.score_windows(model, selected)

    # ── Persist ──────────────────────────────────────────────────────────────
    anomaly.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = anomaly.MODEL_DIR / "isolation_forest_model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": anomaly.MODEL_FEATURES,
            "index_columns": anomaly.INDEX_COLUMNS,
            "contamination": args.contamination,
            "random_state": args.seed,
            "n_estimators": args.n_estimators,
            "trained_on_rows": int(len(selected)),
            "score_orientation": "anomaly_score = -decision_function; higher = more unusual",
        },
        model_path,
    )

    results_path = anomaly.MODEL_DIR / "anomaly_results.csv"
    results.to_csv(results_path, index=False)

    summary = anomaly.build_summary(
        selection, results, args.contamination, args.seed, args.n_estimators
    )
    summary_path = anomaly.write_summary(summary)

    print(f"\n  saved: {model_path.name}, {results_path.name}, {summary_path.name}")
    print(f"         -> {anomaly.MODEL_DIR}")

    # ── Results ──────────────────────────────────────────────────────────────
    rule("RESULTS")
    flagged = summary["flagged"]
    print(f"  windows analyzed : {flagged['windows_analyzed']}")
    print(f"  windows flagged  : {flagged['windows_flagged']}")
    print(f"  percentage       : {flagged['pct_flagged']}%")
    print("\n  per case:")
    per_case = pd.DataFrame(flagged["per_case"])
    print(per_case.to_string(index=False))

    dist = summary["score_distribution"]
    print(f"\n  anomaly-score distribution (higher = more unusual):")
    print(f"    mean {dist['mean']:+.4f}   std {dist['std']:.4f}")
    quantile_line = "   ".join(
        f"{k} {v:+.4f}" for k, v in dist["quantiles"].items()
    )
    print(f"    {quantile_line}")

    # ── Top 10 ───────────────────────────────────────────────────────────────
    rule("TOP 10 MOST ANOMALOUS WINDOWS")
    print("  (statistically unusual monitoring windows — NOT clinical findings)\n")
    header = (f"  {'rank':>4}  {'case':>4}  {'window':>6}  {'time range':>18}  "
              f"{'score':>9}  {'flagged':>7}")
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for entry in summary["extremes"]["most_unusual"]:
        span = f"{entry['window_start_min']:.1f}-{entry['window_end_min']:.1f} min"
        print(f"  {summary['extremes']['most_unusual'].index(entry) + 1:>4}  "
              f"{entry['caseid']:>4}  {entry['window_index']:>6}  {span:>18}  "
              f"{entry['anomaly_score']:>+9.4f}  {'yes' if entry['anomaly_label'] else 'no':>7}")

    rule("TOP 10 — FEATURE VALUES BEHIND EACH WINDOW")
    print("  (connecting each flagged window back to the feature table)\n")
    detail = anomaly.attach_features(results.head(10), table)
    show = ["caseid", "window_index", "anomaly_score",
            "hr_mean", "hr_std", "hr_delta", "hr_coverage_pct",
            "spo2_mean", "spo2_delta", "spo2_coverage_pct",
            "rr_mean", "rr_delta", "rr_coverage_pct"]
    print(detail[show].round(3).to_string(index=False))

    rule("10 LEAST ANOMALOUS WINDOWS")
    print(f"  {'case':>4}  {'window':>6}  {'time range':>18}  {'score':>9}")
    for entry in summary["extremes"]["least_unusual"]:
        span = f"{entry['window_start_min']:.1f}-{entry['window_end_min']:.1f} min"
        print(f"  {entry['caseid']:>4}  {entry['window_index']:>6}  {span:>18}  "
              f"{entry['anomaly_score']:>+9.4f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    rule("PLOTS")
    paths = []
    for caseid in sorted(table["caseid"].unique()):
        paths.append(anomaly.plot_case_with_flags(int(caseid), table, results))
    paths.append(anomaly.plot_score_distribution(results))
    paths.append(anomaly.plot_top_windows(results, table, n=10))
    for path in paths:
        print(f"  {path}")
    print("\n  Each case plot shows HR, SpO2 and RR trajectories plus a coverage")
    print("  panel.  Red = flagged window.  Grey = window not usable (excluded).")

    # ── Framing ──────────────────────────────────────────────────────────────
    rule("INTERPRETATION")
    print("  " + summary["interpretation"].replace(". ", ".\n  "))
    print(f"\n  Ground truth: {summary['ground_truth']}")
    print("\n  Caveats:")
    for caveat in summary["caveats"]:
        print(f"    - {caveat}")

    rule("NEXT")
    print("  Baseline is running end-to-end: features -> model -> scored windows.")
    print("  Stopping here as scoped — no XAI layer, no frontend, no second model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
