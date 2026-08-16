"""Descriptive class profile for the Model B baseline.

The unsupervised analogue of a classification report: same shape (classes,
support, per-class statistics) with descriptive numbers instead of supervised
performance measures, because this data carries no ground-truth anomaly label.

    python class_profile_checkpoint.py
"""

import sys

import pandas as pd

sys.path.insert(0, "src")

from vitaldb_audit import anomaly, class_profile, features  # noqa: E402

pd.set_option("display.width", 200)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    table = pd.read_csv(features.FEATURES_DIR / "feature_table_5min.csv")
    results = pd.read_csv(anomaly.MODEL_DIR / "anomaly_results.csv")
    _, selection = anomaly.select_model_rows(table)

    profile = class_profile.build_profile(results, table, selection)
    path = class_profile.write_profile(profile)

    rule("DESCRIPTIVE CLASS PROFILE — MODEL B")
    print(f"  {profile['model']}")
    print(f"\n  NO GROUND TRUTH. This is an unsupervised model, so there is nothing")
    print(f"  for a prediction to be correct or incorrect against. Precision,")
    print(f"  recall, F1, accuracy, AUROC, AUPRC and a confusion matrix are all")
    print(f"  undefined here and are not reported.")

    # ── Support ──────────────────────────────────────────────────────────────
    rule("CLASS SUPPORT")
    support = profile["support"]
    print(f"  {'class':<14}{'support':>9}{'pct':>9}   meaning")
    for entry in support["classes"]:
        print(f"  {entry['class']:<14}{entry['support']:>9}"
              f"{entry['pct_of_analyzed']:>8.1f}%   {entry['meaning']}")
    print(f"  {'-' * 74}")
    print(f"  {'analyzed':<14}{support['analyzed_total']:>9}")

    unclassified = support["not_classified"]
    print(f"\n  Never classified (not a third class, not errors):")
    print(f"    window not usable (<70% coverage) : "
          f"{unclassified['excluded_window_not_usable']:>3}")
    print(f"    null model feature                : "
          f"{unclassified['excluded_null_model_feature']:>3}")
    print(f"    {'-' * 40}")
    print(f"    feature table total               : "
          f"{unclassified['windows_in_feature_table']:>3}")
    print(f"\n  NOTE: {support['class_balance_note']}")

    # ── Score distribution ───────────────────────────────────────────────────
    rule("SCORE DISTRIBUTION BY CLASS")
    frame = pd.DataFrame(profile["score_distribution_by_class"])
    print(frame.to_string(index=False))

    # ── Per case ─────────────────────────────────────────────────────────────
    rule("PER-CASE COMPOSITION")
    per_case = pd.DataFrame(profile["per_case_composition"])
    print(per_case[["caseid", "analyzed", "flagged", "not_flagged",
                    "pct_flagged", "median_score", "max_score"]].to_string(index=False))

    # ── Separation ───────────────────────────────────────────────────────────
    rule("WHICH FEATURES SEPARATE THE CLASSES")
    print("  Cliff's delta: non-parametric effect size in [-1, +1].")
    print("  Descriptive separation only — NOT a performance measure and NOT")
    print("  feature importance.\n")
    print(f"  {'feature':<20}{'flagged':>10}{'not flagged':>13}"
          f"{'delta':>9}   separation")
    print(f"  {'-' * 72}")
    for row in profile["feature_separation"]["model_features"]:
        flagged = "  n/a" if row["median_flagged"] is None else f"{row['median_flagged']:.2f}"
        others = "  n/a" if row["median_not_flagged"] is None else f"{row['median_not_flagged']:.2f}"
        delta = row["cliffs_delta"]
        print(f"  {row['feature']:<20}{flagged:>10}{others:>13}"
              f"{delta:>+9.3f}   {row['separation']}")

    print(f"\n  Context — coverage columns, which are NOT Model B inputs:")
    for row in profile["feature_separation"]["context_features_not_model_inputs"]:
        flagged = f"{row['median_flagged']:.2f}"
        others = f"{row['median_not_flagged']:.2f}"
        print(f"  {row['feature']:<20}{flagged:>10}{others:>13}"
              f"{row['cliffs_delta']:>+9.3f}   {row['separation']}")

    # ── Evidence composition ─────────────────────────────────────────────────
    composition = profile["flagged_class_evidence_composition"]
    if composition:
        rule("FLAGGED CLASS — INTERPRETABLE EVIDENCE COMPOSITION")
        print(f"  mean shift only  : {composition['mean_shift_only']:>3}")
        print(f"  dispersion only  : {composition['dispersion_only']:>3}")
        print(f"  both             : {composition['both']:>3}")
        print(f"  neither          : {composition['neither']:>3}")
        print(f"  {'-' * 24}")
        print(f"  total            : {composition['flagged_windows_described']:>3}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    rule("FILES")
    plot_path = class_profile.plot_profile(results, table, profile)
    print(f"  {path}")
    print(f"  {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
