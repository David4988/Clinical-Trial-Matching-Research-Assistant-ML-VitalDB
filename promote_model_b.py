"""Promote Model B to results/model/ as the canonical baseline.

Model B (15 physiological features, coverage features removed) won the coverage
ablation and is the prototype baseline.  This script makes results/model/ hold
Model B rather than Model A.

What it does NOT do: recompute, re-rank or otherwise alter Model B's results.
The promoted CSV is copied byte-for-byte from the ablation output and verified
identical afterwards.  The fitted estimator is refitted only because the
ablation never serialised one, and it is refit on the same rows with the same
seed and parameters, then checked to reproduce the copied scores exactly.

Model A is not deleted — it is the control the ablation was measured against, so
it moves to results/model/model_a_control/ instead of being overwritten.
The ablation artifacts in results/ablation/ are left untouched.

    python promote_model_b.py
"""

import filecmp
import json
import shutil
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from vitaldb_audit import ablation, anomaly, features  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    rule("PROMOTE MODEL B TO results/model/")

    model_dir = anomaly.MODEL_DIR
    control_dir = model_dir / "model_a_control"
    source_csv = ablation.ABLATION_DIR / "anomaly_results_model_b.csv"

    # ── Preserve Model A ─────────────────────────────────────────────────────
    control_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for name in ("anomaly_results.csv", "experiment_summary.json",
                 "isolation_forest_model.joblib"):
        current = model_dir / name
        if current.exists():
            shutil.move(str(current), str(control_dir / name))
            moved.append(name)
    print(f"  Model A preserved as the ablation control:")
    for name in moved:
        print(f"    -> {control_dir.relative_to(anomaly.config.PROJECT_ROOT)}/{name}")
    if not moved:
        print("    (nothing to move; already promoted)")

    # ── Copy Model B's results verbatim ──────────────────────────────────────
    target_csv = model_dir / "anomaly_results.csv"
    shutil.copy2(source_csv, target_csv)
    identical = filecmp.cmp(source_csv, target_csv, shallow=False)
    print(f"\n  results copied verbatim from "
          f"{source_csv.relative_to(anomaly.config.PROJECT_ROOT)}")
    print(f"    byte-identical to source: {identical}")
    if not identical:
        raise AssertionError("promoted results differ from the ablation output")

    # ── Refit the estimator on the same rows, seed and parameters ────────────
    table = pd.read_csv(features.FEATURES_DIR / "feature_table_5min.csv")
    selected, selection = anomaly.select_model_rows(table)
    model, results = ablation.run_variant(selected, ablation.MODEL_B_FEATURES)

    promoted = pd.read_csv(target_csv)
    merged = promoted.merge(
        results[["caseid", "window_index", "anomaly_score", "anomaly_label"]],
        on=["caseid", "window_index"], suffixes=("_file", "_refit"),
    )
    scores_match = bool(np.allclose(
        merged["anomaly_score_file"], merged["anomaly_score_refit"], atol=1e-9))
    labels_match = bool(
        (merged["anomaly_label_file"] == merged["anomaly_label_refit"]).all())
    print(f"\n  refit reproduces the promoted results:")
    print(f"    rows matched : {len(merged)} of {len(promoted)}")
    print(f"    scores equal : {scores_match}")
    print(f"    labels equal : {labels_match}")
    if not (scores_match and labels_match and len(merged) == len(promoted)):
        raise AssertionError("refit does not reproduce the promoted Model B results")

    model_path = model_dir / "isolation_forest_model.joblib"
    joblib.dump(
        {
            "model": model,
            "variant": "model_b",
            "feature_columns": ablation.MODEL_B_FEATURES,
            "index_columns": anomaly.INDEX_COLUMNS,
            "excluded_features": ablation.COVERAGE_FEATURES,
            "contamination": anomaly.DEFAULT_CONTAMINATION,
            "random_state": anomaly.RANDOM_SEED,
            "n_estimators": anomaly.DEFAULT_N_ESTIMATORS,
            "trained_on_rows": int(len(selected)),
            "score_orientation":
                "anomaly_score = -decision_function; higher = more unusual",
        },
        model_path,
    )

    # ── Canonical experiment summary ─────────────────────────────────────────
    summary = anomaly.build_summary(
        selection, results, anomaly.DEFAULT_CONTAMINATION,
        anomaly.RANDOM_SEED, anomaly.DEFAULT_N_ESTIMATORS,
    )
    summary["experiment"] = "isolation_forest_baseline_model_b"
    summary["variant"] = {
        "name": "Model B",
        "canonical": True,
        "n_features": len(ablation.MODEL_B_FEATURES),
        "features": ablation.MODEL_B_FEATURES,
        "excluded_features": ablation.COVERAGE_FEATURES,
        "selected_by": (
            "coverage-feature ablation; see results/ablation/ablation_summary.json"
        ),
        "control": "Model A (18 features) preserved in results/model/model_a_control/",
    }
    summary["features"]["n_features"] = len(ablation.MODEL_B_FEATURES)
    summary["features"]["columns"] = ablation.MODEL_B_FEATURES
    summary["features"]["excluded_by_design"] = (
        summary["features"]["excluded_by_design"] + ablation.COVERAGE_FEATURES
    )
    summary_path = model_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ── Report ───────────────────────────────────────────────────────────────
    rule("CANONICAL BASELINE")
    print(f"  {model_dir.relative_to(anomaly.config.PROJECT_ROOT)}/")
    for path in sorted(model_dir.glob("*")):
        if path.is_file():
            print(f"    {path.name:<34} {path.stat().st_size / 1024:>8.1f} KB")
    print(f"    model_a_control/                   (Model A, the ablation control)")

    print(f"\n  variant       : Model B, {len(ablation.MODEL_B_FEATURES)} features")
    print(f"  excluded      : {', '.join(ablation.COVERAGE_FEATURES)}")
    print(f"  analyzed      : {len(promoted)} windows")
    print(f"  flagged       : {int(promoted['anomaly_label'].sum())}")
    print(f"\n  ablation artifacts left untouched in "
          f"{ablation.ABLATION_DIR.relative_to(anomaly.config.PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
