"""One controlled ablation: does dropping the coverage features change the flags?

MODEL A  the current 18-feature set.
MODEL B  the same, minus hr_coverage_pct, spo2_coverage_pct, rr_coverage_pct.

Everything else is held identical by construction: the same selected rows, the
same seed, the same contamination, the same tree count.  The row selection is
computed ONCE from the full feature set and handed to both variants, so the two
models cannot differ through a different training population — which would make
the comparison meaningless without being visible.

There is no ground truth here.  Nothing in this module computes accuracy,
precision, recall, F1, AUROC or AUPRC, because none of them are defined for
this experiment.  The comparison is descriptive: which windows moved, by how
much, and in which direction.
"""

import json
import logging

import numpy as np
import pandas as pd

from vitaldb_audit import anomaly, config

logger = logging.getLogger("vitaldb_audit.ablation")

# The only difference between the two variants.
COVERAGE_FEATURES: list[str] = [
    "hr_coverage_pct", "spo2_coverage_pct", "rr_coverage_pct",
]

MODEL_A_FEATURES: list[str] = list(anomaly.MODEL_FEATURES)
MODEL_B_FEATURES: list[str] = [
    c for c in anomaly.MODEL_FEATURES if c not in COVERAGE_FEATURES
]

ABLATION_DIR = config.RESULTS_DIR / "ablation"
PLOT_DIR = config.RESULTS_DIR / "plots" / "ablation"

# The cluster and the single window called out in the previous checkpoint, kept
# here as named landmarks so the report answers the same questions every run.
CASE4_CLUSTER = {"caseid": 4, "start_min": 285.0, "end_min": 340.0}
CASE8_WINDOW = {"caseid": 8, "window_index": 16}


# ── Running the two variants ─────────────────────────────────────────────────


def run_variant(
    selected: pd.DataFrame,
    feature_columns: list[str],
    contamination: float = anomaly.DEFAULT_CONTAMINATION,
    random_state: int = anomaly.RANDOM_SEED,
    n_estimators: int = anomaly.DEFAULT_N_ESTIMATORS,
):
    """Fit and score one variant on ALREADY-SELECTED rows."""
    model = anomaly.fit_isolation_forest(
        selected,
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        feature_columns=feature_columns,
    )
    results = anomaly.score_windows(model, selected, feature_columns=feature_columns)
    return model, results


def run_ablation(
    table: pd.DataFrame,
    contamination: float = anomaly.DEFAULT_CONTAMINATION,
    random_state: int = anomaly.RANDOM_SEED,
    n_estimators: int = anomaly.DEFAULT_N_ESTIMATORS,
) -> dict:
    """Run both variants over one identical selected population."""
    selected, selection = anomaly.select_model_rows(table)

    model_a, results_a = run_variant(
        selected, MODEL_A_FEATURES, contamination, random_state, n_estimators)
    model_b, results_b = run_variant(
        selected, MODEL_B_FEATURES, contamination, random_state, n_estimators)

    # The guarantee this experiment rests on, checked rather than asserted.
    if len(results_a) != len(results_b) != len(selected):
        raise AssertionError("variants scored different numbers of windows")
    keys_a = set(map(tuple, results_a[["caseid", "window_index"]].to_numpy()))
    keys_b = set(map(tuple, results_b[["caseid", "window_index"]].to_numpy()))
    if keys_a != keys_b:
        raise AssertionError("variants were scored on different windows")

    return {
        "selected": selected,
        "selection": selection,
        "model_a": model_a,
        "model_b": model_b,
        "results_a": results_a,
        "results_b": results_b,
        "config": {
            "contamination": contamination,
            "random_state": random_state,
            "n_estimators": n_estimators,
            "model_a_features": MODEL_A_FEATURES,
            "model_b_features": MODEL_B_FEATURES,
            "removed_features": COVERAGE_FEATURES,
            "held_identical": [
                "selected rows", "seed", "contamination", "n_estimators",
                "feature values", "no standardization", "no imputation",
            ],
        },
    }


# ── Comparison ───────────────────────────────────────────────────────────────


def compare(results_a: pd.DataFrame, results_b: pd.DataFrame) -> pd.DataFrame:
    """Join the two runs window-by-window and classify what happened to each."""
    keys = ["caseid", "window_index", "window_start_s", "window_end_s"]
    left = results_a[keys + ["anomaly_score", "anomaly_rank", "anomaly_label"]]
    right = results_b[keys + ["anomaly_score", "anomaly_rank", "anomaly_label"]]

    merged = left.merge(right, on=keys, suffixes=("_a", "_b"))
    merged["score_change"] = (
        merged["anomaly_score_b"] - merged["anomaly_score_a"]).round(6)
    merged["rank_change"] = merged["anomaly_rank_b"] - merged["anomaly_rank_a"]

    def classify(row):
        was, now = bool(row["anomaly_label_a"]), bool(row["anomaly_label_b"])
        if was and now:
            return "flagged in both"
        if was and not now:
            return "flagged -> not flagged"
        if not was and now:
            return "not flagged -> flagged"
        return "flagged in neither"

    merged["transition"] = merged.apply(classify, axis=1)
    return merged.sort_values("anomaly_score_a", ascending=False).reset_index(drop=True)


def transition_counts(comparison: pd.DataFrame) -> dict:
    order = ["flagged in both", "flagged -> not flagged",
             "not flagged -> flagged", "flagged in neither"]
    counts = comparison["transition"].value_counts().to_dict()
    return {key: int(counts.get(key, 0)) for key in order}


def rank_agreement(comparison: pd.DataFrame) -> dict:
    """How similarly the two variants ORDER the windows.

    Absolute scores from two different feature spaces are not directly
    comparable, so the ordering is the meaningful comparison.  Spearman here is
    a descriptive agreement statistic between two unsupervised rankings, not a
    performance metric — there is nothing to be right about.
    """
    spearman = float(
        comparison["anomaly_rank_a"].corr(comparison["anomaly_rank_b"], method="spearman")
    )
    def top10_keys(column):
        top = comparison.nsmallest(10, column)
        return set(map(tuple, top[["caseid", "window_index"]].to_numpy()))

    top10_a = top10_keys("anomaly_rank_a")
    top10_b = top10_keys("anomaly_rank_b")
    return {
        "spearman_rank_correlation": round(spearman, 4),
        "top10_overlap": len(top10_a & top10_b),
        "mean_abs_rank_change": round(float(comparison["rank_change"].abs().mean()), 2),
        "max_abs_rank_change": int(comparison["rank_change"].abs().max()),
    }


def top_windows(results: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    out = results.head(n)[
        ["caseid", "window_index", "window_start_s", "window_end_s",
         "anomaly_score", "anomaly_label"]
    ].copy()
    out["time_range"] = (
        (out["window_start_s"] / 60).round(1).astype(str) + "-"
        + (out["window_end_s"] / 60).round(1).astype(str) + " min"
    )
    return out


def per_case_counts(results_a: pd.DataFrame, results_b: pd.DataFrame) -> pd.DataFrame:
    def counts(results, suffix):
        return (
            results.groupby("caseid")
            .agg(**{
                f"analyzed": ("anomaly_label", "size"),
                f"flagged_{suffix}": ("anomaly_label", "sum"),
            })
        )

    a = counts(results_a, "a")
    b = counts(results_b, "b").drop(columns=["analyzed"])
    out = a.join(b).reset_index()
    out["change"] = out["flagged_b"] - out["flagged_a"]
    return out


# ── Landmark questions from the previous checkpoint ──────────────────────────


def cluster_status(comparison: pd.DataFrame, landmark: dict = None) -> dict:
    """Does case 4's 285-340 minute cluster survive the ablation?"""
    landmark = landmark or CASE4_CLUSTER
    window = comparison[
        (comparison["caseid"] == landmark["caseid"])
        & (comparison["window_start_s"] >= landmark["start_min"] * 60)
        & (comparison["window_end_s"] <= landmark["end_min"] * 60)
    ]
    return {
        "caseid": landmark["caseid"],
        "range_min": [landmark["start_min"], landmark["end_min"]],
        "windows_in_range": int(len(window)),
        "flagged_model_a": int(window["anomaly_label_a"].sum()),
        "flagged_model_b": int(window["anomaly_label_b"].sum()),
        "window_indices_a": window[window["anomaly_label_a"] == 1]["window_index"]
                                  .astype(int).tolist(),
        "window_indices_b": window[window["anomaly_label_b"] == 1]["window_index"]
                                  .astype(int).tolist(),
    }


def window_status(comparison: pd.DataFrame, landmark: dict = None) -> dict:
    """Does case 8 window 16 stay unusual once coverage is removed?"""
    landmark = landmark or CASE8_WINDOW
    row = comparison[
        (comparison["caseid"] == landmark["caseid"])
        & (comparison["window_index"] == landmark["window_index"])
    ]
    if row.empty:
        return {**landmark, "present": False}
    r = row.iloc[0]
    return {
        "caseid": int(r["caseid"]),
        "window_index": int(r["window_index"]),
        "present": True,
        "score_a": round(float(r["anomaly_score_a"]), 6),
        "score_b": round(float(r["anomaly_score_b"]), 6),
        "rank_a": int(r["anomaly_rank_a"]),
        "rank_b": int(r["anomaly_rank_b"]),
        "flagged_a": bool(r["anomaly_label_a"]),
        "flagged_b": bool(r["anomaly_label_b"]),
        "transition": r["transition"],
    }


def coverage_profile(comparison: pd.DataFrame, table: pd.DataFrame) -> dict:
    """How many flags in each variant sit on a window with imperfect coverage.

    This is the evidence for the interpretability question.  It counts, it does
    not judge: a flag on a low-coverage window is not automatically wrong, it is
    just a flag the coverage features could have produced on their own.
    """
    merged = comparison.merge(
        table[["caseid", "window_index"] + COVERAGE_FEATURES],
        on=["caseid", "window_index"], how="left",
    )
    merged["min_coverage_pct"] = merged[COVERAGE_FEATURES].min(axis=1)

    def profile(label_column):
        flagged = merged[merged[label_column] == 1]
        return {
            "n_flagged": int(len(flagged)),
            "flagged_with_full_coverage": int((flagged["min_coverage_pct"] >= 100).sum()),
            "flagged_below_100pct_coverage": int((flagged["min_coverage_pct"] < 100).sum()),
            "flagged_below_90pct_coverage": int((flagged["min_coverage_pct"] < 90).sum()),
            "min_coverage_of_flagged": round(float(flagged["min_coverage_pct"].min()), 2)
            if len(flagged) else None,
        }

    population = {
        "n_analyzed": int(len(merged)),
        "at_full_coverage": int((merged["min_coverage_pct"] >= 100).sum()),
        "below_100pct_coverage": int((merged["min_coverage_pct"] < 100).sum()),
    }
    return {
        "analyzed_population": population,
        "model_a": profile("anomaly_label_a"),
        "model_b": profile("anomaly_label_b"),
    }


def build_report(bundle: dict, table: pd.DataFrame) -> dict:
    comparison = compare(bundle["results_a"], bundle["results_b"])
    return {
        "experiment": "coverage_feature_ablation",
        "question": (
            "Do the anomaly flags change when the three coverage features are "
            "removed, holding everything else identical?"
        ),
        "interpretation": (
            "Both variants rank windows by how unusual their feature pattern is. "
            "A flagged window is a statistically unusual monitoring window, NOT "
            "an adverse event and NOT a clinical finding. No ground truth exists "
            "here, so no supervised performance measure is reported."
        ),
        "config": {k: v for k, v in bundle["config"].items()},
        "data_selection": bundle["selection"],
        "model_a": {
            "n_features": len(MODEL_A_FEATURES),
            "windows_analyzed": int(len(bundle["results_a"])),
            "windows_flagged": int(bundle["results_a"]["anomaly_label"].sum()),
        },
        "model_b": {
            "n_features": len(MODEL_B_FEATURES),
            "windows_analyzed": int(len(bundle["results_b"])),
            "windows_flagged": int(bundle["results_b"]["anomaly_label"].sum()),
        },
        "transitions": transition_counts(comparison),
        "rank_agreement": rank_agreement(comparison),
        "per_case": per_case_counts(
            bundle["results_a"], bundle["results_b"]).astype(object).to_dict("records"),
        "case4_cluster": cluster_status(comparison),
        "case8_window16": window_status(comparison),
        "coverage_profile": coverage_profile(comparison, table),
        "top10_model_a": top_windows(bundle["results_a"]).astype(object).to_dict("records"),
        "top10_model_b": top_windows(bundle["results_b"]).astype(object).to_dict("records"),
    }


def write_report(report: dict, path=None):
    path = path or (ABLATION_DIR / "ablation_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


# ── Plot ─────────────────────────────────────────────────────────────────────


def plot_comparison(comparison: pd.DataFrame, table: pd.DataFrame, out_dir=None):
    """Score-vs-score scatter plus the per-case flag counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or PLOT_DIR
    merged = comparison.merge(
        table[["caseid", "window_index"] + COVERAGE_FEATURES],
        on=["caseid", "window_index"], how="left",
    )
    merged["min_coverage_pct"] = merged[COVERAGE_FEATURES].min(axis=1)

    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(14.5, 6.2), gridspec_kw={"width_ratios": [1.55, 1]}
    )

    styles = {
        "flagged in both": ("#d62728", "o", 46),
        "flagged -> not flagged": ("#ff7f0e", "v", 74),
        "not flagged -> flagged": ("#1f77b4", "^", 74),
        "flagged in neither": ("#bbbbbb", ".", 22),
    }
    for transition, (colour, marker, size) in styles.items():
        subset = merged[merged["transition"] == transition]
        if subset.empty:
            continue
        ax.scatter(subset["anomaly_score_a"], subset["anomaly_score_b"],
                   c=colour, marker=marker, s=size, alpha=0.85,
                   label=f"{transition} (n={len(subset)})", zorder=3)

    # Ring the windows that did not have perfect coverage: if the ablation is
    # doing what we think, these are the ones that move.
    partial = merged[merged["min_coverage_pct"] < 100]
    if not partial.empty:
        ax.scatter(partial["anomaly_score_a"], partial["anomaly_score_b"],
                   facecolors="none", edgecolors="#000000", s=150, linewidths=1.1,
                   label=f"coverage < 100% (n={len(partial)})", zorder=4)

    limits = [
        min(merged["anomaly_score_a"].min(), merged["anomaly_score_b"].min()) - 0.02,
        max(merged["anomaly_score_a"].max(), merged["anomaly_score_b"].max()) + 0.02,
    ]
    ax.plot(limits, limits, color="#333333", linestyle="--", linewidth=0.9,
            label="unchanged score", zorder=1)
    ax.axhline(0, color="#999999", linewidth=0.7, zorder=1)
    ax.axvline(0, color="#999999", linewidth=0.7, zorder=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("MODEL A anomaly score  (18 features, with coverage)")
    ax.set_ylabel("MODEL B anomaly score  (15 features, coverage removed)")
    ax.set_title("Same windows, both variants\n"
                 "higher = more unusual; above/right of 0 = flagged", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)

    counts = per_case_counts(
        comparison.rename(columns={"anomaly_label_a": "anomaly_label"}),
        comparison.rename(columns={"anomaly_label_b": "anomaly_label"}),
    )
    positions = np.arange(len(counts))
    width = 0.38
    ax_bar.bar(positions - width / 2, counts["flagged_a"], width,
               color="#d62728", alpha=0.85, label="MODEL A")
    ax_bar.bar(positions + width / 2, counts["flagged_b"], width,
               color="#1f77b4", alpha=0.85, label="MODEL B")
    for pos, row in zip(positions, counts.itertuples()):
        ax_bar.text(pos - width / 2, row.flagged_a + 0.12, str(int(row.flagged_a)),
                    ha="center", fontsize=8)
        ax_bar.text(pos + width / 2, row.flagged_b + 0.12, str(int(row.flagged_b)),
                    ha="center", fontsize=8)
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels([f"case {int(c)}\n({int(n)} windows)"
                            for c, n in zip(counts["caseid"], counts["analyzed"])],
                           fontsize=8)
    ax_bar.set_ylabel("windows flagged")
    ax_bar.set_title("Flagged windows per case", fontsize=10)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "coverage_ablation_comparison.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
