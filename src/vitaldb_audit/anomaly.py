"""Isolation Forest baseline over the v1 feature table.

What this is
------------
An UNSUPERVISED baseline that ranks 5-minute monitoring windows by how unusual
their feature pattern is relative to the other windows in this dataset.

What this is NOT
----------------
There is no ground-truth anomaly label in this data, so nothing here is
validated against outcomes.  A flagged window is a STATISTICALLY UNUSUAL
MONITORING WINDOW — an unusual physiological feature pattern.  It is not an
adverse event, not deterioration, and not a clinical finding.  Accuracy,
precision, recall, F1, AUROC and AUPRC are undefined here and are deliberately
not computed anywhere in this module.

Data-handling rules carried through from the earlier stages
-----------------------------------------------------------
NO IMPUTATION
    Rows carrying a null in any model feature are EXCLUDED from the fit and
    reported, never filled.  A null delta means "no earlier usable window to
    compare against", which is not a zero.

NO IDENTIFIERS AS PREDICTORS
    caseid, window_index and the window timestamps are carried alongside the
    matrix for traceability but are never columns of it, so the model cannot
    learn "case 4 is unusual" or "late windows are unusual".

NO USABILITY FLAGS AS PREDICTORS
    ``window_usable`` selects the rows; it is not a feature.  Coverage
    percentages ARE features, because how much of a window was observed is a
    genuine property of the monitoring, but see the caveat in the summary.

NO STANDARDIZATION
    Isolation Forest splits on one feature at a time at randomly drawn
    thresholds, so it is invariant to per-feature monotone rescaling.
    Standardizing would add a fitted artifact to carry around for no gain.
"""

import json
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from vitaldb_audit import config

logger = logging.getLogger("vitaldb_audit.anomaly")

# ── Model configuration ──────────────────────────────────────────────────────

# The 18 compact monitoring features.  Order is fixed and stored with the model
# so that scoring can never silently re-order columns.
MODEL_FEATURES: list[str] = [
    f"{signal}_{stat}"
    for signal in ("hr", "spo2", "rr")
    for stat in ("mean", "std", "min", "max", "delta", "coverage_pct")
]

# Columns kept beside the matrix purely so a result can be traced back to a
# window.  Never passed to the model.
INDEX_COLUMNS: list[str] = [
    "caseid", "window_index", "window_start_s", "window_end_s",
]

RANDOM_SEED = config.RANDOM_SEED
DEFAULT_CONTAMINATION = 0.10
DEFAULT_N_ESTIMATORS = 200

MODEL_DIR = config.RESULTS_DIR / "model"
PLOT_DIR = config.RESULTS_DIR / "plots" / "anomaly"


# ── Row selection ────────────────────────────────────────────────────────────


def select_model_rows(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pick the rows this experiment can honestly model.

    Two filters, both reported rather than applied silently:
      1. ``window_usable`` must be true — all three core signals cleared the
         coverage bar.
      2. No null in any model feature.  Excluded, never imputed.

    Returns the selected frame (index columns + features) and a report.
    """
    missing = set(MODEL_FEATURES + INDEX_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"feature table is missing columns: {sorted(missing)}")

    total = len(table)
    usable = table[table["window_usable"].fillna(False).astype(bool)].copy()

    null_mask = usable[MODEL_FEATURES].isna().any(axis=1)
    excluded = usable[null_mask]
    selected = usable[~null_mask].reset_index(drop=True)

    null_by_column = (
        usable[MODEL_FEATURES].isna().sum().loc[lambda s: s > 0].astype(int).to_dict()
    )

    report = {
        "rows_in_feature_table": int(total),
        "rows_window_usable": int(len(usable)),
        "rows_excluded_unusable_window": int(total - len(usable)),
        "rows_excluded_null_feature": int(len(excluded)),
        "rows_analyzed": int(len(selected)),
        "null_counts_among_usable_rows": null_by_column,
        "excluded_null_windows": excluded[INDEX_COLUMNS].astype(int).to_dict("records"),
        "imputation": "none — rows with a null feature are excluded, not filled",
    }
    logger.info(
        "selected %d of %d rows (%d unusable, %d null-feature)",
        len(selected), total, total - len(usable), len(excluded),
    )
    return selected[INDEX_COLUMNS + MODEL_FEATURES], report


# ── Fit and score ────────────────────────────────────────────────────────────


def fit_isolation_forest(
    selected: pd.DataFrame,
    contamination: float = DEFAULT_CONTAMINATION,
    random_state: int = RANDOM_SEED,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    feature_columns: list[str] | None = None,
) -> IsolationForest:
    """Fit on the model features only.  Seed is fixed for reproducibility.

    ``feature_columns`` defaults to the full v1 set.  It exists so an ablation
    can drop a subset while holding every other input identical; the rows are
    chosen by the caller, never re-selected here.
    """
    feature_columns = list(feature_columns or MODEL_FEATURES)
    matrix = selected[feature_columns]
    if matrix.isna().any().any():
        raise ValueError("model matrix still contains nulls; selection failed")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        max_samples="auto",
        bootstrap=False,
        n_jobs=1,
    )
    model.fit(matrix.to_numpy(dtype=float))
    logger.info(
        "fitted IsolationForest(n=%d, contamination=%s, seed=%d) on %d x %d",
        n_estimators, contamination, random_state, len(matrix), len(feature_columns),
    )
    return model


def score_windows(
    model: IsolationForest,
    selected: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Score every selected window and attach it back to its identifiers.

    ``anomaly_score`` is oriented so that HIGHER MEANS MORE UNUSUAL.  It is the
    negated sklearn decision function, which is the opposite orientation, and
    getting that backwards would invert the entire result — so the raw
    decision-function value is kept alongside it for auditing.
    """
    feature_columns = list(feature_columns or MODEL_FEATURES)
    matrix = selected[feature_columns].to_numpy(dtype=float)

    decision = model.decision_function(matrix)
    predicted = model.predict(matrix)

    results = selected[INDEX_COLUMNS].copy()
    results["anomaly_score"] = np.round(-decision, 6)
    results["iforest_decision_function"] = np.round(decision, 6)
    results["anomaly_label"] = (predicted == -1).astype(int)

    results = results.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    results.insert(len(INDEX_COLUMNS), "anomaly_rank", np.arange(1, len(results) + 1))
    return results


def attach_features(results: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Join the feature values back onto the results, for inspection only."""
    keep = INDEX_COLUMNS[:2] + MODEL_FEATURES
    return results.merge(table[keep], on=INDEX_COLUMNS[:2], how="left")


# ── Reporting ────────────────────────────────────────────────────────────────


def score_distribution(results: pd.DataFrame) -> dict:
    """Descriptive statistics of the score, with no thresholded metrics."""
    scores = results["anomaly_score"]
    quantiles = [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]
    return {
        "count": int(len(scores)),
        "mean": round(float(scores.mean()), 6),
        "std": round(float(scores.std(ddof=1)), 6) if len(scores) > 1 else None,
        "quantiles": {
            f"q{int(q * 100):02d}": round(float(scores.quantile(q)), 6)
            for q in quantiles
        },
    }


def flagged_summary(results: pd.DataFrame) -> dict:
    """How many windows were flagged, overall and per case."""
    flagged = int(results["anomaly_label"].sum())
    analyzed = int(len(results))
    per_case = (
        results.groupby("caseid")
        .agg(windows_analyzed=("anomaly_label", "size"),
             windows_flagged=("anomaly_label", "sum"),
             max_score=("anomaly_score", "max"))
        .reset_index()
    )
    per_case["pct_flagged"] = (
        per_case["windows_flagged"] / per_case["windows_analyzed"] * 100
    ).round(1)
    per_case["max_score"] = per_case["max_score"].round(6)
    return {
        "windows_analyzed": analyzed,
        "windows_flagged": flagged,
        "pct_flagged": round(flagged / analyzed * 100, 2) if analyzed else 0.0,
        "per_case": per_case.astype(object).to_dict("records"),
    }


def extreme_windows(results: pd.DataFrame, n: int = 10) -> dict:
    """The n most and n least unusual windows, with their time ranges."""
    def rows(frame):
        out = []
        for _, r in frame.iterrows():
            out.append({
                "caseid": int(r["caseid"]),
                "window_index": int(r["window_index"]),
                "window_start_min": round(float(r["window_start_s"]) / 60.0, 2),
                "window_end_min": round(float(r["window_end_s"]) / 60.0, 2),
                "anomaly_score": round(float(r["anomaly_score"]), 6),
                "anomaly_label": int(r["anomaly_label"]),
            })
        return out

    return {
        "most_unusual": rows(results.head(n)),
        "least_unusual": rows(results.tail(n).iloc[::-1]),
    }


def build_summary(
    selection_report: dict,
    results: pd.DataFrame,
    contamination: float,
    random_state: int,
    n_estimators: int,
) -> dict:
    """The experiment record written to experiment_summary.json."""
    return {
        "experiment": "isolation_forest_baseline_v1",
        "interpretation": (
            "Scores rank windows by how UNUSUAL their feature pattern is "
            "relative to the other windows in this dataset. A flagged window is "
            "a statistically unusual monitoring window, NOT an adverse event, "
            "NOT clinical deterioration, and NOT a clinically validated finding."
        ),
        "ground_truth": (
            "none available; accuracy/precision/recall/F1/AUROC/AUPRC are "
            "undefined for this experiment and are not reported"
        ),
        "model": {
            "algorithm": "sklearn.ensemble.IsolationForest",
            "n_estimators": n_estimators,
            "contamination": contamination,
            "random_state": random_state,
            "max_samples": "auto",
            "standardization": (
                "none — Isolation Forest is invariant to per-feature monotone "
                "rescaling"
            ),
        },
        "features": {
            "n_features": len(MODEL_FEATURES),
            "columns": MODEL_FEATURES,
            "excluded_by_design": [
                "caseid", "window_index", "window_start_s", "window_end_s",
                "hr_usable", "spo2_usable", "rr_usable", "n_core_usable",
                "window_usable", "consecutive_usable_windows",
                "ART_MBP (all columns)",
            ],
        },
        "data_selection": selection_report,
        "score_orientation": "higher anomaly_score = more unusual",
        "score_distribution": score_distribution(results),
        "flagged": flagged_summary(results),
        "extremes": extreme_windows(results, n=10),
        "caveats": [
            "Contamination is an assumed flag rate, not a discovered one; it "
            "sets how many windows are labelled, so the flagged count is a "
            "parameter choice rather than a measurement.",
            "Coverage percentages are model features, so a window can be "
            "flagged for unusual data quality rather than unusual physiology. "
            "Check the coverage columns before reading any flag as physiology.",
            "Four cases is a very small reference population; 'unusual' here "
            "means unusual relative to these four cases only.",
        ],
    }


def write_summary(summary: dict, path=None):
    path = path or (MODEL_DIR / "experiment_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


# ── Plots ────────────────────────────────────────────────────────────────────


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_case_with_flags(
    caseid: int,
    table: pd.DataFrame,
    results: pd.DataFrame,
    out_dir=None,
):
    """Physiological trajectories for one case with flagged windows highlighted.

    Drawn from the FULL feature table, not just the analyzed rows, so windows
    that were excluded stay visible as gaps instead of being closed over.
    """
    plt = _plt()
    out_dir = out_dir or PLOT_DIR
    case = table[table["caseid"] == caseid].sort_values("window_index")
    scored = results[results["caseid"] == caseid]
    flagged = scored[scored["anomaly_label"] == 1]

    centres = (case["window_start_s"] + case["window_end_s"]) / 120.0  # minutes
    panels = [
        ("hr", "HR (bpm)", "#1f77b4"),
        ("spo2", "SpO2 (%)", "#2ca02c"),
        ("rr", "RR (breaths/min)", "#9467bd"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(13, 9.5), sharex=True)

    for ax, (short, label, colour) in zip(axes, panels):
        mean = pd.to_numeric(case[f"{short}_mean"], errors="coerce")
        low = pd.to_numeric(case[f"{short}_min"], errors="coerce")
        high = pd.to_numeric(case[f"{short}_max"], errors="coerce")
        ax.fill_between(centres, low, high, color=colour, alpha=0.15,
                        label="window min-max")
        ax.plot(centres, mean, marker="o", markersize=3.5, linewidth=1.5,
                color=colour, label="window mean")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")

    # Coverage panel: the quality context behind every flag.
    ax = axes[3]
    for short, colour in (("hr", "#1f77b4"), ("spo2", "#2ca02c"), ("rr", "#9467bd")):
        ax.plot(centres, pd.to_numeric(case[f"{short}_coverage_pct"], errors="coerce"),
                marker=".", linewidth=1.2, color=colour, label=short)
    ax.axhline(70, color="#d62728", linestyle="--", linewidth=1.0,
               label="70% usable bar")
    ax.set_ylabel("coverage (%)", fontsize=9)
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left", ncol=4)
    ax.set_xlabel("Time from case start (minutes)")

    # Shading, applied to every panel.
    unusable = case[~case["window_usable"].fillna(False).astype(bool)]
    for axis in axes:
        for _, r in unusable.iterrows():
            axis.axvspan(r["window_start_s"] / 60.0, r["window_end_s"] / 60.0,
                         color="#7f7f7f", alpha=0.16, zorder=0)
        for _, r in flagged.iterrows():
            axis.axvspan(r["window_start_s"] / 60.0, r["window_end_s"] / 60.0,
                         color="#d62728", alpha=0.20, zorder=0)

    # Label only the highest-scoring few: flagged windows are often adjacent,
    # and one label per window collapses into an unreadable smear.
    for _, r in flagged.nlargest(3, "anomaly_score").iterrows():
        axes[0].annotate(
            f"w{int(r['window_index'])}: {r['anomaly_score']:+.3f}",
            xy=((r["window_start_s"] + r["window_end_s"]) / 120.0, 1.015),
            xycoords=("data", "axes fraction"), ha="center", fontsize=7,
            color="#d62728",
        )

    axes[0].set_title(
        f"Case {caseid} — 5-minute windows, Isolation Forest baseline\n"
        f"Red = statistically unusual monitoring window ({len(flagged)} flagged)  |  "
        f"Grey = window not usable (below 70% coverage, excluded from the model)  |  "
        f"no interpolation",
        fontsize=10,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"case_{caseid}__anomaly_5min.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_score_distribution(results: pd.DataFrame, out_dir=None):
    """Score histogram with the flag boundary marked."""
    plt = _plt()
    out_dir = out_dir or PLOT_DIR
    fig, ax = plt.subplots(figsize=(9, 4.2))

    normal = results[results["anomaly_label"] == 0]["anomaly_score"]
    flagged = results[results["anomaly_label"] == 1]["anomaly_score"]

    bins = 30
    ax.hist(normal, bins=bins, color="#1f77b4", alpha=0.75,
            label=f"not flagged (n={len(normal)})")
    ax.hist(flagged, bins=bins, color="#d62728", alpha=0.8,
            label=f"flagged (n={len(flagged)})")
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0,
               label="decision boundary")
    ax.set_xlabel("anomaly score  (higher = more unusual)")
    ax.set_ylabel("windows")
    ax.set_title("Anomaly-score distribution across analyzed windows", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "score_distribution.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_top_windows(results: pd.DataFrame, table: pd.DataFrame, n=10, out_dir=None):
    """The n most unusual windows, with the feature values behind each."""
    plt = _plt()
    out_dir = out_dir or PLOT_DIR
    top = results.head(n)
    detail = attach_features(top, table)

    fig, (ax_score, ax_feat) = plt.subplots(
        1, 2, figsize=(14, 0.55 * n + 2.8), gridspec_kw={"width_ratios": [1, 1.5]}
    )

    labels = [
        f"case {int(r.caseid)}  w{int(r.window_index)}\n"
        f"{r.window_start_s / 60:.0f}-{r.window_end_s / 60:.0f} min"
        for r in top.itertuples()
    ]
    positions = np.arange(len(top))[::-1]
    ax_score.barh(positions, top["anomaly_score"], color="#d62728", alpha=0.85)
    ax_score.set_yticks(positions)
    ax_score.set_yticklabels(labels, fontsize=7)
    ax_score.set_xlabel("anomaly score (higher = more unusual)")
    ax_score.set_title(f"Top {n} most unusual monitoring windows", fontsize=10)
    ax_score.grid(alpha=0.3, axis="x")

    # Per-feature z-position relative to the analyzed population, so it is
    # visible WHICH feature made each window unusual.  Display only.
    analyzed = results[INDEX_COLUMNS[:2]].merge(
        table[INDEX_COLUMNS[:2] + MODEL_FEATURES], on=INDEX_COLUMNS[:2], how="left"
    )
    mu = analyzed[MODEL_FEATURES].mean()
    sigma = analyzed[MODEL_FEATURES].std(ddof=1).replace(0, np.nan)
    z = ((detail[MODEL_FEATURES] - mu) / sigma).to_numpy(dtype=float)

    limit = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1.0
    mesh = ax_feat.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax_feat.set_xticks(range(len(MODEL_FEATURES)))
    ax_feat.set_xticklabels(MODEL_FEATURES, rotation=90, fontsize=6)
    ax_feat.set_yticks(range(len(top)))
    ax_feat.set_yticklabels([f"case {int(r.caseid)} w{int(r.window_index)}"
                             for r in top.itertuples()], fontsize=7)
    ax_feat.set_title("Feature values, standard deviations from the\n"
                      "analyzed-population mean (display only)", fontsize=9)
    fig.colorbar(mesh, ax=ax_feat, fraction=0.025, label="sd from mean")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"top_{n}_unusual_windows.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
