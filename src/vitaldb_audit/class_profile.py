"""Descriptive class profile for the Model B baseline.

This is the honest analogue of a classification report for an UNSUPERVISED
model with no ground truth.  It has the same shape — classes, support, per-class
statistics — but every number is descriptive.

WHAT IS DELIBERATELY ABSENT AND WHY
-----------------------------------
Precision, recall, F1, accuracy, AUROC, AUPRC and a confusion matrix all require
known-correct labels.  VitalDB carries no anomaly annotation, so there is
nothing for a prediction to be right or wrong against.  Reporting those numbers
would mean inventing a reference and then scoring against it, which would look
authoritative and mean nothing.  ``build_profile`` therefore emits no such key,
and a test enforces it.

WHAT IS REPORTED INSTEAD
------------------------
- Class support: how many windows fall in each class, and what happened to the
  windows that were never classified at all.
- Score distribution per class.
- Per-case composition.
- Which features separate the classes, measured by Cliff's delta — a
  non-parametric effect size, not a performance metric and not feature
  importance.  It answers "how much do these two groups of windows differ on
  this feature", nothing more.
"""

import json
import logging

import numpy as np
import pandas as pd

from vitaldb_audit import ablation, anomaly, config

logger = logging.getLogger("vitaldb_audit.class_profile")

CLASS_LABELS = {1: "flagged", 0: "not_flagged"}

# Coverage columns are NOT Model B inputs.  They are profiled separately so a
# reader can see whether the classes differ on data quality anyway.
CONTEXT_FEATURES = ablation.COVERAGE_FEATURES

PROFILE_PATH = anomaly.MODEL_DIR / "class_profile.json"
PLOT_DIR = config.RESULTS_DIR / "plots" / "profile"


# ── Effect size ──────────────────────────────────────────────────────────────


def cliffs_delta(a, b) -> float | None:
    """Non-parametric effect size between two samples, in [-1, +1].

    +1 means every value in ``a`` exceeds every value in ``b``; 0 means the two
    are completely intermixed.  Chosen over a difference of means because the
    feature distributions here are heavily zero-inflated and non-normal.

    This is a DESCRIPTIVE separation measure. It is not a performance metric,
    not a p-value, and not feature importance.
    """
    a = np.asarray(pd.Series(a).dropna(), dtype=float)
    b = np.asarray(pd.Series(b).dropna(), dtype=float)
    if a.size == 0 or b.size == 0:
        return None
    comparison = np.sign(a[:, None] - b[None, :])
    return round(float(comparison.sum() / (a.size * b.size)), 4)


def magnitude(delta: float | None) -> str:
    """Conventional descriptive bands for Cliff's delta (Romano et al.)."""
    if delta is None:
        return "not computable"
    size = abs(delta)
    if size < 0.147:
        return "negligible"
    if size < 0.33:
        return "small"
    if size < 0.474:
        return "medium"
    return "large"


# ── Profile sections ─────────────────────────────────────────────────────────


def class_support(results: pd.DataFrame, table: pd.DataFrame,
                  selection: dict) -> dict:
    """How many windows are in each class, and what happened to the rest."""
    analyzed = len(results)
    flagged = int(results["anomaly_label"].sum())
    return {
        "classes": [
            {
                "class": "flagged",
                "meaning": "statistically unusual monitoring window",
                "support": flagged,
                "pct_of_analyzed": round(flagged / analyzed * 100, 2),
            },
            {
                "class": "not_flagged",
                "meaning": "window not separated from the population by the model",
                "support": analyzed - flagged,
                "pct_of_analyzed": round((analyzed - flagged) / analyzed * 100, 2),
            },
        ],
        "analyzed_total": analyzed,
        "not_classified": {
            "windows_in_feature_table": int(len(table)),
            "excluded_window_not_usable": selection["rows_excluded_unusable_window"],
            "excluded_null_model_feature": selection["rows_excluded_null_feature"],
            "note": (
                "These windows were never presented to the model. They are not a "
                "third class and not errors; they are windows the pipeline "
                "declined to model, preserved in the feature table."
            ),
        },
        "class_balance_note": (
            "The flagged count is set by the contamination parameter (0.10), not "
            "discovered by the model. Class sizes are a configuration choice."
        ),
    }


def score_profile(results: pd.DataFrame) -> list[dict]:
    """Score distribution within each class."""
    out = []
    for label, name in CLASS_LABELS.items():
        scores = results[results["anomaly_label"] == label]["anomaly_score"]
        if scores.empty:
            out.append({"class": name, "support": 0})
            continue
        out.append({
            "class": name,
            "support": int(len(scores)),
            "min": round(float(scores.min()), 6),
            "q25": round(float(scores.quantile(0.25)), 6),
            "median": round(float(scores.median()), 6),
            "q75": round(float(scores.quantile(0.75)), 6),
            "max": round(float(scores.max()), 6),
            "mean": round(float(scores.mean()), 6),
            "std": round(float(scores.std(ddof=1)), 6) if len(scores) > 1 else None,
        })
    return out


def per_case_composition(results: pd.DataFrame) -> list[dict]:
    """Class composition case by case."""
    grouped = results.groupby("caseid").agg(
        analyzed=("anomaly_label", "size"),
        flagged=("anomaly_label", "sum"),
        max_score=("anomaly_score", "max"),
        median_score=("anomaly_score", "median"),
    ).reset_index()
    grouped["not_flagged"] = grouped["analyzed"] - grouped["flagged"]
    grouped["pct_flagged"] = (
        grouped["flagged"] / grouped["analyzed"] * 100).round(2)
    grouped["max_score"] = grouped["max_score"].round(6)
    grouped["median_score"] = grouped["median_score"].round(6)
    return grouped.astype(object).to_dict("records")


def feature_separation(results: pd.DataFrame, table: pd.DataFrame,
                       features: list[str]) -> list[dict]:
    """Per-feature class medians and Cliff's delta, strongest separation first."""
    merged = results[["caseid", "window_index", "anomaly_label"]].merge(
        table, on=["caseid", "window_index"], how="left")
    flagged = merged[merged["anomaly_label"] == 1]
    others = merged[merged["anomaly_label"] == 0]

    rows = []
    for feature in features:
        delta = cliffs_delta(flagged[feature], others[feature])
        rows.append({
            "feature": feature,
            "median_flagged": round(float(flagged[feature].median()), 4)
            if flagged[feature].notna().any() else None,
            "median_not_flagged": round(float(others[feature].median()), 4)
            if others[feature].notna().any() else None,
            "cliffs_delta": delta,
            "separation": magnitude(delta),
        })
    return sorted(
        rows,
        key=lambda r: abs(r["cliffs_delta"]) if r["cliffs_delta"] is not None else -1,
        reverse=True,
    )


def evidence_composition(evidence_path=None) -> dict | None:
    """What the interpretable evidence flags say about the flagged class."""
    evidence_path = evidence_path or (config.RESULTS_DIR / "xai" / "evidence_cases.json")
    if not evidence_path.exists():
        return None
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    entries = document.get("evidence", [])
    if not entries:
        return None

    mean_only = spread_only = both = neither = 0
    for entry in entries:
        observations = entry["observations"]
        changed = observations["n_signals_changed"] > 0
        spread = observations.get("n_signals_dispersion_unusual", 0) > 0
        if changed and spread:
            both += 1
        elif changed:
            mean_only += 1
        elif spread:
            spread_only += 1
        else:
            neither += 1

    return {
        "flagged_windows_described": len(entries),
        "mean_shift_only": mean_only,
        "dispersion_only": spread_only,
        "both": both,
        "neither": neither,
        "note": (
            "Composition of the flagged class by interpretable evidence flag. "
            "Descriptive only: the evidence flags are a separate rule, not a "
            "reference label, so this is not a correctness check."
        ),
    }


# ── Assembly ─────────────────────────────────────────────────────────────────


def build_profile(
    results: pd.DataFrame,
    table: pd.DataFrame,
    selection: dict,
    model_features: list[str] | None = None,
) -> dict:
    """The full descriptive class profile."""
    model_features = list(model_features or ablation.MODEL_B_FEATURES)
    return {
        "report": "descriptive_class_profile",
        "model": (
            "Isolation Forest, Model B: 15 physiological features, "
            "coverage features excluded as model inputs"
        ),
        "ground_truth": (
            "none available. This is an unsupervised model, so there is nothing "
            "for a prediction to be correct or incorrect against. No supervised "
            "performance measure and no confusion matrix is reported, because "
            "none of them are defined for this experiment."
        ),
        "interpretation": (
            "'flagged' means statistically unusual relative to the other windows "
            "in this dataset. It does NOT mean an adverse event, deterioration, "
            "or any clinically validated finding."
        ),
        "support": class_support(results, table, selection),
        "score_distribution_by_class": score_profile(results),
        "per_case_composition": per_case_composition(results),
        "feature_separation": {
            "measure": (
                "Cliff's delta between the flagged and not-flagged windows: a "
                "non-parametric effect size in [-1, +1]. Descriptive separation "
                "only — not a performance measure and not feature importance."
            ),
            "model_features": feature_separation(results, table, model_features),
            "context_features_not_model_inputs": feature_separation(
                results, table, CONTEXT_FEATURES),
        },
        "flagged_class_evidence_composition": evidence_composition(),
    }


def write_profile(profile: dict, path=None):
    path = path or PROFILE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return path


# ── Plot ─────────────────────────────────────────────────────────────────────


def plot_profile(results: pd.DataFrame, table: pd.DataFrame, profile: dict,
                 out_dir=None):
    """Score distribution by class, and the feature separation ranking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or PLOT_DIR
    fig, (ax_score, ax_sep) = plt.subplots(
        1, 2, figsize=(14.5, 6.4), gridspec_kw={"width_ratios": [1, 1.35]})

    flagged = results[results["anomaly_label"] == 1]["anomaly_score"]
    others = results[results["anomaly_label"] == 0]["anomaly_score"]
    parts = ax_score.boxplot(
        [others, flagged], vert=False, widths=0.55, patch_artist=True,
        tick_labels=[f"not flagged\n(n={len(others)})", f"flagged\n(n={len(flagged)})"])
    for patch, colour in zip(parts["boxes"], ["#bbbbbb", "#d62728"]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
    for group, y, colour in ((others, 1, "#777777"), (flagged, 2, "#d62728")):
        ax_score.scatter(group, np.random.default_rng(3).normal(y, 0.055, len(group)),
                         s=14, color=colour, alpha=0.6, zorder=3)
    ax_score.axvline(0, color="#333333", linestyle="--", linewidth=0.9,
                     label="decision boundary")
    ax_score.set_xlabel("anomaly score  (higher = more unusual)")
    ax_score.set_title("Score distribution by class", fontsize=10)
    ax_score.legend(fontsize=8, loc="lower right")
    ax_score.grid(alpha=0.3, axis="x")

    separation = profile["feature_separation"]["model_features"]
    names = [r["feature"] for r in separation][::-1]
    deltas = [r["cliffs_delta"] or 0.0 for r in separation][::-1]
    colours = ["#d62728" if d > 0 else "#1f77b4" for d in deltas]
    positions = np.arange(len(names))
    ax_sep.barh(positions, deltas, color=colours, alpha=0.85)
    ax_sep.set_yticks(positions)
    ax_sep.set_yticklabels(names, fontsize=8)
    ax_sep.axvline(0, color="#333333", linewidth=0.8)
    for boundary in (0.147, 0.33, 0.474):
        for sign in (1, -1):
            ax_sep.axvline(sign * boundary, color="#999999", linestyle=":",
                           linewidth=0.7)
    ax_sep.set_xlabel("Cliff's delta   (flagged vs not flagged; dotted = "
                      "negligible / small / medium / large)")
    ax_sep.set_title("Which features separate the classes\n"
                     "descriptive effect size, not feature importance", fontsize=10)
    ax_sep.grid(alpha=0.3, axis="x")

    fig.suptitle(
        "Model B descriptive class profile — unsupervised, no ground truth, "
        "no supervised performance measure",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "class_profile.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
