"""Evidence objects for flagged windows: the input an XAI/LLM layer will consume.

ANOMALOUS WINDOW -> EVIDENCE -> (explanation, built later)

This module assembles what is already known about each flagged window into one
structured, machine-readable object.  It does not explain anything, does not
write prompts, does not call a model, and does not produce medical language.

Scope discipline
----------------
Every number here is READ from the v1 feature table or the Model B results.
Nothing is recomputed from raw signals, nothing is imputed, and the anomaly
detector is not touched.  The only derived quantities are the change flags in
``observations``, and their full arithmetic basis is emitted alongside them so
a consumer can audit or override the rule.

Framing
-------
A flagged window is a STATISTICALLY UNUSUAL MONITORING WINDOW.  The evidence
object describes what the signals did; it assigns no cause, no diagnosis and no
clinical meaning, and the JSON carries that statement so a downstream consumer
cannot lose it.

The "changed" rule
------------------
``{signal}_changed`` is true when the signal moved further than its own typical
within-window variability:

    trend_score = delta / pooled_std,  pooled_std = sqrt((std_now^2 + std_ref^2) / 2)
    changed     = |trend_score| > CHANGE_K

This is SELF-REFERENTIAL: "moved more than this signal normally wobbles". It is
a statistical band, not a physiological threshold, and CHANGE_K is a tunable
constant rather than a clinical cut-off.
"""

import json
import logging

import numpy as np
import pandas as pd

from vitaldb_audit import config

logger = logging.getLogger("vitaldb_audit.evidence")

SCHEMA_VERSION = "evidence-1.0"

# Statistical band for the change flags.  Not a clinical threshold.
CHANGE_K = 1.0

SIGNALS = ("hr", "spo2", "rr")
SIGNAL_UNITS = {"hr": "bpm", "spo2": "%", "rr": "breaths/min"}

XAI_DIR = config.RESULTS_DIR / "xai"
MODEL_B_RESULTS = config.RESULTS_DIR / "ablation" / "anomaly_results_model_b.csv"

NOT_A_DIAGNOSIS = (
    "A flagged window is a statistically unusual monitoring window — an unusual "
    "physiological feature pattern relative to the other windows in this "
    "dataset. It is NOT an adverse event, NOT clinical deterioration, and NOT a "
    "clinically validated finding. This object describes what the signals did; "
    "it assigns no cause and no clinical meaning."
)


# ── JSON-safe conversion ─────────────────────────────────────────────────────


def _py(value):
    """Convert numpy/pandas scalars to plain Python, and NA to None.

    Without this the evidence file needs a custom encoder to serialise, which
    would leave numpy types leaking into whatever consumes it next.
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else round(float(value), 4)
    if pd.isna(value):
        return None
    return value


# ── Previous usable window lookup ────────────────────────────────────────────


def previous_usable_window(
    table: pd.DataFrame, caseid: int, window_index: int, signal: str
) -> dict:
    """The most recent EARLIER window in which this signal was usable.

    Searches the FULL feature table, including windows the model never scored,
    because "previous usable" is a property of the monitoring timeline rather
    than of the analysed subset.  Never crosses a case boundary.
    """
    history = table[
        (table["caseid"] == caseid)
        & (table["window_index"] < window_index)
        & (table[f"{signal}_usable"].fillna(False).astype(bool))
        & (table[f"{signal}_mean"].notna())
    ]
    if history.empty:
        return {"mean": None, "window_index": None, "windows_back": None}

    reference = history.loc[history["window_index"].idxmax()]
    return {
        "mean": _py(reference[f"{signal}_mean"]),
        "window_index": _py(reference["window_index"]),
        "windows_back": _py(window_index - int(reference["window_index"])),
    }


# ── Per-signal evidence ──────────────────────────────────────────────────────


def signal_evidence(table: pd.DataFrame, row: pd.Series, signal: str) -> dict:
    """Everything known about one signal in one window."""
    caseid = int(row["caseid"])
    window_index = int(row["window_index"])
    reference = previous_usable_window(table, caseid, window_index, signal)

    return {
        "unit": SIGNAL_UNITS[signal],
        "current_mean": _py(row[f"{signal}_mean"]),
        "previous_usable_mean": reference["mean"],
        "previous_usable_window_index": reference["window_index"],
        "windows_since_reference": reference["windows_back"],
        "delta": _py(row[f"{signal}_delta"]),
        "std": _py(row[f"{signal}_std"]),
        "min": _py(row[f"{signal}_min"]),
        "max": _py(row[f"{signal}_max"]),
        "coverage_pct": _py(row[f"{signal}_coverage_pct"]),
        "usable": _py(row[f"{signal}_usable"]),
    }


def change_basis(
    table: pd.DataFrame, row: pd.Series, signal: str, k: float = CHANGE_K
) -> dict:
    """The arithmetic behind one ``{signal}_changed`` flag, emitted for audit."""
    delta = row[f"{signal}_delta"]
    std_now = row[f"{signal}_std"]
    reference = previous_usable_window(table, int(row["caseid"]),
                                       int(row["window_index"]), signal)

    std_reference = None
    if reference["window_index"] is not None:
        match = table[
            (table["caseid"] == row["caseid"])
            & (table["window_index"] == reference["window_index"])
        ]
        if not match.empty:
            std_reference = match.iloc[0][f"{signal}_std"]

    if pd.isna(delta) or std_now is None or pd.isna(std_now) or std_reference is None \
            or pd.isna(std_reference):
        return {
            "delta": _py(delta),
            "pooled_std": None,
            "trend_score": None,
            "threshold_k": k,
            "changed": None,
            "reason": "not computable — no usable reference window for this signal",
        }

    pooled = float(np.sqrt((float(std_now) ** 2 + float(std_reference) ** 2) / 2.0))
    if pooled == 0.0:
        return {
            "delta": _py(delta),
            "pooled_std": 0.0,
            "trend_score": None,
            "threshold_k": k,
            "changed": bool(float(delta) != 0.0),
            "reason": "zero dispersion in both windows; flag falls back to delta != 0",
        }

    trend_score = float(delta) / pooled
    return {
        "delta": _py(delta),
        "pooled_std": round(pooled, 4),
        "trend_score": round(trend_score, 4),
        "threshold_k": k,
        "changed": bool(abs(trend_score) > k),
        "reason": "|delta| exceeds the signal's own within-window variability"
                  if abs(trend_score) > k else
                  "movement within the signal's own within-window variability",
    }


def direction(delta) -> str | None:
    if delta is None or pd.isna(delta):
        return None
    if float(delta) > 0:
        return "increase"
    if float(delta) < 0:
        return "decrease"
    return "no change"


# ── Window evidence ──────────────────────────────────────────────────────────


def build_window_evidence(
    table: pd.DataFrame, result_row: pd.Series, k: float = CHANGE_K
) -> dict:
    """One complete evidence object for one flagged window."""
    caseid = int(result_row["caseid"])
    window_index = int(result_row["window_index"])

    match = table[
        (table["caseid"] == caseid) & (table["window_index"] == window_index)
    ]
    if match.empty:
        raise ValueError(
            f"case {caseid} window {window_index} is not in the feature table"
        )
    row = match.iloc[0]

    signals = {s: signal_evidence(table, row, s) for s in SIGNALS}
    bases = {s: change_basis(table, row, s, k) for s in SIGNALS}

    changed = {s: bases[s]["changed"] for s in SIGNALS}
    n_changed = sum(1 for value in changed.values() if value is True)

    start_s = float(row["window_start_s"])
    end_s = float(row["window_end_s"])

    return {
        "case_id": caseid,
        "window_index": window_index,
        "time_range": {
            "start_s": round(start_s, 1),
            "end_s": round(end_s, 1),
            "start_min": round(start_s / 60.0, 2),
            "end_min": round(end_s / 60.0, 2),
            "duration_min": round((end_s - start_s) / 60.0, 2),
            "label": f"{start_s / 60.0:.1f}-{end_s / 60.0:.1f} min from case start",
        },
        "anomaly_score": _py(result_row["anomaly_score"]),
        # Cast explicitly: pandas iterrows() collapses a mixed-dtype row to a
        # float Series, which would silently render rank 1 as 1.0.
        "anomaly_rank": int(result_row["anomaly_rank"]),
        "anomaly_label": int(result_row["anomaly_label"]),
        "signals": signals,
        "n_core_signals_usable": _py(row["n_core_usable"]),
        "window_usable": _py(row["window_usable"]),
        "consecutive_usable_windows": _py(row["consecutive_usable_windows"]),
        "observations": {
            "hr_changed": changed["hr"],
            "spo2_changed": changed["spo2"],
            "rr_changed": changed["rr"],
            "multiple_signals_changed": n_changed >= 2,
            "n_signals_changed": n_changed,
            "hr_direction": direction(row["hr_delta"]),
            "spo2_direction": direction(row["spo2_delta"]),
            "rr_direction": direction(row["rr_delta"]),
            "change_basis": bases,
        },
    }


def build_evidence(
    table: pd.DataFrame,
    results: pd.DataFrame,
    k: float = CHANGE_K,
    flagged_only: bool = True,
) -> list[dict]:
    """Evidence objects for the flagged windows, most unusual first."""
    subset = results[results["anomaly_label"] == 1] if flagged_only else results
    subset = subset.sort_values("anomaly_rank")
    return [build_window_evidence(table, row, k) for _, row in subset.iterrows()]


def build_document(
    table: pd.DataFrame,
    results: pd.DataFrame,
    k: float = CHANGE_K,
    model_description: str | None = None,
) -> dict:
    """The full evidence_cases.json payload, provenance included."""
    evidence = build_evidence(table, results, k)
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "Structured evidence for each flagged monitoring window, prepared "
            "for a downstream explanation layer. Contains no explanation, no "
            "prompt and no generated text."
        ),
        "interpretation": NOT_A_DIAGNOSIS,
        "source": {
            "model": model_description or (
                "Isolation Forest, Model B: 15 physiological features "
                "(HR/SpO2/RR x mean, std, min, max, delta), coverage features "
                "excluded as model inputs"
            ),
            "feature_table": "results/features/feature_table_5min.csv",
            "results_table": str(MODEL_B_RESULTS.relative_to(config.PROJECT_ROOT)),
            "window_minutes": 5.0,
            "coverage_note": (
                "Coverage percentages are reported as EVIDENCE for every signal "
                "but were not inputs to Model B."
            ),
        },
        "change_rule": {
            "definition": (
                "changed = |delta / pooled_std| > k, "
                "pooled_std = sqrt((std_now^2 + std_reference^2) / 2)"
            ),
            "k": k,
            "self_referential": True,
            "not_a_clinical_threshold": True,
            "note": (
                "A statistical band meaning 'moved more than this signal "
                "normally varies within a window'. Every flag's arithmetic is "
                "emitted under observations.change_basis so it can be audited "
                "or recomputed with a different k."
            ),
        },
        "counts": {
            "windows_flagged": len(evidence),
            "windows_analyzed": int(len(results)),
            "windows_in_feature_table": int(len(table)),
        },
        "evidence": evidence,
    }


def write_document(document: dict, path=None):
    """Write the evidence file.  Must serialise with no custom encoder."""
    path = path or (XAI_DIR / "evidence_cases.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    logger.info("wrote %d evidence objects to %s", len(document["evidence"]), path)
    return path


# ── Human-readable report ────────────────────────────────────────────────────


def _fmt(value, spec=".2f", missing="  n/a"):
    return missing if value is None else format(value, spec)


def render_window(entry: dict) -> str:
    """One flagged window as a compact readable block."""
    lines = []
    header = (f"#{entry['anomaly_rank']}  case {entry['case_id']}  "
              f"window {entry['window_index']}  |  {entry['time_range']['label']}  |  "
              f"score {entry['anomaly_score']:+.4f}")
    lines.append(header)
    lines.append("-" * len(header))

    lines.append(f"  {'signal':<6}{'mean':>9}{'prev':>9}{'delta':>9}"
                 f"{'std':>8}{'min':>8}{'max':>8}{'cov%':>8}   change")
    for signal in SIGNALS:
        data = entry["signals"][signal]
        basis = entry["observations"]["change_basis"][signal]
        changed = basis["changed"]
        if changed is None:
            verdict = "not computable"
        elif changed:
            score = basis["trend_score"]
            arrow = "up" if (data["delta"] or 0) > 0 else "down"
            verdict = (f"CHANGED {arrow}"
                       + (f" ({score:+.2f} sd)" if score is not None else ""))
        else:
            verdict = "stable"
        lines.append(
            f"  {signal:<6}{_fmt(data['current_mean']):>9}"
            f"{_fmt(data['previous_usable_mean']):>9}"
            f"{_fmt(data['delta'], '+.2f'):>9}{_fmt(data['std']):>8}"
            f"{_fmt(data['min']):>8}{_fmt(data['max']):>8}"
            f"{_fmt(data['coverage_pct'], '.1f'):>8}   {verdict}"
        )

    observations = entry["observations"]
    lines.append(
        f"  signals changed: {observations['n_signals_changed']} of 3"
        f"   |   multiple_signals_changed: "
        f"{str(observations['multiple_signals_changed']).lower()}"
        f"   |   core signals usable: {entry['n_core_signals_usable']}/3"
    )

    references = {
        entry["signals"][s]["windows_since_reference"] for s in SIGNALS
    } - {None, 1}
    if references:
        detail = ", ".join(
            f"{s} compares back {entry['signals'][s]['windows_since_reference']} windows"
            for s in SIGNALS
            if entry["signals"][s]["windows_since_reference"] not in (None, 1)
        )
        lines.append(f"  note: {detail} (a gap in usable monitoring, not 5 minutes)")

    return "\n".join(lines)


def render_report(document: dict, n: int = 10) -> str:
    """Compact readable report for the top n flagged windows."""
    evidence = document["evidence"][:n]
    counts = document["counts"]

    out = [
        "EVIDENCE REPORT — TOP "
        f"{len(evidence)} FLAGGED MONITORING WINDOWS",
        "=" * 78,
        f"model    : {document['source']['model']}",
        f"flagged  : {counts['windows_flagged']} of {counts['windows_analyzed']} "
        f"analyzed windows",
        f"change   : {document['change_rule']['definition']}  (k="
        f"{document['change_rule']['k']}, statistical band, not a clinical threshold)",
        "",
        "Each block below is a statistically unusual monitoring window.",
        "It describes what the signals did. It is not a diagnosis.",
        "",
    ]
    for entry in evidence:
        out.append(render_window(entry))
        out.append("")
    return "\n".join(out)


def write_report(text: str, path=None):
    path = path or (XAI_DIR / "evidence_report_top10.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
