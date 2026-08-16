"""Analysis-only benchmarks for the Model B baseline.

Four questions, none of which require a ground-truth label:

  1. SEED STABILITY     Is the result an artifact of one random seed?
  2. SIMPLE BASELINE    Does a plain robust z-score find the same windows?
  3. TEMPORAL COHERENCE Do flags cluster in time, or scatter like noise?
  4. EVIDENCE REVIEW    What is each flag actually resting on?

Nothing here modifies the canonical model, the feature table, the preprocessing
or any existing artifact.  Seeds 1-5 fit throwaway estimators that are scored
and discarded; the canonical results are read from disk and never rewritten.

NOT VALIDATION.  There is still no ground truth.  Agreement between two
unsupervised methods is agreement, not correctness — if both are wrong in the
same way they will agree perfectly.  Stability across seeds means the procedure
is reproducible, not that its output is right.  No supervised performance
measure is computed anywhere in this module.
"""

import json
import logging

import numpy as np
import pandas as pd

from vitaldb_audit import ablation, anomaly, config

logger = logging.getLogger("vitaldb_audit.benchmark")

BENCHMARK_DIR = config.RESULTS_DIR / "benchmark"

CANONICAL_SEED = anomaly.RANDOM_SEED
ALTERNATE_SEEDS = [1, 7, 42, 1234, 99991]

# Robust z-score baseline: values beyond this many robust sd count as extreme
# under the threshold variant.  The primary comparison instead takes the top-k
# windows with k matched to the Isolation Forest flag count, so the two methods
# are compared at an identical budget.
Z_THRESHOLD = 3.5


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark 1 — seed stability
# ══════════════════════════════════════════════════════════════════════════════


def top_k_keys(results: pd.DataFrame, k: int = 10) -> list[tuple]:
    top = results.nsmallest(k, "anomaly_rank")
    return list(map(tuple, top[["caseid", "window_index"]].to_numpy()))


def seed_stability(selected: pd.DataFrame, canonical: pd.DataFrame,
                   seeds: list[int] | None = None) -> pd.DataFrame:
    """Refit Model B under alternate seeds and compare to the canonical run.

    Same rows, same 15 features, same contamination, same tree count.  Only the
    random_state differs, so any disagreement is seed sensitivity and nothing
    else.
    """
    seeds = list(seeds if seeds is not None else ALTERNATE_SEEDS)
    keys = ["caseid", "window_index"]
    canonical_top = top_k_keys(canonical, 10)

    rows = []
    for seed in [CANONICAL_SEED] + seeds:
        _, results = ablation.run_variant(
            selected, ablation.MODEL_B_FEATURES, random_state=seed)

        merged = canonical[keys + ["anomaly_score", "anomaly_rank", "anomaly_label"]].merge(
            results[keys + ["anomaly_score", "anomaly_rank", "anomaly_label"]],
            on=keys, suffixes=("_canonical", "_seed"))

        spearman = float(merged["anomaly_rank_canonical"].corr(
            merged["anomaly_rank_seed"], method="spearman"))
        label_agreement = float(
            (merged["anomaly_label_canonical"] == merged["anomaly_label_seed"]).mean())
        overlap = len(set(canonical_top) & set(top_k_keys(results, 10)))

        rows.append({
            "seed": seed,
            "is_canonical": seed == CANONICAL_SEED,
            "windows_analyzed": int(len(results)),
            "windows_flagged": int(results["anomaly_label"].sum()),
            "top10_overlap_with_canonical": overlap,
            "spearman_rank_correlation": round(spearman, 4),
            "pct_labels_unchanged": round(label_agreement * 100, 2),
            "top10": "; ".join(f"{c}:{w}" for c, w in top_k_keys(results, 10)),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark 2 — robust z-score baseline
# ══════════════════════════════════════════════════════════════════════════════


def robust_scale(values: pd.Series) -> tuple[float, str]:
    """A robust dispersion estimate, with the estimator actually used.

    MAD is preferred, but 7 of the 15 features have a MAD of exactly zero on
    this data (SpO2 sits at its ceiling; RR is a set ventilator rate), so a
    plain MAD z-score would divide by zero or silently drop the strongest
    separating features.  IQR is the fallback.  A feature with neither is
    effectively constant and is excluded rather than allowed to produce an
    infinite score.
    """
    median = float(values.median())
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    if mad > 0:
        return mad, "MAD"
    iqr = float(values.quantile(0.75) - values.quantile(0.25)) / 1.349
    if iqr > 0:
        return iqr, "IQR"
    return 0.0, "degenerate"


def robust_zscore_baseline(selected: pd.DataFrame, k_flagged: int,
                           features: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Flag windows by their most extreme robust z-score across features.

    Deliberately simple and fully interpretable: no model, no fitting, no
    randomness.  Every window's score is the largest number of robust standard
    deviations any single feature sits from that feature's median, and the
    driving feature is reported alongside it.

    The flag budget is matched to the Isolation Forest's, so overlap between the
    two is a like-for-like comparison rather than an artifact of one method
    flagging more windows.
    """
    features = list(features or ablation.MODEL_B_FEATURES)
    matrix = selected[features]

    z_columns = {}
    scale_report = []
    for feature in features:
        values = matrix[feature].astype(float)
        median = float(values.median())
        scale, estimator = robust_scale(values)
        scale_report.append({
            "feature": feature,
            "median": round(median, 4),
            "scale": round(scale, 4),
            "estimator": estimator,
            "used": estimator != "degenerate",
        })
        if scale > 0:
            z_columns[feature] = (values - median) / scale

    if not z_columns:
        raise ValueError("every feature is degenerate; no baseline can be built")

    z = pd.DataFrame(z_columns, index=selected.index)
    absolute = z.abs()

    out = selected[anomaly.INDEX_COLUMNS].copy()
    out["baseline_score"] = absolute.max(axis=1).round(4)
    out["driving_feature"] = absolute.idxmax(axis=1)
    out["driving_z"] = z.to_numpy()[
        np.arange(len(z)), [z.columns.get_loc(c) for c in out["driving_feature"]]
    ].round(4)
    out["n_features_beyond_threshold"] = (absolute > Z_THRESHOLD).sum(axis=1)

    out = out.sort_values("baseline_score", ascending=False).reset_index(drop=True)
    out.insert(len(anomaly.INDEX_COLUMNS), "baseline_rank",
               np.arange(1, len(out) + 1))
    out["baseline_label"] = (out["baseline_rank"] <= k_flagged).astype(int)

    meta = {
        "method": (
            "robust z-score: z = (x - median) / scale, scale = 1.4826*MAD, "
            "falling back to IQR/1.349 where MAD is zero; window score = "
            "max |z| over features"
        ),
        "deterministic": True,
        "features_available": len(features),
        "features_used": int(sum(1 for r in scale_report if r["used"])),
        "features_excluded_degenerate": [
            r["feature"] for r in scale_report if not r["used"]
        ],
        "flag_rule": f"top {k_flagged} by score, matched to the Isolation Forest count",
        "threshold_variant_z": Z_THRESHOLD,
        "windows_with_any_feature_beyond_threshold": int(
            (out["n_features_beyond_threshold"] > 0).sum()),
        "scales": scale_report,
    }
    return out, meta


def compare_to_isolation_forest(baseline: pd.DataFrame,
                                canonical: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join the two methods window-by-window and describe their agreement."""
    keys = ["caseid", "window_index", "window_start_s", "window_end_s"]
    merged = canonical[keys + ["anomaly_score", "anomaly_rank", "anomaly_label"]].merge(
        baseline[keys + ["baseline_score", "baseline_rank", "baseline_label",
                         "driving_feature", "driving_z"]],
        on=keys)

    def agreement(row):
        iso, base = bool(row["anomaly_label"]), bool(row["baseline_label"])
        if iso and base:
            return "both flagged"
        if iso:
            return "isolation forest only"
        if base:
            return "baseline only"
        return "neither"

    merged["agreement"] = merged.apply(agreement, axis=1)
    merged = merged.sort_values("anomaly_rank").reset_index(drop=True)

    spearman = float(merged["anomaly_rank"].corr(
        merged["baseline_rank"], method="spearman"))
    iso_top = set(map(tuple, merged.nsmallest(10, "anomaly_rank")[
        ["caseid", "window_index"]].to_numpy()))
    base_top = set(map(tuple, merged.nsmallest(10, "baseline_rank")[
        ["caseid", "window_index"]].to_numpy()))
    counts = merged["agreement"].value_counts().to_dict()

    summary = {
        "windows_compared": int(len(merged)),
        "isolation_forest_flagged": int(merged["anomaly_label"].sum()),
        "baseline_flagged": int(merged["baseline_label"].sum()),
        "flagged_by_both": int(counts.get("both flagged", 0)),
        "isolation_forest_only": int(counts.get("isolation forest only", 0)),
        "baseline_only": int(counts.get("baseline only", 0)),
        "flagged_by_neither": int(counts.get("neither", 0)),
        "top10_overlap": len(iso_top & base_top),
        "spearman_rank_correlation": round(spearman, 4),
        "rank_correlation_note": (
            "Both rankings order the same 135 windows, so Spearman is "
            "meaningful as an agreement statistic. It is not a performance "
            "measure: neither ranking is a reference."
        ),
    }
    return merged, summary


def episode_detected(frame: pd.DataFrame, label_column: str,
                     caseid: int = 4, start_min: float = 285.0,
                     end_min: float = 340.0) -> dict:
    """Does a given method flag the case 4 episode?"""
    window = frame[
        (frame["caseid"] == caseid)
        & (frame["window_start_s"] >= start_min * 60)
        & (frame["window_end_s"] <= end_min * 60)
    ]
    flagged = window[window[label_column] == 1]
    return {
        "caseid": caseid,
        "range_min": [start_min, end_min],
        "windows_in_range": int(len(window)),
        "flagged": int(len(flagged)),
        "window_indices": sorted(flagged["window_index"].astype(int).tolist()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark 3 — temporal coherence
# ══════════════════════════════════════════════════════════════════════════════


def flagged_runs(results: pd.DataFrame) -> list[dict]:
    """Maximal runs of consecutive flagged window indices, per case.

    Adjacency is window_index difference of exactly 1.  A window the model never
    analysed therefore breaks a run, which is the conservative reading: we
    cannot claim continuity across a window that was never scored.
    """
    runs = []
    flagged = results[results["anomaly_label"] == 1]
    for caseid, group in flagged.groupby("caseid"):
        indices = sorted(group["window_index"].astype(int).tolist())
        if not indices:
            continue
        start = previous = indices[0]
        for index in indices[1:]:
            if index == previous + 1:
                previous = index
                continue
            runs.append({"caseid": int(caseid), "start_window": start,
                         "end_window": previous, "length": previous - start + 1})
            start = previous = index
        runs.append({"caseid": int(caseid), "start_window": start,
                     "end_window": previous, "length": previous - start + 1})
    return runs


def temporal_coherence(results: pd.DataFrame) -> dict:
    """Do the flags cluster in time, or scatter independently?"""
    runs = flagged_runs(results)
    flagged_total = int(results["anomaly_label"].sum())

    in_runs = sum(r["length"] for r in runs if r["length"] >= 2)
    lengths = [r["length"] for r in runs]

    per_case = []
    for caseid, group in results.groupby("caseid"):
        case_runs = [r for r in runs if r["caseid"] == int(caseid)]
        case_flagged = int(group["anomaly_label"].sum())
        case_in_runs = sum(r["length"] for r in case_runs if r["length"] >= 2)
        per_case.append({
            "caseid": int(caseid),
            "windows_analyzed": int(len(group)),
            "flagged": case_flagged,
            "with_adjacent_flagged_neighbour": case_in_runs,
            "pct_in_contiguous_runs": round(case_in_runs / case_flagged * 100, 2)
            if case_flagged else 0.0,
            "n_runs": len(case_runs),
            "run_lengths": sorted((r["length"] for r in case_runs), reverse=True),
            "longest_run": max((r["length"] for r in case_runs), default=0),
        })

    return {
        "overall": {
            "windows_analyzed": int(len(results)),
            "flagged": flagged_total,
            "with_adjacent_flagged_neighbour": in_runs,
            "pct_in_contiguous_runs": round(in_runs / flagged_total * 100, 2)
            if flagged_total else 0.0,
            "n_runs": len(runs),
            "run_lengths": sorted(lengths, reverse=True),
            "longest_run": max(lengths, default=0),
            "isolated_flags": sum(1 for length in lengths if length == 1),
        },
        "per_case": per_case,
        "runs": runs,
        "adjacency_definition": (
            "consecutive window_index within the same case; a window that was "
            "not analysed breaks a run rather than being assumed continuous"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark 4 — evidence review
# ══════════════════════════════════════════════════════════════════════════════

REVIEW_RULES = {
    "physiologically_supported": (
        "every core signal at 100% coverage AND at least one evidence flag "
        "(mean change or unusual dispersion)"
    ),
    "mainly_data_quality": (
        "at least one core signal below 100% coverage AND no evidence flag"
    ),
    "ambiguous": (
        "anything else: imperfect coverage alongside an evidence flag, or full "
        "coverage with no evidence flag at all"
    ),
}


def review_flagged_windows(evidence_document: dict,
                           table: pd.DataFrame) -> pd.DataFrame:
    """Classify each flagged window by what its flag rests on.

    THIS IS NOT CLINICAL VALIDATION.  "Physiologically supported" means the
    evidence points at signal behaviour rather than at data quality.  It makes
    no claim that the behaviour is abnormal, meaningful, or clinically real.
    """
    coverage_columns = ablation.COVERAGE_FEATURES
    rows = []
    for entry in evidence_document["evidence"]:
        caseid, window_index = entry["case_id"], entry["window_index"]
        match = table[(table["caseid"] == caseid)
                      & (table["window_index"] == window_index)]
        min_coverage = float(match.iloc[0][coverage_columns].min())

        observations = entry["observations"]
        n_changed = observations["n_signals_changed"]
        n_dispersion = observations.get("n_signals_dispersion_unusual", 0)
        has_evidence = (n_changed + n_dispersion) > 0
        full_coverage = min_coverage >= 100.0

        if full_coverage and has_evidence:
            verdict = "physiologically_supported"
        elif not full_coverage and not has_evidence:
            verdict = "mainly_data_quality"
        else:
            verdict = "ambiguous"

        drivers = [f"{s}_change" for s in ("hr", "spo2", "rr")
                   if observations[f"{s}_changed"] is True]
        drivers += [f"{s}_dispersion" for s in ("hr", "spo2", "rr")
                    if observations.get(f"{s}_dispersion_unusual") is True]

        rows.append({
            "caseid": caseid,
            "window_index": window_index,
            "anomaly_rank": entry["anomaly_rank"],
            "anomaly_score": entry["anomaly_score"],
            "time_range_min": entry["time_range"]["label"].replace(
                " from case start", ""),
            "min_coverage_pct": round(min_coverage, 2),
            "n_signals_changed": n_changed,
            "n_signals_dispersion_unusual": n_dispersion,
            "evidence_drivers": "; ".join(drivers) if drivers else "none",
            "review": verdict,
        })
    return pd.DataFrame(rows).sort_values("anomaly_rank").reset_index(drop=True)


def review_summary(review: pd.DataFrame) -> dict:
    counts = review["review"].value_counts().to_dict()
    return {
        "windows_reviewed": int(len(review)),
        "physiologically_supported": int(counts.get("physiologically_supported", 0)),
        "mainly_data_quality": int(counts.get("mainly_data_quality", 0)),
        "ambiguous": int(counts.get("ambiguous", 0)),
        "rules": REVIEW_RULES,
        "not_clinical_validation": (
            "'Physiologically supported' means the flag rests on signal "
            "behaviour rather than on data quality. It is NOT a claim that the "
            "behaviour is abnormal, clinically meaningful, or medically "
            "accurate. No clinical review has taken place."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Markdown table from a frame."""
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, divider]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.4f}" if abs(value) < 1000 else f"{value:.1f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(summary: dict, seed_frame: pd.DataFrame,
                  comparison: pd.DataFrame, coherence: dict,
                  review: pd.DataFrame) -> str:
    """The benchmark report, with observation and interpretation kept apart."""
    seeds = summary["benchmark_1_seed_stability"]
    baseline = summary["benchmark_2_simple_baseline"]
    overall = coherence["overall"]
    reviewed = summary["benchmark_4_evidence_review"]

    out = []
    add = out.append

    add("# Model B Benchmark Report")
    add("")
    add(f"Isolation Forest, 15 physiological features, contamination "
        f"{anomaly.DEFAULT_CONTAMINATION}, {anomaly.DEFAULT_N_ESTIMATORS} trees, "
        f"canonical seed {CANONICAL_SEED}, {summary['windows_analyzed']} analyzed "
        f"windows, {summary['windows_flagged']} flagged.")
    add("")
    add("> **There is no ground-truth anomaly label in this data.** Nothing in this "
        "report is a measure of correctness. No accuracy, precision, recall, F1, "
        "AUROC or AUPRC is computed, because none of them are defined here. "
        "Agreement between two unsupervised methods is agreement, not validation: "
        "two methods wrong in the same way agree perfectly.")
    add("")
    add("Each section separates **OBSERVED RESULTS** — numbers produced by the runs — "
        "from **INTERPRETATION** — what they do and do not license us to say.")
    add("")
    add("---")
    add("")

    # ── Benchmark 1 ──────────────────────────────────────────────────────────
    add("## Benchmark 1 — Seed stability")
    add("")
    add(f"Model B refit under {len(ALTERNATE_SEEDS)} alternate seeds. Same rows, "
        f"same 15 features, same contamination, same tree count; only "
        f"`random_state` differs.")
    add("")
    add("### OBSERVED RESULTS")
    add("")
    add(_table(seed_frame, ["seed", "windows_flagged",
                            "top10_overlap_with_canonical",
                            "spearman_rank_correlation", "pct_labels_unchanged"]))
    add("")
    add(f"- Flagged count across seeds: {seeds['flagged_min']}–{seeds['flagged_max']} "
        f"(canonical {summary['windows_flagged']}).")
    add(f"- Top-10 overlap with canonical: {seeds['top10_overlap_min']}–"
        f"{seeds['top10_overlap_max']} of 10 (mean {seeds['top10_overlap_mean']}).")
    add(f"- Spearman rank correlation: {seeds['spearman_min']}–"
        f"{seeds['spearman_max']} (mean {seeds['spearman_mean']}).")
    add(f"- Labels unchanged: {seeds['pct_labels_unchanged_min']}%–"
        f"{seeds['pct_labels_unchanged_max']}% (mean "
        f"{seeds['pct_labels_unchanged_mean']}%).")
    add("")
    add("### INTERPRETATION")
    add("")
    add(seeds["interpretation"])
    add("")
    add("---")
    add("")

    # ── Benchmark 2 ──────────────────────────────────────────────────────────
    add("## Benchmark 2 — Simple statistical baseline")
    add("")
    add(f"`{baseline['method']}`")
    add("")
    add(f"Deterministic, no fitting, no randomness. "
        f"{baseline['features_used']} of {baseline['features_available']} features "
        f"contribute; {len(baseline['features_excluded_degenerate'])} are excluded "
        f"as degenerate (`{'`, `'.join(baseline['features_excluded_degenerate'])}`) "
        f"because both MAD and IQR are zero — they are effectively constant across "
        f"the analyzed windows and would otherwise divide by zero.")
    add("")
    add("### OBSERVED RESULTS")
    add("")
    add(f"- Isolation Forest flagged **{baseline['isolation_forest_flagged']}**, "
        f"baseline flagged **{baseline['baseline_flagged']}** (budget matched).")
    add(f"- Flagged by both: **{baseline['flagged_by_both']}**. "
        f"Isolation Forest only: {baseline['isolation_forest_only']}. "
        f"Baseline only: {baseline['baseline_only']}.")
    add(f"- Top-10 overlap: **{baseline['top10_overlap']} of 10**.")
    add(f"- Spearman rank correlation: **{baseline['spearman_rank_correlation']}**.")
    add("")
    add("Top 10 by each method:")
    add("")
    top_iso = comparison.nsmallest(10, "anomaly_rank")[
        ["caseid", "window_index", "anomaly_score", "baseline_rank"]]
    add("*Isolation Forest*")
    add("")
    add(_table(top_iso, ["caseid", "window_index", "anomaly_score", "baseline_rank"]))
    add("")
    top_base = comparison.nsmallest(10, "baseline_rank")[
        ["caseid", "window_index", "baseline_score", "driving_feature",
         "anomaly_rank"]]
    add("*Robust z-score baseline*")
    add("")
    add(_table(top_base, ["caseid", "window_index", "baseline_score",
                          "driving_feature", "anomaly_rank"]))
    add("")
    add("Case 4 episode (285–340 min):")
    add("")
    add(f"- Isolation Forest: {baseline['episode_isolation_forest']['flagged']} of "
        f"{baseline['episode_isolation_forest']['windows_in_range']} windows "
        f"→ `{baseline['episode_isolation_forest']['window_indices']}`")
    add(f"- Baseline: {baseline['episode_baseline']['flagged']} of "
        f"{baseline['episode_baseline']['windows_in_range']} windows "
        f"→ `{baseline['episode_baseline']['window_indices']}`")
    add("")
    add("### INTERPRETATION")
    add("")
    add(baseline["interpretation"])
    add("")
    add("---")
    add("")

    # ── Benchmark 3 ──────────────────────────────────────────────────────────
    add("## Benchmark 3 — Temporal coherence")
    add("")
    add(f"Adjacency: {coherence['adjacency_definition']}.")
    add("")
    add("### OBSERVED RESULTS")
    add("")
    add(f"- Flagged windows: **{overall['flagged']}**")
    add(f"- With an adjacent flagged neighbour: **"
        f"{overall['with_adjacent_flagged_neighbour']}** "
        f"(**{overall['pct_in_contiguous_runs']}%** of flags)")
    add(f"- Runs: **{overall['n_runs']}**, lengths `{overall['run_lengths']}`")
    add(f"- Longest run: **{overall['longest_run']}** windows "
        f"({overall['longest_run'] * 5} minutes)")
    add(f"- Isolated single-window flags: **{overall['isolated_flags']}**")
    add("")
    add(_table(pd.DataFrame(coherence["per_case"]),
               ["caseid", "windows_analyzed", "flagged",
                "with_adjacent_flagged_neighbour", "pct_in_contiguous_runs",
                "n_runs", "longest_run"]))
    add("")
    add("### INTERPRETATION")
    add("")
    add(summary["benchmark_3_temporal_coherence"]["interpretation"])
    add("")
    add("---")
    add("")

    # ── Benchmark 4 ──────────────────────────────────────────────────────────
    add("## Benchmark 4 — Evidence review")
    add("")
    add("Each flagged window classified by what its flag rests on, using the "
        "existing evidence objects. Rules:")
    add("")
    for name, rule in REVIEW_RULES.items():
        add(f"- **{name}** — {rule}")
    add("")
    add("### OBSERVED RESULTS")
    add("")
    add(f"- physiologically supported: **{reviewed['physiologically_supported']}**")
    add(f"- mainly data-quality related: **{reviewed['mainly_data_quality']}**")
    add(f"- ambiguous: **{reviewed['ambiguous']}**")
    add("")
    add(_table(review, ["caseid", "window_index", "anomaly_rank",
                        "min_coverage_pct", "evidence_drivers", "review"]))
    add("")
    add("### INTERPRETATION")
    add("")
    add(reviewed["interpretation"])
    add("")
    add(f"> {reviewed['not_clinical_validation']}")
    add("")
    add("---")
    add("")
    add("## What this benchmark does not establish")
    add("")
    for item in summary["limitations"]:
        add(f"- {item}")
    add("")
    return "\n".join(out)
