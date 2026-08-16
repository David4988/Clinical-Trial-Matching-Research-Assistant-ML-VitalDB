"""Checkpoint verification script: fetch → profile → classify → select → probe.

Runs against the live VitalDB API. Prints the mandatory verification report,
then stops.
"""

import json
import sys
import os

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vitaldb_audit import config, fetch, profile, signals, select, probe
from vitaldb_audit.logging_setup import setup_logging

RULE = "=" * 78
DASH = "-" * 78


def main():
    logger = setup_logging(verbose=False)

    # ── Stage 1: Fetch ──────────────────────────────────────────────────
    print(f"\n{RULE}\nSTAGE 1: FETCHING METADATA FROM LIVE VITALDB API\n{RULE}")
    tables, records = fetch.fetch_metadata(force_refresh=False)

    # ── Stage 2: Profile ────────────────────────────────────────────────
    print(f"\n{RULE}\nSTAGE 2: PROFILING\n{RULE}")
    track_inv = profile.build_track_inventory(tables.trks)
    clinical_miss = profile.profile_clinical_missingness(tables.cases)
    dur_stats = profile.compute_duration_stats(tables.cases)
    lab_inv = profile.build_lab_inventory(tables.labs)
    tpc_stats = profile.tracks_per_case_stats(tables.trks)

    # ── Stage 3: Classify ───────────────────────────────────────────────
    print(f"\n{RULE}\nSTAGE 3: CLASSIFYING PHYSIOLOGICAL TRACKS\n{RULE}")
    candidates = signals.build_physiological_candidates(track_inv)
    coverage = signals.category_coverage(candidates)
    unmatched = signals.unmatched_suffixes(track_inv)

    # ── Stage 4: Select ─────────────────────────────────────────────────
    print(f"\n{RULE}\nSTAGE 4: SELECTING CANDIDATE CASES\n{RULE}")
    case_tracks = select.build_case_track_sets(tables.trks)
    scored = select.score_cases(case_tracks, tables.cases)
    selected_ids = select.select_candidates(scored, n=config.N_CANDIDATE_CASES)

    # ── Stage 5: Probe ──────────────────────────────────────────────────
    print(f"\n{RULE}\nSTAGE 5: EMPIRICAL SAMPLING PROBE (selected cases only)\n{RULE}")
    probe_results = probe.probe_cases(
        tables.trks, selected_ids, config.REQUIRED_PANEL
    )

    # ════════════════════════════════════════════════════════════════════
    #  VERIFICATION REPORT
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'#' * 78}")
    print(f"#  CHECKPOINT VERIFICATION REPORT")
    print(f"{'#' * 78}")

    # ── A. Dataset Shapes ───────────────────────────────────────────────
    print(f"\n{RULE}\nA. DATASET SHAPES\n{RULE}")
    for rec in records:
        print(f"  {rec['name']:>6s}: {rec.get('n_rows', '?'):>8} rows × {rec.get('n_cols', '?'):>3} cols  ({rec['n_bytes']:,} bytes, sha256={rec['sha256'][:16]}...)")
    print(f"\n  Distinct track names: {len(track_inv)}")
    print(f"  Devices: {sorted(track_inv['device'].unique())}")
    print(f"\n  Duration stats (case_minutes):")
    cm = dur_stats.get("case_minutes", {})
    for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        print(f"    {k:>6s}: {cm.get(k, 'N/A')}")
    print(f"\n  Tracks per case:")
    for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        print(f"    {k:>6s}: {tpc_stats.get(k, 'N/A')}")

    # ── Ground truth comparison ─────────────────────────────────────────
    print(f"\n{DASH}\n  Ground Truth Comparison\n{DASH}")
    gt = {
        "cases rows": (6388, records[0].get("n_rows")),
        "cases cols": (74, records[0].get("n_cols")),
        "trks rows": (486449, records[1].get("n_rows")),
        "trks cols": (3, records[1].get("n_cols")),
        "labs rows": (928448, records[2].get("n_rows")),
        "labs cols": (4, records[2].get("n_cols")),
        "distinct tracks": (196, len(track_inv)),
    }
    for label, (expected, actual) in gt.items():
        match = "✓" if expected == actual else "✗ MISMATCH"
        print(f"  {label:>20s}: expected={expected}, actual={actual}  {match}")

    # ── B. Candidate Selection ──────────────────────────────────────────
    print(f"\n{RULE}\nB. CANDIDATE SELECTION\n{RULE}")
    n_full_panel = int(scored["has_full_panel"].sum())
    n_qualifying = int(scored["qualifies"].sum())
    print(f"  Full-panel cases: {n_full_panel}")
    print(f"  Qualifying (panel + duration band {config.MIN_CASE_MINUTES}-{config.MAX_CASE_MINUTES} min): {n_qualifying}")
    print(f"  Selected case IDs: {selected_ids}")

    selected_scores = scored[scored["caseid"].isin(selected_ids)].sort_values("caseid")
    print(f"\n  {'caseid':>8s}  {'n_tracks':>8s}  {'n_physio':>8s}  {'minutes':>10s}  {'qualifies':>9s}")
    print(f"  {DASH[:55]}")
    for _, r in selected_scores.iterrows():
        print(f"  {int(r['caseid']):>8d}  {int(r['n_tracks']):>8d}  {int(r['n_physio_categories']):>8d}  {r['case_minutes']:>10.2f}  {bool(r['qualifies'])!s:>9s}")

    # ── C. Track Coverage ───────────────────────────────────────────────
    print(f"\n{RULE}\nC. TRACK COVERAGE (selected cases)\n{RULE}")
    panel_set = set(config.REQUIRED_PANEL)
    for cid in selected_ids:
        case_trk_set = case_tracks.get(cid, set())
        panel_present = panel_set.intersection(case_trk_set)
        panel_missing = panel_set - case_trk_set
        physio_trks = [t for t in sorted(case_trk_set) if signals.classify_track(t) is not None]
        print(f"\n  Case {cid}:")
        print(f"    Total tracks: {len(case_trk_set)}")
        print(f"    Panel coverage: {len(panel_present)}/{len(panel_set)} ({', '.join(sorted(panel_missing)) if panel_missing else 'complete'})")
        print(f"    Physiological tracks: {len(physio_trks)}")
        for t in physio_trks:
            cat = signals.classify_track(t)
            print(f"      - {t}  [{cat}]")

    # ── D. Observed Sampling ────────────────────────────────────────────
    print(f"\n{RULE}\nD. OBSERVED SAMPLING (empirical probe results)\n{RULE}")
    if probe_results.empty:
        print("  No probe results.")
    else:
        print(f"\n  {'caseid':>6s}  {'tname':<24s}  {'obs_hz':>8s}  {'median_dt':>10s}  {'kind':<10s}  {'rows':>6s}  {'bytes':>8s}  {'error'}")
        print(f"  {DASH}")
        for _, r in probe_results.iterrows():
            hz = f"{r['observed_sampling_hz']:.2f}" if r['observed_sampling_hz'] is not None and not (isinstance(r['observed_sampling_hz'], float) and r['observed_sampling_hz'] != r['observed_sampling_hz']) else "N/A"
            dt = f"{r['median_dt_s']:.6f}" if r['median_dt_s'] is not None and not (isinstance(r['median_dt_s'], float) and r['median_dt_s'] != r['median_dt_s']) else "N/A"
            kind = r.get('observed_track_kind', 'N/A') or 'N/A'
            err = r.get('error', '') or ''
            print(f"  {int(r['caseid']):>6d}  {r['tname']:<24s}  {hz:>8s}  {dt:>10s}  {kind:<10s}  {int(r['n_rows_read']):>6d}  {int(r['n_bytes_read']):>8d}  {err}")

        # Cross-case summary
        ok = probe_results[probe_results["error"].isna()]
        if not ok.empty:
            print(f"\n  Cross-case frequency consistency:")
            for tname, group in ok.groupby("tname"):
                hz_vals = group["observed_sampling_hz"].values
                hz_min, hz_max = hz_vals.min(), hz_vals.max()
                consistent = "consistent" if hz_min == hz_max else f"VARIES ({hz_min:.2f}-{hz_max:.2f})"
                print(f"    {tname:<24s}: {hz_vals[0]:.2f} Hz across {len(group)} cases — {consistent}")

        n_errors = len(probe_results[probe_results["error"].notna()])
        print(f"\n  Total probed: {len(probe_results)}, successful: {len(ok)}, errors: {n_errors}")
        print(f"  Sampling rate source: {probe_results.iloc[0].get('sampling_rate_source', 'N/A')}")

    # ── E. Unknowns / Limitations ───────────────────────────────────────
    print(f"\n{RULE}\nE. UNKNOWNS / LIMITATIONS\n{RULE}")
    unknowns = [
        "Intra-track missingness/gaps: /trks records only that a track exists, "
        "never how many samples it holds or where its gaps are.",
        "Signal quality and artifact burden: not assessable from metadata alone.",
        "Whether a track is usable throughout the entire case duration.",
        "Clinical validity of observed anomalies in signal values.",
        "Lab/signal clock alignment: lab timestamps (labs.dt) vs intra-operative signal clock.",
        f"Unmatched configured signal suffixes: {json.dumps(unmatched) if unmatched else 'none (all matched)'}.",
    ]
    for i, u in enumerate(unknowns, 1):
        print(f"  {i}. {u}")

    print(f"\n{'#' * 78}")
    print("CHECKPOINT REACHED — awaiting manual inspection before further implementation.")
    print(f"{'#' * 78}\n")


if __name__ == "__main__":
    main()
