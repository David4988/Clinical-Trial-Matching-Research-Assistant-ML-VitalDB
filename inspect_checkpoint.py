"""Selected-case signal inspection checkpoint (inspection stage only).

Loads four core physiological tracks for four hand-picked cases, measures them,
plots them, and prints the A-G review report.  Stops there: no anomaly
detection, no modelling, no preprocessing decisions taken.

Run:  .venv/bin/python inspect_checkpoint.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from vitaldb_audit import config, fetch, inspect_signals
from vitaldb_audit.logging_setup import setup_logging

RULE = "=" * 100
DASH = "-" * 100

CASES = [2, 4, 8, 9]
SIGNALS = inspect_signals.CORE_SIGNALS


def _f(value, suffix="", dash="—"):
    """Format a possibly-None measurement without inventing a value."""
    return f"{value}{suffix}" if value is not None else dash


def main():
    setup_logging(verbose=False)

    tables, _ = fetch.fetch_metadata(force_refresh=False)
    bundle = inspect_signals.inspect_cases(
        CASES, tables.cases, tables.trks, signals=SIGNALS
    )
    reports = bundle["cases"]

    # ── A ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nA. CASE SUMMARY\n{RULE}")
    print(f"{'case':>6} {'duration_min':>13} {'tracks_found':>13} {'tracks_missing':>15}  missing_names")
    print(DASH)
    for rep in reports:
        found = [n for n, s in rep["signals"].items() if s["available"]]
        missing = [n for n, s in rep["signals"].items() if not s["available"]]
        print(f"{rep['caseid']:>6} {rep['case_duration_min']:>13.2f} "
              f"{len(found):>13} {len(missing):>15}  {', '.join(missing) or '—'}")
    print(f"\nDataset version {bundle['dataset_version']}  |  "
          f"load strategy: {bundle['load_strategy']}")
    print(f"interpolation={bundle['interpolation']}  resampling={bundle['resampling']}  "
          f"gap_factor={bundle['gap_factor']}x median dt")

    # ── B ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nB. SIGNAL COVERAGE\n{RULE}")
    print("span_cov%   = (last_ts - first_ts) / case_duration      -> does it reach end to end?")
    print("density%    = present_values / (case_duration / dt)     -> are the samples all there?")
    print(DASH)
    print(f"{'case':>5} {'track':<22} {'rows':>7} {'first_ts':>10} {'last_ts':>10} "
          f"{'obs_min':>9} {'case_min':>9} {'span%':>8} {'density%':>9}")
    print(DASH)
    for rep in reports:
        for tname, s in rep["signals"].items():
            if not s["available"]:
                print(f"{rep['caseid']:>5} {tname:<22} {'NOT RECORDED FOR THIS CASE':>60}")
                continue
            print(f"{rep['caseid']:>5} {tname:<22} {s['n_rows']:>7} "
                  f"{_f(s['first_time_s']):>10} {_f(s['last_time_s']):>10} "
                  f"{_f(s['observed_duration_min']):>9} {_f(s['case_duration_min']):>9} "
                  f"{_f(s['span_coverage_pct']):>8} {_f(s['sample_coverage_pct']):>9}")

    # ── C ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nC. GAP / MISSINGNESS SUMMARY\n{RULE}")
    print("These are two DIFFERENT failure modes and are counted separately:")
    print("  NaN values   = a row exists at that timestamp but carries no value")
    print("  timestamp gap = no row exists at all for a stretch of time (> "
          f"{inspect_signals.GAP_FACTOR}x median dt)")
    print(DASH)
    print(f"{'case':>5} {'track':<22} {'dt_med':>8} {'dt_min':>8} {'dt_max':>9} "
          f"{'NaN':>6} {'NaN%':>7} {'gaps':>6} {'largest_gap_s':>14} {'time_in_gaps%':>14}")
    print(DASH)
    for rep in reports:
        for tname, s in rep["signals"].items():
            if not s["available"]:
                continue
            print(f"{rep['caseid']:>5} {tname:<22} {_f(s['median_dt_s']):>8} "
                  f"{_f(s['min_dt_s']):>8} {_f(s['max_dt_s']):>9} "
                  f"{s['n_values_missing']:>6} {s['pct_values_missing']:>7} "
                  f"{s['n_gaps']:>6} {s['largest_gap_s']:>14} {s['pct_time_in_gaps']:>14}")

    # ── D ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nD. SIGNAL STATISTICS\n{RULE}")
    print("Computed on present values only. Nothing clipped, filled, or artifact-removed.")
    print(DASH)
    print(f"{'case':>5} {'track':<22} {'min':>9} {'median':>9} {'max':>9} "
          f"{'n_present':>10} {'obs_Hz':>8}")
    print(DASH)
    for rep in reports:
        for tname, s in rep["signals"].items():
            if not s["available"]:
                continue
            print(f"{rep['caseid']:>5} {tname:<22} {_f(s['value_min']):>9} "
                  f"{_f(s['value_median']):>9} {_f(s['value_max']):>9} "
                  f"{s['n_values_present']:>10} {_f(s['observed_sampling_hz']):>8}")

    # ── E ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nE. PLOT PATHS\n{RULE}")
    for rep in reports:
        print(f"\ncase {rep['caseid']}:")
        for tname, path in rep["plots"].items():
            print(f"   {tname:<22} {path}")
        print(f"   {'SYNCHRONIZED PANEL':<22} {rep['synchronized_plot']}")
    print(f"\nMachine-readable bundle: {config.RESULTS_DIR / 'signal_inspection.json'}")

    # ── F ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nF. DATA QUALITY OBSERVATIONS\n{RULE}")
    print("Descriptive only. No clinical inference is drawn from any value below.")
    print(DASH)
    any_flag = False
    for rep in reports:
        for tname, s in rep["signals"].items():
            for flag in s["flags"]:
                any_flag = True
                print(f"  case {rep['caseid']:>4}  {tname:<22} {flag}")
    if not any_flag:
        print("  (none)")
    print("\n  CARRIED FORWARD FROM THE RECON CHECKPOINT (untouched, as instructed):")
    print("    profile.compute_duration_stats -> anesthesia_minutes is corrupt upstream.")
    print("    Case 4476 has aneend = -3.69e9, giving a -61,524,471 minute span, which")
    print("    drags that metric's mean to -9431.65 and std to 769,780.")
    print("    case_minutes is unaffected and reproduces ground truth exactly.")

    # ── G ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nG. PREPROCESSING QUESTIONS THAT MUST BE DECIDED NEXT\n{RULE}")
    for i, q in enumerate(PREPROCESSING_QUESTIONS, 1):
        print(f"\n{i}. {q['q']}")
        print(f"   observed: {q['obs']}")
        print(f"   options : {q['opts']}")

    print(f"\n{RULE}")
    print("STOPPING AFTER INSPECTION STAGE — awaiting manual review.")
    print("No anomaly detection, no resampling, no imputation, no modelling performed.")
    print(RULE)


PREPROCESSING_QUESTIONS = [
    {
        "q": "Sentinel zeros vs. true physiological zeros — how should exact 0 be treated?",
        "obs": "Solar8000 numerics use 0 as a sensor-disconnect sentinel; it is "
               "indistinguishable from a real reading by dtype alone.",
        "opts": "(a) treat 0 as missing for all four tracks; (b) per-track rule "
                "(0 impossible for HR/SPO2/MBP, arguably possible for RR_CO2 during apnoea); "
                "(c) keep raw and carry a companion validity mask.",
    },
    {
        "q": "Common time base — resample, or keep the native irregular grid?",
        "obs": "Median dt is ~2 s but jitters, and the four tracks do not share "
               "timestamps, so they cannot be column-joined without a decision.",
        "opts": "(a) no resampling, align by nearest-timestamp tolerance join; "
                "(b) fixed 2 s grid with last-observation-carried-forward and an explicit "
                "staleness cap; (c) fixed grid leaving gaps as NaN.",
    },
    {
        "q": "Gap policy — what is the maximum gap that may be bridged, and how?",
        "obs": "Gaps are measured here but never filled. GAP_FACTOR=3x median dt "
               "is the current detection threshold, which is a detection choice, not a fill policy.",
        "opts": "(a) never bridge; (b) bridge <= N seconds by carry-forward and mark "
                "the bridged samples; (c) split each case into contiguous segments and "
                "analyse segments independently.",
    },
    {
        "q": "Analysis window — whole case, or the anaesthesia/operation sub-interval?",
        "obs": "Tracks routinely start after casestart and end before caseend, so "
               "span coverage is below 100% even for healthy tracks. anestart/aneend is "
               "the clinically meaningful window but is the corrupt column noted in F.",
        "opts": "(a) whole case (casestart->caseend); (b) opstart->opend; "
                "(c) anestart->aneend once the corrupt rows are excluded.",
    },
    {
        "q": "Out-of-range values — retain, mask, or exclude?",
        "obs": "Plausibility bands are currently used for display and flagging only; "
               "nothing is clipped.",
        "opts": "(a) retain raw and let downstream models see artifacts; "
                "(b) mask out-of-band values as missing; (c) retain but emit a "
                "per-sample quality channel.",
    },
    {
        "q": "Case inclusion — is a case usable when a core track is entirely absent?",
        "obs": "Only case 4 of the four carries all four core tracks; ART_MBP is "
               "absent for cases 2, 8 and 9.",
        "opts": "(a) require the full four-track core; (b) allow a reduced panel with "
                "an explicit feature-availability flag; (c) impute nothing and let the "
                "model handle absence natively.",
    },
]


if __name__ == "__main__":
    main()
