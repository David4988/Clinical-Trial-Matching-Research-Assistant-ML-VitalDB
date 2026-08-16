"""Monitoring-window aggregation checkpoint.

Aggregates the four inspected cases into fixed monitoring windows and prints
the A-I review report.  Stops there: no anomaly detection, no modelling.

Run:  .venv/bin/python aggregate_checkpoint.py
      .venv/bin/python aggregate_checkpoint.py --window 5
      .venv/bin/python aggregate_checkpoint.py --window 30
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd

from vitaldb_audit import aggregate, config, fetch
from vitaldb_audit.logging_setup import setup_logging

RULE = "=" * 104
DASH = "-" * 104

CASES = [2, 4, 8, 9]


def _fmt(value, dash="—"):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return dash
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=float, default=aggregate.WINDOW_MINUTES,
                        help="window width in minutes (default: 10)")
    parser.add_argument("--cases", type=int, nargs="+", default=CASES)
    args = parser.parse_args()

    setup_logging(verbose=False)
    tables, _ = fetch.fetch_metadata(force_refresh=False)

    bundle = aggregate.run_experiment(
        args.cases, tables.cases, tables.trks, window_minutes=args.window
    )
    reports = bundle["cases"]
    short_of = {t: s["short"] for t, s in aggregate.SIGNAL_SPECS.items()}

    print(f"\n{RULE}")
    print(f"MONITORING-WINDOW AGGREGATION CHECKPOINT — window = {args.window:g} min")
    print(f"expected interval {bundle['expected_interval_s']:g}s (~0.5 Hz, empirically probed)  |  "
          f"gap rule {bundle['gap_factor']}x median dt")
    print(f"interpolation={bundle['interpolation']}  fill={bundle['fill']}")
    print(RULE)

    # ── A ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nA. NUMBER OF WINDOWS PER CASE\n{RULE}")
    print(f"{'case':>6} {'case_min':>10} {'windows':>9} {'full_windows':>13} {'tail_window_min':>17}")
    print(DASH)
    for rep in reports:
        frame = pd.read_csv(rep["csv"])
        tail = frame["window_width_min"].iloc[-1]
        full = int((frame["window_width_min"] >= args.window).sum())
        print(f"{rep['caseid']:>6} {rep['case_duration_min']:>10.2f} "
              f"{rep['n_windows']:>9} {full:>13} {tail:>17.2f}")

    # ── B ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nB. COVERAGE DISTRIBUTION\n{RULE}")
    print("coverage = observations / expected, per window. Computed, never assumed.")
    print(DASH)
    print(f"{'case':>6} {'signal':<22} {'mean%':>8} {'median%':>9} {'min%':>8} {'max%':>8}")
    print(DASH)
    for rep in reports:
        for tname, entry in rep["stats"]["per_signal"].items():
            avail = rep["signals"].get(tname, {}).get("available", False)
            label = tname if avail else f"{tname} (ABSENT)"
            print(f"{rep['caseid']:>6} {label:<22} {entry['coverage_mean_pct']:>8} "
                  f"{entry['coverage_median_pct']:>9} {entry['coverage_min_pct']:>8} "
                  f"{entry['coverage_max_pct']:>8}")

    # ── C ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nC. COMPLETELY MISSING WINDOWS (zero observations)\n{RULE}")
    print("Retained in the output with null statistics — never silently dropped.")
    print(DASH)
    print(f"{'case':>6} {'signal':<22} {'empty_windows':>14} {'of_total':>9}")
    print(DASH)
    for rep in reports:
        for tname, entry in rep["stats"]["per_signal"].items():
            print(f"{rep['caseid']:>6} {tname:<22} {entry['n_windows_empty']:>14} "
                  f"{rep['n_windows']:>9}")
    print(DASH)
    for rep in reports:
        print(f"  case {rep['caseid']}: {rep['stats']['n_windows_all_core_missing']} "
              f"window(s) with ALL THREE core signals missing")

    # ── D ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nD. LOW-COVERAGE WINDOWS (<50%, excluding wholly empty)\n{RULE}")
    print(f"{'case':>6} {'signal':<22} {'low_coverage_windows':>22}")
    print(DASH)
    for rep in reports:
        for tname, entry in rep["stats"]["per_signal"].items():
            print(f"{rep['caseid']:>6} {tname:<22} {entry['n_windows_low_coverage']:>22}")

    # ── E ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nE. LARGEST GAPS\n{RULE}")
    print("Gaps are measured and reported. None are bridged or filled.")
    print(DASH)
    print(f"{'case':>6} {'signal':<22} {'largest_gap_s':>15} {'total_gap_s':>13}")
    print(DASH)
    for rep in reports:
        for tname, entry in rep["stats"]["per_signal"].items():
            if not rep["signals"].get(tname, {}).get("available"):
                continue
            print(f"{rep['caseid']:>6} {tname:<22} {entry['largest_gap_seconds']:>15} "
                  f"{entry['total_gap_seconds']:>13}")

    # ── F ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nF. RR_CO2 EXACT-ZERO OBSERVATIONS\n{RULE}")
    print("Zeros are retained as valid observations and included in all statistics.")
    print(DASH)
    print(f"{'case':>6} {'zero_obs':>10} {'total_obs':>11} {'pct_zero':>10} {'windows_with_zeros':>20}")
    print(DASH)
    for rep in reports:
        frame = pd.read_csv(rep["csv"])
        zeros = int(frame["rr_co2_zero_count"].sum())
        total = int(frame["rr_co2_observation_count"].sum())
        n_win = int((frame["rr_co2_zero_count"] > 0).sum())
        pct = round(zeros / total * 100, 2) if total else 0.0
        print(f"{rep['caseid']:>6} {zeros:>10} {total:>11} {pct:>10} {n_win:>20}")

    # ── G ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nG. ART_MBP AVAILABILITY (optional signal)\n{RULE}")
    print(f"{'case':>6} {'available':>11} {'raw_rows':>10} {'note':<50}")
    print(DASH)
    for rep in reports:
        meta = rep["signals"].get("Solar8000/ART_MBP", {})
        avail = meta.get("available", False)
        print(f"{rep['caseid']:>6} {str(avail):>11} {_fmt(meta.get('n_raw_rows'), '—'):>10} "
              f"{meta.get('reason', 'aggregated normally'):<50}")

    # ── H ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nH. GENERATED PLOTS\n{RULE}")
    for rep in reports:
        print(f"\ncase {rep['caseid']}:")
        print(f"   raw         {rep['plots']['raw']}")
        print(f"   aggregated  {rep['plots']['aggregated']}")
        print(f"   coverage    {rep['plots']['coverage']}")
        print(f"   csv         {rep['csv']}")
    print(f"\nSummary JSON: {config.RESULTS_DIR / 'aggregation_summary.json'}")

    # ── I ────────────────────────────────────────────────────────────────────
    print(f"\n{RULE}\nI. EXAMPLE AGGREGATED ROWS\n{RULE}")
    example = reports[1] if len(reports) > 1 else reports[0]
    frame = pd.read_csv(example["csv"])
    cols = [
        "window_index", "window_start_min", "window_end_min",
        "hr_mean", "hr_min", "hr_max", "hr_std", "hr_coverage_percent",
        "spo2_mean", "spo2_min", "rr_co2_mean", "rr_co2_zero_count",
        "art_mbp_mean", "window_data_unavailable",
    ]
    cols = [c for c in cols if c in frame.columns]
    print(f"case {example['caseid']} — first 8 windows "
          f"(all {len(frame)} in {example['csv']}):\n")
    print(frame[cols].head(8).to_string(index=False))

    print(f"\n{RULE}")
    print("STOPPING AFTER AGGREGATION CHECKPOINT — awaiting manual review.")
    print("No anomaly detection, no Isolation Forest, no feature engineering "
          "beyond window statistics, no ML.")
    print(RULE)


if __name__ == "__main__":
    main()
