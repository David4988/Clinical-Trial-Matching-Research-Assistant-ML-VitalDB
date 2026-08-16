"""Build the v1 feature table and print the verification report.

Reads the 5-minute aggregated CSVs, builds the compact 28-column feature table,
writes it to results/features/, and prints the evidence needed to trust it:
schema, example rows, row and null counts, and two leakage checks that are
re-run against the real data every time rather than assumed from the tests.

    python feature_checkpoint.py
"""

import sys

import pandas as pd

sys.path.insert(0, "src")

from vitaldb_audit import features  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    rule("FEATURE SCHEMA v1 — BUILD")

    frames = features.load_aggregated_frames(window_minutes=5.0)
    caseids = [int(f["caseid"].iloc[0]) for f in frames]
    print(f"input   : {len(frames)} cases {caseids} of 5-minute aggregated windows")
    print(f"windows : {sum(len(f) for f in frames)} in, "
          f"coverage bar {features.USABLE_COVERAGE_FRACTION:.0%}")

    table = features.build_feature_table(frames)

    features.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = features.FEATURES_DIR / "feature_table_5min.csv"
    table.to_csv(out_path, index=False)
    print(f"output  : {out_path}")

    # ── Schema ───────────────────────────────────────────────────────────────
    rule(f"FINAL COLUMNS ({len(table.columns)})")
    groups = {
        "identity     ": ["caseid", "window_index", "window_start_s", "window_end_s"],
        "current state": [f"{s}_{k}" for s in features.CORE_SIGNALS
                          for k in features.STATS],
        "change       ": [f"{s}_delta" for s in features.CORE_SIGNALS],
        "data quality ": ([f"{s}_coverage_pct" for s in features.CORE_SIGNALS]
                          + [f"{s}_usable" for s in features.CORE_SIGNALS]
                          + ["n_core_usable", "window_usable"]),
        "temporal     ": ["consecutive_usable_windows"],
    }
    for name, cols in groups.items():
        print(f"  {name} ({len(cols):>2})  {', '.join(cols)}")
    assert sum(len(c) for c in groups.values()) == len(table.columns)

    # ── Row counts ───────────────────────────────────────────────────────────
    rule("ROW COUNTS")
    print(f"  total rows              : {len(table)}")
    print(f"  usable windows          : {int(table['window_usable'].sum())}")
    print(f"  unusable windows (kept) : {int((~table['window_usable']).sum())}")
    print()
    per_case = table.groupby("caseid").agg(
        windows=("window_index", "size"),
        usable=("window_usable", "sum"),
        minutes=("window_end_s", lambda s: round(s.max() / 60.0, 1)),
        hr_delta_present=("hr_delta", "count"),
    )
    print(per_case.to_string())

    # ── Example rows ─────────────────────────────────────────────────────────
    rule("EXAMPLE ROWS")
    show = ["caseid", "window_index", "hr_mean", "hr_delta", "spo2_mean",
            "spo2_delta", "rr_mean", "rr_delta", "hr_coverage_pct",
            "spo2_coverage_pct", "rr_coverage_pct", "n_core_usable",
            "window_usable", "consecutive_usable_windows"]

    print("\nFirst 5 windows of the first case (note: window 0 has null deltas —")
    print("there is no previous window to reference):\n")
    print(table[table["caseid"] == caseids[0]].head(5)[show].to_string(index=False))

    unusable = table[~table["window_usable"].fillna(False)]
    if not unusable.empty:
        pick = unusable.iloc[0]
        case_rows = table[table["caseid"] == pick["caseid"]]
        window = int(pick["window_index"])
        neighbourhood = case_rows[
            case_rows["window_index"].between(window - 1, window + 2)
        ]
        print(f"\nA real unusable window (case {int(pick['caseid'])}, "
              f"window {window}) and its neighbours — the row survives, its")
        print("deltas go null, and the counter resets:\n")
        print(neighbourhood[show].to_string(index=False))

    # ── Nulls ────────────────────────────────────────────────────────────────
    rule("NULL COUNTS (null = genuinely absent, never filled)")
    nulls = features.null_report(table)
    print(nulls.to_string(index=False))
    print(f"\n  columns with zero nulls : "
          f"{int((nulls['nulls'] == 0).sum())} / {len(nulls)}")

    # ── Verification ─────────────────────────────────────────────────────────
    rule("VERIFICATION")

    future = features.verify_no_future_information(frames)
    print(f"  [PASS] no future information used")
    print(f"         {future['rows_rebuilt_from_truncated_history']} rows rebuilt "
          f"from truncated history (windows 0..k only); all identical to the")
    print(f"         rows built from the full case.")

    cross = features.verify_no_cross_case_leakage(frames)
    print(f"\n  [PASS] previous-window references never cross a case boundary")
    print(f"         {cross['cases_checked']} cases identical alone vs. in the "
          f"combined table; every case's first window has null deltas.")

    zero_filled = {
        f"{s}_delta": int((table[f"{s}_delta"] == 0).sum())
        for s in features.CORE_SIGNALS
    }
    print(f"\n  [INFO] exact-zero deltas (should be rare and real, not fill): "
          f"{zero_filled}")

    unusable_with_delta = 0
    for signal in features.CORE_SIGNALS:
        mask = (~table[f"{signal}_usable"].fillna(False)) & table[f"{signal}_delta"].notna()
        unusable_with_delta += int(mask.sum())
    status = "PASS" if unusable_with_delta == 0 else "FAIL"
    print(f"\n  [{status}] no delta computed from an unusable signal "
          f"({unusable_with_delta} violations)")

    rule("NEXT")
    print("  Feature table is ready.  Next milestone: MODEL -> RESULT.")
    print("  Stopping here as scoped — no anomaly detection in this task.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
