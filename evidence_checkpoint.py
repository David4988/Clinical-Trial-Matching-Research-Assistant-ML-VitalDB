"""Build the evidence objects for Model B's flagged windows.

Reads the v1 feature table and the Model B results, writes
results/xai/evidence_cases.json, and prints a compact readable report for the
top 10 flagged windows.

No prompt is written and no model is called. This stage only prepares evidence.

    python evidence_checkpoint.py [--top 10]
"""

import argparse
import json
import sys

import pandas as pd

sys.path.insert(0, "src")

from vitaldb_audit import evidence, features  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--k", type=float, default=evidence.CHANGE_K,
                        help="statistical band for the change flags")
    args = parser.parse_args(argv)

    rule("EVIDENCE PREPARATION — MODEL B FLAGGED WINDOWS")

    table = pd.read_csv(features.FEATURES_DIR / "feature_table_5min.csv")
    results = pd.read_csv(evidence.MODEL_B_RESULTS)
    print(f"  feature table : {len(table)} rows")
    print(f"  model B       : {len(results)} analyzed, "
          f"{int(results['anomaly_label'].sum())} flagged")
    print(f"  change rule   : |delta / pooled_std| > {args.k}   "
          f"(statistical band, not a clinical threshold)")

    document = evidence.build_document(table, results, k=args.k)
    json_path = evidence.write_document(document)

    report = evidence.render_report(document, n=args.top)
    report_path = evidence.write_report(report)

    # ── Readable report ──────────────────────────────────────────────────────
    print()
    print(report)

    # ── Structural check on the emitted file ─────────────────────────────────
    rule("EVIDENCE FILE")
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    entries = reloaded["evidence"]
    print(f"  {json_path}")
    print(f"  {report_path}")
    print(f"\n  schema_version   : {reloaded['schema_version']}")
    print(f"  evidence objects : {len(entries)}")
    print(f"  parsed back from disk with the standard json module (no custom")
    print(f"  encoder needed), so every value is a plain JSON type.")

    required = ["case_id", "window_index", "time_range", "anomaly_score",
                "anomaly_rank", "signals", "n_core_signals_usable",
                "window_usable", "observations"]
    missing = [k for k in required if any(k not in e for e in entries)]
    print(f"\n  required top-level fields present in every object : "
          f"{'yes' if not missing else 'NO -> ' + str(missing)}")

    signal_fields = ["current_mean", "previous_usable_mean", "delta", "std",
                     "min", "max", "coverage_pct"]
    signal_ok = all(
        field in entry["signals"][signal]
        for entry in entries for signal in evidence.SIGNALS for field in signal_fields
    )
    print(f"  all 7 per-signal fields present for HR/SpO2/RR    : "
          f"{'yes' if signal_ok else 'NO'}")

    flags = ["hr_changed", "spo2_changed", "rr_changed", "multiple_signals_changed"]
    flags_ok = all(f in e["observations"] for e in entries for f in flags)
    print(f"  observation flags present in every object         : "
          f"{'yes' if flags_ok else 'NO'}")

    # ── Observation roll-up ──────────────────────────────────────────────────
    rule("OBSERVATION SUMMARY ACROSS ALL FLAGGED WINDOWS")
    frame = pd.DataFrame([
        {
            "case": e["case_id"],
            "window": e["window_index"],
            "rank": e["anomaly_rank"],
            "hr_changed": e["observations"]["hr_changed"],
            "spo2_changed": e["observations"]["spo2_changed"],
            "rr_changed": e["observations"]["rr_changed"],
            "n_changed": e["observations"]["n_signals_changed"],
            "multiple": e["observations"]["multiple_signals_changed"],
        }
        for e in entries
    ])
    print(frame.to_string(index=False))
    print(f"\n  windows where multiple signals changed : "
          f"{int(frame['multiple'].sum())} of {len(frame)}")
    for signal in evidence.SIGNALS:
        column = f"{signal}_changed"
        changed = int(frame[column].fillna(False).sum())
        unknown = int(frame[column].isna().sum())
        print(f"  {signal:<5} changed in {changed:>2} of {len(frame)} windows"
              + (f"  ({unknown} not computable)" if unknown else ""))

    rule("NEXT")
    print("  Evidence objects are ready for an XAI/LLM layer to consume.")
    print("  Stopping here as scoped — no prompt written, no model called.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
