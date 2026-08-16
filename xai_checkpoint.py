"""Generate structured explanations for all Model B flagged windows.

Reads the evidence objects from results/xai/evidence_cases.json, runs the
deterministic explain_window() function on each, and writes:

    results/xai/explanations.json       — structured explanation objects
    results/xai/explanations_report.txt  — human-readable text report

The explainer is a TRANSLATION LAYER.  It describes what the evidence shows.
It does not reconstruct the Isolation Forest's internal decision path and it
does not produce clinical interpretations.

    python xai_checkpoint.py [--top 10]
"""

import argparse
import json
import sys
import textwrap

sys.path.insert(0, "src")

from vitaldb_audit import explain  # noqa: E402
from vitaldb_audit import config   # noqa: E402

XAI_DIR = config.RESULTS_DIR / "xai"
EVIDENCE_PATH = XAI_DIR / "evidence_cases.json"
EXPLANATIONS_PATH = XAI_DIR / "explanations.json"
REPORT_PATH = XAI_DIR / "explanations_report.txt"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ── Render one explanation as a readable block ───────────────────────────────


def render_explanation(exp: dict) -> str:
    """One explanation as a compact human-readable block."""
    lines = []

    header = (
        f"#{exp['anomaly_rank']}  case {exp['case_id']}  "
        f"window {exp['window_index']}  |  {exp['time_label']}  |  "
        f"score {exp['anomaly_score']:+.4f}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    # Headline
    lines.append(f"  HEADLINE: {exp['headline']}")
    lines.append("")

    # Driver summary
    dtype = exp["driver_type"]
    drivers = ", ".join(exp["primary_drivers"]) if exp["primary_drivers"] else "none"
    lines.append(f"  driver type : {dtype}")
    lines.append(f"  drivers     : {drivers}")
    lines.append("")

    # Per-signal narratives
    for signal in explain.SIGNALS:
        label = explain.SIGNAL_LABELS[signal]
        narrative = exp["signal_narratives"][signal]
        wrapped = textwrap.fill(narrative, width=74, initial_indent="    ",
                                subsequent_indent="    ")
        lines.append(f"  {label}:")
        lines.append(wrapped)

    # Supporting detail
    if exp["supporting_detail"]:
        lines.append("")
        lines.append("  Notes:")
        for note in exp["supporting_detail"]:
            lines.append(f"    • {note}")

    # Data quality
    if exp["data_quality_notes"]:
        lines.append("")
        lines.append("  Data quality:")
        for note in exp["data_quality_notes"]:
            lines.append(f"    • {note}")

    return "\n".join(lines)


# ── Full report ──────────────────────────────────────────────────────────────


def render_report(explanations: list[dict], n: int | None = None) -> str:
    """Render the full text report."""
    subset = explanations[:n] if n else explanations
    out = [
        f"EXPLANATION REPORT — {len(subset)} FLAGGED MONITORING WINDOWS",
        "=" * 78,
        "",
        "Each block below describes what the monitoring evidence shows for a",
        "statistically unusual window.  These are evidence-based summaries,",
        "NOT reconstructions of the Isolation Forest's internal reasoning.",
        "",
        explain.DISCLAIMER,
        "",
    ]
    for exp in subset:
        out.append(render_explanation(exp))
        out.append("")
    return "\n".join(out)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=None,
                        help="limit the text report to the top N windows")
    args = parser.parse_args(argv)

    rule("XAI EXPLANATION LAYER — MODEL B FLAGGED WINDOWS")

    # ── Load evidence ────────────────────────────────────────────────────────
    if not EVIDENCE_PATH.exists():
        print(f"  ERROR: {EVIDENCE_PATH} not found.")
        print("  Run evidence_checkpoint.py first.")
        return 1

    document = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    evidence_list = document["evidence"]
    print(f"  evidence file     : {EVIDENCE_PATH}")
    print(f"  evidence objects  : {len(evidence_list)}")

    # ── Generate explanations ────────────────────────────────────────────────
    explanations = explain.explain_all(evidence_list)
    print(f"  explanations      : {len(explanations)}")

    # ── Write JSON ───────────────────────────────────────────────────────────
    output_document = {
        "schema_version": "explanation-1.0",
        "purpose": (
            "Structured plain-English explanations of each flagged monitoring "
            "window. These are evidence-based summaries produced by a "
            "deterministic rule layer, NOT reconstructions of the Isolation "
            "Forest's internal decision path."
        ),
        "interpretation": explain.DISCLAIMER,
        "source_evidence": str(EVIDENCE_PATH.relative_to(config.PROJECT_ROOT)),
        "counts": {
            "explanations": len(explanations),
        },
        "explanations": explanations,
    }

    XAI_DIR.mkdir(parents=True, exist_ok=True)
    EXPLANATIONS_PATH.write_text(
        json.dumps(output_document, indent=2), encoding="utf-8"
    )
    print(f"  wrote             : {EXPLANATIONS_PATH}")

    # ── Write text report ────────────────────────────────────────────────────
    report = render_report(explanations, n=args.top)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  wrote             : {REPORT_PATH}")

    # ── Console summary ──────────────────────────────────────────────────────
    rule("COMPACT SUMMARY")
    for exp in explanations:
        drivers = ", ".join(exp["primary_drivers"]) if exp["primary_drivers"] else "none"
        print(
            f"  #{exp['anomaly_rank']:>2}  case {exp['case_id']}  "
            f"w{exp['window_index']:<3}  [{exp['driver_type']:<13}]  "
            f"drivers: {drivers}"
        )
        # Wrap headline for readability
        wrapped = textwrap.fill(
            exp["headline"], width=74,
            initial_indent="        ", subsequent_indent="        "
        )
        print(wrapped)
        print()

    # ── Driver type distribution ─────────────────────────────────────────────
    rule("DRIVER TYPE DISTRIBUTION")
    from collections import Counter
    dist = Counter(e["driver_type"] for e in explanations)
    for dtype, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {dtype:<15} : {count}")

    # ── Schema check ─────────────────────────────────────────────────────────
    rule("SCHEMA CHECK")
    required = [
        "case_id", "window_index", "anomaly_rank", "anomaly_score",
        "time_label", "headline", "signal_narratives", "primary_drivers",
        "driver_type", "supporting_detail", "data_quality_notes",
        "interpretation",
    ]
    missing = [k for k in required if any(k not in e for e in explanations)]
    print(f"  required fields present in every object : "
          f"{'yes' if not missing else 'NO -> ' + str(missing)}")

    narrative_ok = all(
        signal in e["signal_narratives"]
        for e in explanations for signal in explain.SIGNALS
    )
    print(f"  signal narratives for HR/SpO2/RR        : "
          f"{'yes' if narrative_ok else 'NO'}")

    disclaimer_ok = all(
        e["interpretation"] == explain.DISCLAIMER for e in explanations
    )
    print(f"  disclaimer present in every object      : "
          f"{'yes' if disclaimer_ok else 'NO'}")

    # ── Forbidden terms check ────────────────────────────────────────────────
    rule("FORBIDDEN CLINICAL TERMS CHECK")
    forbidden = [
        "adverse event", "clinical deterioration", "diagnosis",
        "respiratory distress", "cardiac arrest", "patient experienced",
        "clinically validated",
    ]
    # Exclude the disclaimer field — it legitimately uses these words in a
    # negated context ("not a clinical diagnosis").
    scan_objects = [
        {k: v for k, v in e.items() if k != "interpretation"}
        for e in explanations
    ]
    full_text = json.dumps(scan_objects).lower()
    found = [term for term in forbidden if term in full_text]
    if found:
        print(f"  WARNING: found forbidden terms: {found}")
    else:
        print("  OK — no forbidden clinical terms found in explanations.")

    rule("DONE")
    print("  Explanation objects written. No LLM called, no clinical")
    print("  interpretation produced, no Isolation Forest internals claimed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
