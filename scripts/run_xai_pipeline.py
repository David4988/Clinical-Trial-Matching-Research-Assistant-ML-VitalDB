import os
import sys
import json
from vitaldb_audit.llm.pipeline import process_evidence_batch

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evidence_path = os.path.join(repo_root, "results", "xai", "evidence_cases.json")
    cache_path = os.path.join(repo_root, "results", "xai", "llm_cache.json")
    out_json = os.path.join(repo_root, "results", "xai", "llm_explanations.json")
    out_report = os.path.join(repo_root, "results", "xai", "llm_explanations_report.txt")
    out_summary = os.path.join(repo_root, "results", "xai", "llm_run_summary.json")

    print(f"Loading evidence from {evidence_path}")
    try:
        with open(evidence_path, "r") as f:
            data = json.load(f)
            evidence_cases = data.get("evidence", [])
    except Exception as e:
        print(f"Failed to load evidence: {e}")
        return 1

    print(f"Processing {len(evidence_cases)} windows through XAI pipeline...")
    results, metrics = process_evidence_batch(evidence_cases, cache_path)

    print("Saving outputs...")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    with open(out_summary, "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate Report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LLM XAI PIPELINE RUN REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Total windows processed : {metrics['total']}")
    report_lines.append(f"Successful LLM          : {metrics['success']}")
    report_lines.append(f"Failed LLM              : {metrics['failed']}")
    report_lines.append(f"Cache Hits              : {metrics['cache_hits']}")
    report_lines.append(f"Cache Misses            : {metrics['cache_misses']}")
    report_lines.append(f"Total API Calls         : {metrics['total_api_calls']}")
    report_lines.append(f"Schema Validation Fails : {metrics['schema_validation_failures']}")
    report_lines.append(f"Average Latency         : {metrics['average_latency']:.2f} ms")
    report_lines.append(f"Median Latency          : {metrics['median_latency']:.2f} ms")
    report_lines.append("-" * 80)

    # Find 4 representative examples
    # 1. strongest anomaly
    sorted_results = sorted(results, key=lambda x: x["anomaly_rank"])
    strongest = sorted_results[0] if sorted_results else None

    # 2. multi-signal anomaly (driver_type == 'mixed' or len(primary_drivers) > 1)
    multi_signal = next((r for r in results if r["deterministic_explanation"]["driver_type"] == "mixed" or len(r["deterministic_explanation"]["primary_drivers"]) > 1), None)
    
    # 3. dispersion-driven anomaly
    dispersion = next((r for r in results if r["deterministic_explanation"]["driver_type"] == "instability" and r != strongest), None)
    
    # 4. ambiguous (driver_type == 'undetermined' or smallest anomaly score)
    ambiguous = next((r for r in results if r["deterministic_explanation"]["driver_type"] == "undetermined"), None)
    if not ambiguous:
        sorted_by_score = sorted(results, key=lambda x: x["anomaly_score"])
        ambiguous = sorted_by_score[0] if sorted_by_score else None

    examples = {
        "STRONGEST ANOMALY": strongest,
        "MULTI-SIGNAL ANOMALY": multi_signal,
        "DISPERSION-DRIVEN ANOMALY": dispersion,
        "AMBIGUOUS EXAMPLE": ambiguous
    }

    for name, ex in examples.items():
        report_lines.append(f"\n[{name}]")
        if not ex:
            report_lines.append("No matching example found.")
            continue
        report_lines.append(f"Case {ex['case_id']}, Window {ex['window_index']}")
        report_lines.append(f"Rank {ex['anomaly_rank']}, Score {ex['anomaly_score']:.4f}")
        report_lines.append(f"Status: {ex['llm_status']}")
        if ex.get("error"):
            report_lines.append(f"Error: {ex['error']}")
        report_lines.append(f"Headline (Deterministic): {ex['frontend_ready']['headline']}")
        report_lines.append("LLM Summary:")
        report_lines.append(ex['frontend_ready']['summary'])

    with open(out_report, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    # Print to console
    print("\n".join(report_lines))

    print(f"\nOutputs saved to:")
    print(f" - {out_json}")
    print(f" - {out_report}")
    print(f" - {out_summary}")
    print(f" - {cache_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
