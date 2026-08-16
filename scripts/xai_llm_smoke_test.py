"""Minimal LLM XAI connection spike script.

Reads case 4 window 62 from results/xai/evidence_cases.json and
generates a structured explanation using the Gemini provider.
"""

import json
import sys
import time
from pathlib import Path

# Add src to pythonpath
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vitaldb_audit.llm.config import get_llm_config, ConfigError
from vitaldb_audit.llm.gemini import GeminiProvider
from pydantic import ValidationError

EVIDENCE_FILE = PROJECT_ROOT / "results" / "xai" / "evidence_cases.json"
OUTPUT_FILE = PROJECT_ROOT / "results" / "xai" / "llm_smoke_test.json"


def main():
    print("=" * 80)
    print("LLM XAI CONNECTION SPIKE (SMOKE TEST)")
    print("=" * 80)

    # 1. Load config
    try:
        config = get_llm_config()
    except ConfigError as e:
        print(f"\n[ERROR] Configuration failed: {e}")
        return 1

    provider_name = config["provider"]
    model_name = config["model"]
    api_key = config["api_key"]

    print(f"Provider : {provider_name}")
    print(f"Model    : {model_name}")

    if provider_name != "gemini":
        print(f"\n[ERROR] Only Gemini is currently implemented for this spike.")
        return 1

    # 2. Instantiate provider
    provider = GeminiProvider(api_key=api_key, model=model_name)

    # 3. Load evidence
    if not EVIDENCE_FILE.exists():
        print(f"\n[ERROR] Evidence file not found: {EVIDENCE_FILE}")
        return 1

    try:
        with open(EVIDENCE_FILE, "r") as f:
            all_evidence = json.load(f)
    except Exception as e:
        print(f"\n[ERROR] Failed to read evidence file: {e}")
        return 1

    # Find Case 4 / Window 62
    target_evidence = None
    for ev in all_evidence.get("evidence", []):
        if ev.get("case_id") == 4 and ev.get("window_index") == 62:
            target_evidence = ev
            break
            
    if not target_evidence:
        print("\n[ERROR] Could not find Case 4, Window 62 in evidence file.")
        return 1

    print(f"\nLoaded Evidence for Case {target_evidence['case_id']}, Window {target_evidence['window_index']}")

    # 4. Generate Explanation
    print("\nSending request to LLM API...")
    start_time = time.time()
    
    try:
        explanation = provider.explain(target_evidence)
        api_success = True
        schema_valid = True
        error_msg = None
    except ValidationError as e:
        api_success = True
        schema_valid = False
        error_msg = str(e)
    except Exception as e:
        api_success = False
        schema_valid = False
        error_msg = str(e)
        
    latency = time.time() - start_time

    # 5. Report
    print("-" * 80)
    print("RESULTS")
    print("-" * 80)
    print(f"Latency                 : {latency:.2f} seconds")
    print(f"HTTP/API success        : {api_success}")
    print(f"Structured-output valid : {schema_valid}")

    if not api_success:
        print(f"\nAPI Error:\n{error_msg}")
        return 1
        
    if not schema_valid:
        print(f"\nSchema Validation Error:\n{error_msg}")
        return 1

    # 6. Save & Print
    explanation_dict = explanation.model_dump()
    
    # Also attach metadata for the test result file
    output_data = {
        "metadata": {
            "provider": provider_name,
            "model": model_name,
            "latency_seconds": round(latency, 3),
            "evidence_reference": {
                "case_id": target_evidence["case_id"],
                "window_index": target_evidence["window_index"]
            }
        },
        "explanation": explanation_dict
    }
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\nSaved result to: {OUTPUT_FILE}\n")
    print("GENERATED EXPLANATION:")
    print(json.dumps(explanation_dict, indent=2))
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
