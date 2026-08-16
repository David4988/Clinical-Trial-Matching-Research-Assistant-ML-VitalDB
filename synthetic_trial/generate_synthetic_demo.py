"""Generate a clean synthetic XAI demo output."""

import os
import json
from pathlib import Path
import pandas as pd

from synthetic_trial.src.features import build_features
from synthetic_trial.src.model import split_patients, select_training_cohort, prepare_model_matrix, train_anomaly_model, score_observations
from synthetic_trial.src.evaluate import assign_ground_truth_labels
from synthetic_trial.src.evidence import format_synthetic_evidence
from synthetic_trial.src.explain import explain_synthetic_window
from synthetic_trial.src.xai_validation import validate_scenario_consistency
from synthetic_trial.src.llm import SyntheticGeminiProvider

REPORTS_DIR = Path("synthetic_trial/reports")

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup Data & Model
    obs_df = pd.read_csv("synthetic_trial/data/observations.csv")
    feat_df = build_features(obs_df)
    train_patients, eval_patients = split_patients(feat_df, train_ratio=0.5, seed=42)
    train_df = feat_df[feat_df["patient_id"].isin(train_patients)].copy()
    eval_df = feat_df[feat_df["patient_id"].isin(eval_patients)].copy()
    
    model = train_anomaly_model(train_df, contamination=0.10, seed=42)
    scored = score_observations(model, eval_df)
    scored = assign_ground_truth_labels(scored)
    
    # 2. Setup LLM Provider
    api_key = None
    try:
        with open(".env.local") as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.strip().split("=", 1)[1].strip("'\"")
    except FileNotFoundError:
        pass
        
    if api_key:
        llm = SyntheticGeminiProvider(api_key=api_key)
    else:
        print("WARNING: GEMINI_API_KEY not found in .env.local. Skipping LLM generation.")
        llm = None
        
    demo_cases = []
    
    targets = [
        ("Case 1 (STABLE)", "STABLE", "normal"),
        ("Case 2 (SUDDEN_DETERIORATION)", "SUDDEN_DETERIORATION", "acute_change"),
        ("Case 3 (ADVERSE_EVENT)", "ADVERSE_EVENT", "adverse_event")
    ]
    
    for case_name, scenario, gt_state in targets:
        candidates = scored[(scored["scenario"] == scenario) & (scored["ground_truth_state"] == gt_state)]
        if candidates.empty:
            continue
            
        row = candidates.iloc[0]
        
        ev = format_synthetic_evidence(row)
        ex = explain_synthetic_window(ev)
        val = validate_scenario_consistency(ev, ex)
        
        gemini_out = None
        if llm:
            try:
                g_res = llm.explain(ev, ex)
                if g_res:
                    gemini_out = g_res.model_dump()
            except Exception as e:
                print(f"LLM Error on {case_name}: {e}")
                
        demo_cases.append({
            "demo_case": case_name,
            "ground_truth_scenario": scenario,
            "ground_truth_state": gt_state,
            "model_decision": "anomaly" if ev["predicted_anomaly"] == 1 else "normal",
            "physiological_evidence": ev["signals"],
            "deterministic_explanation": ex,
            "scenario_consistency": val,
            "gemini_explanation": gemini_out
        })
        
    # Write output
    out_path = REPORTS_DIR / "synthetic_demo_cases.json"
    with open(out_path, "w") as f:
        json.dump(demo_cases, f, indent=2)
        
    # Print clean console output
    print("\n" + "="*50)
    print("SYNTHETIC XAI DEMO")
    print("="*50)
    
    for case in demo_cases:
        print(f"\n{case['demo_case']}")
        print(f"Scenario: {case['ground_truth_scenario']} -> State: {case['ground_truth_state']}")
        print(f"Model Decision: {case['model_decision']}")
        print(f"Deltas: HR {case['physiological_evidence']['hr']['delta']}, SpO2 {case['physiological_evidence']['spo2']['delta']}, RR {case['physiological_evidence']['rr']['delta']}")
        print(f"Deterministic Summary: {case['deterministic_explanation']['evidence_summary']}")
        print(f"Validation: {case['scenario_consistency']['validation_reason']}")
        
        if case['gemini_explanation']:
            print("\nGemini Output:")
            print(f"  Summary: {case['gemini_explanation']['summary']}")
            print(f"  Key Findings: {case['gemini_explanation']['key_findings']}")
            print(f"  Caveats: {case['gemini_explanation']['caveats']}")
        else:
            print("\nGemini Output: None (Fallback triggered)")
            
        print("-"*50)
        
    print(f"\nFull JSON saved to {out_path}")

if __name__ == "__main__":
    main()
