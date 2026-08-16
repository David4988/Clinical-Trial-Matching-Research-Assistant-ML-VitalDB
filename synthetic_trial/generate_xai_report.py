"""Generate XAI explanation quality report."""

import json
from pathlib import Path
import pandas as pd

from synthetic_trial.src.evidence import format_synthetic_evidence
from synthetic_trial.src.explain import explain_synthetic_window
from synthetic_trial.src.xai_validation import validate_scenario_consistency

REPORTS_DIR = Path("synthetic_trial/reports")

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load previously generated predictions (from the c=0.10 run ideally, 
    # but we can re-run or just use the saved csv if we had one. 
    # Actually, we didn't save the predictions CSV in the experiment script, 
    # but we DO have 'synthetic_trial/reports/evaluation_predictions.csv' from the baseline.
    # Let's run a fresh prediction using c=0.10 and the clean training cohort to ensure it's up to date.)
    from synthetic_trial.src.features import build_features
    from synthetic_trial.src.model import split_patients, train_anomaly_model, score_observations
    from synthetic_trial.src.evaluate import assign_ground_truth_labels
    
    obs_df = pd.read_csv("synthetic_trial/data/observations.csv")
    feat_df = build_features(obs_df)
    train_patients, eval_patients = split_patients(feat_df, train_ratio=0.5, seed=42)
    train_df = feat_df[feat_df["patient_id"].isin(train_patients)].copy()
    eval_df = feat_df[feat_df["patient_id"].isin(eval_patients)].copy()
    
    model = train_anomaly_model(train_df, contamination=0.10, seed=42)
    scored = score_observations(model, eval_df)
    scored = assign_ground_truth_labels(scored)
    
    total_evaluated = len(scored)
    success_count = 0
    complete_count = 0
    consistent_count = 0
    
    scenario_consistency = {s: {"total": 0, "consistent": 0} for s in scored["scenario"].unique()}
    
    # Run pipeline on every window
    for _, row in scored.iterrows():
        # 1. Evidence
        ev = format_synthetic_evidence(row)
        
        # 2. Explanation
        try:
            ex = explain_synthetic_window(ev)
            success_count += 1
        except Exception:
            continue
            
        # 3. Validation
        val = validate_scenario_consistency(ev, ex)
        
        if val["evidence_completeness"]:
            complete_count += 1
        if val["scenario_consistency"]:
            consistent_count += 1
            scenario_consistency[ev["scenario"]]["consistent"] += 1
            
        scenario_consistency[ev["scenario"]]["total"] += 1

    # Report metrics
    report = {
        "total_evaluated_windows": total_evaluated,
        "explanation_generation_success_rate": success_count / total_evaluated if total_evaluated else 0,
        "evidence_completeness_rate": complete_count / total_evaluated if total_evaluated else 0,
        "scenario_consistency_rate": consistent_count / total_evaluated if total_evaluated else 0,
        "per_scenario_consistency": {
            s: (d["consistent"] / d["total"] if d["total"] else 0)
            for s, d in scenario_consistency.items()
        }
    }
    
    # Write JSON
    with open(REPORTS_DIR / "xai_validation.json", "w") as f:
        json.dump(report, f, indent=2)
        
    # Write MD
    md = [
        "# Synthetic XAI Explanation Quality Report",
        "",
        "**NOTE: This is a scenario-grounded engineering validation. It does NOT represent clinical validation of the explanations.**",
        "",
        f"- **Total evaluated windows**: {report['total_evaluated_windows']}",
        f"- **Explanation generation success rate**: {report['explanation_generation_success_rate'] * 100:.2f}%",
        f"- **Evidence completeness rate**: {report['evidence_completeness_rate'] * 100:.2f}%",
        f"- **Overall scenario consistency rate**: {report['scenario_consistency_rate'] * 100:.2f}%",
        "",
        "## Per-Scenario Consistency",
        "| Scenario | Consistency Rate |",
        "|----------|-----------------:|"
    ]
    for s, rate in report["per_scenario_consistency"].items():
        md.append(f"| {s} | {rate * 100:.2f}% |")
        
    with open(REPORTS_DIR / "xai_validation_report.md", "w") as f:
        f.write("\n".join(md))

if __name__ == "__main__":
    main()
