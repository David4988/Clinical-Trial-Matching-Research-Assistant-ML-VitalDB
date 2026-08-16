"""Contamination experiment for the synthetic evaluation pipeline."""

import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from synthetic_trial.src.features import build_features
from synthetic_trial.src.model import split_patients, select_training_cohort, prepare_model_matrix, train_anomaly_model, score_observations
from synthetic_trial.src.evaluate import (
    assign_ground_truth_labels,
    calculate_binary_metrics,
    calculate_per_scenario_metrics,
    calculate_detection_delay,
    calculate_data_quality_metrics
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("contamination_experiment")

DATA_DIR = Path("synthetic_trial/data")
REPORTS_DIR = Path("synthetic_trial/reports")

# Previous baseline values
BASELINE = {
    "Configuration": "Current baseline (All scenarios, 0.10)",
    "Precision": 0.5578,
    "Recall": 0.2447,
    "F1": 0.3402,
    "FPR": 0.0539,
    "Sudden Det. Recall": 0.4802,
    "Adverse Event Recall": 0.9977
}

def run_experiment():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    obs_df = pd.read_csv(DATA_DIR / "observations.csv")
    feat_df = build_features(obs_df)
    
    # Existing deterministic split
    train_patients, eval_patients = split_patients(feat_df, train_ratio=0.5, seed=42)
    train_df = feat_df[feat_df["patient_id"].isin(train_patients)].copy()
    eval_df = feat_df[feat_df["patient_id"].isin(eval_patients)].copy()
    
    # Cleaner training cohort
    cohort_df = select_training_cohort(train_df)
    train_valid = prepare_model_matrix(cohort_df)
    
    logger.info(f"Training patient count (total): {len(train_patients)}")
    logger.info(f"Training patient count (clean cohort): {len(cohort_df['patient_id'].unique())}")
    logger.info(f"Training window count (total): {len(train_df)}")
    logger.info(f"Training window count (clean cohort): {len(cohort_df)}")
    
    # Scenario counts
    scenario_counts = cohort_df.groupby("patient_id")["scenario"].first().value_counts().to_dict()
    logger.info(f"Cohort scenario counts: {scenario_counts}")
    
    # Ground-truth state counts in training
    gt_counts = cohort_df["ground_truth_state"].value_counts().to_dict()
    logger.info(f"Cohort ground-truth state counts: {gt_counts}")
    
    results = []
    detailed_reports = {}
    
    contaminations = [0.10, 0.15, 0.20]
    
    for c in contaminations:
        logger.info(f"Running experiment with contamination = {c}")
        model = train_anomaly_model(train_df, contamination=c, seed=42)
        
        scored_eval = score_observations(model, eval_df)
        scored_eval = assign_ground_truth_labels(scored_eval)
        
        overall = calculate_binary_metrics(scored_eval[scored_eval["ground_truth_state"] != "data_gap"])
        scenarios = calculate_per_scenario_metrics(scored_eval[scored_eval["ground_truth_state"] != "data_gap"])
        delay = calculate_detection_delay(scored_eval)
        
        # Summary row
        sudden_recall = scenarios.get("SUDDEN_DETERIORATION", {}).get("recall_detection_rate", 0.0)
        adverse_recall = scenarios.get("ADVERSE_EVENT", {}).get("recall_detection_rate", 0.0)
        
        results.append({
            "Configuration": f"Stable/Improving training, {c:.2f}",
            "Precision": overall["precision"],
            "Recall": overall["recall_detection_rate"],
            "F1": overall["f1"],
            "FPR": overall["false_positive_rate"],
            "Sudden Det. Recall": sudden_recall,
            "Adverse Event Recall": adverse_recall
        })
        
        detailed_reports[f"c_{c}"] = {
            "overall": overall,
            "scenarios": scenarios,
            "delay": delay,
            "pred_anomaly_count": int(scored_eval["predicted_anomaly"].sum()),
            "pred_anomaly_pct": float(scored_eval["predicted_anomaly"].sum() / len(scored_eval) * 100)
        }
        
    generate_markdown_report(results, detailed_reports, cohort_df)
    
    json_path = REPORTS_DIR / "model_baseline_comparison.json"
    with open(json_path, "w") as f:
        json.dump({
            "baseline": BASELINE,
            "experiment_results": results,
            "detailed_reports": detailed_reports
        }, f, indent=2)

def generate_markdown_report(results, detailed_reports, cohort_df):
    md_path = REPORTS_DIR / "model_baseline_comparison.md"
    
    lines = [
        "# Model Baseline Comparison Experiment",
        "",
        "## 1. Training Cohort Details",
        f"- **Clean Training Patients**: {len(cohort_df['patient_id'].unique())}",
        f"- **Clean Training Windows**: {len(cohort_df)}",
        "- **Scenario Distribution**: " + str(cohort_df.groupby("patient_id")["scenario"].first().value_counts().to_dict()),
        "- **Ground-Truth State Distribution**: " + str(cohort_df["ground_truth_state"].value_counts().to_dict()),
        "",
        "## 2. Baseline vs Cleaned-Training Comparison",
        "| Configuration | Precision | Recall | F1 | FPR | Sudden Det. Recall | Adverse Event Recall |",
        "|---------------|----------:|-------:|---:|----:|-------------------:|---------------------:|"
    ]
    
    all_rows = [BASELINE] + results
    for r in all_rows:
        lines.append(
            f"| {r['Configuration']} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['FPR']:.4f} | {r['Sudden Det. Recall']:.4f} | {r['Adverse Event Recall']:.4f} |"
        )
        
    lines.extend([
        "",
        "## 3. Detailed Results",
    ])
    
    for c in [0.10, 0.15, 0.20]:
        key = f"c_{c}"
        r = detailed_reports[key]
        lines.extend([
            f"### Stable/Improving training, {c:.2f}",
            f"- **Predicted Anomalous Rows**: {r['pred_anomaly_count']} ({r['pred_anomaly_pct']:.2f}%)",
            f"- **Detected Events**: {r['delay']['detected_cases']} detected, {r['delay']['missed_cases']} missed",
            f"- **Mean Delay**: {r['delay']['mean_delay_minutes']} minutes (Worst: {r['delay']['worst_delay_windows'] * 5 if r['delay']['worst_delay_windows'] else 0} mins)",
            "",
            "**Stable/Improving False Positive Inspection:**",
            f"- STABLE FPR: {r['scenarios'].get('STABLE', {}).get('false_positive_rate', 0.0):.4f}",
            f"- IMPROVING FPR: {r['scenarios'].get('IMPROVING', {}).get('false_positive_rate', 0.0):.4f}",
            ""
        ])
        
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Saved comparison report to {md_path}")

if __name__ == "__main__":
    run_experiment()
