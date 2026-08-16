"""Runner script for the synthetic evaluation pipeline."""

import os
import json
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from synthetic_trial.src.features import build_features
from synthetic_trial.src.model import split_patients, train_anomaly_model, score_observations
from synthetic_trial.src.evaluate import (
    assign_ground_truth_labels,
    calculate_binary_metrics,
    calculate_per_scenario_metrics,
    calculate_detection_delay,
    calculate_data_quality_metrics
)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("synthetic_evaluation")

DATA_DIR = Path("synthetic_trial/data")
REPORTS_DIR = Path("synthetic_trial/reports")
PLOTS_DIR = Path("synthetic_trial/plots")

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    logger.info("Loading observations...")
    obs_path = DATA_DIR / "observations.csv"
    if not obs_path.exists():
        logger.error(f"Missing {obs_path}. Run generator first.")
        return
        
    obs_df = pd.read_csv(obs_path)
    
    # 2. Build Features
    logger.info("Building synthetic features...")
    feat_df = build_features(obs_df)
    
    # 3. Train/Eval Split
    logger.info("Splitting patients...")
    train_patients, eval_patients = split_patients(feat_df, train_ratio=0.5, seed=42)
    
    train_df = feat_df[feat_df["patient_id"].isin(train_patients)].copy()
    eval_df = feat_df[feat_df["patient_id"].isin(eval_patients)].copy()
    
    logger.info(f"Train patients: {len(train_patients)}, Eval patients: {len(eval_patients)}")
    logger.info(f"Train windows: {len(train_df)}, Eval windows: {len(eval_df)}")
    
    # 4. Train Model
    logger.info("Training Isolation Forest anomaly model on training set...")
    model = train_anomaly_model(train_df, contamination=0.10, seed=42)
    
    # 5. Score Evaluation Set
    logger.info("Scoring evaluation windows...")
    scored_eval = score_observations(model, eval_df)
    scored_eval = assign_ground_truth_labels(scored_eval)
    
    # 6. Calculate Metrics
    logger.info("Calculating metrics...")
    
    # Physiological overall
    overall_metrics = calculate_binary_metrics(scored_eval[scored_eval["ground_truth_state"] != "data_gap"])
    
    # Per-scenario
    scenario_metrics = calculate_per_scenario_metrics(scored_eval[scored_eval["ground_truth_state"] != "data_gap"])
    
    # Detection delay
    delay_metrics = calculate_detection_delay(scored_eval)
    
    # Data quality
    dq_metrics = calculate_data_quality_metrics(scored_eval)
    
    # 7. Generate Reports
    report = {
        "evaluation_setup": {
            "train_patients": len(train_patients),
            "eval_patients": len(eval_patients),
            "train_windows": len(train_df),
            "eval_windows": len(eval_df)
        },
        "overall_physiological_metrics": overall_metrics,
        "scenario_metrics": scenario_metrics,
        "detection_delay": delay_metrics,
        "data_quality_metrics": dq_metrics
    }
    
    report_path = REPORTS_DIR / "evaluation_metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Saved evaluation metrics to {report_path}")
    
    md_path = REPORTS_DIR / "evaluation_report.md"
    generate_markdown_report(report, md_path)
    logger.info(f"Saved evaluation report to {md_path}")
    
    # Save predictions
    preds_path = REPORTS_DIR / "evaluation_predictions.csv"
    scored_eval.to_csv(preds_path, index=False)
    logger.info(f"Saved predictions to {preds_path}")
    
    # 8. Plots
    plot_representative_patient(scored_eval, "SUDDEN_DETERIORATION", PLOTS_DIR / "sudden_deterioration_eval.png")
    plot_representative_patient(scored_eval, "ADVERSE_EVENT", PLOTS_DIR / "adverse_event_eval.png")
    
    logger.info("Evaluation pipeline complete.")

def generate_markdown_report(report, out_path):
    lines = [
        "# Synthetic Evaluation Report",
        "",
        "## Setup",
        f"- **Train Patients**: {report['evaluation_setup']['train_patients']}",
        f"- **Eval Patients**: {report['evaluation_setup']['eval_patients']}",
        f"- **Train Windows**: {report['evaluation_setup']['train_windows']}",
        f"- **Eval Windows**: {report['evaluation_setup']['eval_windows']}",
        "",
        "## Overall Physiological Metrics (Excluding Data Gaps)",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Windows | {report['overall_physiological_metrics']['total_windows']} |",
        f"| Precision | {report['overall_physiological_metrics']['precision']} |",
        f"| Recall | {report['overall_physiological_metrics']['recall_detection_rate']} |",
        f"| F1 Score | {report['overall_physiological_metrics']['f1']} |",
        f"| False Positive Rate | {report['overall_physiological_metrics']['false_positive_rate']} |",
        "",
        "## Detection Delay (Sudden/Adverse Events)",
        f"- **Detected Cases**: {report['detection_delay']['detected_cases']}",
        f"- **Missed Cases**: {report['detection_delay']['missed_cases']}",
        f"- **Mean Delay (mins)**: {report['detection_delay']['mean_delay_minutes']} | **Worst Delay (mins)**: {report['detection_delay']['worst_delay_windows'] * 5 if report['detection_delay']['worst_delay_windows'] else 0}",
        "",
        "## Per-Scenario Metrics",
        "| Scenario | Total Windows | Precision | Recall | F1 | FPR |",
        "|----------|---------------|-----------|--------|----|-----|"
    ]
    
    for scen, m in report['scenario_metrics'].items():
        if "error" in m:
            continue
        lines.append(f"| {scen} | {m['total_windows']} | {m.get('precision', 'N/A')} | {m.get('recall_detection_rate', 'N/A')} | {m.get('f1', 'N/A')} | {m.get('false_positive_rate', 'N/A')} |")
        
    lines.extend([
        "",
        "## Data Quality Monitoring",
        f"- **Total Gap Windows**: {report['data_quality_metrics']['total_gap_windows']}",
        f"- **Flagged Gap Windows**: {report['data_quality_metrics']['flagged_gap_windows']}",
        f"- **Flag Rate**: {report['data_quality_metrics']['flag_rate']}",
        ""
    ])
    
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

def plot_representative_patient(df, scenario, out_path):
    scen_df = df[df["scenario"] == scenario]
    if scen_df.empty:
        return
        
    pid = scen_df["patient_id"].iloc[0]
    pdf = scen_df[scen_df["patient_id"] == pid].sort_values("window_index")
    
    plt.figure(figsize=(10, 5))
    plt.plot(pdf["window_index"], pdf["anomaly_score"], label="Anomaly Score", color="blue")
    
    # Shade flagged windows
    flags = pdf[pdf["predicted_anomaly"] == 1]
    for _, row in flags.iterrows():
        plt.axvspan(row["window_index"]-0.5, row["window_index"]+0.5, color="red", alpha=0.3)
        
    # Mark transition if event
    transitions = pdf[pdf["ground_truth_state"].isin(["acute_change", "adverse_event"])]
    if not transitions.empty:
        plt.axvline(x=transitions.iloc[0]["window_index"], color="black", linestyle="--", label="Ground Truth Transition")
        
    plt.title(f"Model Prediction vs Ground Truth: {pid} ({scenario})")
    plt.xlabel("Window Index")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    main()
