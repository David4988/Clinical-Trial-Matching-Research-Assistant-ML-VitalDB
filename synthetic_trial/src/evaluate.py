"""Evaluation metrics for synthetic monitoring pipeline."""

import json
import pandas as pd
import numpy as np

POSITIVE_STATES = {"deteriorating", "acute_change", "adverse_event"}
NEGATIVE_STATES = {"normal", "improving", "improved", "recovering", "recovered", "post_event"}

def assign_ground_truth_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map the granular ground_truth_state into binary is_anomaly_ground_truth."""
    df = df.copy()
    
    # 1 for positive states, 0 for negative states, NaN for data_gap or unknown
    df["is_anomaly_ground_truth"] = np.nan
    df.loc[df["ground_truth_state"].isin(POSITIVE_STATES), "is_anomaly_ground_truth"] = 1
    df.loc[df["ground_truth_state"].isin(NEGATIVE_STATES), "is_anomaly_ground_truth"] = 0
    
    return df

def calculate_binary_metrics(df: pd.DataFrame) -> dict:
    """Calculate binary classification metrics, excluding data_gap rows."""
    valid_mask = df["is_anomaly_ground_truth"].notna() & df["predicted_anomaly"].notna()
    eval_df = df[valid_mask]
    
    if eval_df.empty:
        return {"error": "No valid rows for evaluation"}
        
    y_true = eval_df["is_anomaly_ground_truth"].astype(int)
    y_pred = eval_df["predicted_anomaly"].astype(int)
    
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        "total_windows": len(eval_df),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall_detection_rate": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4)
    }

def calculate_per_scenario_metrics(df: pd.DataFrame) -> dict:
    """Calculate metrics broken down by scenario."""
    scenarios = df["scenario"].unique()
    results = {}
    
    for scen in scenarios:
        scen_df = df[df["scenario"] == scen]
        metrics = calculate_binary_metrics(scen_df)
        if "error" not in metrics:
            results[scen] = metrics
            
    return results

def calculate_detection_delay(df: pd.DataFrame) -> dict:
    """Calculate time-to-first-flag for event scenarios.
    
    Primary scenarios: SUDDEN_DETERIORATION, ADVERSE_EVENT
    """
    target_scenarios = ["SUDDEN_DETERIORATION", "ADVERSE_EVENT"]
    event_patients = df[df["scenario"].isin(target_scenarios)]["patient_id"].unique()
    
    delays = []
    detected = 0
    missed = 0
    
    for pid in event_patients:
        pdf = df[df["patient_id"] == pid].sort_values("window_index")
        
        # Find the transition window: where state becomes acute_change or adverse_event
        event_states = {"acute_change", "adverse_event"}
        transitions = pdf[pdf["ground_truth_state"].isin(event_states)]
        
        if transitions.empty:
            continue
            
        transition_idx = transitions.iloc[0]["window_index"]
        
        # Look for flags after the transition
        post_event = pdf[pdf["window_index"] >= transition_idx]
        flags = post_event[post_event["predicted_anomaly"] == 1]
        
        if not flags.empty:
            first_flag_idx = flags.iloc[0]["window_index"]
            delay = int(first_flag_idx - transition_idx)
            delays.append(delay)
            detected += 1
        else:
            missed += 1
            
    if delays:
        mean_delay = float(np.mean(delays))
        median_delay = float(np.median(delays))
        worst_delay = int(np.max(delays))
    else:
        mean_delay = median_delay = worst_delay = None
        
    return {
        "detected_cases": detected,
        "missed_cases": missed,
        "mean_delay_windows": mean_delay,
        "median_delay_windows": median_delay,
        "worst_delay_windows": worst_delay,
        "mean_delay_minutes": mean_delay * 5 if mean_delay is not None else None,
        "median_delay_minutes": median_delay * 5 if median_delay is not None else None,
    }

def calculate_data_quality_metrics(df: pd.DataFrame) -> dict:
    """Evaluate prediction of data_gap windows."""
    # Data gap rows
    gap_df = df[df["ground_truth_state"] == "data_gap"]
    total = len(gap_df)
    if total == 0:
        return {"total_gap_windows": 0}
        
    # How many of these were flagged by the anomaly model?
    # Note: IF might flag these due to weird coverage_percent or missing values
    # Actually, if coverage drops, does the model flag it? We'll see.
    flagged = int((gap_df["predicted_anomaly"] == 1).sum())
    
    return {
        "total_gap_windows": total,
        "flagged_gap_windows": flagged,
        "flag_rate": round(flagged / total, 4) if total > 0 else 0.0
    }
