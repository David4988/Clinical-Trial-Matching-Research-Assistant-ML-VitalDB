"""Deterministic evidence formatting for the synthetic pipeline."""

import pandas as pd

def format_synthetic_evidence(row: pd.Series) -> dict:
    """Format a single scored window into structured XAI evidence."""
    return {
        "patient_id": str(row.get("patient_id", "")),
        "trial_id": str(row.get("trial_id", "")),
        "timestamp_minutes": float(row.get("timestamp", 0.0)),
        "window_index": int(row.get("window_index", 0)),
        "dose_number": int(row.get("dose_number", 0)),
        
        "scenario": str(row.get("scenario", "")),
        "ground_truth_state": str(row.get("ground_truth_state", "")),
        
        "anomaly_score": float(row.get("anomaly_score", 0.0)) if pd.notna(row.get("anomaly_score")) else None,
        "predicted_anomaly": int(row.get("predicted_anomaly", 0)) if pd.notna(row.get("predicted_anomaly")) else None,
        
        "signals": {
            "hr": {
                "current_mean": float(row.get("heart_rate", 0.0)),
                "delta": float(row.get("heart_rate_delta", 0.0)) if pd.notna(row.get("heart_rate_delta")) else None,
            },
            "spo2": {
                "current_mean": float(row.get("spo2", 0.0)),
                "delta": float(row.get("spo2_delta", 0.0)) if pd.notna(row.get("spo2_delta")) else None,
            },
            "rr": {
                "current_mean": float(row.get("respiratory_rate", 0.0)),
                "delta": float(row.get("respiratory_rate_delta", 0.0)) if pd.notna(row.get("respiratory_rate_delta")) else None,
            }
        },
        
        "coverage": float(row.get("coverage_percent", 100.0)),
        "data_quality": str(row.get("data_quality", "GOOD")),
    }

def format_all_evidence(scored_df: pd.DataFrame) -> list[dict]:
    """Format a collection of scored windows."""
    evidence_list = []
    for _, row in scored_df.iterrows():
        evidence_list.append(format_synthetic_evidence(row))
    return evidence_list
