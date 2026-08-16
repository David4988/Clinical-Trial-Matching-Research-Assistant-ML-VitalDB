"""Tests for synthetic evaluation pipeline components."""

import pandas as pd
import numpy as np

from synthetic_trial.src.features import build_features
from synthetic_trial.src.model import split_patients, prepare_model_matrix
from synthetic_trial.src.evaluate import (
    assign_ground_truth_labels,
    calculate_binary_metrics,
    calculate_detection_delay,
    calculate_data_quality_metrics
)
from synthetic_trial.src.evidence import format_synthetic_evidence

def test_feature_extraction_deltas():
    obs = pd.DataFrame({
        "patient_id": ["P1", "P1", "P2", "P2"],
        "window_index": [0, 1, 0, 1],
        "heart_rate": [70.0, 75.0, 80.0, 78.0],
        "spo2": [98.0, 97.0, 99.0, 99.0],
        "respiratory_rate": [15.0, 16.0, 14.0, 14.5]
    })
    
    features = build_features(obs)
    
    # First window deltas should be NaN
    p1_w0 = features.iloc[0]
    assert pd.isna(p1_w0["heart_rate_delta"])
    
    # Second window deltas should be calculated correctly
    p1_w1 = features.iloc[1]
    assert p1_w1["heart_rate_delta"] == 5.0
    assert p1_w1["spo2_delta"] == -1.0
    
    # Patient boundary: P2's first window should be NaN, not compared to P1
    p2_w0 = features.iloc[2]
    assert pd.isna(p2_w0["heart_rate_delta"])
    
def test_evaluation_labels_mapping():
    df = pd.DataFrame({
        "ground_truth_state": ["normal", "adverse_event", "acute_change", "recovering", "data_gap"]
    })
    mapped = assign_ground_truth_labels(df)
    
    assert mapped.iloc[0]["is_anomaly_ground_truth"] == 0  # normal
    assert mapped.iloc[1]["is_anomaly_ground_truth"] == 1  # adverse_event
    assert mapped.iloc[2]["is_anomaly_ground_truth"] == 1  # acute_change
    assert mapped.iloc[3]["is_anomaly_ground_truth"] == 0  # recovering
    assert pd.isna(mapped.iloc[4]["is_anomaly_ground_truth"])  # data_gap
    
def test_data_gap_exclusion():
    df = pd.DataFrame({
        "ground_truth_state": ["data_gap", "adverse_event"],
        "is_anomaly_ground_truth": [np.nan, 1],
        "predicted_anomaly": [1, 1]
    })
    metrics = calculate_binary_metrics(df)
    
    # The data_gap row should be excluded from physiological metrics, leaving only 1 TP
    assert metrics["total_windows"] == 1
    assert metrics["true_positives"] == 1
    
    dq = calculate_data_quality_metrics(df)
    assert dq["total_gap_windows"] == 1
    assert dq["flagged_gap_windows"] == 1

def test_patient_split():
    df = pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(100)]
    })
    train_ids, eval_ids = split_patients(df, train_ratio=0.5, seed=42)
    
    # Deterministic check
    train_ids2, eval_ids2 = split_patients(df, train_ratio=0.5, seed=42)
    assert train_ids == train_ids2
    
    # Overlap check
    assert len(set(train_ids).intersection(set(eval_ids))) == 0
    
    # Coverage check
    assert len(train_ids) + len(eval_ids) == 100

def test_detection_delay_calculation():
    df = pd.DataFrame({
        "patient_id": ["P1", "P1", "P1", "P1", "P2", "P2"],
        "scenario": ["SUDDEN_DETERIORATION"]*4 + ["ADVERSE_EVENT"]*2,
        "window_index": [0, 1, 2, 3, 0, 1],
        "ground_truth_state": ["normal", "acute_change", "acute_change", "acute_change", "normal", "adverse_event"],
        "predicted_anomaly": [0, 0, 1, 1, 0, 0]
    })
    
    delay_metrics = calculate_detection_delay(df)
    
    # P1 transition is at window 1, first flag is at window 2 -> delay = 1
    # P2 transition is at window 1, no flag -> missed
    
    assert delay_metrics["detected_cases"] == 1
    assert delay_metrics["missed_cases"] == 1
    assert delay_metrics["mean_delay_windows"] == 1.0

def test_evidence_formatting():
    row = pd.Series({
        "patient_id": "P001",
        "timestamp": 15.0,
        "window_index": 3,
        "scenario": "STABLE",
        "ground_truth_state": "normal",
        "anomaly_score": 0.45,
        "predicted_anomaly": 1,
        "heart_rate": 80.0,
        "heart_rate_delta": 2.0
    })
    
    evidence = format_synthetic_evidence(row)
    
    assert evidence["patient_id"] == "P001"
    assert evidence["timestamp_minutes"] == 15.0
    assert evidence["anomaly_score"] == 0.45
    assert evidence["signals"]["hr"]["current_mean"] == 80.0
    assert evidence["signals"]["hr"]["delta"] == 2.0

def test_training_cohort_selection():
    from synthetic_trial.src.model import select_training_cohort
    
    # Create mock training dataframe with various scenarios
    df = pd.DataFrame({
        "patient_id": ["P1", "P1", "P2", "P2", "P3", "P4", "P5"],
        "scenario": ["STABLE", "STABLE", "IMPROVING", "IMPROVING", "GRADUAL_DETERIORATION", "SUDDEN_DETERIORATION", "ADVERSE_EVENT"],
        "ground_truth_state": ["normal"] * 7,
        "value": range(7)
    })
    
    filtered = select_training_cohort(df)
    
    # Only P1 (STABLE) and P2 (IMPROVING) should remain
    remaining_patients = set(filtered["patient_id"].unique())
    assert remaining_patients == {"P1", "P2"}
    assert "P3" not in remaining_patients
    assert "P4" not in remaining_patients
    
    # Ensure all their windows are kept
    assert len(filtered[filtered["patient_id"] == "P1"]) == 2
    assert len(filtered[filtered["patient_id"] == "P2"]) == 2
    
    # Ensure no scenario leakage in output
    assert set(filtered["scenario"].unique()) == {"STABLE", "IMPROVING"}
