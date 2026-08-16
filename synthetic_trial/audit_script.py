import pandas as pd
import json

from synthetic_trial.src.features import build_features
from synthetic_trial.src.model import split_patients, MODEL_FEATURES
from synthetic_trial.src.evaluate import assign_ground_truth_labels, calculate_detection_delay

def main():
    obs = pd.read_csv("synthetic_trial/data/observations.csv")
    feat = build_features(obs)
    
    # 1. VERIFY DATA SPLIT
    train_ids, eval_ids = split_patients(feat, train_ratio=0.5, seed=42)
    train_df = feat[feat["patient_id"].isin(train_ids)]
    eval_df = feat[feat["patient_id"].isin(eval_ids)]
    
    print("--- 1. VERIFY DATA SPLIT ---")
    print(f"Train patients: {len(train_ids)}")
    print(f"Eval patients: {len(eval_ids)}")
    print(f"Train windows: {len(train_df)}")
    print(f"Eval windows: {len(eval_df)}")
    print("Train scenarios:", train_df.groupby("patient_id")["scenario"].first().value_counts().to_dict())
    print("Eval scenarios:", eval_df.groupby("patient_id")["scenario"].first().value_counts().to_dict())
    print("Train windows by scenario:", train_df["scenario"].value_counts().to_dict())
    print("Eval windows by scenario:", eval_df["scenario"].value_counts().to_dict())
    print(f"Patient overlap: {set(train_ids).intersection(set(eval_ids))}")
    
    # 3. VERIFY TRAINING CONTAMINATION
    print("\n--- 3. VERIFY TRAINING CONTAMINATION ---")
    train_valid = train_df.dropna(subset=MODEL_FEATURES)
    print(f"Total training rows: {len(train_df)}")
    print(f"Rows used for fitting: {len(train_valid)}")
    print("Fitting rows state distribution:", train_valid["ground_truth_state"].value_counts().to_dict())
    print("Fitting rows scenario distribution:", train_valid["scenario"].value_counts().to_dict())
    
    # 4. VERIFY LABEL MAPPING
    print("\n--- 4. VERIFY LABEL MAPPING ---")
    eval_mapped = assign_ground_truth_labels(eval_df)
    state_mapping = eval_mapped.groupby("ground_truth_state")["is_anomaly_ground_truth"].first().to_dict()
    state_counts = eval_mapped["ground_truth_state"].value_counts().to_dict()
    for state, is_anom in state_mapping.items():
        print(f"{state:20} | {is_anom} (count: {state_counts[state]})")

    # 5. VERIFY ADVERSE EVENT RECALL
    print("\n--- 5. VERIFY ADVERSE EVENT ---")
    eval_scored = pd.read_csv("synthetic_trial/reports/evaluation_predictions.csv")
    ae_patients = eval_scored[eval_scored["scenario"] == "ADVERSE_EVENT"]["patient_id"].unique()
    for pid in ae_patients[:2]:
        print(f"\nPatient {pid}")
        pdf = eval_scored[eval_scored["patient_id"] == pid]
        transitions = pdf[pdf["ground_truth_state"] == "adverse_event"]
        if transitions.empty: continue
        t_idx = transitions.index[0]
        # show window before, at, and after
        window_indices = [t_idx-1, t_idx, t_idx+1]
        for idx in window_indices:
            if idx in pdf.index:
                row = pdf.loc[idx]
                print(f"w{row['window_index']} | {row['timestamp']:.1f} | HR: {row['heart_rate']:.1f} (d:{row['heart_rate_delta']}) | SpO2: {row['spo2']:.1f} (d:{row['spo2_delta']}) | RR: {row['respiratory_rate']:.1f} (d:{row['respiratory_rate_delta']}) | {row['ground_truth_state']:15} | score: {row['anomaly_score']:.3f} | pred: {row['predicted_anomaly']}")

    # 6. VERIFY SUDDEN DETERIORATION DELAY
    print("\n--- 6. VERIFY SUDDEN DETERIORATION ---")
    sd_patients = eval_scored[eval_scored["scenario"] == "SUDDEN_DETERIORATION"]["patient_id"].unique()
    for pid in sd_patients[:2]:
        print(f"\nPatient {pid}")
        pdf = eval_scored[eval_scored["patient_id"] == pid]
        transitions = pdf[pdf["ground_truth_state"] == "acute_change"]
        if transitions.empty: continue
        t_idx = transitions.index[0]
        
        # Check if already positive before
        before = pdf.loc[:t_idx-1]
        flags_before = before[before["predicted_anomaly"] == 1]
        if not flags_before.empty:
            print(f"WARNING: Already positive before event at windows {flags_before['window_index'].tolist()}")
            
        # show around transition
        window_indices = range(t_idx-2, t_idx+3)
        for idx in window_indices:
            if idx in pdf.index:
                row = pdf.loc[idx]
                print(f"w{row['window_index']} | {row['timestamp']:.1f} | HR: {row['heart_rate']:.1f} (d:{row['heart_rate_delta']}) | {row['ground_truth_state']:15} | score: {row['anomaly_score']:.3f} | pred: {row['predicted_anomaly']}")
                
    # 8. CHECK THRESHOLD/CONTAMINATION
    print("\n--- 8. CHECK THRESHOLD/CONTAMINATION ---")
    total_eval = len(eval_scored)
    pred_anom = eval_scored["predicted_anomaly"].sum()
    print(f"Total eval observations: {total_eval}")
    print(f"Predicted anomalous: {pred_anom} ({pred_anom/total_eval*100:.2f}%)")
    actual_pos = eval_scored["is_anomaly_ground_truth"].sum()
    actual_neg = (eval_scored["is_anomaly_ground_truth"] == 0).sum()
    print(f"Actual positive: {actual_pos} ({actual_pos/total_eval*100:.2f}%)")
    print(f"Actual negative: {actual_neg} ({actual_neg/total_eval*100:.2f}%)")

if __name__ == "__main__":
    main()
