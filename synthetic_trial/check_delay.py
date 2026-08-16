import pandas as pd
from synthetic_trial.src.features import build_features
from synthetic_trial.src.model import split_patients, select_training_cohort, prepare_model_matrix, train_anomaly_model, score_observations

obs_df = pd.read_csv("synthetic_trial/data/observations.csv")
feat_df = build_features(obs_df)
train_patients, eval_patients = split_patients(feat_df, train_ratio=0.5, seed=42)
train_df = feat_df[feat_df["patient_id"].isin(train_patients)].copy()
eval_df = feat_df[feat_df["patient_id"].isin(eval_patients)].copy()
cohort_df = select_training_cohort(train_df)

model = train_anomaly_model(train_df, contamination=0.10, seed=42)
scored = score_observations(model, eval_df)

sd_patients = scored[scored["scenario"] == "SUDDEN_DETERIORATION"]["patient_id"].unique()
for pid in sd_patients[:1]:
    print(f"\nPatient {pid}")
    pdf = scored[scored["patient_id"] == pid]
    t_idx = pdf[pdf["ground_truth_state"] == "acute_change"].index[0]
    
    before = pdf.loc[:t_idx-1]
    flags_before = before[before["predicted_anomaly"] == 1]
    print(f"Pre-event false positive windows: {flags_before['window_index'].tolist()}")
    
    for idx in range(t_idx-2, t_idx+3):
        if idx in pdf.index:
            r = pdf.loc[idx]
            print(f"w{r['window_index']} | {r['ground_truth_state']} | HR: {r['heart_rate']:.1f} (d:{r['heart_rate_delta']}) | pred: {r['predicted_anomaly']}")
