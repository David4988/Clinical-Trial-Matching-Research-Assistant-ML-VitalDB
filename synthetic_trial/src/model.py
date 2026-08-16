"""Anomaly detection model for synthetic evaluation pipeline."""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("synthetic_trial.model")

MODEL_FEATURES = [
    "heart_rate", "spo2", "respiratory_rate",
    "heart_rate_delta", "spo2_delta", "respiratory_rate_delta"
]

def split_patients(df: pd.DataFrame, train_ratio: float = 0.5, seed: int = 42) -> tuple[list, list]:
    """Deterministically split patients into train and evaluation sets.
    
    Args:
        df: DataFrame containing at least 'patient_id'.
        train_ratio: Fraction of patients to use for training.
        seed: Random seed for deterministic split.
        
    Returns:
        tuple of (train_patient_ids, eval_patient_ids)
    """
    patients = sorted(df["patient_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    
    split_idx = int(len(patients) * train_ratio)
    train_ids = patients[:split_idx]
    eval_ids = patients[split_idx:]
    
    return train_ids, eval_ids

def prepare_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows with valid features."""
    # We require all model features to be non-null.
    # The first window of each patient will have null deltas.
    valid = df.dropna(subset=MODEL_FEATURES).copy()
    return valid

def train_anomaly_model(train_df: pd.DataFrame, contamination: float = 0.10, seed: int = 42) -> IsolationForest:
    """Train Isolation Forest on the training patients.
    
    The model only sees the physiological monitoring features.
    """
    matrix = prepare_model_matrix(train_df)[MODEL_FEATURES]
    
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=seed,
        max_samples="auto",
        bootstrap=False,
        n_jobs=1,
    )
    model.fit(matrix.to_numpy(dtype=float))
    return model

def score_observations(model: IsolationForest, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Score the evaluation windows using the trained model.
    
    Returns a DataFrame with 'anomaly_score' and 'predicted_anomaly'.
    """
    results = eval_df.copy()
    
    # Initialize with NaNs/0s for rows that can't be scored (e.g. first window with null deltas)
    results["anomaly_score"] = np.nan
    results["predicted_anomaly"] = 0
    
    valid_mask = results[MODEL_FEATURES].notna().all(axis=1)
    valid_matrix = results.loc[valid_mask, MODEL_FEATURES].to_numpy(dtype=float)
    
    if len(valid_matrix) > 0:
        # Orient anomaly_score so higher = more unusual
        decision = model.decision_function(valid_matrix)
        predicted = model.predict(valid_matrix)
        
        results.loc[valid_mask, "anomaly_score"] = np.round(-decision, 6)
        results.loc[valid_mask, "predicted_anomaly"] = (predicted == -1).astype(int)
        
    return results
