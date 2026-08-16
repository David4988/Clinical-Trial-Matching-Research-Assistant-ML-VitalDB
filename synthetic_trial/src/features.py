"""Feature extraction for synthetic observations."""

import pandas as pd

CORE_SIGNALS = ["heart_rate", "spo2", "respiratory_rate"]

def build_features(observations_df: pd.DataFrame) -> pd.DataFrame:
    """Build synthetic monitoring features (means and deltas).
    
    Args:
        observations_df: DataFrame of raw synthetic observations.
        
    Returns:
        DataFrame with added delta columns for each core signal.
    """
    if observations_df.empty:
        return observations_df
        
    df = observations_df.copy()
    
    # Ensure sorted by patient and time
    df = df.sort_values(["patient_id", "window_index"]).reset_index(drop=True)
    
    # Compute deltas from previous window, explicitly isolated per patient
    grouped = df.groupby("patient_id")
    
    for signal in CORE_SIGNALS:
        prev_val = grouped[signal].shift(1)
        # Delta is current - previous
        df[f"{signal}_delta"] = (df[signal] - prev_val).round(4)
        
    return df
