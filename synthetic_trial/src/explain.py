"""Deterministic rule-based XAI layer for synthetic evaluation.

Translates synthetic evidence dictionaries into structured explanations.
Does NOT use the ground truth scenario to generate explanations.
"""

def explain_synthetic_window(evidence: dict) -> dict:
    """Generate a deterministic explanation for a synthetic evidence object."""
    
    # 1. Determine Status
    pred = evidence.get("predicted_anomaly")
    status = "anomaly" if pred == 1 else "normal"
    
    # Check for data quality exception
    if evidence.get("data_quality") != "GOOD":
        status = "data_gap"
    
    # 2. Extract Signals & Directions
    key_signals = []
    directions = []
    
    for sig_key, sig_label in [("hr", "heart_rate"), ("spo2", "spo2"), ("rr", "respiratory_rate")]:
        sig_data = evidence.get("signals", {}).get(sig_key, {})
        delta = sig_data.get("delta")
        
        if delta is not None and abs(delta) > 0.1:  # Small threshold to filter float noise
            key_signals.append(sig_label)
            if delta > 0:
                directions.append(f"{sig_label.upper()} increasing")
            else:
                directions.append(f"{sig_label.upper()} decreasing")
                
    # 3. Determine Explanation Type
    explanation_type = "physiological_stability"
    if status == "data_gap":
        explanation_type = "data_quality_gap"
    elif status == "anomaly":
        if len(key_signals) > 0:
            explanation_type = "acute_physiological_change"
        else:
            explanation_type = "unexplained_anomaly"
    else:
        # normal
        if len(key_signals) > 0:
            # check if changes were small
            explanation_type = "minor_physiological_fluctuation"
            
    # 4. Construct Summary
    if explanation_type == "data_quality_gap":
        cov = evidence.get('coverage_percent', 0.0)
        summary = f"Window excluded or flagged due to data quality issues (coverage: {cov:.1f}%)."
    elif explanation_type == "physiological_stability":
        summary = "Model scored window as normal; physiological signals show stability."
    elif explanation_type == "minor_physiological_fluctuation":
        summary = "Model scored window as normal; minor fluctuations observed in physiological signals."
    elif explanation_type == "acute_physiological_change":
        strongest = evidence.get("strongest_signal", "none")
        summary = (f"Model flagged window as anomalous. Evidence shows significant changes "
                   f"in {', '.join(key_signals)}, most prominently {strongest}.")
    else:
        summary = "Model flagged window, but clear physiological drivers could not be extracted from delta features."

    return {
        "status": status,
        "explanation_type": explanation_type,
        "key_signals": key_signals,
        "direction": directions,
        "evidence_summary": summary,
        "anomaly_score": evidence.get("anomaly_score")
    }
