import json
from pathlib import Path

def main():
    REPORTS_DIR = Path("synthetic_trial/reports")
    
    # Load comparison stats
    with open(REPORTS_DIR / "model_baseline_comparison.json") as f:
        baseline_comp = json.load(f)
        
    c10 = baseline_comp["detailed_reports"]["c_0.1"]
    
    # Load XAI stats
    with open(REPORTS_DIR / "xai_validation.json") as f:
        xai_stats = json.load(f)
        
    # Construct final JSON
    final_json = {
        "dataset": {
            "train_patients": 102,
            "eval_patients": 250,
            "train_windows": 14688,
            "eval_windows": 36000,
            "scenarios": ["STABLE", "IMPROVING", "GRADUAL_DETERIORATION", "SUDDEN_DETERIORATION", "RECOVERY", "ADVERSE_EVENT", "DATA_QUALITY_FAILURE"],
            "ground_truth_states": ["normal", "improving", "improved", "deteriorating", "acute_change", "adverse_event", "recovering", "recovered", "post_event", "data_gap"]
        },
        "model": {
            "type": "IsolationForest",
            "training_cohort": "STABLE + IMPROVING",
            "features": ["heart_rate", "spo2", "respiratory_rate", "heart_rate_delta", "spo2_delta", "respiratory_rate_delta"],
            "contamination": 0.10,
            "random_seed": 42
        },
        "metrics": {
            "precision": c10["overall"]["precision"],
            "recall": c10["overall"]["recall_detection_rate"],
            "f1": c10["overall"]["f1"],
            "fpr": c10["overall"]["false_positive_rate"]
        },
        "per_scenario": c10["scenarios"],
        "detection_delay": c10["delay"],
        "xai": {
            "evidence_completeness_rate": xai_stats["evidence_completeness_rate"],
            "scenario_consistency_rate": xai_stats["scenario_consistency_rate"],
            "explanation_generation_success_rate": xai_stats["explanation_generation_success_rate"],
            "gemini_adapter": "implemented",
            "fallback_behavior": "graceful fallback to deterministic layer on API failure"
        },
        "limitations": [
            "Deterministic physiological shifts",
            "Trivial separability of acute events (SUDDEN_DETERIORATION, ADVERSE_EVENT)",
            "Lack of real-world noisy variance",
            "Requires artificial contamination setting"
        ],
        "reproducibility": {
            "generate_data": "PYTHONPATH=. .venv/bin/python synthetic_trial/src/generator.py",
            "run_evaluation": "PYTHONPATH=. .venv/bin/python generate_synthetic_evaluation.py",
            "run_contamination_experiment": "PYTHONPATH=. .venv/bin/python run_contamination_experiment.py",
            "generate_xai_validation": "PYTHONPATH=. .venv/bin/python synthetic_trial/generate_xai_report.py",
            "generate_demo": "PYTHONPATH=. .venv/bin/python synthetic_trial/generate_synthetic_demo.py",
            "run_tests": "PYTHONPATH=. .venv/bin/pytest synthetic_trial/tests/ -v"
        }
    }
    
    with open(REPORTS_DIR / "SYNTHETIC_EVALUATION_FINAL.json", "w") as f:
        json.dump(final_json, f, indent=2)
        
    # Construct final Markdown
    md = f"""# Synthetic Evaluation Final Report

## Executive Summary
The synthetic simulator provides an isolated, deterministic environment for evaluating the VitalDB anomaly detection stack. Because real VitalDB clinical data lacks perfect ground-truth labels for subtle physiological deterioration, this synthetic dataset acts as a tightly controlled engineering testbed. 

The monitoring model evaluates the core capability of an Isolation Forest to detect physiological deviations from a learned baseline. The XAI layer evaluates whether deterministic logic and an optional Gemini LLM adapter can accurately and consistently summarize the mathematical evidence driving those anomaly detections. 

**This is scenario-grounded engineering evaluation, not clinical validation.**

## Dataset
- **Patients**: 102 Train (clean cohort), 250 Eval
- **Windows per patient**: ~144 (12 hours of 5-minute windows)
- **Scenarios**: STABLE, IMPROVING, GRADUAL_DETERIORATION, SUDDEN_DETERIORATION, RECOVERY, ADVERSE_EVENT, DATA_QUALITY_FAILURE
- **Ground-truth states**: normal, improving, improved, deteriorating, acute_change, adverse_event, recovering, recovered, post_event, data_gap

## Model
- **Training Cohort**: Patients from STABLE and IMPROVING scenarios only.
- **Feature Set**: heart_rate, spo2, respiratory_rate, heart_rate_delta, spo2_delta, respiratory_rate_delta
- **Model Type**: Isolation Forest
- **Contamination**: 0.10
- **Random Seed**: 42

## Evaluation Results
- **Precision**: {final_json['metrics']['precision']*100:.2f}%
- **Recall (Detection Rate)**: {final_json['metrics']['recall']*100:.2f}%
- **F1 Score**: {final_json['metrics']['f1']*100:.2f}%
- **False Positive Rate**: {final_json['metrics']['fpr']*100:.2f}%

### Scenario Highlights
- **Sudden Deterioration Recall**: {final_json['per_scenario']['SUDDEN_DETERIORATION']['recall_detection_rate']*100:.2f}%
- **Adverse Event Recall**: {final_json['per_scenario']['ADVERSE_EVENT']['recall_detection_rate']*100:.2f}%
- **Stable FPR**: {final_json['per_scenario']['STABLE']['false_positive_rate']*100:.2f}%

### Detection Delay
- **Detected Events**: {final_json['detection_delay']['detected_cases']}
- **Missed Events**: {final_json['detection_delay']['missed_cases']}
- **Mean Delay**: {final_json['detection_delay']['mean_delay_minutes']:.1f} minutes

## XAI Results
- **Evidence Schema**: Deterministic JSON containing signals, deltas, and data quality labels.
- **Deterministic Explainer**: Pure python rule-based text generation based on delta magnitude.
- **Gemini Adapter**: `SyntheticGeminiProvider` using Google GenAI SDK and Pydantic structured schema.
- **Fallback Behavior**: Gracefully returns `None` and falls back to deterministic summary on API failure (e.g. 429 Rate Limit).
- **Generation Success Rate**: {final_json['xai']['explanation_generation_success_rate']*100:.2f}%
- **Scenario Consistency Rate**: {final_json['xai']['scenario_consistency_rate']*100:.2f}% (False positives during STABLE periods correctly drive down consistency since they can't be explained cleanly by physiological deterioration).

## Engineering Interpretation
### Demonstrated
- Reproducible synthetic generation
- Ground-truth evaluation
- Anomaly detection pipeline
- Event detection measurement
- Deterministic evidence extraction
- Structured explanation
- Scenario-grounded XAI validation

### Not Demonstrated
- Clinical validity
- Clinical accuracy
- Real-world generalization
- Causal interpretation
- Safety for clinical decision-making

## Limitations
- Deterministic physiological shifts
- Trivial separability of some events
- Lack of real-world noise
- Contamination choices
- Synthetic scenario assumptions

## Reproducibility Commands
1. Generate data: `{final_json['reproducibility']['generate_data']}`
2. Run evaluation: `{final_json['reproducibility']['run_evaluation']}`
3. Run contamination experiment: `{final_json['reproducibility']['run_contamination_experiment']}`
4. Generate XAI validation: `{final_json['reproducibility']['generate_xai_validation']}`
5. Generate demo cases: `{final_json['reproducibility']['generate_demo']}`
6. Run test suite: `{final_json['reproducibility']['run_tests']}`

## File Map
| File | Purpose |
|------|---------|
| `synthetic_trial/src/generator.py` | Synthetic trial generation |
| `synthetic_trial/src/features.py` | Monitoring features |
| `synthetic_trial/src/model.py` | Isolation Forest |
| `synthetic_trial/src/evaluate.py` | Metrics |
| `synthetic_trial/src/evidence.py` | Evidence extraction |
| `synthetic_trial/src/explain.py` | Deterministic XAI |
| `synthetic_trial/src/xai_validation.py` | Scenario validation |
| `synthetic_trial/src/llm.py` | Gemini adapter |
"""
    with open(REPORTS_DIR / "SYNTHETIC_EVALUATION_FINAL.md", "w") as f:
        f.write(md)

if __name__ == "__main__":
    main()
