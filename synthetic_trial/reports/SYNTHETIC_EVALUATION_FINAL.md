# Synthetic Evaluation Final Report

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
- **Precision**: 50.44%
- **Recall (Detection Rate)**: 67.41%
- **F1 Score**: 57.71%
- **False Positive Rate**: 18.41%

### Scenario Highlights
- **Sudden Deterioration Recall**: 99.91%
- **Adverse Event Recall**: 100.00%
- **Stable FPR**: 12.09%

### Detection Delay
- **Detected Events**: 66
- **Missed Events**: 0
- **Mean Delay**: 0.0 minutes

## XAI Results
- **Evidence Schema**: Deterministic JSON containing signals, deltas, and data quality labels.
- **Deterministic Explainer**: Pure python rule-based text generation based on delta magnitude.
- **Gemini Adapter**: `SyntheticGeminiProvider` using Google GenAI SDK and Pydantic structured schema.
- **Fallback Behavior**: Gracefully returns `None` and falls back to deterministic summary on API failure (e.g. 429 Rate Limit).
- **Generation Success Rate**: 100.00%
- **Scenario Consistency Rate**: 70.62% (False positives during STABLE periods correctly drive down consistency since they can't be explained cleanly by physiological deterioration).

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
1. Generate data: `PYTHONPATH=. .venv/bin/python synthetic_trial/src/generator.py`
2. Run evaluation: `PYTHONPATH=. .venv/bin/python generate_synthetic_evaluation.py`
3. Run contamination experiment: `PYTHONPATH=. .venv/bin/python run_contamination_experiment.py`
4. Generate XAI validation: `PYTHONPATH=. .venv/bin/python synthetic_trial/generate_xai_report.py`
5. Generate demo cases: `PYTHONPATH=. .venv/bin/python synthetic_trial/generate_synthetic_demo.py`
6. Run test suite: `PYTHONPATH=. .venv/bin/pytest synthetic_trial/tests/ -v`

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
