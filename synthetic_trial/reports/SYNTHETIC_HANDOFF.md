# Synthetic Workstream: Team Handoff

## 1. What is Finished
The complete synthetic clinical-trial simulator and evaluation environment is ready for use. This includes:
- **Data Generation**: Deterministic physiological data injection mapping directly to realistic states (e.g. STABLE, SUDDEN_DETERIORATION, ADVERSE_EVENT).
- **Evaluation Harness**: Automated splitting, tracking, and evaluation of anomaly detection pipelines at both the window-level and patient-event level.
- **XAI Validation**: A scenario-grounded deterministic extraction pipeline that evaluates both mathematical accuracy and Gemini LLM integration fallback behavior.

## 2. Recommended Baseline
- **Model**: Isolation Forest
- **Training Cohort**: Clean `STABLE` + `IMPROVING` patients only (do not train on data containing anomalies).
- **Contamination**: `c=0.10` (chosen as the best balance between catching 100% of ADVERSE_EVENTS while minimizing FPR).
- **LLM Settings**: `gemini-3.6-flash`, low temperature (`0.1`), structured output (Pydantic schema).

## 3. Where Outputs Are
All critical artifacts have been consolidated:
- **Final MD Report**: `synthetic_trial/reports/SYNTHETIC_EVALUATION_FINAL.md`
- **Final JSON Data**: `synthetic_trial/reports/SYNTHETIC_EVALUATION_FINAL.json`
- **XAI Demo Cases**: `synthetic_trial/reports/synthetic_demo_cases.json`
- **Raw Metric Comparisons**: `synthetic_trial/reports/model_baseline_comparison.json`

## 4. What Should NOT Be Changed
- **Do NOT** change the synthetic generator parameters or scenario logic. It is locked to provide a consistent benchmark.
- **Do NOT** change the evaluation-label mapping (`assign_ground_truth_labels`).
- **Do NOT** use synthetic performance to claim clinical validity on the real VitalDB dataset. The synthetic data serves as an engineering testbed to validate software logic, not medical algorithms.

## 5. What Remains for Final-Product Integration
- **Dashboard Hookup**: The JSON metrics and explanations produced by this pipeline must be wired into the frontend React dashboard.
- **Real VitalDB Training**: The real VitalDB models must be re-trained and evaluated, as this pipeline only proves the *concept* of the architecture, not the real-world weights.
- **LLM Prompt Tuning (VitalDB)**: The current Gemini prompt is optimized for the deterministic features available in the synthetic set; the real VitalDB explainer will need to incorporate intra-window variance statistics (`std`, `min`, `max`).
