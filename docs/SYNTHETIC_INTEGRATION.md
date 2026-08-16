# Synthetic Demonstration Integration

## Purpose
This document describes how the static outputs from the ML synthetic experimentation repository are consumed by the final application. The purpose of this boundary is to allow the application dashboard to demonstrate realistic ML anomalies and LLM explanations (XAI) without carrying the dependencies or computational weight of the research pipeline.

## Data Flow
```text
Synthetic Research Repository
        ↓
generated synthetic_demo_cases.json
        ↓
copied and versioned into the application backend (backend/app/synthetic/artifacts)
        ↓
SyntheticArtifactProvider
        ↓
Application RiskAssessment
        ↓
Dashboard
```

## Provenance
All risk outputs originating from this adapter are explicitly labeled:
`source = synthetic`
This ensures the final application's demonstration modes are never accidentally confused for real clinical data or real patient monitoring anomalies.

## Demo Activation
The synthetic data is activated via the existing `/monitoring/demo/seed` endpoint by passing the flag:
`use_synthetic_ml: bool = true`
If omitted or `false`, the standard deterministic application mock provider is used.

## Update Process
If the research repository produces updated synthetic behaviors or better ML baselines, the artifact must be manually copied:
```bash
cp synthetic_trial/reports/synthetic_demo_cases.json ../Clinical-Trial-Matching---Research-Assistant/backend/app/synthetic/artifacts/
```
No automatic synchronization or background polling is implemented to prevent brittle cross-repository dependencies.

## Limitations
The outputs of this integration are **scenario-grounded engineering demonstrations** meant to evaluate the structural integrity of the application dashboard. They do not constitute clinical validation and are generated purely via deterministic physiological perturbation patterns rather than real biological variance.
