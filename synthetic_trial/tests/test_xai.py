"""Tests for synthetic XAI components."""

import pytest
from unittest.mock import Mock, patch

from synthetic_trial.src.evidence import format_synthetic_evidence
from synthetic_trial.src.explain import explain_synthetic_window
from synthetic_trial.src.xai_validation import validate_scenario_consistency
from synthetic_trial.src.llm import SyntheticGeminiProvider, SyntheticExplanationSchema
import pandas as pd

@pytest.fixture
def sample_evidence():
    return {
        "patient_id": "P1",
        "scenario": "SUDDEN_DETERIORATION",
        "ground_truth_state": "acute_change",
        "predicted_anomaly": 1,
        "anomaly_score": 0.25,
        "coverage_percent": 100.0,
        "data_quality": "GOOD",
        "signals": {
            "hr": {"current_mean": 100.0, "delta": 20.0},
            "spo2": {"current_mean": 90.0, "delta": -5.0},
            "rr": {"current_mean": 24.0, "delta": 8.0}
        },
        "strongest_signal": "heart_rate"
    }

def test_deterministic_explanation_acute_change(sample_evidence):
    explanation = explain_synthetic_window(sample_evidence)
    assert explanation["status"] == "anomaly"
    assert explanation["explanation_type"] == "acute_physiological_change"
    assert "heart_rate" in explanation["key_signals"]
    assert "spo2" in explanation["key_signals"]
    assert "HEART_RATE increasing" in explanation["direction"]
    assert "SPO2 decreasing" in explanation["direction"]
    assert "heart_rate" in explanation["evidence_summary"]

def test_deterministic_explanation_stable():
    evidence = {
        "scenario": "STABLE",
        "ground_truth_state": "normal",
        "predicted_anomaly": 0,
        "data_quality": "GOOD",
        "signals": {
            "hr": {"current_mean": 70.0, "delta": 0.05},
            "spo2": {"current_mean": 98.0, "delta": -0.01},
            "rr": {"current_mean": 14.0, "delta": 0.0}
        }
    }
    explanation = explain_synthetic_window(evidence)
    assert explanation["status"] == "normal"
    assert explanation["explanation_type"] == "physiological_stability"
    assert len(explanation["key_signals"]) == 0

def test_deterministic_explanation_data_gap():
    evidence = {
        "predicted_anomaly": 0,
        "coverage_percent": 60.0,
        "data_quality": "PARTIAL",
        "signals": {
            "hr": {"delta": None}, "spo2": {"delta": None}, "rr": {"delta": None}
        }
    }
    explanation = explain_synthetic_window(evidence)
    assert explanation["status"] == "data_gap"
    assert explanation["explanation_type"] == "data_quality_gap"

def test_scenario_validation_success(sample_evidence):
    explanation = explain_synthetic_window(sample_evidence)
    val = validate_scenario_consistency(sample_evidence, explanation)
    assert val["evidence_completeness"] is True
    assert val["scenario_consistency"] is True
    assert "Correctly identified acute change" in val["validation_reason"]

def test_scenario_validation_failure():
    # Model missed the sudden deterioration
    evidence = {
        "scenario": "SUDDEN_DETERIORATION",
        "ground_truth_state": "acute_change",
        "predicted_anomaly": 0,
        "data_quality": "GOOD",
        "signals": {
            "hr": {"current_mean": 70.0, "delta": 0.05},
            "spo2": {"current_mean": 98.0, "delta": -0.01},
            "rr": {"current_mean": 14.0, "delta": 0.0}
        }
    }
    explanation = explain_synthetic_window(evidence)
    val = validate_scenario_consistency(evidence, explanation)
    assert val["scenario_consistency"] is False
    assert "Failed to identify acute physiological change" in val["validation_reason"]

@patch("google.genai.Client")
def test_gemini_integration_valid(mock_client, sample_evidence):
    # Mock the LLM client
    mock_instance = mock_client.return_value
    mock_response = Mock()
    mock_response.text = '''{
        "status": "anomaly",
        "summary": "Patient exhibits acute changes.",
        "key_findings": ["HR up", "SpO2 down"],
        "signals": ["heart_rate", "spo2"],
        "evidence": "Detailed evidence string.",
        "caveats": "This is synthetic engineering validation."
    }'''
    mock_instance.models.generate_content.return_value = mock_response

    provider = SyntheticGeminiProvider(api_key="fake")
    explanation = explain_synthetic_window(sample_evidence)
    
    result = provider.explain(sample_evidence, explanation)
    
    assert result is not None
    assert result.status == "anomaly"
    assert "synthetic engineering validation" in result.caveats

@patch("google.genai.Client")
def test_gemini_integration_fallback(mock_client, sample_evidence):
    # Mock failure
    mock_instance = mock_client.return_value
    mock_instance.models.generate_content.side_effect = Exception("API error")

    provider = SyntheticGeminiProvider(api_key="fake")
    explanation = explain_synthetic_window(sample_evidence)
    
    result = provider.explain(sample_evidence, explanation)
    assert result is None  # Handled gracefully
