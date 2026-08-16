import os
import json
import pytest
from unittest import mock

from vitaldb_audit.llm.base import ExplanationSchema
from vitaldb_audit.llm.validation import validate_explanation, LLMValidationError
from vitaldb_audit.llm.pipeline import process_evidence_batch

@pytest.fixture
def sample_evidence():
    return {
        "case_id": 4,
        "window_index": 62,
        "anomaly_rank": 1,
        "anomaly_score": 0.23,
        "time_range": {"label": "310.0-315.0 min"},
        "signals": {
            "hr": {"current_mean": 99.7, "std": 0.7, "min": 98.0, "max": 101.0, "delta": -1.8, "previous_usable_mean": 101.5, "usable": True},
            "spo2": {"current_mean": 99.3, "std": 3.1, "min": 81.0, "max": 100.0, "delta": -0.6, "previous_usable_mean": 99.9, "usable": True},
            "rr": {"current_mean": 16.8, "std": 8.9, "min": 5.0, "max": 33.0, "delta": 5.3, "previous_usable_mean": 11.5, "usable": True},
        },
        "observations": {
            "dispersion_basis": {
                "hr": {"threshold": 1.0, "current_std": 0.7},
                "spo2": {"threshold": 1.0, "current_std": 3.1},
                "rr": {"threshold": 2.6, "current_std": 8.9},
            },
            "change_basis": {
                "hr": {"trend_score": -1.0},
                "spo2": {"trend_score": -0.5},
                "rr": {"trend_score": 1.5},
            },
            "hr_dispersion_unusual": False,
            "spo2_dispersion_unusual": True,
            "rr_dispersion_unusual": True,
            "hr_changed": True,
        }
    }


def test_validation_success(sample_evidence):
    explanation = ExplanationSchema(
        summary="HR decreased by 1.8. SpO2 had std 3.1.",
        key_evidence=["SpO2 min was 81.0."],
        data_quality="Good.",
        uncertainty="Statistical.",
        not_a_diagnosis="This is not a diagnosis or adverse event."
    )
    # Should not raise
    validate_explanation(explanation, sample_evidence)


def test_validation_forbidden_term(sample_evidence):
    explanation = ExplanationSchema(
        summary="Patient shows clinical deterioration.",
        key_evidence=["HR decreased."],
        data_quality="Good.",
        uncertainty="Statistical.",
        not_a_diagnosis="This is not a diagnosis or adverse event."
    )
    with pytest.raises(LLMValidationError, match="Forbidden term"):
        validate_explanation(explanation, sample_evidence)


def test_validation_missing_disclaimer(sample_evidence):
    explanation = ExplanationSchema(
        summary="HR decreased.",
        key_evidence=["HR decreased."],
        data_quality="Good.",
        uncertainty="Statistical.",
        not_a_diagnosis="Just numbers."
    )
    with pytest.raises(LLMValidationError, match="missing the mandatory non-diagnosis disclaimer"):
        validate_explanation(explanation, sample_evidence)


def test_validation_numerical_hallucination(sample_evidence):
    explanation = ExplanationSchema(
        summary="HR decreased to 150.0.", # 150.0 not in evidence
        key_evidence=["HR decreased."],
        data_quality="Good.",
        uncertainty="Statistical.",
        not_a_diagnosis="This is not a diagnosis or adverse event."
    )
    with pytest.raises(LLMValidationError, match="Numerical hallucination detected"):
        validate_explanation(explanation, sample_evidence)


@mock.patch("vitaldb_audit.llm.pipeline.GeminiProvider")
def test_pipeline_deterministic_mode(mock_gemini_provider, sample_evidence, tmp_path):
    with mock.patch.dict(os.environ, {"XAI_MODE": "deterministic"}, clear=True):
        cache_file = str(tmp_path / "cache.json")
        results, metrics = process_evidence_batch([sample_evidence], cache_file)
        
        assert len(results) == 1
        assert results[0]["llm_status"] == "skipped"
        assert results[0]["llm_explanation"] is None
        assert "headline" in results[0]["frontend_ready"]
        
        mock_gemini_provider.assert_not_called()
        assert metrics["total"] == 1
        assert metrics["success"] == 0


@mock.patch("vitaldb_audit.llm.pipeline.GeminiProvider")
def test_pipeline_hybrid_mode_success(mock_gemini_provider, sample_evidence, tmp_path):
    # Mock LLM response
    mock_llm_resp = ExplanationSchema(
        summary="HR 99.7.",
        key_evidence=["SpO2 99.3."],
        data_quality="Good.",
        uncertainty="Stat.",
        not_a_diagnosis="Not a diagnosis."
    )
    mock_provider_instance = mock.Mock()
    mock_provider_instance.explain.return_value = mock_llm_resp
    mock_gemini_provider.return_value = mock_provider_instance
    
    with mock.patch.dict(os.environ, {"XAI_MODE": "hybrid", "GEMINI_API_KEY": "test"}, clear=True):
        cache_file = str(tmp_path / "cache.json")
        results, metrics = process_evidence_batch([sample_evidence], cache_file)
        
        assert len(results) == 1
        assert results[0]["llm_status"] == "success"
        assert results[0]["llm_explanation"]["summary"] == "HR 99.7."
        
        # Test Cache Hit on second run
        results_2, metrics_2 = process_evidence_batch([sample_evidence], cache_file)
        assert metrics_2["cache_hits"] == 1
        assert mock_provider_instance.explain.call_count == 1


@mock.patch("vitaldb_audit.llm.pipeline.GeminiProvider")
def test_pipeline_hybrid_mode_fallback(mock_gemini_provider, sample_evidence, tmp_path):
    mock_provider_instance = mock.Mock()
    # Simulate API failure
    mock_provider_instance.explain.side_effect = Exception("API down")
    mock_gemini_provider.return_value = mock_provider_instance
    
    with mock.patch.dict(os.environ, {"XAI_MODE": "hybrid", "GEMINI_API_KEY": "test"}, clear=True):
        cache_file = str(tmp_path / "cache.json")
        results, metrics = process_evidence_batch([sample_evidence], cache_file)
        
        assert len(results) == 1
        assert results[0]["llm_status"] == "failed"
        assert "API Error: API down" in results[0]["error"]
        
        # Frontend ready should fallback to deterministic
        assert results[0]["frontend_ready"]["summary"] != ""
