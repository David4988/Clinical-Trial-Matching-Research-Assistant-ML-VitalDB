"""Tests for LLM providers."""

import json
from unittest import mock

import pytest
from google.genai.types import GenerateContentResponse
from vitaldb_audit.llm.base import ExplanationSchema
from vitaldb_audit.llm.gemini import GeminiProvider


class MockGenerateContentResponse:
    def __init__(self, text):
        self.text = text


@mock.patch("google.genai.Client")
def test_gemini_provider_success(mock_client_class):
    # Setup mock
    mock_client = mock.MagicMock()
    mock_client_class.return_value = mock_client
    
    expected_response = {
        "summary": "Mock summary.",
        "key_evidence": ["Mock point"],
        "data_quality": "Mock quality.",
        "uncertainty": "Mock uncertainty.",
        "not_a_diagnosis": "Mock disclaimer."
    }
    
    mock_client.models.generate_content.return_value = MockGenerateContentResponse(
        text=json.dumps(expected_response)
    )
    
    # Initialize provider
    provider = GeminiProvider(api_key="mock_key")
    
    # Execute
    evidence = {"case_id": 4, "anomaly_score": 0.5}
    result = provider.explain(evidence)
    
    # Assert
    assert isinstance(result, ExplanationSchema)
    assert result.summary == "Mock summary."
    assert result.key_evidence == ["Mock point"]
    
    # Verify the call to the client
    mock_client.models.generate_content.assert_called_once()
    kwargs = mock_client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "gemini-3.6-flash"
    assert "translate the following evidence" in kwargs["contents"].lower()
    
    # Verify config
    config = kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema == ExplanationSchema
    assert config.temperature == 0.1
