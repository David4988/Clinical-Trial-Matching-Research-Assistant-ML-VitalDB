"""Tests for LLM schema validation."""

import pytest
from pydantic import ValidationError
from vitaldb_audit.llm.base import ExplanationSchema


def test_explanation_schema_valid():
    data = {
        "summary": "This is a summary.",
        "key_evidence": ["Point 1", "Point 2"],
        "data_quality": "Data is fine.",
        "uncertainty": "No uncertainty.",
        "not_a_diagnosis": "This is not a diagnosis."
    }
    schema = ExplanationSchema(**data)
    assert schema.summary == "This is a summary."
    assert len(schema.key_evidence) == 2


def test_explanation_schema_missing_fields():
    data = {
        "summary": "This is a summary.",
    }
    with pytest.raises(ValidationError):
        ExplanationSchema(**data)


def test_explanation_schema_invalid_types():
    data = {
        "summary": "This is a summary.",
        "key_evidence": "Not a list",  # Invalid type
        "data_quality": "Data is fine.",
        "uncertainty": "No uncertainty.",
        "not_a_diagnosis": "This is not a diagnosis."
    }
    with pytest.raises(ValidationError):
        ExplanationSchema(**data)
