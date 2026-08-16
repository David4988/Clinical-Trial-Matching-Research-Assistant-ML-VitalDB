"""Base abstraction for XAI LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel, Field


class ExplanationSchema(BaseModel):
    """The required structured output schema for all LLM explanations."""
    summary: str = Field(
        ...,
        description="A one or two sentence plain-English summary of what the evidence shows."
    )
    key_evidence: list[str] = Field(
        ...,
        description="A list of bullet points detailing the specific physiological changes (or lack thereof) identified in the evidence."
    )
    data_quality: str = Field(
        ...,
        description="A brief note on data quality, coverage, or gaps. Say 'Data quality is sufficient' if no issues are present."
    )
    uncertainty: str = Field(
        ...,
        description="A statement on the statistical uncertainty or limitations of the evidence."
    )
    not_a_diagnosis: str = Field(
        ...,
        description="A mandatory disclaimer stating that this is a statistical analysis of monitoring data and NOT a clinical diagnosis or determination of an adverse event."
    )


class LLMProvider(ABC):
    """Abstract base class for LLM XAI providers."""

    @abstractmethod
    def explain(self, evidence: Dict[str, Any]) -> ExplanationSchema:
        """Translate structured evidence into a structured explanation.

        Args:
            evidence: The raw evidence dictionary produced by the deterministic
                      Isolation Forest evidence layer.

        Returns:
            ExplanationSchema: The structured explanation.
        """
        pass
