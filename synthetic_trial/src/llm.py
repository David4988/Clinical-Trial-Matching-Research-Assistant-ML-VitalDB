"""Gemini XAI integration for synthetic evaluation."""

import json
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

class SyntheticExplanationSchema(BaseModel):
    """The required structured output schema for synthetic LLM explanations."""
    status: str = Field(
        ..., 
        description="The anomaly status ('anomaly', 'normal', or 'data_gap')"
    )
    summary: str = Field(
        ..., 
        description="A one-sentence summary of the physiological state based on the evidence."
    )
    key_findings: list[str] = Field(
        ..., 
        description="Bullet points describing the most significant physiological changes or stability."
    )
    signals: list[str] = Field(
        ...,
        description="List of the primary signals that drove the model's decision (e.g. ['heart_rate', 'spo2'])."
    )
    evidence: str = Field(
        ...,
        description="A plain English paragraph elaborating on the provided deterministic explanation."
    )
    caveats: str = Field(
        ...,
        description="A mandatory disclaimer stating this is a synthetic engineering validation, not clinical analysis."
    )

class SyntheticGeminiProvider:
    """XAI provider using Google Gemini's structured output for synthetic data."""

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def explain(self, evidence: Dict[str, Any], deterministic_explanation: Dict[str, Any]) -> Optional[SyntheticExplanationSchema]:
        """Translate synthetic deterministic evidence and explanation into an LLM summary."""
        
        system_instruction = (
            "You are a data translation system evaluating a synthetic clinical trial dataset.\n"
            "Your task is to convert the supplied deterministic evidence and explanation into a human-readable structured format.\n"
            "RULES:\n"
            "1. ONLY use the provided evidence. Do NOT invent new measurements.\n"
            "2. Do NOT diagnose the patient. This is synthetic data.\n"
            "3. Describe the physiological changes (deltas) accurately as presented.\n"
            "4. Acknowledge if the model flagged the window or if it was normal.\n"
            "5. Ensure caveats state this is synthetic engineering validation.\n"
        )

        prompt = (
            "Translate the following synthetic evidence and deterministic explanation into the required JSON schema.\n\n"
            f"EVIDENCE:\n{json.dumps(evidence, indent=2)}\n\n"
            f"DETERMINISTIC EXPLANATION:\n{json.dumps(deterministic_explanation, indent=2)}\n"
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=SyntheticExplanationSchema,
            temperature=0.1,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            parsed_dict = json.loads(response.text)
            return SyntheticExplanationSchema.model_validate(parsed_dict)
        except Exception as e:
            # Fallback behavior: log and return None if Gemini fails
            print(f"Gemini generation failed: {e}")
            return None
