"""Google Gemini provider implementation for LLM XAI."""

import json
from typing import Any, Dict

from google import genai
from google.genai import types

from .base import ExplanationSchema, LLMProvider


class GeminiProvider(LLMProvider):
    """XAI provider using Google Gemini's structured output."""

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def explain(self, evidence: Dict[str, Any]) -> ExplanationSchema:
        """Translate structured evidence into a structured explanation."""
        
        system_instruction = (
            "You are a strict data-translation system. "
            "Your task is to convert the supplied physiological monitoring evidence JSON "
            "into a plain-English structured summary.\n"
            "RULES:\n"
            "1. Use ONLY the supplied evidence. Do NOT invent measurements or events.\n"
            "2. Do NOT diagnose the patient. Do NOT claim an 'adverse event' or 'clinical deterioration'.\n"
            "3. Distinguish statistical unusualness (e.g. 'unusually large variability', 'mean shifted') from clinical interpretation.\n"
            "4. Mention data-quality limitations if they exist in the evidence (e.g. partial coverage, reference gaps).\n"
            "5. Preserve numerical values accurately.\n"
        )

        prompt = (
            "Translate the following evidence object into the required JSON schema.\n\n"
            f"{json.dumps(evidence, indent=2)}"
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ExplanationSchema,
            temperature=0.1,  # Low temperature for deterministic/factual translation
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        # The SDK returns JSON string that matches the schema
        # We parse it into the Pydantic model to ensure it is perfectly valid
        parsed_dict = json.loads(response.text)
        return ExplanationSchema.model_validate(parsed_dict)
