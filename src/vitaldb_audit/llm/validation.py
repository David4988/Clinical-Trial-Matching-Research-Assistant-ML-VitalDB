"""Validation logic for LLM explanations."""

import re
import math
from typing import Any

from .base import ExplanationSchema

FORBIDDEN_TERMS = [
    "clinical deterioration",
    "adverse event",
    "diagnosis",
    "diagnose",
    "statistically significant",
    "disease",
    "symptom",
    "treatment",
]

class LLMValidationError(Exception):
    """Raised when an LLM explanation fails safety or factual validation."""
    pass


def _extract_numbers(text: str) -> list[float]:
    """Extract all numbers from a string."""
    # Matches integers and decimals
    pattern = r'-?\d+\.\d+|-?\d+'
    matches = re.findall(pattern, text)
    return [float(m) for m in matches]


def _collect_evidence_numbers(evidence: Any) -> set[float]:
    """Recursively collect all numeric values from the evidence dictionary."""
    numbers = set()
    if isinstance(evidence, dict):
        for k, v in evidence.items():
            if k in ("case_id", "window_index", "anomaly_rank"):
                continue  # skip identifiers
            numbers.update(_collect_evidence_numbers(v))
    elif isinstance(evidence, list):
        for item in evidence:
            numbers.update(_collect_evidence_numbers(item))
    elif isinstance(evidence, (int, float)) and not isinstance(evidence, bool):
        numbers.add(float(evidence))
    return numbers


def validate_explanation(explanation: ExplanationSchema, evidence: dict):
    """Validate an explanation for forbidden terms and numerical hallucinations.

    Args:
        explanation: The parsed Pydantic schema from the LLM.
        evidence: The raw deterministic evidence object.

    Raises:
        LLMValidationError: If validation fails.
    """
    # 1. Check forbidden terms (excluding the disclaimer field)
    text_to_check = (
        explanation.summary + " " +
        " ".join(explanation.key_evidence) + " " +
        explanation.data_quality + " " +
        explanation.uncertainty
    ).lower()

    for term in FORBIDDEN_TERMS:
        if term in text_to_check:
            raise LLMValidationError(f"Forbidden term found in explanation: '{term}'")

    # 2. Verify disclaimer presence
    disclaimer = explanation.not_a_diagnosis.lower()
    if "diagnosis" not in disclaimer and "adverse event" not in disclaimer:
        raise LLMValidationError("Explanation is missing the mandatory non-diagnosis disclaimer.")

    # 3. Lightweight numeric verification
    # Extract all numbers from the explanation (excluding disclaimer and data quality)
    content_text = explanation.summary + " " + " ".join(explanation.key_evidence)
    llm_numbers = _extract_numbers(content_text)
    
    evidence_numbers = _collect_evidence_numbers(evidence)
    
    # We will only strictly check numbers that have decimals or are large, 
    # to avoid flagging harmless small integers like '1', '2' or '3' used for counting signals.
    for num in llm_numbers:
        # Ignore small integers often used for counting
        if num.is_integer() and -10 <= num <= 10:
            continue
            
        # Check if the number is close to any number in the evidence
        # (allowing for rounding, e.g., LLM says 101.5 but evidence is 101.46)
        # Also check absolute values, because LLM often says "decreased by 1.8" when evidence is -1.8
        matched = False
        for ev_num in evidence_numbers:
            if math.isclose(abs(num), abs(ev_num), rel_tol=0.01, abs_tol=0.1):
                matched = True
                break
        
        if not matched:
            raise LLMValidationError(
                f"Numerical hallucination detected: The number {num} was found in the "
                "explanation but does not match any value in the supplied evidence."
            )
