# LLM API Provider Comparison for XAI

This document compares three major LLM providers (Google Gemini, OpenAI, Anthropic) for translating structured deterministic evidence from the VitalDB Isolation Forest anomaly detector into a richer, human-readable structured explanation.

**Goal**: Convert a JSON evidence object into a structured explanation JSON object containing `summary`, `key_evidence`, `data_quality`, `uncertainty`, and `not_a_diagnosis` fields, relying entirely on the provided evidence and refraining from clinical interpretation.

## 1. Google Gemini API

**Recommended SDK:** `google-genai` (The newest official SDK)
**Recommended Model:** `gemini-2.5-flash`
**Authentication:** Environment variable `GEMINI_API_KEY`
**Local Development:** Very easy via `google-genai` and `pytest`.

* **Structured JSON / JSON Schema Support:** Excellent. The `google-genai` SDK natively supports passing Pydantic models directly to the `response_schema` parameter in `GenerateContentConfig`.
* **Latency:** `gemini-2.5-flash` is extremely fast (typically <1.5s for small payloads), making it ideal for processing explanation batches or serving real-time requests if needed.
* **Cost/Usage:** Very inexpensive for small context windows. Has generous rate limits for standard usage.
* **Suitability:** **Highest.** The combination of fast latency, low cost, native Pydantic support, and ease of integration makes it the strongest candidate for this hackathon spike.

## 2. OpenAI API

**Recommended SDK:** `openai`
**Recommended Model:** `gpt-4o-mini`
**Authentication:** Environment variable `OPENAI_API_KEY`
**Local Development:** Easy.

* **Structured JSON / JSON Schema Support:** Excellent. OpenAI's "Structured Outputs" feature (`response_format={"type": "json_schema"}`) guarantees exact schema adherence. The Python SDK supports parsing directly into Pydantic models via `client.beta.chat.completions.parse`.
* **Latency:** `gpt-4o-mini` is extremely fast and comparable to Gemini Flash.
* **Cost/Usage:** Very inexpensive. Rate limits are generally high.
* **Suitability:** **High.** OpenAI offers robust structured output, but requires `client.beta` endpoints for native Pydantic parsing. It is a solid fallback option if Gemini is not used.

## 3. Anthropic API

**Recommended SDK:** `anthropic`
**Recommended Model:** `claude-3-5-haiku-20241022` or `claude-3-5-sonnet-20241022`
**Authentication:** Environment variable `ANTHROPIC_API_KEY`
**Local Development:** Easy.

* **Structured JSON / JSON Schema Support:** Good, but less direct. Anthropic doesn't have a native Pydantic parser wrapper in its SDK like OpenAI or Gemini. Structured output is typically achieved either by explicit system prompting (e.g., "Return ONLY JSON") or by forcing tool use. Third-party libraries like `instructor` are often used to bridge this gap.
* **Latency:** Haiku is very fast, though Sonnet is slightly slower but highly capable.
* **Cost/Usage:** Haiku is cheap; Sonnet is more expensive.
* **Suitability:** **Moderate.** The lack of a simple, built-in Pydantic wrapper for structured output in the official SDK makes it slightly more complex to integrate quickly for a strict JSON contract compared to Gemini and OpenAI.

---

## Decision

For this hackathon, we will proceed with the **Google Gemini API** (`gemini-2.5-flash`) using the `google-genai` SDK.

**Rationale:**
1. **Native Pydantic Support:** The `google-genai` SDK makes structured output trivial.
2. **Speed & Cost:** `gemini-2.5-flash` is perfectly suited for small, rapid translations.
3. **Simplicity:** We do not need agents, RAG, or search—just a fast, reliable JSON-to-JSON transformation.
