"""Full orchestration pipeline for LLM XAI."""

import json
import time
from typing import List, Dict, Any, Tuple

from vitaldb_audit.llm.config import get_llm_config
from vitaldb_audit.llm.gemini import GeminiProvider
from vitaldb_audit.llm.cache import LLMCache
from vitaldb_audit.llm.validation import validate_explanation, LLMValidationError
from vitaldb_audit.explain import explain_window

SCHEMA_VERSION = "1.0"
PROMPT_VERSION = "1.1"
MAX_LLM_CALLS = 14

def process_evidence_batch(evidence_cases: List[Dict[str, Any]], cache_file: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process a batch of evidence objects through the XAI pipeline."""
    config = get_llm_config()
    xai_mode = config["xai_mode"]
    
    # Initialize provider only if required
    provider_client = None
    if xai_mode in ("llm", "hybrid"):
        if config["provider"] == "gemini":
            provider_client = GeminiProvider(config["api_key"], config["model"])
        else:
            raise ValueError(f"Unsupported provider for pipeline: {config['provider']}")
        
    cache = LLMCache(cache_file)
    
    results = []
    metrics = {
        "total": len(evidence_cases),
        "success": 0,
        "failed": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "total_api_calls": 0,
        "schema_validation_failures": 0,
        "latencies": []
    }
    
    for ev in evidence_cases:
        case_id = ev.get("case_id")
        window_index = ev.get("window_index")
        anomaly_rank = ev.get("anomaly_rank")
        anomaly_score = ev.get("anomaly_score")
        
        # 1. Always generate authoritative deterministic explanation
        deterministic_exp = explain_window(ev)
        
        # Base result payload
        result = {
            "case_id": case_id,
            "window_index": window_index,
            "anomaly_rank": anomaly_rank,
            "anomaly_score": anomaly_score,
            "deterministic_explanation": deterministic_exp,
            "llm_explanation": None,
            "llm_status": "skipped",
            "provider": config["provider"] if provider_client else None,
            "model": config["model"] if provider_client else None,
            "latency_ms": 0,
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
        }
        
        # 2. LLM explanation if required
        if xai_mode in ("llm", "hybrid"):
            cached_llm = cache.get(ev, config["model"], SCHEMA_VERSION, PROMPT_VERSION)
            if cached_llm:
                result["llm_explanation"] = cached_llm
                result["llm_status"] = "success"
                metrics["cache_hits"] += 1
                metrics["success"] += 1
            else:
                metrics["cache_misses"] += 1
                
                # Enforce call budget
                if metrics["total_api_calls"] >= MAX_LLM_CALLS:
                    result["llm_status"] = "failed"
                    result["error"] = "API call budget exceeded"
                    metrics["failed"] += 1
                else:
                    start_t = time.time()
                    try:
                        # Up to 1 retry
                        try:
                            metrics["total_api_calls"] += 1
                            llm_resp = provider_client.explain(ev)
                        except Exception:
                            metrics["total_api_calls"] += 1
                            llm_resp = provider_client.explain(ev)
                            
                        # Validate response safety and truthfulness
                        validate_explanation(llm_resp, ev)
                        
                        latency = (time.time() - start_t) * 1000
                        result["latency_ms"] = int(latency)
                        metrics["latencies"].append(result["latency_ms"])
                        
                        llm_dict = llm_resp.model_dump()
                        result["llm_explanation"] = llm_dict
                        result["llm_status"] = "success"
                        metrics["success"] += 1
                        
                        # Save to cache
                        cache.set(ev, config["model"], SCHEMA_VERSION, PROMPT_VERSION, llm_dict)
                        
                    except LLMValidationError as e:
                        result["llm_status"] = "failed"
                        result["error"] = f"Validation Error: {str(e)}"
                        metrics["failed"] += 1
                        metrics["schema_validation_failures"] += 1
                    except Exception as e:
                        result["llm_status"] = "failed"
                        result["error"] = f"API Error: {str(e)}"
                        metrics["failed"] += 1
        
        # 3. Frontend-ready Object Formatter
        frontend_obj = {
            "case_id": case_id,
            "window_index": window_index,
            "severity": "unusual",
            "headline": deterministic_exp["headline"],  # Headline is always deterministic
            "anomaly_score": anomaly_score,
            "llm_status": result["llm_status"]
        }
        
        if result["llm_status"] == "success" and result["llm_explanation"]:
            frontend_obj["summary"] = result["llm_explanation"]["summary"]
            frontend_obj["key_evidence"] = result["llm_explanation"]["key_evidence"]
            frontend_obj["data_quality"] = result["llm_explanation"]["data_quality"]
            frontend_obj["uncertainty"] = result["llm_explanation"]["uncertainty"]
        else:
            # Deterministic Fallback
            frontend_obj["summary"] = " ".join(deterministic_exp["signal_narratives"].values())
            frontend_obj["key_evidence"] = deterministic_exp["supporting_detail"]
            frontend_obj["data_quality"] = " ".join(deterministic_exp["data_quality_notes"])
            frontend_obj["uncertainty"] = deterministic_exp["interpretation"]

        result["frontend_ready"] = frontend_obj
        results.append(result)

    # Compute Latency Metrics
    if metrics["latencies"]:
        metrics["average_latency"] = sum(metrics["latencies"]) / len(metrics["latencies"])
        sorted_lat = sorted(metrics["latencies"])
        metrics["median_latency"] = sorted_lat[len(sorted_lat) // 2]
    else:
        metrics["average_latency"] = 0
        metrics["median_latency"] = 0
        
    return results, metrics
