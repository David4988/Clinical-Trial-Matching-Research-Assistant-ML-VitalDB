import json
import hashlib
import os
from typing import Any, Dict, Optional

class LLMCache:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _compute_key(self, evidence: dict, model: str, schema_version: str, prompt_version: str) -> str:
        # Create a stable string representation of the evidence dict
        evidence_str = json.dumps(evidence, sort_keys=True)
        payload = f"{evidence_str}|{model}|{schema_version}|{prompt_version}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, evidence: dict, model: str, schema_version: str, prompt_version: str) -> Optional[dict]:
        key = self._compute_key(evidence, model, schema_version, prompt_version)
        return self.cache.get(key)

    def set(self, evidence: dict, model: str, schema_version: str, prompt_version: str, explanation: dict):
        key = self._compute_key(evidence, model, schema_version, prompt_version)
        self.cache[key] = explanation
        self._save_cache()
