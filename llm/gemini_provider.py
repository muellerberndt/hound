"""Gemini provider implementation.

Supports both the legacy `google-generativeai` SDK and the new
`google-genai` SDK. Prefers the new SDK when available.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, TypeVar

from pydantic import BaseModel

from .base_provider import BaseLLMProvider
from .schema_definitions import get_schema_definition

T = TypeVar('T', bound=BaseModel)

# Try new google-genai client first; fallback to legacy google-generativeai
_USE_NEW_GENAI = False
_genai_new = None
_genai_legacy = None
try:  # New client
    from google import genai as _genai_new  # type: ignore
    _USE_NEW_GENAI = True
except Exception:
    try:
        import google.generativeai as _genai_legacy  # type: ignore
    except Exception:
        _genai_legacy = None

# Runtime flags to control legacy fallback (easier debugging when disabled)
_REQUIRE_NEW = (os.environ.get("HOUND_GEMINI_REQUIRE_NEW", "").lower() in {"1", "true", "yes", "on"})
_NO_LEGACY = _REQUIRE_NEW or (os.environ.get("HOUND_GEMINI_NO_LEGACY", "").lower() in {"1", "true", "yes", "on"})

# Provide minimal enums/types shims for legacy path when available
_HarmCategory = None
_HarmBlockThreshold = None
if _genai_legacy is not None:
    try:
        from google.generativeai.types import (  # type: ignore
            HarmBlockThreshold as _HarmBlockThreshold,
            HarmCategory as _HarmCategory,
        )
    except Exception:
        _HarmBlockThreshold = None
        _HarmCategory = None

# Backward-compat alias used by unit tests (patched in tests)
genai = _genai_new or _genai_legacy


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider implementation (new SDK preferred)."""
    
    def __init__(
        self, 
        config: dict[str, Any], 
        model_name: str,
        timeout: int = 120,
        retries: int = 3,
        backoff_min: float = 2.0,
        backoff_max: float = 8.0,
        thinking_enabled: bool = False,
        thinking_budget: int = -1,  # -1 for dynamic, 0 to disable, >0 for fixed budget
        **kwargs
    ):
        """
        Initialize Gemini provider.
        
        Args:
            config: Configuration dictionary
            model_name: Gemini model name (e.g., "gemini-2.0-flash", "gemini-2.5-flash")
            timeout: Request timeout in seconds
            retries: Number of retry attempts
            backoff_min: Minimum backoff time in seconds
            backoff_max: Maximum backoff time in seconds
            thinking_enabled: Whether to enable thinking mode (for 2.5 models)
            thinking_budget: Thinking token budget (-1 for dynamic, 0 to disable)
        """
        self.config = config
        self.model_name = model_name
        self.timeout = timeout
        self.retries = retries
        self.backoff_min = backoff_min
        self.backoff_max = backoff_max
        self.thinking_enabled = thinking_enabled
        self.thinking_budget = thinking_budget
        self._last_token_usage = None
        self._use_new = _USE_NEW_GENAI and (_genai_new is not None)
        self._client = None  # Only for new SDK
        self._client_flavor = "genai-new" if self._use_new else ("genai-legacy" if _genai_legacy is not None else "none")

        # Vertex AI configuration (optional)
        gcfg = config.get("gemini", {}) if isinstance(config, dict) else {}
        vcfg = gcfg.get("vertex_ai", {}) if isinstance(gcfg, dict) else {}
        env_use_vertex = os.environ.get("GOOGLE_USE_VERTEX_AI", "").lower() in {"1","true","yes","on"}
        self._use_vertex = bool(vcfg.get("enabled", False) or env_use_vertex)

        # Resolve project and region when Vertex AI is enabled
        self._vertex_project = None
        self._vertex_region = None
        self._vertex_endpoint_url = None
        if self._use_vertex:
            proj = vcfg.get("project_id") or os.environ.get("VERTEX_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
            loc = vcfg.get("region") or vcfg.get("location") or os.environ.get("VERTEX_LOCATION") or os.environ.get("GOOGLE_CLOUD_REGION") or os.environ.get("GOOGLE_CLOUD_ZONE")
            # Convert zone to region if needed (e.g., "us-central1-a" -> "us-central1")
            if isinstance(loc, str) and loc.count("-") >= 2:
                try:
                    parts = loc.split("-")
                    loc = "-".join(parts[:2])
                except Exception:
                    pass
            self._vertex_project = proj
            self._vertex_region = loc
            if proj and loc:
                base = f"https://{loc}-aiplatform.googleapis.com"
                self._vertex_endpoint_url = f"{base}/v1/projects/{proj}/locations/{loc}"
            # No API key required for Vertex; uses ADC / service account
            api_key_env = gcfg.get("api_key_env", "GOOGLE_API_KEY")
            api_key = os.environ.get(api_key_env)  # May be None; not required in Vertex mode
        else:
            # API key path (AI Studio)
            api_key_env = gcfg.get("api_key_env", "GOOGLE_API_KEY")
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"API key not found in environment variable: {api_key_env}")
        
        # Determine generation defaults; explicitly set max_output_tokens high
        # If not provided, many models default to their maximum output limit.
        # We set it explicitly to be safe and configurable.
        self._default_generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": self._resolve_max_output_tokens(model_name),
            "response_mime_type": "application/json",
        }

        if self._use_new:
            # New SDK client
            try:
                if self._use_vertex:
                    # Vertex AI routing (uses ADC or service account credentials)
                    if self._vertex_project and self._vertex_region:
                        self._client = _genai_new.Client(vertexai=True, project=self._vertex_project, location=self._vertex_region)
                    else:
                        # Let the SDK pick up project/location from env/ADC if not provided
                        self._client = _genai_new.Client(vertexai=True)
                else:
                    self._client = _genai_new.Client(api_key=api_key)
            except TypeError:
                # Fallbacks for older client signatures
                if self._use_vertex:
                    self._client = _genai_new.Client(vertexai=True)
                else:
                    self._client = _genai_new.Client()
            self.model = model_name  # Keep a simple name for new API
        else:
            # Legacy SDK configuration and model instance (or patched test double)
            if _NO_LEGACY:
                raise RuntimeError("HOUND_GEMINI_REQUIRE_NEW/HOUND_GEMINI_NO_LEGACY is set but google-genai is not available. Install 'google-genai' to proceed.")
            legacy_lib = _genai_legacy or genai
            if legacy_lib is None:
                raise RuntimeError(
                    "Neither google-genai nor google-generativeai is available. Install one to use Gemini."
                )
            # Configure if available
            try:
                legacy_lib.configure(api_key=api_key)
            except Exception:
                pass

            # Build permissive safety settings when legacy enums exist
            safety_settings = None
            if _HarmCategory is not None and _HarmBlockThreshold is not None:
                try:
                    safety_settings = {
                        _HarmCategory.HARM_CATEGORY_HATE_SPEECH: _HarmBlockThreshold.BLOCK_NONE,
                        _HarmCategory.HARM_CATEGORY_HARASSMENT: _HarmBlockThreshold.BLOCK_NONE,
                        _HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: _HarmBlockThreshold.BLOCK_NONE,
                        _HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: _HarmBlockThreshold.BLOCK_NONE,
                    }
                    if hasattr(_HarmCategory, 'HARM_CATEGORY_CIVIC_INTEGRITY'):
                        safety_settings[_HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY] = _HarmBlockThreshold.BLOCK_NONE
                except Exception:
                    safety_settings = None

            # Create model instance
            self.model = legacy_lib.GenerativeModel(
                model_name=model_name,
                generation_config=self._filter_generation_config_for_legacy(self._default_generation_config),
                safety_settings=safety_settings,
            )

    def _resolve_max_output_tokens(self, model_name: str) -> int:
        """Choose a high but safe max_output_tokens value.

        If we can fetch model info via the new SDK, we use its output_token_limit.
        Otherwise, fall back to a conservative per-family default.
        """
        # Allow environment override for quick tuning
        try:
            env_val = os.environ.get("HOUND_GEMINI_MAX_OUTPUT_TOKENS")
            if env_val:
                v = int(env_val)
                if v > 0:
                    return v
        except Exception:
            pass

        # Try new SDK to read the model's advertised limit
        if _USE_NEW_GENAI and _genai_new is not None:
            try:
                client = _genai_new.Client()
                # Some client versions use `get_model`, others `get`.
                get_fn = getattr(client.models, 'get_model', None) or getattr(client.models, 'get', None)
                if get_fn:
                    model_id = model_name if model_name.startswith("models/") else model_name
                    info = get_fn(model=model_id) if 'get' in get_fn.__name__ else get_fn(model_id)
                    # Try common attribute names
                    for attr in ("output_token_limit", "outputTokenLimit", "max_output_tokens"):
                        if hasattr(info, attr):
                            val = getattr(info, attr)
                            if isinstance(val, int) and val > 0:
                                return val
            except Exception:
                pass

        # Fallback heuristics
        name = model_name.lower()
        if "2.5" in name:
            return 8192
        if "2.0" in name or "1.5" in name:
            return 8192
        return 4096

    @staticmethod
    def _filter_generation_config_for_legacy(cfg: dict[str, Any]) -> dict[str, Any]:
        """Map generation_config keys to legacy naming where needed."""
        mapped = dict(cfg)
        # Legacy SDK uses snake_case for max_output_tokens as well; keep as-is.
        return mapped

    @staticmethod
    def _normalize_generation_config_for_new(cfg: dict[str, Any]) -> dict[str, Any]:
        """Convert generation_config keys to the camelCase expected by google-genai."""
        mapping = {
            "max_output_tokens": "maxOutputTokens",
            "top_p": "topP",
            "top_k": "topK",
            "response_mime_type": "responseMimeType",
        }
        out = {}
        for k, v in cfg.items():
            out[mapping.get(k, k)] = v
        return out

    @staticmethod
    def _extract_json_from_text(text: str) -> Any:
        """Best-effort extraction of a JSON object/array from text."""
        import re
        # Try direct parse first
        try:
            return json.loads(text)
        except Exception:
            pass
        # Look for fenced code blocks
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # Look for first top-level object or array heuristically
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = text.find(open_ch)
            if start != -1:
                depth = 0
                for i in range(start, len(text)):
                    c = text[i]
                    if c == open_ch:
                        depth += 1
                    elif c == close_ch:
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:i+1]
                            try:
                                return json.loads(candidate)
                            except Exception:
                                break
        raise ValueError("No valid JSON object found in response text")
    
    def parse(self, *, system: str, user: str, schema: type[T]) -> T:
        """Make a structured call returning parsed JSON matching `schema`."""
        # Get schema definition from centralized source
        schema_info = get_schema_definition(schema)
        
        # Combine system and user prompts (Gemini doesn't have separate system messages)
        prompt = f"{system}{schema_info}\n\n{user}"
        
        # Calculate request size for potential debugging
        len(prompt)
        
        last_err = None
        
        for attempt in range(self.retries):
            try:
                attempt_start = time.time()
                
                # Build request
                generation_config = dict(self._default_generation_config)
                # Provide JSON schema to strongly enforce JSON output (new SDK)
                try:
                    json_schema = schema.model_json_schema()
                except Exception:
                    json_schema = None
                if self._use_new and json_schema:
                    # responseJsonSchema is preferred; responseSchema must be omitted when used
                    generation_config["responseJsonSchema"] = json_schema
                if self._use_new and self.thinking_enabled and "2.5" in self.model_name:
                    # New SDK expects thinking config nested under generation_config
                    thinking_obj = {
                        "includeThoughts": False,
                    }
                    if self.thinking_budget and self.thinking_budget > 0:
                        thinking_obj["thinkingBudget"] = self.thinking_budget
                    # Use camelCase key as per API
                    generation_config["thinkingConfig"] = thinking_obj

                if self._use_new:
                    # New SDK call
                    kwargs = {
                        "model": self.model_name,
                        "contents": prompt,
                        "generation_config": self._normalize_generation_config_for_new(generation_config),
                    }
                    try:
                        response = self._client.models.generate_content(**kwargs)
                    except TypeError:
                        # Retry without thinkingConfig if unsupported
                        try:
                            if isinstance(kwargs.get("generation_config"), dict) and "thinkingConfig" in kwargs["generation_config"]:
                                gc = dict(kwargs["generation_config"])
                                gc.pop("thinkingConfig", None)
                                kwargs["generation_config"] = gc
                            response = self._client.models.generate_content(**kwargs)
                        except TypeError:
                            # Fallback without generation_config entirely
                            kwargs.pop("generation_config", None)
                            response = self._client.models.generate_content(**kwargs)
                else:
                    # Legacy SDK call via model instance
                    response = self.model.generate_content(
                        prompt,
                        generation_config=self._filter_generation_config_for_legacy(generation_config),
                        request_options={"timeout": self.timeout}
                    )
                
                # Log response details
                time.time() - attempt_start
                
                # Track token usage if available
                if hasattr(response, 'usage_metadata'):
                    self._last_token_usage = {
                        'input_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
                        'output_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0),
                        'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0)
                    }
                
                # Check if response was blocked
                if getattr(response, 'candidates', None):
                    candidate = response.candidates[0]
                    finish = getattr(candidate, 'finish_reason', None)
                    # Normalize finish reason to string if possible
                    if isinstance(finish, int):
                        finish_map = {0: "UNSPECIFIED", 1: "STOP", 2: "MAX_TOKENS", 3: "SAFETY", 4: "RECITATION", 5: "OTHER"}
                        finish_str = finish_map.get(finish, str(finish))
                    else:
                        finish_str = str(finish).upper() if finish else None

                    # Only hard-fail on explicit safety or prohibited content; otherwise accept
                    if finish_str in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"}:
                        raise RuntimeError(f"Response blocked: {finish_str}")
                
                # Parse the JSON response into the schema
                text = getattr(response, 'text', None)
                if not text and getattr(response, 'candidates', None):
                    # Reconstruct text from candidate parts if needed
                    try:
                        content = response.candidates[0].content
                        parts = getattr(content, 'parts', None) or []
                        text_parts = []
                        for p in parts:
                            if hasattr(p, 'text') and p.text:
                                text_parts.append(p.text)
                        text = "\n".join(text_parts) if text_parts else None
                    except Exception:
                        pass
                if text:
                    json_data = self._extract_json_from_text(text)
                    # Try strict validation first, then prune extras and retry
                    try:
                        return schema.model_validate(json_data)
                    except Exception:
                        if isinstance(json_data, dict):
                            allowed = set(getattr(schema, 'model_fields', {}).keys())
                            pruned = {k: v for k, v in json_data.items() if k in allowed}
                            return schema.model_validate(pruned)
                        raise
                else:
                    raise RuntimeError("Empty response from Gemini")
                    
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    sleep_time = random.uniform(self.backoff_min, self.backoff_max)
                    time.sleep(sleep_time)
        
        raise RuntimeError(f"Gemini call failed after {self.retries} attempts: {last_err}")
    
    def raw(self, *, system: str, user: str) -> str:
        """Make a plain text call."""
        # Combine system and user prompts
        prompt = f"{system}\n\n{user}"
        
        last_err = None
        for attempt in range(self.retries):
            try:
                # Generate without structured output
                generation_config = dict(self._default_generation_config)
                generation_config.pop("response_mime_type", None)
                if self._use_new and self.thinking_enabled and "2.5" in self.model_name:
                    thinking_obj = {"includeThoughts": False}
                    if self.thinking_budget and self.thinking_budget > 0:
                        thinking_obj["thinkingBudget"] = self.thinking_budget
                    generation_config["thinkingConfig"] = thinking_obj

                if self._use_new:
                    kwargs = {
                        "model": self.model_name,
                        "contents": prompt,
                        "generation_config": self._normalize_generation_config_for_new(generation_config),
                    }
                    try:
                        response = self._client.models.generate_content(**kwargs)
                    except TypeError:
                        if isinstance(kwargs.get("generation_config"), dict) and "thinkingConfig" in kwargs["generation_config"]:
                            gc = dict(kwargs["generation_config"])
                            gc.pop("thinkingConfig", None)
                            kwargs["generation_config"] = gc
                            try:
                                response = self._client.models.generate_content(**kwargs)
                            except TypeError:
                                kwargs.pop("generation_config", None)
                                response = self._client.models.generate_content(**kwargs)
                        else:
                            kwargs.pop("generation_config", None)
                            response = self._client.models.generate_content(**kwargs)
                else:
                    response = self.model.generate_content(
                        prompt,
                        generation_config=self._filter_generation_config_for_legacy(generation_config),
                        request_options={"timeout": self.timeout}
                    )
                
                # Track token usage if available
                if hasattr(response, 'usage_metadata'):
                    self._last_token_usage = {
                        'input_tokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
                        'output_tokens': getattr(response.usage_metadata, 'candidates_token_count', 0),
                        'total_tokens': getattr(response.usage_metadata, 'total_token_count', 0)
                    }
                
                return response.text
                
            except Exception as e:
                last_err = e
                if attempt < self.retries - 1:
                    sleep_time = random.uniform(self.backoff_min, self.backoff_max)
                    time.sleep(sleep_time)
        
        raise RuntimeError(f"Gemini raw call failed after {self.retries} attempts: {last_err}")
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "Gemini"
    
    @property
    def supports_thinking(self) -> bool:
        """Gemini 2.5 models support thinking mode."""
        return "2.5" in self.model_name
    
    def get_last_token_usage(self) -> dict[str, int] | None:
        """Return token usage from the last call if available."""
        return self._last_token_usage
