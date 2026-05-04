"""Sanitization utilities for external provider interactions."""
import re
from typing import Optional


def sanitize_text(text: Optional[str]) -> str:
    """
    Redact sensitive information from text (logs, errors, manifests).
    Redacts:
    - Authorization headers (Bearer tokens)
    - api_key fields
    - token fields
    - actual setting values if they appear
    """
    if not text:
        return ""

    # 1. Direct replacement of configured secrets (highest priority)
    from modules.operations.settings import settings
    secrets = [
        getattr(settings, "rodin_api_key", ""),
        getattr(settings, "meshy_api_key", ""),
        getattr(settings, "tripo_api_key", ""),
        getattr(settings, "pilot_api_key", ""),
    ]
    for s in secrets:
        if s and isinstance(s, str):
            text = text.replace(s, "[REDACTED]")

    # 2. Redact Authorization: Bearer ...
    text = re.sub(r"(Authorization:\s*Bearer\s+)[^\s]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    
    # 3. Redact Bearer ... (standalone)
    text = re.sub(r"(Bearer\s+)[^\s]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)

    # 4. Redact api_key=... or api_key: ...
    text = re.sub(r"(api_key\s*[=:]\s*)[^\s&,}\]]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    
    # 5. Redact token=... or token: ...
    text = re.sub(r"(token\s*[=:]\s*)[^\s&,}\]]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)

    # 6. Specific redaction for common key patterns (heuristic)
    text = re.sub(r"([a-zA-Z0-9_-]{20,})", lambda m: "[REDACTED]" if _is_likely_key(m.group(1)) else m.group(1), text)

    return text


def sanitize_external_provider_error(text: Optional[str]) -> str:
    """Backwards-compatible wrapper around sanitize_text."""
    return sanitize_text(text)


def sanitize_json_like(obj):
    """
    Recursively sanitize a dict, list, or string.

    - dict: sanitize all values in-place (returns new dict)
    - list: sanitize all elements (returns new list)
    - str: run through sanitize_text
    - anything else: returned unchanged

    Use for manifest fields that may contain user-supplied or provider-supplied
    data: warnings, errors, provider_failure_reason, sanitized_error,
    worker_metadata, candidates, candidate_ranking, path_diagnostics.

    Note: normal safe paths are NOT aggressively redacted unless they contain
    token-like patterns (handled by sanitize_text heuristics).
    """
    if isinstance(obj, dict):
        return {k: sanitize_json_like(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json_like(item) for item in obj]
    if isinstance(obj, str):
        return sanitize_text(obj)
    return obj


def _is_likely_key(text: str) -> bool:
    """Heuristic to check if a string looks like a secret key."""
    # If it's the exact value of a setting, redact it
    from modules.operations.settings import settings
    secrets = [
        getattr(settings, "rodin_api_key", ""),
        getattr(settings, "meshy_api_key", ""),
        getattr(settings, "tripo_api_key", ""),
        getattr(settings, "pilot_api_key", ""),
    ]
    for s in secrets:
        if s and isinstance(s, str) and s in text:
            return True
    return False
