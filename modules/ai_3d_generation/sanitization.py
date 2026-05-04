"""Sanitization utilities for external provider interactions."""
import re
from typing import Optional


def sanitize_external_provider_error(text: Optional[str]) -> str:
    """
    Redact sensitive information from error messages or logs.
    Redacts:
    - Authorization headers (Bearer tokens)
    - api_key fields
    - token fields
    - actual setting values if they appear
    """
    if not text:
        return ""

    # Redact Authorization: Bearer ...
    text = re.sub(r"(Authorization:\s*Bearer\s+)[^\s]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    
    # Redact Bearer ... (standalone)
    text = re.sub(r"(Bearer\s+)[^\s]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)

    # Redact api_key=... or api_key: ...
    text = re.sub(r"(api_key\s*[=:]\s*)[^\s&,}\]]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    
    # Redact token=... or token: ...
    text = re.sub(r"(token\s*[=:]\s*)[^\s&,}\]]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)

    # Specific redaction for common key patterns
    text = re.sub(r"([a-zA-Z0-9_-]{20,})", lambda m: "[REDACTED]" if _is_likely_key(m.group(1)) else m.group(1), text)

    return text


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
        if s and s in text:
            return True
    return False
