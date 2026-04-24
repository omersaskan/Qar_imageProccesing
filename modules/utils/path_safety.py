import re
from pathlib import Path
from typing import Optional
from modules.shared_contracts.errors import PathSafetyError

def validate_safe_path(base_dir: str | Path, target_path: str | Path) -> Path:
    """
    Validates that target_path is within base_dir and prevents traversal.
    Uses Path.resolve() for canonicalization and is_relative_to() for containment check.
    Returns the absolute Path if safe.
    """
    base_path = Path(base_dir).resolve()
    target_path_obj = Path(target_path)
    
    # If target_path is absolute, check if it's within base_path
    if target_path_obj.is_absolute():
        resolved_target = target_path_obj.resolve()
    else:
        resolved_target = (base_path / target_path_obj).resolve()
    
    if not resolved_target.is_relative_to(base_path):
        raise PathSafetyError(
            f"Path safety violation: '{target_path}' resolves to '{resolved_target}', "
            f"which is outside of root '{base_path}'"
        )
    
    return resolved_target

def validate_identifier(identifier: str, label: str = "Identifier", max_length: int = 64) -> str:
    """
    Validates that an identifier (job_id, asset_id, etc.) only contains safe characters.
    Whitelist: alphanumeric, underscores, hyphens.
    """
    if not identifier:
        raise ValueError(f"{label} cannot be empty.")
        
    if len(identifier) > max_length:
        raise ValueError(f"{label} is too long (max {max_length} chars).")
        
    if not re.match(r"^[a-zA-Z0-9_\-]+$", identifier):
        raise ValueError(f"{label} contains invalid characters: '{identifier}'. Only alphanumeric, underscores, and hyphens are allowed.")
        
    return identifier

def ensure_dir(path: str | Path):
    """Safely ensures a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)
