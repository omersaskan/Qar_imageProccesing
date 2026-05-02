"""
Fallback Ladder — deterministic preset retry sequence on reconstruction failure.

When the chosen preset fails (BA collapse, OpenMVS native crash, OOM, etc.)
the runner consults this ladder for the next preset to try.  Each step
records to manifest so the recovery path is visible.

Default ladder (from primary toward conservative):

    profile_safe → safe_high_quality → safe_low_resolution →
    low_thread_texture → baseline

Crash-class detection (heuristic):
    - "exit code 3221226505"        → OpenMVS TextureMesh native crash → low_thread_texture
    - "out of memory" / "OOM"        → safe_low_resolution
    - "no such file or directory"    → cannot recover; abort ladder
    - generic RuntimeError           → next step in default ladder
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

from .reconstruction_preset_resolver import (
    PRESET_NAME_BASELINE,
    PRESET_NAME_LOW_LIGHT_SAFE,
    PRESET_NAME_LOW_TEXTURE_SAFE,
    PRESET_NAME_PROFILE_SAFE,
    PRESET_NAME_TEXTURE_RETRY_SAFE,
    get_preset_by_name,
)
from .reconstruction_profile import ReconstructionProfile


# Synthetic preset names used only by the ladder (resolver maps to baseline)
LADDER_SAFE_HIGH_QUALITY = "safe_high_quality"
LADDER_SAFE_LOW_RESOLUTION = "safe_low_resolution"
LADDER_LOW_THREAD_TEXTURE = "low_thread_texture"


@dataclass
class FallbackAttempt:
    step_index: int
    preset_name: str
    triggered_by: str        # initial | retry_after_runtime_error | retry_after_oom | retry_after_native_crash
    error_excerpt: str = ""
    preset_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_safe_high_quality(profile: Optional[ReconstructionProfile]) -> Dict[str, Any]:
    """Same as profile_safe but with hard caps on resource use."""
    p = get_preset_by_name(PRESET_NAME_PROFILE_SAFE, profile)
    p["name"] = LADDER_SAFE_HIGH_QUALITY
    p["openmvs"]["max_threads"] = 8
    p["rationale"] = "fallback step — clamped to 8 threads, otherwise profile_safe"
    return p


def _build_safe_low_resolution(profile: Optional[ReconstructionProfile]) -> Dict[str, Any]:
    """OOM recovery — cut image size and atlas to half."""
    p = get_preset_by_name(PRESET_NAME_PROFILE_SAFE, profile)
    p["name"] = LADDER_SAFE_LOW_RESOLUTION
    p["colmap"]["max_image_size"] = max(1200, int(p["colmap"]["max_image_size"] / 2))
    p["colmap"]["patchmatch_resolution_level"] = max(2, p["colmap"]["patchmatch_resolution_level"])
    p["openmvs"]["texture_resolution"] = max(1024, int(p["openmvs"]["texture_resolution"] / 2))
    p["openmvs"]["max_threads"] = 4
    p["rationale"] = (
        "fallback step — OOM recovery: half-size image / patchmatch L2 / "
        "half atlas / 4 threads"
    )
    return p


def _build_low_thread_texture(profile: Optional[ReconstructionProfile]) -> Dict[str, Any]:
    """OpenMVS TextureMesh native-crash recovery."""
    p = get_preset_by_name(PRESET_NAME_TEXTURE_RETRY_SAFE, profile)
    p["name"] = LADDER_LOW_THREAD_TEXTURE
    return p


def get_default_ladder(profile: Optional[ReconstructionProfile]) -> List[Dict[str, Any]]:
    """
    The full deterministic ladder list for a given profile.  Caller iterates
    on consecutive failures, picking `[0]`, `[1]`, ... until one succeeds or
    the list is exhausted (then capture_quality_rejected).
    """
    return [
        get_preset_by_name(PRESET_NAME_PROFILE_SAFE, profile),
        _build_safe_high_quality(profile),
        _build_safe_low_resolution(profile),
        _build_low_thread_texture(profile),
        get_preset_by_name(PRESET_NAME_BASELINE, profile),
    ]


def classify_error(error: Optional[str]) -> str:
    """Detect crash class from a stderr/log excerpt."""
    if not error:
        return "unknown"
    e = error.lower()
    if "3221226505" in e or "native crash" in e or "texturemesh" in e and "exit" in e:
        return "native_crash"
    if "out of memory" in e or "oom" in e or "cuda error: out of memory" in e:
        return "oom"
    if "no such file" in e or "file not found" in e:
        return "missing_file"
    if "runtimeerror" in e or "runtime error" in e:
        return "runtime"
    return "unknown"


def pick_next_preset(
    profile: Optional[ReconstructionProfile],
    error_excerpt: Optional[str],
    attempts_so_far: List[FallbackAttempt],
) -> Optional[FallbackAttempt]:
    """
    Pick the next preset to attempt based on crash class + history.
    Returns None when the ladder is exhausted (caller emits
    capture_quality_rejected upstream).

    Priority: crash-class-specific jump first; otherwise default ladder index.
    """
    crash_class = classify_error(error_excerpt)
    used_names = {a.preset_name for a in attempts_so_far}

    # Crash-class specific routing
    if crash_class == "native_crash" and LADDER_LOW_THREAD_TEXTURE not in used_names:
        preset = _build_low_thread_texture(profile)
        preset["name"] = LADDER_LOW_THREAD_TEXTURE
        return FallbackAttempt(
            step_index=len(attempts_so_far),
            preset_name=LADDER_LOW_THREAD_TEXTURE,
            triggered_by="retry_after_native_crash",
            error_excerpt=(error_excerpt or "")[:240],
            preset_snapshot=preset,
        )

    if crash_class == "oom" and LADDER_SAFE_LOW_RESOLUTION not in used_names:
        preset = _build_safe_low_resolution(profile)
        return FallbackAttempt(
            step_index=len(attempts_so_far),
            preset_name=LADDER_SAFE_LOW_RESOLUTION,
            triggered_by="retry_after_oom",
            error_excerpt=(error_excerpt or "")[:240],
            preset_snapshot=preset,
        )

    if crash_class == "missing_file":
        return None  # not recoverable by preset change

    # Default ladder traversal — pick first preset not yet attempted
    ladder = get_default_ladder(profile)
    for idx, p in enumerate(ladder):
        if p["name"] not in used_names:
            return FallbackAttempt(
                step_index=len(attempts_so_far),
                preset_name=p["name"],
                triggered_by="initial" if not attempts_so_far else "retry_after_runtime_error",
                error_excerpt=(error_excerpt or "")[:240],
                preset_snapshot=p,
            )
    return None  # exhausted
