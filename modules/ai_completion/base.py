"""Abstract base for generative 3D completion providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class CompletionStatus(str, Enum):
    PRODUCTION_CANDIDATE = "production_candidate"   # observed >= production threshold + synth <= prod limit
    REVIEW_READY = "review_ready"                   # observed >= completion threshold + synth <= review limit
    PREVIEW_ONLY = "preview_only"                   # observed too low or synth too high but not catastrophic
    FAILED = "failed"                               # provider error / unsalvageable
    SKIPPED_DISABLED = "skipped_disabled"           # ai_completion_enabled=false
    SKIPPED_SUFFICIENT = "skipped_sufficient"       # observed surface already meets production threshold
    SKIPPED_PROVIDER_NONE = "skipped_provider_none" # provider=none configured


@dataclass
class CompletionRequest:
    session_id: str
    mesh_path: str
    reference_image_paths: List[str] = field(default_factory=list)
    observed_surface_ratio: float = 0.0
    capture_profile_key: str = "small_on_surface"
    material_hint: str = "opaque"
    bbox_extents: Optional[Dict[str, float]] = None  # {"x":..., "y":..., "z":...}
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompletionResult:
    status: CompletionStatus
    provider_name: str
    completed_mesh_path: Optional[str] = None
    synthesized_surface_ratio: float = 0.0
    observed_surface_ratio: float = 0.0
    elapsed_sec: float = 0.0
    log: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class CompletionProvider(ABC):
    """
    Pluggable generative provider.  Concrete subclasses include:
      - NoneProvider                  (passthrough, no completion)
      - Hunyuan3DReplicateProvider    (HTTP API: replicate.com)
      - Hunyuan3DLocalProvider        (local Gradio / inference server)
      - MeshyProvider                 (meshy.ai REST)

    Every provider must be safe to instantiate even when its dependencies
    are missing — `is_available()` must reflect that and `complete()`
    must short-circuit with an explanatory FAILED result.
    """

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def complete(self, req: CompletionRequest, output_dir: Path) -> CompletionResult:
        ...

    def get_status(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "available": self.is_available(),
        }
