"""Passthrough provider — never modifies the mesh."""
from __future__ import annotations

from pathlib import Path
import time

from ..base import (
    CompletionProvider,
    CompletionRequest,
    CompletionResult,
    CompletionStatus,
)


class NoneProvider(CompletionProvider):
    name = "none"

    def is_available(self) -> bool:
        return True

    def complete(self, req: CompletionRequest, output_dir: Path) -> CompletionResult:
        return CompletionResult(
            status=CompletionStatus.SKIPPED_PROVIDER_NONE,
            provider_name=self.name,
            completed_mesh_path=req.mesh_path,
            synthesized_surface_ratio=0.0,
            observed_surface_ratio=req.observed_surface_ratio,
            elapsed_sec=0.0,
            log=["NoneProvider: passthrough; configure AI_3D_PROVIDER to enable real completion."],
        )
