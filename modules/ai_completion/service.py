"""
End-to-end orchestrator for AI completion.

  AICompletionService.assess()  → CompletionDecision (no side effects, cheap)
  AICompletionService.run()     → assess + invoke provider + measure synth + apply gates
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    CompletionProvider,
    CompletionRequest,
    CompletionResult,
    CompletionStatus,
)
from .coverage import compute_observed_surface_ratio, compute_synthesized_ratio
from .policy import CompletionDecision, decide_completion_path, apply_quality_gates
from .providers import build_provider

logger = logging.getLogger("ai_completion")


class AICompletionService:
    def __init__(
        self,
        provider: Optional[CompletionProvider] = None,
        settings_obj: Any = None,
        capture_profile: Any = None,
    ):
        self.provider = provider
        self.settings = settings_obj
        self.capture_profile = capture_profile

    # ------------------------------------------------------------------ assess
    def assess(
        self,
        mesh,
        cameras: Optional[List[Dict]] = None,
        masks: Optional[Dict] = None,
        point_cloud=None,
    ) -> Dict[str, Any]:
        """
        Cheap analysis: compute observed-surface ratio + decision tree result.
        Does NOT call the provider.
        """
        cov = compute_observed_surface_ratio(
            mesh, cameras=cameras, masks=masks, point_cloud=point_cloud
        )
        decision = decide_completion_path(
            observed_ratio=cov["observed_surface_ratio"],
            settings_obj=self.settings,
            capture_profile=self.capture_profile,
            provider_name=getattr(self.provider, "name", "none"),
        )
        return {
            "coverage": cov,
            "decision": decision.to_dict(),
            "provider": getattr(self.provider, "name", "none"),
            "provider_available": bool(self.provider and self.provider.is_available()),
        }

    # ------------------------------------------------------------------ run
    def run(
        self,
        mesh,
        original_face_count: int,
        request: CompletionRequest,
        output_dir: Path,
        cameras: Optional[List[Dict]] = None,
        masks: Optional[Dict] = None,
        point_cloud=None,
    ) -> CompletionResult:
        """
        Full pipeline: assess → invoke provider (if allowed) → measure → grade.
        """
        cov = compute_observed_surface_ratio(
            mesh, cameras=cameras, masks=masks, point_cloud=point_cloud
        )
        observed = cov["observed_surface_ratio"]
        request.observed_surface_ratio = observed

        decision = decide_completion_path(
            observed_ratio=observed,
            settings_obj=self.settings,
            capture_profile=self.capture_profile,
            provider_name=getattr(self.provider, "name", "none"),
        )

        if not decision.should_run:
            return CompletionResult(
                status=decision.target_status,
                provider_name=getattr(self.provider, "name", "none"),
                completed_mesh_path=request.mesh_path,
                observed_surface_ratio=observed,
                synthesized_surface_ratio=0.0,
                log=[
                    f"Coverage method: {cov['method']}",
                    f"Decision: {decision.reason}",
                ],
                metadata={"coverage": cov, "decision": decision.to_dict()},
            )

        if not self.provider or not self.provider.is_available():
            provider_name = getattr(self.provider, "name", "none")
            return CompletionResult(
                status=CompletionStatus.FAILED,
                provider_name=provider_name,
                completed_mesh_path=request.mesh_path,
                observed_surface_ratio=observed,
                error=f"Provider '{provider_name}' not available (missing token / dep / checkpoint)",
                log=[f"Coverage method: {cov['method']}"],
                metadata={"coverage": cov, "decision": decision.to_dict()},
            )

        logger.info(
            f"AI completion run: observed={observed:.2f}, target={decision.target_status.value}, "
            f"provider={self.provider.name}"
        )
        provider_result = self.provider.complete(request, output_dir)

        # If provider failed, propagate as-is (with policy metadata)
        if provider_result.status in (CompletionStatus.FAILED,) or not provider_result.completed_mesh_path:
            provider_result.metadata.update({"coverage": cov, "decision": decision.to_dict()})
            return provider_result

        # Measure synthesized ratio
        completed_face_count = original_face_count
        try:
            import trimesh
            loaded = trimesh.load(provider_result.completed_mesh_path, force="mesh")
            if hasattr(loaded, "faces"):
                completed_face_count = int(len(loaded.faces))
        except Exception as e:
            logger.warning(f"Could not measure completed mesh face count: {e}")

        synth_ratio = compute_synthesized_ratio(original_face_count, completed_face_count)
        provider_result.synthesized_surface_ratio = synth_ratio

        # Apply quality gates
        final_status = apply_quality_gates(decision, synth_ratio)
        provider_result.status = final_status

        provider_result.metadata.update({
            "coverage": cov,
            "decision": decision.to_dict(),
            "original_face_count": original_face_count,
            "completed_face_count": completed_face_count,
        })
        provider_result.log.append(
            f"Final grade: {final_status.value} "
            f"(synth_ratio={synth_ratio:.2%}, prod_max={decision.max_synth_for_production:.2%}, "
            f"review_max={decision.max_synth_for_review:.2%})"
        )
        return provider_result


def build_default_service(
    settings_obj: Any,
    capture_profile: Any = None,
    provider_override: Optional[str] = None,
) -> AICompletionService:
    """
    Build a service from settings.  Provider chosen by AI_3D_PROVIDER env var
    unless `provider_override` is supplied.
    """
    provider_name = provider_override or getattr(settings_obj, "ai_3d_provider", "none")
    provider = build_provider(provider_name)
    return AICompletionService(
        provider=provider,
        settings_obj=settings_obj,
        capture_profile=capture_profile,
    )
