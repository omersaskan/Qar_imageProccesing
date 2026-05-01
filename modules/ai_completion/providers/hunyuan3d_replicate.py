"""
Hunyuan3D-2 via Replicate.

Activates only when REPLICATE_API_TOKEN is present in env. Falls back to
FAILED status when missing — no spurious imports, no global state.

Reference: https://replicate.com/tencent/hunyuan3d-2 (model id changes;
HUNYUAN3D_REPLICATE_MODEL env var lets ops pin a specific version).
"""
from __future__ import annotations

import os
import time
import shutil
from pathlib import Path
from typing import Optional

from ..base import (
    CompletionProvider,
    CompletionRequest,
    CompletionResult,
    CompletionStatus,
)


class Hunyuan3DReplicateProvider(CompletionProvider):
    name = "hunyuan3d_replicate"

    def __init__(
        self,
        api_token: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout_sec: int = 600,
    ):
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN", "")
        self.model_id = model_id or os.getenv(
            "HUNYUAN3D_REPLICATE_MODEL",
            "tencent/hunyuan3d-2",
        )
        self.timeout_sec = timeout_sec

    def is_available(self) -> bool:
        if not self.api_token:
            return False
        try:
            import replicate  # noqa: F401
            return True
        except ImportError:
            return False

    def complete(self, req: CompletionRequest, output_dir: Path) -> CompletionResult:
        t0 = time.time()
        log = []

        if not self.api_token:
            return CompletionResult(
                status=CompletionStatus.FAILED,
                provider_name=self.name,
                error="REPLICATE_API_TOKEN missing",
                observed_surface_ratio=req.observed_surface_ratio,
                log=["No REPLICATE_API_TOKEN in env; cannot call Replicate API."],
            )

        try:
            import replicate  # type: ignore
        except ImportError:
            return CompletionResult(
                status=CompletionStatus.FAILED,
                provider_name=self.name,
                error="replicate python package not installed (pip install replicate)",
                observed_surface_ratio=req.observed_surface_ratio,
            )

        if not req.reference_image_paths:
            return CompletionResult(
                status=CompletionStatus.FAILED,
                provider_name=self.name,
                error="No reference image provided",
                observed_surface_ratio=req.observed_surface_ratio,
                log=["Hunyuan3D-2 requires at least one reference RGB image."],
            )

        ref_image = Path(req.reference_image_paths[0])
        if not ref_image.exists():
            return CompletionResult(
                status=CompletionStatus.FAILED,
                provider_name=self.name,
                error=f"Reference image not found: {ref_image}",
                observed_surface_ratio=req.observed_surface_ratio,
            )

        try:
            os.environ["REPLICATE_API_TOKEN"] = self.api_token
            log.append(f"Calling Replicate {self.model_id} with {ref_image.name}")

            with open(ref_image, "rb") as f:
                output = replicate.run(
                    self.model_id,
                    input={
                        "image": f,
                        "octree_resolution": 256,
                        "num_inference_steps": 30,
                    },
                )

            # `output` is typically a URL or list of URLs to the generated GLB
            output_url = output[0] if isinstance(output, (list, tuple)) else output
            if not isinstance(output_url, str) or not output_url.startswith("http"):
                return CompletionResult(
                    status=CompletionStatus.FAILED,
                    provider_name=self.name,
                    error=f"Unexpected Replicate response: {type(output).__name__}",
                    observed_surface_ratio=req.observed_surface_ratio,
                    log=log,
                )

            output_dir.mkdir(parents=True, exist_ok=True)
            completed_path = output_dir / "ai_completed.glb"

            import urllib.request
            urllib.request.urlretrieve(output_url, completed_path)
            log.append(f"Downloaded {output_url} → {completed_path}")

            elapsed = time.time() - t0
            return CompletionResult(
                status=CompletionStatus.PREVIEW_ONLY,  # service.py refines via policy gates
                provider_name=self.name,
                completed_mesh_path=str(completed_path),
                observed_surface_ratio=req.observed_surface_ratio,
                synthesized_surface_ratio=0.0,  # service.py overrides after measuring
                elapsed_sec=round(elapsed, 2),
                log=log,
                metadata={"replicate_model": self.model_id, "output_url": output_url},
            )

        except Exception as e:
            return CompletionResult(
                status=CompletionStatus.FAILED,
                provider_name=self.name,
                error=str(e),
                observed_surface_ratio=req.observed_surface_ratio,
                elapsed_sec=round(time.time() - t0, 2),
                log=log,
            )
