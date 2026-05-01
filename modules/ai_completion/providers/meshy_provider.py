"""
Meshy AI provider — image-to-3D via REST.

Settings.meshy_enabled + MESHY_API_KEY required. Polls until completion
or timeout, then downloads the GLB.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from ..base import (
    CompletionProvider,
    CompletionRequest,
    CompletionResult,
    CompletionStatus,
)


class MeshyProvider(CompletionProvider):
    name = "meshy"
    BASE_URL = "https://api.meshy.ai/openapi/v1/image-to-3d"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_sec: int = 600,
        poll_interval_sec: int = 5,
    ):
        self.api_key = api_key or os.getenv("MESHY_API_KEY", "")
        self.timeout_sec = timeout_sec
        self.poll_interval_sec = poll_interval_sec

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import requests  # noqa: F401
            return True
        except ImportError:
            return False

    def complete(self, req: CompletionRequest, output_dir: Path) -> CompletionResult:
        t0 = time.time()
        log = []

        if not self.api_key:
            return CompletionResult(
                status=CompletionStatus.FAILED,
                provider_name=self.name,
                error="MESHY_API_KEY missing",
                observed_surface_ratio=req.observed_surface_ratio,
            )

        try:
            import requests
        except ImportError:
            return CompletionResult(
                status=CompletionStatus.FAILED,
                provider_name=self.name,
                error="requests package not installed",
                observed_surface_ratio=req.observed_surface_ratio,
            )

        if not req.reference_image_paths:
            return CompletionResult(
                status=CompletionStatus.FAILED,
                provider_name=self.name,
                error="Reference image required",
                observed_surface_ratio=req.observed_surface_ratio,
            )

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            with open(req.reference_image_paths[0], "rb") as f:
                resp = requests.post(
                    self.BASE_URL,
                    headers=headers,
                    files={"image": f},
                    data={"ai_model": "meshy-4", "topology": "triangle"},
                    timeout=60,
                )
            resp.raise_for_status()
            task_id = resp.json().get("result")
            if not task_id:
                return CompletionResult(
                    status=CompletionStatus.FAILED,
                    provider_name=self.name,
                    error=f"No task id in response: {resp.text[:200]}",
                    observed_surface_ratio=req.observed_surface_ratio,
                )
            log.append(f"Meshy task_id={task_id}")

            # Poll
            deadline = t0 + self.timeout_sec
            glb_url = None
            while time.time() < deadline:
                time.sleep(self.poll_interval_sec)
                poll = requests.get(f"{self.BASE_URL}/{task_id}", headers=headers, timeout=30)
                poll.raise_for_status()
                data = poll.json()
                status = data.get("status")
                progress = data.get("progress", 0)
                log.append(f"poll status={status} progress={progress}")
                if status == "SUCCEEDED":
                    glb_url = (data.get("model_urls") or {}).get("glb")
                    break
                if status in ("FAILED", "EXPIRED", "CANCELED"):
                    return CompletionResult(
                        status=CompletionStatus.FAILED,
                        provider_name=self.name,
                        error=f"Meshy task {status}",
                        observed_surface_ratio=req.observed_surface_ratio,
                        log=log,
                    )

            if not glb_url:
                return CompletionResult(
                    status=CompletionStatus.FAILED,
                    provider_name=self.name,
                    error="Timeout waiting for Meshy result",
                    observed_surface_ratio=req.observed_surface_ratio,
                    log=log,
                )

            output_dir.mkdir(parents=True, exist_ok=True)
            completed_path = output_dir / "ai_completed.glb"
            r = requests.get(glb_url, timeout=120)
            r.raise_for_status()
            completed_path.write_bytes(r.content)
            log.append(f"Downloaded → {completed_path}")

            return CompletionResult(
                status=CompletionStatus.PREVIEW_ONLY,
                provider_name=self.name,
                completed_mesh_path=str(completed_path),
                observed_surface_ratio=req.observed_surface_ratio,
                synthesized_surface_ratio=0.0,
                elapsed_sec=round(time.time() - t0, 2),
                log=log,
                metadata={"task_id": task_id},
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
