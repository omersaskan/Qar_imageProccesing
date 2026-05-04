"""Rodin (Hyper3D Gen-2) AI 3D Provider."""
from __future__ import annotations
import logging
from typing import Dict, Any, Tuple, Optional
from .provider_base import AI3DRemoteAsyncProviderBase
from modules.operations.settings import settings

logger = logging.getLogger("rodin_provider")

class RodinProvider(AI3DRemoteAsyncProviderBase):
    name: str = "rodin"
    license_note: str = "Hyper3D Gen-2 commercial license applies."
    is_experimental: bool = True

    def is_available(self) -> Tuple[bool, str]:
        """Check if Rodin is enabled and API key is present."""
        if not getattr(settings, "rodin_enabled", False):
            return False, "Rodin provider is disabled in settings (RODIN_ENABLED=false)"
        if not getattr(settings, "rodin_api_key", ""):
            return False, "Rodin API key is missing (RODIN_API_KEY)"
        return True, ""

    def create_task(
        self,
        input_image_path: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Mocked Rodin task creation."""
        logger.info(f"[Rodin] Mocked create_task for {input_image_path}")
        # In a real implementation, this would call Hyper3D API
        # For now, return a mock task ID
        return "mock_rodin_task_123", None

    def poll_status(self, task_id: str) -> Tuple[str, Optional[str], float]:
        """Mocked Rodin status polling."""
        logger.info(f"[Rodin] Mocked poll_status for {task_id}")
        # Mock immediate success for testing
        return "succeeded", None, 1.0

    def download_result(self, task_id: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """Mocked Rodin result download."""
        import shutil
        from pathlib import Path
        
        logger.info(f"[Rodin] Mocked download_result for {task_id}")
        
        # Create a dummy GLB for testing
        dest = Path(output_dir) / f"{task_id}.glb"
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest, "wb") as f:
            f.write(b"MOCK_GLB_DATA")
            
        return str(dest), None
