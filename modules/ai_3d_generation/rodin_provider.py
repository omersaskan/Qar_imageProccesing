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
        """Check if Rodin is enabled and correctly configured."""
        if not getattr(settings, "rodin_enabled", False):
            return False, "Rodin provider is disabled in settings (RODIN_ENABLED=false)"
        
        # Production guard for mock mode
        if getattr(settings, "rodin_mock_mode", False) and settings.env != "local_dev":
            return False, "Rodin mock mode is prohibited in non-local_dev environment"

        if not getattr(settings, "rodin_api_key", ""):
            return False, "Rodin API key is missing (RODIN_API_KEY)"
        
        # Real API not implemented check
        if not getattr(settings, "rodin_mock_mode", False):
             # When mock is false, we'd normally check if the real client is initialized.
             # Since it's not implemented yet, we fail fast.
             return False, "rodin_real_api_not_implemented"
             
        return True, ""

    def create_task(
        self,
        input_image_path: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Mocked Rodin task creation."""
        if not getattr(settings, "rodin_mock_mode", False):
             return None, "rodin_real_api_not_implemented"
             
        logger.info(f"[Rodin] Mocked create_task for {input_image_path}")
        return "mock_rodin_task_123", None

    def poll_status(self, task_id: str) -> Tuple[str, Optional[str], float]:
        """Mocked Rodin status polling."""
        if not getattr(settings, "rodin_mock_mode", False):
             return "failed", "rodin_real_api_not_implemented", 0.0
             
        logger.info(f"[Rodin] Mocked poll_status for {task_id}")
        return "succeeded", None, 1.0

    def download_result(self, task_id: str, output_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """Mocked Rodin result download."""
        import shutil
        from pathlib import Path
        
        if not getattr(settings, "rodin_mock_mode", False):
             return None, "rodin_real_api_not_implemented"

        logger.info(f"[Rodin] Mocked download_result for {task_id}")
        
        # Create a dummy GLB for testing
        dest = Path(output_dir) / f"{task_id}.glb"
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest, "wb") as f:
            f.write(b"MOCK_GLB_DATA")
            
        return str(dest), None

