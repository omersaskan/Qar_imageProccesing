"""Apple Depth Pro provider (experimental)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Tuple

from .depth_provider_base import DepthProviderBase


class DepthProProvider(DepthProviderBase):

    name = "depth_pro"
    license_note = (
        "Depth Pro — Apple Research License. "
        "Non-commercial research use only. "
        "Check license before production deployment."
    )
    is_experimental = True

    def __init__(self, checkpoint: str = "", device: str = "cpu"):
        self.checkpoint = checkpoint or os.environ.get("DEPTH_PRO_CHECKPOINT", "")
        self.device = device

    def is_available(self) -> Tuple[bool, str]:
        from modules.operations.settings import settings
        if not settings.depth_pro_enabled:
            return False, "DEPTH_PRO_ENABLED=false"
        
        python_path = settings.depth_pro_python_path
        if not python_path:
            return False, "DEPTH_PRO_PYTHON_PATH not configured"
        
        if not Path(python_path).exists():
            return False, f"Depth Pro Python not found at {python_path}"
            
        return True, ""

    def infer(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        from modules.operations.settings import settings
        import subprocess
        import json

        python_path = settings.depth_pro_python_path
        worker_script = str(Path("scripts/depth_pro_worker.py").resolve())
        
        cmd = [
            str(Path(python_path).resolve()),
            worker_script,
            "--image", str(Path(image_path).resolve()),
            "--output", str(Path(output_dir).resolve()),
            "--device", self.device
        ]
        
        if self.checkpoint:
            cmd.extend(["--checkpoint", str(Path(self.checkpoint).resolve())])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            # Find JSON in stdout (in case of other prints)
            stdout = result.stdout.strip()
            # The worker is expected to print exactly one JSON line
            data = json.loads(stdout)
            
            if data.get("status") == "ok":
                return {
                    "status": "ok",
                    "provider": self.name,
                    "depth_map_path": data["depth_map_path"],
                    "depth_format": data["depth_format"],
                    "model_name": data["model_name"],
                    "warnings": ["isolated_worker"],
                }
            else:
                return {
                    "status": "error",
                    "message": data.get("message", "Unknown worker error"),
                    "provider": self.name
                }

        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "message": f"Worker process failed (exit {e.returncode}): {e.stderr}",
                "provider": self.name
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Unexpected error during isolated inference: {str(e)}",
                "provider": self.name
            }
