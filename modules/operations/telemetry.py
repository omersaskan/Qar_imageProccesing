import json
from enum import Enum
from typing import Any, Dict, List
from datetime import datetime, timezone
from pathlib import Path
from modules.operations.logging_config import get_component_logger


class FailureCodes(str, Enum):
    ERR_CAPTURE_QUALITY = "ERR_CAPTURE_QUALITY"
    ERR_RECON_INPUT = "ERR_RECON_INPUT"
    ERR_RECON_RUNTIME = "ERR_RECON_RUNTIME"
    ERR_CLEANUP_INVALID_MESH = "ERR_CLEANUP_INVALID_MESH"
    ERR_EXPORT_MISSING_ARTIFACT = "ERR_EXPORT_MISSING_ARTIFACT"
    ERR_VALIDATION_FAIL = "ERR_VALIDATION_FAIL"
    ERR_PUBLISH_BLOCKED_REVIEW = "ERR_PUBLISH_BLOCKED_REVIEW"
    ERR_PUBLISH_BLOCKED_FAIL = "ERR_PUBLISH_BLOCKED_FAIL"


class OperationalTelemetry:
    """
    Structured telemetry: writes JSON-line entries to a log file and reports
    them via the standard logger.
    """

    def __init__(self, log_path: str = "data/logs/operational.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_component_logger("telemetry")
        self._entries: List[Dict[str, Any]] = []
        # Load existing entries if the file already exists
        if self.log_path.exists():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
            except Exception:
                self._entries = []

    def _write(self, entry: Dict[str, Any]) -> None:
        self._entries.append(entry)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, indent=2, default=str)

    def log_failure(
        self,
        component: str,
        job_id: str,
        failure_code: FailureCodes,
        reason: str,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "failure",
            "component": component,
            "job_id": job_id,
            "failure_code": failure_code.value,
            "reason": reason,
        }
        self._write(entry)
        self.logger.error(f"Failure in {component}: {reason}")

    def log_action(
        self,
        asset_id: str,
        action: str,
        metadata: Dict[str, Any],
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "action",
            "asset_id": asset_id,
            "action": action,
            "metadata": metadata,
        }
        self._write(entry)
        self.logger.info(f"Action: {action} for asset {asset_id}")
