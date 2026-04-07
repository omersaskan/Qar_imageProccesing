import logging
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from modules.operations.logging_config import setup_logging, get_component_logger

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
    Structured logging for operational visibility using standard logging.
    """
    def __init__(self, log_path: str = "data/logs/operational.log"):
        # Initialize logging global config if not already done
        self.logger = get_component_logger("telemetry")
        setup_logging(log_file=log_path)

    def log_failure(self, component: str, job_id: str, failure_code: FailureCodes, reason: str):
        extra = {
            "component": component,
            "job_id": job_id,
            "failure_code": failure_code.value
        }
        # Log via adapter to inject extra fields
        self.logger.error(f"Failure in {component}: {reason}", extra=extra)

    def log_action(self, asset_id: str, action_type: str, metadata: Dict[str, Any]):
        extra = {
            "asset_id": asset_id,
            "action_type": action_type,
            "metadata": metadata
        }
        self.logger.info(f"Action: {action_type} for asset {asset_id}", extra=extra)
