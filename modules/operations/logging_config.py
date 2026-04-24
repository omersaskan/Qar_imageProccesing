import logging
import json
from typing import List, Dict, Any, Optional
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from .settings import settings, AppEnvironment

class JsonFormatter(logging.Formatter):
    """
    Custom formatter to output logs in JSON format for production observability.
    """
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "env": settings.env.value,
            "component": getattr(record, "component", "root"),
            "message": record.getMessage(),
            "job_id": getattr(record, "job_id", None),
            "asset_id": getattr(record, "asset_id", None),
            "session_id": getattr(record, "session_id", None),
            "stage": getattr(record, "stage", None),
            "duration_ms": getattr(record, "duration_ms", None),
            "failure_code": getattr(record, "failure_code", None),
        }
        # Include exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

class ComponentFilter(logging.Filter):
    """
    Ensures 'component' field exists on every log record.
    """
    def filter(self, record):
        if not hasattr(record, "component"):
            record.component = "root"
        return True

def setup_logging():
    """
    Configures the root logger with both console and file handlers based on Settings.
    """
    log_dir = Path(settings.data_root) / "logs"
    log_file = log_dir / "factory.log"
    
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating log directory: {e}")

    # Determine level based on env
    if settings.env == AppEnvironment.PRODUCTION:
        level = logging.WARNING
    elif settings.env == AppEnvironment.PILOT:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logger = logging.getLogger()
    logger.setLevel(level)

    # Console Handler (Human readable)
    console_handler = logging.StreamHandler()
    console_handler.addFilter(ComponentFilter())
    
    # Prettier console logs for Dev, cleaner for Pilot/Prod
    if settings.is_dev:
        fmt = '[%(asctime)s] %(levelname)s [%(component)s] %(message)s'
    else:
        fmt = '%(levelname)s [%(component)s] %(message)s'
        
    console_formatter = logging.Formatter(fmt, datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    
    # File Handler (JSON for machine reading)
    try:
        file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
        file_handler.setFormatter(JsonFormatter())
        
        # Clear existing handlers to avoid duplicates
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logging: {e}")
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(console_handler)

    logger.info(f"Logging initialized in {settings.env.value} mode")
    return logger

def get_component_logger(name: str):
    """Returns a logger pre-configured for a specific component."""
    logger = logging.getLogger(name)
    logger.propagate = True
    return logging.LoggerAdapter(logger, {"component": name})

@contextmanager
def log_stage(logger: logging.LoggerAdapter, stage_name: str, session_id: Optional[str] = None):
    """Context manager to log the start and end of a pipeline stage with duraton."""
    start_time = time.time()
    extra = {"stage": stage_name, "session_id": session_id}
    
    logger.info(f"Starting stage: {stage_name}", extra=extra)
    try:
        yield
        duration_ms = int((time.time() - start_time) * 1000)
        extra["duration_ms"] = duration_ms
        logger.info(f"Completed stage: {stage_name} (took {duration_ms}ms)", extra=extra)
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        extra["duration_ms"] = duration_ms
        logger.error(f"Failed stage: {stage_name} (failed after {duration_ms}ms): {str(e)}", extra=extra)
        raise
