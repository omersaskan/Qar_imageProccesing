import logging
import json
import os
from datetime import datetime, timezone
from pathlib import Path

class JsonFormatter(logging.Formatter):
    """
    Custom formatter to output logs in JSON format for production observability.
    """
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": getattr(record, "component", "root"),
            "message": record.getMessage(),
            "job_id": getattr(record, "job_id", None),
            "asset_id": getattr(record, "asset_id", None),
            "failure_code": getattr(record, "failure_code", None),
            "action_type": getattr(record, "action_type", None)
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

def setup_logging(log_file: str = "data/logs/factory.log", level=logging.DEBUG):
    """
    Configures the root logger with both console and file handlers.
    Sets default level to DEBUG for troubleshooting.
    """
    log_path = Path(log_file)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating log directory: {e}")

    logger = logging.getLogger()
    logger.setLevel(level)

    # Console Handler (Human readable)
    console_handler = logging.StreamHandler()
    console_handler.addFilter(ComponentFilter())
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(component)s] %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S%z'
    )
    console_handler.setFormatter(console_formatter)
    
    # File Handler (JSON for machine reading)
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
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

    return logger

def get_component_logger(name: str):
    """Returns a logger pre-configured for a specific component."""
    logger = logging.getLogger(name)
    # Ensure the logger propagates to the root logger
    logger.propagate = True
    # We use a LoggerAdapter to inject component name automatically
    return logging.LoggerAdapter(logger, {"component": name})
