import os
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("cleanup")

class CleanupManager:
    """
    Manages automated cleanup of temporary files, old logs, and expired job artifacts.
    """
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.jobs_dir = self.data_root / "reconstructions"
        self.logs_dir = self.data_root / "logs"

    def cleanup_old_jobs(self, max_age_days: int = 30):
        """Deletes job directories older than max_age_days."""
        if not self.jobs_dir.exists():
            return

        now = time.time()
        cutoff = now - (max_age_days * 86400)
        count = 0

        for job_path in self.jobs_dir.iterdir():
            if job_path.is_dir():
                # Check modification time of the directory
                mtime = job_path.stat().st_mtime
                if mtime < cutoff:
                    try:
                        shutil.rmtree(job_path)
                        logger.info(f"Cleaned up old job directory: {job_path.name}")
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete job directory {job_path.name}: {e}")
        
        if count > 0:
            logger.info(f"Total jobs cleaned: {count}")

    def cleanup_temp_files(self):
        """Cleans up 'temp' subdirectories within all job directories."""
        if not self.jobs_dir.exists():
            return

        for job_path in self.jobs_dir.iterdir():
            temp_dir = job_path / "temp"
            if temp_dir.exists() and temp_dir.is_dir():
                try:
                    # Clear contents of temp but keep the directory if needed, 
                    # or just delete and recreate. Let's delete for thoroughness.
                    shutil.rmtree(temp_dir)
                    temp_dir.mkdir()
                    logger.info(f"Cleared temp files for job: {job_path.name}")
                except Exception as e:
                    logger.error(f"Failed to clear temp for {job_path.name}: {e}")

    def cleanup_logs(self, max_size_mb: int = 100):
        """Simple log rotation/pruning by size (basic version)."""
        if not self.logs_dir.exists():
            return

        for log_file in self.logs_dir.glob("*.log"):
            size_mb = log_file.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                try:
                    # Truncate log file
                    with open(log_file, "w") as f:
                        f.write(f"--- Log truncated at {datetime.now(timezone.utc)} due to size ({size_mb:.1f} MB) ---\n")
                    logger.info(f"Truncated large log file: {log_file.name}")
                except Exception as e:
                    logger.error(f"Failed to truncate log {log_file.name}: {e}")

    def run_full_cleanup(self):
        """Runs all cleanup tasks."""
        logger.info("Starting automated cleanup process...")
        self.cleanup_temp_files()
        self.cleanup_old_jobs(max_age_days=14) # Default to 14 days for safety
        self.cleanup_logs()
        logger.info("Cleanup process completed.")

if __name__ == "__main__":
    # Integration smoke test for cleanup
    manager = CleanupManager()
    manager.run_full_cleanup()
