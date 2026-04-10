import shutil
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone, timedelta
from .settings import settings, AppEnvironment
from .logging_config import get_component_logger
from modules.capture_workflow.session_manager import SessionManager
from modules.shared_contracts.lifecycle import AssetStatus

logger = get_component_logger("retention")

class RetentionService:
    def __init__(self, data_root: Optional[str] = None):
        self.data_root = Path(data_root or settings.data_root).resolve()
        self.session_manager = SessionManager(data_root=str(self.data_root))
        
    def run_cleanup(self):
        """
        Main entry point for the retention cycle.
        """
        start_time = time.time()
        logger.info("Starting retention cleanup cycle.")
        
        try:
            self._prune_session_artifacts()
            self._prune_reconstruction_scratch()
        except Exception as e:
            logger.error(f"Retention cycle failed: {str(e)}", exc_info=True)
            
        duration = time.time() - start_time
        logger.info(f"Retention cleanup cycle completed in {duration:.2f}s.")

    def _prune_session_artifacts(self):
        """
        Prunes raw frames and videos based on session status and age.
        """
        sessions_dir = self.session_manager.sessions_dir
        if not sessions_dir.exists():
            return

        now = datetime.now(timezone.utc)
        
        for session_file in sessions_dir.glob("*.json"):
            try:
                session = self.session_manager.get_session(session_file.stem)
                if not session:
                    continue
                
                # Determine threshold
                if session.status == AssetStatus.PUBLISHED:
                    threshold_days = settings.published_frames_days
                elif session.status in [AssetStatus.FAILED, AssetStatus.RECAPTURE_REQUIRED]:
                    threshold_days = settings.failed_frames_days
                else:
                    # Session still in progress or in other states, skip
                    continue
                
                capture_path = self.session_manager.get_capture_path(session.session_id)
                if not capture_path.exists():
                    continue

                # We use the session file's mtime or the session's created_at?
                # Using created_at might be too aggressive if a job took a long time.
                # Let's use mtime of the session file as a proxy for 'last activity'.
                last_activity = datetime.fromtimestamp(session_file.stat().st_mtime, tz=timezone.utc)
                age = now - last_activity
                
                if age > timedelta(days=threshold_days):
                    self._cleanup_raw_data(capture_path)
                    
            except Exception as e:
                logger.warning(f"Failed to process retention for session {session_file.stem}: {e}")

    def _cleanup_raw_data(self, capture_path: Path):
        """Removes 'video' and 'frames' folders while keeping 'reports'."""
        for folder in ["video", "frames"]:
            target = capture_path / folder
            if target.exists():
                logger.info(f"Retention: Pruning raw data folder {target}")
                shutil.rmtree(target, ignore_errors=True)

    def _prune_reconstruction_scratch(self):
        """
        Prunes heavy intermediate folders in data/reconstructions.
        Keeps: manifest.json, log, and final mesh/texture.
        """
        recon_root = self.data_root / "reconstructions"
        if not recon_root.exists():
            return
            
        now = datetime.now(timezone.utc)
        threshold_hours = settings.reconstruction_scratch_hours
        
        for job_dir in recon_root.iterdir():
            if not job_dir.is_dir():
                continue
                
            # Check age of job.json or the dir itself
            job_file = job_dir / "job.json"
            if not job_file.exists():
                # Orphaned reconstruction folder? Prune if older than 24h
                mtime = datetime.fromtimestamp(job_dir.stat().st_mtime, tz=timezone.utc)
                if now - mtime > timedelta(hours=24):
                    logger.info(f"Retention: Pruning orphaned recon folder {job_dir}")
                    shutil.rmtree(job_dir, ignore_errors=True)
                continue
                
            last_activity = datetime.fromtimestamp(job_file.stat().st_mtime, tz=timezone.utc)
            if now - last_activity > timedelta(hours=threshold_hours):
                self._cleanup_recon_scratch(job_dir)

    def _cleanup_recon_scratch(self, job_dir: Path):
        """
        Removes large intermediate COLMAP/OpenMVS folders.
        """
        # Folders to prune
        to_prune = ["images", "masks", "sparse", "dense", "temp"]
        # Files to prune
        to_prune_files = ["database.db"]
        
        found_any = False
        for folder in to_prune:
            target = job_dir / folder
            if target.exists():
                logger.info(f"Retention: Pruning recon scratch folder {target}")
                shutil.rmtree(target, ignore_errors=True)
                found_any = True
                
        for filename in to_prune_files:
            target = job_dir / filename
            if target.exists():
                logger.info(f"Retention: Pruning recon scratch file {target}")
                target.unlink(missing_ok=True)
                found_any = True
        
        if found_any:
            logger.info(f"Retention: Scrubbed scratch from {job_dir.name}")
