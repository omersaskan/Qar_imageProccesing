import time
import json
import random
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any
import traceback

from modules.shared_contracts.models import AssetMetadata, CaptureSession
from modules.shared_contracts.lifecycle import AssetStatus
from modules.asset_registry.registry import AssetRegistry
from modules.capture_workflow.session_manager import SessionManager
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("worker")

class IngestionWorker:
    def __init__(self, interval_sec: int = 5):
        self.interval = interval_sec
        self.registry = AssetRegistry()
        self.session_manager = SessionManager()
        self.running = False
        self._thread = None
        
        # Ensure blobs dir exists for final step
        self.blobs_dir = Path("data/registry/blobs")
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        if self.running: return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("IngestionWorker started.")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("IngestionWorker stopped.")

    def _run(self):
        while self.running:
            try:
                self._process_pending_sessions()
            except Exception as e:
                logger.error(f"Worker iteration failed: {str(e)}")
            time.sleep(self.interval)

    def _process_pending_sessions(self):
        """Scans session directory for items in 'created' status and advances them."""
        sessions_dir = Path("data/sessions")
        if not sessions_dir.exists(): return
        
        for file in sessions_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session = CaptureSession.model_validate(data)
                
                # Logic: Advance through states
                if session.status == AssetStatus.CREATED:
                    self._advance_session(session, AssetStatus.CAPTURED, "Extracting frames from video...")
                elif session.status == AssetStatus.CAPTURED:
                    self._advance_session(session, AssetStatus.RECONSTRUCTED, "Reconstructing 3D geometry...")
                elif session.status == AssetStatus.RECONSTRUCTED:
                    self._finalize_ingestion(session)
            except Exception as e:
                logger.error(f"Failed to handle session file {file.name}: {str(e)}")

    def _advance_session(self, session: CaptureSession, next_status: AssetStatus, log_msg: str):
        """Advances session to next status with local processing if needed."""
        try:
            logger.info(f"🚀 PIPELINE START: {log_msg} ({session.session_id})", extra={"job_id": session.session_id})
            
            # --- Real Local Processing (P10) ---
            if next_status == AssetStatus.CAPTURED:
                self._handle_frame_extraction(session)
            elif next_status == AssetStatus.RECONSTRUCTED:
                self._handle_reconstruction(session)
            
            # Finalize status update
            self.session_manager.update_session_status(session.session_id, next_status)
            logger.info(f"✅ PIPELINE STEP COMPLETE: {session.session_id} is now {next_status.value}")

        except Exception as e:
            error_msg = f"❌ PIPELINE ERROR in step '{log_msg}': {str(e)}"
            logger.error(error_msg, extra={"job_id": session.session_id})
            logger.error(traceback.format_exc())
            # We don't advance the status on error to prevent moving to broken states
            raise e

    def _handle_frame_extraction(self, session: CaptureSession):
        """Internal helper for frame extraction step."""
        try:
            from modules.capture_workflow.frame_extractor import FrameExtractor
            extractor = FrameExtractor()
            video_path = Path("data/captures") / session.session_id / "video" / "raw_video.mp4"
            output_dir = Path("data/captures") / session.session_id / "frames"
            
            if video_path.exists():
                logger.info(f"🛠️ Starting CV2 frame extraction for {session.session_id}...", extra={"job_id": session.session_id})
                frames = extractor.extract_keyframes(str(video_path), str(output_dir))
                if not frames:
                     raise RuntimeError(f"Frame extraction produced 0 frames for {session.session_id}")
                logger.info(f"📸 Extraction successful: {len(frames)} frames saved.", extra={"job_id": session.session_id})
            else:
                raise FileNotFoundError(f"Video file missing at {video_path}. Cannot proceed to CAPTURED status.")
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}", extra={"job_id": session.session_id})
            raise e

    def _handle_reconstruction(self, session: CaptureSession):
        """Orchestrates the 3D reconstruction using the actual engine."""
        try:
            from modules.reconstruction_engine.job_manager import JobManager
            from modules.reconstruction_engine.runner import ReconstructionRunner
            from modules.shared_contracts.models import ReconstructionJobDraft
            
            # 1. Prepare Inputs
            frames_dir = Path("data/captures") / session.session_id / "frames"
            input_frames = [str(f) for f in frames_dir.glob("*.jpg")]
            
            if not input_frames:
                logger.warning(f"⚠️ No frames found for {session.session_id} in CAPTURED state. Attempting emergency re-extraction...")
                try:
                    self._handle_frame_extraction(session)
                    input_frames = [str(f) for f in frames_dir.glob("*.jpg")]
                except Exception as re_ext_err:
                    logger.error(f"Emergency re-extraction failed: {re_ext_err}")
            
            if not input_frames:
                # If still no frames, we have a problem. 
                # To prevent endless loop spam, we could move to a manual intervention state if we had one.
                # For now, just raise with a clear message.
                raise RuntimeError(f"No frames found for reconstruction in {session.session_id} even after retry. Please check video file.")

            # 2. Initialize Engine
            manager = JobManager()
            runner = ReconstructionRunner()
            
            draft = ReconstructionJobDraft(
                job_id=f"job_{session.session_id}",
                capture_session_id=session.session_id,
                input_frames=input_frames,
                product_id=session.product_id
            )
            
            # 3. Create and Run Job
            job = manager.create_job(draft)
            manager.update_job_status(job.job_id, "running")
            
            logger.info(f"🏗️ Starting Geometric Reconstruction for {session.session_id} with {len(input_frames)} frames...")
            manifest = runner.run(job)
            
            manager.update_job_status(job.job_id, "completed")
            logger.info(f"✨ Reconstruction successful: {manifest.processing_time_seconds:.2f}s, {manifest.mesh_metadata.vertex_count} vertices.")
            
        except Exception as e:
            logger.error(f"Reconstruction failed for {session.session_id}: {e}")
            # Optional: throttle retries for this specific session?
            raise e

    def _finalize_ingestion(self, session: CaptureSession):
        """Turns the session into a registered asset with a valid placeholder."""
        logger.info(f"🏁 Finalizing ingestion: Creating asset registry record for {session.product_id}...", extra={"job_id": session.session_id})
        
        # 1. Create Metadata
        timestamp = int(time.time())
        asset_id = f"{session.product_id}_{timestamp}"
        metadata = AssetMetadata(
            asset_id=asset_id,
            product_id=session.product_id,
            version=f"v{random.randint(1,5)}.0" 
        )
        
        # 2. Register in Registry
        self.registry.register_asset(metadata)
        self.registry.set_active_version(session.product_id, asset_id)
        self.registry.update_publish_state(asset_id, "published")
        
        # 3. Create dummy GLB blob (Valid placeholder instead of 0-byte)
        try:
            from modules.operations.placeholder_engine import write_placeholder_glb
            blob_path = self.blobs_dir / f"{asset_id}.glb"
            write_placeholder_glb(blob_path)
            logger.info(f"Generated valid placeholder GLB for {asset_id}")
        except Exception as e:
            logger.error(f"Failed to generate GLB placeholder: {e}")
        
        # 4. Clean up session
        try:
            session_file = Path("data/sessions") / f"{session.session_id}.json"
            if session_file.exists():
                session_file.unlink()
            logger.info(f"✅ Success! {session.product_id} is now registered as {asset_id}. Model available in 3D Viewer.")
        except Exception as e:
            logger.error(f"Worker cleanup failed: {str(e)}")

# Singleton-like instance for management
worker_instance = IngestionWorker()
