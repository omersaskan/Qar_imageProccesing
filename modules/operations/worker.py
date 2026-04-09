import time
import json
import random
import threading
from pathlib import Path
from typing import Dict, Any

from modules.shared_contracts.models import AssetMetadata, CaptureSession
from modules.shared_contracts.lifecycle import AssetStatus
from modules.asset_registry.registry import AssetRegistry
from modules.capture_workflow.session_manager import SessionManager
from modules.operations.logging_config import get_component_logger
from modules.reconstruction_engine.output_manifest import OutputManifest
from modules.export_pipeline.glb_exporter import GLBExporter
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType
from modules.qa_validation.validator import AssetValidator

logger = get_component_logger("worker")


class WorkerError(Exception):
    pass


class RecoverableError(WorkerError):
    pass


class IrrecoverableError(WorkerError):
    pass


class IngestionWorker:
    def __init__(self, interval_sec: int = 5):
        self.interval = interval_sec
        self.registry = AssetRegistry()
        self.session_manager = SessionManager()
        self.running = False
        self._thread = None

        self.blobs_dir = Path("data/registry/blobs")
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = AssetCleaner()
        self.validator = AssetValidator()
        self.exporter = GLBExporter()
        self.job_tracker = self.registry

    def start(self):
        if self.running:
            return
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
        sessions_dir = Path("data/sessions")
        if not sessions_dir.exists():
            return

        for file in sessions_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session = CaptureSession.model_validate(data)

                if session.status == AssetStatus.CREATED:
                    self._advance_session(session, AssetStatus.CAPTURED, "Extracting frames from video...")
                elif session.status == AssetStatus.CAPTURED:
                    self._advance_session(session, AssetStatus.RECONSTRUCTED, "Reconstructing 3D geometry...")
                elif session.status == AssetStatus.RECONSTRUCTED:
                    self._finalize_ingestion(session)
                elif session.status == AssetStatus.FAILED:
                    continue
            except IrrecoverableError as ie:
                logger.error(f"💀 Irrecoverable error for {file.name}: {ie}")
                self._mark_session_failed(file.stem, str(ie))
            except RecoverableError as re:
                logger.warning(f"⏳ Transient error for {file.name}, will retry: {re}")
            except Exception as e:
                logger.error(f"❌ Unexpected failure handling {file.name}: {str(e)}")

    def _mark_session_failed(self, session_id: str, reason: str):
        try:
            session = self.session_manager.get_session(session_id)
            if session:
                session.failure_reason = reason
                self.session_manager.save_session(session)
                self.session_manager.update_session_status(session_id, AssetStatus.FAILED)
                logger.info(f"🚫 Session {session_id} marked as FAILED. Reason: {reason}")
        except Exception as e:
            logger.error(f"Failed to mark session {session_id} as FAILED: {e}")

    def _advance_session(self, session: CaptureSession, next_status: AssetStatus, log_msg: str):
        try:
            logger.info(f"🚀 PIPELINE START: {log_msg} ({session.session_id})", extra={"job_id": session.session_id})

            if next_status == AssetStatus.CAPTURED:
                self._handle_frame_extraction(session)
            elif next_status == AssetStatus.RECONSTRUCTED:
                self._handle_reconstruction(session)

            self.session_manager.update_session_status(session.session_id, next_status)
            logger.info(f"✅ PIPELINE STEP COMPLETE: {session.session_id} is now {next_status.value}")

        except (RecoverableError, IrrecoverableError):
            raise
        except Exception as e:
            raise RecoverableError(f"Unexpected error: {str(e)}")

    def _handle_frame_extraction(self, session: CaptureSession):
        try:
            from modules.capture_workflow.frame_extractor import FrameExtractor

            extractor = FrameExtractor()
            video_path = Path("data/captures") / session.session_id / "video" / "raw_video.mp4"
            output_dir = Path("data/captures") / session.session_id / "frames"

            if video_path.exists():
                logger.info(f"🛠️ Starting CV2 frame extraction for {session.session_id}...", extra={"job_id": session.session_id})
                frames = extractor.extract_keyframes(str(video_path), str(output_dir))
                if not frames:
                    raise IrrecoverableError(f"Frame extraction produced 0 frames for {session.session_id}.")
                logger.info(f"📸 Extraction successful: {len(frames)} frames saved.", extra={"job_id": session.session_id})
            else:
                raise IrrecoverableError(f"Video file missing at {video_path}.")
        except IrrecoverableError:
            raise
        except Exception as e:
            raise RecoverableError(f"Frame extraction failed: {e}")

    def _handle_reconstruction(self, session: CaptureSession):
        try:
            from modules.reconstruction_engine.job_manager import JobManager
            from modules.reconstruction_engine.runner import ReconstructionRunner
            from modules.shared_contracts.models import ReconstructionJobDraft

            frames_dir = Path("data/captures") / session.session_id / "frames"
            if not frames_dir.exists():
                self._handle_frame_extraction(session)

            input_frames = [str(f) for f in frames_dir.glob("*.jpg")]
            if not input_frames:
                raise IrrecoverableError(f"No frames available for reconstruction in {session.session_id}")

            manager = JobManager()
            runner = ReconstructionRunner()

            draft = ReconstructionJobDraft(
                job_id=f"job_{session.session_id}",
                capture_session_id=session.session_id,
                input_frames=input_frames,
                product_id=session.product_id,
            )

            job = manager.create_job(draft)
            manager.update_job_status(job.job_id, "running")

            logger.info(f"🏗️ Starting Geometric Reconstruction for {session.session_id} with {len(input_frames)} frames...")
            manifest = runner.run(job)

            manager.update_job_status(job.job_id, "completed")
            logger.info(
                f"✨ Reconstruction successful: {manifest.processing_time_seconds:.2f}s, "
                f"{manifest.mesh_metadata.vertex_count} vertices."
            )

        except IrrecoverableError:
            raise
        except Exception as e:
            msg = str(e)
            if "not configured" in msg or "prohibited" in msg or "VIOLATION" in msg:
                raise IrrecoverableError(f"Engine configuration error: {msg}")
            raise RecoverableError(f"Engine failure: {e}")

    def _finalize_ingestion(self, session: CaptureSession):
        logger.info(f"🏁 Finalizing ingestion: Processing assets for {session.product_id}...", extra={"job_id": session.session_id})

        job_id = f"job_{session.session_id}"
        job_dir = Path("data/reconstructions") / job_id
        manifest_path = job_dir / "manifest.json"

        if not manifest_path.exists():
            raise IrrecoverableError(f"Reconstruction manifest missing for {session.session_id} at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            manifest = OutputManifest.model_validate(manifest_data)

        logger.info(f"🧹 Starting mesh cleanup for {session.session_id}...")
        try:
            metadata, cleanup_stats, cleaned_mesh_path = self.cleaner.process_cleanup(
                job_id=job_id,
                raw_mesh_path=manifest.mesh_path,
                profile_type=CleanupProfileType.MOBILE_DEFAULT,
                raw_texture_path=manifest.texture_path,
            )
        except Exception as e:
            raise IrrecoverableError(f"Mesh cleanup failed: {e}")

        logger.info(f"⚖️ Validating asset quality for {session.session_id}...")

        import trimesh

        cleaned_mesh = trimesh.load(cleaned_mesh_path)
        if isinstance(cleaned_mesh, trimesh.Scene):
            cleaned_mesh = cleaned_mesh.dump(concatenate=True)

        dimensions = {
            "x": float(cleaned_mesh.bounds[1][0] - cleaned_mesh.bounds[0][0]),
            "y": float(cleaned_mesh.bounds[1][1] - cleaned_mesh.bounds[0][1]),
            "z": float(cleaned_mesh.bounds[1][2] - cleaned_mesh.bounds[0][2]),
        }
        ground_residual = abs(float(cleaned_mesh.bounds[0][2]))

        texture_path = cleanup_stats.get("cleaned_texture_path") or manifest.texture_path
        texture_path_exists = bool(texture_path and Path(texture_path).exists())
        has_uv = bool(cleanup_stats.get("uv_preserved", False))
        has_material = bool(cleanup_stats.get("material_preserved", False))
        texture_status = "complete" if texture_path_exists and has_uv else ("degraded" if texture_path_exists else "missing")

        validation_input = {
            "poly_count": metadata.final_polycount,
            "texture_status": texture_status,
            "bbox": dimensions,
            "ground_offset": ground_residual,
            "cleanup_stats": cleanup_stats,
            "texture_path_exists": texture_path_exists,
            "has_uv": has_uv,
            "has_material": has_material,
            "texture_applied_successfully": texture_path_exists and has_uv,
        }

        timestamp = int(time.time())
        asset_id = f"{session.product_id}_{timestamp}"

        report = self.validator.validate(asset_id, validation_input)
        logger.info(f"📊 Validation Decision: {report.final_decision.upper()} :: {report.contamination_report}")

        if report.final_decision == "fail":
            logger.error(f"❌ Asset FAILED validation: {report.contamination_report}")
            self._mark_session_failed(session.session_id, f"Validation Failed: {report.contamination_report}")
            return

        blob_path = self.blobs_dir / f"{asset_id}.glb"

        logger.info(f"📦 Exporting CLEANED GLB from {cleaned_mesh_path}...")
        try:
            export_result = self.exporter.export(
                mesh_path=cleaned_mesh_path,
                output_path=str(blob_path),
                profile_name="standard",
                texture_path=texture_path if texture_path_exists else None,
                metadata=metadata,
            )
            logger.info(
                f"✅ GLB Export successful: {export_result['filesize']} bytes | "
                f"has_uv={export_result.get('has_uv')} "
                f"texture_ok={export_result.get('texture_applied_successfully')}"
            )
        except Exception as e:
            raise IrrecoverableError(f"GLB Export failed: {e}")

        registry_metadata = AssetMetadata(
            asset_id=asset_id,
            product_id=session.product_id,
            version=f"v{random.randint(1, 5)}.0",
            bbox={"min": metadata.bbox_min, "max": metadata.bbox_max, "dimensions": dimensions},
            pivot_offset=metadata.pivot_offset,
            quality_grade=report.mobile_performance_grade,
        )

        self.registry.register_asset(registry_metadata)
        self.registry.set_active_version(session.product_id, asset_id)

        if manifest.is_stub:
            self.registry.update_publish_state(asset_id, "draft")
        elif report.final_decision == "review":
            self.registry.update_publish_state(asset_id, "draft")
        else:
            self.registry.update_publish_state(asset_id, "published")

        try:
            session_file = Path("data/sessions") / f"{session.session_id}.json"
            if session_file.exists():
                session_file.unlink()
            logger.info(f"✅ Success! {session.product_id} is now registered as {asset_id}.")
        except Exception as e:
            logger.error(f"Worker cleanup failed: {str(e)}")


worker_instance = IngestionWorker()