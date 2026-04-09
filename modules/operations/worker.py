import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType
from modules.asset_registry.registry import AssetRegistry
from modules.capture_workflow.session_manager import SessionManager
from modules.export_pipeline.glb_exporter import GLBExporter
from modules.operations.logging_config import get_component_logger
from modules.qa_validation.validator import AssetValidator
from modules.reconstruction_engine.output_manifest import OutputManifest
from modules.shared_contracts.lifecycle import AssetStatus, ReconstructionStatus
from modules.shared_contracts.models import AssetMetadata, CaptureSession, ValidationReport
from modules.utils.file_persistence import FileLock, atomic_write_json
from modules.utils.path_safety import validate_identifier

logger = get_component_logger("worker")


class WorkerError(Exception):
    pass


class RecoverableError(WorkerError):
    pass


class IrrecoverableError(WorkerError):
    pass


class IngestionWorker:
    def __init__(self, interval_sec: int = 5, data_root: str = "data"):
        self.interval = interval_sec
        self.data_root = Path(data_root).resolve()
        self.registry = AssetRegistry()
        self.session_manager = SessionManager(data_root=str(self.data_root))
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._process_lock: Optional[FileLock] = None

        self.blobs_dir = self.data_root / "registry" / "blobs"
        self.blobs_dir.mkdir(parents=True, exist_ok=True)
        self._worker_lock_path = self.data_root / "worker.process"

        self.cleaner = AssetCleaner(data_root=str(self.data_root))
        self.validator = AssetValidator()
        self.exporter = GLBExporter()
        self.job_tracker = self.registry

    def start(self):
        if self.running:
            return

        process_lock = FileLock(
            self._worker_lock_path,
            timeout=0.1,
            stale_threshold=max(float(self.interval * 6), 30.0),
        )
        try:
            process_lock.__enter__()
        except TimeoutError:
            logger.warning("IngestionWorker start skipped because another process already holds the worker lock.")
            return

        self._process_lock = process_lock
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="meshysiz-worker")
        self._thread.start()
        logger.info("IngestionWorker started.")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

        if self._process_lock is not None:
            self._process_lock.__exit__(None, None, None)
            self._process_lock = None

        logger.info("IngestionWorker stopped.")

    def _run(self):
        while self.running:
            try:
                self._process_pending_sessions()
            except Exception as e:
                logger.error(f"Worker iteration failed: {str(e)}")
            time.sleep(self.interval)

    def _process_pending_sessions(self):
        sessions_dir = self.session_manager.sessions_dir
        if not sessions_dir.exists():
            return

        for file in sessions_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    session = CaptureSession.model_validate(json.load(f))

                if session.status == AssetStatus.CREATED:
                    self._advance_session(session, AssetStatus.CAPTURED, "Extracting frames from video...")
                elif session.status == AssetStatus.CAPTURED:
                    self._advance_session(session, AssetStatus.RECONSTRUCTED, "Reconstructing 3D geometry...")
                elif session.status == AssetStatus.RECAPTURE_REQUIRED:
                    continue
                elif session.status == AssetStatus.RECONSTRUCTED:
                    self._advance_session(session, AssetStatus.CLEANED, "Cleaning reconstructed mesh...")
                elif session.status == AssetStatus.CLEANED:
                    self._advance_session(session, AssetStatus.EXPORTED, "Exporting final GLB...")
                elif session.status == AssetStatus.EXPORTED:
                    self._advance_session(session, AssetStatus.VALIDATED, "Validating exported asset...")
                elif session.status == AssetStatus.VALIDATED:
                    if session.publish_state not in {"draft", "published"}:
                        self._handle_publish(session)
                elif session.status in {AssetStatus.PUBLISHED, AssetStatus.FAILED}:
                    continue

            except IrrecoverableError as ie:
                logger.error(f"Irrecoverable error for {file.name}: {ie}")
                self._mark_session_failed(file.stem, str(ie))
            except RecoverableError as re:
                logger.warning(f"Transient error for {file.name}, will retry: {re}")
            except Exception as e:
                logger.error(f"Unexpected failure handling {file.name}: {str(e)}")

    def _persist_session(self, session: CaptureSession, new_status: Optional[AssetStatus] = None, **fields: Any) -> CaptureSession:
        persisted = self.session_manager.get_session(session.session_id)
        if persisted is None:
            if new_status is not None:
                session.status = new_status
            for field_name, value in fields.items():
                setattr(session, field_name, value)
            return session

        return self.session_manager.update_session(session.session_id, new_status=new_status, **fields)

    def _mark_session_failed(self, session_id: str, reason: str):
        try:
            session = self.session_manager.get_session(session_id)
            if session is None:
                return

            if session.status == AssetStatus.FAILED:
                self.session_manager.update_session(
                    session_id,
                    failure_reason=reason,
                    publish_state="failed",
                    last_pipeline_stage=AssetStatus.FAILED.value,
                )
            else:
                self.session_manager.update_session(
                    session_id,
                    new_status=AssetStatus.FAILED,
                    failure_reason=reason,
                    publish_state="failed",
                    last_pipeline_stage=AssetStatus.FAILED.value,
                )
            logger.info(f"Session {session_id} marked as FAILED. Reason: {reason}")
        except Exception as e:
            logger.error(f"Failed to mark session {session_id} as FAILED: {e}")

    def _mark_session_needs_recapture(
        self,
        session: CaptureSession,
        reason: str,
        coverage_score: float = 0.0,
        extracted_frames: Optional[list[str]] = None,
    ) -> CaptureSession:
        logger.info(f"Session {session.session_id} requires recapture. Reason: {reason}")
        fields: Dict[str, Any] = {
            "publish_state": "needs_recapture",
            "coverage_score": float(coverage_score),
            "failure_reason": reason,
            "last_pipeline_stage": AssetStatus.RECAPTURE_REQUIRED.value,
        }
        if extracted_frames is not None:
            fields["extracted_frames"] = extracted_frames
        return self._persist_session(
            session,
            new_status=AssetStatus.RECAPTURE_REQUIRED,
            **fields,
        )

    def _advance_session(self, session: CaptureSession, next_status: AssetStatus, log_msg: str):
        try:
            logger.info(f"PIPELINE START: {log_msg} ({session.session_id})", extra={"job_id": session.session_id})

            if next_status == AssetStatus.CAPTURED:
                updated = self._handle_frame_extraction(session)
            elif next_status == AssetStatus.RECONSTRUCTED:
                updated = self._handle_reconstruction(session)
            elif next_status == AssetStatus.CLEANED:
                updated = self._handle_cleanup(session)
            elif next_status == AssetStatus.EXPORTED:
                updated = self._handle_export(session)
            elif next_status == AssetStatus.VALIDATED:
                updated = self._handle_validation(session)
            else:
                raise IrrecoverableError(f"Unsupported pipeline transition target: {next_status.value}")

            logger.info(f"PIPELINE STEP COMPLETE: {updated.session_id} is now {updated.status.value}")

        except (RecoverableError, IrrecoverableError):
            raise
        except Exception as e:
            raise RecoverableError(f"Unexpected error: {str(e)}")

    def _handle_frame_extraction(self, session: CaptureSession) -> CaptureSession:
        try:
            from modules.capture_workflow.frame_extractor import FrameExtractor
            from modules.capture_workflow.coverage_analyzer import CoverageAnalyzer

            extractor = FrameExtractor()
            coverage_analyzer = CoverageAnalyzer()
            video_path = self.session_manager.captures_dir / session.session_id / "video" / "raw_video.mp4"
            output_dir = self.session_manager.captures_dir / session.session_id / "frames"
            reports_dir = self.session_manager.get_capture_path(session.session_id) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            if not video_path.exists():
                raise IrrecoverableError(f"Video file missing at {video_path}.")

            frames = extractor.extract_keyframes(str(video_path), str(output_dir))
            if not frames:
                raise IrrecoverableError(f"Frame extraction produced 0 frames for {session.session_id}.")
            min_frames = getattr(getattr(extractor, "config", None), "min_frames", 3)
            if len(frames) < int(min_frames):
                raise IrrecoverableError(
                    f"Frame extraction produced {len(frames)} validated frames for {session.session_id}; "
                    f"minimum required is {int(min_frames)}."
                )

            coverage_report = coverage_analyzer.analyze_coverage(frames)
            atomic_write_json(reports_dir / "coverage_report.json", coverage_report)

            if coverage_report["overall_status"] != "sufficient":
                reasons = "; ".join(coverage_report.get("reasons", [])) or "insufficient viewpoint diversity"
                return self._mark_session_needs_recapture(
                    session,
                    reason=f"Capture quality insufficient: {reasons}",
                    coverage_score=float(coverage_report.get("coverage_score", 0.0)),
                    extracted_frames=frames,
                )

            return self._persist_session(
                session,
                new_status=AssetStatus.CAPTURED,
                extracted_frames=frames,
                coverage_score=float(coverage_report.get("coverage_score", 0.0)),
                publish_state=None,
                last_pipeline_stage=AssetStatus.CAPTURED.value,
                failure_reason=None,
            )
        except IrrecoverableError:
            raise
        except ValueError as e:
            raise IrrecoverableError(f"Frame extraction failed: {e}")
        except Exception as e:
            raise RecoverableError(f"Frame extraction failed: {e}")

    def _handle_reconstruction(self, session: CaptureSession) -> CaptureSession:
        try:
            from modules.capture_workflow.coverage_analyzer import CoverageAnalyzer
            from modules.reconstruction_engine.job_manager import JobManager
            from modules.reconstruction_engine.runner import ReconstructionRunner
            from modules.shared_contracts.models import ReconstructionJobDraft

            frames_dir = self.session_manager.captures_dir / session.session_id / "frames"
            if not frames_dir.exists():
                session = self._handle_frame_extraction(session)
                if session.status == AssetStatus.RECAPTURE_REQUIRED:
                    return session

            input_frames = [str(f) for f in frames_dir.glob("*.jpg")]
            if not input_frames:
                raise IrrecoverableError(f"No frames available for reconstruction in {session.session_id}")

            coverage_report = CoverageAnalyzer().analyze_coverage(input_frames)
            if coverage_report["overall_status"] != "sufficient":
                reasons = "; ".join(coverage_report.get("reasons", [])) or "insufficient viewpoint diversity"
                return self._mark_session_needs_recapture(
                    session,
                    reason=f"Capture quality insufficient before reconstruction: {reasons}",
                    coverage_score=float(coverage_report.get("coverage_score", 0.0)),
                    extracted_frames=input_frames,
                )

            manager = JobManager(data_root=str(self.data_root))
            runner = ReconstructionRunner()
            job_id = f"job_{session.session_id}"

            draft = ReconstructionJobDraft(
                job_id=job_id,
                capture_session_id=session.session_id,
                input_frames=input_frames,
                product_id=session.product_id,
            )

            job = manager.create_job(draft)
            manager.update_job_status(job.job_id, ReconstructionStatus.RUNNING)
            manifest = runner.run(job)
            manager.update_job_status(job.job_id, ReconstructionStatus.COMPLETED)

            return self._persist_session(
                session,
                new_status=AssetStatus.RECONSTRUCTED,
                reconstruction_job_id=job.job_id,
                reconstruction_manifest_path=str(Path(job.job_dir) / "manifest.json"),
                publish_state=None,
                last_pipeline_stage=AssetStatus.RECONSTRUCTED.value,
                failure_reason=None,
            )
        except IrrecoverableError:
            raise
        except Exception as e:
            msg = str(e)
            if "not configured" in msg or "prohibited" in msg or "VIOLATION" in msg:
                raise IrrecoverableError(f"Engine configuration error: {msg}")
            raise RecoverableError(f"Engine failure: {e}")

    def _handle_cleanup(self, session: CaptureSession) -> CaptureSession:
        manifest = self._load_manifest(session)
        logger.info(f"Starting mesh cleanup for {session.session_id}...")

        try:
            metadata, cleanup_stats, cleaned_mesh_path = self.cleaner.process_cleanup(
                job_id=manifest.job_id,
                raw_mesh_path=manifest.mesh_path,
                profile_type=CleanupProfileType.MOBILE_DEFAULT,
                raw_texture_path=manifest.texture_path,
            )
        except Exception as e:
            raise IrrecoverableError(f"Mesh cleanup failed: {e}")

        metadata_path = Path(cleaned_mesh_path).parent / "normalized_metadata.json"
        cleanup_stats_path = Path(cleaned_mesh_path).parent / "cleanup_stats.json"
        atomic_write_json(metadata_path, metadata.model_dump(mode="json"))
        atomic_write_json(cleanup_stats_path, cleanup_stats)

        return self._persist_session(
            session,
            new_status=AssetStatus.CLEANED,
            cleanup_mesh_path=cleaned_mesh_path,
            cleanup_metadata_path=str(metadata_path),
            cleanup_stats_path=str(cleanup_stats_path),
            last_pipeline_stage=AssetStatus.CLEANED.value,
            failure_reason=None,
        )

    def _handle_export(self, session: CaptureSession) -> CaptureSession:
        metadata, cleanup_stats = self._load_cleanup_artifacts(session)
        manifest = self._load_manifest(session)

        asset_id = session.asset_id or self._build_asset_id(session)
        texture_path = cleanup_stats.get("cleaned_texture_path") or manifest.texture_path
        texture_path_exists = bool(texture_path and Path(texture_path).exists())
        blob_path = self.blobs_dir / f"{asset_id}.glb"

        logger.info(f"Exporting cleaned GLB for {session.session_id}...")
        try:
            self.exporter.export(
                mesh_path=session.cleanup_mesh_path,
                output_path=str(blob_path),
                profile_name="standard",
                texture_path=texture_path if texture_path_exists else None,
                metadata=metadata,
            )
        except Exception as e:
            raise IrrecoverableError(f"GLB Export failed: {e}")

        return self._persist_session(
            session,
            new_status=AssetStatus.EXPORTED,
            asset_id=asset_id,
            export_blob_path=str(blob_path),
            last_pipeline_stage=AssetStatus.EXPORTED.value,
            failure_reason=None,
        )

    def _handle_validation(self, session: CaptureSession) -> CaptureSession:
        metadata, cleanup_stats = self._load_cleanup_artifacts(session)
        manifest = self._load_manifest(session)
        asset_id = session.asset_id or self._build_asset_id(session)

        if not session.export_blob_path:
            raise IrrecoverableError(f"Exported GLB path missing for {session.session_id}")

        try:
            export_metrics = self.exporter.inspect_exported_asset(session.export_blob_path)
        except Exception as e:
            raise IrrecoverableError(f"Exported GLB validation failed: {e}")

        texture_path = cleanup_stats.get("cleaned_texture_path") or manifest.texture_path
        texture_path_exists = bool(texture_path and Path(texture_path).exists())
        has_uv = bool(export_metrics["has_uv"])
        has_material = bool(export_metrics["has_material"])
        has_embedded_texture = bool(export_metrics["has_embedded_texture"])
        texture_status = (
            "complete"
            if has_embedded_texture and has_uv
            else ("degraded" if texture_path_exists or has_material else "missing")
        )

        validation_input = {
            "poly_count": int(export_metrics["face_count"]),
            "texture_status": texture_status,
            "bbox": export_metrics["bbox"],
            "ground_offset": float(export_metrics["ground_offset"]),
            "cleanup_stats": cleanup_stats,
            "texture_path_exists": bool(texture_path_exists or has_embedded_texture),
            "has_uv": has_uv,
            "has_material": has_material,
            "texture_applied_successfully": bool(has_embedded_texture and has_uv and has_material),
            "delivery_geometry_count": int(export_metrics["geometry_count"]),
            "delivery_component_count": int(export_metrics["component_count"]),
        }

        report = self.validator.validate(asset_id, validation_input)
        reports_dir = self.session_manager.get_capture_path(session.session_id) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        validation_report_path = reports_dir / "validation_report.json"
        atomic_write_json(validation_report_path, report.model_dump(mode="json"))

        logger.info(f"Validation Decision for {session.session_id}: {report.final_decision.upper()}")
        return self._persist_session(
            session,
            new_status=AssetStatus.VALIDATED,
            asset_id=asset_id,
            validation_report_path=str(validation_report_path),
            publish_state="pending",
            last_pipeline_stage=AssetStatus.VALIDATED.value,
            failure_reason=None,
        )

    def _handle_publish(self, session: CaptureSession) -> CaptureSession:
        if session.publish_state in {"draft", "published"}:
            return session

        report = self._load_validation_report(session)
        manifest = self._load_manifest(session)

        if report.final_decision == "fail":
            reason = f"Validation Failed: {report.contamination_report}"
            self._mark_session_failed(session.session_id, reason)
            session.failure_reason = reason
            session.publish_state = "failed"
            session.status = AssetStatus.FAILED
            session.last_pipeline_stage = AssetStatus.FAILED.value
            return session

        asset_id = session.asset_id or self._build_asset_id(session)
        metadata = self._build_registry_metadata(session, asset_id, report)

        existing_metadata = self.registry.get_asset(asset_id)
        stored_metadata = existing_metadata or self.registry.register_asset(metadata)
        publish_state, next_status = self._publish_asset(asset_id, session, report, manifest)

        fields = {
            "asset_id": asset_id,
            "asset_version": stored_metadata.version,
            "publish_state": publish_state,
            "last_pipeline_stage": next_status.value if next_status else AssetStatus.VALIDATED.value,
            "failure_reason": None,
        }
        return self._persist_session(session, new_status=next_status, **fields)

    def _publish_asset(
        self,
        asset_id: str,
        session: CaptureSession,
        report: ValidationReport,
        manifest: OutputManifest,
    ) -> Tuple[str, Optional[AssetStatus]]:
        if manifest.is_stub or report.final_decision == "review":
            self.registry.update_publish_state(asset_id, "draft")
            return "draft", None

        self.registry.update_publish_state(asset_id, "published")
        self.registry.set_active_version(session.product_id, asset_id)
        return "published", AssetStatus.PUBLISHED

    def _build_asset_id(self, session: CaptureSession) -> str:
        return validate_identifier(f"asset_{session.session_id}", "Asset ID")

    def _build_registry_metadata(self, session: CaptureSession, asset_id: str, report: ValidationReport) -> AssetMetadata:
        metadata, _ = self._load_cleanup_artifacts(session)
        if not session.export_blob_path:
            raise IrrecoverableError(f"Exported GLB path missing for {session.session_id}")
        export_metrics = self.exporter.inspect_exported_asset(session.export_blob_path)
        return AssetMetadata(
            asset_id=asset_id,
            product_id=session.product_id,
            version=None,
            bbox={
                "min": export_metrics["bounds_min"],
                "max": export_metrics["bounds_max"],
                "dimensions": export_metrics["bbox"],
            },
            pivot_offset=metadata.pivot_offset,
            quality_grade=report.mobile_performance_grade,
        )

    def _load_manifest(self, session: CaptureSession) -> OutputManifest:
        manifest_path = session.reconstruction_manifest_path
        if not manifest_path and session.reconstruction_job_id:
            manifest_path = str(self.data_root / "reconstructions" / session.reconstruction_job_id / "manifest.json")
        if not manifest_path:
            manifest_path = str(self.data_root / "reconstructions" / f"job_{session.session_id}" / "manifest.json")

        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise IrrecoverableError(f"Reconstruction manifest missing for {session.session_id} at {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            return OutputManifest.model_validate(json.load(f))

    def _load_cleanup_artifacts(self, session: CaptureSession) -> Tuple[NormalizedMetadata, Dict[str, Any]]:
        if not session.cleanup_metadata_path:
            raise IrrecoverableError(f"Cleanup metadata missing for {session.session_id}")
        if not session.cleanup_stats_path:
            raise IrrecoverableError(f"Cleanup stats missing for {session.session_id}")

        metadata_path = Path(session.cleanup_metadata_path)
        stats_path = Path(session.cleanup_stats_path)
        if not metadata_path.exists():
            raise IrrecoverableError(f"Cleanup metadata artifact missing: {metadata_path}")
        if not stats_path.exists():
            raise IrrecoverableError(f"Cleanup stats artifact missing: {stats_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = NormalizedMetadata.model_validate(json.load(f))
        with open(stats_path, "r", encoding="utf-8") as f:
            cleanup_stats = json.load(f)

        return metadata, cleanup_stats

    def _load_validation_report(self, session: CaptureSession) -> ValidationReport:
        if not session.validation_report_path:
            raise IrrecoverableError(f"Validation report missing for {session.session_id}")

        report_path = Path(session.validation_report_path)
        if not report_path.exists():
            raise IrrecoverableError(f"Validation report artifact missing: {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            return ValidationReport.model_validate(json.load(f))

    def _finalize_ingestion(self, session: CaptureSession) -> CaptureSession:
        """
        Compatibility wrapper for existing tools/tests.
        Executes the remaining phase handlers sequentially starting from the
        current session status.
        """
        current = self.session_manager.get_session(session.session_id) or session

        if current.status == AssetStatus.RECONSTRUCTED:
            current = self._handle_cleanup(current)
        if current.status == AssetStatus.CLEANED:
            current = self._handle_export(current)
        if current.status == AssetStatus.EXPORTED:
            current = self._handle_validation(current)
        if current.status == AssetStatus.VALIDATED and current.publish_state not in {"draft", "published"}:
            current = self._handle_publish(current)

        return current


worker_instance = IngestionWorker()
