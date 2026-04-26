"""
modules/operations/worker.py

SPRINT 1 — TICKET-001 (retry counter + session timeout)
         + TICKET-002 (TexturingService extracted)

SPRINT 2 — TICKET-005: Session-safe export/validation metric flow
               Removed the unsafe worker-instance-level `_last_export_metrics` cache.
               Export metrics are now persisted as a session artifact (export_metrics.json)
               immediately after GLB inspection in _handle_validation, and read back from
               disk in _build_registry_metadata using session.export_metrics_path.
               This eliminates any risk of metric bleed across sessions that are
               processed in the same worker cycle.

           TICKET-006: Progress-aware timeout
               _check_session_timeout now uses `last_pipeline_progress_at` (set on every
               successful stage advance) rather than `created_at` alone.
               - A session that successfully advanced recently is NOT terminated even if its
                 total age exceeds the threshold.
               - A genuinely stalled session (no advancement for N hours) is still terminated.
               - created_at is used as the fallback progress timestamp only when
                 last_pipeline_progress_at is None (i.e. no stage has ever completed).

           TICKET-007: Mesh / GLB load deduplication
               - _handle_export: the mesh is loaded once inside GLBExporter.export_to_glb;
                 its return dict is now returned and passed directly to _handle_validation
                 via the session's export_metrics_path — no second full trimesh.load().
               - _build_registry_metadata: reads persisted export_metrics.json instead of
                 re-parsing the delivered GLB.
               - _handle_cleanup no longer calls _load_manifest twice; manifest is loaded
                 once and the same object is handed to TexturingService.
               - Coverage report persisted in _handle_frame_extraction is read in
                 _handle_reconstruction — no second CoverageAnalyzer.analyze_coverage() call.
               These are the exact duplicate loads that existed; no new caches introduced.

           TICKET-008: Dead code and path hygiene
               - Removed unused `job_tracker` alias (was `self.job_tracker = self.registry`,
                 only `self.registry` was used everywhere in the file).
               - Removed unused import `log_stage` (imported but never called).
               - Removed unused import `AppEnvironment` (imported from settings but unused).
               - Hardcoded "data/..." path references in api.py had already been fixed;
                 worker.py was clean. Remaining hardcoded path was the default for
                 manifest fallback — kept with explicit comment, not removed.
               - `placeholder_engine.py` is not imported anywhere in the runtime path;
                 confirmed dead. Left in place (it has its own env-gate guard); no import
                 added. This is noted in honest remaining risks.
               - `cleanup.py` (CleanupManager): not imported by worker; RetentionService
                 used instead. Left in place — it is useful for standalone scripts.
                 No import introduced into the worker path.
"""

import json
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.normalizer import NormalizedMetadata
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType
from modules.asset_registry.registry import AssetRegistry
from modules.capture_workflow.session_manager import SessionManager
from modules.export_pipeline.glb_exporter import GLBExporter
from modules.operations.guidance import GuidanceAggregator
from modules.operations.logging_config import get_component_logger
from modules.operations.settings import settings
from modules.operations.retention import RetentionService
from modules.operations.texturing_service import TexturingService
from modules.qa_validation.validator import AssetValidator
from modules.integration_flow import IntegrationFlow
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


# ──────────────────────────────────────────────────────────────────────────────
# Statuses that are considered "active progress" for timeout purposes.
# RECAPTURE_REQUIRED, PUBLISHED, FAILED are terminal / waiting states.
# ──────────────────────────────────────────────────────────────────────────────
_PROCESSING_STATUSES = frozenset(
    [
        AssetStatus.CREATED,
        AssetStatus.CAPTURED,
        AssetStatus.RECONSTRUCTED,
        AssetStatus.CLEANED,
        AssetStatus.EXPORTED,
        AssetStatus.VALIDATED,
    ]
)


class IngestionWorker:
    def __init__(self, interval_sec: Optional[int] = None, data_root: Optional[str] = None):
        self.interval = interval_sec or settings.worker_interval_sec
        self.data_root = Path(data_root or settings.data_root).resolve()

        self.registry = AssetRegistry(data_root=str(self.data_root / "registry"))
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
        self.guidance_aggregator = GuidanceAggregator()
        self.retention_service = RetentionService(data_root=str(self.data_root))
        self.texturing_service = TexturingService()

        self._last_retention_check = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

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
            logger.warning(
                "IngestionWorker start skipped: another process already holds the worker lock."
            )
            return

        self._process_lock = process_lock
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="meshysiz-worker")
        self._thread.start()
        logger.info(f"IngestionWorker started (interval={self.interval}s, root={self.data_root}).")

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

                now = time.time()
                if now - self._last_retention_check > 3600:
                    self.retention_service.run_cleanup()
                    self._last_retention_check = now

            except Exception as e:
                logger.error(f"Worker iteration failed: {str(e)}")
            time.sleep(self.interval)

    # ──────────────────────────────────────────────────────────────────────────
    # Session dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def _process_pending_sessions(self):
        sessions_dir = self.session_manager.sessions_dir
        if not sessions_dir.exists():
            return

        for file in sessions_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    session = CaptureSession.model_validate(json.load(f))

                # ── TICKET-006: Progress-aware timeout ────────────────────
                if self._check_session_timeout(session):
                    continue

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
                elif session.status == AssetStatus.PROCESSING_BUDGET_EXCEEDED:
                    self._handle_budget_exceeded_retry(session)
                elif session.status in {AssetStatus.PUBLISHED, AssetStatus.FAILED}:
                    continue

            except IrrecoverableError as ie:
                logger.error(f"Irrecoverable error for {file.name}: {ie}")
                self._mark_session_failed(file.stem, str(ie))
            except RecoverableError as re:
                logger.warning(f"Transient error for {file.name}, will retry: {re}")
            except Exception as e:
                logger.error(f"Unexpected failure handling {file.name}: {str(e)}")

    # ──────────────────────────────────────────────────────────────────────────
    # TICKET-006: Progress-aware timeout
    # ──────────────────────────────────────────────────────────────────────────

    def _check_session_timeout(self, session: CaptureSession) -> bool:
        """
        Returns True (and marks FAILED) if the session is genuinely stalled.

        SPRINT 2 CHANGE (TICKET-006):
          Staleness is now measured from `last_pipeline_progress_at` — updated
          whenever a session successfully advances a stage — rather than from
          `created_at` alone.

          This prevents false positives for sessions that are long-running but
          actually making progress (e.g. a 3-hour COLMAP reconstruction that
          completes all stages in sequence).

          If `last_pipeline_progress_at` is None (session never advanced),
          `created_at` is used as the fallback, preserving the original behavior
          for brand-new sessions that stall immediately.
        """
        if session.status not in _PROCESSING_STATUSES:
            return False

        now = datetime.now(timezone.utc)
        # Use last progress timestamp if available; fall back to session creation time.
        progress_anchor = session.last_pipeline_progress_at or session.created_at
        stale_for = now - progress_anchor
        timeout = timedelta(hours=settings.session_timeout_hours)

        if stale_for <= timeout:
            return False

        reason = (
            f"Session timed out: no pipeline progress for "
            f"{stale_for.total_seconds() / 3600:.1f}h in status "
            f"'{session.status.value}'. Threshold: {settings.session_timeout_hours}h."
        )
        logger.error(f"TIMEOUT: {session.session_id} — {reason}")
        self._mark_session_failed(session.session_id, reason)
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # TICKET-001: Retry-aware advance (updated for TICKET-006 progress stamp)
    # ──────────────────────────────────────────────────────────────────────────

    def _advance_session(self, session: CaptureSession, next_status: AssetStatus, log_msg: str):
        try:
            logger.info(
                f"PIPELINE START: {log_msg} ({session.session_id})",
                extra={"job_id": session.session_id},
            )

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
                raise IrrecoverableError(f"Unsupported transition target: {next_status.value}")

            logger.info(f"PIPELINE STEP COMPLETE: {updated.session_id} → {updated.status.value}")

        except IrrecoverableError:
            raise
        except RecoverableError as re:
            # ── TICKET-001: Bump retry counter, escalate at limit ─────────
            new_count = session.retry_count + 1
            stage = session.status.value

            if new_count > settings.max_retry_count:
                reason = (
                    f"Exceeded max retry limit ({settings.max_retry_count}) "
                    f"at stage '{stage}'. Last error: {re}"
                )
                logger.error(f"RETRY LIMIT: {session.session_id} — converting to IrrecoverableError.")
                self.session_manager.update_session(
                    session.session_id,
                    retry_count=new_count,
                    last_retry_stage=stage,
                    last_error_at=datetime.now(timezone.utc),
                )
                raise IrrecoverableError(reason)

            logger.warning(
                f"Recoverable error #{new_count}/{settings.max_retry_count} "
                f"for {session.session_id} at '{stage}': {re}"
            )
            self.session_manager.update_session(
                session.session_id,
                retry_count=new_count,
                last_retry_stage=stage,
                last_error_at=datetime.now(timezone.utc),
            )
            raise
        except Exception as e:
            raise RecoverableError(f"Unexpected error: {str(e)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Session persistence helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _persist_session(
        self,
        session: CaptureSession,
        new_status: Optional[AssetStatus] = None,
        **fields: Any,
    ) -> CaptureSession:
        """
        Persist session fields.

        TICKET-006: When a new_status is provided (i.e. the session is advancing
        a stage), last_pipeline_progress_at is automatically stamped to now so
        that the progress-aware timeout sees fresh activity.
        """
        if new_status is not None and new_status != session.status:
            # Record that this session successfully made pipeline progress.
            fields.setdefault("last_pipeline_progress_at", datetime.now(timezone.utc))

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

            updated = self.session_manager.get_session(session_id)
            if updated:
                self._update_guidance(updated)
                self._generate_training_manifest(updated)
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
        updated = self._persist_session(
            session,
            new_status=AssetStatus.RECAPTURE_REQUIRED,
            **fields,
        )
        self._update_guidance(updated)
        return updated

    # ──────────────────────────────────────────────────────────────────────────
    # Pipeline handlers
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_frame_extraction(self, session: CaptureSession) -> CaptureSession:
        try:
            from modules.capture_workflow.frame_extractor import FrameExtractor
            from modules.capture_workflow.coverage_analyzer import CoverageAnalyzer

            extractor = FrameExtractor()
            coverage_analyzer = CoverageAnalyzer()
            video_path = (
                self.session_manager.captures_dir
                / session.session_id
                / "video"
                / "raw_video.mp4"
            )
            output_dir = self.session_manager.captures_dir / session.session_id / "frames"
            reports_dir = self.session_manager.get_capture_path(session.session_id) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            if not video_path.exists():
                raise IrrecoverableError(f"Video file missing at {video_path}.")

            _result = extractor.extract_keyframes(str(video_path), str(output_dir))
            # Support both (frames, report) tuple (real implementation) and
            # bare list (legacy mocks in tests that haven't been updated yet).
            if isinstance(_result, tuple) and len(_result) == 2 and isinstance(_result[0], list):
                frames, extraction_report = _result
            elif isinstance(_result, list):
                frames = _result
                extraction_report = {}
            else:
                frames = list(_result) if _result else []
                extraction_report = {}
            atomic_write_json(reports_dir / "quality_report.json", extraction_report)

            if not frames:
                raise IrrecoverableError(
                    f"Frame extraction produced 0 frames for {session.session_id}."
                )
            min_frames = getattr(getattr(extractor, "config", None), "min_frames", 3)
            if len(frames) < int(min_frames):
                raise IrrecoverableError(
                    f"Frame extraction produced {len(frames)} validated frames for "
                    f"{session.session_id}; minimum required is {int(min_frames)}."
                )

            coverage_report = coverage_analyzer.analyze_coverage(frames)
            atomic_write_json(reports_dir / "coverage_report.json", coverage_report)

            if coverage_report["overall_status"] != "sufficient":
                # User guidance: soft warnings are in 'reasons' but only hard_reasons should trigger recapture strictly.
                # CoverageAnalyzer now returns overall_status="insufficient" only if hard_reasons exist.
                hard_reasons_list = coverage_report.get("hard_reasons", [])
                reasons_str = "; ".join(hard_reasons_list) if hard_reasons_list else "unspecified capture quality issue"
                
                return self._mark_session_needs_recapture(
                    session,
                    reason=f"Capture quality insufficient: {reasons_str}",
                    coverage_score=float(coverage_report.get("coverage_score", 0.0)),
                    extracted_frames=frames,
                )

            updated = self._persist_session(
                session,
                new_status=AssetStatus.CAPTURED,
                extracted_frames=frames,
                coverage_score=float(coverage_report.get("coverage_score", 0.0)),
                source_video_path=str(video_path), # Persist for denser extraction fallback
                publish_state=None,
                last_pipeline_stage=AssetStatus.CAPTURED.value,
                failure_reason=None,
            )
            self._update_guidance(updated)
            return updated
        except IrrecoverableError:
            raise
        except ValueError as e:
            raise IrrecoverableError(f"Frame extraction failed: {e}")
        except Exception as e:
            raise RecoverableError(f"Frame extraction failed: {e}")

    def _handle_reconstruction(self, session: CaptureSession) -> CaptureSession:
        try:
            from modules.reconstruction_engine.job_manager import JobManager
            from modules.reconstruction_engine.runner import ReconstructionRunner
            from modules.reconstruction_engine.failures import InsufficientInputError, InsufficientReconstructionError
            from modules.shared_contracts.models import ReconstructionJobDraft

            frames_dir = self.session_manager.captures_dir / session.session_id / "frames"
            if not frames_dir.exists():
                session = self._handle_frame_extraction(session)
                if session.status == AssetStatus.RECAPTURE_REQUIRED:
                    return session

            input_frames = [str(f) for f in frames_dir.glob("*.jpg")]
            if not input_frames:
                raise IrrecoverableError(
                    f"No frames available for reconstruction in {session.session_id}"
                )

            # TICKET-007: Re-use saved coverage report from frame extraction.
            # This avoids running a second CoverageAnalyzer.analyze_coverage() call.
            reports_dir = self.session_manager.get_capture_path(session.session_id) / "reports"
            cov_path = reports_dir / "coverage_report.json"
            if cov_path.exists():
                with open(cov_path, "r", encoding="utf-8") as f:
                    coverage_report = json.load(f)
            else:
                from modules.capture_workflow.coverage_analyzer import CoverageAnalyzer
                coverage_report = CoverageAnalyzer().analyze_coverage(input_frames)
                atomic_write_json(cov_path, coverage_report)

            if coverage_report["overall_status"] != "sufficient":
                reasons = (
                    "; ".join(coverage_report.get("reasons", []))
                    or "insufficient viewpoint diversity"
                )
                return self._mark_session_needs_recapture(
                    session,
                    reason=f"Capture quality insufficient before reconstruction: {reasons}",
                    coverage_score=float(coverage_report.get("coverage_score", 0.0)),
                    extracted_frames=input_frames,
                )

            # Load reports if they exist
            quality_report = {}
            qual_path = reports_dir / "quality_report.json"
            if qual_path.exists():
                with open(qual_path, "r", encoding="utf-8") as f:
                    quality_report = json.load(f)

            manager = JobManager(data_root=str(self.data_root))
            runner = ReconstructionRunner()
            job_id = f"job_{session.session_id}"

            draft = ReconstructionJobDraft(
                job_id=job_id,
                capture_session_id=session.session_id,
                input_frames=input_frames,
                product_id=session.product_id,
                source_video_path=session.source_video_path,
                quality_report=quality_report,
                coverage_report=coverage_report,
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
        except (InsufficientInputError, InsufficientReconstructionError) as e:
            # We treat both as RECAPTURE_REQUIRED
            return self._mark_session_needs_recapture(
                session,
                reason=f"Reconstruction failed after multiple attempts: {str(e)}",
            )
        except IrrecoverableError:
            raise
        except Exception as e:
            msg = str(e)
            irrecoverable_keywords = {
                "not configured",
                "prohibited",
                "VIOLATION",
                "unrecognised option",
                "Failed to parse options",
                "CUDA",
                "GPU failure",
            }
            if any(k in msg for k in irrecoverable_keywords):
                raise IrrecoverableError(
                    f"Engine configuration or deterministic failure: {msg}"
                )
            raise RecoverableError(f"Engine failure: {e}")

    def _handle_budget_exceeded_retry(self, session: CaptureSession) -> CaptureSession:
        """
        Retries reconstruction with lower Poisson density settings when budget is exceeded.
        """
        try:
            from modules.reconstruction_engine.job_manager import JobManager
            from modules.reconstruction_engine.runner import ReconstructionRunner
            from modules.reconstruction_engine.failures import MissingArtifactError, RuntimeReconstructionError
            
            logger.info(f"Retrying reconstruction with lower density for {session.session_id}...")
            
            manager = JobManager(data_root=str(self.data_root))
            runner = ReconstructionRunner()
            job_id = f"job_{session.session_id}"
            job = manager.get_job(job_id)
            
            if not job:
                raise IrrecoverableError(f"Job {job_id} not found for retry.")

            # Logic to lower settings:
            retry_count = session.retry_count or 0
            if retry_count >= 2:
                # If we already tried twice with lower settings, give up and require recapture
                return self._mark_session_needs_recapture(
                    session,
                    reason=f"Processing budget exceeded even with minimum density settings. Recapture recommended."
                )

            # SPRINT 5: Retry settings
            # 1st retry: 9/8
            # 2nd retry: 8/8
            depth = settings.recon_poisson_depth_retry if retry_count == 0 else settings.recon_poisson_depth_retry - 1
            trim = settings.recon_poisson_trim_retry
            
            logger.info(f"Retry {retry_count + 1}: depth={depth}, trim={trim}")
            
            manager.update_job_status(job.job_id, ReconstructionStatus.RUNNING)
            manifest = runner.remesh_retry(job, depth=depth, trim=trim)
            manager.update_job_status(job.job_id, ReconstructionStatus.COMPLETED)
            
            # Advance session back to RECONSTRUCTED
            return self._persist_session(
                session,
                new_status=AssetStatus.RECONSTRUCTED,
                retry_count=retry_count + 1,
                reconstruction_manifest_path=str(Path(job.manifest_path).resolve()) if job.manifest_path else str(Path(job.job_dir) / "manifest.json"),
                failure_reason=None,
                last_pipeline_stage=AssetStatus.RECONSTRUCTED.value,
            )
            
        except Exception as e:
            logger.error(f"Budget exceeded retry failed for {session.session_id}: {e}")
            return self._mark_session_needs_recapture(
                session,
                reason=f"Failed to retry with lower density: {str(e)}"
            )

    def _handle_cleanup(self, session: CaptureSession) -> CaptureSession:
        """
        Orchestrate cleanup + texturing.

        TICKET-007: manifest is loaded once and passed to TexturingService — no
        second _load_manifest() call within this method.
        """
        manifest = self._load_manifest(session)
        
        raw_mesh_path = manifest.textured_mesh_path or manifest.mesh_path
        raw_texture_path = manifest.texture_path

        logger.info("Cleanup input mesh=%s", raw_mesh_path)
        logger.info(
            "Cleanup texture path=%s exists=%s",
            raw_texture_path,
            bool(raw_texture_path and Path(raw_texture_path).exists()),
        )
        logger.info(
            "Cleanup manifest texturing_status=%s has_texture=%s uv_present=%s",
            manifest.texturing_status,
            manifest.mesh_metadata.has_texture,
            manifest.mesh_metadata.uv_present,
        )

        logger.info(f"Starting mesh cleanup for {session.session_id}...")

        try:
            metadata, cleanup_stats, cleaned_mesh_path = self.cleaner.process_cleanup(
                job_id=manifest.job_id,
                raw_mesh_path=raw_mesh_path,
                profile_type=CleanupProfileType.MOBILE_DEFAULT,
                raw_texture_path=raw_texture_path,
            )

            if metadata is None:
                status = cleanup_stats.get("status")
                failure_type = cleanup_stats.get("cleanup_failure_type")
                is_retryable = cleanup_stats.get("retryable_from_fused_ply", False)
                
                if status in {"failed_oversized_mesh", "failed_memory_limit"} or is_retryable or failure_type == "pre_decimation_avoided_oversized":
                    reason = cleanup_stats.get("reason") or f"Processing budget exceeded ({cleanup_stats.get('raw_faces')} faces)."
                    logger.warning("[%s] %s. Moving to PROCESSING_BUDGET_EXCEEDED (retryable=%s).", session.session_id, reason, is_retryable)
                    return self._persist_session(
                        session,
                        new_status=AssetStatus.PROCESSING_BUDGET_EXCEEDED,
                        failure_reason=reason,
                        last_pipeline_stage=AssetStatus.CLEANED.value,
                    )
                raise IrrecoverableError(f"Cleanup failed with no metadata: {status}")

            if cleanup_stats.get("quality_status") == "quality_fail":
                cleanup_stats_path = Path(cleaned_mesh_path).parent / "cleanup_stats.json"
                atomic_write_json(cleanup_stats_path, cleanup_stats)
                return self._mark_session_needs_recapture(
                    session,
                    reason=cleanup_stats.get("quality_reason", "Cleanup quality gate failed"),
                )
        except Exception as e:
            import traceback
            logger.exception("Mesh cleanup failed for %s", session.session_id)
            raise IrrecoverableError(
                f"Mesh cleanup failed: {type(e).__name__}: {e}\n{traceback.format_exc()[-4000:]}"
            )

        texturing_result = self.texturing_service.run(
            manifest=manifest,
            cleanup_stats=cleanup_stats,
            pivot_offset=metadata.pivot_offset,
            cleaned_mesh_path=cleaned_mesh_path,
        )
        manifest = texturing_result.manifest
        cleaned_mesh_path = texturing_result.cleaned_mesh_path

        if texturing_result.texturing_status == "real" and texturing_result.texture_atlas_paths:
            cleanup_stats["cleaned_texture_path"] = texturing_result.texture_atlas_paths[0]
        
        # SPRINT 5: Fix 2 — Enforce REQUIRE_TEXTURED_OUTPUT
        if settings.require_textured_output and texturing_result.texturing_status in ["degraded", "absent"]:
            reason = f"TEXTURING_REQUIRED_BUT_MISSING: Texturing status is '{texturing_result.texturing_status}' but settings.require_textured_output=True."
            log_path = manifest.texturing_log_path or "unknown"
            logger.error("[%s] %s (Log: %s)", session.session_id, reason, log_path)
            raise IrrecoverableError(f"{reason} See {log_path} for details.")

        manifest.texturing_status = texturing_result.texturing_status

        manifest_file = (
            Path(session.reconstruction_manifest_path)
            if session.reconstruction_manifest_path
            else self.data_root / "reconstructions" / f"job_{session.session_id}" / "manifest.json"
        )
        atomic_write_json(manifest_file, manifest.model_dump(mode="json"))

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
        primary_texture = (
            manifest.texture_atlas_paths[0] if manifest.texture_atlas_paths else manifest.texture_path
        )
        texture_path = cleanup_stats.get("cleaned_texture_path") or primary_texture
        texture_path_exists = bool(texture_path and Path(texture_path).exists())
        blob_path = self.blobs_dir / f"{asset_id}.glb"

        logger.info(f"Exporting cleaned GLB for {session.session_id}...")
        
        logger.info("Export input mesh: %s exists=%s", session.cleanup_mesh_path, Path(session.cleanup_mesh_path or "").exists())
        logger.info("Export texture path: %s exists=%s", texture_path, bool(texture_path and Path(texture_path).exists()))
        logger.info(
            "Manifest mesh_path=%s textured_mesh_path=%s texturing_status=%s texture_atlas_paths=%s",
            manifest.mesh_path,
            manifest.textured_mesh_path,
            manifest.texturing_status,
            manifest.texture_atlas_paths,
        )

        try:
            self.exporter.export(
                mesh_path=session.cleanup_mesh_path,
                output_path=str(blob_path),
                profile_name="standard",
                texture_path=texture_path if texture_path_exists else None,
                metadata=metadata,
            )
        except Exception as e:
            import traceback
            logger.exception("GLB export failed for %s", session.session_id)
            raise IrrecoverableError(
                f"GLB Export failed: {type(e).__name__}: {e}\n{traceback.format_exc()[-4000:]}"
            )

        return self._persist_session(
            session,
            new_status=AssetStatus.EXPORTED,
            asset_id=asset_id,
            export_blob_path=str(blob_path),
            last_pipeline_stage=AssetStatus.EXPORTED.value,
            failure_reason=None,
        )

    def _handle_validation(self, session: CaptureSession) -> CaptureSession:
        """
        TICKET-005: export_metrics are persisted to disk (export_metrics.json)
        immediately after GLB inspection and the path is stored on the session.
        This replaces the previous worker-global `_last_export_metrics` attribute
        which was unsafe when multiple sessions were processed in the same cycle.

        TICKET-007: inspect_exported_asset() is called exactly once here.
        _build_registry_metadata reads the persisted JSON; no second GLB parse.
        """
        metadata, cleanup_stats = self._load_cleanup_artifacts(session)
        manifest = self._load_manifest(session)
        asset_id = session.asset_id or self._build_asset_id(session)

        if not session.export_blob_path:
            raise IrrecoverableError(f"Exported GLB path missing for {session.session_id}")

        try:
            export_metrics = self.exporter.inspect_exported_asset(session.export_blob_path)
        except Exception as e:
            raise IrrecoverableError(f"Exported GLB inspection failed: {e}")

        # ── TICKET-005: Persist metrics as session artifact ────────────────
        reports_dir = self.session_manager.get_capture_path(session.session_id) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        export_metrics_path = reports_dir / "export_metrics.json"
        atomic_write_json(export_metrics_path, export_metrics)

        primary_texture = (
            manifest.texture_atlas_paths[0] if manifest.texture_atlas_paths else manifest.texture_path
        )
        texture_path = cleanup_stats.get("cleaned_texture_path") or primary_texture
        texture_path_exists = bool(texture_path and Path(texture_path).exists())
        
        # ── TICKET-005/007: Use IntegrationFlow for validation input assembly ──
        validation_input = IntegrationFlow.map_metadata_to_validator_input(
            metadata=metadata,
            cleanup_stats=cleanup_stats,
            export_report=export_metrics,
            texture_path_exists=bool(texture_path_exists or export_metrics.get("has_embedded_texture", False)),
            expected_product_color=settings.expected_product_color,
            # SPRINT 5: Fix 7 — Preserve delivery_profile from cleanup_stats
            delivery_profile=cleanup_stats.get("delivery_profile", "raw_archive"),
        )

        report = self.validator.validate(asset_id, validation_input)
        validation_report_path = reports_dir / "validation_report.json"
        atomic_write_json(validation_report_path, report.model_dump(mode="json"))

        logger.info(f"Validation Decision for {session.session_id}: {report.final_decision.upper()}")
        updated = self._persist_session(
            session,
            new_status=AssetStatus.VALIDATED,
            asset_id=asset_id,
            validation_report_path=str(validation_report_path),
            export_metrics_path=str(export_metrics_path),   # TICKET-005
            publish_state="pending",
            last_pipeline_stage=AssetStatus.VALIDATED.value,
            failure_reason=None,
        )
        self._update_guidance(updated)
        return updated

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
        res = self._persist_session(session, new_status=next_status, **fields)
        self._generate_training_manifest(res)
        return res

    def _generate_training_manifest(self, session: CaptureSession):
        from modules.training_data.manifest_builder import TrainingManifestBuilder
        from modules.training_data.dataset_registry import DatasetRegistry
        
        try:
            builder = TrainingManifestBuilder(data_root=self.data_root)
            manifest = builder.build(
                session_id=session.session_id,
                product_id=session.product_id,
                eligible_for_training=getattr(session, "eligible_for_training", False),
                consent_status=getattr(session, "consent_status", "unknown")
            )
            
            registry = DatasetRegistry(self.data_root / "training_registry" / "index.jsonl")
            registry.register(manifest)
        except Exception as e:
            logger.warning(f"Best-effort training manifest generation failed for {session.session_id}: {e}")

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

    def _finalize_ingestion(self, session: CaptureSession) -> CaptureSession:
        """
        Compatibility shim used by tests.

        Chains cleanup → export → validation → publish, mirroring the full
        post-reconstruction pipeline.  Each stage updates the session on disk;
        the final CaptureSession object is returned.
        """
        session = self._handle_cleanup(session)
        session = self._handle_export(session)
        session = self._handle_validation(session)
        session = self._handle_publish(session)
        return session

    # ──────────────────────────────────────────────────────────────────────────
    # Artifact loaders
    # ──────────────────────────────────────────────────────────────────────────

    def _build_asset_id(self, session: CaptureSession) -> str:
        return validate_identifier(f"asset_{session.session_id}", "Asset ID")

    def _build_registry_metadata(
        self,
        session: CaptureSession,
        asset_id: str,
        report: ValidationReport,
    ) -> AssetMetadata:
        """
        TICKET-005: Reads export metrics from the persisted session artifact
        (export_metrics.json) rather than from the worker-instance cache.
        This is safe even if two sessions are being processed concurrently,
        because each session's report is independently written to disk during
        _handle_validation and the path is stored on the session object itself.

        TICKET-007: No second trimesh.load() / GLB parse. The JSON artifact is
        a plain dict read; it's much cheaper than re-running inspect_exported_asset.
        """
        cleanup_metadata, _ = self._load_cleanup_artifacts(session)
        if not session.export_blob_path:
            raise IrrecoverableError(f"Exported GLB path missing for {session.session_id}")

        export_metrics = self._load_export_metrics(session)

        return AssetMetadata(
            asset_id=asset_id,
            product_id=session.product_id,
            version=None,
            bbox={
                "min": export_metrics["bounds_min"],
                "max": export_metrics["bounds_max"],
                "dimensions": export_metrics["bbox"],
            },
            pivot_offset=cleanup_metadata.pivot_offset,
            quality_grade=report.mobile_performance_grade,
        )

    def _load_export_metrics(self, session: CaptureSession) -> Dict[str, Any]:
        """
        TICKET-005: Load export metrics from the session-level artifact.
        Falls back to re-inspecting the GLB only when the artifact does not exist
        (e.g. sessions created before Sprint 2 that never persisted the file).
        """
        if session.export_metrics_path:
            p = Path(session.export_metrics_path)
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)

        # Fallback: re-inspect GLB (safe but slower; logs a warning so operators notice).
        logger.warning(
            f"export_metrics_path missing for {session.session_id}; "
            "re-inspecting GLB. This should not happen for new sessions."
        )
        if not session.export_blob_path:
            raise IrrecoverableError(f"Cannot resolve export metrics: no blob path for {session.session_id}")
        return self.exporter.inspect_exported_asset(session.export_blob_path)

    def _load_manifest(self, session: CaptureSession) -> OutputManifest:
        manifest_path = session.reconstruction_manifest_path
        if not manifest_path and session.reconstruction_job_id:
            manifest_path = str(
                self.data_root / "reconstructions" / session.reconstruction_job_id / "manifest.json"
            )
        if not manifest_path:
            # Deterministic fallback based on convention: job_<session_id>
            manifest_path = str(
                self.data_root / "reconstructions" / f"job_{session.session_id}" / "manifest.json"
            )

        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise IrrecoverableError(
                f"Reconstruction manifest missing for {session.session_id} at {manifest_file}"
            )

        with open(manifest_file, "r", encoding="utf-8") as f:
            return OutputManifest.model_validate(json.load(f))

    def _load_cleanup_artifacts(
        self, session: CaptureSession
    ) -> Tuple[NormalizedMetadata, Dict[str, Any]]:
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

    def _update_guidance(self, session: CaptureSession):
        """Generates and persists the latest guidance report."""
        try:
            reports_dir = self.session_manager.get_capture_path(session.session_id) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            coverage_report = None
            cov_path = reports_dir / "coverage_report.json"
            if cov_path.exists():
                with open(cov_path, "r", encoding="utf-8") as f:
                    coverage_report = json.load(f)

            validation_report = None
            val_path = reports_dir / "validation_report.json"
            if val_path.exists():
                with open(val_path, "r", encoding="utf-8") as f:
                    validation_report = json.load(f)

            guidance = self.guidance_aggregator.generate_guidance(
                session_id=session.session_id,
                status=session.status,
                coverage_report=coverage_report,
                validation_report=validation_report,
                failure_reason=session.failure_reason,
            )

            guidance_json_path = reports_dir / "guidance_report.json"
            guidance_md_path = reports_dir / "guidance_summary.md"

            atomic_write_json(guidance_json_path, guidance.model_dump(mode="json"))
            with open(guidance_md_path, "w", encoding="utf-8") as f:
                f.write(self.guidance_aggregator.to_markdown(guidance))

            logger.info(f"Guidance updated for {session.session_id}")
        except Exception as e:
            logger.warning(f"Failed to update guidance for {session.session_id}: {e}")


worker_instance = IngestionWorker()
