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
from modules.operations.guidance import GuidanceAggregator
from modules.operations.logging_config import get_component_logger, log_stage
from modules.operations.settings import settings, AppEnvironment
from modules.operations.retention import RetentionService
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
        self.job_tracker = self.registry
        self.guidance_aggregator = GuidanceAggregator()
        self.retention_service = RetentionService(data_root=str(self.data_root))
        
        self._last_retention_check = 0.0

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
                
                # Retention check every hour
                now = time.time()
                if now - self._last_retention_check > 3600:
                    self._handle_retention()
                    self._last_retention_check = now
                    
            except Exception as e:
                logger.error(f"Worker iteration failed: {str(e)}")
            time.sleep(self.interval)

    def _handle_retention(self):
        """Invoke the retention service logic."""
        self.retention_service.run_cleanup()

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
            
            # Refresh guidance on failure
            updated = self.session_manager.get_session(session_id)
            if updated:
                self._update_guidance(updated)
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

            updated = self._persist_session(
                session,
                new_status=AssetStatus.CAPTURED,
                extracted_frames=frames,
                coverage_score=float(coverage_report.get("coverage_score", 0.0)),
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
            from modules.capture_workflow.coverage_analyzer import CoverageAnalyzer
            from modules.reconstruction_engine.job_manager import JobManager
            from modules.reconstruction_engine.runner import ReconstructionRunner
            from modules.reconstruction_engine.failures import InsufficientInputError
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
        except InsufficientInputError as e:
            return self._mark_session_needs_recapture(
                session,
                reason=f"Reconstruction failed due to insufficient masked input: {str(e)}"
            )
        except IrrecoverableError:
            raise
        except Exception as e:
            msg = str(e)
            # Gate deterministic failures (Config, Security, or CLI errors)
            irrerecoverable_keywords = {
                "not configured", "prohibited", "VIOLATION", 
                "unrecognised option", "Failed to parse options",
                "CUDA", "GPU failure"
            }
            if any(k in msg for k in irrerecoverable_keywords):
                raise IrrecoverableError(f"Engine configuration or deterministic failure: {msg}")
            
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
            
        # PHASE 2.2A TEXTURING: Execute on the cleaned final geometry BEFORE it was aligned
        texturing_status = "absent"
        if manifest.engine_type == "colmap":
            mesh_parent = Path(manifest.mesh_path).parent
            colmap_dir = mesh_parent.parent if mesh_parent.name == "dense" else mesh_parent
            
            if colmap_dir.joinpath("dense").exists():
                from modules.reconstruction_engine.openmvs_texturer import OpenMVSTexturer
                from modules.utils.file_persistence import calculate_checksum
                import trimesh
                
                texturer = OpenMVSTexturer()
                texturing_dir = Path(cleaned_mesh_path).parent / "texturing"
                texturing_dir.mkdir(exist_ok=True, parents=True)
                
                try:
                    # 1. Texture the un-shifted geometry
                    texture_results = texturer.run_texturing(
                        colmap_workspace=colmap_dir,
                        dense_workspace=colmap_dir / "dense",
                        selected_mesh=cleanup_stats["pre_aligned_mesh_path"],
                        output_dir=texturing_dir
                    )
                    
                    textured_path = texture_results["textured_mesh_path"]
                    tex_mesh = trimesh.load(textured_path, force="mesh")
                    scene_mesh = tex_mesh.dump(concatenate=True) if isinstance(tex_mesh, trimesh.Scene) else tex_mesh
                    has_uv = False
                    if hasattr(scene_mesh.visual, 'uv') and scene_mesh.visual.uv is not None and len(scene_mesh.visual.uv) > 0:
                        has_uv = True
                    
                    if has_uv:
                        texturing_status = "real"
                        
                        # 2. Re-apply the alignment shift to the textured OBJ directly (preserves UVs/MTL reliably)
                        pivot = metadata.pivot_offset
                        aligned_textured_obj = str(Path(cleaned_mesh_path).parent / "textured_aligned_mesh.obj")
                        
                        with open(textured_path, "r", encoding="utf-8") as f_in, open(aligned_textured_obj, "w", encoding="utf-8") as f_out:
                            for line in f_in:
                                if line.startswith("v "):
                                    parts = line.strip().split()
                                    if len(parts) >= 4:
                                        x = float(parts[1]) + pivot["x"]
                                        y = float(parts[2]) + pivot["y"]
                                        z = float(parts[3]) + pivot["z"]
                                        f_out.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                                        continue
                                f_out.write(line)
                                
                        texture_results["textured_mesh_path"] = aligned_textured_obj
                        
                        # Update output manifest source of truth to reflect the textured mesh
                        manifest.textured_mesh_path = texture_results["textured_mesh_path"]
                        manifest.texture_atlas_paths = texture_results["texture_atlas_paths"]
                        manifest.texturing_engine = texture_results["texturing_engine"]
                        manifest.texturing_log_path = texture_results["log_path"]
                        manifest.mesh_metadata.uv_present = has_uv
                        
                        # ALIGN MANIFEST TRUTH
                        manifest.mesh_path = manifest.textured_mesh_path
                        manifest.mesh_metadata.vertex_count = len(scene_mesh.vertices)
                        manifest.mesh_metadata.face_count = len(scene_mesh.faces)
                        manifest.mesh_metadata.has_texture = True
                        manifest.checksum = calculate_checksum(manifest.mesh_path)
                        
                        cleanup_stats["cleaned_texture_path"] = manifest.texture_atlas_paths[0] if manifest.texture_atlas_paths else None
                        cleaned_mesh_path = manifest.textured_mesh_path
                        session.cleanup_mesh_path = cleaned_mesh_path
                    else:
                        texturing_status = "degraded"
                        logger.warning("Texturing produced mesh but no UV coordinates detected.")
                        
                except Exception as e:
                    logger.warning(f"Texturing stage failed/degraded: {e}")
                    texturing_status = "degraded"
                    
        manifest.texturing_status = texturing_status
        manifest_file = Path(session.reconstruction_manifest_path) if session.reconstruction_manifest_path else self.data_root / "reconstructions" / f"job_{session.session_id}" / "manifest.json"
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
        primary_texture = manifest.texture_atlas_paths[0] if manifest.texture_atlas_paths else manifest.texture_path
        texture_path = cleanup_stats.get("cleaned_texture_path") or primary_texture
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

        primary_texture = manifest.texture_atlas_paths[0] if manifest.texture_atlas_paths else manifest.texture_path
        texture_path = cleanup_stats.get("cleaned_texture_path") or primary_texture
        texture_path_exists = bool(texture_path and Path(texture_path).exists())
        has_uv = bool(export_metrics["has_uv"])
        has_material = bool(export_metrics["has_material"])
        has_embedded_texture = bool(export_metrics["has_embedded_texture"])
        texture_status = export_metrics.get("texture_integrity_status", "missing")

        validation_input = {
            "poly_count": int(export_metrics["face_count"]),
            "texture_status": texture_status,
            "bbox": export_metrics["bbox"],
            "ground_offset": float(export_metrics["ground_offset"]),
            "cleanup_stats": cleanup_stats,
            "texture_path_exists": bool(texture_path_exists or has_embedded_texture),
            "has_uv": has_uv,
            "has_material": has_material,
            "has_embedded_texture": has_embedded_texture,
            "texture_count": export_metrics.get("texture_count", 0),
            "material_count": export_metrics.get("material_count", 0),
            "texture_integrity_status": texture_status,
            "material_semantic_status": export_metrics.get("material_semantic_status", "geometry_only"),
            "basecolor_present": bool(export_metrics.get("basecolor_present", False)),
            "normal_present": bool(export_metrics.get("normal_present", False)),
            "metallic_roughness_present": bool(export_metrics.get("metallic_roughness_present", False)),
            "occlusion_present": bool(export_metrics.get("occlusion_present", False)),
            "emissive_present": bool(export_metrics.get("emissive_present", False)),
            "material_integrity_status": export_metrics.get("material_integrity_status", "missing"),
            "delivery_geometry_count": int(export_metrics["geometry_count"]),
            "delivery_component_count": int(export_metrics["component_count"]),
        }

        report = self.validator.validate(asset_id, validation_input)
        reports_dir = self.session_manager.get_capture_path(session.session_id) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        validation_report_path = reports_dir / "validation_report.json"
        atomic_write_json(validation_report_path, report.model_dump(mode="json"))

        logger.info(f"Validation Decision for {session.session_id}: {report.final_decision.upper()}")
        updated = self._persist_session(
            session,
            new_status=AssetStatus.VALIDATED,
            asset_id=asset_id,
            validation_report_path=str(validation_report_path),
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

        return current

    def _update_guidance(self, session: CaptureSession):
        """
        Generates and persists the latest guidance report based on current session state.
        """
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
                failure_reason=session.failure_reason
            )

            # Persist artifacts
            guidance_json_path = reports_dir / "guidance_report.json"
            guidance_md_path = reports_dir / "guidance_summary.md"
            
            atomic_write_json(guidance_json_path, guidance.model_dump(mode="json"))
            with open(guidance_md_path, "w", encoding="utf-8") as f:
                f.write(self.guidance_aggregator.to_markdown(guidance))

            logger.info(f"Guidance updated for {session.session_id}")
        except Exception as e:
            logger.warning(f"Failed to update guidance for {session.session_id}: {e}")


worker_instance = IngestionWorker()
