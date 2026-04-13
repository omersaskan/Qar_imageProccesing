import os
import shutil
import time
from pathlib import Path
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import trimesh

from modules.shared_contracts.models import (
    ReconstructionJob, 
    ReconstructionAttemptResult, 
    ReconstructionAttemptType, 
    ReconstructionAudit
)
from modules.utils.file_persistence import atomic_write_json, calculate_checksum
from .adapter import COLMAPAdapter, ReconstructionAdapter, SimulatedAdapter
from .failures import (
    InsufficientInputError, 
    MissingArtifactError, 
    RuntimeReconstructionError,
    InsufficientReconstructionError
)
from .output_manifest import MeshMetadata, OutputManifest
from modules.operations.settings import settings


class ReconstructionRunner:
    def __init__(self, adapter: Optional[ReconstructionAdapter] = None):
        self.is_production = os.getenv("ENV", "development").lower() == "production"
        self.allow_simulated = os.getenv("ALLOW_SIMULATED_RECONSTRUCTION", "false").lower() == "true"

        if not adapter:
            engine_choice = os.getenv("RECON_ENGINE", "colmap").lower()

            if engine_choice == "colmap":
                adapter = COLMAPAdapter()
                if self.is_production and not adapter._engine_path:
                    raise RuntimeError("Production run aborted: RECON_ENGINE_PATH must be configured for COLMAP.")
            elif engine_choice == "simulated":
                if self.is_production:
                    raise RuntimeError(
                        "Production run aborted: RECON_ENGINE=colmap must be configured when ENV=production. "
                        "Simulated engine is strictly prohibited."
                    )
                if not self.allow_simulated:
                    raise RuntimeError(
                        "Simulated reconstruction is disabled by default. "
                        "Set ALLOW_SIMULATED_RECONSTRUCTION=true only for explicit test flows."
                    )
                adapter = SimulatedAdapter()
            else:
                raise RuntimeError(f"Unsupported reconstruction engine '{engine_choice}'.")

        if self.is_production and adapter.is_stub:
            raise RuntimeError(
                "SECURITY/INTEGRITY VIOLATION: A stub/simulated engine was detected in a PRODUCTION environment. "
                "The reconstruction pipeline has been halted to prevent placeholder data leakage."
            )

        self.adapter = adapter

    def _read_image(self, image_path: Path):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None:
            return image

        try:
            image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
            if image_bytes.size == 0:
                return None
            return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _validate_input_frames(self, input_frames: List[str]) -> List[str]:
        valid_frames: List[str] = []
        invalid_reasons: List[str] = []

        for frame_path_str in input_frames:
            frame_path = Path(frame_path_str)
            if not frame_path.exists():
                invalid_reasons.append(f"{frame_path}: missing")
                continue
            if frame_path.stat().st_size <= 0:
                invalid_reasons.append(f"{frame_path}: zero-byte")
                continue

            frame = self._read_image(frame_path)
            if frame is None or frame.size == 0:
                invalid_reasons.append(f"{frame_path}: unreadable")
                continue

            valid_frames.append(str(frame_path))

        if invalid_reasons:
            raise InsufficientInputError(
                "Invalid reconstruction input frames detected: " + "; ".join(invalid_reasons)
            )

        if len(valid_frames) < 3:
            raise InsufficientInputError(
                f"At least 3 readable frames are required for reconstruction; got {len(valid_frames)}."
            )

        return valid_frames

    def _is_ascii_safe_path(self, path: Path) -> bool:
        try:
            str(path).encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    def _prepare_execution_workspace(self, job: ReconstructionJob) -> Tuple[Path, Optional[Path]]:
        job_dir = Path(job.job_dir).resolve()
        if os.name != "nt" or self._is_ascii_safe_path(job_dir):
            return job_dir, None

        ascii_root = Path(os.getenv("MESHYSIZ_ASCII_WORKROOT", r"C:\meshysiz_ascii"))
        ascii_workspace = ascii_root / job.job_id
        if ascii_workspace.exists():
            shutil.rmtree(ascii_workspace, ignore_errors=True)
        ascii_workspace.mkdir(parents=True, exist_ok=True)
        return ascii_workspace, job_dir

    def _sync_workspace_back(self, source_dir: Path, target_dir: Path) -> None:
        if source_dir.resolve() == target_dir.resolve():
            return

        for root, _, files in os.walk(source_dir):
            root_path = Path(root)
            relative_root = root_path.relative_to(source_dir)
            destination_root = target_dir / relative_root
            destination_root.mkdir(parents=True, exist_ok=True)
            for name in files:
                shutil.copy2(root_path / name, destination_root / name)

    def _remap_artifact_paths(self, results: dict, source_dir: Path, target_dir: Path) -> dict:
        remapped = dict(results)
        for key in ("mesh_path", "texture_path", "log_path"):
            value = remapped.get(key)
            if not value:
                continue

            source_path = Path(value)
            try:
                relative = source_path.resolve().relative_to(source_dir.resolve())
            except Exception:
                continue
            remapped[key] = str((target_dir / relative).resolve())

        return remapped

    def _validate_mesh_artifact(self, mesh_path: Path) -> Tuple[int, int]:
        if not mesh_path.exists():
            raise MissingArtifactError(mesh_path.name)

        try:
            mesh = trimesh.load(str(mesh_path))
        except Exception as e:
            raise MissingArtifactError(f"{mesh_path.name}: unreadable mesh ({e})")

        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        if not isinstance(mesh, trimesh.Trimesh):
            raise MissingArtifactError(f"{mesh_path.name}: not a polygon mesh")

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise MissingArtifactError(f"{mesh_path.name}: empty or face-less mesh")

        return int(len(mesh.vertices)), int(len(mesh.faces))

    def _score_attempt(self, results: dict) -> float:
        """
        Ranks reconstruction attempts based on quality metrics.
        Higher is better.
        """
        if not results:
            return -1.0
        
        # Base score from registered images (critical for sparse success)
        score = results.get("registered_images", 0) * 100.0
        
        # Contribution from point counts
        score += results.get("sparse_points", 0) * 0.5
        score += results.get("dense_points_fused", 0) * 0.1
        
        # Penalty for failed meshing
        if results.get("mesher_used") == "failed":
            score -= 5000.0
        elif results.get("mesher_used") in ["poisson", "delaunay"]:
            score += 2000.0
            
        return score

    def run(self, job: ReconstructionJob) -> OutputManifest:
        if not job.input_frames or len(job.input_frames) < 3:
            raise InsufficientInputError("At least 3 high-quality frames are required for reconstruction.")

        validated_frames = self._validate_input_frames(job.input_frames)
        
        # Determine fallback sequence from settings
        fallback_steps = settings.recon_fallback_steps
        if not fallback_steps:
            fallback_steps = ["default"]
            
        audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
        job_dir = Path(job.job_dir).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)
        
        best_results = None
        best_score = -1.0
        best_index = -1
        
        for i, step_name in enumerate(fallback_steps):
            attempt_start = time.time()
            attempt_type = ReconstructionAttemptType(step_name)
            
            # Skip unmasked if disabled
            if attempt_type == ReconstructionAttemptType.UNMASKED and not settings.recon_unmasked_fallback_enabled:
                logging.warning(f"Skipping UNMASKED fallback as it is disabled in settings.")
                continue

            # Create isolated sub-workspace for this attempt
            attempt_dir = job_dir / f"attempt_{i}_{step_name}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Starting reconstruction attempt {i}: type={step_name}")
            
            # Density & Masking logic
            density = 1.0
            enforce_masks = True
            current_frames = validated_frames
            sampling_rate_used = None
            reextracted_frames_dir = None
            
            if attempt_type == ReconstructionAttemptType.DENSER_FRAMES:
                if job.source_video_path and Path(job.source_video_path).exists():
                    sampling_rate_used = settings.recon_fallback_sample_rate
                    logging.info(f"Attempt {i}: Re-extracting frames with denser sampling rate={sampling_rate_used}")
                    try:
                        from modules.capture_workflow.frame_extractor import FrameExtractor
                        
                        # Create denser extractor
                        extractor = FrameExtractor()
                        extractor.thresholds.frame_sample_rate = sampling_rate_used
                        
                        # Extract to isolated subdir
                        extract_dir = attempt_dir / "extracted_frames"
                        extract_dir.mkdir(parents=True, exist_ok=True)
                        reextracted_frames_dir = str(extract_dir)
                        
                        new_frames = extractor.extract_keyframes(job.source_video_path, reextracted_frames_dir)
                        if new_frames:
                            current_frames = new_frames
                            logging.info(f"Attempt {i}: Denser extraction successful. count={len(new_frames)}")
                        else:
                            logging.warning(f"Attempt {i}: Denser extraction produced no frames. Falling back to default list.")
                    except Exception as ex_err:
                        logging.error(f"Attempt {i}: Denser extraction failed: {ex_err}")
                else:
                    logging.warning(f"Attempt {i}: source_video_path missing or invalid. Cannot densify accurately.")

            if attempt_type == ReconstructionAttemptType.DEFAULT:
                # Default behavior might use a slightly reduced density to speed up/simplify
                # but for this specific sprint, we'll keep it at 1.0 unless otherwise specified.
                # However, if 'denser_frames' is a thing, usually 'default' is less dense.
                # We'll use 0.5 for default if denser_frames is in the list.
                if ReconstructionAttemptType.DENSER_FRAMES in fallback_steps:
                    density = 0.5
            elif attempt_type == ReconstructionAttemptType.DENSER_FRAMES:
                density = 1.0
            elif attempt_type == ReconstructionAttemptType.UNMASKED:
                enforce_masks = False
                density = 1.0 # Unmasked usually wants all frames
            
            try:
                # Execution workspace handling (ASCII path safety for Windows)
                execution_dir, final_dir = self._prepare_execution_workspace(job)
                # Adjust final_dir to use attempt subfolder
                if final_dir:
                    final_dir = attempt_dir
                else:
                    execution_dir = attempt_dir

                results = self.adapter.run_reconstruction(
                    current_frames, 
                    execution_dir,
                    density=density,
                    enforce_masks=enforce_masks
                )
                
                # Sync back if using ASCII workroot
                if final_dir:
                    self._sync_workspace_back(execution_dir, final_dir)
                    results = self._remap_artifact_paths(results, execution_dir, final_dir)
                    shutil.rmtree(execution_dir, ignore_errors=True)
                
                score = self._score_attempt(results)
                
                attempt_res = ReconstructionAttemptResult(
                    attempt_type=attempt_type,
                    status="success",
                    frames_used=len(current_frames),
                    registered_images=results.get("registered_images", 0),
                    sparse_points=results.get("sparse_points", 0),
                    dense_points_fused=results.get("dense_points_fused", 0),
                    mesher_used=results.get("mesher_used", "none"),
                    mesh_path=results.get("mesh_path"),
                    log_path=results.get("log_path"),
                    sampling_rate_used=sampling_rate_used,
                    source_video_path=job.source_video_path if attempt_type == ReconstructionAttemptType.DENSER_FRAMES else None,
                    reextracted_frames_dir=reextracted_frames_dir,
                    metrics_rank_score=score
                )
                
                # If we have a successful result that passes basic gating, we might stop or continue
                # The user says "Select the best honest result", so we should try all and pick.
                if score > best_score:
                    best_score = score
                    best_results = results
                    best_index = i
                
            except (InsufficientReconstructionError, RuntimeReconstructionError, InsufficientInputError) as e:
                attempt_res = ReconstructionAttemptResult(
                    attempt_type=attempt_type,
                    status="failed" if isinstance(e, RuntimeReconstructionError) else "weak",
                    frames_used=0,
                    error_message=str(e),
                    sampling_rate_used=sampling_rate_used,
                    source_video_path=job.source_video_path if attempt_type == ReconstructionAttemptType.DENSER_FRAMES else None,
                    reextracted_frames_dir=reextracted_frames_dir,
                    metrics_rank_score=-100.0
                )
                logging.warning(f"Attempt {i} ({step_name}) were weak or failed: {e}")
            except Exception as e:
                attempt_res = ReconstructionAttemptResult(
                    attempt_type=attempt_type,
                    status="failed",
                    frames_used=0,
                    error_message=f"Unexpected error: {str(e)}",
                    metrics_rank_score=-1000.0
                )
                logging.error(f"Attempt {i} ({step_name}) crashed: {e}")
            
            audit.attempts.append(attempt_res)
            
            # OPTIONAL: Early exit if default is EXTREMELY good? 
            # User wants honest comparison, so we'll likely continue unless configured otherwise.

        # After all attempts, select best
        if best_results is None or best_score < 0:
            audit.final_status = "recapture_required"
            self._save_audit(audit, job_dir)
            # Find the most descriptive error
            last_err = audit.attempts[-1].error_message if audit.attempts else "All reconstruction attempts failed."
            raise InsufficientReconstructionError(f"All fallback attempts failed. Last error: {last_err}")

        audit.selected_best_index = best_index
        audit.final_status = "success"
        self._save_audit(audit, job_dir)
        
        logging.info(f"Reconstruction complete. Selected best attempt index: {best_index} ({audit.attempts[best_index].attempt_type})")

        return self._finalize_best_attempt(best_results, job, job_dir)

    def _save_audit(self, audit: ReconstructionAudit, job_dir: Path):
        audit_path = job_dir / "reconstruction_audit.json"
        atomic_write_json(audit_path, audit.model_dump(mode="json"))

    def _finalize_best_attempt(self, results: dict, job: ReconstructionJob, job_dir: Path) -> OutputManifest:
        mesh_path = Path(results["mesh_path"])
        texture_path = Path(results["texture_path"])
        log_path = Path(results["log_path"])

        vertex_count, face_count = self._validate_mesh_artifact(mesh_path)
        checksum = calculate_checksum(mesh_path)

        manifest = OutputManifest(
            job_id=job.job_id,
            mesh_path=str(mesh_path),
            texture_path=str(texture_path),
            log_path=str(log_path),
            processing_time_seconds=0.0, # Could calculate total
            engine_type=self.adapter.engine_type,
            is_stub=self.adapter.is_stub,
            mesh_metadata=MeshMetadata(
                vertex_count=vertex_count,
                face_count=face_count,
                has_texture=texture_path.exists(),
            ),
            checksum=checksum,
        )
        
        # Overwrite root manifest with the best one
        manifest_path = job_dir / "manifest.json"
        atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
        return manifest
