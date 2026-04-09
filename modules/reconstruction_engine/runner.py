import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import trimesh

from modules.shared_contracts.models import ReconstructionJob
from modules.utils.file_persistence import atomic_write_json, calculate_checksum
from .adapter import COLMAPAdapter, ReconstructionAdapter, SimulatedAdapter
from .failures import InsufficientInputError, MissingArtifactError, RuntimeReconstructionError
from .output_manifest import MeshMetadata, OutputManifest


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

    def run(self, job: ReconstructionJob) -> OutputManifest:
        if not job.input_frames or len(job.input_frames) < 3:
            raise InsufficientInputError("At least 3 high-quality frames are required for reconstruction.")

        validated_frames = self._validate_input_frames(job.input_frames)

        start_time = time.time()
        job_dir = Path(job.job_dir).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)
        execution_dir, final_dir = self._prepare_execution_workspace(job)

        try:
            results = self.adapter.run_reconstruction(validated_frames, execution_dir)
        except Exception as e:
            if final_dir is not None and execution_dir.exists():
                self._sync_workspace_back(execution_dir, final_dir)
            raise RuntimeReconstructionError(f"Engine ({self.adapter.engine_type}) failed: {str(e)}")
        else:
            if final_dir is not None:
                self._sync_workspace_back(execution_dir, final_dir)
                results = self._remap_artifact_paths(results, execution_dir, final_dir)
                workspace_dir = final_dir
            else:
                workspace_dir = job_dir
        finally:
            if final_dir is not None and execution_dir.exists():
                shutil.rmtree(execution_dir, ignore_errors=True)

        mesh_path = Path(results["mesh_path"])
        texture_path = Path(results["texture_path"])
        log_path = Path(results["log_path"])

        if not log_path.exists():
            raise MissingArtifactError(log_path.name)

        vertex_count, face_count = self._validate_mesh_artifact(mesh_path)
        checksum = calculate_checksum(mesh_path)

        processing_time = time.time() - start_time
        manifest = OutputManifest(
            job_id=job.job_id,
            mesh_path=str(mesh_path),
            texture_path=str(texture_path),
            log_path=str(log_path),
            processing_time_seconds=processing_time,
            engine_type=self.adapter.engine_type,
            is_stub=self.adapter.is_stub,
            mesh_metadata=MeshMetadata(
                vertex_count=vertex_count,
                face_count=face_count,
                has_texture=texture_path.exists(),
            ),
            checksum=checksum,
        )

        manifest_path = workspace_dir / "manifest.json"
        atomic_write_json(manifest_path, manifest.model_dump(mode="json"))

        return manifest
