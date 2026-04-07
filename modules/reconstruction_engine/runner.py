import os
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from shared_contracts.models import ReconstructionJob
from shared_contracts.lifecycle import ReconstructionStatus
from .output_manifest import OutputManifest, MeshMetadata
from .failures import RuntimeReconstructionError, MissingArtifactError, InsufficientInputError
from .adapter import ReconstructionAdapter, COLMAPAdapter, SimulatedAdapter

class ReconstructionRunner:
    def __init__(self, adapter: Optional[ReconstructionAdapter] = None):
        # Determine strictness level
        self.is_production = os.getenv("ENV", "development").lower() == "production"
        
        # 1. Automatic Adapter Selection if none provided
        if not adapter:
            engine_choice = os.getenv("RECON_ENGINE", "colmap").lower()
            
            if engine_choice == "colmap":
                adapter = COLMAPAdapter()
                if self.is_production and not adapter._engine_path:
                    raise RuntimeError("Production run aborted: RECON_ENGINE_PATH must be configured for COLMAP.")
            elif not self.is_production:
                # Fallback to simulated ONLY in non-production
                adapter = SimulatedAdapter()
            else:
                raise RuntimeError(
                    "Production run aborted: RECON_ENGINE=colmap must be configured when ENV=production. "
                    "Simulated engine is strictly prohibited."
                )

        # 2. Strict Production Guard
        if self.is_production and adapter.is_stub:
            raise RuntimeError(
                "SECURITY/INTEGRITY VIOLATION: A stub/simulated engine was detected in a PRODUCTION environment. "
                "The reconstruction pipeline has been halted to prevent placeholder data leakage."
            )

        self.adapter = adapter

    def run(self, job: ReconstructionJob) -> OutputManifest:
        """
        Runs the reconstruction process using the configured adapter.
        """
        # 1. Validation check
        if not job.input_frames or len(job.input_frames) < 3:
            raise InsufficientInputError("At least 3 high-quality frames are required for reconstruction.")

        start_time = time.time()
        job_dir = Path(job.job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

        # 2. Run Adapter
        try:
            results = self.adapter.run_reconstruction(job.input_frames, job_dir)
        except Exception as e:
            raise RuntimeReconstructionError(f"Engine ({self.adapter.engine_type}) failed: {str(e)}")

        # 3. Artifact Verification
        mesh_path = Path(results["mesh_path"])
        texture_path = Path(results["texture_path"])
        log_path = Path(results["log_path"])

        if not mesh_path.exists():
            raise MissingArtifactError(f"Engine output missing: {mesh_path.name}")

        # 4. Create Output Manifest
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
                vertex_count=results.get("vertex_count"),
                face_count=results.get("face_count"),
                has_texture=texture_path.exists()
            )
        )

        # 5. Save manifest.json inside job directory
        manifest_path = job_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        return manifest
