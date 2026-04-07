import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from shared_contracts.models import ReconstructionJob
from shared_contracts.lifecycle import ReconstructionStatus
from .output_manifest import OutputManifest, MeshMetadata
from .failures import RuntimeReconstructionError, MissingArtifactError, InsufficientInputError

class ReconstructionRunner:
    def __init__(self):
        pass

    def run(self, job: ReconstructionJob) -> OutputManifest:
        """
        Simulates the reconstruction process and generates raw outputs.
        """
        # 1. Validation check
        if not job.input_frames or len(job.input_frames) < 3:
            raise InsufficientInputError("At least 3 high-quality frames are required for reconstruction.")

        start_time = time.time()
        job_dir = Path(job.job_dir)
        
        # Simulating processing delay
        # time.sleep(0.1) 

        # 2. Simulate Output Generation
        mesh_filename = "raw_mesh.obj"
        texture_filename = "raw_texture.png"
        log_filename = "logs/reconstruction.log"

        mesh_path = job_dir / mesh_filename
        texture_path = job_dir / texture_filename
        log_path = job_dir / log_filename

        # Creating dummy files for the stub
        with open(mesh_path, "w") as f:
            f.write("# Raw Mesh Placeholder\n")
        
        with open(texture_path, "w") as f:
            f.write("Raw Texture Placeholder\n")

        with open(log_path, "w") as f:
            f.write(f"Reconstruction started at {datetime.utcnow()}\n")
            f.write(f"Processing {len(job.input_frames)} high-quality frames...\n")
            f.write("Geometric reconstruction completed.\n")

        # 3. Artifact Verification (as requested)
        if not mesh_path.exists():
            raise MissingArtifactError(mesh_filename)
        if not texture_path.exists():
            raise MissingArtifactError(texture_filename)

        # 4. Create Output Manifest
        processing_time = time.time() - start_time
        manifest = OutputManifest(
            job_id=job.job_id,
            mesh_path=str(mesh_path),
            texture_path=str(texture_path),
            log_path=str(log_path),
            processing_time_seconds=processing_time,
            mesh_metadata=MeshMetadata(
                vertex_count=len(job.input_frames) * 150,
                face_count=len(job.input_frames) * 280,
                has_texture=True
            )
        )

        # 5. Save manifest.json inside job directory
        manifest_path = job_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        return manifest
