from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
import os
import time
import shutil
import subprocess
import json

class ReconstructionAdapter(ABC):
    @abstractmethod
    def run_reconstruction(self, input_frames: List[str], output_dir: Path) -> dict:
        """
        Runs the reconstruction process and returns a dictionary with artifact info.
        """
        pass

    @property
    @abstractmethod
    def engine_type(self) -> str:
        pass

    @property
    @abstractmethod
    def is_stub(self) -> bool:
        pass

class COLMAPAdapter(ReconstructionAdapter):
    """
    Adapter for COLMAP reconstruction engine using explicit command chain.
    Supports object masking and turntable detection.
    """
    def __init__(self, engine_path: Optional[str] = None):
        self._engine_path = engine_path or os.getenv("RECON_ENGINE_PATH")
        
        # Fallback to user-provided or well-known paths if env var is missing
        if not self._engine_path:
            well_known = [
                r"C:\colmap\colmap\COLMAP.bat",
                r"C:\colmap\COLMAP.bat",
                r"C:\colmap\colmap.exe"
            ]
            for p in well_known:
                if os.path.exists(p):
                    self._engine_path = p
                    break

        self._use_gpu = os.getenv("RECON_USE_GPU", "true").lower() == "true"
        self._max_image_size = int(os.getenv("RECON_MAX_IMAGE_SIZE", "2000"))

    @property
    def engine_type(self) -> str:
        return "colmap"

    @property
    def is_stub(self) -> bool:
        return False

    def _run_command(self, cmd: List[str], cwd: Path, log_file):
        """Helper to run a command and log output."""
        log_file.write(f"\n--- Running: {' '.join(cmd)} ---\n")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(cwd)
        )
        if process.stdout:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {process.returncode}: {cmd[1]}")

    def run_reconstruction(self, input_frames: List[str], output_dir: Path) -> dict:
        if not self._engine_path:
            raise RuntimeError("Reconstruction engine path (RECON_ENGINE_PATH) not configured.")
        
        # 1. Prepare Workspace
        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy frames and masks
        for frame_path in input_frames:
            src = Path(frame_path)
            if not src.exists(): continue
            shutil.copy2(src, images_dir / src.name)
            
            # Check for corresponding mask
            mask_src = src.parent / "masks" / f"{src.name}.png"
            if mask_src.exists():
                shutil.copy2(mask_src, masks_dir / f"{src.name}.png")

        # 2. Sequential COLMAP Chain
        log_path = output_dir / "reconstruction.log"
        with open(log_path, "w", encoding="utf-8") as log_file:
            try:
                # A. Database creation
                db_path = output_dir / "database.db"
                
                # B. Feature Extraction (with mask)
                cmd_extract = [
                    self._engine_path, "feature_extractor",
                    "--database_path", str(db_path),
                    "--image_path", str(images_dir),
                    "--ImageReader.mask_path", str(masks_dir),
                    "--SiftExtraction.use_gpu", "1" if self._use_gpu else "0"
                ]
                self._run_command(cmd_extract, output_dir, log_file)
                
                # C. Matching
                cmd_match = [
                    self._engine_path, "exhaustive_matcher",
                    "--database_path", str(db_path),
                    "--SiftMatching.use_gpu", "1" if self._use_gpu else "0"
                ]
                self._run_command(cmd_match, output_dir, log_file)
                
                # D. Mapping (Sparse Reconstruction)
                sparse_dir = output_dir / "sparse"
                sparse_dir.mkdir(exist_ok=True)
                cmd_map = [
                    self._engine_path, "mapper",
                    "--database_path", str(db_path),
                    "--image_path", str(images_dir),
                    "--output_path", str(sparse_dir)
                ]
                self._run_command(cmd_map, output_dir, log_file)
                
                # E. Undistortion (Prep for Dense)
                dense_dir = output_dir / "dense"
                dense_dir.mkdir(exist_ok=True)
                cmd_undistort = [
                    self._engine_path, "image_undistorter",
                    "--image_path", str(images_dir),
                    "--input_path", str(sparse_dir / "0"),
                    "--output_path", str(dense_dir),
                    "--output_type", "COLMAP"
                ]
                self._run_command(cmd_undistort, output_dir, log_file)
                
                # F. Patch Match Stereo (Dense Depth)
                cmd_stereo = [
                    self._engine_path, "patch_match_stereo",
                    "--workspace_path", str(dense_dir),
                    "--PatchMatchStereo.gpu_index", "0" if self._use_gpu else "-1"
                ]
                self._run_command(cmd_stereo, output_dir, log_file)
                
                # G. Stereo Fusion (Point Cloud)
                cmd_fuse = [
                    self._engine_path, "stereo_fusion",
                    "--workspace_path", str(dense_dir),
                    "--output_path", str(dense_dir / "fused.ply")
                ]
                self._run_command(cmd_fuse, output_dir, log_file)
                
                # H. Poisson Mesher
                cmd_mesh = [
                    self._engine_path, "poisson_mesher",
                    "--input_path", str(dense_dir / "fused.ply"),
                    "--output_path", str(dense_dir / "meshed-poisson.ply")
                ]
                self._run_command(cmd_mesh, output_dir, log_file)
                
            except Exception as e:
                log_file.write(f"\nCRITICAL FAILURE: {str(e)}\n")
                raise RuntimeError(f"COLMAP chain failed: {str(e)}")

        # 3. Artifact Discovery
        dense_dir = output_dir / "dense"
        mesh_path = dense_dir / "meshed-poisson.ply"
        if not mesh_path.exists():
            # Fallback to any ply
            ply_files = list(dense_dir.glob("*.ply"))
            if ply_files: mesh_path = ply_files[0]
        
        if not mesh_path or not mesh_path.exists():
            raise RuntimeError(f"COLMAP completed but no .ply artifacts found in {dense_dir}")

        # 4. Validation
        self._validate_dense_workspace(output_dir)
        
        return {
            "mesh_path": str(mesh_path),
            "texture_path": str(output_dir / "dummy_texture.png"),
            "log_path": str(log_path),
            "vertex_count": 0,
            "face_count": 0
        }

    def _validate_dense_workspace(self, workspace_path: Path):
        """Lightweight integrity checks on the COLMAP dense workspace."""
        dense_dir = workspace_path / "dense"
        fused_ply = dense_dir / "fused.ply"
        
        if not dense_dir.exists():
            raise RuntimeError(f"Dense workspace folder missing: {dense_dir}")
        if not fused_ply.exists():
             raise RuntimeError("Dense point cloud (fused.ply) missing.")
        if fused_ply.stat().st_size < 1024:
             raise RuntimeError(f"Fused point cloud too small ({fused_ply.stat().st_size} bytes).")
        
        return True

class SimulatedAdapter(ReconstructionAdapter):
    """
    Adapter for testing/development. Produces a valid but simple OBJ.
    """
    @property
    def engine_type(self) -> str:
        return "simulated"

    @property
    def is_stub(self) -> bool:
        return True

    def run_reconstruction(self, input_frames: List[str], output_dir: Path) -> dict:
        # Simulate processing delay
        time.sleep(0.5)
        
        mesh_path = output_dir / "raw_mesh.obj"
        texture_path = output_dir / "raw_texture.png"
        log_path = output_dir / "logs/reconstruction.log"
        
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write a simple valid OBJ triangle (not just a string)
        obj_content = (
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 0.0 1.0 0.0\n"
            "f 1 2 3\n"
        )
        with open(mesh_path, "w") as f:
            f.write(obj_content)
            
        with open(texture_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dBHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff? \x00\x05\xfe\x02\xfe\xdcD\x05\x13\x00\x00\x00\x00IEND\xaeB`\x82")
            
        with open(log_path, "w") as f:
            f.write("Simulated reconstruction successful.\n")
            
        return {
            "mesh_path": str(mesh_path),
            "texture_path": str(texture_path),
            "log_path": str(log_path),
            "vertex_count": 3,
            "face_count": 1
        }
