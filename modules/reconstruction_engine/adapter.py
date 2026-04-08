from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path
import os
import time

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

import shutil
import subprocess

class COLMAPAdapter(ReconstructionAdapter):
    """
    Adapter for COLMAP reconstruction engine using automatic_reconstructor.
    """
    def __init__(self, engine_path: Optional[str] = None):
        self._engine_path = engine_path or os.getenv("RECON_ENGINE_PATH")
        
        # Fallback to user-provided or well-known paths if env var is missing
        if not self._engine_path:
            well_known = [
                r"C:\colmap\colmap\COLMAP.bat",
                r"C:\colmap\COLMAP.bat"
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

    def run_reconstruction(self, input_frames: List[str], output_dir: Path) -> dict:
        if not self._engine_path:
            raise RuntimeError("Reconstruction engine path (RECON_ENGINE_PATH) not configured. Production run aborted.")
        
        # 1. Prepare Workspace
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # User requirement: Copy frames (safer for Windows permissions)
        for frame_path in input_frames:
            src = Path(frame_path)
            if not src.exists():
                continue
            dst = images_dir / src.name
            shutil.copy2(src, dst)

        # 2. Build automatic_reconstructor command
        # Note: --max_image_size is removed for compatibility with typical Windows builds
        # if quality is needed, we could add --quality {low, medium, high, extreme}
        cmd = [
            self._engine_path,
            "automatic_reconstructor",
            "--workspace_path", str(output_dir),
            "--image_path", str(images_dir),
            "--use_gpu", "1" if self._use_gpu else "0"
        ]

        # 3. Execute
        log_path = output_dir / "reconstruction.log"
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"Starting COLMAP with command: {' '.join(cmd)}\n\n")
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                if process.stdout:
                    for line in process.stdout:
                        log_file.write(line)
                
                process.wait()
            except Exception as e:
                raise RuntimeError(f"Failed to launch COLMAP: {str(e)}")

        if process.returncode != 0:
            raise RuntimeError(f"COLMAP failed with exit code {process.returncode}. Check logs at {log_path}")

        # 4. Artifact Discovery (User requirement: prioritize meshes)
        potential_meshes = [
            output_dir / "dense" / "0" / "meshed-poisson.ply",
            output_dir / "dense" / "0" / "meshed-delaunay.ply",
            output_dir / "meshed-poisson.ply",
            output_dir / "meshed-delaunay.ply",
            output_dir / "dense" / "0" / "fused.ply" # Check dense subfolder if exists
        ]
        
        mesh_path = None
        for p in potential_meshes:
            if p.exists():
                mesh_path = p
                break
        
        if not mesh_path:
             # Deep search if standard paths fail
             ply_files = list(output_dir.rglob("*.ply"))
             if ply_files:
                 mesh_path = ply_files[0]
        
        if not mesh_path:
            raise RuntimeError(f"COLMAP completed but no .ply artifacts found in {output_dir}")

        # 5. OpenMVS Texturing Stage (Photorealism)
        openmvs_dir = os.getenv("OPENMVS_BIN_PATH", r"C:\OpenMVS")
        texture_path = str(output_dir / "dummy_texture.png")
        
        # Check if OpenMVS tools exist
        interface_colmap = Path(openmvs_dir) / "InterfaceCOLMAP.exe"
        texture_mesh = Path(openmvs_dir) / "TextureMesh.exe"
        
        if interface_colmap.exists() and texture_mesh.exists():
            log_file = open(log_path, "a", encoding="utf-8")
            log_file.write("\n\n--- Starting OpenMVS Texturing Pipeline ---\n")
            
            try:
                # Setup paths
                scene_mvs = output_dir / "scene.mvs"
                dense_dir = output_dir / "dense" / "0"
                scene_texture_obj = output_dir / "scene_texture.obj"
                
                # Command 1: Convert COLMAP workspace to MVS format
                cmd_interface = [
                    str(interface_colmap),
                    "-i", str(dense_dir),
                    "-o", str(scene_mvs)
                ]
                log_file.write(f"Running InterfaceCOLMAP: {' '.join(cmd_interface)}\n")
                subprocess.run(cmd_interface, stdout=log_file, stderr=subprocess.STDOUT, check=True)
                
                # Command 2: Project original images onto the Poisson Mesh
                cmd_texture = [
                    str(texture_mesh),
                    str(scene_mvs),
                    "-m", str(mesh_path),
                    "-o", str(scene_texture_obj)
                ]
                log_file.write(f"Running TextureMesh: {' '.join(cmd_texture)}\n")
                subprocess.run(cmd_texture, stdout=log_file, stderr=subprocess.STDOUT, check=True)
                
                # If success, update outputs to MVS results
                if scene_texture_obj.exists():
                    mesh_path = scene_texture_obj
                    texture_path = str(output_dir / "scene_texture.png")
                    log_file.write("\nOpenMVS Texturing completed successfully.\n")
                    
            except subprocess.CalledProcessError as e:
                log_file.write(f"\nOpenMVS Process failed with code {e.returncode}. Falling back to default COLMAP output.\n")
            except Exception as e:
                log_file.write(f"\nOpenMVS Error: {str(e)}. Falling back to default COLMAP output.\n")
            finally:
                log_file.close()

        return {
            "mesh_path": str(mesh_path),
            "texture_path": texture_path, # will point to scene_texture.png if MVS succeeded
            "log_path": str(log_path),
            "vertex_count": 0, # Placeholder, could be parsed from PLY header
            "face_count": 0
        }

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
