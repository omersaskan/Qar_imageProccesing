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

        # 5. Pre-flight Validation (User requirement: verify dense workspace integrity)
        try:
            self._validate_dense_workspace(output_dir)
        except RuntimeError as e:
             # We log but continue to attempt discovery, though OpenMVS will likely fail
             with open(log_path, "a", encoding="utf-8") as log_f:
                 log_f.write(f"\n[WARNING] Pre-flight Validation Failed: {str(e)}\n")
             # If we want to hard-abort OpenMVS on inconsistency, we could re-raise here.
             # Given Sample 2's failure, a hard abort is safer to prevent cryptical OpenMVS errors.
             raise e

        # 6. OpenMVS Texturing Stage (Photorealism)
        openmvs_dir = os.getenv("OPENMVS_BIN_PATH", r"C:\OpenMVS")
        texture_path = str(output_dir / "dummy_texture.png")
        final_vertex_count = 0
        final_face_count = 0
        
        # Check if OpenMVS tools exist
        interface_colmap = Path(openmvs_dir) / "InterfaceCOLMAP.exe"
        texture_mesh_bin = Path(openmvs_dir) / "TextureMesh.exe"
        reconstruct_mesh_bin = Path(openmvs_dir) / "ReconstructMesh.exe"
        
        if interface_colmap.exists() and texture_mesh_bin.exists():
            with open(log_path, "a", encoding="utf-8") as log_file_handle:
                log_file_handle.write("\n\n--- Starting OpenMVS Texturing Pipeline ---\n")
                
                try:
                    # Configuration
                    res_level = os.getenv("OPENMVS_RESOLUTION_LEVEL", "2")
                    
                    # Setup paths (all relative to output_dir since we use cwd=output_dir)
                    dense_dir = Path("dense") / "0"
                    scene_mvs = output_dir / "scene.mvs"
                    
                    # ── Step 1: InterfaceCOLMAP ──────────────────────────────
                    cmd_interface = [
                        str(interface_colmap),
                        "-i", str(dense_dir),
                        "-o", "scene.mvs"
                    ]
                    log_file_handle.write(f"Running InterfaceCOLMAP in {output_dir}: {' '.join(cmd_interface)}\n")
                    
                    res_interface = subprocess.run(
                        cmd_interface, 
                        cwd=str(output_dir),
                        capture_output=True, 
                        text=True, 
                        errors="replace",
                        check=False
                    )
                    log_file_handle.write(res_interface.stdout)
                    if res_interface.stderr:
                        log_file_handle.write(f"\nSTDERR:\n{res_interface.stderr}\n")
                    
                    if res_interface.returncode != 0:
                        raise RuntimeError(f"InterfaceCOLMAP failed with code {res_interface.returncode}")
                    
                    # ── Step 2: Choose best mesh for texturing ────────────────
                    # PREFERRED: ReconstructMesh produces a clean Delaunay mesh
                    # that survives OpenMVS seam leveling (proven in Phase 3).
                    # FALLBACK: Decimated Poisson mesh from COLMAP.
                    texturing_mesh_rel = None
                    
                    if reconstruct_mesh_bin.exists():
                        log_file_handle.write("\n--- ReconstructMesh: Building Delaunay mesh ---\n")
                        cmd_reconstruct = [
                            str(reconstruct_mesh_bin),
                            "scene.mvs"
                        ]
                        log_file_handle.write(f"Running ReconstructMesh in {output_dir}: {' '.join(cmd_reconstruct)}\n")
                        
                        res_reconstruct = subprocess.run(
                            cmd_reconstruct,
                            cwd=str(output_dir),
                            capture_output=True,
                            text=True,
                            errors="replace",
                            check=False
                        )
                        log_file_handle.write(res_reconstruct.stdout)
                        if res_reconstruct.stderr:
                            log_file_handle.write(f"\nSTDERR:\n{res_reconstruct.stderr}\n")
                        
                        scene_mesh_ply = output_dir / "scene_mesh.ply"
                        if res_reconstruct.returncode == 0 and scene_mesh_ply.exists():
                            texturing_mesh_rel = "scene_mesh.ply"
                            log_file_handle.write(f"ReconstructMesh succeeded: {scene_mesh_ply}\n")
                        else:
                            log_file_handle.write(f"ReconstructMesh failed (code {res_reconstruct.returncode}). Falling back to Poisson mesh.\n")
                    
                    # Fallback: Decimate the Poisson mesh
                    if texturing_mesh_rel is None:
                        log_file_handle.write("\n--- Fallback: Decimating Poisson mesh for texturing ---\n")
                        target_faces = int(os.getenv("RECON_DECIMATE_FACES", "200000"))
                        decimated_mesh_path = output_dir / "meshed-poisson-decimated.ply"
                        try:
                            import trimesh
                            master_mesh = trimesh.load(str(mesh_path))
                            current_faces = len(master_mesh.faces)
                            if current_faces > target_faces:
                                log_file_handle.write(f"Decimating mesh from {current_faces} to {target_faces} faces...\n")
                                decimated_mesh = master_mesh.simplify_quadric_decimation(face_count=target_faces)
                                decimated_mesh.export(str(decimated_mesh_path))
                                texturing_mesh_rel = "meshed-poisson-decimated.ply"
                                log_file_handle.write(f"Decimation complete: {decimated_mesh_path}\n")
                            else:
                                texturing_mesh_rel = os.path.relpath(mesh_path, output_dir)
                                log_file_handle.write(f"Mesh density ({current_faces}) below threshold. Using as-is.\n")
                        except Exception as dec_err:
                            texturing_mesh_rel = os.path.relpath(mesh_path, output_dir)
                            log_file_handle.write(f"Warning: Decimation failed ({dec_err}). Using master mesh.\n")
                    
                    # ── Step 3: TextureMesh ──────────────────────────────────
                    cmd_texture = [
                        str(texture_mesh_bin),
                        "scene.mvs",
                        "-m", texturing_mesh_rel,
                        "-o", "scene_texture.ply",
                        "--resolution-level", res_level
                    ]
                    log_file_handle.write(f"\nRunning TextureMesh (resolution-level {res_level}, mesh: {texturing_mesh_rel}) in {output_dir}:\n  {' '.join(cmd_texture)}\n")
                    
                    res_texture = subprocess.run(
                        cmd_texture, 
                        cwd=str(output_dir),
                        capture_output=True, 
                        text=True, 
                        errors="replace",
                        check=False
                    )
                    log_file_handle.write(res_texture.stdout)
                    if res_texture.stderr:
                        log_file_handle.write(f"\nSTDERR:\n{res_texture.stderr}\n")
                    
                    if res_texture.returncode != 0:
                        raise RuntimeError(f"TextureMesh failed with code {res_texture.returncode}")
                    
                    # ── Step 4: Discover textured output files ────────────────
                    # OpenMVS outputs: scene_texture.ply + scene_texture0.png
                    # (indexed atlas naming; could also be .obj depending on version)
                    textured_mesh = None
                    for candidate in ["scene_texture.ply", "scene_texture.obj"]:
                        p = output_dir / candidate
                        if p.exists() and p.stat().st_size > 0:
                            textured_mesh = p
                            break
                    
                    # Discover texture atlas (indexed naming: scene_texture0.png, scene_texture1.png, ...)
                    texture_atlas = None
                    for candidate in sorted(output_dir.glob("scene_texture*.png")):
                        if candidate.stat().st_size > 0:
                            texture_atlas = candidate
                            break  # Use the first/primary atlas
                    # Also check non-indexed name
                    if texture_atlas is None:
                        p = output_dir / "scene_texture.png"
                        if p.exists() and p.stat().st_size > 0:
                            texture_atlas = p
                    
                    if textured_mesh:
                        mesh_path = textured_mesh
                        texture_path = str(texture_atlas) if texture_atlas else str(output_dir / "dummy_texture.png")
                        log_file_handle.write(f"\nOpenMVS Texturing completed successfully.\n")
                        log_file_handle.write(f"  Textured mesh: {textured_mesh} ({textured_mesh.stat().st_size} bytes)\n")
                        if texture_atlas:
                            log_file_handle.write(f"  Texture atlas: {texture_atlas} ({texture_atlas.stat().st_size} bytes)\n")
                        
                        # Extract real counts from the textured mesh
                        try:
                            import trimesh
                            tmesh = trimesh.load(str(textured_mesh))
                            final_vertex_count = len(tmesh.vertices) if hasattr(tmesh, 'vertices') else 0
                            final_face_count = len(tmesh.faces) if hasattr(tmesh, 'faces') else 0
                            log_file_handle.write(f"  Vertices: {final_vertex_count}, Faces: {final_face_count}\n")
                        except Exception:
                            pass
                    else:
                        log_file_handle.write("\nOpenMVS TextureMesh finished but no textured output was found.\n")
                        
                except Exception as e:
                    with open(log_path, "a", encoding="utf-8") as lf:
                        lf.write(f"\nOpenMVS Stage Error: {str(e)}. Falling back to default COLMAP output.\n")

        return {
            "mesh_path": str(mesh_path),
            "texture_path": texture_path,
            "log_path": str(log_path),
            "vertex_count": final_vertex_count,
            "face_count": final_face_count
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

    def _validate_dense_workspace(self, workspace_path: Path):
        """
        Performs lightweight integrity checks on the COLMAP dense workspace.
        Raises RuntimeError if inconsistencies are found.
        """
        dense_dir = workspace_path / "dense" / "0"
        sparse_dir = dense_dir / "sparse"
        images_dir = dense_dir / "images"
        
        # 1. Check folder existence
        if not dense_dir.exists():
            raise RuntimeError(f"Dense workspace folder missing: {dense_dir}")
        if not sparse_dir.exists():
            raise RuntimeError(f"Sparse reconstruction folder missing: {sparse_dir}")
        if not images_dir.exists():
            raise RuntimeError(f"Dense images folder missing: {images_dir}")

        # 2. Check binary files existence and size
        required_bins = ["cameras.bin", "images.bin", "points3D.bin"]
        for bin_file in required_bins:
            p = sparse_dir / bin_file
            if not p.exists():
                raise RuntimeError(f"Required COLMAP binary missing: {bin_file}")
            if p.stat().st_size == 0:
                raise RuntimeError(f"Required COLMAP binary is empty: {bin_file}")

        # 3. Check fused.ply
        fused_ply = dense_dir / "fused.ply"
        if not fused_ply.exists():
             raise RuntimeError("Dense point cloud (fused.ply) missing.")
        if fused_ply.stat().st_size < 1024: # Minimal sane size for a PLY
             raise RuntimeError(f"Dense point cloud (fused.ply) is suspiciously small ({fused_ply.stat().st_size} bytes).")

        # 4. Image count consistency check (feasilbe via folder listing)
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        if not image_files:
            raise RuntimeError(f"No images found in dense workspace: {images_dir}")
        
        # Note: A full images.bin parser is heavy, but we can verify that the images 
        # folder matches the expected set from input_frames if available.
        # For now, we rely on the existence of the folder and non-zero counts.
        
        return True
