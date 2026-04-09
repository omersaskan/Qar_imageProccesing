from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
import time
import shutil
import subprocess
import json

import cv2
import trimesh

from .mesh_selector import MeshSelector


class ReconstructionAdapter(ABC):
    @abstractmethod
    def run_reconstruction(self, input_frames: List[str], output_dir: Path) -> dict:
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
    Explicit COLMAP chain with:
    - frame + mask copying
    - lightweight mask-quality filtering
    - artifact discovery
    - best mesh selection via MeshSelector
    """

    def __init__(self, engine_path: Optional[str] = None):
        self._engine_path = engine_path or os.getenv("RECON_ENGINE_PATH")

        if not self._engine_path:
            well_known = [
                r"C:\colmap\colmap\COLMAP.bat",
                r"C:\colmap\COLMAP.bat",
                r"C:\colmap\colmap.exe",
            ]
            for p in well_known:
                if os.path.exists(p):
                    self._engine_path = p
                    break

        self._use_gpu = os.getenv("RECON_USE_GPU", "true").lower() == "true"
        self._max_image_size = int(os.getenv("RECON_MAX_IMAGE_SIZE", "2000"))
        self._matcher = os.getenv("RECON_MATCHER", "exhaustive").lower()  # exhaustive | sequential
        self._input_mode = os.getenv("RECON_INPUT_MODE", "full").lower()   # full only for now

        self.mesh_selector = MeshSelector()

    @property
    def engine_type(self) -> str:
        return "colmap"

    @property
    def is_stub(self) -> bool:
        return False

    def _run_command(self, cmd: List[str], cwd: Path, log_file):
        log_file.write(f"\n--- Running: {' '.join(cmd)} ---\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(cwd),
        )

        if process.stdout:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(cmd)}")

    def _mask_is_usable(self, mask_path: Path) -> bool:
        if not mask_path.exists():
            return False

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False

        h, w = mask.shape[:2]
        occupancy = float(np.sum(mask > 0) / max(h * w, 1))
        return 0.04 < occupancy < 0.90

    def _prepare_workspace(self, input_frames: List[str], output_dir: Path) -> Dict[str, Any]:
        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        accepted = 0
        rejected_missing_mask = 0
        rejected_bad_mask = 0

        for frame_path in input_frames:
            src = Path(frame_path)
            if not src.exists():
                continue

            mask_src = src.parent / "masks" / f"{src.name}.png"

            if not mask_src.exists():
                rejected_missing_mask += 1
                continue

            if not self._mask_is_usable(mask_src):
                rejected_bad_mask += 1
                continue

            shutil.copy2(src, images_dir / src.name)
            shutil.copy2(mask_src, masks_dir / f"{src.name}.png")
            accepted += 1

        return {
            "images_dir": images_dir,
            "masks_dir": masks_dir,
            "accepted_frames": accepted,
            "rejected_missing_mask": rejected_missing_mask,
            "rejected_bad_mask": rejected_bad_mask,
        }

    def _discover_candidates(self, dense_dir: Path) -> List[str]:
        candidates: List[str] = []

        # High-priority mesh outputs
        for name in ["meshed-poisson.ply", "meshed-delaunay.ply", "delaunay_mesh.ply", "poisson_mesh.ply"]:
            p = dense_dir / name
            if p.exists():
                candidates.append(str(p))

        # Fallback mesh-ish PLYs
        for p in dense_dir.glob("*.ply"):
            if str(p) not in candidates and "fused" not in p.name.lower():
                candidates.append(str(p))

        # Last-resort point cloud fallback
        fused = dense_dir / "fused.ply"
        if fused.exists() and str(fused) not in candidates:
            candidates.append(str(fused))

        return candidates

    def _mesh_stats(self, mesh_path: str) -> Dict[str, int]:
        try:
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            return {
                "vertex_count": int(len(mesh.vertices)) if hasattr(mesh, "vertices") else 0,
                "face_count": int(len(mesh.faces)) if hasattr(mesh, "faces") else 0,
            }
        except Exception:
            return {"vertex_count": 0, "face_count": 0}

    def _validate_dense_workspace(self, workspace_path: Path) -> bool:
        dense_dir = workspace_path / "dense"
        fused_ply = dense_dir / "fused.ply"

        if not dense_dir.exists():
            raise RuntimeError(f"Dense workspace folder missing: {dense_dir}")
        if not fused_ply.exists():
            raise RuntimeError("Dense point cloud (fused.ply) missing.")
        if fused_ply.stat().st_size < 1024:
            raise RuntimeError(f"Fused point cloud too small ({fused_ply.stat().st_size} bytes).")

        return True

    def run_reconstruction(self, input_frames: List[str], output_dir: Path) -> dict:
        if not self._engine_path:
            raise RuntimeError("Reconstruction engine path (RECON_ENGINE_PATH) not configured.")

        prep = self._prepare_workspace(input_frames, output_dir)
        images_dir: Path = prep["images_dir"]
        masks_dir: Path = prep["masks_dir"]

        if prep["accepted_frames"] < 3:
            raise RuntimeError(
                f"Not enough usable masked frames for reconstruction. "
                f"accepted={prep['accepted_frames']} "
                f"missing_mask={prep['rejected_missing_mask']} "
                f"bad_mask={prep['rejected_bad_mask']}"
            )

        log_path = output_dir / "reconstruction.log"
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(
                f"Workspace prepared. accepted={prep['accepted_frames']} "
                f"missing_mask={prep['rejected_missing_mask']} "
                f"bad_mask={prep['rejected_bad_mask']}\n"
            )

            try:
                db_path = output_dir / "database.db"

                cmd_extract = [
                    self._engine_path,
                    "feature_extractor",
                    "--database_path", str(db_path),
                    "--image_path", str(images_dir),
                    "--ImageReader.mask_path", str(masks_dir),
                    "--SiftExtraction.use_gpu", "1" if self._use_gpu else "0",
                    "--SiftExtraction.max_image_size", str(self._max_image_size),
                ]
                self._run_command(cmd_extract, output_dir, log_file)

                if self._matcher == "sequential":
                    cmd_match = [
                        self._engine_path,
                        "sequential_matcher",
                        "--database_path", str(db_path),
                        "--SiftMatching.use_gpu", "1" if self._use_gpu else "0",
                    ]
                else:
                    cmd_match = [
                        self._engine_path,
                        "exhaustive_matcher",
                        "--database_path", str(db_path),
                        "--SiftMatching.use_gpu", "1" if self._use_gpu else "0",
                    ]
                self._run_command(cmd_match, output_dir, log_file)

                sparse_dir = output_dir / "sparse"
                sparse_dir.mkdir(exist_ok=True)

                cmd_map = [
                    self._engine_path,
                    "mapper",
                    "--database_path", str(db_path),
                    "--image_path", str(images_dir),
                    "--output_path", str(sparse_dir),
                ]
                self._run_command(cmd_map, output_dir, log_file)

                dense_dir = output_dir / "dense"
                dense_dir.mkdir(exist_ok=True)

                cmd_undistort = [
                    self._engine_path,
                    "image_undistorter",
                    "--image_path", str(images_dir),
                    "--input_path", str(sparse_dir / "0"),
                    "--output_path", str(dense_dir),
                    "--output_type", "COLMAP",
                ]
                self._run_command(cmd_undistort, output_dir, log_file)

                cmd_stereo = [
                    self._engine_path,
                    "patch_match_stereo",
                    "--workspace_path", str(dense_dir),
                    "--PatchMatchStereo.gpu_index", "0" if self._use_gpu else "-1",
                ]
                self._run_command(cmd_stereo, output_dir, log_file)

                cmd_fuse = [
                    self._engine_path,
                    "stereo_fusion",
                    "--workspace_path", str(dense_dir),
                    "--output_path", str(dense_dir / "fused.ply"),
                ]
                self._run_command(cmd_fuse, output_dir, log_file)

                # try poisson mesher first
                try:
                    cmd_mesh = [
                        self._engine_path,
                        "poisson_mesher",
                        "--input_path", str(dense_dir / "fused.ply"),
                        "--output_path", str(dense_dir / "meshed-poisson.ply"),
                    ]
                    self._run_command(cmd_mesh, output_dir, log_file)
                except Exception as poisson_err:
                    log_file.write(f"\nPoisson mesher failed: {poisson_err}\n")
                    try:
                        cmd_mesh = [
                            self._engine_path,
                            "delaunay_mesher",
                            "--input_path", str(dense_dir / "fused.ply"),
                            "--output_path", str(dense_dir / "meshed-delaunay.ply"),
                        ]
                        self._run_command(cmd_mesh, output_dir, log_file)
                    except Exception as delaunay_err:
                        log_file.write(f"\nDelaunay mesher failed: {delaunay_err}\n")

            except Exception as e:
                log_file.write(f"\nCRITICAL FAILURE: {str(e)}\n")
                raise RuntimeError(f"COLMAP chain failed: {str(e)}")

        self._validate_dense_workspace(output_dir)

        dense_dir = output_dir / "dense"
        candidates = self._discover_candidates(dense_dir)
        if not candidates:
            raise RuntimeError(f"COLMAP completed but no reconstruction artifacts found in {dense_dir}")

        selected_mesh = self.mesh_selector.select_best_mesh(candidates) or candidates[0]
        stats = self._mesh_stats(selected_mesh)

        return {
            "mesh_path": str(selected_mesh),
            "texture_path": str(output_dir / "dummy_texture.png"),
            "log_path": str(log_path),
            "vertex_count": stats["vertex_count"],
            "face_count": stats["face_count"],
        }


class SimulatedAdapter(ReconstructionAdapter):
    @property
    def engine_type(self) -> str:
        return "simulated"

    @property
    def is_stub(self) -> bool:
        return True

    def run_reconstruction(self, input_frames: List[str], output_dir: Path) -> dict:
        time.sleep(0.5)

        mesh_path = output_dir / "raw_mesh.obj"
        texture_path = output_dir / "raw_texture.png"
        log_path = output_dir / "logs/reconstruction.log"

        log_path.parent.mkdir(parents=True, exist_ok=True)

        obj_content = (
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 0.0 1.0 0.0\n"
            "f 1 2 3\n"
        )
        with open(mesh_path, "w", encoding="utf-8") as f:
            f.write(obj_content)

        with open(texture_path, "wb") as f:
            f.write(
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR"
                b"\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x02\x00\x00\x00\x90wS\xde"
                b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff? \x00\x05\xfe\x02\xfe\xdcD\x05\x13"
                b"\x00\x00\x00\x00IEND\xaeB`\x82"
            )

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Simulated reconstruction successful.\n")

        return {
            "mesh_path": str(mesh_path),
            "texture_path": str(texture_path),
            "log_path": str(log_path),
            "vertex_count": 3,
            "face_count": 1,
        }