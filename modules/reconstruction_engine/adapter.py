from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
import time
import shutil
import subprocess

import cv2
import numpy as np
import trimesh

from .mesh_selector import MeshSelector
from .failures import InsufficientInputError, RuntimeReconstructionError, InsufficientReconstructionError
from modules.utils.mask_resolution import resolve_mask_path
from modules.operations.settings import settings


class ColmapCommandBuilder:
    """
    Centralized builder for COLMAP commands.
    Targets COLMAP 4.0.3 flag semantics.
    """

    def __init__(self, binary_path: str, use_gpu: bool = True):
        self.bin = binary_path
        self.use_gpu = use_gpu

    def feature_extractor(self, db_path: Path, images_dir: Path, masks_dir: Path, max_size: int) -> List[str]:
        cmd = [
            self.bin, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.mask_path", str(masks_dir),
            "--FeatureExtraction.use_gpu", "1" if self.use_gpu else "0",
            "--FeatureExtraction.max_image_size", str(max_size),
        ]
        return cmd

    def matcher(self, mode: str, db_path: Path) -> List[str]:
        matcher_type = "exhaustive_matcher" if mode == "exhaustive" else "sequential_matcher"
        cmd = [
            self.bin, matcher_type,
            "--database_path", str(db_path),
            # COLMAP 4.0.3 uses FeatureMatching instead of SiftMatching for GPU toggle
            "--FeatureMatching.use_gpu", "1" if self.use_gpu else "0",
        ]
        return cmd

    def mapper(self, db_path: Path, images_dir: Path, output_path: Path) -> List[str]:
        cmd = [
            self.bin, "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(output_path),
            # Optional GPU BA for 4.x
            "--Mapper.ba_use_gpu", "1" if self.use_gpu else "0",
        ]
        return cmd

    def image_undistorter(self, images_dir: Path, input_path: Path, output_path: Path) -> List[str]:
        return [
            self.bin, "image_undistorter",
            "--image_path", str(images_dir),
            "--input_path", str(input_path),
            "--output_path", str(output_path),
            "--output_type", "COLMAP",
        ]

    def patch_match_stereo(self, workspace_path: Path) -> List[str]:
        return [
            self.bin, "patch_match_stereo",
            "--workspace_path", str(workspace_path),
            "--PatchMatchStereo.gpu_index", "0" if self.use_gpu else "-1",
            "--PatchMatchStereo.geom_consistency", "1",
            "--PatchMatchStereo.filter", "1",
        ]

    def stereo_fusion(self, workspace_path: Path, output_path: Path) -> List[str]:
        return [
            self.bin, "stereo_fusion",
            "--workspace_path", str(workspace_path),
            "--output_path", str(output_path),
        ]

    def poisson_mesher(self, input_path: Path, output_path: Path) -> List[str]:
        return [
            self.bin, "poisson_mesher",
            "--input_path", str(input_path),
            "--output_path", str(output_path),
        ]

    def delaunay_mesher(self, workspace_path: Path, output_path: Path) -> List[str]:
        return [
            self.bin, "delaunay_mesher",
            "--input_path", str(workspace_path),
            "--output_path", str(output_path),
        ]

    def model_analyzer(self, model_path: Path) -> List[str]:
        return [
            self.bin, "model_analyzer",
            "--path", str(model_path),
        ]


class ReconstructionAdapter(ABC):
    @abstractmethod
    def run_reconstruction(
        self, 
        input_frames: List[str], 
        output_dir: Path,
        density: float = 1.0,
        enforce_masks: bool = True
    ) -> dict:
        """
        Runs the reconstruction process and returns a dictionary with artifact info.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def engine_type(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_stub(self) -> bool:
        raise NotImplementedError


class COLMAPAdapter(ReconstructionAdapter):
    """
    Product-focused COLMAP reconstruction adapter.

    Improvements over the old version:
    - filters frames by usable mask occupancy
    - keeps explicit COLMAP chain
    - discovers multiple mesh candidates
    - selects best candidate with MeshSelector
    - returns real vertex_count / face_count
    - attempts to discover any texture-like artifact instead of always returning a dummy path
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
        self.mesh_selector = MeshSelector()
        self.builder = ColmapCommandBuilder(self._engine_path, self._use_gpu)

    @property
    def engine_type(self) -> str:
        return "colmap"

    @property
    def is_stub(self) -> bool:
        return False

    def _run_command(self, cmd: List[str], cwd: Path, log_file) -> None:
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

        first_error_line = None
        if process.stdout:
            for line in process.stdout:
                if not first_error_line and ("Failed" in line or "Error" in line or "unrecognised" in line):
                    first_error_line = line.strip()
                log_file.write(line)
                log_file.flush()

        process.wait()
        if process.returncode != 0:
            msg = f"Command failed with exit code {process.returncode}: {' '.join(cmd)}"
            raise RuntimeReconstructionError(msg, output_snippet=first_error_line)

    def _mask_is_usable(self, mask_path: Path) -> bool:
        if not mask_path.exists():
            return False

        mask = self._read_image(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False

        h, w = mask.shape[:2]
        occupancy = float(np.sum(mask > 0) / max(h * w, 1))
        return 0.04 < occupancy < 0.90

    def _read_image(self, image_path: Path, read_flag: int):
        image = cv2.imread(str(image_path), read_flag)
        if image is not None:
            return image

        try:
            image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
            if image_bytes.size == 0:
                return None
            return cv2.imdecode(image_bytes, read_flag)
        except Exception:
            return None

    def _frame_is_usable(self, frame_path: Path) -> bool:
        if not frame_path.exists():
            return False
        if frame_path.stat().st_size <= 0:
            return False

        frame = self._read_image(frame_path, cv2.IMREAD_COLOR)
        return frame is not None and frame.size > 0

    def _prepare_workspace(
        self, 
        input_frames: List[str], 
        output_dir: Path,
        sampling_step: int = 1,
        enforce_masks: bool = True
    ) -> Dict[str, Any]:
        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        accepted_frames = 0
        rejected_missing_mask = 0
        rejected_bad_mask = 0
        rejected_unreadable_frame = 0
        rejected_sampling = 0
        
        match_mode_counts = {"stem": 0, "legacy": 0, "none": 0}

        for i, frame_path in enumerate(input_frames):
            if i % sampling_step != 0:
                rejected_sampling += 1
                continue

            src = Path(frame_path)
            if not src.exists():
                rejected_unreadable_frame += 1
                continue

            if not self._frame_is_usable(src):
                rejected_unreadable_frame += 1
                continue

            mask_src, match_mode = resolve_mask_path(src)
            match_mode_counts[match_mode] += 1
            
            if enforce_masks:
                if mask_src is None:
                    rejected_missing_mask += 1
                    continue

                if not self._mask_is_usable(mask_src):
                    rejected_bad_mask += 1
                    continue

            shutil.copy2(src, images_dir / src.name)
            if mask_src and mask_src.exists():
                shutil.copy2(mask_src, masks_dir / f"{src.name}.png")
            accepted_frames += 1

        return {
            "images_dir": images_dir,
            "masks_dir": masks_dir,
            "accepted_frames": accepted_frames,
            "rejected_missing_mask": rejected_missing_mask,
            "rejected_bad_mask": rejected_bad_mask,
            "rejected_unreadable_frame": rejected_unreadable_frame,
            "rejected_sampling": rejected_sampling,
            "match_mode_counts": match_mode_counts,
        }

    def _validate_dense_workspace(self, workspace_path: Path) -> int:
        dense_dir = workspace_path / "dense"
        fused_ply = dense_dir / "fused.ply"

        if not dense_dir.exists():
            raise RuntimeError(f"Dense workspace folder missing: {dense_dir}")
        if not fused_ply.exists():
            raise RuntimeError("Dense point cloud (fused.ply) missing.")
        
        file_size = fused_ply.stat().st_size
        if file_size < 1024:
            raise RuntimeError(f"Fused point cloud too small ({file_size} bytes).")

        return self._get_ply_point_count(fused_ply)

    def _get_ply_point_count(self, ply_path: Path) -> int:
        """Roughly estimate point count from PLY file size if parsing is too slow."""
        try:
            with open(ply_path, "rb") as f:
                header = ""
                for _ in range(30):
                    line = f.readline().decode("ascii", errors="ignore")
                    header += line
                    if "end_header" in line:
                        break
                
                if "element vertex" in header:
                    for line in header.splitlines():
                        if "element vertex" in line:
                            return int(line.split()[-1])
            return 0
        except Exception:
            return 0

    def _parse_model_stats(self, sparse_dir: Path, log_file) -> Dict[str, int]:
        cmd = self.builder.model_analyzer(sparse_dir)
        try:
            # Using Popen-style capture for maximum compatibility with .bat files on Windows
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(sparse_dir.parent.parent or "."),
                shell=True if self._engine_path.lower().endswith(".bat") else False
            )
            stdout, _ = process.communicate(timeout=30)
            
            # Log for debugging if empty
            if not stdout.strip():
                log_file.write(f"\nWarning: model_analyzer returned empty output for {sparse_dir.name}\n")
            
            stats = {"registered_images": 0, "points_3d": 0}
            for line in stdout.splitlines():
                if "Registered images" in line:
                    try:
                        stats["registered_images"] = int(line.split()[-1])
                    except (ValueError, IndexError):
                        pass
                if "Points3D" in line and "observations" not in line:
                    try:
                        # Line format: "Points3D : 1234"
                        parts = line.split(":")
                        if len(parts) > 1:
                            stats["points_3d"] = int(parts[1].strip())
                    except (ValueError, IndexError):
                        pass
            
            return stats
        except Exception as e:
            log_file.write(f"\nWarning: model_analyzer failed on {sparse_dir.name}: {e}\n")
            return {"registered_images": 0, "points_3d": 0}
    def _select_best_sparse_model(self, sparse_dir: Path, log_file) -> Optional[Dict[str, Any]]:
        """
        Iterates through all sub-models in the sparse directory and selects the best one.
        Criteria:
        1. Highest registered_images
        2. Tie-break: highest sparse_points
        3. Tie-break: lowest index (stable)
        """
        if not sparse_dir.exists():
            return None

        candidates = []
        # Sub-models are usually named '0', '1', '2', etc.
        for item in sorted(sparse_dir.glob("*"), key=lambda x: x.name):
            if not item.is_dir():
                continue
            
            stats = self._parse_model_stats(item, log_file)
            if stats["registered_images"] > 0:
                candidates.append({
                    "path": item,
                    "registered_images": stats["registered_images"],
                    "points_3d": stats["points_3d"]
                })
        
        if not candidates:
            return None

        # Apply multi-level sorting
        # Sort by images desc, points desc, then path name asc (for stability)
        candidates.sort(
            key=lambda x: (-x["registered_images"], -x["points_3d"], x["path"].name)
        )
        
        best = candidates[0]
        
        log_file.write("\n--- Sparse Model Candidates ---\n")
        for c in candidates:
            mark = " (SELECTED)" if c == best else ""
            log_file.write(f"Model {c['path'].name}: {c['registered_images']} images, {c['points_3d']} points{mark}\n")
        log_file.write("-" * 31 + "\n")
        
        return best

    def _is_valid_mesh_candidate(self, mesh_path: Path) -> bool:
        if not mesh_path.exists() or mesh_path.stat().st_size <= 0:
            return False

        try:
            mesh = trimesh.load(str(mesh_path))
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            return isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0 and len(mesh.faces) > 0
        except Exception:
            return False

    def _discover_mesh_candidates(self, dense_dir: Path) -> List[str]:
        candidates: List[str] = []

        # preferred mesh outputs
        preferred = [
            "meshed-poisson.ply",
            "meshed-delaunay.ply",
            "delaunay_mesh.ply",
            "poisson_mesh.ply",
        ]

        for name in preferred:
            p = dense_dir / name
            if self._is_valid_mesh_candidate(p):
                candidates.append(str(p))

        # any other non-fused mesh-like ply
        for p in dense_dir.glob("*.ply"):
            if "fused" in p.name.lower():
                continue
            if str(p) not in candidates and self._is_valid_mesh_candidate(p):
                candidates.append(str(p))

        return candidates

    def _discover_texture_candidate(self, workspace_dir: Path) -> str:
        """
        COLMAP poisson/delaunay outputs usually won't give a real textured mesh.
        But if some downstream step or future extension dumps a texture image,
        we can pick it up here. Otherwise return a known non-existing sentinel path.
        """
        search_roots = [
            workspace_dir,
            workspace_dir / "dense",
        ]

        exts = ["*.png", "*.jpg", "*.jpeg"]
        for root in search_roots:
            if not root.exists():
                continue
            for ext in exts:
                for p in root.glob(ext):
                    # skip obvious non-texture inputs if needed
                    if "texture" in p.name.lower() or "albedo" in p.name.lower():
                        return str(p)

        return str(workspace_dir / "_no_texture.png")

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

    def run_reconstruction(
        self, 
        input_frames: List[str], 
        output_dir: Path,
        density: float = 1.0,
        enforce_masks: bool = True
    ) -> dict:
        if not self._engine_path:
            raise RuntimeError("Reconstruction engine path (RECON_ENGINE_PATH) not configured.")

        sampling_step = max(1, int(1.0 / density)) if density < 1.0 else 1
        prep = self._prepare_workspace(input_frames, output_dir, sampling_step=sampling_step, enforce_masks=enforce_masks)
        images_dir: Path = prep["images_dir"]
        masks_dir: Path = prep["masks_dir"]

        if prep["accepted_frames"] < 3:
            counts = prep["match_mode_counts"]
            raise InsufficientInputError(
                "Not enough usable masked frames for reconstruction. "
                f"accepted={prep['accepted_frames']} "
                f"unreadable={prep['rejected_unreadable_frame']} "
                f"missing_mask={prep['rejected_missing_mask']} "
                f"bad_mask={prep['rejected_bad_mask']} "
                f"modes(stem={counts['stem']}, legacy={counts['legacy']}, none={counts['none']})"
            )

        log_path = output_dir / "reconstruction.log"
        with open(log_path, "w", encoding="utf-8") as log_file:
            counts = prep["match_mode_counts"]
            log_file.write(
                f"Workspace prepared. accepted={prep['accepted_frames']} "
                f"unreadable={prep['rejected_unreadable_frame']} "
                f"missing_mask={prep['rejected_missing_mask']} "
                f"bad_mask={prep['rejected_bad_mask']} "
                f"modes(stem={counts['stem']}, legacy={counts['legacy']}, none={counts['none']})\n"
            )

            try:
                db_path = output_dir / "database.db"

                # Feature extraction
                cmd_extract = self.builder.feature_extractor(
                    db_path, images_dir, masks_dir, self._max_image_size
                )
                self._run_command(cmd_extract, output_dir, log_file)

                # Matching
                cmd_match = self.builder.matcher(self._matcher, db_path)
                self._run_command(cmd_match, output_dir, log_file)

                # Sparse map
                sparse_dir = output_dir / "sparse"
                sparse_dir.mkdir(exist_ok=True)
                cmd_map = self.builder.mapper(db_path, images_dir, sparse_dir)
                self._run_command(cmd_map, output_dir, log_file)

                # Stage gating
                best_model = self._select_best_sparse_model(sparse_dir, log_file)
                if not best_model:
                    raise RuntimeReconstructionError("Sparse reconstruction finished but no valid sub-model found.")
                
                model_path = best_model["path"]
                registered_images = best_model["registered_images"]
                sparse_points = best_model["points_3d"]
                selected_model_name = model_path.name
                
                if registered_images < 5 or sparse_points < 100:
                    raise InsufficientReconstructionError(
                        f"Best sparse model too small for densification. model={selected_model_name}, images={registered_images}, points={sparse_points}"
                    )

                # Dense prep
                dense_dir = output_dir / "dense"
                dense_dir.mkdir(exist_ok=True)
                cmd_undistort = self.builder.image_undistorter(
                    images_dir, model_path, dense_dir
                )
                self._run_command(cmd_undistort, output_dir, log_file)

                # Dense depth
                cmd_stereo = self.builder.patch_match_stereo(dense_dir)
                self._run_command(cmd_stereo, output_dir, log_file)

                # Fusion
                cmd_fuse = self.builder.stereo_fusion(dense_dir, dense_dir / "fused.ply")
                self._run_command(cmd_fuse, output_dir, log_file)
                
                fused_points = self._validate_dense_workspace(output_dir)
                log_file.write(f"Dense fusion successful. Fused points: {fused_points}\n")

                # Meshing: try poisson, then delaunay
                poisson_ok = False
                try:
                    cmd_mesh = self.builder.poisson_mesher(
                        dense_dir / "fused.ply", dense_dir / "meshed-poisson.ply"
                    )
                    self._run_command(cmd_mesh, output_dir, log_file)
                    poisson_ok = True
                    mesher_used = "poisson"
                except Exception as poisson_err:
                    log_file.write(f"\nPoisson mesher failed: {poisson_err}\n")

                if not poisson_ok:
                    try:
                        # FIX: delaunay_mesher expects dense root, not fused.ply
                        cmd_mesh = self.builder.delaunay_mesher(
                            dense_dir, dense_dir / "meshed-delaunay.ply"
                        )
                        self._run_command(cmd_mesh, output_dir, log_file)
                        mesher_used = "delaunay"
                    except Exception as delaunay_err:
                        log_file.write(f"\nDelaunay mesher failed: {delaunay_err}\n")
                        mesher_used = "failed"

            except Exception as e:
                log_file.write(f"\nCRITICAL FAILURE: {str(e)}\n")
                raise RuntimeError(f"COLMAP chain failed: {str(e)}")

        self._validate_dense_workspace(output_dir)

        dense_dir = output_dir / "dense"
        candidates = self._discover_mesh_candidates(dense_dir)
        if not candidates:
            raise RuntimeError(f"COLMAP completed but no usable mesh artifacts found in {dense_dir}")

        selected_mesh = self.mesh_selector.select_best_mesh(candidates) or candidates[0]
        selected_stats = self._mesh_stats(selected_mesh)
        texture_candidate = self._discover_texture_candidate(output_dir)

        return {
            "mesh_path": str(selected_mesh),
            "texture_path": texture_candidate,
            "log_path": str(log_path),
            "vertex_count": selected_stats["vertex_count"],
            "face_count": selected_stats["face_count"],
            "registered_images": registered_images,
            "sparse_points": sparse_points,
            "dense_points_fused": fused_points,
            "mesher_used": mesher_used if 'mesher_used' in locals() else "unknown",
            "selected_sparse_model": selected_model_name if 'selected_model_name' in locals() else "none"
        }


class SimulatedAdapter(ReconstructionAdapter):
    @property
    def engine_type(self) -> str:
        return "simulated"

    @property
    def is_stub(self) -> bool:
        return True

    def run_reconstruction(
        self, 
        input_frames: List[str], 
        output_dir: Path,
        density: float = 1.0,
        enforce_masks: bool = True
    ) -> dict:
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
