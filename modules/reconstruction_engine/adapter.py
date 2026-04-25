import re
import json
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
from .failures import (
    ReconstructionError,
    InsufficientInputError,
    RuntimeReconstructionError,
    InsufficientReconstructionError,
    DenseMaskAlignmentError,
)
from modules.utils.mask_resolution import resolve_mask_path
from modules.operations.settings import settings


class ColmapCapabilityManager:
    """
    Handles runtime discovery and caching of COLMAP capabilities.
    Prevents "unrecognised option" errors by probing -h output.
    """

    _cache: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_capabilities(cls, binary_path: str) -> Dict[str, Any]:
        if binary_path in cls._cache:
            return cls._cache[binary_path]

        caps = {
            "extraction_prefix": "FeatureExtraction",
            "matching_prefix": "FeatureMatching",
            "has_ba_gpu": True,
            "has_cuda": False,
        }

        if not binary_path or not os.path.exists(binary_path):
            return caps

        try:
            # Probe feature_extractor
            res = subprocess.run(
                [binary_path, "feature_extractor", "-h"],
                capture_output=True,
                text=True,
                check=False,
            )
            output = res.stdout + res.stderr
            # Specifically check if the "Sift" variant of the GPU flag exists
            if "--SiftExtraction.use_gpu" in output:
                caps["extraction_prefix"] = "SiftExtraction"
            
            # CUDA detection: check if use_gpu exists in help
            if "use_gpu" in output.lower():
                caps["has_cuda"] = True

            # Probe exhaustive_matcher
            res = subprocess.run(
                [binary_path, "exhaustive_matcher", "-h"],
                capture_output=True,
                text=True,
                check=False,
            )
            output = res.stdout + res.stderr
            if "--SiftMatching.use_gpu" in output:
                caps["matching_prefix"] = "SiftMatching"

            # Probe mapper
            res = subprocess.run(
                [binary_path, "mapper", "-h"],
                capture_output=True,
                text=True,
                check=False,
            )
            output = res.stdout + res.stderr
            if "ba_use_gpu" not in output:
                caps["has_ba_gpu"] = False

        except Exception:
            # Fallback to defaults if probing fails
            pass

        cls._cache[binary_path] = caps
        return caps


class ColmapCommandBuilder:
    """
    Centralized builder for COLMAP commands.
    Now capability-aware to support versions from 3.6 to 4.0+.
    """

    def __init__(self, binary_path: str, use_gpu: bool = True):
        self.bin = binary_path
        self._requested_gpu = use_gpu
        self.caps = ColmapCapabilityManager.get_capabilities(binary_path)
        
        # Override GPU request if the build doesn't support it
        self.use_gpu = use_gpu and self.caps["has_cuda"]

    def feature_extractor(
        self,
        db_path: Path,
        images_dir: Path,
        masks_dir: Optional[Path],
        max_size: int,
    ) -> List[str]:
        prefix = self.caps["extraction_prefix"]
        cmd = [
            self.bin,
            "feature_extractor",
            "--database_path",
            str(db_path),
            "--image_path",
            str(images_dir),
            f"--{prefix}.use_gpu",
            "1" if self.use_gpu else "0",
            f"--{prefix}.max_image_size",
            str(max_size),
        ]

        if masks_dir is not None:
            cmd += ["--ImageReader.mask_path", str(masks_dir)]

        return cmd

    def matcher(self, mode: str, db_path: Path) -> List[str]:
        matcher_type = "exhaustive_matcher" if mode == "exhaustive" else "sequential_matcher"
        prefix = self.caps["matching_prefix"]
        cmd = [
            self.bin,
            matcher_type,
            "--database_path",
            str(db_path),
            f"--{prefix}.use_gpu",
            "1" if self.use_gpu else "0",
        ]
        return cmd

    def mapper(self, db_path: Path, images_dir: Path, output_path: Path) -> List[str]:
        cmd = [
            self.bin,
            "mapper",
            "--database_path",
            str(db_path),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(output_path),
        ]
        
        if self.caps["has_ba_gpu"]:
            cmd += ["--Mapper.ba_use_gpu", "1" if self.use_gpu else "0"]
            
        return cmd

    def image_undistorter(self, images_dir: Path, input_path: Path, output_path: Path) -> List[str]:
        return [
            self.bin,
            "image_undistorter",
            "--image_path",
            str(images_dir),
            "--input_path",
            str(input_path),
            "--output_path",
            str(output_path),
            "--output_type",
            "COLMAP",
        ]

    def patch_match_stereo(self, workspace_path: Path) -> List[str]:
        cmd = [
            self.bin,
            "patch_match_stereo",
            "--workspace_path",
            str(workspace_path),
            "--PatchMatchStereo.geom_consistency",
            "1",
            "--PatchMatchStereo.filter",
            "1",
        ]
        
        # GPU index -1 means CPU mode in COLMAP
        if self.use_gpu:
            cmd += ["--PatchMatchStereo.gpu_index", "0"]
        else:
            cmd += ["--PatchMatchStereo.gpu_index", "-1"]
            
        return cmd

    def stereo_fusion(self, workspace_path: Path, output_path: Path, mask_path: Optional[Path] = None) -> List[str]:
        cmd = [
            self.bin,
            "stereo_fusion",
            "--workspace_path",
            str(workspace_path),
            "--output_path",
            str(output_path),
            "--StereoFusion.min_num_pixels",
            "2",
        ]
        if mask_path is not None:
            cmd += ["--StereoFusion.mask_path", str(mask_path)]
        return cmd

    def poisson_mesher(self, input_path: Path, output_path: Path) -> List[str]:
        return [
            self.bin,
            "poisson_mesher",
            "--input_path",
            str(input_path),
            "--output_path",
            str(output_path),
        ]

    def delaunay_mesher(self, workspace_path: Path, output_path: Path) -> List[str]:
        return [
            self.bin,
            "delaunay_mesher",
            "--input_path",
            str(workspace_path),
            "--output_path",
            str(output_path),
        ]

    def model_analyzer(self, model_path: Path) -> List[str]:
        return [
            self.bin,
            "model_analyzer",
            "--path",
            str(model_path),
        ]


class OpenMVSCommandBuilder:
    """
    Builder for OpenMVS pipeline commands.
    """

    def __init__(self, bin_path: str):
        self.bin = Path(bin_path)

    def _get_bin(self, name: str) -> str:
        p = self.bin / name
        if p.with_suffix(".exe").exists():
            return str(p.with_suffix(".exe"))
        return str(p)

    def interface_colmap(self, workspace_path: Path, output_mvs: Path) -> List[str]:
        return [
            self._get_bin("InterfaceCOLMAP"),
            "-i",
            str(workspace_path),
            "-o",
            str(output_mvs),
            "--working-folder",
            str(workspace_path),
        ]

    def densify_point_cloud(self, input_mvs: Path, output_mvs: Path) -> List[str]:
        return [
            self._get_bin("DensifyPointCloud"),
            "-i",
            str(input_mvs),
            "-o",
            str(output_mvs),
            "--working-folder",
            str(input_mvs.parent),
            "--resolution-level",
            "1",
            "--number-views",
            "0",
            "--max-threads",
            "0",
        ]

    def reconstruct_mesh(self, input_mvs: Path, output_mesh_ply: Path) -> List[str]:
        return [
            self._get_bin("ReconstructMesh"),
            "-i",
            str(input_mvs),
            "-o",
            str(output_mesh_ply),
            "--working-folder",
            str(input_mvs.parent),
        ]

    def refine_mesh(self, input_mvs: Path, output_mvs: Path) -> List[str]:
        return [
            self._get_bin("RefineMesh"),
            "-i",
            str(input_mvs),
            "-o",
            str(output_mvs),
            "--resolution-level",
            "1",
            "--max-iterations",
            "5",
        ]

    def texture_mesh(self, input_scene_mvs: Path, input_mesh_ply: Path, output_obj: Path) -> List[str]:
        return [
            self._get_bin("TextureMesh"),
            "-i",
            str(input_scene_mvs),
            "--mesh-file",
            str(input_mesh_ply),
            "-o",
            str(output_obj),
            "--export-type",
            "obj",
            "--working-folder",
            str(input_scene_mvs.parent),
            "--resolution-level",
            "0",
        ]


class ReconstructionAdapter(ABC):
    @abstractmethod
    def run_reconstruction(
        self,
        input_frames: List[str],
        output_dir: Path,
        density: float = 1.0,
        enforce_masks: bool = True,
    ) -> dict:
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

    Key fixes in this version:
    - mask_path is only passed to COLMAP when masks are actually available
    - copied_mask_count is tracked explicitly
    - unmasked fallback is now truly unmasked
    - logging makes the mask decision visible in reconstruction.log
    """

    def __init__(self, engine_path: Optional[str] = None):
        self._engine_path = engine_path or os.getenv("RECON_ENGINE_PATH")

        if not self._engine_path:
            well_known = [
                r"C:\colmap\colmap\COLMAP.bat",
                r"C:\colmap\COLMAP.bat",
                r"C:\colmap\colmap.exe",
                "/usr/local/bin/colmap",
                "/usr/bin/colmap",
            ]
            for p in well_known:
                if os.path.exists(p):
                    self._engine_path = p
                    break

        self._use_gpu = os.getenv("RECON_USE_GPU", "true").lower() == "true"
        self._max_image_size = int(os.getenv("RECON_MAX_IMAGE_SIZE", "2000"))
        self._matcher = os.getenv("RECON_MATCHER", "exhaustive").lower()
        self.mesh_selector = MeshSelector()
        self.builder = ColmapCommandBuilder(self._engine_path, self._use_gpu)

    @property
    def engine_type(self) -> str:
        return "colmap"

    @property
    def is_stub(self) -> bool:
        return False

    def _run_command(self, cmd: List[str], cwd: Path, log_file, timeout: Optional[int] = None) -> None:
        log_file.write(f"\n--- Running: {' '.join(cmd)} (timeout={timeout}s) ---\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd),
        )

        first_error_line = None
        try:
            # communicate() handles the timeout correctly and avoids blocking issues
            # though it buffers output, for these specific meshing steps it is acceptable.
            stdout, _ = process.communicate(timeout=timeout)
            if stdout:
                for line in stdout.splitlines():
                    if not first_error_line and (
                        "Failed" in line or "Error" in line or "unrecognised" in line
                    ):
                        first_error_line = line.strip()
                    log_file.write(line + "\n")
                log_file.flush()

        except subprocess.TimeoutExpired:
            process.kill()
            stdout, _ = process.communicate() # Drain any remaining output
            if stdout:
                log_file.write(stdout)
            msg = f"Command timed out after {timeout}s: {' '.join(cmd)}"
            log_file.write(f"\nERROR: {msg}\n")
            raise RuntimeReconstructionError(msg, output_snippet="Timeout reached")

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
        enforce_masks: bool = True,
    ) -> Dict[str, Any]:
        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        accepted_frames = 0
        copied_mask_count = 0
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

            # Copy mask only if it exists. Naming here intentionally matches
            # COLMAP's expectation: frame_0000.jpg -> frame_0000.jpg.png
            if mask_src and mask_src.exists():
                shutil.copy2(mask_src, masks_dir / f"{src.name}.png")
                copied_mask_count += 1

            accepted_frames += 1

        return {
            "images_dir": images_dir,
            "masks_dir": masks_dir,
            "accepted_frames": accepted_frames,
            "copied_mask_count": copied_mask_count,
            "rejected_missing_mask": rejected_missing_mask,
            "rejected_bad_mask": rejected_bad_mask,
            "rejected_unreadable_frame": rejected_unreadable_frame,
            "rejected_sampling": rejected_sampling,
            "match_mode_counts": match_mode_counts,
        }

    def _resolve_effective_masks_dir(
        self,
        prep: Dict[str, Any],
        enforce_masks: bool,
        min_required_frames: int,
    ) -> Optional[Path]:
        accepted_frames = int(prep["accepted_frames"])
        copied_mask_count = int(prep["copied_mask_count"])

        if accepted_frames < min_required_frames:
            counts = prep["match_mode_counts"]
            raise InsufficientInputError(
                "Not enough usable frames for reconstruction. "
                f"accepted={accepted_frames} "
                f"copied_masks={copied_mask_count} "
                f"unreadable={prep['rejected_unreadable_frame']} "
                f"missing_mask={prep['rejected_missing_mask']} "
                f"bad_mask={prep['rejected_bad_mask']} "
                f"modes(stem={counts['stem']}, legacy={counts['legacy']}, none={counts['none']})"
            )

        # If masks are enforced, every accepted image must have a copied mask.
        if enforce_masks:
            if copied_mask_count != accepted_frames:
                raise InsufficientInputError(
                    "Mask enforcement requested, but not all accepted frames have copied masks. "
                    f"accepted={accepted_frames}, copied_masks={copied_mask_count}"
                )
            return prep["masks_dir"]

        # Unmasked fallback: do NOT pass a mask directory to COLMAP.
        return prep["masks_dir"] if copied_mask_count == accepted_frames and copied_mask_count > 0 else None

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

    def _parse_analyzer_output(self, output: str) -> Dict[str, int]:
        stats = {"registered_images": 0, "points_3d": 0}

        for line in output.splitlines():
            reg_match = re.search(r"Registered\s+images\s*:\s*(\d+)", line)
            if reg_match:
                stats["registered_images"] = int(reg_match.group(1))
                continue

            if "observations" not in line:
                points_match = re.search(r"Points(?:3D)?\s*:\s*(\d+)", line)
                if points_match:
                    stats["points_3d"] = int(points_match.group(1))

        return stats

    def _parse_model_stats(self, sparse_dir: Path, log_file) -> Dict[str, int]:
        cmd = self.builder.model_analyzer(sparse_dir)
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(sparse_dir.parent.parent or "."),
                shell=True if self._engine_path and self._engine_path.lower().endswith(".bat") else False,
            )
            stdout, _ = process.communicate(timeout=30)

            if not stdout.strip():
                log_file.write(f"\nWarning: model_analyzer returned empty output for {sparse_dir.name}\n")

            return self._parse_analyzer_output(stdout)

        except Exception as e:
            log_file.write(f"\nWarning: model_analyzer failed on {sparse_dir.name}: {e}\n")
            return {"registered_images": 0, "points_3d": 0}

    def _select_best_sparse_model(self, sparse_dir: Path, log_file) -> Optional[Dict[str, Any]]:
        if not sparse_dir.exists():
            return None

        candidates = []
        for item in sorted(sparse_dir.glob("*"), key=lambda x: x.name):
            if not item.is_dir():
                continue

            stats = self._parse_model_stats(item, log_file)
            if stats["registered_images"] > 0:
                candidates.append(
                    {
                        "path": item,
                        "registered_images": stats["registered_images"],
                        "points_3d": stats["points_3d"],
                    }
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: (-x["registered_images"], -x["points_3d"], x["path"].name)
        )

        best = candidates[0]

        log_file.write("\n--- Sparse Model Candidates ---\n")
        for c in candidates:
            mark = " (SELECTED)" if c == best else ""
            log_file.write(
                f"Model {c['path'].name}: {c['registered_images']} images, "
                f"{c['points_3d']} points{mark}\n"
            )
        log_file.write("-" * 31 + "\n")

        return best

    def _is_valid_mesh_candidate(self, mesh_path: Path) -> bool:
        if not mesh_path.exists() or mesh_path.stat().st_size <= 0:
            return False

        try:
            mesh = trimesh.load(str(mesh_path))
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            return (
                isinstance(mesh, trimesh.Trimesh)
                and len(mesh.vertices) > 0
                and len(mesh.faces) > 0
            )
        except Exception:
            return False

    def _discover_mesh_candidates(self, dense_dir: Path) -> List[str]:
        candidates: List[str] = []

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

        for p in dense_dir.glob("*.ply"):
            if "fused" in p.name.lower():
                continue
            if str(p) not in candidates and self._is_valid_mesh_candidate(p):
                candidates.append(str(p))

        return candidates

    def _discover_texture_candidate(self, workspace_dir: Path) -> str:
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
        enforce_masks: bool = True,
    ) -> dict:
        if not self._engine_path:
            raise RuntimeError("Reconstruction engine path (RECON_ENGINE_PATH) not configured.")

        sampling_step = max(1, int(1.0 / density)) if density < 1.0 else 1
        prep = self._prepare_workspace(
            input_frames,
            output_dir,
            sampling_step=sampling_step,
            enforce_masks=enforce_masks,
        )
        images_dir: Path = prep["images_dir"]

        effective_masks_dir = self._resolve_effective_masks_dir(
            prep=prep,
            enforce_masks=enforce_masks,
            min_required_frames=3,
        )

        log_path = output_dir / "reconstruction.log"
        with open(log_path, "w", encoding="utf-8") as log_file:
            counts = prep["match_mode_counts"]
            log_file.write(
                f"Workspace prepared. accepted={prep['accepted_frames']} "
                f"copied_masks={prep['copied_mask_count']} "
                f"unreadable={prep['rejected_unreadable_frame']} "
                f"missing_mask={prep['rejected_missing_mask']} "
                f"bad_mask={prep['rejected_bad_mask']} "
                f"modes(stem={counts['stem']}, legacy={counts['legacy']}, none={counts['none']}) "
                f"mask_mode={'masked' if effective_masks_dir is not None else 'unmasked'}\n"
            )

            try:
                # Safe defaults for return dict (avoid locals() fragility)
                mesher_used = "unknown"
                selected_model_name = "none"
                db_path = output_dir / "database.db"

                cmd_extract = self.builder.feature_extractor(
                    db_path,
                    images_dir,
                    effective_masks_dir,
                    self._max_image_size,
                )
                self._run_command(cmd_extract, output_dir, log_file)

                cmd_match = self.builder.matcher(self._matcher, db_path)
                self._run_command(cmd_match, output_dir, log_file)

                sparse_dir = output_dir / "sparse"
                sparse_dir.mkdir(exist_ok=True)
                cmd_map = self.builder.mapper(db_path, images_dir, sparse_dir)
                self._run_command(cmd_map, output_dir, log_file)

                best_model = self._select_best_sparse_model(sparse_dir, log_file)
                if not best_model:
                    raise RuntimeReconstructionError(
                        "Sparse reconstruction finished but no valid sub-model found."
                    )

                model_path = best_model["path"]
                registered_images = best_model["registered_images"]
                sparse_points = best_model["points_3d"]
                selected_model_name = model_path.name

                if registered_images < 5 or sparse_points < 100:
                    raise InsufficientReconstructionError(
                        f"Best sparse model too small for densification. "
                        f"model={selected_model_name}, images={registered_images}, points={sparse_points}"
                    )

                dense_dir = output_dir / "dense"
                dense_dir.mkdir(exist_ok=True)
                cmd_undistort = self.builder.image_undistorter(images_dir, model_path, dense_dir)
                self._run_command(cmd_undistort, output_dir, log_file)

                # Initialize dense masking control variables
                force_unmasked_fusion = False
                stereo_masks_dir = dense_dir / "stereo" / "masks"
                if effective_masks_dir and effective_masks_dir.exists():
                    stereo_masks_dir.mkdir(parents=True, exist_ok=True)
                    
                    undistorted_images_dir = dense_dir / "images"
                    if undistorted_images_dir.exists():
                        img_files = list(undistorted_images_dir.glob("*.jpg"))
                        total_frames = len(img_files)
                        fallback_count = 0
                        
                        # Production Guards: Area-based occupancy sanity checks
                        MIN_OCCUPANCY = 0.005 # 0.5% area - subject must exist
                        MAX_OCCUPANCY = 0.98  # 98% area  - background must be removed
                        
                        for img_file in img_files:
                            # Unicode-safe image read for Windows
                            img_array = np.fromfile(str(img_file), np.uint8)
                            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            
                            if img is None:
                                log_file.write(f"WARNING: Could not read undistorted image for masking: {img_file}\n")
                                continue
                                
                            # Extractor produces pure black (0,0,0) background.
                            # We detect subject by keeping any pixel where at least one channel > 0.
                            thresh = (np.any(img > 0, axis=-1).astype(np.uint8) * 255)
                            
                            # Dilate slightly to relax the mask and avoid subject edge clipping 
                            kernel = np.ones((7, 7), np.uint8)
                            gen_mask = cv2.dilate(thresh, kernel, iterations=1)
                            
                            if gen_mask.shape != img.shape[:2]:
                                raise DenseMaskAlignmentError(
                                    f"Shape mismatch: img={img.shape[:2]}, mask={gen_mask.shape}",
                                    image_path=str(img_file)
                                )

                            # Occupancy Sanity Check
                            pixels = gen_mask.shape[0] * gen_mask.shape[1]
                            occupancy = np.count_nonzero(gen_mask) / pixels
                            
                            final_mask = gen_mask
                            if occupancy < MIN_OCCUPANCY or occupancy > MAX_OCCUPANCY:
                                fallback_count += 1
                                log_file.write(
                                    f"WARNING: Mask for {img_file.name} failed occupancy sanity check ({occupancy:.4f}). "
                                    f"Falling back to relaxed (full-white) mask.\n"
                                )
                                final_mask = np.full(gen_mask.shape, 255, dtype=np.uint8)
                            
                            mask_filename = img_file.name + ".png"
                            mask_out_path = stereo_masks_dir / mask_filename
                            # Unicode-safe image write for Windows
                            _, mask_buff = cv2.imencode(".png", final_mask)
                            mask_buff.tofile(str(mask_out_path))
                            
                        # Global fallback assessment
                        if total_frames > 0:
                            fallback_ratio = fallback_count / total_frames
                            log_file.write("\n--- Dense Masking Summary ---\n")
                            log_file.write(f"Strategy: Non-black pixel detection (Any channel > 0)\n")
                            log_file.write(f"Total Frames Processed: {total_frames}\n")
                            log_file.write(f"Fallback (Relaxed) Counter: {fallback_count}\n")
                            log_file.write(f"Fallback Ratio: {fallback_ratio:.2%}\n")
                            log_file.write(f"Min/Max Occupancy Thresholds: {MIN_OCCUPANCY}/{MAX_OCCUPANCY}\n")
                            
                            if fallback_ratio > 0.3:
                                log_file.write(
                                    "CRITICAL WARNING: High mask fallback ratio detected (>30%). "
                                    "Reverting to UNMASKED dense fusion to prevent geometric bias from low-quality masks.\n"
                                )
                                force_unmasked_fusion = True
                            else:
                                force_unmasked_fusion = False
                            log_file.write("-----------------------------\n\n")

                cmd_stereo = self.builder.patch_match_stereo(dense_dir)
                self._run_command(cmd_stereo, output_dir, log_file)

                # Final guard for stereo fusion: Use masks ONLY if they are reliable
                effective_mask_path = stereo_masks_dir if (stereo_masks_dir.exists() and not force_unmasked_fusion) else None
                
                cmd_fuse = self.builder.stereo_fusion(
                    dense_dir, 
                    dense_dir / "fused.ply", 
                    mask_path=effective_mask_path
                )
                self._run_command(cmd_fuse, output_dir, log_file)

                fused_points = self._validate_dense_workspace(output_dir)
                log_file.write(f"Dense fusion successful. Fused points: {fused_points}\n")

                # Guard: If point count is extremely high/low, log it
                if fused_points > 2_000_000:
                    log_file.write(f"WARNING: Extremely high fused point count ({fused_points}). Poisson may take a long time.\n")
                elif fused_points < 1000:
                    log_file.write(f"WARNING: Very low fused point count ({fused_points}). Meshing might fail.\n")

                poisson_ok = False
                mesher_timeout = settings.recon_poisson_timeout_sec
                
                try:
                    cmd_mesh = self.builder.poisson_mesher(
                        dense_dir / "fused.ply",
                        dense_dir / "meshed-poisson.ply",
                    )
                    log_file.write(f"Starting Poisson mesher (timeout={mesher_timeout}s)...\n")
                    self._run_command(cmd_mesh, output_dir, log_file, timeout=mesher_timeout)
                    
                    if self._is_valid_mesh_candidate(dense_dir / "meshed-poisson.ply"):
                        poisson_ok = True
                        mesher_used = "poisson"
                        log_file.write("Poisson meshing successful.\n")
                    else:
                        log_file.write("\nPoisson mesher finished but produced an invalid or empty mesh.\n")

                except (subprocess.TimeoutExpired, RuntimeReconstructionError) as poisson_err:
                    log_file.write(f"\nPoisson mesher FAIL/TIMEOUT: {poisson_err}\n")
                    log_file.write("Falling back to Delaunay mesher...\n")

                if not poisson_ok:
                    try:
                        cmd_mesh = self.builder.delaunay_mesher(
                            dense_dir,
                            dense_dir / "meshed-delaunay.ply",
                        )
                        log_file.write("Starting Delaunay mesher fallback...\n")
                        self._run_command(cmd_mesh, output_dir, log_file)
                        mesher_used = "delaunay"
                        log_file.write("Delaunay meshing successful.\n")
                    except Exception as delaunay_err:
                        log_file.write(f"\nDelaunay mesher failed: {delaunay_err}\n")
                        mesher_used = "failed"

            except ReconstructionError:
                # Propagate typed reconstruction errors directly to the runner
                raise
            except Exception as e:
                # Wrap unknown runtime errors
                log_file.write(f"\nCRITICAL FAILURE: {str(e)}\n")
                raise RuntimeReconstructionError(f"COLMAP chain failed: {str(e)}")

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
            "mesher_used": mesher_used,
            "selected_sparse_model": selected_model_name,
        }


class OpenMVSAdapter(COLMAPAdapter):
    """
    Advanced adapter that uses COLMAP for SfM and OpenMVS for MVS/Texturing.
    """

    def __init__(self, colmap_path: Optional[str] = None, openmvs_path: Optional[str] = None):
        super().__init__(colmap_path)
        self._openmvs_path = openmvs_path or settings.openmvs_path
        self.mvs_builder = OpenMVSCommandBuilder(self._openmvs_path)

    @property
    def engine_type(self) -> str:
        return "colmap_openmvs"

    def run_reconstruction(
        self,
        input_frames: List[str],
        output_dir: Path,
        density: float = 1.0,
        enforce_masks: bool = True,
    ) -> dict:
        log_path = output_dir / "reconstruction.log"
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n--- OpenMVS Pipeline Start ({density=}, {enforce_masks=}) ---\n")

            sampling_step = max(1, int(1.0 / density)) if density < 1.0 else 1
            prep = self._prepare_workspace(
                input_frames,
                output_dir,
                sampling_step=sampling_step,
                enforce_masks=enforce_masks,
            )

            images_dir = prep["images_dir"]
            effective_masks_dir = self._resolve_effective_masks_dir(
                prep=prep,
                enforce_masks=enforce_masks,
                min_required_frames=5,
            )

            counts = prep["match_mode_counts"]
            log_file.write(
                f"OpenMVS workspace prepared. accepted={prep['accepted_frames']} "
                f"copied_masks={prep['copied_mask_count']} "
                f"missing_mask={prep['rejected_missing_mask']} "
                f"bad_mask={prep['rejected_bad_mask']} "
                f"modes(stem={counts['stem']}, legacy={counts['legacy']}, none={counts['none']}) "
                f"mask_mode={'masked' if effective_masks_dir is not None else 'unmasked'}\n"
            )

            # --- MASK REFINEMENT PASS ---
            if enforce_masks:
                self._refine_texture_masks(prep, input_frames, log_file)

            db_path = output_dir / "database.db"
            sparse_dir = output_dir / "sparse"
            sparse_dir.mkdir(exist_ok=True)
            dense_dir = output_dir / "dense"
            dense_dir.mkdir(exist_ok=True)

            try:
                self._run_command(
                    self.builder.feature_extractor(
                        db_path,
                        images_dir,
                        effective_masks_dir,
                        self._max_image_size,
                    ),
                    output_dir,
                    log_file,
                )
                self._run_command(self.builder.matcher(self._matcher, db_path), output_dir, log_file)
                self._run_command(self.builder.mapper(db_path, images_dir, sparse_dir), output_dir, log_file)

                best_model = self._select_best_sparse_model(sparse_dir, log_file)
                if not best_model:
                    raise RuntimeReconstructionError("SfM failed to produce a valid sparse model.")

                registered_images = best_model["registered_images"]
                sparse_points = best_model["points_3d"]

                if registered_images < 5:
                    raise InsufficientReconstructionError(
                        f"SfM only registered {registered_images} images."
                    )

                self._run_command(
                    self.builder.image_undistorter(images_dir, best_model["path"], dense_dir),
                    output_dir,
                    log_file,
                )

                mvs_project = dense_dir / "project.mvs"
                mvs_dense = dense_dir / "project_dense.mvs"
                project_mesh_ply = dense_dir / "project_mesh.ply"
                project_textured_obj = dense_dir / "project_textured.obj"

                self._run_command(
                    self.mvs_builder.interface_colmap(dense_dir, mvs_project),
                    dense_dir,
                    log_file,
                )
                self._run_command(
                    self.mvs_builder.densify_point_cloud(mvs_project, mvs_dense),
                    dense_dir,
                    log_file,
                )
                self._run_command(
                    self.mvs_builder.reconstruct_mesh(mvs_dense, project_mesh_ply),
                    dense_dir,
                    log_file,
                )

                if not project_mesh_ply.exists():
                    raise RuntimeReconstructionError(
                        f"ReconstructMesh failed: {project_mesh_ply} not found."
                    )

                self._run_command(
                    self.mvs_builder.texture_mesh(mvs_dense, project_mesh_ply, project_textured_obj),
                    dense_dir,
                    log_file,
                )

                final_mesh = project_textured_obj
                # OpenMVS 2.4 generates project_textured_material_*_map_Kd.jpg or .png
                # We need to discover the real texture path
                discovered_texture = None
                for ext in ["*.jpg", "*.png"]:
                    for p in dense_dir.glob(f"project_textured_material_*_map_Kd{ext}"):
                        discovered_texture = p
                        break
                    if discovered_texture:
                        break

                if not project_textured_obj.exists() or not discovered_texture:
                    if settings.require_textured_output:
                        raise RuntimeReconstructionError(
                            f"Textured output required but missing. mesh_exists={project_textured_obj.exists()}, texture_found={bool(discovered_texture)}"
                        )
                    
                    log_file.write("\nWarning: Texturing failed or produced incomplete results. Falling back to geometry-only mesh.\n")
                    if project_mesh_ply.exists():
                        final_mesh = project_mesh_ply
                        discovered_texture = None
                    else:
                        raise RuntimeReconstructionError(
                            "OpenMVS completed but no mesh output found (textured or untextured)."
                        )

                stats = self._mesh_stats(str(final_mesh))

                return {
                    "mesh_path": str(final_mesh),
                    "texture_path": str(discovered_texture) if discovered_texture else str(output_dir / "_no_texture.png"),
                    "log_path": str(log_path),
                    "vertex_count": stats["vertex_count"],
                    "face_count": stats["face_count"],
                    "registered_images": registered_images,
                    "sparse_points": sparse_points,
                    "dense_points_fused": stats.get("vertex_count", 0),  # OpenMVS fused point proxy
                    "mesher_used": "openmvs_reconstruct_mesh",
                    "engine_type": self.engine_type,
                    "textured": bool(discovered_texture),
                    "selected_sparse_model": best_model["path"].name,
                }

            except Exception as e:
                log_file.write(f"\nOpenMVS Pipeline Error: {str(e)}\n")
                if settings.openmvs_fail_hard:
                    raise
                log_file.write("openmvs_fail_hard is False, bubbling up for COLMAP fallback.\n")
                raise RuntimeReconstructionError(f"OpenMVS failed: {e}")

    def _refine_texture_masks(self, prep: Dict[str, Any], input_frames: List[str], log_file) -> None:
        """
        Refines masks in the reconstruction workspace.
        - Erodes masks to avoid edge contamination.
        - Rejects frames with known quality issues (clipping, support) if enough frames remain.
        """
        masks_dir = prep["masks_dir"]
        if not masks_dir or not masks_dir.exists():
            return

        from modules.operations.settings import settings
        erode_px = settings.texture_mask_erode_px
        reject_support = settings.texture_reject_support_contamination
        reject_clipped = settings.texture_reject_subject_clipped
        min_clean = settings.texture_min_clean_frames

        log_file.write(f"Mask refinement: erode={erode_px}px, reject_support={reject_support}, "
                      f"reject_clipped={reject_clipped}, min_clean={min_clean}\n")

        mask_infos = []
        for frame_path in input_frames:
            src = Path(frame_path)
            mask_path = masks_dir / f"{src.name}.png"
            if not mask_path.exists():
                continue
            
            # Find metadata in the original capture structure
            # .../frames/frame_0001.jpg -> .../frames/masks/frame_0001.json
            meta_path = src.parent / "masks" / f"{src.stem}.json"
            meta = {}
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass
            
            reasons = meta.get("reasons", []) or meta.get("failure_reasons", [])
            is_clipped = (
                meta.get("is_clipped", False)
                or meta.get("subject_clipped", False)
                or "subject_clipped" in reasons
            )
            support_suspected = (
                meta.get("support_suspected", False)
                or meta.get("support_contamination_detected", False)
                or "support_contamination_detected" in reasons
            )

            mask_infos.append({
                "path": mask_path,
                "meta": meta,
                "is_clipped": bool(is_clipped),
                "support_suspected": bool(support_suspected),
                "occupancy": float(meta.get("occupancy", 0.0))
            })

        if not mask_infos:
            return

        # 1. Filter frames
        clean_infos = []
        for info in mask_infos:
            rejected = False
            if reject_clipped and info["is_clipped"]:
                rejected = True
            if reject_support and info["support_suspected"]:
                rejected = True
            
            if not rejected:
                clean_infos.append(info)

        # 2. Safe Fallback
        if len(clean_infos) < min_clean:
            log_file.write(f"WARNING: texture mask refinement fallback: too few clean frames ({len(clean_infos)} < {min_clean}). Using all available masks.\n")
            final_selection = mask_infos
        else:
            log_file.write(f"Mask refinement: rejected {len(mask_infos) - len(clean_infos)} contaminated/clipped frames. Remaining: {len(clean_infos)}\n")
            final_selection = clean_infos
            
            # Blank out rejected masks to prevent them from being used in OpenMVS
            clean_paths = {info["path"] for info in final_selection}
            for info in mask_infos:
                if info["path"] not in clean_paths:
                    mask_img = cv2.imdecode(np.fromfile(str(info["path"]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        blank = np.zeros_like(mask_img)
                        _, buff = cv2.imencode(".png", blank)
                        buff.tofile(str(info["path"]))

        # 3. Erosion
        if erode_px > 0:
            kernel = np.ones((erode_px, erode_px), np.uint8)
            for info in final_selection:
                mask_path = info["path"]
                # Unicode-safe read for Windows
                mask_img = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    eroded = cv2.erode(mask_img, kernel, iterations=1)
                    # Unicode-safe write for Windows
                    _, buff = cv2.imencode(".png", eroded)
                    buff.tofile(str(mask_path))



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
        enforce_masks: bool = True,
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