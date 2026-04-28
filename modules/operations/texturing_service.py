"""
modules/operations/texturing_service.py

SPRINT 1 — TICKET-002: Worker Refactor

Extracts the OpenMVS texturing + OBJ vertex alignment logic that was previously
embedded inside IngestionWorker._handle_cleanup() (the "god method").

Responsibilities:
  - Locate COLMAP/dense workspace from manifest path
  - Invoke OpenMVSTexturer
  - Re-apply pivot alignment to the textured OBJ (preserves UVs/MTL reliably)
  - Update the OutputManifest in place and return it
  - Report texturing_status: "real" | "degraded" | "absent"

The worker delegates entirely to this service; it does NOT know the internals.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from modules.operations.logging_config import get_component_logger
from modules.reconstruction_engine.output_manifest import OutputManifest
from modules.utils.file_persistence import atomic_write_json, calculate_checksum
import trimesh

logger = get_component_logger("texturing_service")


class TexturingResult:
    """Value object returned by TexturingService.run()."""

    __slots__ = (
        "texturing_status",
        "cleaned_mesh_path",
        "texture_atlas_paths",
        "manifest",
    )

    def __init__(
        self,
        texturing_status: str,
        cleaned_mesh_path: str,
        texture_atlas_paths: list[str],
        manifest: OutputManifest,
    ) -> None:
        self.texturing_status = texturing_status          # "real" | "degraded" | "absent"
        self.cleaned_mesh_path = cleaned_mesh_path        # path to use for export
        self.texture_atlas_paths = texture_atlas_paths    # may be empty
        self.manifest = manifest                          # updated manifest (not yet persisted)


class TexturingService:
    """
    Owns the OpenMVS texturing pipeline and the post-texturing OBJ alignment step.

    Usage:
        service = TexturingService()
        result = service.run(
            manifest=manifest,
            cleanup_stats=cleanup_stats,
            pivot_offset=metadata.pivot_offset,
            cleaned_mesh_path=cleaned_mesh_path,
        )
    """

    def run(
        self,
        manifest: OutputManifest,
        cleanup_stats: Dict[str, Any],
        pivot_offset: Dict[str, float],
        cleaned_mesh_path: str,
        expected_color: str = "unknown",
    ) -> TexturingResult:
        """
        Execute texturing if the manifest indicates a COLMAP reconstruction.

        Returns a TexturingResult with the resolved mesh path, atlas paths,
        status string, and an updated (but not yet persisted) manifest.
        """
        if cleanup_stats.get("cleanup_mode") == "texture_safe_copy":
            logger.info("Cleanup mode is texture_safe_copy. Skipping OpenMVS texturing.")
            
            cleaned_mesh = cleanup_stats["cleaned_mesh_path"]
            cleaned_texture = cleanup_stats["cleaned_texture_path"]
            
            valid = self._validate_texture_safe_bundle(cleaned_mesh, cleaned_texture)
            if not valid:
                logger.warning("Texture safe bundle validation failed - degraded.")
                return TexturingResult(
                    texturing_status="degraded",
                    cleaned_mesh_path=cleaned_mesh_path,
                    texture_atlas_paths=[],
                    manifest=manifest,
                )

            # Update manifest for successful skip
            manifest.mesh_path = cleaned_mesh
            manifest.textured_mesh_path = cleaned_mesh
            manifest.texture_path = cleaned_texture
            manifest.texture_atlas_paths = [cleaned_texture]
            manifest.texturing_engine = "texture_safe_copy"
            manifest.texturing_status = "real"
            manifest.mesh_metadata.has_texture = True
            manifest.mesh_metadata.uv_present = True
            
            from modules.utils.file_persistence import calculate_checksum
            try:
                manifest.checksum = calculate_checksum(cleaned_mesh)
            except Exception:
                pass

            return TexturingResult(
                texturing_status="real",
                cleaned_mesh_path=cleaned_mesh,
                texture_atlas_paths=[cleaned_texture],
                manifest=manifest,
            )

        # Locate the COLMAP dense workspace relative to the raw mesh path.
        mesh_parent = Path(manifest.mesh_path).parent
        colmap_dir = mesh_parent.parent if mesh_parent.name == "dense" else mesh_parent

        if not colmap_dir.joinpath("dense").exists():
            logger.info(
                f"Texturing skipped: dense/ workspace not found under {colmap_dir} "
                f"(declared engine: {manifest.engine_type})."
            )
            return TexturingResult(
                texturing_status="absent",
                cleaned_mesh_path=cleaned_mesh_path,
                texture_atlas_paths=[],
                manifest=manifest,
            )

        return self._run_openmvs(
            manifest=manifest,
            colmap_dir=colmap_dir,
            cleanup_stats=cleanup_stats,
            pivot_offset=pivot_offset,
            cleaned_mesh_path=cleaned_mesh_path,
            expected_color=expected_color,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_openmvs(
        self,
        manifest: OutputManifest,
        colmap_dir: Path,
        cleanup_stats: Dict[str, Any],
        pivot_offset: Dict[str, float],
        cleaned_mesh_path: str,
        expected_color: str = "unknown",
    ) -> TexturingResult:
        from modules.reconstruction_engine.openmvs_texturer import OpenMVSTexturer

        texturer = OpenMVSTexturer()
        texturing_dir = Path(cleaned_mesh_path).parent / "texturing"
        texturing_dir.mkdir(exist_ok=True, parents=True)

        try:
            texture_results = texturer.run_texturing(
                colmap_workspace=colmap_dir,
                dense_workspace=colmap_dir / "dense",
                selected_mesh=cleanup_stats["pre_aligned_mesh_path"],
                output_dir=texturing_dir,
                expected_color=expected_color,
                neutralization_type=settings.texture_neutralization_type,
            )
        except Exception as exc:
            logger.warning(f"OpenMVS texturing failed (degraded): {exc}")
            
            # Fix 7: Fix log path propagation even on failure
            log_path = texturing_dir / "texturing.log"
            manifest.texturing_log_path = str(log_path)
            
            from modules.reconstruction_engine.failures import TexturingFailed
            if isinstance(exc, TexturingFailed) and exc.log_path:
                manifest.texturing_log_path = exc.log_path

            return TexturingResult(
                texturing_status="degraded",
                cleaned_mesh_path=cleaned_mesh_path,
                texture_atlas_paths=[],
                manifest=manifest,
            )

        textured_path = texture_results["textured_mesh_path"]
        generated_textures = texture_results.get("texture_atlas_paths", [])

        # SPRINT 5C: Atlas Repair Service + Retry Loop
        if generated_textures:
            from modules.operations.atlas_repair_service import AtlasRepairService
            repair_service = AtlasRepairService()
            primary_atlas = generated_textures[0]
            repair_results = repair_service.repair_atlas(primary_atlas, expected_color=expected_color)
            
            # Check for background contamination failure
            stats = repair_results.get("repaired_stats") or repair_results.get("stats") or {}
            bg_ratio = stats.get("dominant_background_color_ratio", 0.0)
            
            # If fail due to background contamination, try stricter retry
            if bg_ratio > 0.25: # Hard limit for white_cream
                logger.warning(f"Excessive background contamination ({bg_ratio:.3f} > 0.25). Retrying with stricter frames...")
                
                # Retry 1: Top 12 frames
                try:
                    retry_results = texturer.run_texturing(
                        colmap_workspace=colmap_dir,
                        dense_workspace=colmap_dir / "dense",
                        selected_mesh=cleanup_stats["pre_aligned_mesh_path"],
                        output_dir=texturing_dir / "retry_top12",
                        expected_color=expected_color,
                        top_n=12,
                        neutralization_type=settings.texture_neutralization_type,
                    )
                    if retry_results.get("texture_atlas_paths"):
                        primary_atlas = retry_results["texture_atlas_paths"][0]
                        repair_results = repair_service.repair_atlas(primary_atlas, expected_color=expected_color)
                        stats = repair_results.get("repaired_stats") or repair_results.get("stats") or {}
                        bg_ratio = stats.get("dominant_background_color_ratio", 0.0)
                        
                        if bg_ratio <= 0.25:
                            logger.info("Retry with Top 12 frames successful.")
                            texture_results = retry_results
                            generated_textures = retry_results["texture_atlas_paths"]
                        else:
                            logger.warning(f"Retry with Top 12 frames still contaminated ({bg_ratio:.3f}). Trying masked sources...")
                            
                            # Retry 2: Masked sources
                            # We need to find the masked_images_dir from previous run if possible
                            # Actually, we can just point to it if it exists
                            masked_images_dir = texturing_dir / "selected_images_masked"
                            if masked_images_dir.exists() and any(masked_images_dir.glob("*.jpg")):
                                retry_masked_results = texturer.run_texturing(
                                    colmap_workspace=colmap_dir,
                                    dense_workspace=colmap_dir / "dense",
                                    selected_mesh=cleanup_stats["pre_aligned_mesh_path"],
                                    output_dir=texturing_dir / "retry_masked",
                                    expected_color=expected_color,
                                    image_folder_override=masked_images_dir,
                                    neutralization_type=settings.texture_neutralization_type,
                                )
                                if retry_masked_results.get("texture_atlas_paths"):
                                    primary_atlas = retry_masked_results["texture_atlas_paths"][0]
                                    repair_results = repair_service.repair_atlas(primary_atlas, expected_color=expected_color)
                                    stats = repair_results.get("repaired_stats") or repair_results.get("stats") or {}
                                    bg_ratio = stats.get("dominant_background_color_ratio", 0.0)
                                    
                                    logger.info(f"Retry with masked sources: bg_ratio={bg_ratio:.3f}")
                                    texture_results = retry_masked_results
                                    generated_textures = retry_masked_results["texture_atlas_paths"]
                except Exception as retry_exc:
                    logger.warning(f"Texturing retry failed: {retry_exc}")

            if repair_results["status"] == "repaired":
                logger.info(f"Using repaired atlas: {repair_results['repaired_path']}")
                # Replace the primary atlas in the results
                generated_textures[0] = repair_results["repaired_path"]
                texture_results["texture_atlas_paths"] = generated_textures
                final_texture_stats = repair_results["repaired_stats"]
                final_texture_stats["repaired_atlas_path"] = repair_results["repaired_path"]
            else:
                final_texture_stats = repair_results.get("stats")

            # Update paths to ensure they point to the retry results if applicable
            textured_path = texture_results["textured_mesh_path"]
            generated_textures = texture_results.get("texture_atlas_paths", [])

            # Save texture quality report
            if final_texture_stats:
                report_path = texturing_dir / "texture_quality_report.json"
                atomic_write_json(report_path, final_texture_stats)
                cleanup_stats["texture_quality_report_path"] = str(report_path)

        # Verify UV presence before committing to "real" status.
        has_uv = self._check_uv(textured_path, trimesh)
        if not has_uv:
            logger.warning("Texturing produced mesh but no UV coordinates detected — degraded.")
            return TexturingResult(
                texturing_status="degraded",
                cleaned_mesh_path=cleaned_mesh_path,
                texture_atlas_paths=texture_results.get("texture_atlas_paths", []),
                manifest=manifest,
            )

        # Re-apply alignment shift directly on the OBJ (preserves UVs/MTL).
        aligned_textured_obj = self._apply_pivot_to_obj(textured_path, pivot_offset, cleaned_mesh_path)

        # Relocate MTL and textures from texturing scratch dir to the final parent output dir
        import shutil
        target_dir = Path(cleaned_mesh_path).parent
        source_dir = Path(textured_path).parent
        
        for mtl_file in source_dir.glob("*.mtl"):
            try:
                dest_mtl = target_dir / mtl_file.name
                with open(mtl_file, "r", encoding="utf-8") as fm_in:
                    lines = fm_in.readlines()
                with open(dest_mtl, "w", encoding="utf-8") as fm_out:
                    for line in lines:
                        if line.strip().startswith(("map_Kd", "map_bump", "bump", "map_Ks", "map_Ns", "map_d", "norm", "map_Ka")):
                            last_slash_idx = max(line.rfind('/'), line.rfind('\\'))
                            if last_slash_idx != -1:
                                basename = line[last_slash_idx + 1:]
                                space_idx = line.rfind(' ', 0, last_slash_idx)
                                if space_idx != -1:
                                    new_line = line[:space_idx + 1] + basename
                                else:
                                    new_line = line
                                fm_out.write(new_line)
                                continue
                        fm_out.write(line)
            except Exception as e:
                logger.warning(f"Failed to copy or rewrite MTL file {mtl_file}: {e}")
                
        new_atlas_paths = []
        for atlas in texture_results.get("texture_atlas_paths", []):
            try:
                dest = target_dir / Path(atlas).name
                shutil.copy2(atlas, dest)
                new_atlas_paths.append(str(dest))
            except Exception as e:
                new_atlas_paths.append(atlas)
                
        texture_results["texture_atlas_paths"] = new_atlas_paths

        # Update texture_results to reflect the aligned path.
        texture_results["textured_mesh_path"] = aligned_textured_obj

        # Refresh manifest fields — caller is still responsible for persisting.
        manifest.textured_mesh_path = aligned_textured_obj
        manifest.texture_atlas_paths = texture_results.get("texture_atlas_paths", [])
        manifest.texturing_engine = texture_results.get("texturing_engine", "openmvs")
        manifest.texturing_log_path = texture_results.get("log_path")
        manifest.mesh_metadata.uv_present = True
        manifest.mesh_metadata.has_texture = True

        # The canonical mesh path now points to the textured+aligned mesh.
        manifest.mesh_path = aligned_textured_obj
        try:
            manifest.checksum = calculate_checksum(aligned_textured_obj)
        except Exception:
            pass  # Non-critical; checksum absence doesn't block the pipeline.

        # Refresh vertex/face counts from the textured geometry.
        try:
            reloaded = trimesh.load(aligned_textured_obj, force="scene")
            if hasattr(reloaded, "geometry"):
                # It's a Scene
                manifest.mesh_metadata.vertex_count = sum(len(g.vertices) for g in reloaded.geometry.values() if hasattr(g, 'vertices'))
                manifest.mesh_metadata.face_count = sum(len(g.faces) for g in reloaded.geometry.values() if hasattr(g, 'faces'))
            else:
                # It's a single Trimesh
                manifest.mesh_metadata.vertex_count = int(len(reloaded.vertices))
                manifest.mesh_metadata.face_count = int(len(reloaded.faces))
        except Exception:
            pass

        return TexturingResult(
            texturing_status="real",
            cleaned_mesh_path=aligned_textured_obj,
            texture_atlas_paths=texture_results.get("texture_atlas_paths", []),
            manifest=manifest,
        )

    @staticmethod
    def _check_uv(mesh_path: str, trimesh_module: Any) -> bool:
        """Load a mesh and return True if it has UV coordinates."""
        try:
            m = trimesh_module.load(mesh_path, force="scene")
            if hasattr(m, "geometry"):
                # Iterate through scene geometry
                return any(
                    hasattr(g, "visual")
                    and hasattr(g.visual, "uv")
                    and g.visual.uv is not None
                    and len(g.visual.uv) > 0
                    for g in m.geometry.values()
                )
            else:
                return bool(
                    hasattr(m, "visual")
                    and hasattr(m.visual, "uv")
                    and m.visual.uv is not None
                    and len(m.visual.uv) > 0
                )
        except Exception:
            return False

    @staticmethod
    def _apply_pivot_to_obj(
        source_obj: str,
        pivot: Dict[str, float],
        cleaned_mesh_path: str,
    ) -> str:
        """
        Re-apply the alignment pivot shift to a textured OBJ file without
        touching UV or material lines.

        Only lines beginning with exactly "v " (geometric vertices) are shifted.
        All other lines (vt, vn, f, mtllib, usemtl, etc.) pass through unchanged.

        Returns the path to the written file.
        """
        out_path = str(Path(cleaned_mesh_path).parent / "textured_aligned_mesh.obj")
        px = pivot.get("x", 0.0)
        py = pivot.get("y", 0.0)
        pz = pivot.get("z", 0.0)

        with (
            open(source_obj, "r", encoding="utf-8") as f_in,
            open(out_path, "w", encoding="utf-8") as f_out,
        ):
            for line in f_in:
                if line.startswith("v "):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        try:
                            x = float(parts[1]) + px
                            y = float(parts[2]) + py
                            z = float(parts[3]) + pz
                            f_out.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                            continue
                        except ValueError:
                            pass  # Malformed vertex line — write as-is.
                elif line.startswith("mtllib "):
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        mtl_basename = Path(parts[1]).name
                        f_out.write(f"mtllib {mtl_basename}\n")
                        continue
                f_out.write(line)

        return out_path

    def _validate_texture_safe_bundle(self, mesh_path: str, texture_path: str) -> bool:
        """Verify the cleaned OBJ/MTL/texture bundle."""
        p_mesh = Path(mesh_path)
        p_tex = Path(texture_path)
        
        if not p_mesh.exists() or not p_tex.exists():
            return False
            
        # OBJ contains vt lines
        vt_found = False
        mtllib_found = False
        mtl_filename = None
        try:
            with open(p_mesh, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("vt "):
                        vt_found = True
                    if stripped.startswith("mtllib "):
                        mtllib_found = True
                        parts = stripped.split(None, 1)
                        if len(parts) > 1:
                            mtl_filename = parts[1].strip()
                    if vt_found and mtllib_found:
                        break
        except Exception:
            return False
        
        if not vt_found or not mtllib_found or not mtl_filename:
            return False
            
        # MTL exists next to OBJ
        p_mtl = p_mesh.parent / mtl_filename
        if not p_mtl.exists():
            return False
            
        # MTL has map_Kd and target exists
        map_kd_target = None
        try:
            with open(p_mtl, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("map_Kd "):
                        parts = stripped.split(None, 1)
                        if len(parts) > 1:
                            map_kd_target = parts[1].strip()
                        break
        except Exception:
            return False
        
        if not map_kd_target:
            return False

        # Resolve relative to MTL parent
        try:
            # Handle potential relative paths like ./texture.jpg
            target_path = (p_mtl.parent / map_kd_target)
            if not target_path.exists():
                return False
            
            # Consistency check with provided texture_path
            # In a valid bundle, the MTL map_Kd should point to the same file as p_tex
            if target_path.resolve() != p_tex.resolve():
                # If resolves are different, check if at least the filenames match
                # (Sometimes resolution fails in complex environments, but the relative link must be sound)
                if target_path.name != p_tex.name:
                    return False
        except Exception:
            return False
        
        return True
