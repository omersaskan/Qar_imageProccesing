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
    ) -> TexturingResult:
        """
        Execute texturing if the manifest indicates a COLMAP reconstruction.

        Returns a TexturingResult with the resolved mesh path, atlas paths,
        status string, and an updated (but not yet persisted) manifest.
        """
        if manifest.engine_type != "colmap":
            logger.info("Texturing skipped: engine_type is not 'colmap'.")
            return TexturingResult(
                texturing_status="absent",
                cleaned_mesh_path=cleaned_mesh_path,
                texture_atlas_paths=[],
                manifest=manifest,
            )

        # Locate the COLMAP dense workspace relative to the raw mesh path.
        mesh_parent = Path(manifest.mesh_path).parent
        colmap_dir = mesh_parent.parent if mesh_parent.name == "dense" else mesh_parent

        if not colmap_dir.joinpath("dense").exists():
            logger.info(
                f"Texturing skipped: dense/ workspace not found under {colmap_dir}."
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
    ) -> TexturingResult:
        from modules.reconstruction_engine.openmvs_texturer import OpenMVSTexturer
        from modules.utils.file_persistence import calculate_checksum
        import trimesh

        texturer = OpenMVSTexturer()
        texturing_dir = Path(cleaned_mesh_path).parent / "texturing"
        texturing_dir.mkdir(exist_ok=True, parents=True)

        try:
            texture_results = texturer.run_texturing(
                colmap_workspace=colmap_dir,
                dense_workspace=colmap_dir / "dense",
                selected_mesh=cleanup_stats["pre_aligned_mesh_path"],
                output_dir=texturing_dir,
            )
        except Exception as exc:
            logger.warning(f"OpenMVS texturing failed (degraded): {exc}")
            return TexturingResult(
                texturing_status="degraded",
                cleaned_mesh_path=cleaned_mesh_path,
                texture_atlas_paths=[],
                manifest=manifest,
            )

        textured_path = texture_results["textured_mesh_path"]

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
                shutil.copy2(mtl_file, target_dir / mtl_file.name)
            except Exception:
                pass
                
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
            reloaded = trimesh.load(aligned_textured_obj, force="mesh")
            if isinstance(reloaded, trimesh.Scene):
                reloaded = reloaded.dump(concatenate=True)
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
            m = trimesh_module.load(mesh_path, force="mesh")
            if isinstance(m, trimesh_module.Scene):
                m = m.dump(concatenate=True)
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
