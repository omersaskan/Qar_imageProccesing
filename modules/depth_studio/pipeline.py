"""Top-level Depth Studio pipeline orchestrator."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from .input_router import route_input
from .image_preflight import check_image
from .video_frame_selector import select_best_frame
from .depth_anything_provider import DepthAnythingV2Provider
from .depth_pro_provider import DepthProProvider
from .depth_output import write_depth_preview, load_depth_png16
from .depth_refinement import refine_depth
from .texture_projection import prepare_texture
from .glb_builder import build_glb
from .manifest import build_manifest, write_manifest
from .subject_masker import compute_subject_mask, apply_mask_to_depth
from modules.operations.settings import settings


def _get_provider(provider_name: str):
    if provider_name == "depth_pro":
        return DepthProProvider(device=settings.depth_anything_device)
    return DepthAnythingV2Provider(
        checkpoint=settings.depth_anything_checkpoint,
        device=settings.depth_anything_device,
    )


def run_depth_studio(
    session_id: str,
    input_file_path: str,
    output_base_dir: str,
    provider_name: Optional[str] = None,
    explicit_final_override: bool = False,
    prompt_box: Optional[tuple] = None,
) -> Dict[str, Any]:
    """
    Full Depth Studio pipeline.
    Returns the completed manifest dict.
    """
    provider_name = provider_name or settings.depth_studio_default_provider
    provider = _get_provider(provider_name)

    input_dir = Path(output_base_dir) / "input"
    derived_dir = Path(output_base_dir) / "derived"
    manifests_dir = Path(output_base_dir) / "manifests"
    input_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    warnings: list = []
    selected_frame_path: Optional[str] = None
    depth_map_path: Optional[str] = None
    depth_format: Optional[str] = None
    glb_path: Optional[str] = None
    mesh_vertex_count = 0
    mesh_face_count = 0
    refinement_applied = False
    final_status = "failed"
    mask_method: Optional[str] = None
    mask_fg_ratio: Optional[float] = None
    mask_bbox: Optional[list] = None
    mask_full_frame_fallback: bool = False
    mask_overlay_path: Optional[str] = None
    mask_stats_path: Optional[str] = None
    mask_quality: Optional[str] = None
    mask_component_count: int = 0
    mask_selected_area_ratio: Optional[float] = None
    _subject_mask = None   # kept for GLB face culling

    # ── 1. Route input ────────────────────────────────────────────────────────
    try:
        input_type, _ = route_input(input_file_path)
    except ValueError as e:
        manifest = build_manifest(
            session_id=session_id, input_type="unknown", input_path=input_file_path,
            provider=provider_name, model_name=None, provider_status="failed",
            license_note=provider.license_note, selected_frame_path=None,
            depth_map_path=None, depth_format=None, refinement_applied=False,
            mesh_mode=settings.depth_mesh_mode, mesh_vertex_count=0, mesh_face_count=0,
            glb_path=None, status="failed", warnings=[str(e)],
        )
        write_manifest(manifest, str(manifests_dir))
        return manifest

    # Copy original to session input/
    dest_name = "input" + Path(input_file_path).suffix
    dest_input = input_dir / dest_name
    shutil.copy2(input_file_path, str(dest_input))

    # ── 2. Frame selection (video) or direct image path ───────────────────────
    if input_type == "video":
        if not settings.depth_studio_allow_video_input:
            warnings.append("video_input_disabled")
            final_status = "failed"
            manifest = build_manifest(
                session_id=session_id, input_type=input_type, input_path=input_file_path,
                provider=provider_name, model_name=None, provider_status="failed",
                license_note=provider.license_note, selected_frame_path=None,
                depth_map_path=None, depth_format=None, refinement_applied=False,
                mesh_mode=settings.depth_mesh_mode, mesh_vertex_count=0, mesh_face_count=0,
                glb_path=None, status=final_status, warnings=warnings,
            )
            write_manifest(manifest, str(manifests_dir))
            return manifest

        frame_path = str(derived_dir / "selected_frame.jpg")
        ok, reason, frame_meta = select_best_frame(str(dest_input), frame_path)
        if not ok:
            warnings.append(f"frame_selection_failed: {reason}")
            final_status = "failed"
        else:
            warnings.append("input_video_best_frame_used")
            selected_frame_path = frame_path
        image_path_for_depth = frame_path if ok else None
    else:
        ok, reason, _ = check_image(str(dest_input))
        if not ok:
            warnings.append(f"image_preflight_failed: {reason}")
            image_path_for_depth = None
        else:
            image_path_for_depth = str(dest_input)

    if not image_path_for_depth:
        final_status = "failed"
        manifest = build_manifest(
            session_id=session_id, input_type=input_type, input_path=input_file_path,
            provider=provider_name, model_name=None, provider_status="failed",
            license_note=provider.license_note, selected_frame_path=selected_frame_path,
            depth_map_path=None, depth_format=None, refinement_applied=False,
            mesh_mode=settings.depth_mesh_mode, mesh_vertex_count=0, mesh_face_count=0,
            glb_path=None, status=final_status, warnings=warnings,
        )
        write_manifest(manifest, str(manifests_dir))
        return manifest

    # ── 3. Depth inference ────────────────────────────────────────────────────
    depth_result = provider.safe_infer(image_path_for_depth, str(derived_dir))
    model_name = depth_result.get("model_name")
    provider_status = depth_result.get("status", "failed")
    warnings.extend(depth_result.get("warnings", []))

    if provider.is_experimental:
        warnings.append("experimental_provider")

    warnings.append("single_view_geometry")
    warnings.append("backside_not_observed")
    warnings.append("preview_only_asset")

    if provider_status != "ok":
        manifest = build_manifest(
            session_id=session_id, input_type=input_type, input_path=input_file_path,
            provider=provider_name, model_name=model_name,
            provider_status=provider_status,
            license_note=provider.license_note, selected_frame_path=selected_frame_path,
            depth_map_path=None, depth_format=None, refinement_applied=False,
            mesh_mode=settings.depth_mesh_mode, mesh_vertex_count=0, mesh_face_count=0,
            glb_path=None, status="failed", warnings=warnings,
        )
        write_manifest(manifest, str(manifests_dir))
        return manifest

    depth_map_path = depth_result["depth_map_path"]
    depth_format = depth_result.get("depth_format", "png16")

    # ── 4. Depth preview ──────────────────────────────────────────────────────
    try:
        import cv2
        raw = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
        if raw is not None:
            import numpy as np
            d_f = raw.astype(np.float32) / 65535.0
            write_depth_preview(d_f, str(derived_dir / "depth_preview.png"))
    except Exception:
        pass

    # ── 5. Depth refinement ───────────────────────────────────────────────────
    if settings.depth_edge_cleanup_enabled:
        try:
            import cv2
            import numpy as np
            d_raw = load_depth_png16(depth_map_path)
            img_bgr = cv2.imread(image_path_for_depth)
            ref = refine_depth(d_raw, img_bgr, edge_cleanup=True)
            refined_path = str(derived_dir / "depth_refined_16.png")
            d16 = (ref["depth"] * 65535).astype(np.uint16)
            cv2.imwrite(refined_path, d16)
            depth_map_path = refined_path
            refinement_applied = True
        except Exception:
            pass

    # ── 5b. Subject masking → masked depth ───────────────────────────────────
    try:
        import numpy as np
        import cv2 as _cv2
        d_for_mask = _cv2.imread(depth_map_path, _cv2.IMREAD_UNCHANGED)
        if d_for_mask is not None:
            d_float = d_for_mask.astype(np.float32) / 65535.0
            mask_result = compute_subject_mask(
                image_path=image_path_for_depth,
                depth_norm=d_float,
                output_dir=str(derived_dir),
                prompt_box=prompt_box,
            )
            mask_method = mask_result["method_used"]
            mask_fg_ratio = mask_result["fg_ratio"]
            mask_bbox = mask_result["bbox"]
            mask_full_frame_fallback = mask_result["full_frame_fallback_used"]
            mask_overlay_path = mask_result.get("overlay_path")
            mask_stats_path = mask_result.get("stats_path")
            _subject_mask = mask_result.get("mask")
            mask_quality = mask_result.get("mask_quality")
            mask_component_count = mask_result.get("component_count", 0)
            mask_selected_area_ratio = mask_result.get("selected_component_area_ratio")
            warnings.extend([f"mask:{w}" for w in mask_result.get("warnings", [])])

            # Apply mask: push background depth to far value
            masked_depth = apply_mask_to_depth(d_float, mask_result["mask"])
            masked_d16 = (masked_depth * 65535).astype(np.uint16)
            masked_depth_path = str(derived_dir / "depth_masked_16.png")
            _cv2.imwrite(masked_depth_path, masked_d16)
            depth_map_path = masked_depth_path
    except Exception as _mask_err:
        warnings.append(f"subject_masking_failed:{_mask_err}")

    # ── 6. Texture preparation ────────────────────────────────────────────────
    tex_result = prepare_texture(image_path_for_depth, str(derived_dir))
    texture_path = tex_result.get("texture_path")
    if not texture_path:
        warnings.append("texture_preparation_failed")
        texture_path = image_path_for_depth  # fallback to original

    # ── 7. GLB build ──────────────────────────────────────────────────────────
    glb_out = str(derived_dir / "preview_mesh.glb")
    glb_result = build_glb(
        depth_map_path=depth_map_path,
        texture_path=texture_path,
        output_glb_path=glb_out,
        grid_resolution=settings.depth_grid_resolution,
        mask=_subject_mask,
    )
    glb_path = glb_result.get("glb_path")
    mesh_vertex_count = glb_result.get("mesh_vertex_count", 0)
    mesh_face_count = glb_result.get("mesh_face_count", 0)

    if glb_result.get("status") == "ok":
        if mask_quality in ("review", "low_confidence"):
            final_status = "partial"
        else:
            final_status = "ok"
    else:
        warnings.append(f"glb_build_failed: {glb_result.get('reason', '')}")
        final_status = "partial"

    # ── 8. Manifest ───────────────────────────────────────────────────────────
    manifest = build_manifest(
        session_id=session_id,
        input_type=input_type,
        input_path=input_file_path,
        provider=provider_name,
        model_name=model_name,
        provider_status=provider_status,
        license_note=provider.license_note,
        selected_frame_path=selected_frame_path,
        depth_map_path=depth_map_path,
        depth_format=depth_format,
        refinement_applied=refinement_applied,
        mesh_mode=settings.depth_mesh_mode,
        mesh_vertex_count=mesh_vertex_count,
        mesh_face_count=mesh_face_count,
        glb_path=glb_path,
        status=final_status,
        warnings=list(dict.fromkeys(warnings)),  # deduplicate while preserving order
        mask_method=mask_method,
        mask_fg_ratio=mask_fg_ratio,
        mask_bbox=mask_bbox,
        mask_full_frame_fallback=mask_full_frame_fallback,
        mask_overlay_path=mask_overlay_path,
        mask_stats_path=mask_stats_path,
        mask_quality=mask_quality,
        mask_component_count=mask_component_count,
        mask_selected_area_ratio=mask_selected_area_ratio,
    )
    write_manifest(manifest, str(manifests_dir))
    return manifest
