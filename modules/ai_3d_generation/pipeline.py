"""
AI 3D Generation pipeline orchestrator.

Entry point: generate_ai_3d()

Steps:
  1. Route input (image → direct, video → best frame)
  2. Preprocess (crop / square pad / resize → ai3d_input.png)
  3. Provider selection
  4. Generate (via provider.safe_generate)
  5. Postprocess stubs
  6. Quality gate
  7. Write manifest

Does not touch Depth Studio or photogrammetry pipelines.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from .manifest import build_manifest, write_manifest
from .input_preprocessor import preprocess_input
from .postprocess import run_postprocess
from .quality_gate import evaluate as quality_evaluate

logger = logging.getLogger("ai_3d_generation.pipeline")


def _get_provider(provider_name: str):
    if provider_name == "sf3d":
        from .sf3d_provider import SF3DProvider
        return SF3DProvider()
    # Future providers go here
    from .sf3d_provider import SF3DProvider
    return SF3DProvider()


def generate_ai_3d(
    session_id: str,
    input_file_path: str,
    output_base_dir: str,
    provider_name: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Full AI 3D generation pipeline.
    Returns the completed manifest dict.
    """
    from modules.operations.settings import settings

    provider_name = provider_name or settings.ai_3d_default_provider
    provider = _get_provider(provider_name)
    opts = options or {}

    input_dir    = Path(output_base_dir) / "input"
    derived_dir  = Path(output_base_dir) / "derived"
    manifests_dir = Path(output_base_dir) / "manifests"
    for d in (input_dir, derived_dir, manifests_dir):
        d.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    errors: list[str] = []
    selected_frame_path: Optional[str] = None
    prepared_image_path: Optional[str] = None
    output_glb_path: Optional[str] = None
    preview_image_path: Optional[str] = None
    provider_result: Dict[str, Any] = {}
    preprocessing_meta: Dict[str, Any] = {}
    postprocessing_meta: Dict[str, Any] = {}

    # ── 1. Copy input ─────────────────────────────────────────────────────────
    suffix = Path(input_file_path).suffix or ".jpg"
    dest_input = input_dir / f"upload{suffix}"
    shutil.copy2(input_file_path, str(dest_input))

    # ── 2. Route & frame selection ────────────────────────────────────────────
    from modules.depth_studio.input_router import route_input
    try:
        input_type, _ = route_input(input_file_path)
    except ValueError:
        input_type = "image"

    if input_type == "video":
        frame_path = str(derived_dir / "selected_frame.jpg")
        try:
            from modules.depth_studio.video_frame_selector import select_best_frame
            ok, reason, _ = select_best_frame(str(dest_input), frame_path)
            if ok:
                selected_frame_path = frame_path
                warnings.append("input_video_best_frame_used")
                image_path_for_gen = frame_path
            else:
                warnings.append(f"frame_selection_failed:{reason}")
                image_path_for_gen = None
        except Exception as exc:
            warnings.append(f"frame_selection_error:{exc}")
            image_path_for_gen = None
    else:
        image_path_for_gen = str(dest_input)

    if not image_path_for_gen:
        errors.append("no_usable_input_image")
        manifest = _build_failed_manifest(
            session_id, input_file_path, input_type, provider, errors, warnings,
            selected_frame_path,
        )
        write_manifest(manifest, str(manifests_dir))
        return manifest

    # ── 3. Preprocess ─────────────────────────────────────────────────────────
    if settings.ai_3d_preprocess_enabled:
        preprocessing_meta = preprocess_input(
            source_image_path=image_path_for_gen,
            output_dir=str(derived_dir),
            input_size=opts.get("input_size", settings.sf3d_input_size),
        )
        prepared_image_path = preprocessing_meta.get("prepared_image_path")
        warnings.extend(preprocessing_meta.get("warnings", []))
    else:
        prepared_image_path = image_path_for_gen
        preprocessing_meta = {"enabled": False}

    generation_input = prepared_image_path or image_path_for_gen

    # ── 4. Generate ───────────────────────────────────────────────────────────
    provider_result = provider.safe_generate(
        input_image_path=generation_input,
        output_dir=str(derived_dir),
        options=opts,
    )
    warnings.extend(provider_result.get("warnings", []))
    if provider_result.get("error"):
        errors.append(provider_result["error"])

    output_glb_path    = provider_result.get("output_path")
    preview_image_path = provider_result.get("preview_image_path")

    # ── 5. Postprocess ────────────────────────────────────────────────────────
    postprocessing_meta = run_postprocess(
        glb_path=output_glb_path,
        enabled=settings.ai_3d_postprocess_enabled,
    )

    # ── 6. Quality gate ───────────────────────────────────────────────────────
    review_required = settings.sf3d_require_review or settings.ai_3d_require_review
    gate = quality_evaluate(provider_result, output_glb_path, review_required)
    warnings.extend(gate.get("warnings", []))
    final_status = gate["verdict"]

    # ── 7. Manifest ───────────────────────────────────────────────────────────
    _prov_status = provider_result.get("status", "failed")
    manifest = build_manifest(
        session_id=session_id,
        source_input_path=input_file_path,
        input_type=input_type,
        provider=provider.name,
        provider_status=_prov_status,
        model_name=provider_result.get("model_name"),
        license_note=provider.license_note,
        selected_frame_path=selected_frame_path,
        prepared_image_path=prepared_image_path,
        preprocessing=preprocessing_meta,
        postprocessing=postprocessing_meta,
        quality_gate=gate,
        output_glb_path=output_glb_path,
        output_format=provider.output_format,
        preview_image_path=preview_image_path,
        status=final_status,
        warnings=list(dict.fromkeys(warnings)),
        errors=errors,
        review_required=review_required,
        # Phase 4D
        execution_mode=getattr(settings, "sf3d_execution_mode", "disabled"),
        worker_metadata=provider_result.get("metadata", {}),
        provider_failure_reason=(
            provider_result.get("error")
            if _prov_status != "ok" else None
        ),
    )
    write_manifest(manifest, str(manifests_dir))

    logger.info(
        "AI 3D generation: provider=%s status=%s gate=%s",
        provider.name, provider_result.get("status"), final_status,
    )
    return manifest


def _build_failed_manifest(
    session_id, input_file_path, input_type, provider,
    errors, warnings, selected_frame_path,
) -> Dict[str, Any]:
    from .quality_gate import _gate
    return build_manifest(
        session_id=session_id,
        source_input_path=input_file_path,
        input_type=input_type,
        provider=provider.name,
        provider_status="failed",
        model_name=None,
        license_note=provider.license_note,
        selected_frame_path=selected_frame_path,
        prepared_image_path=None,
        preprocessing={},
        postprocessing={},
        quality_gate=_gate("failed", False, [], reason="no_usable_input"),
        output_glb_path=None,
        output_format=provider.output_format,
        preview_image_path=None,
        status="failed",
        warnings=warnings,
        errors=errors,
    )
