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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from .manifest import build_manifest, write_manifest
from .input_preprocessor import preprocess_input
from .postprocess import run_postprocess
from .quality_gate import evaluate as quality_evaluate
from .quality_profiles import resolve_quality_profile

logger = logging.getLogger("ai_3d_generation.pipeline")


def _get_provider(provider_name: str):
    """Resolve the provider by name.

    Accepted providers: sf3d, rodin.
    Unknown names raise ValueError — no silent fallback.
    """
    if provider_name == "sf3d":
        from .sf3d_provider import SF3DProvider
        return SF3DProvider()
    if provider_name == "rodin":
        from .rodin_provider import RodinProvider
        return RodinProvider()
    raise ValueError(f"unknown_ai3d_provider:{provider_name}")


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

    # Resolve quality profile
    q_mode = opts.get("quality_mode") or settings.ai_3d_quality_mode
    try:
        resolved_quality = resolve_quality_profile(q_mode, settings, opts)
    except Exception as exc:
        logger.warning(f"Quality profile resolution failed: {exc}. Falling back to settings.")
        resolved_quality = {
            "input_size": opts.get("input_size", settings.sf3d_input_size),
            "texture_resolution": opts.get("texture_resolution", settings.sf3d_texture_resolution),
            "max_candidates": settings.ai_3d_max_candidates,
            "video_topk_frames": settings.ai_3d_video_topk_frames,
            "remesh": opts.get("remesh", settings.sf3d_remesh),
            "quality_mode": "fallback",
            "warnings": [f"resolution_error:{exc}"]
        }
    
    # Inject resolved quality into options for provider
    opts["input_size"] = resolved_quality["input_size"]
    opts["texture_resolution"] = resolved_quality["texture_resolution"]
    opts["remesh"] = resolved_quality["remesh"]

    # Resolve background removal toggle
    bg_removal = opts.get("background_removal_enabled")
    if bg_removal is None:
        bg_removal = getattr(settings, "ai_3d_background_removal_enabled", False)

    # Consent check for external providers
    external_providers = ("rodin", "meshy", "tripo")
    if provider.name in external_providers:
        consent = opts.get("external_provider_consent")
        if not consent:
            logger.error(f"Consent missing for external provider: {provider.name}")
            manifest = _build_failed_manifest(
                session_id, input_file_path, "image", provider, 
                ["external_provider_consent_required"], [], None
            )
            # Write it so it can be picked up
            manifests_dir = Path(output_base_dir) / "manifests"
            manifests_dir.mkdir(parents=True, exist_ok=True)
            write_manifest(manifest, str(manifests_dir))
            return manifest

    input_dir    = Path(output_base_dir) / "input"
    derived_dir  = Path(output_base_dir) / "derived"
    manifests_dir = Path(output_base_dir) / "manifests"
    for d in (input_dir, derived_dir, manifests_dir):
        d.mkdir(parents=True, exist_ok=True)
    derived_dir_abs = derived_dir.resolve()

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
    src_input = Path(input_file_path)
    try:
        src_resolved = src_input.resolve()
        dest_resolved = dest_input.resolve()
    except Exception:
        src_resolved = src_input
        dest_resolved = dest_input

# Upload endpoint may already store the file at data/ai_3d/<session>/input/upload.*
# In that case copy2(src, src) raises SameFileError; skip the copy safely.
    if src_resolved != dest_resolved:
        shutil.copy2(str(src_resolved), str(dest_input))
    else:
        logger.info("AI 3D input already in session input dir; skipping copy: %s", dest_input)
    # ── 2. Route & candidate resolution ───────────────────────────────────────
    from modules.depth_studio.input_router import route_input
    try:
        input_type, _ = route_input(input_file_path)
    except ValueError:
        input_type = "image"

    from .multi_input import load_session_inputs, resolve_candidate_sources
    session_inputs = load_session_inputs(output_base_dir)

    candidates = []
    candidate_ranking = []
    selected_candidate_id = None
    selected_candidate_reason = None
    uploaded_files_count = session_inputs.get("uploaded_files_count", 1) if session_inputs else 1
    generation_input: Optional[str] = None
    _started_at = None
    _finished_at = None
    _duration_sec = None

    if settings.ai_3d_multi_candidate_enabled and (session_inputs or input_type == "video"):
        logger.info("Using multi-candidate flow for session %s", session_id)
        
        res = resolve_candidate_sources(output_base_dir, str(dest_input), session_inputs)
        input_type = res["input_mode"]
        sources = res["sources"]
        uploaded_files_count = res.get("uploaded_files_count", 1)
        
        if input_type == "video":
            from .video_candidates import select_top_k_frames
            sources = select_top_k_frames(
                video_path=sources[0],
                out_dir=str(derived_dir / "extracted_frames"),
                top_k=resolved_quality["video_topk_frames"],
                min_spacing_sec=settings.ai_3d_video_frame_min_spacing_sec,
            )
            if not sources:
                errors.append("no_usable_video_frames")
                manifest = _build_failed_manifest(
                    session_id, input_file_path, input_type, provider, errors, warnings,
                    None,
                )
                write_manifest(manifest, str(manifests_dir))
                return manifest
                
        from .candidate_runner import run_candidates_sequential
        
        _t0 = time.monotonic()
        _started_at = datetime.now(tz=timezone.utc).isoformat()
        
        candidates = run_candidates_sequential(
            session_dir=output_base_dir,
            source_paths=sources,
            provider=provider,
            options=opts,
            max_candidates=resolved_quality["max_candidates"],
            input_size=resolved_quality["input_size"],
            input_mode=input_type,
            bbox_padding_ratio=resolved_quality.get("bbox_padding_ratio", 0.12),
            background_removal_enabled=bg_removal,
        )
        
        _t1 = time.monotonic()
        _finished_at = datetime.now(tz=timezone.utc).isoformat()
        _duration_sec = round(_t1 - _t0, 2)
        
        from .candidate_selector import select_best
        best, candidate_ranking, selected_candidate_reason = select_best(candidates)
        
        if not best:
            errors.append("all_candidates_failed")
            provider_result = {
                "status": "failed",
                "output_path": None,
                "preview_image_path": None,
                "warnings": warnings,
                "error": "all_candidates_failed",
                "metadata": {}
            }
            output_glb_path = None
            prepared_image_path = None
            selected_frame_path = None
        else:
            best["selected"] = True
            selected_candidate_id = best["candidate_id"]
            
            if best.get("output_glb_path") and Path(best["output_glb_path"]).exists():
                shutil.copy2(best["output_glb_path"], str(derived_dir / "output.glb"))
                output_glb_path = str(derived_dir / "output.glb")
            else:
                output_glb_path = None
                
            if best.get("prepared_image_path") and Path(best["prepared_image_path"]).exists():
                shutil.copy2(best["prepared_image_path"], str(derived_dir / "ai3d_input.png"))
                prepared_image_path = str(derived_dir / "ai3d_input.png")
            else:
                prepared_image_path = None
                
            if input_type == "video" and best.get("source_path") and Path(best["source_path"]).exists():
                shutil.copy2(best["source_path"], str(derived_dir / "selected_frame.jpg"))
                selected_frame_path = str(derived_dir / "selected_frame.jpg")

            provider_result = {
                "status": best.get("provider_status", "failed"),
                "output_path": output_glb_path,
                "preview_image_path": None,
                "warnings": best.get("warnings", []),
                "error": None,
                "metadata": best.get("worker_metadata", {}),
                "model_name": best.get("model_name") or provider.name,
            }
            warnings.extend(best.get("warnings", []))
            if best.get("errors"):
                errors.extend(best["errors"])
                
            generation_input = best.get("prepared_image_path") or best.get("source_path")
            preprocessing_meta = best.get("preprocessing") or {"enabled": True}

    else:
        logger.info("Using single-candidate flow for session %s", session_id)
        # ── 2 (Legacy). Route & frame selection ───────────────────────────────────
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
                input_size=resolved_quality["input_size"],
                bbox_padding_ratio=resolved_quality.get("bbox_padding_ratio", 0.12),
                background_removal_enabled=bg_removal,
            )
            prepared_image_path = preprocessing_meta.get("prepared_image_path")
            warnings.extend(preprocessing_meta.get("warnings", []))
        else:
            prepared_image_path = image_path_for_gen
            preprocessing_meta = {"enabled": False}
    
        generation_input = prepared_image_path or image_path_for_gen
    
        # ── 4. Generate ───────────────────────────────────────────────────────────
        # Normalise generation_input to absolute path before handing to provider
        if generation_input:
            try:
                generation_input = str(Path(generation_input).resolve())
            except Exception:
                pass
    
        _t0 = time.monotonic()
        _started_at = datetime.now(tz=timezone.utc).isoformat()
    
        provider_result = provider.safe_generate(
            input_image_path=generation_input,
            output_dir=str(derived_dir_abs),
            options=opts,
        )
    
        _t1 = time.monotonic()
        _finished_at = datetime.now(tz=timezone.utc).isoformat()
        _duration_sec = round(_t1 - _t0, 2)
    
        warnings.extend(provider_result.get("warnings", []))
        if provider_result.get("error"):
            errors.append(provider_result["error"])
    
        output_glb_path    = provider_result.get("output_path")
        preview_image_path = provider_result.get("preview_image_path")
    
        # Normalise GLB path to absolute
        if output_glb_path:
            try:
                output_glb_path = str(Path(output_glb_path).resolve())
            except Exception:
                pass

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

    # ── compute output_size_bytes ─────────────────────────────────────────────
    _output_size_bytes: Optional[int] = None
    if output_glb_path:
        try:
            _glb_p = Path(output_glb_path)
            if _glb_p.exists():
                _output_size_bytes = _glb_p.stat().st_size
        except Exception:
            pass

    # ── compute missing_outputs ───────────────────────────────────────────────
    _expected_outputs = {
        "prepared_input": derived_dir_abs / "ai3d_input.png",
        "output_glb":     derived_dir_abs / "output.glb",
    }
    _missing_outputs: List[str] = [
        k for k, p in _expected_outputs.items() if not p.exists()
    ]

    # ── build path_diagnostics ────────────────────────────────────────────────
    _path_diag: Dict[str, Any] = {
        "source_input_path":    str(Path(input_file_path).resolve()),
        "generation_input":     generation_input,
        "output_dir":           str(derived_dir_abs),
        "output_glb_path":      output_glb_path,
    }
    _exec_mode = getattr(settings, "sf3d_execution_mode", "disabled")
    if _exec_mode == "wsl_subprocess":
        try:
            from .sf3d_provider import _windows_to_wsl_path
            _path_diag["generation_input_wsl"] = (
                _windows_to_wsl_path(generation_input) if generation_input else None
            )
            _path_diag["output_dir_wsl"] = _windows_to_wsl_path(str(derived_dir_abs))
        except Exception:
            pass

    def _normalize_input_mode(t: str) -> str:
        return "single_image" if t == "image" else t

    manifest = build_manifest(
        session_id=session_id,
        source_input_path=input_file_path,
        input_type=_normalize_input_mode(input_type),
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
        execution_mode=_exec_mode,
        worker_metadata=provider_result.get("metadata", {}),
        provider_failure_reason=(
            provider_result.get("error")
            if _prov_status != "ok" else None
        ),
        missing_outputs=_missing_outputs,
        generation_started_at=_started_at,
        generation_finished_at=_finished_at,
        duration_sec=_duration_sec,
        output_size_bytes=_output_size_bytes,
        path_diagnostics=_path_diag,

        # Phase 1.5 External Provider
        external_provider=provider_result.get("metadata", {}).get("external_provider", False),
        external_provider_name=provider_result.get("metadata", {}).get("external_provider_name"),
        external_task_id=provider_result.get("metadata", {}).get("external_task_id"),
        external_status=provider_result.get("metadata", {}).get("external_status"),
        downloaded_output_glb_path=provider_result.get("metadata", {}).get("downloaded_output_glb_path"),
        privacy_notice=provider_result.get("metadata", {}).get("privacy_notice"),
        provider_latency_sec=provider_result.get("metadata", {}).get("provider_latency_sec"),
        provider_poll_count=provider_result.get("metadata", {}).get("provider_poll_count"),
        sanitized_error=provider_result.get("metadata", {}).get("sanitized_error"),
        external_provider_consent=opts.get("external_provider_consent") is True,

        # Phase 1 multi-candidate
        input_mode=_normalize_input_mode(input_type),

        uploaded_files_count=uploaded_files_count,
        candidate_count=len(candidates),
        selected_candidate_id=selected_candidate_id,
        selected_candidate_reason=selected_candidate_reason,
        candidate_ranking=candidate_ranking,
        candidates=candidates,
        quality_mode=resolved_quality["quality_mode"],
        resolved_quality=resolved_quality,
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
    
    def _normalize_input_mode(t: str) -> str:
        return "single_image" if t == "image" else t
        
    return build_manifest(
        session_id=session_id,
        source_input_path=input_file_path,
        input_type=_normalize_input_mode(input_type),
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
