"""Build and persist the ai_3d_generation manifest block."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


def build_manifest(
    session_id: str,
    source_input_path: str,
    input_type: str,
    provider: str,
    provider_status: str,
    model_name: Optional[str],
    license_note: str,
    selected_frame_path: Optional[str],
    prepared_image_path: Optional[str],
    preprocessing: Dict[str, Any],
    postprocessing: Dict[str, Any],
    quality_gate: Dict[str, Any],
    output_glb_path: Optional[str],
    output_format: str,
    preview_image_path: Optional[str],
    status: str,
    warnings: List[str],
    errors: List[str],
    review_required: bool = True,
    enabled: bool = True,
    # ── Phase 4D additions (all optional for backward compat) ─────────────────
    execution_mode: str = "disabled",
    worker_metadata: Optional[Dict[str, Any]] = None,
    provider_failure_reason: Optional[str] = None,
    missing_outputs: Optional[List[str]] = None,
    # ── Phase 4E additions (all optional for backward compat) ─────────────────
    generation_started_at: Optional[str] = None,
    generation_finished_at: Optional[str] = None,
    duration_sec: Optional[float] = None,
    output_size_bytes: Optional[int] = None,
    path_diagnostics: Optional[Dict[str, Any]] = None,
    # ── Phase 1 multi-candidate additions (all optional for backward compat) ──
    input_mode: Optional[str] = None,
    uploaded_files_count: Optional[int] = None,
    candidate_count: Optional[int] = None,
    selected_candidate_id: Optional[str] = None,
    selected_candidate_reason: Optional[str] = None,
    candidate_ranking: Optional[List[Dict[str, Any]]] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    quality_mode: Optional[str] = None,
    # ── Phase 1.5 External Provider additions ─────────────────────────────────
    external_provider: bool = False,
    external_provider_name: Optional[str] = None,
    external_task_id: Optional[str] = None,
    external_status: Optional[str] = None,
    external_model_urls: Optional[List[str]] = None,
    downloaded_output_glb_path: Optional[str] = None,
    cost_credits: Optional[float] = None,
    privacy_notice: Optional[str] = None,
    external_provider_consent: bool = False,
    provider_latency_sec: Optional[float] = None,
    provider_poll_count: Optional[int] = None,
    sanitized_error: Optional[str] = None,
) -> Dict[str, Any]:
    _worker_meta = worker_metadata or {}
    return {
        "enabled": enabled,
        "mode": "ai_generated_3d",
        "asset_type": "ai_generated",
        "session_id": session_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),

        # Provenance — must never be confused with real reconstruction
        "is_true_scan": False,
        "geometry_confidence": "estimated",
        "source": "single_image_or_best_frame",

        # Provider
        "provider": provider,
        "provider_status": provider_status,
        "model_name": model_name,
        "license_note": license_note,
        "execution_mode": execution_mode,

        # External Provider (Phase 1.5)
        "external_provider": external_provider,
        "external_provider_name": external_provider_name,
        "external_task_id": external_task_id,
        "external_status": external_status,
        "external_model_urls": external_model_urls or [],
        "downloaded_output_glb_path": downloaded_output_glb_path,
        "cost_credits": cost_credits,
        "privacy_notice": privacy_notice,
        "external_provider_consent": external_provider_consent,
        "provider_latency_sec": provider_latency_sec,
        "provider_poll_count": provider_poll_count,
        "sanitized_error": sanitized_error,

        # Input
        "source_input_path": source_input_path,
        "input_type": input_type,
        "selected_frame_path": selected_frame_path,
        "prepared_image_path": prepared_image_path,

        # Preprocessing / postprocessing / QA
        "preprocessing": preprocessing,
        "postprocessing": postprocessing,
        "quality_gate": quality_gate,

        # Output
        "output_glb_path": output_glb_path,
        "output_format": output_format,
        "preview_image_path": preview_image_path,
        "missing_outputs": missing_outputs or [],
        "output_size_bytes": output_size_bytes,

        # Worker runtime metadata
        "worker_metadata": _worker_meta,
        "peak_mem_mb": _worker_meta.get("peak_mem_mb"),

        # Runtime timing (Phase 4E)
        "generation_started_at": generation_started_at,
        "generation_finished_at": generation_finished_at,
        "duration_sec": duration_sec,

        # Path diagnostics (Phase 4E)
        "path_diagnostics": path_diagnostics or {},

        # Status
        "status": status,
        "review_required": review_required,
        "warnings": warnings,
        "errors": errors,
        "provider_failure_reason": provider_failure_reason,

        # Phase 1 — multi-candidate orchestration
        "input_mode": input_mode,
        "uploaded_files_count": uploaded_files_count,
        "candidate_count": candidate_count,
        "selected_candidate_id": selected_candidate_id,
        "selected_candidate_reason": selected_candidate_reason,
        "candidate_ranking": candidate_ranking or [],
        "candidates": candidates or [],
        "quality_mode": quality_mode,
    }


def write_manifest(manifest: Dict[str, Any], output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(output_dir) / "ai3d_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return str(out)
