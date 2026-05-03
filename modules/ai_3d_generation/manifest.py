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
) -> Dict[str, Any]:
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

        # Status
        "status": status,
        "review_required": review_required,
        "warnings": warnings,
        "errors": errors,
    }


def write_manifest(manifest: Dict[str, Any], output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(output_dir) / "ai3d_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return str(out)
