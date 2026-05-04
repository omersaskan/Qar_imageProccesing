"""
Phase 1 — Sequential candidate runner.

Processes candidates one-at-a-time (never parallel) to respect the
single-job GPU guard in sf3d_provider.

Responsibilities:
  - For each candidate source image:
      1. Copy source to candidate directory.
      2. Run preprocess_input.
      3. Call provider.safe_generate.
      4. Collect metadata (status, duration, warnings, errors).
      5. Write per-candidate candidate_manifest.json.
  - Return a list of candidate metadata dicts.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from .input_preprocessor import preprocess_input
from .candidate_selector import score_candidate

logger = logging.getLogger("ai_3d_generation.candidate_runner")


def _ensure_candidate_dir(session_dir: str, cand_idx: int) -> Path:
    """Create and return the candidate directory path."""
    cand_id = f"cand_{cand_idx:03d}"
    cand_dir = Path(session_dir) / "derived" / "candidates" / cand_id
    cand_dir.mkdir(parents=True, exist_ok=True)
    return cand_dir


def run_candidates_sequential(
    session_dir: str,
    source_paths: List[str],
    provider,
    options: Optional[Dict[str, Any]] = None,
    max_candidates: int = 5,
    input_size: int = 512,
    input_mode: str = "single_image",
) -> List[Dict[str, Any]]:
    """
    Process each candidate source image sequentially.

    Parameters
    ----------
    session_dir : str
        Root session directory (e.g. ``data/ai_3d/<session_id>``).
    source_paths : list[str]
        Absolute paths to candidate source images.
    provider : AI3DProviderBase
        The AI 3D provider instance (will call .safe_generate).
    options : dict, optional
        Options passed through to safe_generate.
    max_candidates : int
        Maximum number of candidates to process.
    input_size : int
        Target input size for preprocessing.

    Returns
    -------
    list[dict]
        One metadata dict per candidate, each containing:
        candidate_id, source_path, prepared_image_path, output_glb_path,
        provider_status, status, score, score_breakdown, duration_sec,
        warnings, errors, selected (initially False).
    """
    opts = options or {}
    results: List[Dict[str, Any]] = []

    for i, src_path in enumerate(source_paths[:max_candidates], start=1):
        cand_id = f"cand_{i:03d}"
        cand_dir = _ensure_candidate_dir(session_dir, i)
        source_type = "single_image"
        if input_mode == "video":
            source_type = "video_frame"
        elif input_mode == "multi_image":
            source_type = "uploaded_image"

        cand_meta: Dict[str, Any] = {
            "candidate_id": cand_id,
            "source_path": src_path,
            "source_type": source_type,
            "prepared_image_path": None,
            "output_glb_path": None,
            "provider_status": "failed",
            "status": "failed",
            "score": 0.0,
            "score_breakdown": {},
            "duration_sec": 0.0,
            "peak_mem_mb": None,
            "warnings": [],
            "errors": [],
            "selected": False,
        }

        t0 = time.monotonic()

        try:
            # 1. Copy source to candidate dir
            src_ext = Path(src_path).suffix or ".jpg"
            dest_source = cand_dir / f"source{src_ext}"
            shutil.copy2(src_path, str(dest_source))
            logger.info("[%s] Copied source → %s", cand_id, dest_source)

            # 2. Preprocess
            prep_result = preprocess_input(
                source_image_path=str(dest_source),
                output_dir=str(cand_dir),
                input_size=input_size,
            )
            cand_meta["prepared_image_path"] = prep_result.get("prepared_image_path")
            cand_meta["warnings"].extend(prep_result.get("warnings", []))

            generation_input = prep_result.get("prepared_image_path") or str(dest_source)

            # Resolve to absolute path
            try:
                generation_input = str(Path(generation_input).resolve())
            except Exception:
                pass

            # 3. Generate via provider (sequential — no parallel calls)
            logger.info("[%s] Running provider.safe_generate …", cand_id)
            prov_result = provider.safe_generate(
                input_image_path=generation_input,
                output_dir=str(cand_dir.resolve()),
                options=opts,
            )

            cand_meta["provider_status"] = prov_result.get("status", "failed")
            cand_meta["warnings"].extend(prov_result.get("warnings", []))
            if prov_result.get("error"):
                cand_meta["errors"].append(prov_result["error"])

            output_path = prov_result.get("output_path")
            if output_path:
                try:
                    output_path = str(Path(output_path).resolve())
                except Exception:
                    pass
            cand_meta["output_glb_path"] = output_path

            # Extract peak memory and metadata
            worker_meta = prov_result.get("metadata", {})
            cand_meta["worker_metadata"] = worker_meta
            cand_meta["model_name"] = prov_result.get("model_name")
            cand_meta["peak_mem_mb"] = worker_meta.get("peak_mem_mb")
            
            output_size_bytes = worker_meta.get("output_size_bytes")
            if not output_size_bytes and output_path and Path(output_path).exists():
                output_size_bytes = Path(output_path).stat().st_size
            cand_meta["output_size_bytes"] = output_size_bytes

            # Determine candidate status
            if prov_result.get("status") == "ok" and output_path and Path(output_path).exists():
                cand_meta["status"] = "ok"
            else:
                cand_meta["status"] = "failed"

        except Exception as exc:
            logger.error("[%s] Candidate failed with exception: %s", cand_id, exc)
            cand_meta["errors"].append(str(exc))
            cand_meta["status"] = "failed"

        t1 = time.monotonic()
        cand_meta["duration_sec"] = round(t1 - t0, 2)

        # 4. Score
        score, breakdown = score_candidate(cand_meta)
        cand_meta["score"] = score
        cand_meta["score_breakdown"] = breakdown

        # 5. Write per-candidate manifest
        try:
            cand_manifest_path = cand_dir / "candidate_manifest.json"
            cand_manifest_path.write_text(
                json.dumps(cand_meta, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("[%s] Failed to write candidate_manifest.json: %s", cand_id, exc)

        results.append(cand_meta)
        logger.info(
            "[%s] status=%s score=%.1f duration=%.1fs",
            cand_id, cand_meta["status"], score, cand_meta["duration_sec"],
        )

    return results
