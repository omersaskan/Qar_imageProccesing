"""
Phase 1 — Multi-input session resolver.

Pure helper module. Does NOT import FastAPI or any ML package.

Responsibilities:
  - Load and parse session_inputs.json from a session's input/ directory.
  - Detect input mode: single_image, video, multi_image.
  - Return a list of source file paths for downstream candidate processing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("ai_3d_generation.multi_input")

# File extensions recognised as video
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


def detect_input_mode(file_path: str) -> str:
    """Determine mode from a single file path's extension."""
    ext = Path(file_path).suffix.lower()
    if ext in _VIDEO_EXTENSIONS:
        return "video"
    return "single_image"


def load_session_inputs(session_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load session_inputs.json from ``<session_dir>/input/session_inputs.json``.

    Returns the parsed dict or None if the file does not exist.
    Schema expected::

        {
          "input_mode": "multi_image" | "video" | "single_image",
          "uploaded_files_count": int,
          "input_files": ["upload_001.jpg", ...]
        }
    """
    manifest_path = Path(session_dir) / "input" / "session_inputs.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        logger.debug("Loaded session_inputs.json from %s: mode=%s count=%s",
                      session_dir, data.get("input_mode"), data.get("uploaded_files_count"))
        return data
    except Exception as exc:
        logger.warning("Failed to parse session_inputs.json at %s: %s", manifest_path, exc)
        return None


def resolve_candidate_sources(
    session_dir: str,
    input_file_path: str,
    session_inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a list of candidate source images from session state.

    Returns::

        {
            "input_mode": str,
            "sources": [str, ...],   # absolute paths to candidate source images
        }

    Logic:
      - If session_inputs is provided and input_mode == "multi_image",
        resolve each file in input_files relative to <session_dir>/input/.
      - If input_mode == "video", return the single video path (caller must
        run video_candidates.select_top_k_frames separately).
      - Otherwise fall back to the single input_file_path.
    """
    input_dir = Path(session_dir) / "input"

    if session_inputs and session_inputs.get("input_mode") == "multi_image":
        files = session_inputs.get("input_files", [])
        sources = []
        for fname in files:
            p = input_dir / fname
            if p.exists():
                sources.append(str(p.resolve()))
            else:
                logger.warning("multi_input source file missing: %s", p)
        if not sources:
            # Fallback to the provided input path
            logger.warning("No multi-image sources resolved; falling back to input_file_path")
            return {"input_mode": "single_image", "sources": [str(Path(input_file_path).resolve())]}
        return {"input_mode": "multi_image", "sources": sources}

    # Single file — detect video vs image
    mode = detect_input_mode(input_file_path)
    return {"input_mode": mode, "sources": [str(Path(input_file_path).resolve())]}


def write_session_inputs(
    session_dir: str,
    input_mode: str,
    input_files: List[str],
) -> str:
    """
    Write session_inputs.json to ``<session_dir>/input/session_inputs.json``.
    Returns the path to the written file.
    """
    input_dir = Path(session_dir) / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_mode": input_mode,
        "uploaded_files_count": len(input_files),
        "input_files": input_files,
    }
    out_path = input_dir / "session_inputs.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(out_path)
