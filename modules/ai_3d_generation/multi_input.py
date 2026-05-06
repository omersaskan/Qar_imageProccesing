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
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_VALID_INPUT_MODES = {"single_image", "video", "multi_image"}
_VALID_PROVIDERS = {"sf3d", "rodin", "meshy", "tripo", "hunyuan3d_21"}



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
        if not isinstance(data, dict):
            logger.warning("session_inputs.json is not a dict")
            return None
        if data.get("input_mode") not in _VALID_INPUT_MODES:
            logger.warning("Invalid input_mode in session_inputs.json")
            return None
        if not isinstance(data.get("input_files"), list):
            logger.warning("input_files is not a list")
            return None
        if data.get("provider") and data.get("provider") not in _VALID_PROVIDERS:
            logger.error("Invalid provider in session_inputs.json: %s", data.get("provider"))
            raise ValueError(f"Invalid provider in session_inputs.json: {data.get('provider')}")
        
        logger.debug("Loaded session_inputs.json from %s: mode=%s count=%s provider=%s",
                      session_dir, data.get("input_mode"), data.get("uploaded_files_count"), data.get("provider"))
        return data

    except ValueError:
        raise
    except Exception as exc:
        logger.warning("Failed to parse session_inputs.json at %s: %s", manifest_path, exc)
        return None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


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
        uploaded_files_count = session_inputs.get("uploaded_files_count", len(files))
        
        for fname in files:
            p = input_dir / fname
            
            # Path traversal guard
            try:
                resolved_p = p.resolve(strict=False)
                if not _is_relative_to(resolved_p, input_dir.resolve()):
                    logger.warning("Path traversal attempt in multi_input: %s", fname)
                    continue
            except Exception:
                continue
                
            # Skip non-images in multi_image mode
            if resolved_p.suffix.lower() not in _IMAGE_EXTENSIONS:
                logger.warning("Skipping non-image file in multi_image mode: %s", fname)
                continue
                
            if resolved_p.exists():
                sources.append(str(resolved_p))
            else:
                logger.warning("multi_input source file missing: %s", resolved_p)
                
        if not sources:
            # Fallback to the provided input path
            logger.warning("No multi-image sources resolved; falling back to input_file_path")
            return {"input_mode": "single_image", "sources": [str(Path(input_file_path).resolve())], "uploaded_files_count": 1}
        return {"input_mode": "multi_image", "sources": sources, "uploaded_files_count": uploaded_files_count}

    # Single file — detect video vs image
    mode = detect_input_mode(input_file_path)
    return {"input_mode": mode, "sources": [str(Path(input_file_path).resolve())], "uploaded_files_count": 1}


def write_session_inputs(
    session_dir: str,
    input_mode: str,
    input_files: List[str],
    provider: Optional[str] = None,
) -> str:
    """
    Write session_inputs.json to ``<session_dir>/input/session_inputs.json``.
    Returns the path to the written file.
    """
    if input_mode not in _VALID_INPUT_MODES:
        raise ValueError(f"Invalid input_mode: {input_mode}")
    if input_mode == "multi_image" and not input_files:
        raise ValueError("multi_image mode requires at least one input file")
    if provider and provider not in _VALID_PROVIDERS:
        raise ValueError(f"Invalid provider: {provider}")

    input_dir = Path(session_dir) / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_files_basenames = [Path(f).name for f in input_files]
    manifest = {
        "input_mode": input_mode,
        "uploaded_files_count": len(input_files_basenames),
        "input_files": input_files_basenames,
        "provider": provider,
    }
    out_path = input_dir / "session_inputs.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(out_path)

