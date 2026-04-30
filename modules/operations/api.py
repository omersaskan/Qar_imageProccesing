"""
modules/operations/api.py

SPRINT 1 — TICKET-004: Disk Space + Binary Preflight Hardening

Changes:
  - /api/ready: COLMAP binary is now probed with --help (not just path existence).
    Disk space is checked and surfaced. Both are shown in the response.
  - /api/sessions/upload: Disk space preflight blocks upload in pilot/production
    when free space is below settings.min_free_disk_gb.
  - /api/products list_products: Uses settings.data_root instead of hardcoded "data/registry/meta".
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import json
import shutil
import uuid

from modules.asset_registry.registry import AssetRegistry
from modules.capture_workflow.session_manager import SessionManager
from modules.shared_contracts.lifecycle import AssetStatus
from modules.operations.logging_config import get_component_logger, setup_logging
from modules.operations.worker import worker_instance
from modules.operations.settings import settings, AppEnvironment
import mimetypes
import cv2
import numpy as np
from modules.ai_segmentation.preview_providers import get_preview_provider

# Initialize unified logging
setup_logging()

logger = get_component_logger("api")

embedded_worker_enabled = os.getenv("MESHYSIZ_EMBEDDED_WORKER", "true").lower() == "true"


# SPRINT 3 TICKET-012: Replaced deprecated @app.on_event with lifespan handler.
# This eliminates the FastAPI DeprecationWarning that polluted test output.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    try:
        settings.validate_setup()
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        # Validation failure is logged but does not abort startup;
        # /api/ready will surface the problem to the operator.

    if embedded_worker_enabled:
        worker_instance.start()

    yield  # Application runs

    # ── Shutdown ─────────────────────────────────────────────────────────────
    if embedded_worker_enabled:
        worker_instance.stop()


app = FastAPI(title="Meshysiz Asset Factory API", lifespan=lifespan)

# Enable CORS with settings-controlled allowlist
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Dependency
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if settings.is_dev:
        return  # Optional in dev

    if not x_api_key or x_api_key != settings.pilot_api_key:
        logger.warning(
            f"Unauthorized API access attempt from {settings.env.value} environment"
        )
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")


registry = AssetRegistry(data_root=str(Path(settings.data_root) / "registry"))
session_manager = SessionManager(data_root=settings.data_root)


@app.get("/api/health")
async def health_check():
    """Lightweight alive check."""
    return {"status": "ok", "env": settings.env.value}


@app.get("/api/ready", dependencies=[Depends(verify_api_key)])
async def readiness_check():
    """
    Checks if the system is fully configured and ready for production jobs.

    Sprint 1 additions:
      - COLMAP binary is probed with --help (not just path existence).
      - Disk space is checked and included in the response.
    """
    issues = []

    # 1. Structural Checks
    dr = Path(settings.data_root)
    if not dr.exists():
        issues.append(f"Data root missing: {dr}")

    # ── TICKET-004: Binary probe (--help) instead of mere path check ──────
    colmap_probe = settings.probe_colmap_binary()
    if not colmap_probe["ok"]:
        issues.append(f"COLMAP binary not usable: {colmap_probe['error']}")

    # ── OpenMVS Readiness Probe ───────────────────────────────────────────
    openmvs_probe = settings.probe_openmvs_binaries()
    if not openmvs_probe["ok"]:
        issues.append(f"OpenMVS binary not usable: {openmvs_probe['error']}")

    # ── FFmpeg / ffprobe Readiness Probes ──────────────────────────────────
    ffmpeg_probe = settings.probe_ffmpeg()
    if not ffmpeg_probe["ok"]:
        issues.append(f"FFmpeg binary not usable: {ffmpeg_probe['error']}")
    
    ffprobe_probe = settings.probe_ffprobe()
    if not ffprobe_probe["ok"]:
        issues.append(f"ffprobe binary not usable: {ffprobe_probe['error']}")

    # ── Pilot/Prod Simulated Guard ─────────────────────────────────────────
    if not settings.is_dev:
        if getattr(settings, 'recon_pipeline', '') == "simulated":
            issues.append(f"Simulated reconstruction pipeline is not allowed in {getattr(settings.env, 'value', settings.env)}.")

    # ── TICKET-004: Disk space reporting ──────────────────────────────────
    free_disk_gb = settings.check_free_disk_gb()
    disk_ok = free_disk_gb >= settings.min_free_disk_gb
    if not disk_ok:
        issues.append(
            f"Low disk space: {free_disk_gb:.1f} GB free, "
            f"minimum required: {settings.min_free_disk_gb} GB"
        )

    # 2. Dependency Checks
    missing_ml = settings.check_ml_deps()
    missing_proc = settings.check_processing_deps()

    if missing_ml:
        issues.append(f"Missing ML Segmentation dependencies: {', '.join(missing_ml)}")
    if missing_proc:
        issues.append(f"Missing Critical Processing dependencies: {', '.join(missing_proc)}")

    # Determination: in Pilot/Prod any issue is a hard stop.
    is_ready = True
    if issues and not settings.is_dev:
        is_ready = False

    return {
        "status": "ready" if is_ready else "not_ready",
        "env": settings.env.value,
        "issues": issues,
        "dependencies": {
            "ml_segmentation_ready": not bool(missing_ml),
            "critical_processing_ready": not bool(missing_proc),
        },
        "preflight": {
            "colmap_probe_ok": colmap_probe["ok"],
            "colmap_version_line": colmap_probe.get("version_line"),
            "openmvs_probe_ok": openmvs_probe["ok"],
            "ffmpeg_probe_ok": ffmpeg_probe["ok"],
            "ffmpeg_version_line": ffmpeg_probe.get("version_line"),
            "ffprobe_probe_ok": ffprobe_probe["ok"],
            "free_disk_gb": round(free_disk_gb, 2) if free_disk_gb != float("inf") else None,
            "min_required_disk_gb": settings.min_free_disk_gb,
            "disk_ok": disk_ok,
        },
    }


@app.post("/api/ar/mask-preview", dependencies=[Depends(verify_api_key)])
async def get_mask_preview(
    file: UploadFile = File(...),
):
    """
    Lightweight endpoint for real-time AR mask preview.
    Does NOT create a session or persist data.
    """
    if not settings.sam_mask_preview_enabled:
        return {
            "provider": settings.segmentation_preview_provider,
            "mask_format": "polygon",
            "mask": [],
            "confidence": 0.0,
            "fallback_used": True,
            "detail": "SAM_MASK_PREVIEW_ENABLED is false"
        }

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Resize for performance if needed
        h, w = img.shape[:2]
        max_size = settings.sam_mask_preview_max_image_size
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        provider = get_preview_provider()
        mask_data = provider.get_mask(img)
        
        return mask_data
    except HTTPException:
        # Re-raise HTTP exceptions (like 400 for invalid image) so they aren't caught by the fallback block
        raise
    except Exception as e:
        logger.error(f"Mask preview failed during inference: {e}")
        return {
            "provider": settings.segmentation_preview_provider,
            "mask_format": "polygon",
            "mask": [],
            "confidence": 0.0,
            "fallback_used": True,
            "error": str(e)
        }


@app.post("/api/sessions/upload", dependencies=[Depends(verify_api_key)])
async def upload_video(
    product_id: str = Form(...),
    operator_id: str = Form("dashboard_user"),
    quality_manifest: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """
    Handles video upload, creates a new CaptureSession, and saves the file.
    Includes strict FFmpeg normalization and quality manifest validation.
    """
    from modules.utils.path_safety import validate_identifier
    from modules.utils.video_utils import normalize_video, validate_video_file

    # ── 1. Identifier Validation ──────────────────────────────────────────
    try:
        validate_identifier(product_id)
        validate_identifier(operator_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid identifier: {str(e)}")

    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
        raise HTTPException(
            status_code=400,
            detail="Invalid video format. Supported: .mp4, .mov, .avi, .webm",
        )

    session_id = f"cap_{uuid.uuid4().hex[:8]}"
    capture_path = session_manager.get_capture_path(session_id)
    session_created = False

    def cleanup_on_failure(reason: str):
        if not session_created:
            if capture_path.exists():
                try:
                    rejected_root = Path(settings.data_root) / "rejected"
                    rejected_root.mkdir(parents=True, exist_ok=True)
                    target_path = rejected_root / session_id
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.move(str(capture_path), str(target_path))
                    # Save the reason
                    with open(target_path / "rejection_reason.txt", "w") as f:
                        f.write(reason)
                    logger.info(f"Rejected session {session_id} moved to rejected folder for diagnostics. Reason: {reason}")
                except Exception as e:
                    logger.error(f"Failed to move rejected session to rejected folder: {e}")
                    shutil.rmtree(capture_path, ignore_errors=True)
        else:
            session_manager.update_session(
                session_id, 
                new_status=AssetStatus.FAILED, 
                failure_reason=f"Upload processing failed: {reason}"
            )

    # ── 2. Environment & Disk Preflight ───────────────────────────────────
    ffmpeg_resolved = settings.resolve_executable(settings.ffmpeg_path)
    ffprobe_resolved = settings.resolve_executable(settings.ffprobe_path)
    
    if not ffmpeg_resolved or not ffprobe_resolved:
        missing = []
        if not ffmpeg_resolved: missing.append("ffmpeg")
        if not ffprobe_resolved: missing.append("ffprobe")
        raise HTTPException(
            status_code=503, 
            detail=f"System Environment Incomplete: {', '.join(missing)} binary missing. Contact administrator."
        )

    missing_ml = settings.check_ml_deps()
    missing_proc = settings.check_processing_deps()

    if (missing_ml or missing_proc) and not settings.is_dev:
        raise HTTPException(status_code=503, detail="System Environment Incomplete")

    if not settings.is_dev:
        free_disk_gb = settings.check_free_disk_gb()
        if free_disk_gb < settings.min_free_disk_gb:
            raise HTTPException(status_code=507, detail="Insufficient disk space")

    # ── 3. Storage Setup & Direct Streaming ───────────────────────────────
    video_dir = capture_path / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    (capture_path / "reports").mkdir(parents=True, exist_ok=True)

    # Save original directly to captures (shows progress on disk)
    original_ext = Path(file.filename).suffix
    original_path = video_dir / f"original_capture{original_ext}"
    
    try:
        logger.info(f"Streaming upload for {session_id} directly to {original_path}")
        with open(original_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        file_size_mb = os.path.getsize(original_path) / (1024 * 1024)
        if file_size_mb == 0 or file_size_mb < 0.1:
            raise HTTPException(status_code=400, detail="Video file is too small or empty.")
        if file_size_mb > settings.max_upload_mb:
            raise HTTPException(status_code=400, detail="File size exceeds maximum.")

        # ── 4. Normalization ──────────────────────────────────────────────────
        video_path = video_dir / "raw_video.mp4"
        
        # FFmpeg normalization handles WebM/MOV/AVI and converts to standard H.264
        normalize_video(
            original_path, 
            video_path, 
            ffmpeg_path=ffmpeg_resolved, 
            ffprobe_path=ffprobe_resolved,
            timeout=settings.video_normalize_timeout_sec
        )
        
        # ── 5. Post-Normalization Validation ──────────────────────────────────
        # Now that it's a standard MP4, OpenCV MUST be able to read it.
        ok, error, video_meta = validate_video_file(
            video_path, 
            min_fps=settings.min_video_fps,
            min_duration=settings.min_video_duration_sec,
            max_duration=settings.max_video_duration_sec
        )
        if not ok:
            raise HTTPException(status_code=400, detail=f"Video validation failed: {error}")

        # Basic resolution gate (Orientation Agnostic)
        short_edge = min(video_meta["width"], video_meta["height"])
        long_edge = max(video_meta["width"], video_meta["height"])
        
        if short_edge < settings.min_video_short_edge or long_edge < settings.min_video_long_edge:
             msg = (f"Video resolution too low: {video_meta['width']}x{video_meta['height']}. "
                    f"Short edge must be >= {settings.min_video_short_edge}, Long edge must be >= {settings.min_video_long_edge}")
             logger.warning(f"Upload rejected for {session_id}: {msg}")
             raise HTTPException(
                 status_code=400, 
                 detail=msg
             )

        # ── 6. Quality Manifest Validation ────────────────────────────────────
        if not quality_manifest or quality_manifest.strip() in ("", "null"):
            raise HTTPException(status_code=422, detail="Missing quality_manifest for AR capture.")

        try:
            manifest_data = json.loads(quality_manifest)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Malformed JSON in quality_manifest")
        
        # Demo mode guard
        if manifest_data.get("is_demo") and not settings.is_dev:
             raise HTTPException(status_code=422, detail="Demo mode not permitted in production.")

        # Manifest numeric sanity
        m_total_frames = manifest_data.get("total_frame_count", 0)
        m_accepted_frames = manifest_data.get("accepted_frame_count", 0)
        m_blur_rejections = manifest_data.get("blur_rejection_count", 0)
        
        if m_total_frames <= 0:
            raise HTTPException(status_code=422, detail="Manifest total_frame_count must be > 0")
        if m_accepted_frames < 0 or m_accepted_frames > m_total_frames:
            raise HTTPException(status_code=422, detail="Manifest accepted_frame_count inconsistency")
        if m_blur_rejections < 0 or m_blur_rejections > m_total_frames:
            raise HTTPException(status_code=422, detail="Manifest blur_rejection_count inconsistency")
            
        # Manifest-vs-Video consistency
        v_frame_count = video_meta["frame_count"]
        tolerance = settings.ar_manifest_frame_count_tolerance
        if m_total_frames > v_frame_count * (1.0 + tolerance):
            msg = f"Manifest total_frame_count ({m_total_frames}) exceeds video frame count ({v_frame_count}) by >{tolerance*100}%"
            logger.warning(f"Upload rejected for {session_id}: {msg}")
            raise HTTPException(
                status_code=422, 
                detail=msg
            )
        
        # ── 7. AR Quality Gate Enforcement ───────────────────────────────────
        rejection_reasons = []
        coverage = manifest_data.get("coverage_summary", {}).get("percent", 0)
        max_gap = manifest_data.get("coverage_summary", {}).get("maxGap", 360)
        
        if coverage < settings.ar_min_coverage:
            rejection_reasons.append(f"Coverage too low: {coverage:.1f}%")
        if max_gap > settings.ar_max_gap:
            rejection_reasons.append(f"Gap too large: {max_gap:.1f}°")
        if m_accepted_frames < settings.ar_min_accepted_frames:
            rejection_reasons.append(f"Insufficient accepted frames: {m_accepted_frames}")

        # Blur ratio gate
        blur_ratio = m_blur_rejections / m_total_frames
        if blur_ratio > settings.ar_max_blur_ratio:
            rejection_reasons.append(f"Blur ratio too high: {blur_ratio:.2%} > {settings.ar_max_blur_ratio:.2%}")

        # Accepted ratio gate
        accepted_ratio = m_accepted_frames / m_total_frames
        if accepted_ratio < settings.ar_min_accepted_ratio:
            rejection_reasons.append(f"Accepted frame ratio too low: {accepted_ratio:.2%} < {settings.ar_min_accepted_ratio:.2%}")

        # Profile-specific gates
        profile_type = manifest_data.get("profile_type", "generic").lower()
        comp = manifest_data.get("profile_completion")
        
        if profile_type in ("box", "bottle"):
            if not comp:
                rejection_reasons.append(f"Missing profile_completion data for {profile_type}")
            else:
                if profile_type == "box":
                    faces = comp.get("faces", [])
                    if len(faces) < 6:
                        rejection_reasons.append(f"Box requires 6 completed faces, got {len(faces)}")
                elif profile_type == "bottle":
                    if not comp.get("cap") or not comp.get("base"):
                        rejection_reasons.append("Bottle requires both cap and base coverage")

        if rejection_reasons:
            logger.warning(
                f"Capture failed quality gate for {session_id}. Reasons: {rejection_reasons}"
            )
            raise HTTPException(
                status_code=422, 
                detail={"message": "Capture failed quality gate.", "reasons": rejection_reasons}
            )

        # ── 8. Session Finalization ───────────────────────────────────────────
        with open(capture_path / "reports" / "ar_quality_manifest.json", "w") as f:
            json.dump(manifest_data, f, indent=2)

        session = session_manager.create_session(session_id, product_id, operator_id)
        session_created = True
        
        session_manager.update_session(
            session_id, 
            new_status=AssetStatus.CAPTURED,
            manifest_validation_status="passed",
            test_mode=bool(manifest_data.get("is_demo"))
        )

        logger.info(f"Session {session_id} finalized successfully.")
        return {
            "session_id": session_id,
            "product_id": product_id,
            "status": "uploaded",
            "manifest_validation_status": "passed"
        }

    except HTTPException as he:
        cleanup_on_failure(he.detail if isinstance(he.detail, str) else "HTTP Exception")
        raise
    except Exception as e:
        logger.exception(f"Upload processing failed for {session_id}")
        cleanup_on_failure(str(e))
        raise HTTPException(status_code=500, detail=f"Internal upload error: {str(e)}")
    finally:
        # Direct streaming to captures; cleanup handles directory if needed
        pass


@app.get("/api/products", dependencies=[Depends(verify_api_key)])
async def list_products():
    """Returns a unified list of registered products and active capture sessions."""
    products_map = {}

    # 1. Scan Registered Assets — use settings.data_root, not hardcoded path
    meta_dir = Path(settings.data_root) / "registry" / "meta"
    if meta_dir.exists():
        for file in meta_dir.glob("*.json"):
            product_id = file.stem
            active_id = registry._get_active_id(product_id)
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                asset_count = len(data.get("assets", {}))

            products_map[product_id] = {
                "id": product_id,
                "active_id": active_id,
                "asset_count": asset_count,
                "last_updated": file.stat().st_mtime,
                "status": "registered",
            }

    # 2. Scan Active Sessions (In-progress uploads)
    sessions_dir = Path(settings.data_root) / "sessions"
    if sessions_dir.exists():
        for file in sessions_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                    p_id = session_data.get("product_id")
                    s_status = session_data.get("status")
                    
                    if p_id:
                        is_active = s_status not in ["published", "failed"]
                        
                        if p_id not in products_map:
                            products_map[p_id] = {
                                "id": p_id,
                                "active_id": None,
                                "asset_count": 0,
                                "last_updated": file.stat().st_mtime,
                                "status": "processing" if is_active else "registered",
                                "has_active_session": is_active
                            }
                        else:
                            if is_active:
                                products_map[p_id]["has_active_session"] = True
                                products_map[p_id]["status"] = "processing"
            except Exception:
                continue

    return sorted(products_map.values(), key=lambda x: x["last_updated"], reverse=True)


@app.get("/api/worker/status", dependencies=[Depends(verify_api_key)])
async def worker_status():
    return {
        "embedded": embedded_worker_enabled,
        "running": worker_instance.running,
    }


@app.get("/api/products/{product_id}/history", dependencies=[Depends(verify_api_key)])
async def get_history(product_id: str):
    """Returns full version history. Falls back to session data if not yet registered."""
    from modules.utils.path_safety import validate_identifier
    try:
        validate_identifier(product_id)
        return registry.get_history(product_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        sessions = []
        sessions_dir = Path(settings.data_root) / "sessions"
        if sessions_dir.exists():
            for file in sessions_dir.glob("*.json"):
                with open(file, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                    if s_data.get("product_id") == product_id:
                        sessions.append(
                            {
                                "asset_id": s_data.get("session_id"),
                                "version": "In-Progress",
                                "status": s_data.get("status"),
                                "is_active": False,
                                "approved": False,
                                "audit": [
                                    {
                                        "action": "session_created",
                                        "asset_id": s_data.get("session_id"),
                                    }
                                ],
                            }
                        )
        if not sessions:
            raise HTTPException(status_code=404, detail="Product history not found")
        return sessions


@app.get("/api/logs", dependencies=[Depends(verify_api_key)])
async def get_logs(limit: int = 50):
    """Reads the last N lines from the operational log file."""
    log_file = Path(settings.data_root) / "logs" / "factory.log"
    if not log_file.exists():
        return []

    logs = []
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()[-limit:]
        for line in lines:
            try:
                logs.append(json.loads(line))
            except Exception:
                logs.append({"message": line.strip(), "level": "INFO"})
    return logs[::-1]  # Newest first


@app.get("/api/sessions/{session_id}/guidance", dependencies=[Depends(verify_api_key)])
async def get_session_guidance(session_id: str):
    """Returns the structured guidance report for a session."""
    from modules.utils.path_safety import validate_identifier
    try:
        validate_identifier(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    reports_dir = Path(settings.data_root) / "captures" / session_id / "reports"
    guidance_path = reports_dir / "guidance_report.json"

    if not guidance_path.exists():
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session_id,
            "status": session.status,
            "next_action": "Guidance not yet generated. Please wait...",
            "messages": [],
        }

    with open(guidance_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/sessions/{session_id}/guidance/summary", dependencies=[Depends(verify_api_key)])
async def get_session_guidance_summary(session_id: str):
    """Returns the human-readable Markdown summary of the guidance."""
    from modules.utils.path_safety import validate_identifier
    try:
        validate_identifier(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    reports_dir = Path(settings.data_root) / "captures" / session_id / "reports"
    summary_path = reports_dir / "guidance_summary.md"

    if not summary_path.exists():
        return "# Guidance Pending\nPlease wait for the system to analyze your capture."

    with open(summary_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/sessions/{session_id}/cancel", dependencies=[Depends(verify_api_key)])
async def cancel_session(session_id: str):
    """Marks a session as failed/cancelled to stop worker processing."""
    from modules.utils.path_safety import validate_identifier
    try:
        validate_identifier(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status in [AssetStatus.PUBLISHED, AssetStatus.FAILED]:
        return {"status": "already_closed", "session_id": session_id, "current_status": session.status.value}
        
    try:
        session_manager.update_session(
            session_id, 
            new_status=AssetStatus.FAILED,
            failure_reason="Cancelled by user via Dashboard"
        )
        logger.info(f"Session {session_id} cancelled by user.")
        return {"status": "cancelled", "session_id": session_id}
    except Exception as e:
        logger.error(f"Failed to cancel session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Cancellation failed: {str(e)}")


@app.get("/api/training/manifests", dependencies=[Depends(verify_api_key)])
async def list_training_manifests():
    """
    Returns the list of generated training data manifests.
    Internal/Admin only in production.
    """
    from modules.training_data.dataset_registry import DatasetRegistry
    
    try:
        registry = DatasetRegistry(Path(settings.data_root) / "training_registry" / "index.jsonl")
        manifests = registry.get_all()
        return manifests
    except Exception as e:
        logger.error(f"Failed to fetch training manifests: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch training manifests")


# Asset Blobs (GLB Files)
blobs_dir = Path(settings.data_root) / "registry" / "blobs"
blobs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/assets/blobs", StaticFiles(directory=str(blobs_dir)), name="blobs")

# Mount the UI directory if it exists
ui_dir = Path("ui")
if ui_dir.exists():
    app.mount("/", StaticFiles(directory="ui", html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    from pathlib import Path
    
    cert_path = Path(__file__).parent.parent.parent / "certs" / "cert.pem"
    key_path = Path(__file__).parent.parent.parent / "certs" / "key.pem"
    
    ssl_config = {}
    if cert_path.exists() and key_path.exists():
        logger.info(f"Starting server with SSL: {cert_path}")
        ssl_config = {
            "ssl_certfile": str(cert_path),
            "ssl_keyfile": str(key_path)
        }
    else:
        logger.info("Starting server without SSL (Certs not found)")

    uvicorn.run(app, host="0.0.0.0", port=8001, **ssl_config)
