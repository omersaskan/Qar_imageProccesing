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
from modules.operations.logging_config import get_component_logger, setup_logging
from modules.operations.worker import worker_instance
from modules.operations.settings import settings, AppEnvironment
import mimetypes

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

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
            "free_disk_gb": round(free_disk_gb, 2) if free_disk_gb != float("inf") else None,
            "min_required_disk_gb": settings.min_free_disk_gb,
            "disk_ok": disk_ok,
        },
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

    Sprint 1 additions (TICKET-004):
      - Disk space is checked before accepting the file in pilot/production.
      - The error message is explicit and actionable.
    """
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
        raise HTTPException(
            status_code=400,
            detail="Invalid video format. Supported: .mp4, .mov, .avi, .webm",
        )

    session_id = f"cap_{uuid.uuid4().hex[:8]}"

    # ── Dependency Gating (existing) ──────────────────────────────────────
    missing_ml = settings.check_ml_deps()
    missing_proc = settings.check_processing_deps()

    if (missing_ml or missing_proc) and not settings.is_dev:
        detail = "System Environment Incomplete: "
        if missing_ml:
            detail += f"Missing ML dependencies ({', '.join(missing_ml)}). "
        if missing_proc:
            detail += f"Missing Processing dependencies ({', '.join(missing_proc)}). "
        detail += "Consult the Operator Runbook for installation guidance."
        logger.error(f"Upload blocked for session {session_id} due to environment issues.")
        raise HTTPException(status_code=503, detail=detail)

    # ── TICKET-004: Disk space preflight ──────────────────────────────────
    if not settings.is_dev:
        free_disk_gb = settings.check_free_disk_gb()
        if free_disk_gb < settings.min_free_disk_gb:
            detail = (
                f"Insufficient disk space: {free_disk_gb:.1f} GB free. "
                f"Minimum required for safe processing: {settings.min_free_disk_gb} GB. "
                "Free disk space before accepting new uploads."
            )
            logger.error(
                f"Upload blocked for session {session_id}: "
                f"only {free_disk_gb:.1f} GB free (min {settings.min_free_disk_gb} GB)."
            )
            raise HTTPException(status_code=507, detail=detail)

    # Preflight Check: Video Validation
    import tempfile
    import cv2
    import os
    
    fd, temp_path = tempfile.mkstemp(suffix=Path(file.filename).suffix)
    try:
        with os.fdopen(fd, 'wb') as f:
            shutil.copyfileobj(file.file, f)
            
        file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        
        if file_size_mb == 0 or file_size_mb < 0.1:
            raise HTTPException(status_code=400, detail="Video file is too small or empty.")
            
        if file_size_mb > settings.max_upload_mb:
            raise HTTPException(status_code=400, detail=f"File size exceeds maximum allowed ({settings.max_upload_mb} MB).")

        cap = cv2.VideoCapture(temp_path)
        try:
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Video is unreadable or uses an unsupported codec.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
            
        if width < settings.min_video_width or height < settings.min_video_height:
            raise HTTPException(status_code=400, detail=f"Video resolution too low: {width}x{height}. Minimum required: {settings.min_video_width}x{settings.min_video_height}.")
            
        if fps < settings.min_video_fps:
            raise HTTPException(status_code=400, detail=f"Video FPS too low: {fps:.1f}. Minimum required: {settings.min_video_fps}.")
            
        duration = frame_count / fps if fps > 0 else 0
        duration = max(0.0, duration) # Ensure non-negative
        if duration < settings.min_video_duration_sec:
            raise HTTPException(status_code=400, detail=f"Video duration too short: {duration:.1f}s. Minimum required: {settings.min_video_duration_sec}s.")
        if duration > settings.max_video_duration_sec:
            raise HTTPException(status_code=400, detail=f"Video duration too long: {duration:.1f}s. Maximum allowed: {settings.max_video_duration_sec}s.")
            
        # If all checks pass, move the file safely later
    except HTTPException:
        os.remove(temp_path)
        raise
    except Exception as e:
        os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Error validating video: {e}")

    try:
        # 1. Create Session folders and record
        session = session_manager.create_session(session_id, product_id, operator_id)

        # 2. Setup video folder
        capture_path = session_manager.get_capture_path(session_id)
        video_dir = capture_path / "video"
        video_dir.mkdir(parents=True, exist_ok=True)

        # 3. Save uploaded file
        original_ext = Path(file.filename).suffix.lower()
        if original_ext == ".webm":
            original_path = video_dir / "original_capture.webm"
            shutil.move(temp_path, str(original_path))
            
            video_path = video_dir / "raw_video.mp4"
            try:
                from modules.utils.video_utils import normalize_video
                normalize_video(original_path, video_path, ffmpeg_path=settings.ffmpeg_path)
                logger.info(f"Transcoded {original_path.name} to {video_path.name}")
            except Exception as e:
                if settings.env.value == "local_dev":
                    logger.warning(f"Transcoding failed for {session_id}, falling back to copy (LOCAL_DEV ONLY): {e}")
                    shutil.copy(original_path, video_path)
                else:
                    logger.error(f"Normalization failed for {session_id}: {e}")
                    shutil.rmtree(capture_path)
                    session_file = Path(settings.data_root) / "sessions" / f"{session_id}.json"
                    if session_file.exists(): session_file.unlink()
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Video normalization failed: FFmpeg error or unavailable. {str(e)}"
                    )
        else:
            video_path = video_dir / "raw_video.mp4"
            shutil.move(temp_path, str(video_path))

        # ── Quality Gate Enforcement ──────────────────────────────────────────
        manifest_valid = True
        rejection_reasons = []

        if not quality_manifest:
            # If it's an AR guided session, we might want to flag it. 
            # For now, let's assume if it comes from the AR modal, it MUST have a manifest.
            # We can't strictly know if it's AR unless we add a flag, but if it has a manifest, we validate it.
            # The prompt says: "Reject if quality_manifest is missing for AR guided uploads."
            # Since we are moving to AR-only automatic uploads, we can require it.
            logger.error(f"Upload rejected for {session_id}: Missing quality_manifest")
            shutil.rmtree(capture_path)
            session_file = Path(settings.data_root) / "sessions" / f"{session_id}.json"
            if session_file.exists(): session_file.unlink()
            raise HTTPException(status_code=422, detail="Missing quality_manifest for AR capture.")

        try:
            manifest_data = json.loads(quality_manifest)
            
            # 1. Demo Mode Check
            is_demo = manifest_data.get("is_demo", False)
            if is_demo and not settings.is_dev:
                logger.error(f"Upload rejected for {session_id}: Demo mode not allowed in {settings.env.value}")
                shutil.rmtree(capture_path)
                session_file = Path(settings.data_root) / "sessions" / f"{session_id}.json"
                if session_file.exists(): session_file.unlink()
                raise HTTPException(
                    status_code=422, 
                    detail=f"Demo mode is only permitted in LOCAL_DEV. Current environment: {settings.env.value}"
                )

            # 2. Metric Validation
            coverage = manifest_data.get("coverage_summary", {}).get("percent", 0)
            max_gap = manifest_data.get("coverage_summary", {}).get("maxGap", 360)
            accepted_frames = manifest_data.get("accepted_frame_count", 0)
            total_frames = manifest_data.get("total_frame_count", 0)
            
            blur_rejections = manifest_data.get("rejection_stats", {}).get("Move slower (blur detected)", 0)
            blur_ratio = blur_rejections / total_frames if total_frames > 0 else 0
            
            profile = manifest_data.get("product_profile", "generic")
            completion = manifest_data.get("profile_completion")
            
            # Duration (if available in manifest, otherwise we check video duration later)
            # The manifest should probably include the actual recording duration
            
            if coverage < settings.ar_min_coverage:
                rejection_reasons.append(f"Coverage too low: {coverage:.1f}% (min {settings.ar_min_coverage}%)")
            if max_gap > settings.ar_max_gap:
                rejection_reasons.append(f"Gap too large: {max_gap:.1f}° (max {settings.ar_max_gap}°)")
            if accepted_frames < settings.ar_min_accepted_frames:
                rejection_reasons.append(f"Insufficient accepted frames: {accepted_frames} (min {settings.ar_min_accepted_frames})")
            if blur_ratio > settings.ar_max_blur_ratio:
                rejection_reasons.append(f"Too much blur: {blur_ratio:.2f} (max {settings.ar_max_blur_ratio})")
            
            # Profile specific
            if profile == "box":
                completed_faces = completion.get("faces", []) if completion else []
                if len(completed_faces) < 6:
                    rejection_reasons.append(f"Incomplete box profile: {len(completed_faces)}/6 faces captured")
            elif profile == "bottle":
                cap = completion.get("cap", False) if completion else False
                base = completion.get("base", False) if completion else False
                if not cap or not base:
                    rejection_reasons.append(f"Incomplete bottle profile: {'missing CAP' if not cap else ''} {'missing BASE' if not base else ''}")

            if rejection_reasons:
                manifest_valid = False

        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid quality_manifest JSON.")
        except Exception as e:
            logger.warning(f"Metadata parsing failed during quality gate check: {e}")
            # Fallback: if we can't parse it, we can't validate it.
            manifest_valid = False
            rejection_reasons.append(f"Manifest parsing error: {str(e)}")

        # ── Final Decision & Persistence ──────────────────────────────────────
        reports_dir = capture_path / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        if manifest_valid:
            with open(reports_dir / "ar_quality_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
            
            # Tag demo if applicable (only reaches here if in LOCAL_DEV)
            if manifest_data.get("is_demo"):
                session_file = Path(settings.data_root) / "sessions" / f"{session_id}.json"
                if session_file.exists():
                    with open(session_file, "r+", encoding="utf-8") as f:
                        s_data = json.load(f)
                        s_data["test_mode"] = True
                        s_data["status"] = "demo_capture"
                        f.seek(0)
                        json.dump(s_data, f, indent=2)
                        f.truncate()
        else:
            # Save rejected manifest for debugging
            rejected_dir = reports_dir / "rejected"
            rejected_dir.mkdir(parents=True, exist_ok=True)
            with open(rejected_dir / "rejected_ar_quality_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)
            
            # Cleanup session
            logger.error(f"Upload rejected for {session_id}: {'; '.join(rejection_reasons)}")
            shutil.rmtree(capture_path)
            session_file = Path(settings.data_root) / "sessions" / f"{session_id}.json"
            if session_file.exists(): session_file.unlink()
            
            raise HTTPException(
                status_code=422, 
                detail={
                    "message": "Capture failed quality gate.",
                    "reasons": rejection_reasons,
                    "manifest_validation_status": "rejected"
                }
            )

        logger.info(
            f"Video uploaded successfully for session {session_id}. Size: {file_size_mb:.2f} MB, Res: {width}x{height}, Dur: {duration:.1f}s",
            extra={"job_id": session_id},
        )

        return {
            "session_id": session_id,
            "product_id": product_id,
            "status": "uploaded",
            "path": str(video_path),
            "manifest_validation_status": "passed"
        }
    except HTTPException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during upload: {str(e)}",
        )


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
    try:
        return registry.get_history(product_id)
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
    reports_dir = Path(settings.data_root) / "captures" / session_id / "reports"
    summary_path = reports_dir / "guidance_summary.md"

    if not summary_path.exists():
        return "# Guidance Pending\nPlease wait for the system to analyze your capture."

    with open(summary_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/sessions/{session_id}/cancel", dependencies=[Depends(verify_api_key)])
async def cancel_session(session_id: str):
    """Marks a session as failed/cancelled to stop worker processing."""
    session_file = Path(settings.data_root) / "sessions" / f"{session_id}.json"
    if not session_file.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        with open(session_file, "r+", encoding="utf-8") as f:
            s_data = json.load(f)
            if s_data.get("status") in ["published", "failed"]:
                return {"status": "already_closed", "session_id": session_id}
            
            s_data["status"] = "failed"
            s_data["error"] = "Cancelled by user via Dashboard"
            f.seek(0)
            json.dump(s_data, f, indent=2)
            f.truncate()
            
        logger.info(f"Session {session_id} cancelled by user.")
        return {"status": "cancelled", "session_id": session_id}
    except Exception as e:
        logger.error(f"Failed to cancel session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
