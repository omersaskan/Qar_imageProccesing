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

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Header, Depends, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

embedded_worker_enabled = settings.embedded_worker_enabled


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
    capture_profile_size: str = Form("small"),
    capture_profile_scene: str = Form("on_surface"),
    material_hint: str = Form("opaque"),
    file: UploadFile = File(...),
):
    """
    Handles video upload, creates a new CaptureSession, and saves the file.
    Includes strict FFmpeg normalization and quality manifest validation.

    Capture profile (size_class × scene_type) controls reconstruction tuning,
    upload size/duration limits, and isolation behavior end-to-end.
    """
    from modules.utils.path_safety import validate_identifier
    from modules.utils.video_utils import normalize_video, validate_video_file
    from modules.operations.capture_profile import (
        SizeClass, SceneType, MaterialHint,
        resolve_capture_profile, apply_profile_to_settings,
    )

    # Resolve capture profile (UI-supplied; falls back to default on garbage input)
    try:
        sz = SizeClass(capture_profile_size.lower())
    except ValueError:
        sz = SizeClass.SMALL
    try:
        sc = SceneType(capture_profile_scene.lower())
    except ValueError:
        sc = SceneType.ON_SURFACE
    try:
        mh = MaterialHint(material_hint.lower())
    except ValueError:
        mh = MaterialHint.OPAQUE
    profile = resolve_capture_profile(sz, sc, mh)
    eff_settings = apply_profile_to_settings(profile, settings)
    logger.info(
        f"Upload received with capture_profile={profile.preset_key} "
        f"material={mh.value} → max_upload_mb={eff_settings.max_upload_mb}, "
        f"max_dur={eff_settings.max_video_duration_sec}s, "
        f"long_edge_min={eff_settings.min_video_long_edge}"
    )

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

    # ── 2. Quality Manifest JSON Parse (Early MALFORMED check) ────────────
    manifest_data = None
    if quality_manifest and quality_manifest.strip() not in ("", "null"):
        try:
            manifest_data = json.loads(quality_manifest)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Malformed JSON in quality_manifest")

    # ── 3. Profile Resolution & Effective Settings ────────────────────────
    try:
        sz = SizeClass(capture_profile_size.lower())
    except ValueError:
        sz = SizeClass.SMALL
    try:
        sc = SceneType(capture_profile_scene.lower())
    except ValueError:
        sc = SceneType.ON_SURFACE
    try:
        mh = MaterialHint(material_hint.lower())
    except ValueError:
        mh = MaterialHint.OPAQUE
        
    profile = resolve_capture_profile(sz, sc, mh)
    eff_settings = apply_profile_to_settings(profile, settings)
    
    logger.info(
        f"Upload received with capture_profile={profile.preset_key} "
        f"material={mh.value} \u2192 max_upload_mb={eff_settings.max_upload_mb}, "
        f"min_dur={eff_settings.min_video_duration_sec}s, "
        f"max_dur={eff_settings.max_video_duration_sec}s"
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

    # ── 4. Environment & Disk Preflight ───────────────────────────────────
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

    # ── 5. Storage Setup & Direct Streaming ───────────────────────────────
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
        
        # ── 6. AR Quality Manifest CONTENT Validation (Precedence High) ───────
        if manifest_data:
            # A. Demo Mode & Numeric Sanity
            if manifest_data.get("is_demo") and not settings.is_dev:
                 raise HTTPException(status_code=422, detail="Demo mode not permitted in production.")

            m_total_frames = manifest_data.get("total_frame_count", 0)
            m_accepted_frames = manifest_data.get("accepted_frame_count", 0)
            m_blur_rejections = manifest_data.get("blur_rejection_count", 0)
            
            if m_total_frames <= 0:
                raise HTTPException(status_code=422, detail="Manifest total_frame_count must be > 0")
            if m_accepted_frames < 0 or m_accepted_frames > m_total_frames:
                raise HTTPException(status_code=422, detail="Manifest accepted_frame_count inconsistency")
            if m_blur_rejections < 0 or m_blur_rejections > m_total_frames:
                raise HTTPException(status_code=422, detail="Manifest blur_rejection_count inconsistency")

            # B. AR Quality Gate Enforcement (Precedence 3)
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
                logger.warning(f"Capture failed quality gate for {session_id}. Reasons: {rejection_reasons}")
                raise HTTPException(
                    status_code=422, 
                    detail={"message": "Capture failed quality gate.", "reasons": rejection_reasons}
                )

        # ── 7. Physical Video Validation (Precedence 4) ───────────────────────
        preflight_errors = []
        
        # A. File Size
        if file_size_mb == 0 or file_size_mb < 0.1:
            preflight_errors.append("Video file is too small or empty.")
        elif file_size_mb > eff_settings.max_upload_mb:
            preflight_errors.append(
                f"File size {file_size_mb:.1f}MB exceeds maximum allowed {eff_settings.max_upload_mb}MB "
                f"for capture_profile={profile.preset_key}."
            )

        # B. Video Metadata & Integrity
        ok, error, video_meta = validate_video_file(
            original_path,
            min_fps=0,        # Manual check below using eff_settings
            min_duration=0,   
            max_duration=9999 
        )
        if not ok:
            preflight_errors.append(f"Integrity check failed: {error}")
        else:
            # C. FPS
            if video_meta["fps"] < eff_settings.min_video_fps:
                preflight_errors.append(f"FPS too low: {video_meta['fps']:.1f} < {eff_settings.min_video_fps}")
            
            # D. Duration
            v_dur = video_meta.get("duration", 0)
            if v_dur < eff_settings.min_video_duration_sec:
                preflight_errors.append(f"Video too short: {v_dur:.1f}s < {eff_settings.min_video_duration_sec}s")
            if v_dur > eff_settings.max_video_duration_sec:
                preflight_errors.append(f"Video too long: {v_dur:.1f}s > {eff_settings.max_video_duration_sec}s")

            _w, _h = video_meta["width"], video_meta["height"]
            _short = min(_w, _h)
            _long  = max(_w, _h)
            
            logger.debug(f"DEBUG: Resolving {_w}x{_h} (short={_short}, long={_long}) vs limits (short={eff_settings.min_video_short_edge}, long={eff_settings.min_video_long_edge})")
            
            if _short < eff_settings.min_video_short_edge or _long < eff_settings.min_video_long_edge:
                preflight_errors.append(
                    f"Resolution too low: {_w}x{_h}. "
                    f"Short edge must be >= {eff_settings.min_video_short_edge}, "
                    f"Long edge must be >= {eff_settings.min_video_long_edge}"
                )

        if preflight_errors:
            msg = " | ".join(preflight_errors)
            logger.warning(f"Upload rejected for {session_id}: {msg}")
            raise HTTPException(status_code=400, detail=f"Video validation failed: {msg}")

        # ── 8. Manifest Missing Check (Precedence Final for legacy compatibility) ──
        if not manifest_data:
             raise HTTPException(status_code=422, detail="Missing quality_manifest for AR capture.")

        # Manifest-vs-Video consistency (now we have both)
        v_frame_count = video_meta["frame_count"]
        m_total_frames = manifest_data.get("total_frame_count", 0)
        tolerance = settings.ar_manifest_frame_count_tolerance
        if m_total_frames > v_frame_count * (1.0 + tolerance):
            msg = f"Manifest total_frame_count ({m_total_frames}) exceeds video frame count ({v_frame_count}) by >{tolerance*100}%"
            logger.warning(f"Upload rejected for {session_id}: {msg}")
            raise HTTPException(status_code=422, detail=msg)

        # ── 9. Normalization & Persist Profile ────────────────────────────────
        try:
            profile_manifest = capture_path / "session_capture_profile.json"
            with open(profile_manifest, "w", encoding="utf-8") as pf:
                json.dump(profile.to_dict(), pf, indent=2)
        except Exception as e:
            logger.warning(f"Could not persist session_capture_profile.json: {e}")

        video_path = video_dir / "raw_video.mp4"
        try:
            normalize_video(
                original_path,
                video_path,
                ffmpeg_path=ffmpeg_resolved,
                ffprobe_path=ffprobe_resolved,
                timeout=settings.video_normalize_timeout_sec
            )
        except RuntimeError as _norm_err:
            raise HTTPException(status_code=400, detail=f"Video processing failed: {_norm_err}")

        # ── 10. Session Finalization ──────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# SAM2 Manual Track (UI-driven) — operatör bbox/point çizer, video propagation
# tüm frame'lerin masks'ini üretir. SAM2_ENABLED=true ve checkpoint gerekli.
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/capture-gate", dependencies=[Depends(verify_api_key)])
async def get_capture_gate(session_id: str):
    """
    Sprint 2 — return the capture quality gate report from extraction_manifest.json.
    UI polls this after upload to show the 3×8 matrix overlay + reasons + suggestions.

    Returns 404 until extraction has finished writing the manifest.
    """
    capture_path = Path(settings.data_root) / "captures" / session_id
    manifest_path = capture_path / "frames" / "extraction_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Capture gate not yet available — frame extraction still running or failed",
        )
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manifest read failed: {e}")

    gate = manifest.get("capture_gate") or {}
    if not gate:
        return {
            "session_id": session_id,
            "status": "pending",
            "message": "extraction completed before Sprint 2 gate; legacy session",
        }
    return {
        "session_id": session_id,
        "status": "ready",
        "decision": gate.get("decision", "unknown"),
        "reasons": gate.get("reasons", []),
        "suggestions": gate.get("suggestions", []),
        "matrix_3x8": gate.get("matrix_3x8", []),
        "blur": gate.get("blur", {}),
        "elevation": gate.get("elevation", {}),
        "azimuth": gate.get("azimuth", {}),
        "thresholds": gate.get("gate_thresholds", {}),
    }


@app.get("/api/sessions/{session_id}/first-frame", dependencies=[Depends(verify_api_key)])
async def get_session_first_frame(session_id: str):
    """
    Returns the first extracted frame for UI bbox/point prompting.
    """
    frames_dir = Path(settings.data_root) / "captures" / session_id / "frames"
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session frames dir not found: {frames_dir}")

    frame_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    frame_files = [f for f in frame_files if f.is_file()]
    if not frame_files:
        raise HTTPException(status_code=404, detail="No frames in session")

    return FileResponse(str(frame_files[0]), media_type="image/jpeg")


@app.post("/api/sessions/{session_id}/sam2_track", dependencies=[Depends(verify_api_key)])
async def sam2_video_track(
    session_id: str,
    payload: Dict[str, Any] = Body(...),
):
    """
    Run SAM2 video propagation across all session frames, seeded by a manual
    prompt (bbox or points) on a chosen frame. Writes masks to
    `data/captures/{session_id}/frames/masks/<frame>.png`.

    Body schema:
        {
          "seed_frame_idx": 0,
          "seed_box": [x1, y1, x2, y2]              # OR
          "seed_points": [[x, y], ...],
          "seed_labels": [1, 1, ...]                # 1=foreground, 0=background
        }
    """
    if not settings.sam2_enabled:
        raise HTTPException(status_code=400, detail="SAM2 disabled (SAM2_ENABLED=false)")

    seed_frame_idx = int(payload.get("seed_frame_idx", 0))
    seed_box = payload.get("seed_box")
    seed_points = payload.get("seed_points")
    seed_labels = payload.get("seed_labels")

    if not seed_box and not seed_points:
        raise HTTPException(status_code=400, detail="Provide seed_box or seed_points")

    if seed_box is not None and (not isinstance(seed_box, (list, tuple)) or len(seed_box) != 4):
        raise HTTPException(status_code=400, detail="seed_box must be [x1, y1, x2, y2]")

    frames_dir = Path(settings.data_root) / "captures" / session_id / "frames"
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session frames dir not found: {frames_dir}")

    masks_dir = frames_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    try:
        from modules.ai_segmentation.sam2_video_backend import SAM2VideoBackend
        backend = SAM2VideoBackend(
            model_cfg=settings.sam2_model_cfg,
            checkpoint=settings.sam2_checkpoint,
            device=settings.sam2_device,
        )
    except Exception as e:
        logger.error(f"SAM2 video backend init failed: {e}")
        raise HTTPException(status_code=500, detail=f"SAM2 backend init failed: {e}")

    if not backend.is_available():
        raise HTTPException(
            status_code=503,
            detail=f"SAM2 video predictor not loaded; check checkpoint at {settings.sam2_checkpoint}",
        )

    np_points = None
    np_labels = None
    if seed_points is not None:
        try:
            np_points = np.array(seed_points, dtype=np.float32)
            if np_points.ndim != 2 or np_points.shape[1] != 2:
                raise ValueError("seed_points must be a list of [x, y] pairs")
            label_list = seed_labels if seed_labels else [1] * len(seed_points)
            np_labels = np.array(label_list, dtype=np.int32)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid seed_points/seed_labels: {e}")

    logger.info(
        f"[SAM2 track] session={session_id} seed_frame={seed_frame_idx} "
        f"seed_box={seed_box} seed_points={seed_points}"
    )

    masks_dict = backend.segment_video(
        frames_dir=frames_dir,
        seed_frame_idx=seed_frame_idx,
        seed_box=seed_box,
        seed_points=np_points,
        seed_labels=np_labels,
        seed_prompt_source="ui_manual",
        output_dir=masks_dir,
    )

    return {
        "session_id": session_id,
        "masks_generated": backend.masks_generated,
        "expected_frame_count": backend.expected_frame_count,
        "video_propagation_failed": backend.video_propagation_failed,
        "masks_dir": str(masks_dir),
        "status": backend.get_status(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI Completion (generative 3D for unobserved surfaces)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/ai-complete/assess", dependencies=[Depends(verify_api_key)])
async def assess_ai_completion(session_id: str):
    """
    Cheap analysis: load reconstruction mesh, compute observed surface ratio,
    show what would happen with the configured provider — without invoking it.
    """
    capture_path = Path(settings.data_root) / "captures" / session_id
    job_dir = capture_path / "reconstruction"
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Reconstruction dir not found: {job_dir}")

    manifest_path = job_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="manifest.json missing — reconstruction incomplete?")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mesh_path = manifest.get("mesh_path")
    if not mesh_path or not Path(mesh_path).exists():
        raise HTTPException(status_code=404, detail=f"Mesh not found: {mesh_path}")

    try:
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mesh load failed: {e}")

    # Resolve capture profile (UI-supplied or env)
    from modules.operations.capture_profile import CaptureProfile, resolve_from_setting
    profile = None
    spp = capture_path / "session_capture_profile.json"
    if spp.exists():
        try:
            profile = CaptureProfile.from_dict(json.loads(spp.read_text(encoding="utf-8")))
        except Exception:
            pass
    if profile is None:
        profile = resolve_from_setting(settings.capture_profile, settings.material_hint)

    from modules.ai_completion import build_default_service
    svc = build_default_service(settings, capture_profile=profile)
    result = svc.assess(mesh)
    result["mesh_path"] = mesh_path
    result["capture_profile"] = profile.preset_key
    return result


@app.post("/api/sessions/{session_id}/ai-complete", dependencies=[Depends(verify_api_key)])
async def run_ai_completion(
    session_id: str,
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    """
    Trigger AI completion for a session.  Body fields (all optional):
        provider:  override AI_3D_PROVIDER (e.g. 'hunyuan3d_replicate', 'meshy')
        notes:     free text passed to provider metadata
        force:     bool — bypass observed/production gate (review-only output)
    """
    body = payload or {}

    capture_path = Path(settings.data_root) / "captures" / session_id
    job_dir = capture_path / "reconstruction"
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Reconstruction dir not found: {job_dir}")

    manifest_path = job_dir / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="manifest.json missing — reconstruction incomplete?")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mesh_path = manifest.get("mesh_path")
    if not mesh_path or not Path(mesh_path).exists():
        raise HTTPException(status_code=404, detail=f"Mesh not found: {mesh_path}")

    try:
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
        original_face_count = int(len(mesh.faces))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mesh load failed: {e}")

    # Reference image: first extracted frame
    frames_dir = capture_path / "frames"
    ref_images: List[str] = []
    if frames_dir.exists():
        candidates = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
        ref_images = [str(c) for c in candidates[:1]]

    # Capture profile
    from modules.operations.capture_profile import CaptureProfile, resolve_from_setting
    profile = None
    spp = capture_path / "session_capture_profile.json"
    if spp.exists():
        try:
            profile = CaptureProfile.from_dict(json.loads(spp.read_text(encoding="utf-8")))
        except Exception:
            pass
    if profile is None:
        profile = resolve_from_setting(settings.capture_profile, settings.material_hint)

    from modules.ai_completion import (
        build_default_service, CompletionRequest, CompletionStatus,
    )
    provider_override = body.get("provider")
    svc = build_default_service(
        settings, capture_profile=profile, provider_override=provider_override,
    )

    request = CompletionRequest(
        session_id=session_id,
        mesh_path=mesh_path,
        reference_image_paths=ref_images,
        observed_surface_ratio=0.0,  # service computes
        capture_profile_key=profile.preset_key,
        material_hint=profile.material_hint.value,
        notes=str(body.get("notes", "")),
    )

    output_dir = capture_path / "ai_completion"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = svc.run(
        mesh=mesh,
        original_face_count=original_face_count,
        request=request,
        output_dir=output_dir,
    )

    # Persist result for traceability
    try:
        with open(output_dir / "completion_result.json", "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
    except Exception:
        pass

    return result.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3B: DEPTH STUDIO ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

_depth_sessions: Dict[str, Dict[str, Any]] = {}
_depth_data_root = Path(settings.data_root) / "depth_studio"


def _depth_session_dir(session_id: str) -> Path:
    return _depth_data_root / session_id


def _depth_session_summary(session_id: str) -> Dict[str, Any]:
    """
    Return a compact status summary dict for a session.
    Reads the persisted manifest if available; falls back to in-memory info.
    Used to enrich 404 responses on image/artifact endpoints.
    """
    info = _depth_sessions.get(session_id, {})
    manifest_path = info.get("manifest_path")
    manifest: Dict[str, Any] = {}
    if manifest_path:
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except Exception:
            pass

    status = manifest.get("status") or info.get("status", "unknown")
    provider_status = manifest.get("provider_status", "unknown")
    warnings = manifest.get("warnings", [])
    provider = manifest.get("provider") or info.get("provider", "unknown")

    # Determine which expected outputs are missing
    derived = _depth_session_dir(session_id) / "derived"
    expected = {
        "depth_preview":  derived / "depth_preview.png",
        "subject_mask":   derived / "subject_mask.png",
        "mask_overlay":   derived / "mask_overlay.png",
        "cropped_subject": derived / "cropped_subject.jpg",
        "glb":            derived / "preview_mesh.glb",
    }
    missing_outputs = [name for name, path in expected.items() if not path.exists()]

    summary: Dict[str, Any] = {
        "session_status": status,
        "provider": provider,
        "provider_status": provider_status,
        "warnings": warnings,
        "missing_outputs": missing_outputs,
    }

    # Friendly reason for provider failure
    _PROVIDER_FAILURE_STATUSES = ("unavailable", "failed", "disabled", "error")
    if provider_status in _PROVIDER_FAILURE_STATUSES:
        if "depth_pro" in provider:
            if provider_status == "error":
                summary["provider_failure_reason"] = (
                    "Depth Pro worker process failed: check DEPTH_PRO_PYTHON_PATH, "
                    "venv installation, and worker logs"
                )
            else:
                summary["provider_failure_reason"] = (
                    "Depth Pro unavailable: DEPTH_PRO_ENABLED=false or depth_pro package not installed"
                )
        else:
            summary["provider_failure_reason"] = (
                f"Provider '{provider}' reported status '{provider_status}'"
            )

    return summary


@app.post("/api/depth-studio/upload", dependencies=[Depends(verify_api_key)])
async def depth_studio_upload(
    file: UploadFile = File(...),
    provider: str = Form(default="depth_anything_v2"),
    product_id: str = Form(default=""),
):
    """Accept an image or video, create a Depth Studio session."""
    if not settings.depth_studio_enabled:
        raise HTTPException(status_code=503, detail="Depth Studio is disabled (DEPTH_STUDIO_ENABLED=false)")

    session_id = f"ds_{uuid.uuid4().hex[:12]}"
    session_dir = _depth_session_dir(session_id)
    input_dir = session_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    dest = input_dir / f"upload{suffix}"
    with open(str(dest), "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    _depth_sessions[session_id] = {
        "session_id": session_id,
        "status": "uploaded",
        "input_path": str(dest),
        "provider": provider,
        "product_id": product_id,
        "manifest_path": None,
    }
    return {"session_id": session_id, "status": "uploaded", "input_path": str(dest)}


class _DepthProcessRequest(BaseModel):
    prompt_box: Optional[list] = None   # [x0, y0, x1, y1]


@app.post("/api/depth-studio/process/{session_id}", dependencies=[Depends(verify_api_key)])
async def depth_studio_process(session_id: str, body: Optional[_DepthProcessRequest] = None):
    """Run depth inference + mesh + GLB for the uploaded file.

    Optional JSON body: {"prompt_box": [x0, y0, x1, y1]}
    """
    if session_id not in _depth_sessions:
        raise HTTPException(status_code=404, detail=f"Depth Studio session not found: {session_id}")

    info = _depth_sessions[session_id]
    if info["status"] not in ("uploaded", "failed"):
        return {"session_id": session_id, "status": info["status"], "detail": "Already processed or processing"}

    info["status"] = "processing"
    session_dir = _depth_session_dir(session_id)

    prompt_box = None
    if body and body.prompt_box and len(body.prompt_box) == 4:
        prompt_box = tuple(int(v) for v in body.prompt_box)

    try:
        from modules.depth_studio.pipeline import run_depth_studio
        manifest = run_depth_studio(
            session_id=session_id,
            input_file_path=info["input_path"],
            output_base_dir=str(session_dir),
            provider_name=info.get("provider"),
            prompt_box=prompt_box,
        )
        info["status"] = manifest.get("status", "failed")
        info["manifest_path"] = str(session_dir / "manifests" / "depth_studio_manifest.json")

        # Build enriched response so the UI can act on failure without extra round-trips.
        # missing_outputs comes from disk (written by the pipeline); provider_failure_reason
        # is derived from the in-hand manifest dict so it works even when the manifest file
        # was never written (e.g. provider mock in tests).
        summary = _depth_session_summary(session_id)
        provider_status = manifest.get("provider_status", "unknown")
        provider_name = manifest.get("provider") or info.get("provider", "unknown")
        provider_failure_reason: Optional[str] = None
        _PROVIDER_FAILURE_STATUSES = ("unavailable", "failed", "disabled", "error")
        if provider_status in _PROVIDER_FAILURE_STATUSES:
            if "depth_pro" in (provider_name or ""):
                if provider_status == "error":
                    provider_failure_reason = (
                        "Depth Pro worker process failed: check DEPTH_PRO_PYTHON_PATH, "
                        "venv installation, and worker logs"
                    )
                else:
                    provider_failure_reason = (
                        "Depth Pro unavailable: DEPTH_PRO_ENABLED=false or depth_pro package not installed"
                    )
            else:
                provider_failure_reason = (
                    f"Provider '{provider_name}' reported status '{provider_status}'"
                )

        response: Dict[str, Any] = {
            "session_id": session_id,
            "status": info["status"],
            "provider_status": provider_status,
            "warnings": manifest.get("warnings", []),
            "missing_outputs": summary["missing_outputs"],
            "manifest": manifest,
        }
        if provider_failure_reason:
            response["provider_failure_reason"] = provider_failure_reason
        return response
    except Exception as e:
        info["status"] = "failed"
        raise HTTPException(status_code=500, detail=f"Depth Studio processing failed: {e}")


@app.get("/api/depth-studio/status/{session_id}", dependencies=[Depends(verify_api_key)])
async def depth_studio_status(session_id: str):
    """Return current status and summary for a Depth Studio session."""
    if session_id not in _depth_sessions:
        raise HTTPException(status_code=404, detail=f"Depth Studio session not found: {session_id}")
    info = _depth_sessions[session_id]
    return {
        "session_id": session_id,
        "status": info["status"],
        "provider": info.get("provider"),
        "manifest_path": info.get("manifest_path"),
    }


@app.get("/api/depth-studio/manifest/{session_id}", dependencies=[Depends(verify_api_key)])
async def depth_studio_manifest(session_id: str):
    """Return full depth studio manifest JSON."""
    if session_id not in _depth_sessions:
        raise HTTPException(status_code=404, detail=f"Depth Studio session not found: {session_id}")
    manifest_path = _depth_sessions[session_id].get("manifest_path")
    if not manifest_path or not Path(manifest_path).exists():
        raise HTTPException(status_code=404, detail="Manifest not yet generated")
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _artifact_404(session_id: str, artifact_name: str) -> HTTPException:
    """Build a 404 with session summary embedded so callers understand why the artifact is missing."""
    summary = _depth_session_summary(session_id)
    detail: Dict[str, Any] = {
        "error": f"{artifact_name} not available",
        "session_status": summary["session_status"],
        "provider_status": summary["provider_status"],
        "warnings": summary["warnings"],
        "missing_outputs": summary["missing_outputs"],
    }
    if summary.get("provider_failure_reason"):
        detail["provider_failure_reason"] = summary["provider_failure_reason"]
    return HTTPException(status_code=404, detail=detail)


@app.get("/api/depth-studio/preview/{session_id}", dependencies=[Depends(verify_api_key)])
async def depth_studio_preview(session_id: str):
    """Return depth preview image if available."""
    if session_id not in _depth_sessions:
        raise HTTPException(status_code=404, detail=f"Depth Studio session not found: {session_id}")
    preview = _depth_session_dir(session_id) / "derived" / "depth_preview.png"
    if preview.exists():
        return FileResponse(str(preview), media_type="image/png")
    raise _artifact_404(session_id, "depth_preview")


@app.get("/api/depth-studio/subject-mask/{session_id}", dependencies=[Depends(verify_api_key)])
async def depth_studio_subject_mask(session_id: str):
    """Return binary subject mask PNG."""
    if session_id not in _depth_sessions:
        raise HTTPException(status_code=404, detail=f"Depth Studio session not found: {session_id}")
    p = _depth_session_dir(session_id) / "derived" / "subject_mask.png"
    if p.exists():
        return FileResponse(str(p), media_type="image/png")
    raise _artifact_404(session_id, "subject_mask")


@app.get("/api/depth-studio/cropped-subject/{session_id}", dependencies=[Depends(verify_api_key)])
async def depth_studio_cropped_subject(session_id: str):
    """Return cropped subject JPEG."""
    if session_id not in _depth_sessions:
        raise HTTPException(status_code=404, detail=f"Depth Studio session not found: {session_id}")
    p = _depth_session_dir(session_id) / "derived" / "cropped_subject.jpg"
    if p.exists():
        return FileResponse(str(p), media_type="image/jpeg")
    raise _artifact_404(session_id, "cropped_subject")


@app.get("/api/depth-studio/mask-overlay/{session_id}", dependencies=[Depends(verify_api_key)])
async def depth_studio_mask_overlay(session_id: str):
    """Return subject mask overlay image (green tint on detected foreground)."""
    if session_id not in _depth_sessions:
        raise HTTPException(status_code=404, detail=f"Depth Studio session not found: {session_id}")
    overlay = _depth_session_dir(session_id) / "derived" / "mask_overlay.png"
    if overlay.exists():
        return FileResponse(str(overlay), media_type="image/png")
    raise _artifact_404(session_id, "mask_overlay")


@app.get("/api/depth-studio/mask-stats/{session_id}", dependencies=[Depends(verify_api_key)])
async def depth_studio_mask_stats(session_id: str):
    """Return subject mask statistics JSON."""
    if session_id not in _depth_sessions:
        raise HTTPException(status_code=404, detail=f"Depth Studio session not found: {session_id}")
    stats_path = _depth_session_dir(session_id) / "derived" / "mask_stats.json"
    if stats_path.exists():
        return json.loads(stats_path.read_text(encoding="utf-8"))
    raise _artifact_404(session_id, "mask_stats")


# ─────────────────────────────────────────────────────────────────────────────
# AI 3D GENERATION ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

_ai3d_sessions: Dict[str, Dict[str, Any]] = {}
_ai3d_data_root = Path(settings.data_root) / "ai_3d"


def _ai3d_session_dir(session_id: str) -> Path:
    return _ai3d_data_root / session_id


def _ai3d_artifact_404(session_id: str, artifact_name: str) -> HTTPException:
    """Structured 404 with session summary for missing AI 3D artifacts."""
    info = _ai3d_sessions.get(session_id, {})
    manifest_path = info.get("manifest_path")
    manifest: Dict[str, Any] = {}
    if manifest_path:
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except Exception:
            pass
    detail: Dict[str, Any] = {
        "error": f"{artifact_name} not available",
        "session_status": manifest.get("status") or info.get("status", "unknown"),
        "provider": manifest.get("provider") or info.get("provider", "unknown"),
        "provider_status": manifest.get("provider_status", "unknown"),
        "warnings": manifest.get("warnings", []),
        "errors": manifest.get("errors", []),
    }
    return HTTPException(status_code=404, detail=detail)


@app.get("/api/ai-3d/preflight", dependencies=[Depends(verify_api_key)])
async def ai3d_preflight():
    """
    Run WSL2 preflight checks without inference.

    Returns the 5-step check result:
      wsl_exe, distro, python, worker_script, dry_run_contract

    No GPU is allocated — safe to call at any time to verify the
    wsl_subprocess execution mode is correctly configured.
    """
    try:
        from modules.ai_3d_generation.sf3d_provider import SF3DProvider
        provider = SF3DProvider()
        result = provider.preflight_wsl()
        status_code = 200 if result.get("ok") else 503
        from fastapi.responses import JSONResponse
        return JSONResponse(content=result, status_code=status_code)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preflight error: {exc}")


@app.post("/api/ai-3d/upload", dependencies=[Depends(verify_api_key)])
async def ai3d_upload(
    file: UploadFile = File(...),
    provider: str = Form(default="sf3d"),
):
    """Accept an image or video, create an AI 3D session."""

    session_id = f"ai3d_{uuid.uuid4().hex[:12]}"
    session_dir = _ai3d_session_dir(session_id)
    input_dir = session_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    dest = input_dir / f"upload{suffix}"
    with open(str(dest), "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    _ai3d_sessions[session_id] = {
        "session_id": session_id,
        "status": "uploaded",
        "input_path": str(dest),
        "provider": provider,
        "manifest_path": None,
    }

    from modules.ai_3d_generation.multi_input import write_session_inputs
    write_session_inputs(str(session_dir), "single_image", [dest.name], provider=provider)

    return {

        "session_id": session_id,
        "status": "uploaded",
        "provider": provider,
        "input_path": str(dest),
    }


@app.post("/api/ai-3d/upload-multi", dependencies=[Depends(verify_api_key)])
async def ai3d_upload_multi(
    files: list[UploadFile] = File(...),
    provider: str = Form(default="sf3d"),
):
    """Accept multiple images or videos, create an AI 3D session."""

    if not settings.ai_3d_multi_input_enabled:
        raise HTTPException(status_code=400, detail="Multi-input is disabled")
        
    if len(files) > settings.ai_3d_multi_input_max_files:
        raise HTTPException(status_code=400, detail=f"Too many files. Max: {settings.ai_3d_multi_input_max_files}")

    session_id = f"ai3d_{uuid.uuid4().hex[:12]}"
    session_dir = _ai3d_session_dir(session_id)
    input_dir = session_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    input_files = []
    
    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="upload-multi accepts image files only")

    # Save the first file to upload.ext for backward compatibility of process endpoint
    first_suffix = Path(files[0].filename or "upload.jpg").suffix or ".jpg"
    dest_first = input_dir / f"upload{first_suffix}"
    
    for i, file in enumerate(files):
        suffix = Path(file.filename or f"upload_{i:03d}.jpg").suffix or ".jpg"
        dest = input_dir / f"upload_{i:03d}{suffix}"
        with open(str(dest), "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)
        input_files.append(dest.name)
        
        if i == 0:
            shutil.copy2(str(dest), str(dest_first))

    from modules.ai_3d_generation.multi_input import write_session_inputs
    write_session_inputs(str(session_dir), "multi_image", input_files, provider=provider)

    _ai3d_sessions[session_id] = {
        "session_id": session_id,
        "status": "uploaded",
        "input_path": str(dest_first),
        "provider": provider,
        "manifest_path": None,
    }

    return {
        "session_id": session_id,
        "status": "uploaded",
        "provider": provider,
        "input_path": str(dest_first),
        "uploaded_files_count": len(files),
    }


class _AI3DProcessRequest(BaseModel):
    options: Optional[dict] = None


@app.post("/api/ai-3d/process/{session_id}", dependencies=[Depends(verify_api_key)])
async def ai3d_process(session_id: str, body: Optional[_AI3DProcessRequest] = None):
    """Run AI 3D generation for the uploaded file."""
    if not settings.ai_3d_generation_enabled:
        raise HTTPException(
            status_code=503,
            detail="AI 3D Generation is disabled (AI_3D_GENERATION_ENABLED=false)",
        )

    if session_id not in _ai3d_sessions:
        raise HTTPException(status_code=404, detail=f"AI 3D session not found: {session_id}")

    info = _ai3d_sessions[session_id]
    if info["status"] not in ("uploaded", "failed"):
        return {"session_id": session_id, "status": info["status"],
                "detail": "Already processed or processing"}

    session_dir = _ai3d_session_dir(session_id)
    opts = (body.options if body and body.options else {}) or {}

    # Resolve provider
    from modules.ai_3d_generation.multi_input import load_session_inputs
    session_inputs = load_session_inputs(str(session_dir))
    session_provider = session_inputs.get("provider") if session_inputs else None
    request_provider = opts.get("provider")
    
    resolved_provider = session_provider
    if not resolved_provider:
        resolved_provider = request_provider or info.get("provider") or settings.ai_3d_default_provider

    # Mismatch check
    if request_provider and session_provider and request_provider != session_provider:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "provider_mismatch",
                "message": f"Requested provider '{request_provider}' does not match session provider '{session_provider}'.",
                "session_id": session_id,
            }
        )

    # Authoritative consent enforcement for external providers
    external_providers = {"rodin", "meshy", "tripo"}
    if resolved_provider in external_providers:
        consent = opts.get("external_provider_consent")
        if consent is not True:
            logger.warning(f"[API] Denying external provider '{resolved_provider}' for session {session_id}: consent missing")
            raise HTTPException(
                status_code=400,
                detail={"error": "external_provider_consent_required"}
            )

    info["status"] = "processing"


    try:
        from modules.ai_3d_generation.pipeline import generate_ai_3d
        manifest = generate_ai_3d(
            session_id=session_id,
            input_file_path=info["input_path"],
            output_base_dir=str(session_dir),
            provider_name=resolved_provider,
            options=opts,
        )

        info["status"] = manifest.get("status", "failed")
        info["manifest_path"] = str(session_dir / "manifests" / "ai3d_manifest.json")

        # Determine which expected outputs are missing
        derived = session_dir / "derived"
        expected = {
            "prepared_input": derived / "ai3d_input.png",
            "output_glb":     derived / "output.glb",
        }
        missing_outputs = [k for k, p in expected.items() if not p.exists()]

        provider_status = manifest.get("provider_status", "unknown")

        # 409 — GPU lock held (transient; client should retry)
        if provider_status == "busy":
            info["status"] = "uploaded"  # Reset so the session can be retried
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "sf3d_job_already_running",
                    "message": "An SF3D inference job is already running. Retry after it completes.",
                    "session_id": session_id,
                },
            )

        provider_failure_reason: Optional[str] = None
        if provider_status in ("unavailable", "failed", "disabled", "error"):
            p_name = manifest.get("provider", provider_name)
            if "sf3d" in (p_name or ""):
                errors = manifest.get("errors", [])
                err_str = errors[0] if errors else provider_status
                if "sf3d_disabled" in err_str:
                    provider_failure_reason = (
                        "SF3D is disabled (SF3D_ENABLED=false)"
                    )
                elif "sf3d_python_missing" in err_str:
                    provider_failure_reason = (
                        "SF3D Python not found — install under "
                        "external/stable-fast-3d/.venv_sf3d and set SF3D_PYTHON_PATH"
                    )
                elif "sf3d_worker_missing" in err_str:
                    provider_failure_reason = (
                        "SF3D worker script not found at SF3D_WORKER_SCRIPT path"
                    )
                else:
                    provider_failure_reason = (
                        f"SF3D provider unavailable: {err_str}"
                    )

        _worker_meta = manifest.get("worker_metadata") or {}
        response: Dict[str, Any] = {
            "session_id": session_id,
            "status": info["status"],
            "execution_mode": manifest.get("execution_mode",
                                           getattr(settings, "sf3d_execution_mode", "disabled")),
            "provider_status": provider_status,
            "output_glb_path": manifest.get("output_glb_path"),
            "missing_outputs": missing_outputs,
            "peak_mem_mb": manifest.get("peak_mem_mb") or _worker_meta.get("peak_mem_mb"),
            "worker_metadata": _worker_meta,
            "warnings": manifest.get("warnings", []),
            "errors": manifest.get("errors", []),
            "manifest": manifest,
            
            # Phase 1: multi-candidate fields
            "input_mode": manifest.get("input_mode"),
            "uploaded_files_count": manifest.get("uploaded_files_count"),
            "candidate_count": manifest.get("candidate_count"),
            "selected_candidate_id": manifest.get("selected_candidate_id"),
            "candidate_summary": manifest.get("candidate_ranking"),

            # Phase 4C: mesh geometry + AR readiness (top-level for easy consumption)
            "mesh_stats": manifest.get("mesh_stats"),
            "ar_readiness": manifest.get("ar_readiness"),

            # Phase 4D: GLB structural validation summary
            "glb_validation": manifest.get("glb_validation"),
        }
        if provider_failure_reason:
            response["provider_failure_reason"] = provider_failure_reason
        return response
    except HTTPException:
        raise
    except ValueError as e:
        info["status"] = "failed"
        err_str = str(e)
        if err_str.startswith("unknown_ai3d_provider:"):
            provider_name_bad = err_str.split(":", 1)[1]
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "unknown_ai3d_provider",
                    "provider": provider_name_bad,
                },
            )
        from modules.ai_3d_generation.sanitization import sanitize_external_provider_error
        err_msg = sanitize_external_provider_error(err_str)
        raise HTTPException(status_code=400, detail=f"Invalid request: {err_msg}")
    except Exception as e:
        info["status"] = "failed"
        from modules.ai_3d_generation.sanitization import sanitize_external_provider_error
        err_msg = sanitize_external_provider_error(str(e))
        raise HTTPException(status_code=500, detail=f"AI 3D processing failed: {err_msg}")



@app.get("/api/ai-3d/status/{session_id}", dependencies=[Depends(verify_api_key)])
async def ai3d_status(session_id: str):
    """Return session status."""
    if session_id not in _ai3d_sessions:
        raise HTTPException(status_code=404, detail=f"AI 3D session not found: {session_id}")
    info = _ai3d_sessions[session_id]
    manifest: Dict[str, Any] = {}
    if info.get("manifest_path"):
        try:
            manifest = json.loads(
                Path(info["manifest_path"]).read_text(encoding="utf-8")
            )
        except Exception:
            pass
    return {
        "session_id": session_id,
        "status": info["status"],
        "provider": info.get("provider"),
        "provider_status": manifest.get("provider_status"),
        "manifest_path": info.get("manifest_path"),
    }


@app.get("/api/ai-3d/output/{session_id}", dependencies=[Depends(verify_api_key)])
async def ai3d_output(session_id: str):
    """Serve generated GLB file."""
    if session_id not in _ai3d_sessions:
        raise HTTPException(status_code=404, detail=f"AI 3D session not found: {session_id}")
    derived = _ai3d_session_dir(session_id) / "derived"
    # Try manifest-reported path first, then fallback scan
    info = _ai3d_sessions[session_id]
    glb_path = None
    if info.get("manifest_path"):
        try:
            m = json.loads(Path(info["manifest_path"]).read_text(encoding="utf-8"))
            glb_path = m.get("output_glb_path")
        except Exception:
            pass
    if not glb_path or not Path(glb_path).exists():
        candidates = list(derived.glob("*.glb"))
        glb_path = str(candidates[0]) if candidates else None
    if glb_path and Path(glb_path).exists():
        return FileResponse(str(glb_path), media_type="model/gltf-binary",
                            filename="output.glb")
    raise _ai3d_artifact_404(session_id, "output_glb")


@app.get("/api/ai-3d/prepared-input/{session_id}", dependencies=[Depends(verify_api_key)])
async def ai3d_prepared_input(session_id: str):
    """Serve prepared input image (ai3d_input.png)."""
    if session_id not in _ai3d_sessions:
        raise HTTPException(status_code=404, detail=f"AI 3D session not found: {session_id}")
    p = _ai3d_session_dir(session_id) / "derived" / "ai3d_input.png"
    if p.exists():
        return FileResponse(str(p), media_type="image/png")
    raise _ai3d_artifact_404(session_id, "prepared_input")


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
