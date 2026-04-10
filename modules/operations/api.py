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
app = FastAPI(title="Meshysiz Asset Factory API")

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
        return # Optional in dev
    
    if not x_api_key or x_api_key != settings.pilot_api_key:
        # Note: We do NOT log the key itself as per instruction
        logger.warning(f"Unauthorized API access attempt from {settings.env.value} environment")
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

registry = AssetRegistry(data_root=str(Path(settings.data_root) / "registry"))
session_manager = SessionManager(data_root=settings.data_root)
embedded_worker_enabled = os.getenv("MESHYSIZ_EMBEDDED_WORKER", "true").lower() == "true"


@app.on_event("startup")
async def startup_event():
    try:
        settings.validate_setup()
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        # Don't stop the whole server if just config is wrong, 
        # but the ready check will fail.
        
    if embedded_worker_enabled:
        worker_instance.start()


@app.on_event("shutdown")
async def shutdown_event():
    if embedded_worker_enabled:
        worker_instance.stop()

@app.get("/api/health")
async def health_check():
    """Lightweight alive check."""
    return {"status": "ok", "env": settings.env.value}

@app.get("/api/ready", dependencies=[Depends(verify_api_key)])
async def readiness_check():
    """Checks if the system is fully configured and ready to process jobs."""
    issues = []
    
    # Check data root
    dr = Path(settings.data_root)
    if not dr.exists(): issues.append(f"Data root missing: {dr}")
    
    # Check binaries
    cp = Path(settings.colmap_path)
    if not cp.exists() and not cp.with_suffix(".exe").exists() and not cp.with_suffix(".bat").exists():
        issues.append(f"COLMAP bin missing: {cp}")
        
    return {
        "status": "ready" if not issues else "not_ready",
        "issues": issues,
        "env": settings.env.value
    }

@app.post("/api/sessions/upload", dependencies=[Depends(verify_api_key)])
async def upload_video(
    product_id: str = Form(...),
    operator_id: str = Form("dashboard_user"),
    file: UploadFile = File(...)
):
    """
    Handles video upload, creates a new CaptureSession, and saves the file.
    """
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: .mp4, .mov, .avi")

    # Generate unique session ID
    session_id = f"cap_{uuid.uuid4().hex[:8]}"
    
    try:
        # 1. Create Session folders and record
        session = session_manager.create_session(session_id, product_id, operator_id)
        
        # 2. Setup video folder
        capture_path = session_manager.get_capture_path(session_id)
        video_dir = capture_path / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Save uploaded file
        video_path = video_dir / "raw_video.mp4" # Normalize to mp4 for internal use
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        logger.info(f"Video uploaded successfully for session {session_id}. Size: {file_size_mb:.2f} MB", 
                    extra={"job_id": session_id})
        
        return {
            "session_id": session_id,
            "product_id": product_id,
            "status": "uploaded",
            "path": str(video_path)
        }
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error during upload: {str(e)}")

@app.get("/api/products", dependencies=[Depends(verify_api_key)])
async def list_products():
    """Returns a unified list of registered products and active capture sessions."""
    products_map = {}
    
    # 1. Scan Registered Assets
    meta_dir = Path("data/registry/meta")
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
                "status": "registered"
            }

    # 2. Scan Active Sessions (In-progress uploads)
    sessions_dir = Path("data/sessions")
    if sessions_dir.exists():
        for file in sessions_dir.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                    p_id = session_data.get("product_id")
                    if p_id and p_id not in products_map:
                        # Product exists in sessions but not yet in registry
                        products_map[p_id] = {
                            "id": p_id,
                            "active_id": None,
                            "asset_count": 0,
                            "last_updated": file.stat().st_mtime,
                            "status": "processing"
                        }
                    elif p_id in products_map and products_map[p_id]["status"] == "registered":
                        # If it has both, we can flag it as having active updates
                        if session_data.get("status") not in ["published", "failed"]:
                            products_map[p_id]["has_active_session"] = True
            except:
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
        # 1. Try Registry first
        return registry.get_history(product_id)
    except Exception:
        # 2. Fallback: Scan sessions for this product
        sessions = []
        sessions_dir = Path("data/sessions")
        if sessions_dir.exists():
            for file in sessions_dir.glob("*.json"):
                with open(file, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                    if s_data.get("product_id") == product_id:
                        sessions.append({
                            "asset_id": s_data.get("session_id"),
                            "version": "In-Progress",
                            "status": s_data.get("status"),
                            "is_active": False,
                            "approved": False,
                            "audit": [{"action": "session_created", "asset_id": s_data.get("session_id")}]
                        })
        if not sessions:
            raise HTTPException(status_code=404, detail="Product history not found")
        return sessions

@app.get("/api/logs", dependencies=[Depends(verify_api_key)])
async def get_logs(limit: int = 50):
    """Reads the last N lines from the operational log file."""
    log_file = Path("data/logs/factory.log")
    if not log_file.exists():
        return []
        
    logs = []
    with open(log_file, "r", encoding="utf-8") as f:
        # Simple tail implementation
        lines = f.readlines()[-limit:]
        for line in lines:
            try:
                logs.append(json.loads(line))
            except:
                logs.append({"message": line.strip(), "level": "INFO"})
    return logs[::-1] # Newest first

@app.get("/api/sessions/{session_id}/guidance", dependencies=[Depends(verify_api_key)])
async def get_session_guidance(session_id: str):
    """Returns the structured guidance report for a session."""
    reports_dir = Path(settings.op.data_root) / "captures" / session_id / "reports"
    guidance_path = reports_dir / "guidance_report.json"
    
    if not guidance_path.exists():
        # Minimal fallback if not yet generated by worker
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session_id,
            "status": session.status,
            "next_action": "Guidance not yet generated. Please wait...",
            "messages": []
        }
        
    with open(guidance_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/sessions/{session_id}/guidance/summary", dependencies=[Depends(verify_api_key)])
async def get_session_guidance_summary(session_id: str):
    """Returns the human-readable Markdown summary of the guidance."""
    reports_dir = Path(settings.op.data_root) / "captures" / session_id / "reports"
    summary_path = reports_dir / "guidance_summary.md"
    
    if not summary_path.exists():
        return "# Guidance Pending\nPlease wait for the system to analyze your capture."
        
    with open(summary_path, "r", encoding="utf-8") as f:
        return f.read()

# Asset Blobs (GLB Files)
blobs_dir = Path("data/registry/blobs")
blobs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/assets/blobs", StaticFiles(directory="data/registry/blobs"), name="blobs")

# Mount the UI directory if it exists
ui_dir = Path("ui")
if ui_dir.exists():
    app.mount("/", StaticFiles(directory="ui", html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
