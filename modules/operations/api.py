from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import List, Dict, Any
import os
import json
import shutil
import uuid

from modules.asset_registry.registry import AssetRegistry
from modules.capture_workflow.session_manager import SessionManager
from modules.operations.logging_config import get_component_logger, setup_logging
from modules.operations.worker import worker_instance
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

registry = AssetRegistry()
session_manager = SessionManager()
embedded_worker_enabled = os.getenv("MESHYSIZ_EMBEDDED_WORKER", "true").lower() == "true"


@app.on_event("startup")
async def startup_event():
    if embedded_worker_enabled:
        worker_instance.start()


@app.on_event("shutdown")
async def shutdown_event():
    if embedded_worker_enabled:
        worker_instance.stop()

@app.post("/api/sessions/upload")
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

@app.get("/api/products")
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

@app.get("/api/worker/status")
async def worker_status():
    return {
        "embedded": embedded_worker_enabled,
        "running": worker_instance.running,
    }

@app.get("/api/products/{product_id}/history")
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

@app.get("/api/logs")
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
