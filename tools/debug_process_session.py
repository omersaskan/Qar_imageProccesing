import sys
import os
from pathlib import Path
import json
import logging

# Add project root and modules to sys.path
project_root = Path(__file__).parent.parent.resolve()
modules_path = project_root / "modules"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(modules_path) not in sys.path:
    sys.path.insert(0, str(modules_path))

from modules.operations.worker import IngestionWorker
from modules.operations.logging_config import setup_logging
from modules.shared_contracts.models import CaptureSession
from modules.shared_contracts.lifecycle import AssetStatus

def debug_session(session_id: str):
    """
    Manually processes a single session in the foreground to catch errors.
    """
    print(f"--- Debugging Session: {session_id} ---")
    
    # 1. Setup Logging to Console and File
    setup_logging()
    logger = logging.getLogger("debug_tool")
    
    # 2. Check if session file exists
    session_file = project_root / "data" / "sessions" / f"{session_id}.json"
    if not session_file.exists():
        print(f"ERROR: Session file not found at {session_file}")
        return

    # 3. Load Session
    with open(session_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        session = CaptureSession.model_validate(data)
    
    print(f"Current Status: {session.status}")
    print(f"Product ID: {session.product_id}")
    
    # 4. Initialize Worker (stop background loop immediately for manual debug)
    worker = IngestionWorker()
    worker.stop()
    
    # 5. Process Step by Step
    try:
        if session.status == AssetStatus.CREATED:
            print("\n>>> Stepping from CREATED to CAPTURED...")
            worker._advance_session(session, AssetStatus.CAPTURED, "Extracting frames...")
            # Reload session
            with open(session_file, "r") as f: session = CaptureSession.model_validate(json.load(f))
            
        if session.status == AssetStatus.CAPTURED:
            print("\n>>> Stepping from CAPTURED to RECONSTRUCTED...")
            worker._advance_session(session, AssetStatus.RECONSTRUCTED, "Reconstructing geometry...")
            # Reload session
            with open(session_file, "r") as f: session = CaptureSession.model_validate(json.load(f))

        if session.status == AssetStatus.RECONSTRUCTED:
            print("\n>>> Stepping from RECONSTRUCTED to CLEANED (Texturing)...")
            worker._advance_session(session, AssetStatus.CLEANED, "Cleaning and texturing...")
            # Reload session
            with open(session_file, "r") as f: session = CaptureSession.model_validate(json.load(f))

        if session.status == AssetStatus.CLEANED:
            print("\n>>> Finalizing Ingestion...")
            worker._finalize_ingestion(session)
            print("\n✅ Session finalized and registered successfully!")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR DURING DEBUG: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Scan for existing sessions if none provided
        sessions_dir = project_root / "data" / "sessions"
        if sessions_dir.exists():
            files = list(sessions_dir.glob("*.json"))
            if files:
                print("Available sessions to debug:")
                for f in files:
                    print(f"  - {f.stem}")
                print(f"\nUsage: python {sys.argv[0]} <session_id>")
            else:
                print("No active sessions found in data/sessions/")
        else:
            print("Session directory data/sessions/ does not exist.")
    else:
        debug_session(sys.argv[1])
