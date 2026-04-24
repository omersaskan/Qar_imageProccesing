import uvicorn
import os
import sys
from pathlib import Path

# Add project root and modules to sys.path for universal import resolution
project_root = Path(__file__).parent.resolve()
modules_path = project_root / "modules"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(modules_path) not in sys.path:
    sys.path.insert(0, str(modules_path))

if __name__ == "__main__":
    print(f"--- Meshysiz Asset Factory Dashboard ---")
    print(f"Starting server on http://localhost:8001")
    
    # 1. Start Ingestion Worker (Simulated pipeline)
    try:
        from modules.operations.worker import worker_instance
        worker_instance.start()
    except Exception as e:
        print(f"Failed to start worker: {e}")

    # 2. Start API
    from modules.operations.api import app
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001)
    finally:
        # Shutdown worker gracefully
        print("Shutting down worker...")
        if 'worker_instance' in locals():
            worker_instance.stop()
    
