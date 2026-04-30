import uvicorn
import os
import sys
from pathlib import Path

# Add project root and modules to sys.path for universal import resolution
project_root = Path(__file__).parent.resolve()
modules_path = project_root / "modules"
cert_path = Path(__file__).parent / "certs" / "cert.pem"
key_path = Path(__file__).parent / "certs" / "key.pem"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(modules_path) not in sys.path:
    sys.path.insert(0, str(modules_path))

if __name__ == "__main__":
    print(f"Starting server on http://localhost:8001 (or https if certs found)")
    
    ssl_config = {}
    if cert_path.exists() and key_path.exists():
        ssl_config = {
            "ssl_certfile": str(cert_path),
            "ssl_keyfile": str(key_path)
        }
    # 1. Start Ingestion Worker (Simulated pipeline)
    try:
        from modules.operations.worker import worker_instance
        worker_instance.start()
    except Exception as e:
        print(f"Failed to start worker: {e}")

    # 2. Start API
    from modules.operations.api import app
    try:
        uvicorn.run(app, host="0.0.0.0", port=8001, **ssl_config)
    finally:
        # Shutdown worker gracefully
        print("Shutting down worker...")
        if 'worker_instance' in locals():
            worker_instance.stop()
    
