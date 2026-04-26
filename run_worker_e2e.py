import sys
import os
import time
from pathlib import Path

# Add the project root to sys.path
sys.path.append(os.getcwd())

from modules.operations.worker import IngestionWorker
from modules.operations.settings import settings

def run_worker():
    print(f"Starting worker on data_root: {settings.data_root}")
    worker = IngestionWorker(data_root=settings.data_root)
    # Use a shorter interval for the E2E run
    worker.interval = 5
    
    # We want to run it synchronously for the E2E test so we can see output
    print("Worker loop starting. Press Ctrl+C to stop.")
    try:
        while True:
            print(f"[{time.strftime('%H:%M:%S')}] Processing iteration...")
            worker._process_pending_sessions()
            time.sleep(worker.interval)
    except KeyboardInterrupt:
        print("Worker stopped by user.")

if __name__ == "__main__":
    run_worker()
