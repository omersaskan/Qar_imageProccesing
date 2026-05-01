import os
import shutil
import time
from pathlib import Path
from modules.capture_workflow.session_manager import SessionManager
from modules.operations.worker import worker_instance

def run():
    source_video = Path(r"C:\Users\Lenovo\.gemini\antigravity\scratch\Qar_imageProccesing\data\captures\cap_48333fc5\video\raw_video.mp4")
    if not source_video.exists():
        print(f"Source video not found: {source_video}")
        return

    session_id = f"cap_{int(time.time())}"
    print(f"Creating new session: {session_id}")
    
    manager = SessionManager(Path("data"))
    session = manager.create_session(session_id, "prod_debug", "op_debug")
    
    # Copy video
    dest_video_dir = Path("data/captures") / session_id / "video"
    dest_video_dir.mkdir(parents=True, exist_ok=True)
    dest_video = dest_video_dir / "raw_video.mp4"
    shutil.copy2(source_video, dest_video)
    
    print(f"Video copied to {dest_video}")
    print("Starting worker process...")
    
    try:
        os.system(f"py tools/debug_process_session.py {session_id}")
        print("Worker finished processing.")
    except Exception as e:
        print(f"Worker encountered an error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run()
