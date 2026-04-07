import os
import json
import time
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Dict

try:
    import msvcrt
    _HAS_MSVCRT = True
except ImportError:
    _HAS_MSVCRT = False
    try:
        import fcntl
        _HAS_FCNTL = True
    except ImportError:
        _HAS_FCNTL = False

class FileLock:
    """
    A simple, robust file locking context manager.
    Uses O_CREAT | O_EXCL for atomic locking.
    Supports stale lock recovery and Windows deletion retries.
    """
    def __init__(self, file_path: str | Path, timeout: float = 10.0, delay: float = 0.05, stale_threshold: float = 10.0):
        self.lock_file = Path(str(file_path) + ".lock")
        self.timeout = timeout
        self.delay = delay
        self.stale_threshold = stale_threshold
        self.fd = None

    def _write_metadata(self):
        """Writes current PID and timestamp to the lock file for diagnostics."""
        try:
            if self.fd is not None:
                metadata = {
                    "pid": os.getpid(),
                    "timestamp": time.time(),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                os.write(self.fd, json.dumps(metadata).encode('utf-8'))
        except Exception:
            pass

    def _safe_remove(self, path: Path, retries: int = 3):
        """Attempts to remove a file with retries, especially for Windows file system lag."""
        for i in range(retries):
            try:
                if path.exists():
                    os.remove(str(path))
                return True
            except OSError:
                if i < retries - 1:
                    time.sleep(0.05)
        return False

    def __enter__(self):
        start_time = time.time()
        while True:
            try:
                # Open with O_CREAT | O_EXCL is atomic
                self.fd = os.open(str(self.lock_file), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                self._write_metadata()
                return self
            except FileExistsError:
                # Check for stale lock
                try:
                    stats = self.lock_file.stat()
                    mtime = stats.st_mtime
                    size = stats.st_size
                    
                    is_stale = (time.time() - mtime) > self.stale_threshold
                    
                    # User requirement: Cleanup empty/corrupt ONLY if stale
                    if is_stale:
                        # Attempt to read metadata to see if it's corrupt
                        is_corrupt = False
                        if size > 0:
                            try:
                                with open(self.lock_file, 'r') as f:
                                    json.load(f)
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                is_corrupt = True
                        else:
                            is_corrupt = True # Empty file
                        
                        if is_corrupt or is_stale:
                            # If it's stale, we recover regardless of corruption
                            # but we log if it was also corrupt
                            reason = "stale" if not is_corrupt else "stale and corrupt/empty"
                            # We don't have a logger here, but we could use print or just proceed
                            # Since this is a low-level util, we just recover.
                            if self._safe_remove(self.lock_file):
                                continue # Retry acquisition
                except (OSError, FileNotFoundError):
                    pass

                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Could not acquire lock on {self.lock_file} after {self.timeout}s.")
                time.sleep(self.delay)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
            
        self._safe_remove(self.lock_file)

def atomic_write_json(file_path: str | Path, data: Any, indent: int = 2):
    """
    Writes JSON data to a file atomically using a temporary file and os.replace.
    Ensures that the file is never in a partially written state.
    """
    path = Path(file_path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Use a temporary file in the same directory to ensure os.replace is atomic (same FS)
    with tempfile.NamedTemporaryFile('w', dir=str(parent), delete=False, encoding='utf-8') as tf:
        json.dump(data, tf, indent=indent, ensure_ascii=False)
        temp_name = tf.name

    try:
        os.replace(temp_name, str(path))
    except Exception:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        raise

def calculate_checksum(file_path: str | Path) -> str:
    """Calculates the SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
