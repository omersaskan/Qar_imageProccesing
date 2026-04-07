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
    Uses msvcrt on Windows and fcntl on Unix.
    """
    def __init__(self, file_path: str | Path, timeout: float = 10.0, delay: float = 0.05):
        self.lock_file = Path(str(file_path) + ".lock")
        self.timeout = timeout
        self.delay = delay
        self.fd = None

    def __enter__(self):
        start_time = time.time()
        while True:
            try:
                # Open with O_CREAT | O_EXCL is atomic
                self.fd = os.open(str(self.lock_file), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return self
            except FileExistsError:
                # Check for stale lock (e.g., if a process crashed while holding the lock)
                try:
                    stats = os.stat(str(self.lock_file))
                    mtime = stats.st_mtime
                    if time.time() - mtime > 15.0: # Increased threshold to 15s to be safer
                        try:
                            os.remove(str(self.lock_file))
                            # Small sleep to allow FS to settle on Windows
                            time.sleep(0.1)
                            continue
                        except OSError:
                            pass
                except OSError:
                    pass

                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Could not acquire lock on {self.lock_file} after {self.timeout}s. Possible deadlock or long-running process.")
                time.sleep(self.delay)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            
            try:
                if self.lock_file.exists():
                    os.remove(str(self.lock_file))
            except OSError:
                pass

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
