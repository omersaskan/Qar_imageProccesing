import pytest
import time
import os
import json
import threading
from pathlib import Path
from modules.utils.file_persistence import FileLock

def test_file_lock_stale_recovery(tmp_path):
    lock_file_base = tmp_path / "test_file"
    lock_file = tmp_path / "test_file.lock"
    
    # 1. Create a stale lock file manually
    lock_file.write_text(json.dumps({"pid": 9999, "timestamp": time.time() - 20}))
    # Update mtime to 20s ago so it's actually stale according to stat()
    stale_time = time.time() - 20
    os.utime(str(lock_file), (stale_time, stale_time))
    
    # 2. Try to acquire lock with stale_threshold < 20
    with FileLock(lock_file_base, stale_threshold=5.0, timeout=1.0) as lock:
        assert lock.fd is not None
        assert lock_file.exists()
        
    # Lock should be released now
    assert not lock_file.exists()

def test_file_lock_timeout_on_active_lock(tmp_path):
    lock_file_base = tmp_path / "test_file_active"
    
    # 1. Hold a lock in another thread or just open it
    # We use a dummy fd to pretend it's active
    with FileLock(lock_file_base) as lock1:
        # 2. Attempt to acquire again should timeout
        with pytest.raises(TimeoutError):
            with FileLock(lock_file_base, timeout=0.1) as lock2:
                pass

def test_file_lock_corrupt_recovery_only_if_stale(tmp_path):
    lock_file_base = tmp_path / "test_corrupt"
    lock_file = tmp_path / "test_corrupt.lock"
    
    # 1. Create a corrupt lock file that's NOT stale
    lock_file.write_text("not-json")
    
    # 2. Attempt to acquire with high threshold (should timeout because it's not stale)
    with pytest.raises(TimeoutError):
        with FileLock(lock_file_base, stale_threshold=100.0, timeout=0.1) as lock:
            pass
            
    # 3. Now make it stale
    old_time = time.time() - 200
    os.utime(lock_file, (old_time, old_time))
    
    # 4. Attempt to acquire (should recover)
    with FileLock(lock_file_base, stale_threshold=10.0, timeout=1.0) as lock:
        assert lock.fd is not None

def test_file_lock_does_not_steal_stale_lock_from_live_pid(tmp_path):
    lock_file_base = tmp_path / "test_live_pid"
    lock_file = tmp_path / "test_live_pid.lock"

    lock_file.write_text(json.dumps({"pid": os.getpid(), "timestamp": time.time() - 200}))
    old_time = time.time() - 200
    os.utime(lock_file, (old_time, old_time))

    with pytest.raises(TimeoutError):
        with FileLock(lock_file_base, stale_threshold=1.0, timeout=0.1):
            pass
