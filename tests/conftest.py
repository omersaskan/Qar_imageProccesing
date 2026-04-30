import os
import sys
from pathlib import Path
import pytest

# Ensure the repository root is in sys.path
root_dir = Path(__file__).parent.parent.absolute()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """
    Ensure tests run in a controlled environment.
    Redirects data_root to a temporary directory for all tests.
    """
    # Force LOCAL_DEV for tests unless explicitly overridden
    monkeypatch.setenv("ENV", "local_dev")
    
    # We don't want tests writing to the real data folder
    test_data_root = root_dir / "tests" / "tmp_data"
    test_data_root.mkdir(exist_ok=True)
    monkeypatch.setenv("DATA_ROOT", str(test_data_root))
    
    # Mock missing binaries if needed, or ensure they don't crash startup
    # (Settings handles this with shutil.which, but we can force it here)
    
    yield
    
    # Cleanup could go here, but usually better to leave it for individual tests 
    # or use a temp directory fixture.
