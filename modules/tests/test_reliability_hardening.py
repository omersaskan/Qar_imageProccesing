import pytest
import os
import shutil
from pathlib import Path
from modules.utils.path_safety import validate_safe_path, validate_identifier, PathSafetyError
from modules.utils.file_persistence import atomic_write_json, FileLock, calculate_checksum
from modules.asset_registry.registry import AssetRegistry
from modules.shared_contracts.models import AssetMetadata

@pytest.fixture
def temp_data():
    path = Path("temp_test_data").resolve()
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True)
    yield path
    # On Windows, we might need a small delay for handles to clear
    import time
    time.sleep(0.1)
    if path.exists():
        try:
            shutil.rmtree(path)
        except PermissionError:
            # If it still fails, just leave it for now
            pass

def test_path_safety():
    base = Path("data").resolve()
    # Safe path
    assert validate_safe_path(base, "job_1").is_relative_to(base)
    
    # Traversal attempt
    with pytest.raises(PathSafetyError):
        validate_safe_path(base, "../secret.txt")
    
    # Absolute path outside root
    with pytest.raises(PathSafetyError):
        validate_safe_path(base, "/etc/passwd")

def test_identifier_validation():
    assert validate_identifier("job-123") == "job-123"
    assert validate_identifier("asset_456") == "asset_456"
    
    with pytest.raises(ValueError):
        validate_identifier("job#1") # Invalid char
    
    with pytest.raises(ValueError):
        validate_identifier("a" * 65) # Too long

def test_atomic_write(temp_data):
    file_path = temp_data / "test.json"
    data = {"key": "value"}
    atomic_write_json(file_path, data)
    
    assert file_path.exists()
    with open(file_path, "r") as f:
        import json
        assert json.load(f) == data

def test_json_registry(temp_data):
    registry_path = temp_data / "registry"
    registry = AssetRegistry(data_root=str(registry_path))
    
    metadata = AssetMetadata(
        asset_id="test_asset",
        product_id="prod_1",
        version="1.0"
    )
    
    registry.register_asset(metadata)
    
    # Verify directory structure
    assert (registry_path / "meta" / "prod_1.json").exists()
    
    # Verify persistence
    asset = registry.get_asset("test_asset")
    assert asset.asset_id == "test_asset"
    assert asset.product_id == "prod_1"
    
    # Verify active version
    registry.set_active_version("prod_1", "test_asset")
    assert registry._get_active_id("prod_1") == "test_asset"
    
    # Duplicate check
    from modules.shared_contracts.errors import DuplicateAssetError
    with pytest.raises(DuplicateAssetError):
        registry.register_asset(metadata)

def test_checksum(temp_data):
    file_path = temp_data / "hello.txt"
    with open(file_path, "w") as f:
        f.write("hello world")
    
    checksum = calculate_checksum(file_path)
    # Expected SHA-256 for "hello world"
    import hashlib
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert checksum == expected
