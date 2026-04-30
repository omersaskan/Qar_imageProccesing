import pytest
import shutil
import os
from pathlib import Path
from modules.operations.settings import settings
from modules.asset_registry.registry import AssetRegistry
from modules.shared_contracts.models import AssetMetadata
from modules.shared_contracts.lifecycle import AssetStatus

def test_resolve_executable_path_exists():
    # Test with a file that exists (using current file as dummy)
    this_file = str(Path(__file__).resolve())
    resolved = settings.resolve_executable(this_file)
    assert resolved == this_file

def test_resolve_executable_path_name_only():
    # Test with a command that should be in PATH (like 'py' or 'python' on windows)
    # Using 'py' as it's common on Windows as per user env
    cmd = "py"
    resolved = settings.resolve_executable(cmd)
    if resolved:
        assert Path(resolved).exists()
    else:
        # Fallback for systems where 'py' might not be there but 'python' is
        resolved = settings.resolve_executable("python")
        if resolved:
            assert Path(resolved).exists()

def test_asset_metadata_phase_b_defaults():
    meta = AssetMetadata(asset_id="test_id", product_id="prod_id")
    assert meta.geometry_source == "photogrammetry"
    assert meta.production_status == "production_candidate"
    assert meta.may_override_recapture_required is False
    assert meta.ai_generated is False
    assert meta.requires_manual_review is False

def test_registry_publish_guard_ai_generated(tmp_path):
    registry = AssetRegistry(data_root=str(tmp_path))
    product_id = "test_prod"
    asset_id = "test_asset_ai"
    
    meta = AssetMetadata(
        asset_id=asset_id, 
        product_id=product_id, 
        ai_generated=True
    )
    registry.register_asset(meta)
    
    with pytest.raises(ValueError, match="AI-generated"):
        registry.publish_asset(product_id, asset_id)

def test_registry_publish_guard_manual_review(tmp_path):
    registry = AssetRegistry(data_root=str(tmp_path))
    product_id = "test_prod"
    asset_id = "test_asset_review"
    
    meta = AssetMetadata(
        asset_id=asset_id, 
        product_id=product_id, 
        requires_manual_review=True
    )
    registry.register_asset(meta)
    
    # Should reject because it's not approved
    with pytest.raises(ValueError, match="manual review and has not been approved"):
        registry.publish_asset(product_id, asset_id)
        
    # Grant approval
    registry.grant_approval(asset_id, "review")
    
    # Now it should publish
    registry.publish_asset(product_id, asset_id)
    history = registry.get_history(product_id)
    assert any(h["asset_id"] == asset_id and h["status"] == "published" for h in history)

def test_registry_publish_normal_photogrammetry(tmp_path):
    registry = AssetRegistry(data_root=str(tmp_path))
    product_id = "test_prod"
    asset_id = "test_asset_normal"
    
    meta = AssetMetadata(
        asset_id=asset_id, 
        product_id=product_id,
        ai_generated=False,
        requires_manual_review=False
    )
    registry.register_asset(meta)
    
    # Should publish normally
    registry.publish_asset(product_id, asset_id)
    history = registry.get_history(product_id)
    assert any(h["asset_id"] == asset_id and h["status"] == "published" for h in history)

def test_readme_ports():
    readme_path = Path("README.md")
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
        # Should NOT contain 8000 for UI/Docs
        # (Checking for common patterns used in README)
        assert "localhost:8000" not in content
        assert "localhost:8001" in content
