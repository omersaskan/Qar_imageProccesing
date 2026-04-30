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
    # Test with a command that should be in PATH
    cmd = "py"
    resolved = settings.resolve_executable(cmd)
    if resolved:
        assert Path(resolved).exists()
    else:
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

def test_registry_publish_guard_geometry_source(tmp_path):
    registry = AssetRegistry(data_root=str(tmp_path))
    product_id = "test_prod"
    asset_id = "test_asset_sam3d"
    
    meta = AssetMetadata(
        asset_id=asset_id, 
        product_id=product_id, 
        geometry_source="sam3d"
    )
    registry.register_asset(meta)
    
    with pytest.raises(ValueError, match="source is 'sam3d'"):
        registry.publish_asset(product_id, asset_id)

def test_registry_set_active_guards(tmp_path):
    registry = AssetRegistry(data_root=str(tmp_path))
    product_id = "test_prod"
    
    # 1. AI Generated
    meta_ai = AssetMetadata(asset_id="ai", product_id=product_id, ai_generated=True)
    registry.register_asset(meta_ai)
    with pytest.raises(ValueError, match="AI-generated"):
        registry.set_active_version(product_id, "ai")
        
    # 2. Requires Review
    meta_rev = AssetMetadata(asset_id="rev", product_id=product_id, requires_manual_review=True)
    registry.register_asset(meta_rev)
    with pytest.raises(ValueError, match="requires manual review"):
        registry.set_active_version(product_id, "rev")
        
    # 3. Non-photogrammetry
    meta_source = AssetMetadata(asset_id="src", product_id=product_id, geometry_source="meshy")
    registry.register_asset(meta_source)
    with pytest.raises(ValueError, match="source is 'meshy'"):
        registry.set_active_version(product_id, "src")

def test_registry_update_publish_state_guard(tmp_path):
    registry = AssetRegistry(data_root=str(tmp_path))
    meta = AssetMetadata(asset_id="a1", product_id="p1")
    registry.register_asset(meta)
    
    # OK state
    registry.update_publish_state("a1", "draft")
    
    # Blocked state
    with pytest.raises(ValueError, match="Direct state update to 'published' is prohibited"):
        registry.update_publish_state("a1", "published")

def test_registry_publish_normal_photogrammetry(tmp_path):
    registry = AssetRegistry(data_root=str(tmp_path))
    product_id = "test_prod"
    asset_id = "test_asset_normal"
    
    meta = AssetMetadata(
        asset_id=asset_id, 
        product_id=product_id,
        ai_generated=False,
        requires_manual_review=False,
        geometry_source="photogrammetry"
    )
    registry.register_asset(meta)
    
    # Should publish normally
    registry.publish_asset(product_id, asset_id)
    history = registry.get_history(product_id)
    assert any(h["asset_id"] == asset_id and h["status"] == "published" for h in history)
    assert registry._get_active_id(product_id) == asset_id

def test_readme_ports():
    readme_path = Path("README.md")
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
        assert "localhost:8000" not in content
        assert "localhost:8001" in content

@pytest.mark.anyio
async def test_api_upload_preflight_binary_resolution(monkeypatch):
    from modules.operations.api import upload_video
    from fastapi import HTTPException
    import shutil
    
    # Mock shutil.which to fail for ffprobe
    def mock_which(cmd, path=None):
        if "ffprobe" in cmd: return None
        return "/path/to/ffmpeg"
        
    # Mock Path.exists to fail for ffprobe
    original_exists = Path.exists
    def mock_exists(self):
        if "ffprobe" in str(self): return False
        return original_exists(self)
        
    monkeypatch.setattr(shutil, "which", mock_which)
    monkeypatch.setattr(Path, "exists", mock_exists)
    
    from unittest.mock import MagicMock
    mock_file = MagicMock()
    mock_file.filename = "test.mp4"
    
    with pytest.raises(HTTPException) as exc:
        await upload_video(
            product_id="p1", 
            operator_id="test_op", 
            quality_manifest="{}", 
            file=mock_file
        )
    
    assert exc.value.status_code == 503
    assert "ffprobe binary missing" in exc.value.detail
