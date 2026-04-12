import pytest
from pathlib import Path
from modules.utils.mask_resolution import resolve_mask_path, resolve_meta_path

def test_resolve_mask_path_stem(tmp_path):
    frame_path = tmp_path / "frame_0001.jpg"
    frame_path.touch()
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    
    # Preferred: stem-based
    mask_path = masks_dir / "frame_0001.png"
    mask_path.touch()
    
    resolved, mode = resolve_mask_path(frame_path)
    assert mode == "stem"
    assert resolved == mask_path

def test_resolve_mask_path_legacy(tmp_path):
    frame_path = tmp_path / "frame_0001.jpg"
    frame_path.touch()
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    
    # Fallback: legacy double-extension
    mask_path = masks_dir / "frame_0001.jpg.png"
    mask_path.touch()
    
    resolved, mode = resolve_mask_path(frame_path)
    assert mode == "legacy"
    assert resolved == mask_path

def test_resolve_mask_path_none(tmp_path):
    frame_path = tmp_path / "frame_0001.jpg"
    frame_path.touch()
    
    resolved, mode = resolve_mask_path(frame_path)
    assert mode == "none"
    assert resolved is None

def test_resolve_meta_path_priority(tmp_path):
    frame_path = tmp_path / "frame_0001.jpg"
    frame_path.touch()
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    
    # Create both
    stem_path = masks_dir / "frame_0001.json"
    legacy_path = masks_dir / "frame_0001.jpg.json"
    stem_path.touch()
    legacy_path.touch()
    
    # Should pick stem
    resolved, mode = resolve_meta_path(frame_path)
    assert mode == "stem"
    assert resolved == stem_path
