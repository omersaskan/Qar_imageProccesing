import pytest
import numpy as np
import trimesh
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType

def test_texture_safe_copy_with_guidance(tmp_path):
    """
    Verify that TEXTURE_SAFE_COPY correctly passes guidance data to the isolator
    and can produce mask_guided status.
    """
    cleaner = AssetCleaner(data_root=str(tmp_path))
    
    # Setup dummy files for safe copy
    raw_mesh = tmp_path / "model.obj"
    raw_mesh.write_text("v 0 0 0\nv 1 1 1\nf 1 1 1")
    raw_tex = tmp_path / "texture.jpg"
    raw_tex.write_text("dummy")
    
    # Mock data
    cameras = [{"name": "frame_0000.jpg", "P": np.eye(3, 4)}]
    masks = {"frame_0000.jpg": np.ones((100, 100), dtype=np.uint8) * 255}
    
    mock_isolation_stats = {
        "object_isolation_status": "success",
        "object_isolation_method": "mask_guided",
        "mask_support_ratio": 0.85
    }
    
    mock_isolated_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_isolated_mesh.faces = np.array([[0, 1, 2]])
    mock_isolated_mesh.vertices = np.array([[0,0,0], [1,1,1], [2,2,2]])
    
    # Helper to create the 'cleaned_mesh.obj' file so face count reading works
    def side_effect_align(work_mesh, output_path):
        Path(output_path).write_text("f 1 2 3\n") # One face
        return {"x":0,"y":0,"z":0}, {"x":0,"y":0,"z":0}, {"x":1,"y":1,"z":1}

    with patch("modules.asset_cleanup_pipeline.cleaner.trimesh.load") as mock_load, \
         patch.object(cleaner, "_is_valid_textured_obj_bundle", return_value=(True, "Valid", str(raw_tex))), \
         patch.object(cleaner, "_safe_align_obj", side_effect=side_effect_align), \
         patch.object(cleaner, "_inspect_visuals", return_value=(True, True)), \
         patch.object(cleaner.isolator, "isolate_product", return_value=(mock_isolated_mesh, mock_isolation_stats)) as mock_isolate:
        
        mock_load.return_value = mock_isolated_mesh
        
        metadata, stats, mesh_path = cleaner.process_cleanup(
            job_id="job_test_safe_copy",
            raw_mesh_path=str(raw_mesh),
            raw_texture_path=str(raw_tex),
            cameras=cameras,
            masks=masks
        )
        
        assert stats["isolation"]["object_isolation_method"] == "mask_guided"
        assert stats["cleanup_mode"] == "texture_safe_copy"
        assert stats["final_polycount"] == 1
        
        mock_isolate.assert_called_once()
        _, kwargs = mock_isolate.call_args
        assert kwargs["cameras"] == cameras
        assert kwargs["masks"] == masks

def test_texture_safe_copy_geometric_fallback_not_ready(tmp_path):
    """
    Verify that if TEXTURE_SAFE_COPY falls back to geometric_only, 
    delivery_ready becomes False.
    """
    cleaner = AssetCleaner(data_root=str(tmp_path))
    
    raw_mesh = tmp_path / "model_fallback.obj"
    raw_mesh.write_text("v 0 0 0\nv 1 1 1\nf 1 1 1")
    raw_tex = tmp_path / "texture_fallback.jpg"
    raw_tex.write_text("dummy")
    
    mock_isolation_stats = {
        "object_isolation_status": "success",
        "object_isolation_method": "geometric_only",
    }
    
    mock_isolated_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_isolated_mesh.faces = np.array([[0, 1, 2]])
    mock_isolated_mesh.vertices = np.array([[0,0,0], [1,1,1], [2,2,2]])
    
    def side_effect_align(work_mesh, output_path):
        Path(output_path).write_text("f 1 2 3\n")
        return {"x":0,"y":0,"z":0}, {"x":0,"y":0,"z":0}, {"x":1,"y":1,"z":1}

    with patch("modules.asset_cleanup_pipeline.cleaner.trimesh.load") as mock_load, \
         patch.object(cleaner, "_is_valid_textured_obj_bundle", return_value=(True, "Valid", str(raw_tex))), \
         patch.object(cleaner, "_safe_align_obj", side_effect=side_effect_align), \
         patch.object(cleaner, "_inspect_visuals", return_value=(True, True)), \
         patch.object(cleaner.isolator, "isolate_product", return_value=(mock_isolated_mesh, mock_isolation_stats)):
        
        mock_load.return_value = mock_isolated_mesh
        
        metadata, stats, mesh_path = cleaner.process_cleanup(
            job_id="job_test_safe_fallback",
            raw_mesh_path=str(raw_mesh),
            raw_texture_path=str(raw_tex),
            cameras=None,
            masks=None
        )
        
        assert stats["isolation"]["object_isolation_method"] == "geometric_only"
        assert stats["delivery_ready"] is False
