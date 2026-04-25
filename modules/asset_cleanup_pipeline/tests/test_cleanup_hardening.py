import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType

def test_cleanup_polycount_rejection(tmp_path):
    cleaner = AssetCleaner(data_root=str(tmp_path))
    
    # Mock internal methods to simulate a low polycount result
    mock_metadata = MagicMock()
    mock_stats = {
        "isolation": {"component_count": 5},
        "final_polycount": 3000, # Below 5000 threshold
        "bbox_min": {"x": 0, "y": 0, "z": 0},
        "bbox_max": {"x": 1, "y": 1, "z": 1},
        "pivot_offset": {"x": 0, "y": 0, "z": 0},
        "cleaned_mesh_path": "dummy.obj"
    }
    
    with patch("modules.asset_cleanup_pipeline.cleaner.trimesh.load") as mock_load, \
         patch.object(cleaner, "_is_valid_textured_obj_bundle", return_value=(False, "not textured", None)), \
         patch.object(cleaner, "_safe_align_obj", return_value=({}, {}, {})), \
         patch.object(cleaner, "_inspect_visuals", return_value=(True, True)), \
         patch.object(cleaner.isolator, "isolate_product", return_value=(MagicMock(), mock_stats["isolation"])), \
         patch.object(cleaner.remesher, "process", return_value=3000), \
         patch.object(cleaner.alignment, "align_to_ground", return_value=(None, {})), \
         patch.object(cleaner.bbox_extractor, "extract", return_value=({}, {})), \
         patch.object(cleaner.normalizer, "generate_metadata", return_value=mock_metadata), \
         patch.object(cleaner.normalizer, "save_metadata"), \
         patch("os.path.exists", return_value=True):
        
        # Setup mock mesh
        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[1, 2, 3]])
        mock_mesh.faces = np.array([[0, 0, 0]])
        mock_load.return_value = mock_mesh

        # Setup isolated mesh mock
        mock_isolated = MagicMock()
        mock_isolated.vertices = np.array([[1, 2, 3]])
        mock_isolated.faces = np.array([[0, 0, 0]])
        
        # Create dummy file to satisfy existence check
        dummy_mesh = tmp_path / "raw.obj"
        dummy_mesh.write_text("v 0 0 0\nf 1 2 3")

        # Create output files to satisfy existence checks
        job_dir = tmp_path / "cleaned" / "job1"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "cleaned_mesh.obj").write_text("dummy")
        (job_dir / "normalized_metadata.json").write_text("{}")
        (job_dir / "pre_aligned_mesh.obj").write_text("dummy")
        
        with patch.object(cleaner.isolator, "isolate_product", return_value=(mock_isolated, mock_stats["isolation"])):
            _, cleanup_stats, _ = cleaner.process_cleanup("job1", str(dummy_mesh))
        
        assert cleanup_stats["quality_status"] == "quality_fail"
        assert cleanup_stats["recapture_recommended"] is True
        assert "below the asset-quality threshold" in cleanup_stats["quality_reason"]

def test_cleanup_island_warning(tmp_path):
    cleaner = AssetCleaner(data_root=str(tmp_path))
    
    mock_stats = {
        "isolation": {"component_count": 250}, # Above 200 threshold
        "final_polycount": 10000
    }
    
    with patch("modules.asset_cleanup_pipeline.cleaner.trimesh.load") as mock_load, \
         patch.object(cleaner, "_is_valid_textured_obj_bundle", return_value=(False, "not textured", None)), \
         patch.object(cleaner.remesher, "process", return_value=10000), \
         patch.object(cleaner.alignment, "align_to_ground", return_value=(None, {})), \
         patch.object(cleaner.bbox_extractor, "extract", return_value=({}, {})), \
         patch.object(cleaner.normalizer, "generate_metadata"), \
         patch.object(cleaner.normalizer, "save_metadata"), \
         patch("os.path.exists", return_value=True):
        
        # Setup mock mesh
        mock_mesh = MagicMock()
        mock_mesh.vertices = np.array([[1, 2, 3]])
        mock_mesh.faces = np.array([[0, 0, 0]])
        mock_load.return_value = mock_mesh

        # Setup isolated mesh mock
        mock_isolated = MagicMock()
        mock_isolated.vertices = np.array([[1, 2, 3]])
        mock_isolated.faces = np.array([[0, 0, 0]])
        
        dummy_mesh = tmp_path / "raw.obj"
        dummy_mesh.write_text("v 0 0 0\nf 1 2 3")

        # Create output files to satisfy existence checks
        job_dir = tmp_path / "cleaned" / "job2"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "cleaned_mesh.obj").write_text("dummy")
        (job_dir / "normalized_metadata.json").write_text("{}")
        (job_dir / "pre_aligned_mesh.obj").write_text("dummy")
        
        with patch.object(cleaner.isolator, "isolate_product", return_value=(mock_isolated, mock_stats["isolation"])):
            _, cleanup_stats, _ = cleaner.process_cleanup("job2", str(dummy_mesh))
        
        assert cleanup_stats["quality_status"] == "warning"
        assert "Large number of small islands" in cleanup_stats["quality_reason"]
