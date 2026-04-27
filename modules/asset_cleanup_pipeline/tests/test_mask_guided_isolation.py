import pytest
import numpy as np
import trimesh
from pathlib import Path
import json
from modules.asset_cleanup_pipeline.isolation import MeshIsolator

def test_mask_guided_scoring_logic():
    isolator = MeshIsolator()
    
    # Create a dummy component
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    
    # Case 1: Geometric only
    scores_geom = isolator._score_component(mesh)
    assert "total_score" in scores_geom
    assert scores_geom["mask_score"] == 0.0
    
    # Case 2: Mask support (High)
    mask_support_high = {
        "avg_support": 0.9,
        "hit_ratio": 0.95,
        "view_count": 10,
        "supported_view_count": 9
    }
    scores_mask_high = isolator._score_component(mesh, mask_support=mask_support_high)
    assert scores_mask_high["mask_score"] > 0.8
    assert scores_mask_high["total_score"] > scores_geom["total_score"]
    
    # Case 3: Mask support (Low/Penalty)
    mask_support_low = {
        "avg_support": 0.1,
        "hit_ratio": 0.1,
        "view_count": 20,
        "supported_view_count": 1
    }
    scores_mask_low = isolator._score_component(mesh, mask_support=mask_support_low)
    # Should be heavily penalized
    assert scores_mask_low["total_score"] < scores_geom["total_score"] * 0.6

def test_isolate_product_with_guidance(tmp_path):
    isolator = MeshIsolator()
    
    # Create a mesh with two components: one "product" and one "table"
    product = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    product.apply_translation((0, 0, 0.5)) # Above ground
    
    table = trimesh.creation.box(extents=(2, 2, 0.1))
    table.apply_translation((0, 0, 0)) # At ground
    
    combined = trimesh.util.concatenate([product, table])
    
    # Mock cameras and masks that support only the product
    cameras = [
        {"P": np.eye(3, 4), "name": "cam1", "width": 100, "height": 100}
    ]
    # Mask that covers the center where product is
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    masks = {"cam1": mask}
    
    # Run isolation
    final_mesh, stats = isolator.isolate_product(
        combined, 
        cameras=cameras, 
        masks=masks,
        output_dir=tmp_path
    )
    
    assert stats["object_isolation_method"] == "mask_guided"
    assert (tmp_path / "component_scores.json").exists()
    assert (tmp_path / "mask_projection_report.json").exists()
    
    # The product should have higher score than table due to mask support
    with open(tmp_path / "component_scores.json", "r") as f:
        scores = json.load(f)
        # We expect at least 2 components
        assert len(scores) >= 2

def test_point_cloud_support():
    isolator = MeshIsolator()
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    
    # Point cloud that matches the mesh
    pc = trimesh.points.PointCloud(vertices=mesh.vertices)
    
    pc_supports = isolator.isolate_by_point_cloud(mesh, pc)
    assert 0 in pc_supports
    assert pc_supports[0]["support_ratio"] > 0.99
    
    # Point cloud that is far away
    pc_far = trimesh.points.PointCloud(vertices=mesh.vertices + 10.0)
    pc_supports_far = isolator.isolate_by_point_cloud(mesh, pc_far)
    assert pc_supports_far[0]["support_ratio"] < 0.01
