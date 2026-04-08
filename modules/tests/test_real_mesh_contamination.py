import pytest
import trimesh
import numpy as np
from modules.asset_cleanup_pipeline.isolation import MeshIsolator
from modules.qa_validation.validator import AssetValidator

def create_product_like_mesh():
    """Creates a sphere-like product mesh."""
    return trimesh.creation.icosphere(subdivisions=3, radius=0.5)

def create_plane_mesh():
    """Creates a large flat plane (table)."""
    plane = trimesh.creation.box(extents=[10, 10, 0.01])
    # Subdivide heavily to increase density
    for _ in range(6):
        plane = plane.subdivide()
    return plane

def test_product_vs_large_plane():
    isolator = MeshIsolator()
    
    product = create_product_like_mesh()
    plane = create_plane_mesh()
    plane.vertices += [0, 0, -0.6] # Put product on plane
    
    scene_mesh = trimesh.util.concatenate([product, plane])
    
    isolated_mesh, stats = isolator.isolate_product(scene_mesh)
    
    # Assertions
    assert stats["removed_planes"] >= 1
    assert stats["final_faces"] < stats["initial_faces"]
    # Should have selected the icosphere (product)
    assert abs(len(isolated_mesh.faces) - len(product.faces)) < 10
    assert stats["flatness_score"] > 0.1 # Sphere is not flat

def test_floating_islands_removal():
    isolator = MeshIsolator()
    
    product = create_product_like_mesh()
    island = trimesh.creation.icosphere(subdivisions=1, radius=0.05)
    island.vertices += [2, 2, 2] # Far away
    
    scene_mesh = trimesh.util.concatenate([product, island])
    
    isolated_mesh, stats = isolator.isolate_product(scene_mesh)
    
    assert stats["island_count"] >= 1
    assert abs(len(isolated_mesh.faces) - len(product.faces)) < 10

def test_flatness_validation():
    validator = AssetValidator()
    
    # Case 1: Thin slab (should be reviewed/failed)
    slab = create_plane_mesh()
    stats = {
        "isolation": {
            "flatness_score": 0.001,
            "compactness_score": 0.9,
            "final_faces": 12,
            "initial_faces": 12,
            "component_count": 1
        }
    }
    
    input_data = {
        "poly_count": 12,
        "texture_status": "missing",
        "bbox": {"x": 10, "y": 10, "z": 0.01},
        "ground_offset": 0.0,
        "cleanup_stats": stats,
        "has_uv": False,
        "has_texture": False
    }
    
    report = validator.validate("test_slab", input_data)
    assert report.contamination_report["flatness"] == "review"

if __name__ == "__main__":
    pytest.main([__file__])
