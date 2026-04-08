import os
import cv2
import numpy as np
import trimesh
from pathlib import Path
from modules.capture_workflow.object_masker import ObjectMasker
from modules.capture_workflow.quality_analyzer import QualityAnalyzer
from modules.asset_cleanup_pipeline.isolation import MeshIsolator
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType

def test_object_masking():
    print("\n--- Testing Object Masker ---")
    masker = ObjectMasker()
    # Create a dummy frame (white circle on black background)
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.circle(frame, (250, 250), 100, (200, 200, 200), -1)
    
    mask, meta = masker.generate_mask(frame)
    print(f"Mask generated. Occupancy: {meta['occupancy']:.2%}, Confidence: {meta['confidence']}")
    assert meta['occupancy'] > 0.05
    assert meta['confidence'] > 0.5
    print("[SUCCESS] Object Masker Passed")

def test_mesh_isolation():
    print("\n--- Testing Mesh Isolator ---")
    isolator = MeshIsolator()
    
    # Create a dummy mesh with an 'object' (sphere) and a 'plane' (box)
    sphere = trimesh.creation.icosphere(radius=1.0)
    sphere.apply_translation([0, 0, 1.5]) # Floating
    
    plane = trimesh.creation.box(extents=[10, 10, 0.1])
    plane.apply_translation([0, 0, 0]) # Floor
    
    scene_mesh = sphere + plane
    
    isolated_mesh, stats = isolator.isolate_product(scene_mesh)
    print(f"Isolation Stats: {stats}")
    
    # Check if plane was removed
    # Isolated mesh should have faces belonging mostly to the sphere
    assert stats['removed_planes'] >= 1
    assert len(isolated_mesh.faces) < len(scene_mesh.faces)
    print("[SUCCESS] Mesh Isolator Passed")

def test_cleanup_orchestration():
    print("\n--- Testing Cleanup Orchestration ---")
    # Need a temp file
    temp_raw = "test_raw.obj"
    sphere = trimesh.creation.icosphere(radius=1.0)
    sphere.export(temp_raw)
    
    cleaner = AssetCleaner(data_root="temp_test_data")
    try:
        metadata, stats = cleaner.process_cleanup("test_job", temp_raw, CleanupProfileType.MOBILE_DEFAULT)
        print(f"Cleanup Metadata: {metadata}")
        print(f"Final Polycount: {stats['final_polycount']}")
        assert stats['final_polycount'] > 0
        assert metadata.final_polycount == stats['final_polycount']
        print("[SUCCESS] Cleanup Orchestration Passed")
    finally:
        if os.path.exists(temp_raw): os.remove(temp_raw)
        import shutil
        if os.path.exists("temp_test_data"): shutil.rmtree("temp_test_data")

if __name__ == "__main__":
    try:
        test_object_masking()
        test_mesh_isolation()
        test_cleanup_orchestration()
        print("\nPASSED ALL SMOKE TESTS!")
    except Exception as e:
        print(f"\n[FAILED] SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
