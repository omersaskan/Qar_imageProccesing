import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import trimesh

from modules.capture_workflow.object_masker import ObjectMasker
from modules.capture_workflow.quality_analyzer import QualityAnalyzer
from modules.asset_cleanup_pipeline.isolation import MeshIsolator
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType


def write_dummy_png(path: Path):
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff? \x00\x05\xfe\x02\xfe\xdcD\x05\x13"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def test_object_masking():
    print("\n--- Testing Object Masker ---")
    masker = ObjectMasker()

    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.circle(frame, (250, 250), 90, (220, 220, 220), -1)

    mask, meta = masker.generate_mask(frame)

    print(
        f"Mask generated. occupancy={meta['occupancy']:.2%}, "
        f"confidence={meta['confidence']:.2f}, "
        f"fragments={meta.get('fragment_count')}, "
        f"largest_ratio={meta.get('largest_contour_ratio'):.2f}"
    )

    assert meta["occupancy"] > 0.05
    assert meta["confidence"] > 0.4
    assert meta["fragment_count"] >= 1
    assert meta["largest_contour_ratio"] > 0.7
    print("[SUCCESS] ObjectMasker passed")


def test_quality_analysis():
    print("\n--- Testing Quality Analyzer ---")
    analyzer = QualityAnalyzer()
    masker = ObjectMasker()

    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(frame, (120, 100), (280, 320), (200, 200, 200), -1)

    mask, meta = masker.generate_mask(frame)
    report = analyzer.analyze_frame(frame, mask, meta)

    print(f"Quality report: {report}")
    assert "ground_offset" not in report
    assert report["occupancy"] > 0.05
    assert report["overall_pass"] is True
    print("[SUCCESS] QualityAnalyzer passed")


def test_mesh_isolation():
    print("\n--- Testing Mesh Isolator ---")
    isolator = MeshIsolator()

    sphere = trimesh.creation.icosphere(radius=1.0)
    sphere.apply_translation([0, 0, 1.5])

    # Create a more complex plane to satisfy heuristic thresholds
    plane = trimesh.creation.box(extents=[10, 10, 0.1])
    plane = plane.subdivide()
    plane = plane.subdivide()
    plane = plane.subdivide()
    plane.apply_translation([0, 0, 0])

    scene_mesh = sphere + plane
    isolated_mesh, stats = isolator.isolate_product(scene_mesh)

    print(f"Isolation stats: {stats}")
    assert stats["removed_planes"] >= 1
    assert stats["component_count"] >= 1
    assert len(isolated_mesh.faces) > 0
    assert stats["removed_plane_face_share"] >= 0.0
    assert stats["removed_plane_vertex_ratio"] >= 0.0
    print("[SUCCESS] MeshIsolator passed")


def test_cleanup_orchestration():
    print("\n--- Testing Cleanup Orchestration ---")

    temp_root = Path("temp_test_data")
    temp_root.mkdir(parents=True, exist_ok=True)

    temp_raw = temp_root / "test_raw.obj"
    temp_tex = temp_root / "test_raw.png"

    mesh = trimesh.creation.icosphere(radius=1.0)
    mesh.export(temp_raw)
    write_dummy_png(temp_tex)

    cleaner = AssetCleaner(data_root=str(temp_root))

    try:
        metadata, stats, cleaned_mesh_path = cleaner.process_cleanup(
            "test_job",
            str(temp_raw),
            CleanupProfileType.MOBILE_DEFAULT,
            raw_texture_path=str(temp_tex),
        )

        print(f"Cleanup metadata: {metadata}")
        print(f"Cleanup stats: {stats}")
        print(f"Cleaned mesh path: {cleaned_mesh_path}")

        assert stats["final_polycount"] > 0
        assert metadata.final_polycount == stats["final_polycount"]
        assert Path(cleaned_mesh_path).exists()
        assert Path(stats["metadata_path"]).exists()
        assert "uv_preserved" in stats
        assert "material_preserved" in stats
        assert stats["cleaned_texture_path"] == str(temp_tex)

        print("[SUCCESS] AssetCleaner passed")
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)


if __name__ == "__main__":
    try:
        test_object_masking()
        test_quality_analysis()
        test_mesh_isolation()
        test_cleanup_orchestration()
        print("\nPASSED ALL SMOKE TESTS!")
    except Exception as e:
        print(f"\n[FAILED] SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()