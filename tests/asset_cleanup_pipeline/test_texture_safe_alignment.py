import os
import pytest
from pathlib import Path
import numpy as np
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType
from modules.export_pipeline.glb_exporter import GLBExporter

@pytest.fixture
def temp_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "data").mkdir()
    return workspace

def test_safe_align_obj_math(temp_workspace):
    src_dir = temp_workspace / "source"
    src_dir.mkdir()
    
    mesh_path = src_dir / "input.obj"
    # Create a mesh with min_z = 3 and centroid X=1, Y=1
    with open(mesh_path, "w") as f:
        f.write("mtllib material.mtl\n")
        f.write("v 0 0 3\n")
        f.write("v 2 0 3\n")
        f.write("v 1 2 3\n")
        f.write("v 1 1 5\n")
        f.write("vt 0 0\n")
        f.write("f 1/1 2/1 3/1\n")
    
    cleaner = AssetCleaner(data_root=str(temp_workspace / "data"))
    output_path = temp_workspace / "aligned.obj"
    
    pivot, bmin, bmax = cleaner._safe_align_obj(mesh_path, output_path)
    
    # centroid: (0+2+1+1)/4 = 1.0, (0+0+2+1)/4 = 0.75, (3+3+3+5)/4 = 3.5
    # min_z: 3.0
    # Expected shift: x=-1.0, y=-0.75, z=-3.0
    assert pivot["x"] == pytest.approx(-1.0)
    assert pivot["y"] == pytest.approx(-0.75)
    assert pivot["z"] == pytest.approx(-3.0)
    
    # Check output OBJ content
    with open(output_path, "r") as f:
        lines = f.readlines()
        v_lines = [l for l in lines if l.startswith("v ")]
        assert len(v_lines) == 4
        # First vertex (0,0,3) -> (-1, -0.75, 0)
        v1 = [float(x) for x in v_lines[0].split()[1:]]
        assert v1[0] == pytest.approx(-1.0)
        assert v1[1] == pytest.approx(-0.75)
        assert v1[2] == pytest.approx(0.0)
        
        # vt and f lines should be preserved
        assert any(l.startswith("vt ") for l in lines)
        assert any(l.startswith("f ") for l in lines)
        assert any(l.startswith("mtllib ") for l in lines)

def test_texture_safe_copy_alignment_integration(temp_workspace):
    src_dir = temp_workspace / "source"
    src_dir.mkdir()
    
    mesh_path = src_dir / "input.obj"
    mtl_path = src_dir / "material.mtl"
    tex_path = src_dir / "texture.png"
    
    with open(mesh_path, "w") as f:
        f.write("mtllib material.mtl\n")
        f.write("usemtl old_mat\n")
        f.write("v 10 10 10\n")
        f.write("v 11 10 10\n")
        f.write("v 10 11 10\n")
        f.write("vt 0 0\n")
        f.write("vt 1 0\n")
        f.write("vt 0 1\n")
        f.write("f 1/1 2/2 3/3\n")
        
    with open(mtl_path, "w") as f:
        f.write("newmtl old_mat\n")
        f.write("map_Kd texture.png\n")
        
    # Need a real small image for GLB export to work in some environments
    from PIL import Image
    img = Image.new('RGB', (16, 16), color = 'red')
    img.save(tex_path)
    
    cleaner = AssetCleaner(data_root=str(temp_workspace / "data"))
    
    metadata, stats, cleaned_path = cleaner.process_cleanup(
        job_id="job_align",
        raw_mesh_path=str(mesh_path),
        profile_type=CleanupProfileType.TEXTURE_SAFE_COPY,
        raw_texture_path=str(tex_path)
    )
    
    assert stats["pivot_offset"]["z"] == pytest.approx(-10.0)
    assert metadata.pivot_offset["z"] == pytest.approx(-10.0)
    assert metadata.bbox_min["z"] == 0.0
    
    # Export to GLB and verify texture
    exporter = GLBExporter()
    export_dir = temp_workspace / "export"
    export_dir.mkdir()
    glb_path = export_dir / "final.glb"
    
    # We need to manually construct the export call similar to how the worker does it
    metrics = exporter.export_to_glb(
        mesh_path=stats["cleaned_mesh_path"],
        output_path=str(glb_path),
        texture_path=stats["cleaned_texture_path"],
        metadata=metadata
    )
    
    assert metrics["has_uv"] is True
    assert metrics["has_embedded_texture"] is True
    assert metrics["material_semantic_status"] == "diffuse_textured"

def test_validation_guidance_classification():
    from modules.operations.guidance import GuidanceAggregator
    from modules.shared_contracts.models import AssetStatus
    
    agg = GuidanceAggregator()
    
    # Case: Validation failed normally
    report = {
        "final_decision": "fail",
        "ground_aligned": False,
        "mobile_performance_grade": "D",
        "poly_count": 900000,
        "contamination_score": 0.6
    }
    
    guidance = agg.generate_guidance(
        session_id="test_fail",
        status=AssetStatus.FAILED,
        failure_reason="Validation Failed",
        validation_report=report
    )
    
    codes = [m["code"] for m in guidance.messages]
    assert "ASSET_VALIDATION_FAILED" in codes
    assert "GROUND_ALIGNMENT_FAILED" in codes
    assert "MOBILE_GRADE_LOW" in codes
    assert "CONTAMINATION_HIGH" in codes
    assert "SYSTEM_FAILURE_PIPELINE" not in codes
