import os
import json
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch
from typing import Dict, Any

import cv2
import numpy as np
import trimesh
from PIL import Image
from fastapi.testclient import TestClient

from modules.operations.api import app
from modules.operations.settings import settings
from modules.operations.worker import worker_instance
from modules.shared_contracts.models import AssetStatus
from modules.reconstruction_engine.output_manifest import OutputManifest

client = TestClient(app)

def status_value(status):
    return status.value if hasattr(status, "value") else status

def create_valid_dummy_video(path: Path):
    fps = 20.0
    duration = 8.5
    frames = int(fps * duration)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, (720, 720))
    
    frame = np.zeros((720, 720, 3), dtype=np.uint8)
    cv2.circle(frame, (360, 360), 100, (255, 255, 255), -1)
    
    for _ in range(frames):
        out.write(frame)
        
    out.release()


@patch("modules.operations.api.verify_api_key")
def test_pilot_smoke_pipeline_shell(mock_auth):
    """
    Purpose: Workflow progression shell test.
    Uses local workspace path to avoid Windows encoding issues in temp paths.
    """
    test_root = Path("test_data_shell").absolute()
    if test_root.exists():
        shutil.rmtree(test_root)
    test_root.mkdir(parents=True, exist_ok=True)

    orig_data_root = settings.data_root
    settings.data_root = str(test_root / "data")
    Path(settings.data_root).mkdir(parents=True, exist_ok=True)
    worker_instance.data_root = Path(settings.data_root)
    worker_instance.registry.registry_file = Path(settings.data_root) / "registry" / "catalog.json"
    worker_instance.session_manager.data_root = Path(settings.data_root)
    worker_instance.session_manager.sessions_dir = Path(settings.data_root) / "sessions"
    worker_instance.session_manager.captures_dir = Path(settings.data_root) / "captures"
    worker_instance.session_manager.sessions_dir.mkdir(parents=True, exist_ok=True)
    worker_instance.session_manager.captures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from modules.operations.api import session_manager as api_session_manager
        api_session_manager.data_root = Path(settings.data_root)
        api_session_manager.sessions_dir = Path(settings.data_root) / "sessions"
        api_session_manager.captures_dir = Path(settings.data_root) / "captures"
        api_session_manager.sessions_dir.mkdir(parents=True, exist_ok=True)
        api_session_manager.captures_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Upload returns 200
        valid_vid = test_root / "valid.mp4"
        create_valid_dummy_video(valid_vid)
        
        with open(valid_vid, "rb") as f:
            resp = client.post("/api/sessions/upload", data={"product_id": "prod_pilot", "operator_id": "op_pilot"}, files={"file": ("v.mp4", f, "video/mp4")})
            assert resp.status_code == 200, f"Upload failed: {resp.text}"
            session_id = resp.json()["session_id"]
            
        # 2. Session created
        sess = worker_instance.session_manager.get_session(session_id)
        assert sess is not None
        assert status_value(sess.status) == AssetStatus.CREATED.value
        
        # Mocks
        def mock_recon(*args, **kwargs):
            mesh = trimesh.creation.box()
            mesh_dir = Path(settings.data_root) / "reconstructions" / f"job_{session_id}" / "final"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = mesh_dir / "mesh.ply"
            mesh.export(str(mesh_path))
            
            img = Image.new("RGBA", (10, 10), color="blue")
            tex_path = mesh_dir / "texture.png"
            img.save(tex_path)
            
            manifest = OutputManifest(
                job_id=session_id,
                status="success",
                mesh_path=str(mesh_path),
                texture_path=str(tex_path),
                log_path=str(mesh_dir / "recon.log"),
                is_stub=False,
                processing_time_seconds=10.0
            )
            
            job = args[0] if args else None
            if job and hasattr(job, 'job_dir'):
                manifest_path = Path(job.job_dir) / "manifest.json"
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest.model_dump(mode="json"), f)
            return manifest
            
        def mock_texture(*args, **kwargs):
            from modules.operations.texturing_service import TexturingResult
            manifest = kwargs.get("manifest")
            if not manifest:
                for arg in args:
                    if isinstance(arg, OutputManifest):
                        manifest = arg
                        break

            return TexturingResult(
                texturing_status="mocked",
                cleaned_mesh_path=str(Path(settings.data_root) / "reconstructions" / f"job_{session_id}" / "final" / "mesh.ply"),
                texture_atlas_paths=[],
                manifest=manifest
            )

        def mock_extract(video_path, output_dir, *args, **kwargs):
            frames_dir = Path(output_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for i in range(16):
                frame = np.zeros((720, 720, 3), dtype=np.uint8)
                cv2.circle(frame, (360, 360), 120, (255, 255, 255), -1)
                frame_path = frames_dir / f"frame_{i:04d}.jpg"
                ok = cv2.imwrite(str(frame_path), frame)
                if not ok:
                    # Fallback to absolute local string if Path object fails
                    ok = cv2.imwrite(os.path.abspath(str(frame_path)), frame)
                assert ok, f"Failed to write frame at {frame_path}"
                assert frame_path.exists()
                paths.append(str(frame_path))
            
            print(f"MOCK EXTRACT RETURNING {len(paths)} FRAMES to {output_dir}")
            return paths, {"fps": 30.0, "duration_sec": 8.5, "frame_count": len(paths)}

        def mock_coverage(*args, **kwargs):
            return {
                "overall_status": "sufficient",
                "coverage_score": 0.95,
                "reasons": []
            }

        def mock_validate(asset_id, asset_data):
            from modules.shared_contracts.models import ValidationReport
            
            return ValidationReport(
                asset_id=asset_id,
                poly_count=asset_data.get("poly_count", 0),
                texture_status="complete",
                bbox_reasonable=True,
                ground_aligned=True,
                component_count=1,
                largest_component_share=1.0,
                contamination_score=0.0,
                contamination_report={"semantic_guard": "pass"},
                mobile_performance_grade="A",
                material_quality_grade="B",
                material_semantic_status="diffuse_textured",
                final_decision="pass",
            )

        with patch("modules.reconstruction_engine.runner.ReconstructionRunner.run", side_effect=mock_recon):
            with patch("modules.operations.texturing_service.TexturingService.run", side_effect=mock_texture):
                with patch("modules.capture_workflow.frame_extractor.FrameExtractor.extract_keyframes", side_effect=mock_extract):
                    with patch("modules.capture_workflow.coverage_analyzer.CoverageAnalyzer.analyze_coverage", side_effect=mock_coverage):
                        with patch("modules.qa_validation.validator.AssetValidator.validate", side_effect=mock_validate):
                            for _ in range(12):
                                worker_instance._process_pending_sessions()

        sess = worker_instance.session_manager.get_session(session_id)
        assert status_value(sess.status) == AssetStatus.PUBLISHED.value
        
        reports_dir = Path(settings.data_root) / "captures" / session_id / "reports"
        assert (reports_dir / "export_metrics.json").exists()
        assert (reports_dir / "validation_report.json").exists()
        
        manifest1 = Path(settings.data_root) / "training_manifests" / f"{session_id}.json"
        manifest2 = reports_dir / "training_manifest.json"
        assert manifest1.exists()
        assert manifest2.exists()
        
        resp = client.get("/api/training/manifests")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert any(d["session_id"] == session_id for d in data)
        
    finally:
        settings.data_root = orig_data_root
        if test_root.exists():
            shutil.rmtree(test_root)


def test_pilot_smoke_glb_quality():
    """
    Purpose: Real asset quality validation.
    Uses local workspace path to avoid Windows encoding issues in temp paths.
    """
    test_root = Path("test_data_quality").absolute()
    if test_root.exists():
        shutil.rmtree(test_root)
    test_root.mkdir(parents=True, exist_ok=True)

    orig_data_root = settings.data_root
    settings.data_root = str(test_root / "data")
    Path(settings.data_root).mkdir(parents=True, exist_ok=True)
    
    try:
        from modules.export_pipeline.glb_exporter import GLBExporter
        from modules.qa_validation.validator import AssetValidator

        vertices = np.array([
            [-1.0, -1.0, 0.0],
            [ 1.0, -1.0, 0.0],
            [ 1.0,  1.0, 0.0],
            [-1.0,  1.0, 0.0],
            [-1.0, -1.0, 0.01],
            [ 1.0, -1.0, 0.01],
            [ 1.0,  1.0, 0.01],
            [-1.0,  1.0, 0.01],
        ], dtype=float)

        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ], dtype=int)

        uv = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        ], dtype=float)

        texture_image = Image.new("RGBA", (256, 256), color="red")

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv,
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture_image,
                metallicFactor=0.0,
                roughnessFactor=1.0,
            ),
        )
        
        input_dir = test_root / "recon_input"
        input_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = input_dir / "mesh.obj"
        mesh.export(str(mesh_path))
        
        exporter = GLBExporter()
        glb_output = test_root / "data" / "test.glb"
        glb_output.parent.mkdir(parents=True, exist_ok=True)
        
        exporter.export_to_glb(
            mesh_path=str(mesh_path),
            texture_path=None,
            output_path=str(glb_output)
        )
        
        assert glb_output.exists()
        inspection = exporter.inspect_exported_asset(str(glb_output))
        
        assert inspection["has_uv"] is True
        assert inspection["has_embedded_texture"] is True
        assert inspection["material_semantic_status"] in {"diffuse_textured", "pbr_partial", "pbr_complete"}
        
        validator = AssetValidator()
        validation_input = {
            "poly_count": int(inspection["face_count"]),
            "texture_status": inspection["texture_integrity_status"],
            "bbox": inspection["bbox"],
            "ground_offset": float(inspection["ground_offset"]),
            "cleanup_stats": {
                "isolation": {
                    "component_count": 1,
                    "final_faces": inspection["face_count"],
                    "initial_faces": inspection["face_count"],
                    "compactness_score": 1.0,
                    "selected_component_score": 1.0,
                }
            },
            "texture_path_exists": True,
            "has_uv": inspection["has_uv"],
            "has_material": inspection["has_material"],
            "has_embedded_texture": inspection["has_embedded_texture"],
            "texture_count": inspection.get("texture_count", 0),
            "material_count": inspection.get("material_count", 0),
            "texture_integrity_status": inspection["texture_integrity_status"],
            "material_integrity_status": inspection.get("material_integrity_status", "present"),
            "material_semantic_status": inspection["material_semantic_status"],
            "basecolor_present": inspection.get("basecolor_present", False),
            "normal_present": inspection.get("normal_present", False),
            "metallic_roughness_present": inspection.get("metallic_roughness_present", False),
            "delivery_geometry_count": inspection.get("geometry_count", 1),
            "delivery_component_count": inspection.get("component_count", 1),
        }

        report = validator.validate("test_asset", validation_input)
        assert report.final_decision in {"pass", "review"}

    finally:
        settings.data_root = orig_data_root
        if test_root.exists():
            shutil.rmtree(test_root)

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
