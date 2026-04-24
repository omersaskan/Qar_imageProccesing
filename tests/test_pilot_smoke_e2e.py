import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch

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
def test_pilot_smoke_pipeline_shell(mock_auth, tmp_path):
    """
    Purpose: Workflow progression shell test.
    upload -> session created -> worker progresses -> reports generated -> training manifest generated.
    Uses mocks for heavy processing to ensure the shell state machine is correct.
    """
    orig_data_root = settings.data_root
    settings.data_root = str(tmp_path / "data")
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
        valid_vid = tmp_path / "valid.mp4"
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
            
            # The job object is usually the first arg to runner.run
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

        def mock_validate(*args, **kwargs):
            from modules.capture_workflow.quality_analyzer import ValidationReport
            return ValidationReport(
                status="pass",
                issues=[],
                metrics={
                    "semantic_guard": "pass",
                    "component_share": "pass",
                    "component_count": "pass",
                    "plane_contamination": "pass",
                    "compactness": "pass",
                    "selection_quality": "pass",
                    "texture_uv_integrity": "pass",
                    "texture_application": "pass",
                    "material_integrity": "pass",
                    "material_semantics": "pass",
                    "delivery_geometry": "pass",
                    "delivery_fragmentation": "pass",
                }
            )

        with patch("modules.reconstruction_engine.runner.ReconstructionRunner.run", side_effect=mock_recon):
            with patch("modules.operations.texturing_service.TexturingService.run", side_effect=mock_texture):
                with patch("modules.capture_workflow.frame_extractor.FrameExtractor.extract_keyframes", side_effect=mock_extract):
                    with patch("modules.capture_workflow.coverage_analyzer.CoverageAnalyzer.analyze_coverage", side_effect=mock_coverage):
                        with patch("modules.qa_validation.validator.AssetValidator.validate", side_effect=mock_validate):
                            # 3. Worker progresses
                            # Need enough cycles to reach PUBLISHED
                            # (CREATED -> EXTRACTING -> RECONSTRUCTING -> TEXTURING -> CLEANING -> VALIDATING -> PUBLISHED)
                            for _ in range(8):
                                worker_instance._process_pending_sessions()

        sess = worker_instance.session_manager.get_session(session_id)
        assert status_value(sess.status) == AssetStatus.PUBLISHED.value
        
        # 4. export_metrics.json & validation_report.json exist
        reports_dir = Path(settings.data_root) / "captures" / session_id / "reports"
        assert (reports_dir / "export_metrics.json").exists()
        assert (reports_dir / "validation_report.json").exists()
        
        # 5. Training manifest exists in both locations
        manifest1 = Path(settings.data_root) / "training_manifests" / f"{session_id}.json"
        manifest2 = reports_dir / "training_manifest.json"
        assert manifest1.exists()
        assert manifest2.exists()
        
        # 6. /api/training/manifests returns the session
        resp = client.get("/api/training/manifests")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert any(d["session_id"] == session_id for d in data)
        
    finally:
        settings.data_root = orig_data_root


def test_pilot_smoke_glb_quality(tmp_path):
    """
    Purpose: Real asset quality validation.
    Textured fixture -> GLB export -> inspection -> validation pass.
    No patching of internal GLB texture/UV logic.
    """
    orig_data_root = settings.data_root
    settings.data_root = str(tmp_path / "data")
    Path(settings.data_root).mkdir(parents=True, exist_ok=True)
    
    try:
        from modules.export_pipeline.glb_exporter import GLBExporter
        from modules.qa_validation.validator import AssetValidator
        from modules.reconstruction_engine.output_manifest import OutputManifest

        # 1. Create a real textured mesh manually as fixture
        mesh = trimesh.creation.box()
        # Add real UVs and a red texture
        uvs = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1],  # front
            [0, 0], [1, 0], [1, 1], [0, 1],  # back
            [0, 0], [1, 0], [1, 1], [0, 1],  # left
            [0, 0], [1, 0], [1, 1], [0, 1],  # right
            [0, 0], [1, 0], [1, 1], [0, 1],  # top
            [0, 0], [1, 0], [1, 1], [0, 1],  # bottom
        ])
        # Box has 24 vertices usually in trimesh creation
        if len(mesh.vertices) != 24:
            mesh = mesh.subdivide() # ensure enough verts or just use a simpler mesh
            uvs = np.random.rand(len(mesh.vertices), 2)

        tex_img = Image.new("RGBA", (256, 256), color="red")
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=uvs[:len(mesh.vertices)],
            image=tex_img
        )
        
        input_dir = tmp_path / "recon_input"
        input_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = input_dir / "mesh.obj"
        # Exporting as OBJ with texture
        mesh.export(str(mesh_path))
        
        manifest = OutputManifest(
            job_id="test_job",
            status="success",
            mesh_path=str(mesh_path),
            texture_path="", # Embedded in OBJ/Visuals for this test
            log_path="",
            is_stub=False,
            processing_time_seconds=1.0
        )
        
        # 2. Export to GLB
        exporter = GLBExporter()
        glb_output = tmp_path / "data" / "test.glb"
        glb_output.parent.mkdir(parents=True, exist_ok=True)
        
        metrics = exporter.export_to_glb(
            mesh_path=str(mesh_path),
            texture_path=None,
            output_path=str(glb_output)
        )
        
        assert glb_output.exists()
        
        # 3. Inspect Exported Asset
        # Verify metrics match real quality requirements
        assert metrics["has_uv"] is True
        assert metrics["has_embedded_texture"] is True
        assert metrics["material_semantic_status"] == "diffuse_textured"
        
        # 4. Real Validator Pass
        validator = AssetValidator()
        report = validator.validate(str(glb_output))
        assert report.status == "pass", f"Validator failed: {report.issues}"

    finally:
        settings.data_root = orig_data_root

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
