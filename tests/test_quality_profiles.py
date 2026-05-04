import pytest
from unittest.mock import MagicMock
from modules.ai_3d_generation.quality_profiles import resolve_quality_profile

def test_balanced_profile():
    settings = MagicMock()
    settings.ai_3d_max_candidates = 5
    res = resolve_quality_profile("balanced", settings)
    assert res["input_size"] == 768
    assert res["texture_resolution"] == 1024
    assert res["max_candidates"] == 3
    assert res["video_topk_frames"] == 3
    assert res["quality_mode"] == "balanced"

def test_high_profile():
    settings = MagicMock()
    settings.ai_3d_max_candidates = 10
    res = resolve_quality_profile("high", settings)
    assert res["input_size"] == 1024
    assert res["texture_resolution"] == 1024
    assert res["max_candidates"] == 5

def test_ultra_profile():
    settings = MagicMock()
    settings.ai_3d_max_candidates = 10
    res = resolve_quality_profile("ultra", settings)
    assert res["input_size"] == 1024
    assert res["texture_resolution"] == 2048
    assert res["max_candidates"] == 8

def test_unknown_fallback():
    settings = MagicMock()
    settings.ai_3d_max_candidates = 5
    res = resolve_quality_profile("unknown_mode", settings)
    assert res["quality_mode"] == "balanced"
    assert any("unknown_quality_mode_fallback_to_balanced" in w for w in res["warnings"])

def test_overrides_and_clamping():
    settings = MagicMock()
    settings.ai_3d_max_candidates = 10
    
    # Override within limits
    res = resolve_quality_profile("balanced", settings, overrides={"input_size": 1000})
    assert res["input_size"] == 1000
    
    # Clamp input_size
    res = resolve_quality_profile("balanced", settings, overrides={"input_size": 2000})
    assert res["input_size"] == 1024
    
    # Clamp texture_resolution
    res = resolve_quality_profile("balanced", settings, overrides={"texture_resolution": 4096})
    assert res["texture_resolution"] == 2048

def test_max_candidates_clamping_to_settings():
    settings = MagicMock()
    settings.ai_3d_max_candidates = 2
    
    # Ultra wants 8, but settings limit to 2
    res = resolve_quality_profile("ultra", settings)
    assert res["max_candidates"] == 2

def test_pipeline_manifest_includes_resolved_quality(tmp_path):
    from modules.ai_3d_generation.pipeline import generate_ai_3d
    from modules.operations.settings import settings as global_settings
    from unittest.mock import patch
    import os
    
    # Minimal setup to run generate_ai_3d far enough to build manifest
    sess_id = "test_q_manifest"
    output_dir = tmp_path / sess_id
    output_dir.mkdir()
    
    input_img = tmp_path / "input.jpg"
    from PIL import Image
    Image.new("RGB", (10, 10)).save(input_img)
    
    with patch("modules.ai_3d_generation.pipeline._get_provider") as mock_get_p, \
         patch("modules.ai_3d_generation.pipeline.preprocess_input") as mock_pre, \
         patch("modules.ai_3d_generation.pipeline.run_postprocess") as mock_post, \
         patch("modules.ai_3d_generation.pipeline.quality_evaluate") as mock_eval, \
         patch("modules.operations.settings.settings.ai_3d_multi_candidate_enabled", False):
        
        mock_p = MagicMock()
        mock_p.name = "sf3d"
        mock_p.license_note = "test"
        mock_p.output_format = "glb"
        mock_p.safe_generate.return_value = {
            "status": "ok",
            "output_path": str(output_dir / "out.glb"),
            "metadata": {}
        }
        mock_get_p.return_value = mock_p
        mock_pre.return_value = {"prepared_image_path": str(input_img), "warnings": []}
        mock_eval.return_value = {"verdict": "ok", "warnings": []}
        
        # Create dummy GLB
        (output_dir / "out.glb").write_bytes(b"dummy")
        
        manifest = generate_ai_3d(sess_id, str(input_img), str(output_dir), "sf3d", {"quality_mode": "ultra"})
        
        assert "resolved_quality" in manifest
        assert manifest["resolved_quality"]["quality_mode"] == "ultra"
        assert manifest["resolved_quality"]["input_size"] == 1024
        assert manifest["quality_mode"] == "ultra"
