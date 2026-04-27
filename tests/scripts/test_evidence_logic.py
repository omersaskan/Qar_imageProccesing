import pytest
import json
from pathlib import Path
from scripts.export_reconstruction_evidence import generate_checklist

@pytest.fixture
def base_evidence():
    return {
        "job_id": "test_job",
        "config_snapshot": {
            "require_textured_output": True
        },
        "reports": {
            "capture_report": {"overall_status": "PASS"},
            "extraction_report": {"saved_count": 40},
            "reconstruction_audit": {
                "selected_best_index": 0,
                "attempts": [{
                    "registered_images": 40,
                    "dense_points_fused": 30000,
                    "metadata": {
                        "dense_mask_exact_matches": 40,
                        "dense_mask_dimension_matches": 40,
                        "dense_mask_fallback_white_ratio": 0.02
                    }
                }]
            },
            "cleanup_stats": {
                "isolation": {
                    "object_isolation_method": "mask_guided",
                    "isolation_confidence": 0.9
                }
            },
            "export_metrics": {
                "texture_count": 1,
                "material_count": 1,
                "all_textured_primitives_have_texcoord_0": True,
                "texture_applied": True
            },
            "validation_report": {
                "final_decision": "pass"
            }
        }
    }

def test_production_ready_success(base_evidence):
    checklist = generate_checklist(base_evidence)
    assert checklist["final_status"] == "production_ready"

def test_review_ready_capture_warn(base_evidence):
    base_evidence["reports"]["capture_report"]["overall_status"] = "warn"
    checklist = generate_checklist(base_evidence)
    assert checklist["final_status"] == "review_ready"
    assert checklist["capture_status"]["status"] == "review_ready"

def test_fail_fallback_white_too_high(base_evidence):
    base_evidence["reports"]["reconstruction_audit"]["attempts"][0]["metadata"]["dense_mask_fallback_white_ratio"] = 0.15
    checklist = generate_checklist(base_evidence)
    assert checklist["final_status"] == "failed"
    assert "Excessive mask fallback" in checklist["failure_reasons"][0]

def test_review_ready_fallback_white_moderate(base_evidence):
    base_evidence["reports"]["reconstruction_audit"]["attempts"][0]["metadata"]["dense_mask_fallback_white_ratio"] = 0.08
    checklist = generate_checklist(base_evidence)
    assert checklist["final_status"] == "review_ready"

def test_fail_texture_missing_when_required(base_evidence):
    base_evidence["reports"]["export_metrics"]["texture_applied"] = False
    checklist = generate_checklist(base_evidence)
    assert checklist["final_status"] == "failed"
    assert any("texture" in r.lower() for r in checklist["failure_reasons"])

def test_fail_dense_mask_count_mismatch(base_evidence):
    # exact matches 35 but total images 40
    base_evidence["reports"]["reconstruction_audit"]["attempts"][0]["metadata"]["dense_mask_exact_matches"] = 35
    checklist = generate_checklist(base_evidence)
    # 35 > 0 so it's review_ready according to my logic, but user said:
    # "fail when dense mask counts are nonzero but not equal to dense_image_count"
    # Wait, user said: "Do not treat nonzero counts as success."
    # If not production, and > 0, is it review or fail?
    # User said: "dense masks must have exact filename matches and dimension matches for all dense images." for production.
    # If not exact, it should probably be REVIEW if we allow it, but user says "fail when... not equal".
    # I'll update my logic to fail if not exact if that's what they want.
    # Actually, "Do not treat nonzero counts as success" refers to the PREVIOUS check.
    # User requirement: "fail when dense mask counts are nonzero but not equal to dense_image_count"
    # OK, I'll update the script logic for masks to fail if not equal.
    assert checklist["final_status"] == "failed"

def test_review_ready_geometric_isolation(base_evidence):
    base_evidence["reports"]["cleanup_stats"]["isolation"]["object_isolation_method"] = "geometric_only"
    checklist = generate_checklist(base_evidence)
    assert checklist["final_status"] == "review_ready"
    assert checklist["object_isolation_method"]["status"] == "review_ready"

def test_fail_low_points(base_evidence):
    base_evidence["reports"]["reconstruction_audit"]["attempts"][0]["dense_points_fused"] = 5000
    checklist = generate_checklist(base_evidence)
    assert checklist["final_status"] == "failed"
