import os
import sys
import json
import shutil
import logging
from pathlib import Path

# SPRINT v6: Set environment variables first
os.environ["TEXTURE_NEUTRALIZATION_TYPE"] = "black_mask"
os.environ["SAM2_ENABLED"] = "false"
os.environ["REQUIRE_TEXTURED_OUTPUT"] = "true"
os.environ["EXPECTED_PRODUCT_COLOR"] = "white_cream"

ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT))

from modules.operations.settings import settings
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType
from modules.export_pipeline.glb_exporter import GLBExporter
from modules.qa_validation.validator import AssetValidator
from modules.integration_flow import IntegrationFlow
from modules.operations.texturing_service import TexturingService
import trimesh
from modules.asset_cleanup_pipeline.camera_projection import load_reconstruction_cameras, load_reconstruction_masks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("V6Diagnostic")

def run():
    capture_id = "cap_29ab6fa1"
    src_job_id = "cap_29ab6fa1_v5_cream"
    dst_job_id = "cap_29ab6fa1_v6_diag"
    
    src_dir = ROOT / "data" / "reconstructions" / src_job_id
    dst_dir = ROOT / "data" / "reconstructions" / dst_job_id
    
    # 1. Setup destination
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy recon artifacts
    logger.info(f"Copying reconstruction artifacts from {src_job_id} for v6 diagnostic...")
    shutil.copytree(src_dir / "recon", dst_dir / "recon")
    
    # 2. Load Manifest
    manifest_path = dst_dir / "recon" / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)
    
    from modules.reconstruction_engine.output_manifest import OutputManifest
    manifest = OutputManifest.model_validate(manifest_data)
    
    # Update paths in manifest to point to dst_dir
    def fix_path(p):
        if p and src_job_id in p:
            return p.replace(src_job_id, dst_job_id)
        return p

    manifest.mesh_path = fix_path(manifest.mesh_path)
    manifest.textured_mesh_path = fix_path(manifest.textured_mesh_path)
    manifest.texture_path = fix_path(manifest.texture_path)
    manifest.texture_atlas_paths = [fix_path(p) for p in manifest.texture_atlas_paths]
    
    # 3. Run Cleanup (Includes strict Mask QA via TextureFrameFilter later)
    logger.info("--- Running Cleanup (v6) ---")
    mesh_path = Path(manifest.mesh_path)
    best_attempt_dir = mesh_path.parent.parent if mesh_path.parent.name == "dense" else mesh_path.parent
    dense_dir = best_attempt_dir / "dense"
    
    cameras = load_reconstruction_cameras(best_attempt_dir)
    cam0 = cameras[0] if cameras else None
    masks = load_reconstruction_masks(best_attempt_dir, [c["name"] for c in cameras], cam0["width"], cam0["height"]) if cameras else None
    
    point_cloud = None
    pc_paths = [dense_dir / "fused.ply", dense_dir / "project_dense.ply"]
    for pc_path in pc_paths:
        if pc_path.exists():
            point_cloud = trimesh.load(str(pc_path))
            break

    cleaner = AssetCleaner(data_root=str(ROOT / "data"))
    metadata, cleanup_stats, cleaned_mesh_path = cleaner.process_cleanup(
        dst_job_id,
        manifest.mesh_path,
        CleanupProfileType.MOBILE_HIGH,
        raw_texture_path=manifest.texture_path,
        cameras=cameras,
        masks=masks,
        point_cloud=point_cloud
    )
    
    # 4. Texturing (Includes Mask QA & Temporal Filter)
    logger.info("--- Running Texturing (v6) ---")
    texturing_service = TexturingService()
    texturing_result = texturing_service.run(
        manifest=manifest,
        cleanup_stats=cleanup_stats,
        pivot_offset=metadata.pivot_offset,
        cleaned_mesh_path=cleaned_mesh_path,
        expected_color="white_cream"
    )
    manifest = texturing_result.manifest
    cleaned_mesh_path = texturing_result.cleaned_mesh_path
    
    # 5. Export
    logger.info("--- Running Export (v6) ---")
    exporter = GLBExporter()
    export_dir = dst_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_output = export_dir / f"{dst_job_id}.glb"
    
    primary_texture = manifest.texture_atlas_paths[0] if manifest.texture_atlas_paths else manifest.texture_path
    
    export_report = exporter.export(
        cleaned_mesh_path,
        str(export_output),
        profile_name="mobile_high",
        texture_path=primary_texture,
        metadata=metadata
    )
    
    # 6. Validation
    logger.info("--- Running Validation (v6) ---")
    validator = AssetValidator()
    validator_input = IntegrationFlow.map_metadata_to_validator_input(metadata, cleanup_stats=cleanup_stats, export_report=export_report)
    validation_report = validator.validate(dst_job_id, validator_input)
    
    # Save artifacts
    with open(dst_dir / "export_metrics.json", "w") as f: json.dump(export_report, f, indent=2)
    with open(dst_dir / "validation_report.json", "w") as f: json.dump(validation_report.model_dump(mode="json"), f, indent=2)
    
    # --- v6 Classification ---
    logger.info("--- Final v6 Diagnostic Classification ---")
    
    # Load metrics
    try:
        with open(ROOT / "data" / "cleaned" / dst_job_id / "texturing" / "selected_texture_frames.json", "r") as f:
            t_report = json.load(f)
        with open(ROOT / "data" / "cleaned" / dst_job_id / "texturing" / "texture_quality_report.json", "r") as f:
            q_report = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load diagnostic reports: {e}")
        return

    classification = "unknown"
    reasons = []
    
    # 1. Mask Quality
    mqa = t_report.get("mask_qa_report", {})
    frames = mqa.get("frames", {})
    temporal = mqa.get("temporal_summary", {})
    
    bad_mask_count = sum(1 for f in t_report.get("rejected_frames", []) if "mask" in str(f.get("rejection_reasons", "")))
    if bad_mask_count > len(frames) * 0.5:
        classification = "mask_quality_outliers"
        reasons.append("Majority of frames have poor mask quality or temporal jumps.")
    
    # 2. View Coverage
    if t_report.get("coverage_risk"):
        classification = "view_coverage_gap"
        reasons.append(f"Max gap of {t_report.get('max_gap_degrees'):.1f} degrees.")
        if t_report.get("recapture_required"):
            classification = "recapture_required"
            reasons.append("Missing side coverage requires recapture.")

    # 3. Projection / Detail
    leakage = q_report.get("neutralized_background_leakage", 0)
    detail = q_report.get("texture_detail_entropy", 0)
    if leakage > 0.3:
        classification = "OpenMVS_projection_issue"
        reasons.append(f"High background leakage ({leakage:.2f}) into visible mesh surfaces.")
    
    # 4. False Positive Check
    if q_report.get("texture_quality_status") == "fail" and leakage < 0.1 and q_report.get("expected_product_color_match_score", 0) < 0.5:
        classification = "validation_metric_false_positive_for_white_cream_product"
        reasons.append("Failed color match but low background leakage. Validation is too strict for white product.")
    
    # 5. Improvement
    if validation_report.final_decision == "pass":
        classification = "improved"
        reasons.append("V6 diagnostic measures resolved the issues.")

    v6_report = {
        "classification": classification,
        "reasons": reasons,
        "metrics": {
            "max_gap": t_report.get("max_gap_degrees"),
            "leakage": leakage,
            "detail_entropy": detail,
            "mask_temporal_area_std": temporal.get("occupancy_ratio", {}).get("std"),
            "frame_0021_status": t_report.get("frame_0021_status")
        }
    }
    
    with open(dst_dir / "v6_diagnostic_report.json", "w") as f:
        json.dump(v6_report, f, indent=2)
    
    logger.info(f"V6 CLASSIFICATION: {classification}")
    logger.info(f"Reasons: {reasons}")

if __name__ == "__main__":
    run()
