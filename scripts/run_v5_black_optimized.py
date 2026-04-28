import os
os.environ["TEXTURE_NEUTRALIZATION_TYPE"] = "black_mask"

import sys
import json
import shutil
import logging
from pathlib import Path

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
logger = logging.getLogger("OptimizedBlackRun")

def run():
    capture_id = "cap_29ab6fa1"
    src_job_id = "cap_29ab6fa1_v5_cream"
    dst_job_id = "cap_29ab6fa1_v5_black"
    
    src_dir = ROOT / "data" / "reconstructions" / src_job_id
    dst_dir = ROOT / "data" / "reconstructions" / dst_job_id
    
    # 1. Verification
    manifest_path_src = src_dir / "recon" / "manifest.json"
    if not manifest_path_src.exists():
        logger.error(f"Source job {src_job_id} manifest not found.")
        return

    # 2. Setup destination
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy recon artifacts
    logger.info(f"Copying reconstruction artifacts from {src_job_id} to {dst_job_id}...")
    shutil.copytree(src_dir / "recon", dst_dir / "recon")
    # Also copy frames if needed (AssetCleaner might need them if it re-loads images)
    if (src_dir / "frames").exists():
        shutil.copytree(src_dir / "frames", dst_dir / "frames")
    
    # 3. Load Manifest
    manifest_path = dst_dir / "recon" / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)
    
    from modules.reconstruction_engine.output_manifest import OutputManifest
    manifest = OutputManifest.model_validate(manifest_data)
    
    # Update paths in manifest to point to dst_dir
    # (Assuming manifest paths were absolute, which they often are in this codebase)
    def fix_path(p):
        if p and src_job_id in p:
            return p.replace(src_job_id, dst_job_id)
        return p

    manifest.mesh_path = fix_path(manifest.mesh_path)
    manifest.textured_mesh_path = fix_path(manifest.textured_mesh_path)
    manifest.texture_path = fix_path(manifest.texture_path)
    manifest.texture_atlas_paths = [fix_path(p) for p in manifest.texture_atlas_paths]
    manifest.log_path = fix_path(manifest.log_path)
    
    # 4. Configure Environment for Black Mask
    os.environ["TEXTURE_NEUTRALIZATION_TYPE"] = "black_mask"
    os.environ["SAM2_ENABLED"] = "false"
    os.environ["REQUIRE_TEXTURED_OUTPUT"] = "true"
    os.environ["EXPECTED_PRODUCT_COLOR"] = "white_cream"
    # Re-init settings to pick up env
    from modules.operations.settings import Settings
    global settings
    settings = Settings()
    
    # 5. Run Cleanup
    logger.info("--- Running Cleanup (Optimized) ---")
    mesh_path = Path(manifest.mesh_path)
    if mesh_path.parent.name == "dense":
        best_attempt_dir = mesh_path.parent.parent
    else:
        best_attempt_dir = mesh_path.parent
    
    dense_dir = best_attempt_dir / "dense"
    
    cameras = load_reconstruction_cameras(best_attempt_dir)
    masks = None
    if cameras:
        cam0 = cameras[0]
        masks = load_reconstruction_masks(
            best_attempt_dir, 
            [c["name"] for c in cameras],
            expected_width=cam0["width"],
            expected_height=cam0["height"],
        )
    
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
    
    # 6. Texturing
    logger.info("--- Running Texturing (Black Mask) ---")
    texturing_service = TexturingService()
    texturing_result = texturing_service.run(
        manifest=manifest,
        cleanup_stats=cleanup_stats,
        pivot_offset=metadata.pivot_offset,
        cleaned_mesh_path=cleaned_mesh_path,
        expected_color=settings.expected_product_color
    )
    manifest = texturing_result.manifest
    cleaned_mesh_path = texturing_result.cleaned_mesh_path
    
    if texturing_result.texturing_status == "real" and texturing_result.texture_atlas_paths:
        atlas_path = texturing_result.texture_atlas_paths[0]
        cleanup_stats["cleaned_texture_path"] = atlas_path
        cleanup_stats["texture_path"] = atlas_path

    # 7. Export
    logger.info("--- Running Export ---")
    exporter = GLBExporter()
    export_dir = dst_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_output = export_dir / f"{dst_job_id}.glb"
    
    primary_texture = manifest.texture_atlas_paths[0] if manifest.texture_atlas_paths else manifest.texture_path
    texture_path = cleanup_stats.get("cleaned_texture_path") or primary_texture
    
    export_report = exporter.export(
        cleaned_mesh_path,
        str(export_output),
        profile_name="mobile_high",
        texture_path=texture_path if (texture_path and Path(texture_path).exists()) else None,
        metadata=metadata
    )
    
    # 8. Validation
    logger.info("--- Running Validation ---")
    validator = AssetValidator()
    validator_input = IntegrationFlow.map_metadata_to_validator_input(
        metadata,
        cleanup_stats=cleanup_stats,
        export_report=export_report
    )
    validation_report = validator.validate(dst_job_id, validator_input)
    
    # Save artifacts
    with open(dst_dir / "cleanup_stats.json", "w") as f: json.dump(cleanup_stats, f, indent=2)
    with open(dst_dir / "export_metrics.json", "w") as f: json.dump(export_report, f, indent=2)
    with open(dst_dir / "validation_report.json", "w") as f: json.dump(validation_report.model_dump(mode="json"), f, indent=2)
    
    # Export Evidence
    evidence_dir = ROOT / "evidence_cap_29ab6fa1_v5" / "run_b_black_mask"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    
    cmd_export = [
        "py", "scripts/export_reconstruction_evidence.py",
        "--job-id", dst_job_id,
        "--workspace", str(dst_dir),
        "--output-dir", str(evidence_dir)
    ]
    import subprocess
    subprocess.run(cmd_export, check=True)
    
    # Copy texturing_metrics.json
    metrics_candidates = list((dst_dir / "recon").glob("**/texturing_metrics.json"))
    if metrics_candidates:
         shutil.copy2(metrics_candidates[0], evidence_dir / "texturing_metrics.json")
    
    logger.info("OPTIMIZED BLACK RUN COMPLETED.")
    print(f"Final GLB: {export_output}")

if __name__ == "__main__":
    run()
