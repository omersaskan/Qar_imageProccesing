import os
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
logger = logging.getLogger("FinishCreamRun")

def run():
    capture_id = "cap_29ab6fa1"
    job_id = "cap_29ab6fa1_v5_cream"
    
    workspace_dir = ROOT / "data" / "reconstructions" / job_id
    
    # 1. Verification
    manifest_path = workspace_dir / "recon" / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"Manifest not found in {workspace_dir}. Reconstruction might have failed earlier.")
        return

    # 2. Load Manifest
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)
    
    from modules.reconstruction_engine.output_manifest import OutputManifest
    manifest = OutputManifest.model_validate(manifest_data)
    
    # 3. Configure Environment for Cream
    os.environ["TEXTURE_NEUTRALIZATION_TYPE"] = "cream"
    os.environ["SAM2_ENABLED"] = "false"
    os.environ["REQUIRE_TEXTURED_OUTPUT"] = "true"
    os.environ["EXPECTED_PRODUCT_COLOR"] = "white_cream"
    
    # 4. Run Cleanup
    logger.info("--- Finishing Cleanup for Cream ---")
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
        job_id,
        manifest.mesh_path,
        CleanupProfileType.MOBILE_HIGH,
        raw_texture_path=manifest.texture_path,
        cameras=cameras,
        masks=masks,
        point_cloud=point_cloud
    )
    
    # 5. Texturing
    logger.info("--- Running Texturing (Cream) ---")
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
    
    # 6. Export
    logger.info("--- Running Export ---")
    exporter = GLBExporter()
    export_dir = workspace_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_output = export_dir / f"{job_id}.glb"
    
    primary_texture = manifest.texture_atlas_paths[0] if manifest.texture_atlas_paths else manifest.texture_path
    
    export_report = exporter.export(
        cleaned_mesh_path,
        str(export_output),
        profile_name="mobile_high",
        texture_path=primary_texture,
        metadata=metadata
    )
    
    # 7. Validation
    logger.info("--- Running Validation ---")
    validator = AssetValidator()
    validator_input = IntegrationFlow.map_metadata_to_validator_input(metadata, cleanup_stats=cleanup_stats, export_report=export_report)
    validation_report = validator.validate(job_id, validator_input)
    
    # Save artifacts
    with open(workspace_dir / "export_metrics.json", "w") as f: json.dump(export_report, f, indent=2)
    with open(workspace_dir / "validation_report.json", "w") as f: json.dump(validation_report.model_dump(mode="json"), f, indent=2)
    
    # Export Evidence
    evidence_dir = ROOT / "evidence_cap_29ab6fa1_v5" / "run_a_cream"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    import subprocess
    subprocess.run(["py", "scripts/export_reconstruction_evidence.py", "--job-id", job_id, "--workspace", str(workspace_dir), "--output-dir", str(evidence_dir)], check=True)
    
    metrics_candidates = list((workspace_dir / "recon").glob("**/texturing_metrics.json"))
    if metrics_candidates:
         shutil.copy2(metrics_candidates[0], evidence_dir / "texturing_metrics.json")
    
    logger.info("CREAM FINISH COMPLETED.")

if __name__ == "__main__":
    run()
