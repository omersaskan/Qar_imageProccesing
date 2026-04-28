import os
import sys
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime

# Root path setup
ROOT = Path(__file__).parent.absolute()
sys.path.append(str(ROOT))

from modules.operations.settings import settings
from modules.capture_workflow.frame_extractor import FrameExtractor
from modules.reconstruction_engine.runner import ReconstructionRunner
from modules.asset_cleanup_pipeline.cleaner import AssetCleaner
from modules.asset_cleanup_pipeline.profiles import CleanupProfileType
from modules.export_pipeline.glb_exporter import GLBExporter
from modules.qa_validation.validator import AssetValidator
from modules.integration_flow import IntegrationFlow
from modules.shared_contracts.models import ReconstructionJob
from modules.operations.texturing_service import TexturingService
import trimesh
from modules.asset_cleanup_pipeline.camera_projection import load_reconstruction_cameras, load_reconstruction_masks

# Configure logging
log_file = ROOT / "reconstruction.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(str(log_file)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RealRecon")

import argparse

def run():
    parser = argparse.ArgumentParser(description="Real End-to-End Reconstruction")
    parser.add_argument("--capture-id", type=str, default="cap_29ab6fa1")
    parser.add_argument("--job-id", type=str)
    parser.add_argument("--session-id", type=str, default="real_session_001")
    args = parser.parse_args()

    capture_id = args.capture_id
    video_path = str(ROOT / "data" / "captures" / capture_id / "video" / "raw_video.mp4")
    session_id = args.session_id
    job_id = args.job_id or f"job_real_{capture_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    work_dir = ROOT / "data" / "reconstructions" / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"STARTING REAL END-TO-END RECONSTRUCTION: {job_id}")
    logger.info(f"Video: {video_path}")
    
    # 1. Extraction
    logger.info("--- 1. Extracting Frames ---")
    extractor = FrameExtractor()
    frames_dir = work_dir / "frames"
    try:
        frames, extraction_report = extractor.extract_keyframes(video_path, str(frames_dir))
        logger.info(f"Extracted {len(frames)} frames. Report saved to denser_extraction_report.json")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return
    
    # 2. Reconstruction
    logger.info("--- 2. Running Reconstruction ---")
    job = ReconstructionJob(
        job_id=job_id,
        capture_session_id=session_id,
        product_id="test_product",
        input_frames=frames,
        job_dir=str(work_dir / "recon"),
        source_video_path=video_path
    )
    
    runner = ReconstructionRunner()
    try:
        manifest = runner.run(job)
        logger.info("Reconstruction successful.")
        logger.info(f"Mesh path: {manifest.mesh_path}")
        logger.info(f"Texture path: {manifest.texture_path}")
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # 3. Cleanup
    logger.info("--- 3. Running Cleanup ---")
    
    # Load guidance data from the successful attempt
    # Fix: selected_attempt_dir resolution
    mesh_path = Path(manifest.mesh_path)
    if mesh_path.parent.name == "dense":
        best_attempt_dir = mesh_path.parent.parent
    else:
        best_attempt_dir = mesh_path.parent
    dense_dir = best_attempt_dir / "dense"
    logger.info(f"Loading guidance data from: {best_attempt_dir}")
    
    cameras = load_reconstruction_cameras(best_attempt_dir)
    masks = None
    if cameras:
        masks = load_reconstruction_masks(best_attempt_dir, [c["name"] for c in cameras])
    
    point_cloud = None
    # For OpenMVS, use project_dense.ply; for COLMAP, use fused.ply
    pc_paths = [
        dense_dir / "fused.ply",
        dense_dir / "project_dense.ply"
    ]
    for pc_path in pc_paths:
        if pc_path.exists():
            try:
                point_cloud = trimesh.load(str(pc_path))
                if isinstance(point_cloud, trimesh.points.PointCloud):
                    logger.info(f"Loaded guidance point cloud: {pc_path.name}")
                    break
                else:
                    point_cloud = None
            except Exception as pc_err:
                logger.warning(f"Failed to load point cloud {pc_path.name}: {pc_err}")

    logger.info("Cleanup Guidance: cameras=%s, masks=%s, point_cloud=%s", 
                len(cameras) if cameras else 0, len(masks) if masks else 0, bool(point_cloud))

    cleaner = AssetCleaner(data_root=str(ROOT / "data"))
    try:
        metadata, cleanup_stats, cleaned_mesh_path = cleaner.process_cleanup(
            job_id,
            manifest.mesh_path,
            CleanupProfileType.MOBILE_HIGH,
            raw_texture_path=manifest.texture_path,
            cameras=cameras,
            masks=masks,
            point_cloud=point_cloud
        )
        logger.info(f"Cleanup successful. Isolation Method: {cleanup_stats.get('isolation', {}).get('object_isolation_method')}")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return

    # 3.1. Texturing (Task 5)
    logger.info("--- 3.1. Running Texturing ---")
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
    
    # Mirror worker cleanup_stats updates (Task 5-6)
    if texturing_result.texturing_status == "real" and texturing_result.texture_atlas_paths:
        atlas_path = texturing_result.texture_atlas_paths[0]
        cleanup_stats["cleaned_texture_path"] = atlas_path
        cleanup_stats["texture_path"] = atlas_path
        cleanup_stats["has_uv"] = True
        cleanup_stats["has_material"] = True
        cleanup_stats["textured_mesh_path"] = texturing_result.cleaned_mesh_path
        cleanup_stats["texture_integrity_status"] = "complete"
        cleanup_stats["material_semantic_status"] = "diffuse_textured"
        
        if "decimation" in cleanup_stats:
            cleanup_stats["decimation"]["uv_preserved"] = True
            cleanup_stats["decimation"]["material_preserved"] = True
            cleanup_stats["decimation"]["texture_preserved"] = True

    # Task 6: Fail explicitly if REQUIRE_TEXTURED_OUTPUT=true and texturing failed
    if settings.require_textured_output and texturing_result.texturing_status in ["degraded", "absent"]:
        reason = f"TEXTURING_REQUIRED_BUT_MISSING: Status '{texturing_result.texturing_status}'"
        log_path = manifest.texturing_log_path or "unknown"
        logger.error(f"{reason} (Log: {log_path})")
        return

    # Task 7: Strict Gating
    isolation_method = cleanup_stats.get("isolation", {}).get("object_isolation_method")
    if isolation_method == "geometric_only":
        logger.error("Gating: ABORTED - Isolation fell back to geometric_only because no guidance.")
        return
        
    texture_count = 0
    if texturing_result.texturing_status == "real" and texturing_result.texture_atlas_paths:
        texture_count = len(texturing_result.texture_atlas_paths)
    
    if settings.require_textured_output and texture_count == 0:
        logger.error("Gating: ABORTED - texture_count=0 while REQUIRE_TEXTURED_OUTPUT=true")
        return

    # 4. Export
    logger.info("--- 4. Running Export ---")
    exporter = GLBExporter()
    export_dir = work_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_output = export_dir / f"{job_id}.glb"
    
    primary_texture = manifest.texture_atlas_paths[0] if manifest.texture_atlas_paths else manifest.texture_path
    texture_path = cleanup_stats.get("cleaned_texture_path") or primary_texture
    
    export_report = exporter.export(
        cleaned_mesh_path,
        str(export_output),
        profile_name="mobile_high",
        texture_path=texture_path if (texture_path and Path(texture_path).exists()) else None,
        metadata=metadata
    )
    
    if export_report.get("export_status") not in ["success", "review"]:
        logger.error(f"Gating: ABORTED - Export status '{export_report.get('export_status')}' is invalid.")
        return
        
    logger.info("Export complete.")
    
    # 5. Validation
    logger.info("--- 5. Running Validation ---")
    validator = AssetValidator()
    validator_input = IntegrationFlow.map_metadata_to_validator_input(
        metadata,
        cleanup_stats=cleanup_stats,
        export_report=export_report
    )
    
    validation_report = validator.validate(job_id, validator_input)
    logger.info(f"Validation Result: {validation_report.final_decision}")
    
    # Save final artifacts for user output
    with open(work_dir / "cleanup_stats.json", "w") as f: json.dump(cleanup_stats, f, indent=2)
    with open(work_dir / "export_metrics.json", "w") as f: json.dump(export_report, f, indent=2)
    with open(work_dir / "validation_report.json", "w") as f: json.dump(validation_report.model_dump(mode="json"), f, indent=2)
    
    logger.info("--- ALL STEPS COMPLETED ---")
    print(f"\nFinal GLB saved at: {export_output}")

if __name__ == "__main__":
    run()


