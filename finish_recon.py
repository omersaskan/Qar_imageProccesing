import os
import sys
import json
import logging
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
log_file = ROOT / "reconstruction_finish.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(str(log_file)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FinishRecon")

def run():
    job_id = "legacy_cap_29ab6fa1_compare_v3"
    work_dir = ROOT / "data" / "reconstructions" / job_id
    recon_dir = work_dir / "recon"
    
    # Load manifest from the best attempt
    manifest_path = recon_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return
        
    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)
    
    # Create a simple results class to mimic ReconstructionRunner output
    class MeshMetadataMock:
        def __init__(self, data):
            self.uv_present = data.get("uv_present", False)
            self.has_texture = data.get("has_texture", False)
            self.vertex_count = data.get("vertex_count", 0)
            self.face_count = data.get("face_count", 0)

    class ManifestMock:
        def __init__(self, data):
            self.mesh_path = data["mesh_path"]
            self.texture_path = data["texture_path"]
            self.texture_atlas_paths = data.get("texture_atlas_paths", [])
            self.job_id = data["job_id"]
            self.mesh_metadata = MeshMetadataMock(data.get("mesh_metadata", {}))
    
    manifest = ManifestMock(manifest_data)
    logger.info(f"Loaded manifest for {job_id}. Mesh: {manifest.mesh_path}")

    # 3. Cleanup
    logger.info("--- 3. Running Cleanup (RE-RUN) ---")
    
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
        cam0 = cameras[0]
        masks = load_reconstruction_masks(
            best_attempt_dir, 
            [c["name"] for c in cameras],
            expected_width=cam0["width"],
            expected_height=cam0["height"],
        )
    
    point_cloud = None
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
    logger.info(f"Cleanup successful. Isolation Method: {cleanup_stats.get('isolation', {}).get('object_isolation_method')}")

    # 3.1. Texturing
    logger.info("--- 3.1. Running Texturing (RE-RUN) ---")
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
        cleanup_stats["has_uv"] = True
        cleanup_stats["has_material"] = True
        cleanup_stats["textured_mesh_path"] = texturing_result.cleaned_mesh_path
        cleanup_stats["texture_integrity_status"] = "complete"
        cleanup_stats["material_semantic_status"] = "diffuse_textured"
        cleanup_stats["texture_applied"] = True

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
    logger.info(f"Export status: {export_report.get('export_status')}")
    
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
    
    # Save final artifacts
    with open(work_dir / "cleanup_stats.json", "w") as f: json.dump(cleanup_stats, f, indent=2)
    with open(work_dir / "export_metrics.json", "w") as f: json.dump(export_report, f, indent=2)
    with open(work_dir / "validation_report.json", "w") as f: json.dump(validation_report.model_dump(mode="json"), f, indent=2)
    
    logger.info("--- ALL STEPS COMPLETED ---")
    print(f"\nFinal GLB saved at: {export_output}")

if __name__ == "__main__":
    run()
