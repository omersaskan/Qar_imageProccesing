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
    parser.add_argument("--capture-id", type=str, default="cap_50ab7977")
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
    cleaner = AssetCleaner(data_root=str(ROOT / "data"))
    try:
        # Use OpenMVS output directly
        # If it was textured, OpenMVS output is textured.
        metadata, cleanup_stats, cleaned_mesh_path = cleaner.process_cleanup(
            job_id,
            manifest.mesh_path,
            CleanupProfileType.MOBILE_HIGH,
            raw_texture_path=manifest.texture_path
        )
        logger.info("Cleanup successful.")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return
    
    # 4. Export
    logger.info("--- 4. Running Export ---")
    exporter = GLBExporter()
    export_dir = work_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_output = export_dir / f"{job_id}.glb"
    
    export_report = exporter.export(
        cleaned_mesh_path,
        str(export_output),
        profile_name="mobile_high",
        texture_path=cleanup_stats.get("cleaned_texture_path")
    )
    logger.info("Export complete.")
    
    # 5. Validation
    logger.info("--- 5. Running Validation ---")
    validator = AssetValidator()
    validator_input = IntegrationFlow.map_metadata_to_validator_input(
        metadata,
        cleanup_stats=cleanup_stats,
        export_report=export_report,
        filtering_status="object_isolated"
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
