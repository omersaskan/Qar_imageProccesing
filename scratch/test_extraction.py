import sys
import os
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sys.path.append(str(Path(__file__).parent.parent))

from modules.capture_workflow.frame_extractor import FrameExtractor

def main():
    video_path = r"c:\modelPlate\data\captures\cap_9b6ff69b\video\raw_video.mp4"
    output_dir = r"c:\modelPlate\scratch\test_extraction_output"
    
    # Clean output dir
    import shutil
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
        
    print("Running newly configured frame extractor on raw_video.mp4 (rate=5)...")
    extractor = FrameExtractor()
    extractor.thresholds.frame_sample_rate = 5 # Emulate denser_frames
    extracted_paths, report = extractor.extract_keyframes(video_path, output_dir)
    print("\n\n--- Extraction Report ---")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
