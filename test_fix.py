import logging
from pathlib import Path
from modules.reconstruction_engine.texture_frame_filter import TextureFrameFilter

logging.basicConfig(level=logging.INFO)
filter = TextureFrameFilter()
image_folder = Path("data/reconstructions/legacy_cap_29ab6fa1_compare_v3/recon/attempt_1_denser_frames/dense/images")
output_dir = Path("data/reconstructions/legacy_cap_29ab6fa1_compare_v3/test_filter")
output_dir.mkdir(parents=True, exist_ok=True)

report = filter.filter_session_images(image_folder, output_dir)
print(f"Masks available: {report['has_masks_available']}")
print(f"Selected count: {report['selected_count']}")
if report['has_masks_available']:
    print(f"Masked images generated: {any(f.get('masked_source_generated') for f in report['selected_frames'])}")
