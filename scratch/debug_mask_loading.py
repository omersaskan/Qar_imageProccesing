import sys
sys.path.insert(0, r'c:\Users\Lenovo\.gemini\antigravity\scratch\Qar_imageProccesing')
from pathlib import Path
from modules.asset_cleanup_pipeline.camera_projection import load_reconstruction_cameras, load_reconstruction_masks

# Simulate what run_real_recon.py does (correct path)
attempt_dir = Path(r'c:\Users\Lenovo\.gemini\antigravity\scratch\Qar_imageProccesing\data\reconstructions\legacy_cap_29ab6fa1_compare_v2\recon\attempt_1_denser_frames')

cameras = load_reconstruction_cameras(attempt_dir)
print(f"Cameras loaded (attempt_dir): {len(cameras)}")

if cameras:
    cam_names = [c['name'] for c in cameras]
    masks = load_reconstruction_masks(attempt_dir, cam_names)
    print(f"Masks loaded (attempt_dir): {len(masks)}")
    if masks:
        first = list(masks.keys())[0]
        print(f"First mask name: {first}, shape: {masks[first].shape}")
        cam0 = cameras[0]
        mask0 = masks.get(cam0['name'])
        if mask0 is not None:
            cw = cam0['width']
            ch = cam0['height']
            mw = mask0.shape[1]
            mh = mask0.shape[0]
            print(f"Camera resolution: {cw}x{ch}")
            print(f"Mask resolution: {mw}x{mh}")
            print(f"DIMENSION MATCH: {mw == cw and mh == ch}")
    else:
        print("NO MASKS LOADED - this is the bug!")
else:
    print("NO CAMERAS LOADED!")
