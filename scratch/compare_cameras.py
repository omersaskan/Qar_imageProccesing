import sys
sys.path.insert(0, r'c:\Users\Lenovo\.gemini\antigravity\scratch\Qar_imageProccesing')
from pathlib import Path
from modules.asset_cleanup_pipeline.camera_projection import read_colmap_cameras_bin, read_colmap_images_bin

# Compare original vs undistorted camera params
original = Path(r'c:\Users\Lenovo\.gemini\antigravity\scratch\Qar_imageProccesing\data\reconstructions\legacy_cap_29ab6fa1_compare_v2\recon\attempt_1_denser_frames\sparse\1')
undist = Path(r'c:\Users\Lenovo\.gemini\antigravity\scratch\Qar_imageProccesing\data\reconstructions\legacy_cap_29ab6fa1_compare_v2\recon\attempt_1_denser_frames\dense\sparse')

orig_cams = read_colmap_cameras_bin(original / 'cameras.bin')
undist_cams = read_colmap_cameras_bin(undist / 'cameras.bin')

print("=== Original Camera ===")
for cid, cam in orig_cams.items():
    print(f"  id={cid}, model={cam['model']}, WxH={cam['width']}x{cam['height']}")
    print(f"  params={cam['params']}")
    print(f"  K=\n{cam['K']}")

print("\n=== Undistorted Camera ===")
for cid, cam in undist_cams.items():
    print(f"  id={cid}, model={cam['model']}, WxH={cam['width']}x{cam['height']}")
    print(f"  params={cam['params']}")
    print(f"  K=\n{cam['K']}")

# Check extrinsics too (1 image)
orig_imgs = read_colmap_images_bin(original / 'images.bin')
undist_imgs = read_colmap_images_bin(undist / 'images.bin')

# Find same image
name = 'frame_0000.jpg'
orig_img = None
undist_img = None
for img in orig_imgs.values():
    if img['name'] == name:
        orig_img = img
        break
for img in undist_imgs.values():
    if img['name'] == name:
        undist_img = img
        break

if orig_img and undist_img:
    print(f"\n=== Extrinsics for {name} ===")
    print(f"Original tvec: {orig_img['tvec']}")
    print(f"Undist tvec: {undist_img['tvec']}")
    print(f"Same? {all(abs(a - b) < 1e-6 for a, b in zip(orig_img['tvec'], undist_img['tvec']))}")
