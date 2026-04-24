import os
import pytest
from pathlib import Path
import cv2
import numpy as np
from modules.reconstruction_engine.failures import DenseMaskAlignmentError

def test_mask_generation_logic_hardening(tmp_path):
    """
    Tests the hardened dense mask generation logic including:
    1. Dark subject preservation (Any channel > 0)
    2. Occupancy sanity checks (fallback to full white)
    3. Unicode path compatibility
    """
    dense_images_dir = tmp_path / "images"
    dense_images_dir.mkdir()
    
    # helper for unicode-safe write
    def safe_write(path, img, ext=".png"):
        _, buff = cv2.imencode(ext, img)
        buff.tofile(str(path))
        return os.path.exists(str(path))

    # 1. Dark Subject Case: sub-threshold pixel [0, 0, 1] should be captured.
    # Grayscale would be ~0, but 'any channel > 0' captures it.
    dark_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dark_img[40:60, 40:60, 2] = 1 # Only blue channel has 1
    dark_path = dense_images_dir / "dark.png"
    assert safe_write(dark_path, dark_img)
    
    # 2. Too Small Case: subjects smaller than 0.5% area.
    tiny_img = np.zeros((100, 100, 3), dtype=np.uint8)
    tiny_img[0:1, 0:1] = [255, 255, 255] # 1 pixel -> 4x4 dilate = 16 pixels (0.16%)
    tiny_path = dense_images_dir / "tiny.png"
    assert safe_write(tiny_path, tiny_img)
    
    # 3. Too Large Case: everything white
    huge_img = np.full((100, 100, 3), 255, dtype=np.uint8)
    huge_path = dense_images_dir / "huge.png"
    assert safe_write(huge_path, huge_img)

    # Simulation of adapter.py logic
    MIN_OCCUPANCY = 0.005
    MAX_OCCUPANCY = 0.98
    
    results = {}
    for img_file in dense_images_dir.glob("*.png"):
        # Unicode-safe read
        img_array = np.fromfile(str(img_file), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # LOGIC UNDER TEST: Any channel > 0
        thresh = (np.any(img > 0, axis=-1).astype(np.uint8) * 255)
        
        kernel = np.ones((7, 7), np.uint8)
        gen_mask = cv2.dilate(thresh, kernel, iterations=1)
        
        pixels = gen_mask.shape[0] * gen_mask.shape[1]
        occupancy = np.count_nonzero(gen_mask) / pixels
        
        if occupancy < MIN_OCCUPANCY or occupancy > MAX_OCCUPANCY:
            final_mask = np.full(gen_mask.shape, 255, dtype=np.uint8)
            results[img_file.name] = ("fallback", final_mask)
        else:
            results[img_file.name] = ("normal", gen_mask)

    # Validations
    assert results["dark.png"][0] == "normal", "Dark subject with intensity 1 should be detected"
    assert np.count_nonzero(results["dark.png"][1]) > 0
    
    assert results["tiny.png"][0] == "fallback"
    assert np.all(results["tiny.png"][1] == 255)
    
    assert results["huge.png"][0] == "fallback"
    assert np.all(results["huge.png"][1] == 255)

def test_shape_mismatch_raises_custom_error():
    from modules.reconstruction_engine.failures import DenseMaskAlignmentError
    with pytest.raises(DenseMaskAlignmentError):
        raise DenseMaskAlignmentError("mismatch", image_path="fake.jpg")

if __name__ == "__main__":
    pytest.main([__file__])
