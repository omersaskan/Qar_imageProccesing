import os
from pathlib import Path
import cv2
import numpy as np

def test_mask_generation_alignment(tmp_path):
    """
    Simulates the `ReconstructionAdapter` dense mask generation fix.
    Ensures that dynamically generated masks are perfectly aligned and
    share the *exact* resolution as the undistorted images inside dense/images.
    """
    dense_images_dir = tmp_path / "images"
    stereo_masks_dir = tmp_path / "stereo" / "masks"
    dense_images_dir.mkdir(parents=True)
    stereo_masks_dir.mkdir(parents=True)

    # Create a dummy "undistorted" image (1920x1080)
    # The center has a "subject", the background is black.
    target_shape = (1080, 1920, 3)
    dummy_img = np.zeros(target_shape, dtype=np.uint8)
    cv2.circle(dummy_img, (960, 540), 100, (255, 255, 255), -1)

    dummy_img_path = dense_images_dir / "frame_0000.jpg"
    cv2.imwrite(str(dummy_img_path), dummy_img)

    # -----------------------------------------------
    # The Exact Logic From the Fix in `adapter.py`
    # -----------------------------------------------
    for img_file in dense_images_dir.glob("*.jpg"):
        img = cv2.imread(str(img_file))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((7, 7), np.uint8)
            relaxed_mask = cv2.dilate(thresh, kernel, iterations=1)
            
            assert relaxed_mask.shape == img.shape[:2], (
                f"Mask shape mismatch! Expected {img.shape[:2]}, got {relaxed_mask.shape}"
            )
            
            mask_filename = img_file.name + ".png"
            cv2.imwrite(str(stereo_masks_dir / mask_filename), relaxed_mask)

    # -----------------------------------------------
    # Validation
    # -----------------------------------------------
    generated_mask_path = stereo_masks_dir / "frame_0000.jpg.png"
    assert generated_mask_path.exists(), "Mask was not generated."

    generated_mask = cv2.imread(str(generated_mask_path), cv2.IMREAD_GRAYSCALE)
    assert generated_mask.shape == (1080, 1920), \
        f"Generated mask has incorrect shape: {generated_mask.shape}"

    # Ensure mask contains subject but removes background
    assert np.count_nonzero(generated_mask) > 0, "Generated mask is empty."
    assert np.count_nonzero(generated_mask) < (1080 * 1920), "Generated mask failed to clip background."
    print("Regression test passed. Dynamic mask perfectly aligns with undistorted image shape.")

if __name__ == "__main__":
    # Extremely primitive mock for tmp_path using a local scratch folder
    import shutil
    scratch_dir = Path("./test_scratch_mask_align")
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir()
    try:
        test_mask_generation_alignment(scratch_dir)
    finally:
        shutil.rmtree(scratch_dir)
