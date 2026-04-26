import cv2
import numpy as np
import json
from modules.qa_validation.texture_quality import TextureQualityAnalyzer

def get_results(name, img, color="white_cream"):
    cv2.imwrite(f"{name}.jpg", img)
    analyzer = TextureQualityAnalyzer()
    res = analyzer.analyze_path(f"{name}.jpg", expected_product_color=color)
    print(f"\n--- {name.upper()} ---")
    print(json.dumps(res, indent=2))
    # cleanup
    for f in [f"{name}.jpg", "texture_atlas_preview.png", "texture_atlas_histogram.json"]:
        if os.path.exists(f): os.remove(f)

import os
# Case 1: Solid Black
get_results("solid_black", np.zeros((512, 512, 3), dtype=np.uint8))

# Case 2: Neon Green
hsv = np.zeros((512, 512, 3), dtype=np.uint8)
hsv[:, :, 0] = 60 # Green
hsv[:, :, 1] = 255 # Full saturation
hsv[:, :, 2] = 255 # Full value
get_results("neon", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

# Case 3: White/Cream
get_results("white_cream", np.full((512, 512, 3), (240, 245, 250), dtype=np.uint8), "white_cream")
