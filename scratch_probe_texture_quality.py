from pathlib import Path
import numpy as np
import cv2
import json

from modules.qa_validation.texture_quality import TextureQualityAnalyzer

out = Path("scratch_texture_quality_probe")
out.mkdir(exist_ok=True)

# 1) siyah atlas
black = np.zeros((256, 256, 3), dtype=np.uint8)
black_path = out / "black_atlas.png"
cv2.imwrite(str(black_path), black)

# 2) neon/rastgele artifact atlas
neon = np.zeros((256, 256, 3), dtype=np.uint8)
neon[:, :] = (0, 255, 0)  # OpenCV BGR: neon green
neon[40:90, 40:90] = (255, 0, 0)
neon[120:170, 120:170] = (0, 0, 255)
neon_path = out / "neon_atlas.png"
cv2.imwrite(str(neon_path), neon)

# 3) white/cream atlas
cream = np.zeros((256, 256, 3), dtype=np.uint8)
cream[:, :] = (220, 230, 240)  # BGR, warm-ish light cream/white
cream_path = out / "cream_atlas.png"
cv2.imwrite(str(cream_path), cream)

analyzer = TextureQualityAnalyzer()

for name, path in [
    ("BLACK", black_path),
    ("NEON", neon_path),
    ("CREAM", cream_path),
]:
    result = analyzer.analyze_path(str(path), expected_product_color="white_cream")
    print(f"\n--- {name} ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
