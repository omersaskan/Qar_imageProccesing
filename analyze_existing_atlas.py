import sys, json
from modules.qa_validation.texture_quality import TextureQualityAnalyzer

atlas = sys.argv[1]
result = TextureQualityAnalyzer().analyze_path(atlas, expected_product_color="white_cream")
print(json.dumps(result, indent=2, ensure_ascii=False))
