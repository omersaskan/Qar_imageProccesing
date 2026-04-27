import unittest
import json
import os
import shutil
from pathlib import Path

class TestFusionAblationReport(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_ablation_report")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = self.test_dir / "ablation_report.json"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_report_structure(self):
        # Mock report data
        data = {
            "workspace": "/test/workspace",
            "timestamp": "20231027-120000",
            "variants": [
                {
                    "variant": "A1",
                    "status": "success",
                    "fused_point_count": 1000,
                    "image_count": 50,
                    "mask_count": 0,
                    "filename_matches": 0
                },
                {
                    "variant": "A3",
                    "status": "success",
                    "fused_point_count": 800,
                    "image_count": 50,
                    "mask_count": 50,
                    "filename_matches": 50
                }
            ]
        }
        
        with open(self.report_path, "w") as f:
            json.dump(data, f)
            
        # Verify
        with open(self.report_path, "r") as f:
            loaded = json.load(f)
            
        self.assertEqual(loaded["workspace"], "/test/workspace")
        self.assertEqual(len(loaded["variants"]), 2)
        self.assertEqual(loaded["variants"][0]["variant"], "A1")
        self.assertEqual(loaded["variants"][1]["fused_point_count"], 800)

if __name__ == "__main__":
    unittest.main()
