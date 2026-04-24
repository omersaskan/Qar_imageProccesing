import unittest
from pathlib import Path
from unittest.mock import patch
from modules.reconstruction_engine.adapter import ColmapCommandBuilder, ColmapCapabilityManager

class TestColmapCommandBuilder(unittest.TestCase):
    def setUp(self):
        self.bin = "C:\\colmap\\colmap\\COLMAP.bat"
        self.db = Path("data/sessions/cap_123/database.db")
        self.images = Path("data/sessions/cap_123/images")
        self.masks = Path("data/sessions/cap_123/masks")
        self.output = Path("data/sessions/cap_123/sparse")

        # Default capabilities (Legacy style, with CUDA)
        self.default_caps = {
            "extraction_prefix": "FeatureExtraction",
            "matching_prefix": "FeatureMatching",
            "has_ba_gpu": True,
            "has_cuda": True,
        }

    @patch("modules.reconstruction_engine.adapter.ColmapCapabilityManager.get_capabilities")
    def test_feature_extractor_grouped_modern(self, mock_get):
        # Mock modern COLMAP 3.8+ behavior
        mock_get.return_value = {
            "extraction_prefix": "SiftExtraction",
            "matching_prefix": "SiftMatching",
            "has_ba_gpu": True,
            "has_cuda": True,
        }
        
        builder = ColmapCommandBuilder(self.bin, use_gpu=True)
        cmd = builder.feature_extractor(self.db, self.images, self.masks, 2000)
        
        self.assertIn("--SiftExtraction.use_gpu", cmd)
        self.assertIn("--SiftExtraction.max_image_size", cmd)
        self.assertNotIn("--FeatureExtraction.use_gpu", cmd)

    @patch("modules.reconstruction_engine.adapter.ColmapCapabilityManager.get_capabilities")
    def test_gpu_fallback_on_no_cuda_build(self, mock_get):
        # Mock build "without CUDA"
        mock_get.return_value = {
            "extraction_prefix": "FeatureExtraction",
            "matching_prefix": "FeatureMatching",
            "has_ba_gpu": True,
            "has_cuda": False,
        }
        
        # User wants GPU, but build doesn't support it
        builder = ColmapCommandBuilder(self.bin, use_gpu=True)
        self.assertFalse(builder.use_gpu, "Should auto-disable GPU if build has no CUDA")
        
        cmd = builder.feature_extractor(self.db, self.images, self.masks, 2000)
        idx = cmd.index("--FeatureExtraction.use_gpu")
        self.assertEqual(cmd[idx + 1], "0", "Flag should be 0 since GPU is unsupported")

    @patch("modules.reconstruction_engine.adapter.ColmapCapabilityManager.get_capabilities")
    def test_mapper_skips_ba_gpu_if_unsupported(self, mock_get):
        mock_get.return_value = {
            "extraction_prefix": "FeatureExtraction",
            "matching_prefix": "FeatureMatching",
            "has_ba_gpu": False,
            "has_cuda": True,
        }
        
        builder = ColmapCommandBuilder(self.bin, use_gpu=True)
        cmd = builder.mapper(self.db, self.images, self.output)
        
        self.assertIn("mapper", cmd)
        self.assertNotIn("--Mapper.ba_use_gpu", cmd, "Should skip --Mapper.ba_use_gpu if unsupported")

    @patch("modules.reconstruction_engine.adapter.ColmapCapabilityManager.get_capabilities")
    def test_exhaustive_matcher_gpu_on(self, mock_get):
        mock_get.return_value = self.default_caps
        builder = ColmapCommandBuilder(self.bin, use_gpu=True)
        cmd = builder.matcher("exhaustive", self.db)
        
        self.assertIn("exhaustive_matcher", cmd)
        self.assertIn("--FeatureMatching.use_gpu", cmd)
        idx = cmd.index("--FeatureMatching.use_gpu")
        self.assertEqual(cmd[idx + 1], "1")

    @patch("modules.reconstruction_engine.adapter.ColmapCapabilityManager.get_capabilities")
    def test_patch_match_stereo_no_cuda(self, mock_get):
        mock_get.return_value = {
            "extraction_prefix": "FeatureExtraction",
            "matching_prefix": "FeatureMatching",
            "has_ba_gpu": True,
            "has_cuda": False,
        }
        builder = ColmapCommandBuilder(self.bin, use_gpu=True)
        cmd = builder.patch_match_stereo(Path("dense"))
        
        idx = cmd.index("--PatchMatchStereo.gpu_index")
        self.assertEqual(cmd[idx + 1], "-1", "Should use -1 for CPU mode in patch_match_stereo")

    def test_image_undistorter(self):
        builder = ColmapCommandBuilder(self.bin)
        cmd = builder.image_undistorter(self.images, self.output, Path("dense"))
        
        self.assertIn("image_undistorter", cmd)
        self.assertIn("--output_type", cmd)
        self.assertIn("COLMAP", cmd)

    def test_stereo_fusion(self):
        builder = ColmapCommandBuilder(self.bin)
        cmd = builder.stereo_fusion(Path("dense"), Path("fused.ply"))
        self.assertIn("stereo_fusion", cmd)

    def test_meshing_commands(self):
        builder = ColmapCommandBuilder(self.bin)
        
        poisson = builder.poisson_mesher(Path("f"), Path("o"))
        self.assertIn("poisson_mesher", poisson)
        
        delaunay = builder.delaunay_mesher(Path("dense_dir"), Path("o"))
        self.assertIn("delaunay_mesher", delaunay)

    def test_model_analyzer(self):
        builder = ColmapCommandBuilder(self.bin)
        cmd = builder.model_analyzer(Path("sparse/0"))
        self.assertIn("model_analyzer", cmd)

if __name__ == '__main__':
    unittest.main()
