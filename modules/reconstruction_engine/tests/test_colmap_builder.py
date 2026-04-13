import unittest
from pathlib import Path
from modules.reconstruction_engine.adapter import ColmapCommandBuilder

class TestColmapCommandBuilder(unittest.TestCase):
    def setUp(self):
        self.bin = "C:\\colmap\\colmap\\COLMAP.bat"
        self.db = Path("data/sessions/cap_123/database.db")
        self.images = Path("data/sessions/cap_123/images")
        self.masks = Path("data/sessions/cap_123/masks")
        self.output = Path("data/sessions/cap_123/sparse")

    def test_feature_extractor_gpu_on(self):
        builder = ColmapCommandBuilder(self.bin, use_gpu=True)
        cmd = builder.feature_extractor(self.db, self.images, self.masks, 2000)
        
        self.assertIn("feature_extractor", cmd)
        self.assertIn("--FeatureExtraction.use_gpu", cmd)
        idx = cmd.index("--FeatureExtraction.use_gpu")
        self.assertEqual(cmd[idx + 1], "1")

    def test_feature_extractor_gpu_off(self):
        builder = ColmapCommandBuilder(self.bin, use_gpu=False)
        cmd = builder.feature_extractor(self.db, self.images, self.masks, 2000)
        
        idx = cmd.index("--FeatureExtraction.use_gpu")
        self.assertEqual(cmd[idx + 1], "0")

    def test_exhaustive_matcher_gpu_on(self):
        builder = ColmapCommandBuilder(self.bin, use_gpu=True)
        # Verify 4.0.3 flag: --FeatureMatching.use_gpu
        cmd = builder.matcher("exhaustive", self.db)
        
        self.assertIn("exhaustive_matcher", cmd)
        self.assertIn("--FeatureMatching.use_gpu", cmd)
        self.assertNotIn("--SiftMatching.use_gpu", cmd)
        
        idx = cmd.index("--FeatureMatching.use_gpu")
        self.assertEqual(cmd[idx + 1], "1")

    def test_sequential_matcher_gpu_off(self):
        builder = ColmapCommandBuilder(self.bin, use_gpu=False)
        cmd = builder.matcher("sequential", self.db)
        
        self.assertIn("sequential_matcher", cmd)
        self.assertIn("--FeatureMatching.use_gpu", cmd)
        
        idx = cmd.index("--FeatureMatching.use_gpu")
        self.assertEqual(cmd[idx + 1], "0")

    def test_mapper_gpu_on(self):
        builder = ColmapCommandBuilder(self.bin, use_gpu=True)
        cmd = builder.mapper(self.db, self.images, self.output)
        
        self.assertIn("mapper", cmd)
        self.assertIn("--Mapper.ba_use_gpu", cmd)
        idx = cmd.index("--Mapper.ba_use_gpu")
        self.assertEqual(cmd[idx + 1], "1")

    def test_image_undistorter(self):
        builder = ColmapCommandBuilder(self.bin)
        cmd = builder.image_undistorter(self.images, self.output, Path("dense"))
        
        self.assertIn("image_undistorter", cmd)
        self.assertIn("--output_type", cmd)
        self.assertIn("COLMAP", cmd)

    def test_patch_match_stereo_gpu_on(self):
        builder = ColmapCommandBuilder(self.bin, use_gpu=True)
        cmd = builder.patch_match_stereo(Path("dense"))
        
        self.assertIn("patch_match_stereo", cmd)
        idx = cmd.index("--PatchMatchStereo.gpu_index")
        self.assertEqual(cmd[idx + 1], "0")
        
        # Verify hardening flags
        self.assertIn("--PatchMatchStereo.geom_consistency", cmd)
        self.assertIn("--PatchMatchStereo.filter", cmd)
        
    def test_stereo_fusion(self):
        builder = ColmapCommandBuilder(self.bin)
        cmd = builder.stereo_fusion(Path("dense"), Path("fused.ply"))
        self.assertIn("stereo_fusion", cmd)

    def test_meshing_commands(self):
        builder = ColmapCommandBuilder(self.bin)
        
        poisson = builder.poisson_mesher(Path("f"), Path("o"))
        self.assertIn("poisson_mesher", poisson)
        
        # FIX check: delaunay now takes dir
        delaunay = builder.delaunay_mesher(Path("dense_dir"), Path("o"))
        self.assertIn("delaunay_mesher", delaunay)
        self.assertIn("dense_dir", delaunay)
        self.assertNotIn("fused.ply", delaunay)

    def test_model_analyzer(self):
        builder = ColmapCommandBuilder(self.bin)
        cmd = builder.model_analyzer(Path("sparse/0"))
        self.assertIn("model_analyzer", cmd)
        self.assertIn("--path", cmd)

if __name__ == '__main__':
    unittest.main()
