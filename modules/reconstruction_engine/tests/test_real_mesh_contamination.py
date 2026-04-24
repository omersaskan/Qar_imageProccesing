import unittest
import shutil
from pathlib import Path
import sys

import numpy as np
import trimesh

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.asset_cleanup_pipeline.isolation import MeshIsolator
from modules.reconstruction_engine.mesh_selector import MeshSelector
from modules.qa_validation.validator import AssetValidator


class TestRealMeshContamination(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_real_mesh_contamination")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.isolator = MeshIsolator()
        self.selector = MeshSelector()
        self.validator = AssetValidator()

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _build_product_plus_table(self) -> trimesh.Trimesh:
        product = trimesh.creation.icosphere(radius=0.8, subdivisions=2)
        product.apply_translation([0, 0, 1.0])

        table = trimesh.creation.box(extents=[8.0, 8.0, 0.15])
        table_vertices, table_faces = trimesh.remesh.subdivide_to_size(
            table.vertices,
            table.faces,
            max_edge=0.35,
            max_iter=8,
        )
        table = trimesh.Trimesh(vertices=table_vertices, faces=table_faces, process=False)
        table.apply_translation([0, 0, 0.0])

        return product + table

    def _build_product_plus_diagonal_slab(self) -> trimesh.Trimesh:
        product = trimesh.creation.icosphere(radius=0.8, subdivisions=2)
        product.apply_translation([0, 0, 1.2])

        slab = trimesh.creation.box(extents=[6.0, 0.2, 0.05])
        rot = trimesh.transformations.rotation_matrix(np.deg2rad(38), [0, 1, 0])
        slab.apply_transform(rot)
        slab.apply_translation([0.5, 0.0, 0.4])

        return product + slab

    def _build_fragmented_scene(self) -> trimesh.Trimesh:
        product = trimesh.creation.icosphere(radius=0.7, subdivisions=2)
        product.apply_translation([0, 0, 1.0])

        islands = []
        offsets = [
            [3.0, 0.0, 0.4],
            [-2.8, 0.5, 0.6],
            [1.8, 2.2, 0.5],
            [-1.5, -2.4, 0.7],
        ]
        for off in offsets:
            chunk = trimesh.creation.box(extents=[0.18, 0.18, 0.18])
            chunk.apply_translation(off)
            islands.append(chunk)

        scene = product.copy()
        for island in islands:
            scene = scene + island
        return scene

    def test_isolator_removes_table_plane_and_keeps_product(self):
        scene_mesh = self._build_product_plus_table()

        isolated_mesh, stats = self.isolator.isolate_product(scene_mesh)

        self.assertGreater(stats["removed_planes"], 0)
        self.assertGreaterEqual(stats["removed_plane_face_share"], 0.0)
        self.assertGreaterEqual(stats["removed_plane_vertex_ratio"], 0.0)
        self.assertGreater(stats["component_count"], 0)
        self.assertGreater(stats["selected_component_score"], 0.0)
        self.assertLess(len(isolated_mesh.faces), len(scene_mesh.faces))

        # isolated object should be much more compact than the full table scene
        isolated_extents = isolated_mesh.bounds[1] - isolated_mesh.bounds[0]
        self.assertLess(np.max(isolated_extents), 5.0)

    def test_isolator_penalizes_diagonal_slab_scene(self):
        scene_mesh = self._build_product_plus_diagonal_slab()

        isolated_mesh, stats = self.isolator.isolate_product(scene_mesh)

        self.assertGreater(stats["component_count"], 0)
        self.assertIn("flatness_score", stats)
        self.assertIn("compactness_score", stats)
        self.assertIn("selected_component_score", stats)
        self.assertGreater(stats["selected_component_score"], 0.0)

        # final mesh should still be bounded and product-like rather than huge slab-like
        final_extents = isolated_mesh.bounds[1] - isolated_mesh.bounds[0]
        self.assertLess(np.max(final_extents), 5.0)

    def test_validator_flags_fragmented_scene(self):
        """
        Validator should not happily pass a clearly fragmented / contaminated scene.
        """
        scene_mesh = self._build_fragmented_scene()
        isolated_mesh, iso_stats = self.isolator.isolate_product(scene_mesh)

        asset_data = {
            "poly_count": int(len(isolated_mesh.faces)),
            "texture_status": "missing",
            "bbox": {
                "x": float(isolated_mesh.bounds[1][0] - isolated_mesh.bounds[0][0]),
                "y": float(isolated_mesh.bounds[1][1] - isolated_mesh.bounds[0][1]),
                "z": float(isolated_mesh.bounds[1][2] - isolated_mesh.bounds[0][2]),
            },
            "ground_offset": 0.0,
            "cleanup_stats": {"isolation": iso_stats},
            "texture_path_exists": False,
            "has_uv": False,
            "has_material": False,
            "texture_applied_successfully": False,
        }

        report = self.validator.validate("fragmented_asset", asset_data)

        self.assertIn(report.final_decision, {"review", "fail"})
        self.assertGreaterEqual(report.component_count, 1)
        self.assertGreaterEqual(report.contamination_score, 0.0)
        self.assertTrue(
            "component_count" in report.contamination_report
            or "plane_contamination" in report.contamination_report
            or "component_share" in report.contamination_report
        )

    def test_mesh_selector_prefers_compact_product_over_flat_slab(self):
        """
        Given two candidates, selector should prefer compact product-like mesh over slab-like mesh.
        """
        compact_path = self.test_dir / "compact_product.obj"
        slab_path = self.test_dir / "flat_slab.obj"

        compact = trimesh.creation.icosphere(radius=0.9, subdivisions=2)
        compact.apply_translation([0, 0, 1.0])
        compact.export(compact_path)

        slab = trimesh.creation.box(extents=[8.0, 8.0, 0.08])
        slab.apply_translation([0, 0, 0.0])
        slab.export(slab_path)

        best = self.selector.select_best_mesh([str(slab_path), str(compact_path)])
        self.assertEqual(best, str(compact_path))

    def test_mesh_selector_scores_product_higher_than_fragmented_candidate(self):
        product_path = self.test_dir / "product.obj"
        fragmented_path = self.test_dir / "fragmented.obj"

        product = trimesh.creation.icosphere(radius=0.9, subdivisions=2)
        product.apply_translation([0, 0, 1.0])
        product.export(product_path)

        fragmented = self._build_fragmented_scene()
        fragmented.export(fragmented_path)

        best = self.selector.select_best_mesh([str(fragmented_path), str(product_path)])
        self.assertEqual(best, str(product_path))


if __name__ == "__main__":
    unittest.main()
