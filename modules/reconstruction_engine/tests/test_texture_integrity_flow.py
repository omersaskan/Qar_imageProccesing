import unittest
import shutil
from pathlib import Path
import sys

import numpy as np
import trimesh

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from modules.export_pipeline.glb_exporter import GLBExporter
from modules.qa_validation.validator import AssetValidator


def write_dummy_png(path: Path):
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff? \x00\x05\xfe\x02\xfe\xdcD\x05\x13"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def make_uv_mesh_obj(path: Path):
    """
    Create a simple mesh with UVs and export as OBJ.
    """
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

    verts = mesh.vertices.copy()
    x = verts[:, 0]
    y = verts[:, 1]

    x_span = max(float(x.max() - x.min()), 1e-8)
    y_span = max(float(y.max() - y.min()), 1e-8)

    uv = np.zeros((len(verts), 2), dtype=np.float64)
    uv[:, 0] = (x - x.min()) / x_span
    uv[:, 1] = (y - y.min()) / y_span

    mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
    mesh.export(path)


def make_plain_mesh_obj(path: Path):
    """
    Create a simple mesh without UVs and export as OBJ.
    """
    mesh = trimesh.creation.icosphere(radius=0.5)
    mesh.export(path)


class TestTextureIntegrityFlow(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("data/test_texture_integrity")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.exporter = GLBExporter()
        self.validator = AssetValidator()

        self.texture_path = self.test_dir / "test_texture.png"
        write_dummy_png(self.texture_path)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_export_with_uv_and_texture(self):
        source_mesh = self.test_dir / "uv_mesh.obj"
        output_glb = self.test_dir / "uv_mesh.glb"

        make_uv_mesh_obj(source_mesh)

        result = self.exporter.export(
            mesh_path=str(source_mesh),
            output_path=str(output_glb),
            profile_name="standard",
            texture_path=str(self.texture_path),
            metadata=None,
        )

        self.assertTrue(output_glb.exists())
        self.assertGreater(result["filesize"], 0)
        self.assertTrue(result["has_uv"])
        self.assertEqual(result["used_texture_path"], str(self.texture_path))
        self.assertTrue(result["texture_applied_successfully"])

    def test_export_without_uv_reports_degraded_texture(self):
        source_mesh = self.test_dir / "plain_mesh.obj"
        output_glb = self.test_dir / "plain_mesh.glb"

        make_plain_mesh_obj(source_mesh)

        result = self.exporter.export(
            mesh_path=str(source_mesh),
            output_path=str(output_glb),
            profile_name="standard",
            texture_path=str(self.texture_path),
            metadata=None,
        )

        self.assertTrue(output_glb.exists())
        self.assertGreater(result["filesize"], 0)
        self.assertFalse(result["has_uv"])
        self.assertEqual(result["used_texture_path"], str(self.texture_path))
        self.assertFalse(result["texture_applied_successfully"])

    def test_validator_reviews_texture_without_uv(self):
        """
        Texture file exists, but cleaned mesh has no UV/material.
        Expected behavior: not hard crash, but review/fail style integrity warning.
        """
        asset_data = {
            "poly_count": 1200,
            "texture_status": "degraded",
            "bbox": {"x": 10.0, "y": 10.0, "z": 5.0},
            "ground_offset": 0.0,
            "cleanup_stats": {
                "isolation": {
                    "component_count": 1,
                    "initial_faces": 1000,
                    "final_faces": 900,
                    "removed_plane_face_share": 0.0,
                    "removed_plane_vertex_ratio": 0.0,
                    "compactness_score": 0.5,
                    "selected_component_score": 0.8,
                }
            },
            "texture_path_exists": True,
            "has_uv": False,
            "has_material": False,
            "texture_applied_successfully": False,
        }

        report = self.validator.validate("asset_texture_degraded", asset_data)

        self.assertIn(report.final_decision, {"review", "fail"})
        self.assertIn("texture_uv_integrity", report.contamination_report)
        self.assertIn("texture_application", report.contamination_report)
        self.assertGreaterEqual(report.contamination_score, 0.0)

    def test_validator_passes_texture_with_uv(self):
        asset_data = {
            "poly_count": 1200,
            "texture_status": "complete",
            "bbox": {"x": 10.0, "y": 10.0, "z": 5.0},
            "ground_offset": 0.0,
            "cleanup_stats": {
                "isolation": {
                    "component_count": 1,
                    "initial_faces": 1000,
                    "final_faces": 950,
                    "removed_plane_face_share": 0.0,
                    "removed_plane_vertex_ratio": 0.0,
                    "compactness_score": 0.6,
                    "selected_component_score": 0.9,
                }
            },
            "texture_path_exists": True,
            "has_uv": True,
            "has_material": True,
            "texture_applied_successfully": True,
        }

        report = self.validator.validate("asset_texture_ok", asset_data)

        # Could still become review if other thresholds change later,
        # but texture integrity itself should not be the reason.
        self.assertEqual(report.contamination_report.get("texture_uv_integrity"), "pass")
        self.assertEqual(report.contamination_report.get("texture_application"), "pass")
        self.assertEqual(report.contamination_report.get("material_integrity"), "pass")


if __name__ == "__main__":
    unittest.main()