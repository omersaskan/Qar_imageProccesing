from pathlib import Path

import pytest

from modules.export_pipeline.usdz_exporter import USDZExporter


def test_usdz_exporter_blocks_placeholder_output_by_default(tmp_path):
    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text(
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n",
        encoding="utf-8",
    )

    exporter = USDZExporter()

    with pytest.raises(RuntimeError, match="disabled by default"):
        exporter.export(str(mesh_path), str(tmp_path / "mesh.usdz"))


def test_usdz_exporter_placeholder_requires_explicit_opt_in(tmp_path, monkeypatch):
    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text(
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "f 1 2 3\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "mesh.usdz"

    monkeypatch.setenv("MESHYSIZ_ENABLE_PLACEHOLDER_USDZ", "true")
    exporter = USDZExporter()
    result = exporter.export(str(mesh_path), str(output_path))

    assert output_path.exists()
    assert result["stub"] is True
