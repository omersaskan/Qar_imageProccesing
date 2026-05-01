"""Sprint 1 — camera_model_resolver tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from modules.reconstruction_engine.camera_model_resolver import (
    ColmapCameraModel,
    _hfov_from_focal_mm,
    _model_for_hfov,
    _parse_focal_mm,
    resolve_for_frames,
)


def test_hfov_from_focal():
    # 35mm-equivalent focal → horizontal FOV
    # 26mm equiv → ~69.4° (iPhone main, classic ~73° rating uses different sensor)
    assert 67 < _hfov_from_focal_mm(26) < 72
    # 13mm equiv → ~108° (ultrawide)
    assert 100 < _hfov_from_focal_mm(13) < 115
    # 50mm → ~40° (portrait)
    assert 35 < _hfov_from_focal_mm(50) < 45


def test_model_for_hfov_buckets():
    assert _model_for_hfov(60) == ColmapCameraModel.RADIAL
    assert _model_for_hfov(74.9) == ColmapCameraModel.RADIAL
    assert _model_for_hfov(85) == ColmapCameraModel.OPENCV
    assert _model_for_hfov(105) == ColmapCameraModel.OPENCV_FISHEYE
    assert _model_for_hfov(0) == ColmapCameraModel.RADIAL


def test_parse_focal_mm_handles_ifd_rational():
    exif = {"FocalLength": IFDRational(26, 1)}
    assert _parse_focal_mm(exif) == 26.0


def test_parse_focal_mm_handles_tuple():
    exif = {"FocalLength": (26, 1)}
    assert _parse_focal_mm(exif) == 26.0


def test_parse_focal_mm_prefers_35mm_field():
    # If FocalLengthIn35mmFilm exists it wins
    exif = {"FocalLength": (5, 1), "FocalLengthIn35mmFilm": 26}
    assert _parse_focal_mm(exif) == 26.0


def test_resolve_no_frames_falls_back_to_default():
    decision = resolve_for_frames([])
    assert decision.model == ColmapCameraModel.RADIAL
    assert decision.source == "default"


def test_resolve_unreadable_paths_falls_back():
    decision = resolve_for_frames(["/no/such/file.jpg"])
    assert decision.model == ColmapCameraModel.RADIAL
    assert decision.source == "default"


def _write_jpeg_with_exif(path: Path, focal_mm: int, make: str = "", model: str = ""):
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    exif = img.getexif()
    # FocalLength tag 0x920A; Make 0x010F; Model 0x0110
    exif[0x920A] = (focal_mm, 1)
    if make:
        exif[0x010F] = make
    if model:
        exif[0x0110] = model
    img.save(path, "JPEG", exif=exif)


def test_resolve_iphone_main_camera_db_match(tmp_path):
    p = tmp_path / "iphone_main.jpg"
    _write_jpeg_with_exif(p, focal_mm=6, make="Apple", model="iPhone 15 Pro")
    decision = resolve_for_frames([str(p)])
    # iPhone main 5–7.5mm focal → RADIAL via device DB
    assert decision.model == ColmapCameraModel.RADIAL
    assert decision.source == "device_db"
    assert "Apple iPhone 15 Pro" in decision.devices_seen


def test_resolve_iphone_ultrawide_db_match(tmp_path):
    p = tmp_path / "iphone_ultrawide.jpg"
    _write_jpeg_with_exif(p, focal_mm=2, make="Apple", model="iPhone 15 Pro")
    decision = resolve_for_frames([str(p)])
    # iPhone 1.5–3mm → OPENCV_FISHEYE via device DB
    assert decision.model == ColmapCameraModel.OPENCV_FISHEYE
    assert decision.source == "device_db"


def test_resolve_unknown_device_uses_hfov(tmp_path):
    # No make/model → falls through to HFOV computation
    p = tmp_path / "generic.jpg"
    _write_jpeg_with_exif(p, focal_mm=13)
    decision = resolve_for_frames([str(p)])
    assert decision.model == ColmapCameraModel.OPENCV_FISHEYE
    assert decision.source == "exif_hfov"
    assert decision.estimated_hfov_deg > 100


def test_resolve_filename_hint_when_no_exif(tmp_path):
    p = tmp_path / "frame_ultrawide_001.jpg"
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    img.save(p, "JPEG")  # no EXIF
    decision = resolve_for_frames([str(p)])
    assert decision.model == ColmapCameraModel.OPENCV_FISHEYE
    assert decision.source == "filename_hint"


def test_resolve_mixed_lens_capture_steps_up_to_opencv(tmp_path):
    # Two frames with very different HFOVs and no DB hit
    p1 = tmp_path / "narrow.jpg"
    p2 = tmp_path / "wide.jpg"
    _write_jpeg_with_exif(p1, focal_mm=50)  # ~40°
    _write_jpeg_with_exif(p2, focal_mm=15)  # ~95°
    decision = resolve_for_frames([str(p1), str(p2)])
    assert decision.model == ColmapCameraModel.OPENCV
    assert "mixed-lens" in decision.reason
