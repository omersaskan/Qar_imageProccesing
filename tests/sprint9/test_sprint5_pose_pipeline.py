"""Sprint 9 — Sprint 5 pose-backed coverage pipeline tests."""
from __future__ import annotations

import math
import textwrap
from pathlib import Path

import pytest

# ─────────────────────────── colmap_sparse_parser ───────────────────────────

from modules.reconstruction_engine.colmap_sparse_parser import (
    parse_cameras_txt,
    parse_images_txt,
    find_sparse_model_dir,
    load_sparse_model,
)


def write_cameras_txt(path: Path, content: str) -> Path:
    p = path / "cameras.txt"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def write_images_txt(path: Path, content: str) -> Path:
    p = path / "images.txt"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def test_parse_cameras_txt_basic(tmp_path):
    write_cameras_txt(tmp_path, """\
        # comment
        1 PINHOLE 1920 1080 960.0 540.0 960.0 540.0
        2 SIMPLE_RADIAL 640 480 400.0 320.0 240.0 0.01
    """)
    cams = parse_cameras_txt(tmp_path / "cameras.txt")
    assert len(cams) == 2
    assert cams[1]["model"] == "PINHOLE"
    assert cams[1]["width"] == 1920
    assert cams[2]["height"] == 480
    assert len(cams[2]["params"]) == 4


def test_parse_cameras_txt_missing_returns_empty(tmp_path):
    cams = parse_cameras_txt(tmp_path / "cameras.txt")
    assert cams == {}


def test_parse_cameras_txt_malformed_lines_skipped(tmp_path):
    write_cameras_txt(tmp_path, """\
        1 PINHOLE 1920 1080 960.0 540.0 960.0 540.0
        GARBAGE LINE
    """)
    cams = parse_cameras_txt(tmp_path / "cameras.txt")
    assert len(cams) == 1


def _make_image_line(img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name):
    header = f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {name}"
    points = "0.0 0.0 -1"
    return f"{header}\n{points}\n"


def test_parse_images_txt_basic(tmp_path):
    content = "# comment\n" + _make_image_line(1, 1, 0, 0, 0, 0, 0, 5, 1, "img001.jpg")
    content += _make_image_line(2, 0.707, 0.707, 0, 0, 1, 0, 5, 1, "img002.jpg")
    (tmp_path / "images.txt").write_text(content, encoding="utf-8")
    imgs = parse_images_txt(tmp_path / "images.txt")
    assert len(imgs) == 2
    assert imgs[0]["name"] == "img001.jpg"
    assert imgs[0]["qvec"] == [1.0, 0.0, 0.0, 0.0]
    assert imgs[1]["image_id"] == 2


def test_parse_images_txt_missing_returns_empty(tmp_path):
    assert parse_images_txt(tmp_path / "images.txt") == []


def test_find_sparse_model_dir_top_level(tmp_path):
    (tmp_path / "cameras.txt").write_text("", encoding="utf-8")
    assert find_sparse_model_dir(tmp_path) == tmp_path


def test_find_sparse_model_dir_numbered_subdir(tmp_path):
    sub = tmp_path / "0"
    sub.mkdir()
    (sub / "cameras.txt").write_text("", encoding="utf-8")
    assert find_sparse_model_dir(tmp_path) == sub


def test_find_sparse_model_dir_none_when_absent(tmp_path):
    assert find_sparse_model_dir(tmp_path) is None


def test_load_sparse_model_missing_sparse_dir(tmp_path):
    cams, imgs, model_dir = load_sparse_model(tmp_path)
    assert cams == {} and imgs == [] and model_dir is None


def test_load_sparse_model_with_files(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    write_cameras_txt(sparse, "1 PINHOLE 1920 1080 960.0 540.0 960.0 540.0\n")
    content = _make_image_line(1, 1, 0, 0, 0, 0, 0, 3, 1, "a.jpg")
    (sparse / "images.txt").write_text(content, encoding="utf-8")
    cams, imgs, model_dir = load_sparse_model(tmp_path)
    assert len(cams) == 1
    assert len(imgs) == 1
    assert model_dir == sparse


# ─────────────────────────── pose_geometry ───────────────────────────

from modules.reconstruction_engine.pose_geometry import (
    qvec_to_rotation_matrix,
    camera_center_from_pose,
    cartesian_to_spherical,
    compute_scene_centroid,
    centres_relative_to_centroid,
)


def test_qvec_identity_gives_identity_matrix():
    R = qvec_to_rotation_matrix([1, 0, 0, 0])
    assert abs(R[0][0] - 1) < 1e-9
    assert abs(R[1][1] - 1) < 1e-9
    assert abs(R[2][2] - 1) < 1e-9


def test_qvec_normalised_internally():
    # Unnormalised quaternion should give same result as normalised
    R1 = qvec_to_rotation_matrix([2, 0, 0, 0])
    R2 = qvec_to_rotation_matrix([1, 0, 0, 0])
    for i in range(3):
        for j in range(3):
            assert abs(R1[i][j] - R2[i][j]) < 1e-9


def test_camera_center_identity_rotation_no_translation():
    cx, cy, cz = camera_center_from_pose([1, 0, 0, 0], [0, 0, 0])
    assert abs(cx) < 1e-9 and abs(cy) < 1e-9 and abs(cz) < 1e-9


def test_camera_center_pure_translation():
    # Identity rotation, translation [tx, ty, tz] → centre = [-tx, -ty, -tz]
    cx, cy, cz = camera_center_from_pose([1, 0, 0, 0], [1, 2, 3])
    assert abs(cx - (-1)) < 1e-9
    assert abs(cy - (-2)) < 1e-9
    assert abs(cz - (-3)) < 1e-9


def test_cartesian_to_spherical_on_x_axis():
    r, az, el = cartesian_to_spherical(5, 0, 0)
    assert abs(r - 5) < 1e-9
    assert abs(az - 0) < 1e-9
    assert abs(el - 0) < 1e-9


def test_cartesian_to_spherical_top():
    r, az, el = cartesian_to_spherical(0, 0, 3)
    assert abs(r - 3) < 1e-9
    assert abs(el - 90) < 1e-6


def test_compute_scene_centroid_empty():
    cx, cy, cz = compute_scene_centroid([])
    assert cx == cy == cz == 0.0


def test_compute_scene_centroid_basic():
    pts = [(1, 0, 0), (-1, 0, 0), (0, 2, 0)]
    cx, cy, cz = compute_scene_centroid(pts)
    assert abs(cx) < 1e-9
    assert abs(cy - 2/3) < 1e-9


def test_centres_relative_to_centroid_shifts():
    pts = [(1, 1, 1), (3, 3, 3)]
    centroid = (2, 2, 2)
    rel = centres_relative_to_centroid(pts, centroid)
    assert rel[0] == (-1, -1, -1)
    assert rel[1] == (1, 1, 1)


# ─────────────────────────── pose_coverage_matrix ───────────────────────────

from modules.reconstruction_engine.pose_coverage_matrix import (
    build_coverage_matrix,
    coverage_from_attempt_dir,
    _azimuth_sector,
    _elevation_band,
    _unavailable,
)


def _make_image_dict(qvec, tvec, img_id=1, name="img.jpg"):
    return {"image_id": img_id, "qvec": qvec, "tvec": tvec, "camera_id": 1, "name": name}


def test_build_coverage_matrix_empty_images():
    result = build_coverage_matrix([])
    assert result["status"] == "unavailable"


def test_build_coverage_matrix_single_camera():
    img = _make_image_dict([1, 0, 0, 0], [0, 0, 3])
    result = build_coverage_matrix([img])
    assert result["status"] == "ok"
    assert result["registered_count"] == 1
    assert result["coverage_ratio"] >= 0


def test_build_coverage_matrix_full_orbit_coverage():
    # 8 cameras evenly spaced around +XY plane, all at z=0 centre offset
    images = []
    for i in range(8):
        angle = i * (2 * math.pi / 8)
        tx = -math.cos(angle) * 5   # tvec = -R @ centre → identity R
        ty = -math.sin(angle) * 5
        img = _make_image_dict([1, 0, 0, 0], [tx, ty, 0], img_id=i+1)
        images.append(img)
    result = build_coverage_matrix(images)
    assert result["status"] == "ok"
    assert result["azimuth_span_degrees"] > 300  # near full orbit
    assert result["covered_cells"] >= 8


def test_azimuth_sector_wraps_correctly():
    assert _azimuth_sector(0) == 0
    assert _azimuth_sector(45) == 1
    assert _azimuth_sector(359) == 7
    assert _azimuth_sector(-1) == 7


def test_elevation_band_boundaries():
    assert _elevation_band(-90) == 0   # below low → clamped low
    assert _elevation_band(0) == 1     # mid band
    assert _elevation_band(60) == 2    # high band


def test_unavailable_block_has_required_keys():
    block = _unavailable("test reason")
    for key in ("status", "reason", "coverage_ratio", "total_cells"):
        assert key in block
    assert block["status"] == "unavailable"


def test_coverage_from_attempt_dir_missing_dir(tmp_path):
    result = coverage_from_attempt_dir(tmp_path / "nonexistent")
    assert result["status"] == "unavailable"


# ─────────────────────────── orbit_validation ───────────────────────────

from modules.reconstruction_engine.orbit_validation import (
    validate_orbit,
    OrbitThresholds,
    OrbitValidationResult,
)


def _good_coverage():
    return {
        "status": "ok",
        "coverage_ratio": 0.60,
        "azimuth_span_degrees": 350.0,
        "elevation_spread_degrees": 45.0,
        "registered_count": 30,
    }


def test_orbit_validation_pass_on_good_coverage():
    result = validate_orbit(_good_coverage(), total_input_frames=30)
    assert result.verdict == "pass"
    assert result.registered_ratio == pytest.approx(1.0)


def test_orbit_validation_review_on_low_elevation():
    cov = {**_good_coverage(), "elevation_spread_degrees": 10.0}
    result = validate_orbit(cov)
    assert result.verdict in ("review", "fail")
    assert any("elevation" in r for r in result.reasons)


def test_orbit_validation_fail_on_poor_azimuth():
    cov = {**_good_coverage(), "azimuth_span_degrees": 100.0}
    result = validate_orbit(cov, thresholds=OrbitThresholds(min_azimuth_span_degrees=270))
    assert result.verdict == "fail"


def test_orbit_validation_unavailable_on_missing_status():
    result = validate_orbit({"status": "unavailable", "reason": "no sparse"})
    assert result.verdict == "unavailable"


def test_orbit_validation_low_registered_ratio_review():
    cov = {**_good_coverage(), "registered_count": 10}
    result = validate_orbit(cov, total_input_frames=30)
    # 10/30 = 0.33 < 0.70 threshold
    assert result.verdict in ("review", "fail")
    assert result.registered_ratio == pytest.approx(1/3)


def test_orbit_validation_result_to_dict_serialisable():
    import json
    result = validate_orbit(_good_coverage())
    d = result.to_dict()
    json.dumps(d)  # must not raise


# ─────────────────────────── pose_feedback ───────────────────────────

from modules.reconstruction_engine.pose_feedback import generate_pose_feedback


def test_pose_feedback_missing_attempt_dir(tmp_path):
    block = generate_pose_feedback(tmp_path / "no_such_dir", input_frame_count=10)
    assert block["status"] == "unavailable"


def test_pose_feedback_missing_sparse(tmp_path):
    block = generate_pose_feedback(tmp_path, input_frame_count=10)
    assert block["status"] == "unavailable"
    assert "sparse_model_dir" in block
    assert "orbit_validation" in block


def test_pose_feedback_with_real_sparse(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 1920 1080 960.0 540.0 960.0 540.0\n", encoding="utf-8"
    )
    # 8 cameras forming a ring
    lines = "# images\n"
    for i in range(8):
        angle = i * (2 * math.pi / 8)
        tx = -math.cos(angle) * 5
        ty = -math.sin(angle) * 5
        lines += f"{i+1} 1.0 0.0 0.0 0.0 {tx:.4f} {ty:.4f} 0.0 1 img{i:03d}.jpg\n"
        lines += "0.0 0.0 -1\n"
    (sparse / "images.txt").write_text(lines, encoding="utf-8")

    block = generate_pose_feedback(tmp_path, input_frame_count=8)
    assert block["status"] == "ok"
    assert block["coverage"]["registered_count"] == 8
    assert block["orbit_validation"]["verdict"] in ("pass", "review", "fail")
