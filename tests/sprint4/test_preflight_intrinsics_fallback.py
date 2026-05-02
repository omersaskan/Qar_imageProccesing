"""Sprint 4 — preflight + intrinsics_cache + fallback_ladder tests."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from modules.reconstruction_engine.fallback_ladder import (
    LADDER_LOW_THREAD_TEXTURE, LADDER_SAFE_LOW_RESOLUTION,
    classify_error, get_default_ladder, pick_next_preset,
)
from modules.reconstruction_engine.intrinsics_cache import (
    IntrinsicsCache, build_cache_key, disabled_lookup,
)
from modules.reconstruction_engine.reconstruction_preflight import (
    PreflightDecision, evaluate_preflight,
)
from modules.reconstruction_engine.reconstruction_preset_resolver import (
    PRESET_NAME_BASELINE, PRESET_NAME_PROFILE_SAFE,
)


# ─── Preflight ──────────────────────────────────────────────────────────────

def _write_frame(path: Path, w: int = 64, h: int = 64):
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_preflight_empty_keyframes_rejects():
    rep = evaluate_preflight([])
    assert rep.decision == PreflightDecision.REJECT
    assert any("no selected" in r for r in rep.reasons)


def test_preflight_below_hard_min_rejects(tmp_path):
    p = tmp_path / "f.jpg"
    _write_frame(p)
    rep = evaluate_preflight([str(p)])
    assert rep.decision == PreflightDecision.REJECT
    assert rep.selected_count == 1


def test_preflight_below_review_threshold_reviews(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"f_{i}.jpg"
        _write_frame(p)
        paths.append(str(p))
    rep = evaluate_preflight(paths)
    assert rep.decision == PreflightDecision.REVIEW


def test_preflight_passes_with_solid_capture(tmp_path):
    paths = []
    for i in range(20):
        p = tmp_path / f"f_{i}.jpg"
        _write_frame(p)
        paths.append(str(p))
    gate = {
        "matrix_3x8": [[1, 1, 1, 1, 1, 1, 0, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0],
                       [1, 1, 1, 1, 1, 1, 0, 0]],
        "blur": {"median_score": 200.0},
        "azimuth": {"frame_count": 20, "max_consecutive_static_frames": 2},
    }
    rep = evaluate_preflight(paths, capture_gate=gate)
    assert rep.decision == PreflightDecision.PASS
    assert rep.coverage_ratio > 0.6


def test_preflight_low_blur_reviews(tmp_path):
    paths = []
    for i in range(15):
        p = tmp_path / f"f_{i}.jpg"
        _write_frame(p)
        paths.append(str(p))
    gate = {"blur": {"median_score": 25.0}}  # below review threshold
    rep = evaluate_preflight(paths, capture_gate=gate)
    assert rep.decision in (PreflightDecision.REVIEW, PreflightDecision.REJECT)


def test_preflight_very_blurry_rejects(tmp_path):
    paths = []
    for i in range(15):
        p = tmp_path / f"f_{i}.jpg"
        _write_frame(p)
        paths.append(str(p))
    gate = {"blur": {"median_score": 5.0}}
    rep = evaluate_preflight(paths, capture_gate=gate)
    assert rep.decision == PreflightDecision.REJECT


def test_preflight_dimension_mismatch_rejects(tmp_path):
    paths = []
    # 10 frames at 64×64, 5 at 128×128 → mismatch ratio ≈ 33% > 20%
    for i in range(10):
        p = tmp_path / f"a_{i}.jpg"
        _write_frame(p, w=64, h=64)
        paths.append(str(p))
    for i in range(5):
        p = tmp_path / f"b_{i}.jpg"
        _write_frame(p, w=128, h=128)
        paths.append(str(p))
    rep = evaluate_preflight(paths)
    assert rep.decision == PreflightDecision.REJECT
    assert any("dimension" in r.lower() for r in rep.reasons)


def test_preflight_serializable():
    rep = evaluate_preflight([])
    json.dumps(rep.to_dict())


# ─── Intrinsics cache ────────────────────────────────────────────────────────

def test_cache_key_is_deterministic():
    a = build_cache_key(1920, 1080, "iPhone 15 Pro", focal_mm=6.0)
    b = build_cache_key(1920, 1080, "iphone 15 pro", focal_mm=6.0)
    assert a == b


def test_cache_key_focal_binning():
    # Focal 6.1 and 6.2 round to same 0.5-mm bin (6.0 and 6.0)
    a = build_cache_key(1920, 1080, "iPhone", focal_mm=6.1)
    b = build_cache_key(1920, 1080, "iPhone", focal_mm=6.2)
    assert a == b


def test_cache_miss_then_hit(tmp_path):
    cache = IntrinsicsCache(tmp_path / "intrinsics.json")
    r1 = cache.lookup(1920, 1080, "iPhone 15 Pro", focal_mm=6.0)
    assert r1.status == "miss"
    assert r1.record is not None
    # Second lookup should be a hit
    r2 = cache.lookup(1920, 1080, "iPhone 15 Pro", focal_mm=6.0)
    assert r2.status == "hit"
    assert r2.record.use_count >= 1


def test_cache_atomic_write(tmp_path):
    p = tmp_path / "intrinsics.json"
    cache = IntrinsicsCache(p)
    cache.lookup(1920, 1080, "iPhone")
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert "entries" in data
    # No leftover .tmp file
    assert not (tmp_path / "intrinsics.json.tmp").exists()


def test_disabled_lookup_returns_default():
    res = disabled_lookup(1920, 1080)
    assert res.status == "disabled"
    assert res.record is not None
    assert res.record.fx > 0


def test_cache_corrupt_file_recovers(tmp_path):
    p = tmp_path / "intrinsics.json"
    p.write_text("{ this is not json", encoding="utf-8")
    cache = IntrinsicsCache(p)
    res = cache.lookup(1920, 1080, "iPhone")
    # Should not crash, should have started fresh
    assert res.status == "miss"


# ─── Fallback ladder ─────────────────────────────────────────────────────────

def test_classify_error_native_crash():
    assert classify_error("OpenMVS TextureMesh exit code 3221226505") == "native_crash"


def test_classify_error_oom():
    assert classify_error("CUDA error: out of memory") == "oom"


def test_classify_error_missing_file():
    assert classify_error("OSError: No such file or directory") == "missing_file"


def test_classify_error_unknown():
    assert classify_error(None) == "unknown"
    assert classify_error("") == "unknown"


def test_pick_next_preset_initial_attempt():
    nxt = pick_next_preset(profile=None, error_excerpt=None, attempts_so_far=[])
    assert nxt is not None
    assert nxt.preset_name == PRESET_NAME_PROFILE_SAFE
    assert nxt.triggered_by == "initial"


def test_pick_next_preset_native_crash_jumps():
    nxt = pick_next_preset(
        profile=None,
        error_excerpt="OpenMVS exit code 3221226505",
        attempts_so_far=[],
    )
    assert nxt.preset_name == LADDER_LOW_THREAD_TEXTURE
    assert nxt.triggered_by == "retry_after_native_crash"


def test_pick_next_preset_oom_jumps():
    nxt = pick_next_preset(
        profile=None,
        error_excerpt="CUDA error: out of memory",
        attempts_so_far=[],
    )
    assert nxt.preset_name == LADDER_SAFE_LOW_RESOLUTION
    assert nxt.triggered_by == "retry_after_oom"


def test_pick_next_preset_missing_file_aborts():
    nxt = pick_next_preset(
        profile=None,
        error_excerpt="No such file or directory: /missing/path",
        attempts_so_far=[],
    )
    assert nxt is None


def test_default_ladder_includes_baseline_last():
    ladder = get_default_ladder(None)
    assert ladder[0]["name"] == PRESET_NAME_PROFILE_SAFE
    assert ladder[-1]["name"] == PRESET_NAME_BASELINE
    assert len(ladder) == 5


def test_ladder_exhaustion_returns_none():
    from modules.reconstruction_engine.fallback_ladder import FallbackAttempt
    used = [
        FallbackAttempt(0, PRESET_NAME_PROFILE_SAFE, "initial"),
        FallbackAttempt(1, "safe_high_quality", "retry_after_runtime_error"),
        FallbackAttempt(2, LADDER_SAFE_LOW_RESOLUTION, "retry_after_runtime_error"),
        FallbackAttempt(3, LADDER_LOW_THREAD_TEXTURE, "retry_after_runtime_error"),
        FallbackAttempt(4, PRESET_NAME_BASELINE, "retry_after_runtime_error"),
    ]
    nxt = pick_next_preset(profile=None, error_excerpt="generic", attempts_so_far=used)
    assert nxt is None
