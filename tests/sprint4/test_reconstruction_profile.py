"""Sprint 4 — reconstruction_profile derivation tests."""
from __future__ import annotations

import pytest

from modules.reconstruction_engine.reconstruction_profile import (
    MaterialProfile,
    MotionProfile,
    SceneProfile,
    SizeProfile,
    derive_profile,
)


def test_empty_inputs_returns_unknown_profile():
    rep = derive_profile()
    assert rep.material_profile == MaterialProfile.UNKNOWN
    assert rep.size_profile == SizeProfile.UNKNOWN
    assert rep.scene_profile == SceneProfile.UNKNOWN
    assert rep.motion_profile == MotionProfile.UNKNOWN
    assert rep.confidence == 0.0


def test_glossy_material_from_capture_profile():
    em = {"capture_profile": {"material_hint": "glossy", "size_class": "small"}}
    rep = derive_profile(em)
    assert rep.material_profile == MaterialProfile.GLOSSY
    assert rep.size_profile == SizeProfile.SMALL


def test_metallic_maps_to_glossy_class():
    em = {"capture_profile": {"material_hint": "metallic", "size_class": "large"}}
    rep = derive_profile(em)
    assert rep.material_profile == MaterialProfile.GLOSSY
    assert rep.size_profile == SizeProfile.LARGE


def test_transparent_maps_to_transparent_reflective():
    em = {"capture_profile": {"material_hint": "transparent", "size_class": "medium"}}
    rep = derive_profile(em)
    assert rep.material_profile == MaterialProfile.TRANSPARENT_REFLECTIVE


def test_black_color_low_texture_risk():
    em = {
        "capture_profile": {"material_hint": "opaque", "size_class": "small"},
        "color_profile": {"category": "black"},
        "saved_count": 20,
    }
    rep = derive_profile(em)
    assert rep.material_profile == MaterialProfile.LOW_TEXTURE


def test_low_light_scene_from_dark_color():
    em = {
        "color_profile": {"category": "black", "product_rgb": [10, 12, 15]},
    }
    rep = derive_profile(em)
    assert rep.scene_profile == SceneProfile.LOW_LIGHT


def test_controlled_scene_from_pass_gate():
    em = {
        "capture_gate": {
            "decision": "pass",
            "elevation": {"multi_height_score": 0.66},
            "blur": {"median_score": 200},
        },
    }
    rep = derive_profile(em)
    assert rep.scene_profile == SceneProfile.CONTROLLED


def test_static_poor_motion_profile():
    em = {
        "capture_gate": {
            "azimuth": {"cumulative_orbit_progress": 0.05,
                        "max_consecutive_static_frames": 30,
                        "frame_count": 40},
            "blur": {"burst_ratio": 0.0},
        }
    }
    rep = derive_profile(em)
    assert rep.motion_profile == MotionProfile.STATIC_POOR


def test_fast_motion_from_high_burst_ratio():
    em = {
        "capture_gate": {
            "azimuth": {"cumulative_orbit_progress": 0.50,
                        "max_consecutive_static_frames": 0,
                        "frame_count": 30},
            "blur": {"burst_ratio": 0.30},
        }
    }
    rep = derive_profile(em)
    assert rep.motion_profile == MotionProfile.FAST_MOTION


def test_stable_orbit_motion():
    em = {
        "capture_gate": {
            "azimuth": {"cumulative_orbit_progress": 0.85,
                        "max_consecutive_static_frames": 0,
                        "frame_count": 30},
            "blur": {"burst_ratio": 0.05},
        }
    }
    rep = derive_profile(em)
    assert rep.motion_profile == MotionProfile.STABLE_ORBIT


def test_confidence_increases_with_signals():
    em = {
        "capture_profile": {"material_hint": "opaque", "size_class": "small"},
        "color_profile": {"category": "mid", "product_rgb": [128, 128, 128]},
        "capture_gate": {
            "decision": "pass",
            "elevation": {"multi_height_score": 0.66},
            "blur": {"median_score": 100, "burst_ratio": 0.05},
            "azimuth": {"cumulative_orbit_progress": 0.85,
                        "max_consecutive_static_frames": 0, "frame_count": 30},
        },
    }
    rep = derive_profile(em)
    assert rep.confidence == 1.0
    assert len(rep.signals_used) == 4


def test_to_dict_serializable():
    rep = derive_profile()
    import json
    d = rep.to_dict()
    json.dumps(d)
    assert d["material_profile"] == "unknown"
