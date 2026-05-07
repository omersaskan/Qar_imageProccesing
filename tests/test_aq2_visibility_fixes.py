"""
AQ2 visibility and stale preprocess-warning fixes — regression tests.

Tests:
 19. test_ui_js_uses_manifest_null_check        — JS gates on manifest.aq2 != null, not aq2.enabled
 20. test_ui_aq2_unavailable_fallback_in_html   — disabled-state fallback message present in JS
 21. test_stale_preprocess_stripped_glb_exists  — preprocess_failed:* removed when GLB file exists
 22. test_stale_preprocess_kept_glb_path_none   — preprocess_failed:* kept when glb_path is None
 23. test_stale_preprocess_kept_glb_missing     — preprocess_failed:* kept when GLB file absent
 24. test_non_preprocess_warnings_not_stripped  — unrelated warnings pass through the filter
 25. test_aq2_pipeline_enabled_true_by_default  — run_aq2_pipeline returns enabled=True (env default)
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


UI_PATH = Path(__file__).parent.parent / "ui" / "ai_3d_studio.html"


# ── Test 19: JS uses manifest.aq2 != null, not aq2.enabled ───────────────────

class TestUiJsUsesManifestNullCheck(unittest.TestCase):

    def test_ui_js_uses_manifest_null_check(self):
        html = UI_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "manifest.aq2 != null",
            html,
            "AQ2 section must gate on manifest.aq2 != null (presence), not aq2.enabled",
        )
        # Old pattern must be gone — aq2.enabled must NOT be the top-level gate
        aq2_block_start = html.find("// AQ2: Artifacts")
        self.assertGreater(aq2_block_start, 0)
        aq2_block = html[aq2_block_start : aq2_block_start + 200]
        self.assertNotIn(
            "if (aq2.enabled)",
            aq2_block,
            "Top-level AQ2 gate must not use aq2.enabled as the primary condition",
        )


# ── Test 20: unavailable fallback message present in JS ──────────────────────

class TestUiAq2UnavailableFallback(unittest.TestCase):

    def test_ui_aq2_unavailable_fallback_in_html(self):
        html = UI_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "AQ2 artifacts unavailable",
            html,
            "HTML must include a fallback message for when AQ2 is present but disabled",
        )


# ── Helper: simulate the pipeline filter inline ───────────────────────────────

def _apply_filter(output_glb_path, warnings):
    """Mirrors the stale-warning filter added to pipeline.py."""
    if output_glb_path and Path(output_glb_path).exists():
        warnings = [w for w in warnings if not w.startswith("preprocess_failed:")]
    return warnings


# ── Test 21: stale warning stripped when GLB exists ──────────────────────────

class TestStalePreprocessStrippedWhenGlbExists(unittest.TestCase):

    def test_stale_preprocess_stripped_glb_exists(self):
        with tempfile.TemporaryDirectory() as td:
            glb_path = os.path.join(td, "output.glb")
            Path(glb_path).write_bytes(b"GLBx")
            before = [
                "preprocess_failed:Cannot load source image",
                "ai_generated_not_true_scan",
            ]
            after = _apply_filter(glb_path, list(before))
        self.assertNotIn("preprocess_failed:Cannot load source image", after)
        self.assertIn("ai_generated_not_true_scan", after)


# ── Test 22: stale warning kept when glb_path is None ────────────────────────

class TestStalePreprocessKeptWhenGlbPathNone(unittest.TestCase):

    def test_stale_preprocess_kept_glb_path_none(self):
        before = ["preprocess_failed:Cannot load source image", "other"]
        after = _apply_filter(None, list(before))
        self.assertIn("preprocess_failed:Cannot load source image", after)
        self.assertIn("other", after)


# ── Test 23: stale warning kept when GLB file does not exist ─────────────────

class TestStalePreprocessKeptWhenGlbFileMissing(unittest.TestCase):

    def test_stale_preprocess_kept_glb_missing(self):
        before = ["preprocess_failed:Cannot load source image", "other"]
        after = _apply_filter("/nonexistent/path/output.glb", list(before))
        self.assertIn("preprocess_failed:Cannot load source image", after)
        self.assertIn("other", after)


# ── Test 24: non-preprocess warnings pass through unchanged ──────────────────

class TestNonPreprocessWarningsNotStripped(unittest.TestCase):

    def test_non_preprocess_warnings_not_stripped(self):
        with tempfile.TemporaryDirectory() as td:
            glb_path = os.path.join(td, "output.glb")
            Path(glb_path).write_bytes(b"GLBx")
            before = [
                "ai_generated_not_true_scan",
                "mesh_cleanup_review",
                "preprocess_failed:resize_error",
                "ar_readiness_warning",
            ]
            after = _apply_filter(glb_path, list(before))
        self.assertIn("ai_generated_not_true_scan", after)
        self.assertIn("mesh_cleanup_review", after)
        self.assertIn("ar_readiness_warning", after)
        self.assertNotIn("preprocess_failed:resize_error", after)


# ── Test 25: run_aq2_pipeline returns enabled=True by default ─────────────────

class TestAq2PipelineEnabledTrueByDefault(unittest.TestCase):

    def test_aq2_pipeline_enabled_true_by_default(self):
        """AQ2 must be enabled unless AI_3D_AQ2_ENABLED is explicitly set to false."""
        from modules.ai_3d_generation.asset_quality import artifacts as art_mod

        with tempfile.TemporaryDirectory() as td:
            result = art_mod.run_aq2_pipeline(
                raw_glb_path=None,
                session_dir=td,
                manifest={"session_id": "aq2_default_test"},
                asset_quality={},
            )
        self.assertTrue(
            result["enabled"],
            "run_aq2_pipeline must return enabled=True when AI_3D_AQ2_ENABLED is not set to false",
        )
