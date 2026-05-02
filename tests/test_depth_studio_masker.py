"""
Tests for Depth Studio subject masker — connected component filtering.

Synthetic scenario: small object (key) on large textured table.
  - Table component: bottom-border-touching, large area → rejected
  - Key component: near center, small-medium area → selected
"""
import numpy as np
import pytest


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_synthetic_mask_and_depth(h=720, w=720):
    """
    Returns (binary_mask, depth_norm) with two components:
      1. Table surface: bottom 40% of image, full width  (background)
      2. Key object: 80x60 rectangle near center         (subject)
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    # Table component — bottom 40%, full width
    table_y0 = int(h * 0.60)
    mask[table_y0:h, 0:w] = 255

    # Key object — centred
    ky0, kx0 = int(h * 0.35), int(w * 0.45)
    ky1, kx1 = ky0 + 60, kx0 + 80
    mask[ky0:ky1, kx0:kx1] = 255

    # Depth: key is near (0.1), table is mid (0.4), background far (0.9)
    depth = np.full((h, w), 0.9, dtype=np.float32)
    depth[table_y0:h, 0:w] = 0.4
    depth[ky0:ky1, kx0:kx1] = 0.1

    return mask, depth, (kx0, ky0, kx1, ky1)


# ── unit tests ────────────────────────────────────────────────────────────────

class TestBackgroundRejectReason:
    def setup_method(self):
        from modules.depth_studio.subject_masker import _background_reject_reason
        self.reject = _background_reject_reason

    def test_bottom_border_large_area_rejected(self):
        # touches bottom, area > 25%
        reason = self.reject(x0=0, y0=432, x1=720, y1=720, area_ratio=0.28, h=720, w=720)
        assert reason == "bottom_border_large_area"

    def test_bottom_border_full_width_rejected(self):
        # touches bottom, bbox spans full width
        reason = self.reject(x0=0, y0=500, x1=720, y1=720, area_ratio=0.10, h=720, w=720)
        assert reason == "bottom_border_full_width"

    def test_full_width_background_rejected(self):
        # spans left+right, very wide
        reason = self.reject(x0=0, y0=0, x1=720, y1=200, area_ratio=0.10, h=720, w=720)
        assert reason == "full_width_background"

    def test_center_object_not_rejected(self):
        # small centred box — should not be rejected
        reason = self.reject(x0=300, y0=250, x1=420, y1=330, area_ratio=0.02, h=720, w=720)
        assert reason is None


class TestFilterComponents:
    def test_table_component_rejected_key_selected(self):
        """Bottom-border table component must be rejected; key component must be selected."""
        from modules.depth_studio.subject_masker import _filter_components

        mask, depth, (kx0, ky0, kx1, ky1) = _make_synthetic_mask_and_depth()
        selected_mask, stats = _filter_components(mask, 720, 720, depth, prompt_box=None)

        assert selected_mask is not None, "Expected a selected component, got None"

        # Rejected list should contain at least the table component
        assert len(stats["rejected"]) >= 1
        reject_reasons = [r["reject_reason"] for r in stats["rejected"]]
        assert any("bottom_border" in r for r in reject_reasons), (
            f"Expected bottom_border rejection, got: {reject_reasons}"
        )

        # Selected mask should cover the key area
        key_pixels = selected_mask[ky0:ky1, kx0:kx1]
        assert key_pixels.max() == 255, "Key object pixels should be in selected mask"

        # Selected mask should NOT span full image width at bottom
        bottom_row = selected_mask[710, :]
        assert bottom_row.max() == 0, "Bottom border should not be in selected mask"

    def test_prompt_box_boosts_correct_component(self):
        """When prompt_box overlaps key, key must win over anything else."""
        from modules.depth_studio.subject_masker import _filter_components

        mask, depth, (kx0, ky0, kx1, ky1) = _make_synthetic_mask_and_depth()
        prompt = (kx0 - 5, ky0 - 5, kx1 + 5, ky1 + 5)
        selected_mask, stats = _filter_components(mask, 720, 720, depth, prompt_box=prompt)

        assert selected_mask is not None
        key_pixels = selected_mask[ky0:ky1, kx0:kx1]
        assert key_pixels.max() == 255

    def test_all_border_touching_falls_back_to_none(self):
        """If every component touches the border, _filter_components returns None."""
        from modules.depth_studio.subject_masker import _filter_components

        h, w = 100, 100
        # Single component that spans full width and touches bottom
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[60:100, 0:100] = 255
        depth = np.full((h, w), 0.5, dtype=np.float32)

        selected_mask, stats = _filter_components(mask, h, w, depth, prompt_box=None)
        assert selected_mask is None
        assert len(stats["rejected"]) >= 1


class TestAssessQuality:
    def setup_method(self):
        from modules.depth_studio.subject_masker import _assess_quality
        self.assess = _assess_quality

    def test_full_frame_fallback_is_low_confidence(self):
        q = self.assess(0.5, [0, 0, 720, 720], 720, 720, "depth_threshold", True, True)
        assert q == "low_confidence"

    def test_bbox_area_over_60pct_is_low_confidence(self):
        # bbox covers 70% of image
        q = self.assess(0.70, [0, 0, 720, 504], 720, 720, "depth_threshold", False, True)
        assert q == "low_confidence"

    def test_depth_threshold_with_sam2_unavailable_is_review(self):
        q = self.assess(0.10, [200, 200, 500, 500], 720, 720, "depth_threshold", False, True)
        assert q == "review"

    def test_sam2_with_reasonable_mask_is_ok(self):
        q = self.assess(0.10, [200, 200, 500, 500], 720, 720, "sam2", False, False)
        assert q == "ok"

    def test_bottom_border_touch_with_large_area_is_review(self):
        # bbox touches bottom, fg_ratio > 0.20
        q = self.assess(0.25, [0, 400, 720, 719], 720, 720, "depth_threshold", False, True)
        assert q == "review"


class TestComputeSubjectMaskIntegration:
    """Integration test using synthetic in-memory image + depth (no disk I/O for mask read)."""

    def test_table_scenario_rejects_table(self, tmp_path):
        import cv2
        from modules.depth_studio.subject_masker import compute_subject_mask

        h, w = 360, 360
        # Create synthetic image (grey gradient)
        img = np.full((h, w, 3), 128, dtype=np.uint8)
        img_path = str(tmp_path / "test_img.jpg")
        cv2.imwrite(img_path, img)

        # Depth: near region centred (key), far everywhere else
        depth = np.full((h, w), 0.9, dtype=np.float32)
        ky0, kx0, ky1, kx1 = 120, 130, 180, 210
        depth[ky0:ky1, kx0:kx1] = 0.08   # very near = subject

        result = compute_subject_mask(
            image_path=img_path,
            depth_norm=depth,
            output_dir=str(tmp_path),
        )

        assert result["mask"] is not None
        assert result["fg_ratio"] < 0.50, f"fg_ratio too large: {result['fg_ratio']}"
        assert result["method_used"] in ("depth_threshold", "center_crop_fallback")
        assert (tmp_path / "subject_mask.png").exists()
        assert (tmp_path / "mask_overlay.png").exists()
        assert (tmp_path / "mask_stats.json").exists()

        import json
        stats = json.loads((tmp_path / "mask_stats.json").read_text())
        assert "component_count" in stats
        assert "mask_quality" in stats
