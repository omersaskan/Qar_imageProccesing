import unittest

class CoverageTracker:
    def __init__(self, num_sectors=36):
        self.num_sectors = num_sectors
        self.sector_size = 360 / num_sectors
        self.sectors = [False] * num_sectors

    def add_frame(self, azimuth, is_accepted):
        if not is_accepted:
            return
        normalized = ((azimuth % 360) + 360) % 360
        index = int(normalized // self.sector_size)
        if index < self.num_sectors:
            self.sectors[index] = True

    def get_percent(self):
        covered = sum(1 for s in self.sectors if s)
        return (covered / self.num_sectors) * 100

    def get_max_gap(self):
        max_gap = 0
        current_gap = 0
        extended_sectors = self.sectors + self.sectors
        for s in extended_sectors:
            if not s:
                current_gap += self.sector_size
            else:
                max_gap = max(max_gap, current_gap)
                current_gap = 0
        return min(360, max(max_gap, current_gap))

class TestUILogic(unittest.TestCase):
    def test_full_coverage(self):
        tracker = CoverageTracker(num_sectors=36)
        for a in range(0, 360, 10):
            tracker.add_frame(a, True)
        self.assertEqual(tracker.get_percent(), 100.0)
        self.assertEqual(tracker.get_max_gap(), 0.0)

    def test_partial_coverage(self):
        tracker = CoverageTracker(num_sectors=36)
        # Cover 0-180
        for a in range(0, 180, 10):
            tracker.add_frame(a, True)
        self.assertEqual(tracker.get_percent(), 50.0)
        self.assertEqual(tracker.get_max_gap(), 180.0)

    def test_gap_wrap_around(self):
        tracker = CoverageTracker(num_sectors=36)
        # Cover 10-350 (gap at 0)
        for a in range(10, 350, 10):
            tracker.add_frame(a, True)
        # Gap should be 20 degrees (from 350 to 10)
        self.assertEqual(tracker.get_max_gap(), 20.0)

    def test_rejection_logic(self):
        tracker = CoverageTracker(num_sectors=36)
        tracker.add_frame(0, False)
        self.assertEqual(tracker.get_percent(), 0.0)

    def test_gate_logic(self):
        # Helper to simulate app.js checkGate logic
        def can_finish(percent, max_gap, accepted_count, total_count, blur_rejections, is_demo):
            blur_ratio = blur_rejections / total_count if total_count > 0 else 0
            has_enough_frames = accepted_count > 50
            blur_is_ok = blur_ratio < 0.4
            coverage_is_ok = percent > 90 and max_gap < 45
            return (coverage_is_ok and has_enough_frames and blur_is_ok) or is_demo

        # Case 1: All pass
        self.assertTrue(can_finish(95, 20, 100, 120, 10, False))
        
        # Case 2: Gap too large
        self.assertFalse(can_finish(95, 50, 100, 120, 10, False))
        
        # Case 3: Too many blurs
        self.assertFalse(can_finish(95, 20, 100, 200, 100, False)) # 50% blur
        
        # Case 4: Demo mode always passes gate (for testing UX)
        self.assertTrue(can_finish(10, 300, 5, 10, 8, True))

if __name__ == '__main__':
    unittest.main()
