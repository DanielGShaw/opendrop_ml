import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from opendrop_ml.modules.image.select_regions_geom import (
    initial_baseline_seed,
    longest_contour_arc_above_baseline,
    points_above_baseline,
    rotate_baseline,
)


class TestPointsAboveBaseline(unittest.TestCase):
    def test_horizontal_baseline_keeps_visually_above_points(self):
        # Horizontal baseline left-to-right at y=10. In image coordinates the
        # "above" side is y < 10, so only the y=5 points should survive.
        contour = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 15.0], [4.0, 20.0]])
        out = points_above_baseline(contour, ix=0, iy=10, fx=100, fy=10)
        assert_array_equal(out, np.array([[1.0, 5.0], [2.0, 5.0]]))

    def test_baseline_on_point_excludes_it(self):
        # Strict inequality (xp > 0): points lying exactly on the baseline are
        # filtered out, matching the original loop behavior.
        contour = np.array([[5.0, 10.0], [5.0, 0.0]])
        out = points_above_baseline(contour, ix=0, iy=10, fx=10, fy=10)
        assert_array_equal(out, np.array([[5.0, 0.0]]))

    def test_tilted_baseline(self):
        # Baseline from (0,10) to (10,0): slope -1. Points with y < 10 - x are
        # above it. (2, 2): 10 - 2 = 8, 2 < 8 → above. (8, 8): 10 - 8 = 2,
        # 8 > 2 → below.
        contour = np.array([[2.0, 2.0], [8.0, 8.0]])
        out = points_above_baseline(contour, ix=0, iy=10, fx=10, fy=0)
        assert_array_equal(out, np.array([[2.0, 2.0]]))

    def test_empty_contour_returns_empty(self):
        contour = np.empty((0, 2))
        out = points_above_baseline(contour, ix=0, iy=10, fx=100, fy=10)
        self.assertEqual(out.shape, (0, 2))

    def test_preserves_dtype(self):
        contour = np.array([[1.0, 5.0], [2.0, 15.0]])
        out = points_above_baseline(contour, ix=0, iy=10, fx=100, fy=10)
        self.assertEqual(out.dtype, contour.dtype)


class TestInitialBaselineSeed(unittest.TestCase):
    def test_horizontal_just_below_lowest_point(self):
        contour = np.array([[10.0, 50.0], [20.0, 80.0], [30.0, 60.0]])
        ix, iy, fx, fy = initial_baseline_seed(
            contour, canvas_width=200, canvas_height=300
        )
        self.assertEqual((ix, iy, fx, fy), (0, 81, 200, 81))

    def test_clamped_to_canvas_bottom(self):
        # Lowest contour y near canvas bottom — must stay at least 5 px above.
        contour = np.array([[10.0, 395.0]])
        ix, iy, fx, fy = initial_baseline_seed(
            contour, canvas_width=500, canvas_height=400
        )
        self.assertEqual((ix, iy, fx, fy), (0, 395, 500, 395))

    def test_returns_horizontal_line(self):
        contour = np.array([[0.0, 10.0], [50.0, 25.0]])
        ix, iy, fx, fy = initial_baseline_seed(
            contour, canvas_width=100, canvas_height=200
        )
        self.assertEqual(iy, fy)
        self.assertEqual(ix, 0)
        self.assertEqual(fx, 100)


class TestRotateBaseline(unittest.TestCase):
    def test_zero_rotation_returns_endpoints_unchanged(self):
        out = rotate_baseline(10, 50, 90, 50, theta_deg=0.0)
        self.assertEqual(out, (10, 50, 90, 50))

    def test_midpoint_preserved(self):
        ix, iy, fx, fy = 10, 40, 90, 60
        mid_in = ((ix + fx) / 2, (iy + fy) / 2)
        rix, riy, rfx, rfy = rotate_baseline(ix, iy, fx, fy, theta_deg=37.0)
        mid_out = ((rix + rfx) / 2, (riy + rfy) / 2)
        assert_allclose(mid_out, mid_in, atol=1.0)

    def test_ninety_degrees_clockwise_in_image(self):
        # Horizontal baseline (0,50)→(100,50), midpoint (50,50). Positive theta
        # is visually clockwise in image coords (y flipped vs math), so the
        # baseline becomes vertical with the start moving down to (50, 100)
        # and end moving up to (50, 0). Allclose because cos(pi/2) is ~6e-17,
        # not exactly zero.
        rix, riy, rfx, rfy = rotate_baseline(0, 50, 100, 50, theta_deg=90.0)
        assert_allclose((rix, riy, rfx, rfy), (50, 0, 50, 100), atol=1e-9)

    def test_negative_theta_rotates_counter_clockwise(self):
        rix, riy, rfx, rfy = rotate_baseline(0, 50, 100, 50, theta_deg=-90.0)
        assert_allclose((rix, riy, rfx, rfy), (50, 100, 50, 0), atol=1e-9)


class TestLongestContourArcAboveBaseline(unittest.TestCase):
    def test_returns_only_longest_contiguous_run(self):
        # Drop arc (5 above-baseline points) followed by a spike tip (2 above
        # points), separated by below-baseline gap. Baseline at y=10.
        contour = np.array(
            [
                # drop arc (y < 10)
                [0.0, 5.0],
                [1.0, 4.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                # surface dip back below baseline
                [5.0, 12.0],
                [6.0, 13.0],
                [7.0, 12.0],
                # spike tip (y < 10)
                [8.0, 8.0],
                [9.0, 9.0],
                # back below baseline
                [10.0, 14.0],
            ]
        )
        out = longest_contour_arc_above_baseline(contour, ix=0, iy=10, fx=20, fy=10)
        assert_array_equal(out, contour[0:5])

    def test_handles_wrap_around(self):
        # Contour starts mid-drop-arc: above-baseline run wraps from index 9
        # (last) back through 0,1,2,3 (first four).
        contour = np.array(
            [
                [3.0, 5.0],
                [4.0, 4.0],
                [5.0, 5.0],
                [6.0, 6.0],  # 0..3 above
                [7.0, 12.0],
                [8.0, 13.0],
                [4.0, 12.0],  # 4..6 below
                [3.0, 11.5],  # 7  below
                [2.0, 7.0],
                [1.0, 5.0],  # 8..9 above (wraps to 0..3)
            ]
        )
        out = longest_contour_arc_above_baseline(contour, ix=0, iy=10, fx=20, fy=10)
        # Expected: indices 8, 9, 0, 1, 2, 3 in that order.
        expected = np.concatenate([contour[8:10], contour[0:4]])
        assert_array_equal(out, expected)

    def test_picks_longest_when_multiple_runs(self):
        # Two above-baseline runs: short (1 point) then long (3 points).
        # Longest should win regardless of which appears first.
        contour = np.array(
            [
                [0.0, 5.0],  # 0  above (short run)
                [1.0, 12.0],
                [2.0, 13.0],  # 1..2 below
                [3.0, 4.0],
                [4.0, 3.0],
                [5.0, 4.0],  # 3..5 above (long run)
                [6.0, 11.0],  # 6  below
            ]
        )
        out = longest_contour_arc_above_baseline(contour, ix=0, iy=10, fx=20, fy=10)
        assert_array_equal(out, contour[3:6])

    def test_all_above_returns_full_contour(self):
        contour = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 5.0]])
        out = longest_contour_arc_above_baseline(contour, ix=0, iy=10, fx=20, fy=10)
        assert_array_equal(out, contour)

    def test_all_below_returns_empty(self):
        contour = np.array([[1.0, 15.0], [2.0, 14.0], [3.0, 15.0]])
        out = longest_contour_arc_above_baseline(contour, ix=0, iy=10, fx=20, fy=10)
        self.assertEqual(out.shape, (0, 2))

    def test_empty_contour_returns_empty(self):
        out = longest_contour_arc_above_baseline(
            np.empty((0, 2)), ix=0, iy=10, fx=20, fy=10
        )
        self.assertEqual(out.shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
