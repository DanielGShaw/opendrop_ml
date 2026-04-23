import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

from opendrop_ml.modules.contact_angle.contact_line import (
    _line_intersection,
    find_first_intercept,
    find_last_intercept,
    fit_contact_line,
)


def test_line_intersection_crossing_segments():
    p1, p2 = [0.0, 0.0], [10.0, 10.0]
    q1, q2 = [0.0, 10.0], [10.0, 0.0]
    pt = _line_intersection(p1, p2, q1, q2)
    assert pt is not None
    np.testing.assert_allclose(pt, [5.0, 5.0], atol=1e-6)


def test_line_intersection_parallel_returns_none():
    assert _line_intersection([0, 0], [10, 0], [0, 5], [10, 5]) is None


def test_line_intersection_disjoint_segments_returns_none():
    # Segments are on crossing lines but the crossing point is outside
    # both segments' parametric ranges [0, 1].
    assert _line_intersection([0, 0], [1, 1], [5, 5], [5, 10]) is None


def test_find_first_intercept_happy_case():
    # Triangle contour going up and back down across y=5.
    contour = np.array(
        [
            [0.0, 0.0],
            [5.0, 10.0],
            [10.0, 0.0],
        ]
    )
    line = np.array([[-5.0, 5.0], [15.0, 5.0]])  # horizontal at y=5
    idx = find_first_intercept(contour, line)
    # First crossing is on segment [0]-[1] (rising edge).
    assert idx == 0


def test_find_first_intercept_no_crossing_returns_minus_one():
    contour = np.array([[0.0, 0.0], [5.0, 2.0], [10.0, 0.0]])
    line = np.array([[-5.0, 100.0], [15.0, 100.0]])  # line far above contour
    assert find_first_intercept(contour, line) == -1


def test_find_last_intercept_returns_final_segment():
    # Same triangle as find_first_intercept happy case, but iterating
    # backward the first hit is the falling edge (segment index 1).
    contour = np.array(
        [
            [0.0, 0.0],
            [5.0, 10.0],
            [10.0, 0.0],
        ]
    )
    line = np.array([[-5.0, 5.0], [15.0, 5.0]])  # horizontal at y=5
    assert find_last_intercept(contour, line) == 1


def test_find_last_intercept_no_crossing_returns_minus_one():
    contour = np.array([[0.0, 0.0], [5.0, 2.0], [10.0, 0.0]])
    line = np.array([[-5.0, 100.0], [15.0, 100.0]])  # far above contour
    assert find_last_intercept(contour, line) == -1


def test_find_last_intercept_max_y_filters_high_intercepts():
    # Triangle crosses a sloped line at two points with different y
    # values. With max_y set between them, the backward iteration must
    # skip the upper hit and return the lower one.
    contour = np.array(
        [
            [0.0, 0.0],
            [5.0, 10.0],
            [10.0, 0.0],
        ]
    )
    # Sloped line: y = x  (passes through (0,0) and (10,10))
    # Rising segment [0]-[1]: crosses where 2*t = 5*t (t from linear
    # interp) — concretely at (x, y) with x = 5*t, y = 10*t, and y = x
    # gives 10*t = 5*t → t=0, i.e. the (0,0) endpoint.
    # Use a shifted line instead so both crossings land mid-segment.
    line = np.array([[0.0, 2.0], [10.0, 8.0]])  # slope 0.6, y = 0.6x + 2
    # Rising edge [0]-[1]: y_contour = 2x vs y_line = 0.6x + 2 →
    #   2x = 0.6x + 2 → x = 2/1.4 ≈ 1.43, y ≈ 2.86
    # Falling edge [1]-[2]: y_contour = -2x + 20 vs y_line = 0.6x + 2 →
    #   -2x + 20 = 0.6x + 2 → x = 18/2.6 ≈ 6.92, y ≈ 6.15
    # Without max_y, last crossing (backward) is segment 1 (y≈6.15).
    assert find_last_intercept(contour, line) == 1
    # max_y=4 rejects the falling-edge hit (y≈6.15) — falls back to
    # rising-edge hit (y≈2.86) at segment 0.
    assert find_last_intercept(contour, line, max_y=4.0) == 0


def test_fit_contact_line_recovers_slope_with_outlier():
    # Points along y = 0.5 * x + 10, plus one big outlier that IQR should drop.
    xs = np.arange(10, dtype=float)
    ys = 0.5 * xs + 10
    cps = np.column_stack([xs, ys])
    cps[5, 1] += 50  # outlier

    line = fit_contact_line(cps)
    assert line.shape == (2, 2)

    # Spans the original x range.
    assert line[0, 0] == pytest.approx(xs.min())
    assert line[1, 0] == pytest.approx(xs.max())

    # Slope of the returned line should match the inlier slope within ODR noise.
    dx = line[1, 0] - line[0, 0]
    dy = line[1, 1] - line[0, 1]
    slope = dy / dx
    assert slope == pytest.approx(0.5, abs=0.02)


def test_fit_contact_line_flat_when_all_points_collinear_horizontal():
    cps = np.array([[0.0, 5.0], [5.0, 5.0], [10.0, 5.0]])
    line = fit_contact_line(cps)
    # Endpoints span x range at y=5 (flat line).
    assert line[0, 0] == pytest.approx(0.0)
    assert line[1, 0] == pytest.approx(10.0)
    assert abs(line[1, 1] - line[0, 1]) < 1e-6


if __name__ == "__main__":
    pytest.main(["-v", __file__])
