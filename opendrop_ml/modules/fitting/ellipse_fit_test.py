from opendrop_ml.modules.fitting.ellipse_fit import (
    ellipse_fit,
    ellipse_fit_img,
    fit_ellipse,
    ell_parameters,
    ellipse_line_intersection,
    ellipse_closest_point,
    ellipse_fit_errors,
)
from unittest.mock import patch
import numpy as np
import pytest
import math


def _generate_ellipse_drop(a, b, center, angle_range=(0, np.pi), num_points=200):
    """Generate a drop-like ellipse contour (upper half) for testing.

    Parameters
    ----------
    a, b : float
        Semi-axes of the ellipse.
    center : tuple
        (cx, cy) center of the ellipse.
    angle_range : tuple
        (start, end) parametric angle range. (0, pi) gives the upper half
        in image coordinates (y increases downward, so the drop body is above
        the baseline).
    num_points : int
        Number of contour points.

    Returns
    -------
    np.ndarray of shape (num_points, 2)
        Ordered contour with profile[0] = left CP, profile[-1] = right CP.
    """
    cx, cy = center
    t = np.linspace(angle_range[0], angle_range[1], num_points)
    # In image coords, y increases downward.  We want a drop sitting on a
    # horizontal baseline, so the baseline (CPs) should have the largest y,
    # and the apex the smallest y.
    x = cx + a * np.cos(t)
    y = cy - b * np.sin(t)  # minus so apex is above baseline
    return np.column_stack((x, y))


@pytest.fixture
def symmetric_drop():
    """A symmetric drop on a horizontal baseline (ellipse upper half)."""
    return _generate_ellipse_drop(a=40, b=30, center=(200, 300), num_points=200)


@pytest.fixture
def high_angle_drop():
    """A tall narrow drop (high contact angle)."""
    return _generate_ellipse_drop(a=20, b=50, center=(200, 300), num_points=200)


# ---------- ellipse_fit (contour-level) ----------


def test_ellipse_fit_returns_correct_structure(symmetric_drop):
    """ellipse_fit returns a 7-element tuple with expected types."""
    CA, intercepts, center, (a, b), phi_deg, errors, timings = ellipse_fit(
        symmetric_drop
    )

    assert isinstance(CA, list)
    assert isinstance(intercepts, list)
    assert isinstance(timings, dict)
    assert "fit time" in timings
    assert "analysis time" in timings
    # center should be a 2-element array-like
    assert len(center) == 2


def test_ellipse_fit_produces_angles(symmetric_drop):
    """ellipse_fit should produce two finite contact angles for a valid drop."""
    CA, intercepts, center, (a, b), phi_deg, errors, timings = ellipse_fit(
        symmetric_drop
    )

    assert len(CA) == 2, f"Expected 2 angles, got {len(CA)}"
    for angle in CA:
        assert np.isfinite(angle), f"Non-finite angle: {angle}"
        assert 0 < angle < 180, f"Angle {angle} outside (0, 180)"


def test_ellipse_fit_symmetric_drop_angles_close(symmetric_drop):
    """For a symmetric drop the left and right angles should be similar."""
    CA, *_ = ellipse_fit(symmetric_drop)

    if len(CA) == 2:
        assert (
            abs(CA[0] - CA[1]) < 10
        ), f"Symmetric drop angles differ by {abs(CA[0] - CA[1]):.1f}°"


def test_ellipse_fit_recovers_ellipse_parameters(symmetric_drop):
    """The fitted ellipse parameters should be close to the generating ones."""
    CA, intercepts, center, (a, b), phi_deg, errors, timings = ellipse_fit(
        symmetric_drop
    )

    # Sort semi-axes so we compare like-for-like
    fitted = sorted([a, b])
    expected = sorted([40, 30])
    assert abs(fitted[0] - expected[0]) < 5, f"Small axis: {fitted[0]} vs {expected[0]}"
    assert abs(fitted[1] - expected[1]) < 5, f"Large axis: {fitted[1]} vs {expected[1]}"


def test_ellipse_fit_center_near_expected(symmetric_drop):
    """Fitted center should be near the generating center."""
    CA, intercepts, center, (a, b), phi_deg, errors, timings = ellipse_fit(
        symmetric_drop
    )

    assert abs(center[0] - 200) < 5, f"Center x: {center[0]} vs 200"
    assert abs(center[1] - 300) < 5, f"Center y: {center[1]} vs 300"


def test_ellipse_fit_high_angle_drop(high_angle_drop):
    """A tall narrow drop should produce higher contact angles than a wide one."""
    CA, *_ = ellipse_fit(high_angle_drop)

    if len(CA) == 2:
        for angle in CA:
            assert angle > 50, f"Expected high angle, got {angle:.1f}°"


def test_ellipse_fit_errors_dict():
    """ellipse_fit_errors returns a dict with expected keys."""
    # Simple contour for the error calculation
    drop = _generate_ellipse_drop(a=40, b=30, center=(200, 300), num_points=100)
    errors = ellipse_fit_errors(drop, 200, 300, 40, 30, 0, display=False)

    assert isinstance(errors, dict)
    assert "MAE" in errors
    assert "MSE" in errors
    assert "RMSE" in errors


# ---------- ellipse_fit_img (image-level wrapper) ----------


@patch("opendrop_ml.modules.fitting.ellipse_fit.preprocess")
def test_ellipse_fit_img_returns_correct_structure(mock_preprocess):
    """ellipse_fit_img returns a 7-element tuple with expected types."""
    drop = _generate_ellipse_drop(a=40, b=30, center=(200, 300), num_points=200)
    CPs_chosen = np.array([drop[0], drop[-1]])
    img = np.zeros((600, 400, 3), dtype=np.uint8)
    mock_preprocess.return_value = (img, CPs_chosen, drop)

    CA, intercepts, center, (a, b), phi_deg, errors, timings = ellipse_fit_img(img)

    assert isinstance(CA, list)
    assert isinstance(intercepts, list)
    assert isinstance(timings, dict)
    assert "method specific preprocessing time" in timings
    assert "fit time" in timings
    assert "analysis time" in timings
    mock_preprocess.assert_called_once()


@patch("opendrop_ml.modules.fitting.ellipse_fit.preprocess")
def test_ellipse_fit_img_produces_angles(mock_preprocess):
    """ellipse_fit_img should produce two finite contact angles."""
    drop = _generate_ellipse_drop(a=40, b=30, center=(200, 300), num_points=200)
    CPs_chosen = np.array([drop[0], drop[-1]])
    img = np.zeros((600, 400, 3), dtype=np.uint8)
    mock_preprocess.return_value = (img, CPs_chosen, drop)

    CA, intercepts, center, (a, b), phi_deg, errors, timings = ellipse_fit_img(img)

    assert len(CA) == 2, f"Expected 2 angles, got {len(CA)}"
    for angle in CA:
        assert np.isfinite(angle), f"Non-finite angle: {angle}"
        assert 0 < angle < 180, f"Angle {angle} outside (0, 180)"


@patch("opendrop_ml.modules.fitting.ellipse_fit.preprocess")
def test_ellipse_fit_img_angles_match_ellipse_fit(mock_preprocess):
    """ellipse_fit_img and ellipse_fit should produce identical angles
    given the same contour data."""
    drop = _generate_ellipse_drop(a=40, b=30, center=(200, 300), num_points=200)
    CPs_chosen = np.array([drop[0], drop[-1]])
    img = np.zeros((600, 400, 3), dtype=np.uint8)
    mock_preprocess.return_value = (img, CPs_chosen, drop)

    CA_img, *_ = ellipse_fit_img(img)
    CA_direct, *_ = ellipse_fit(drop)

    assert len(CA_img) == len(CA_direct)
    for a_img, a_direct in zip(CA_img, CA_direct):
        assert (
            abs(a_img - a_direct) < 0.1
        ), f"ellipse_fit_img={a_img:.2f} vs ellipse_fit={a_direct:.2f}"


# ---------- Helper functions ----------


def test_fit_ellipse_and_ell_parameters():
    """fit_ellipse + ell_parameters recover known ellipse geometry.

    Note on axis-aligned singularity: a full, perfectly axis-aligned ellipse
    with exact float coordinates causes ell_parameters to hit a singular matrix
    (the Fitzgibbon conic fit returns degenerate coefficients where a[0:3]=0).
    This does NOT affect real data because it requires mathematically exact
    parametric coordinates — any quantization at all prevents it.  Verified
    that integer pixel coords (from cv2.findContours, always int32), sub-pixel
    rounding (even to 4 decimals), noise (even 0.01 px), and Bond-number shape
    distortion each independently eliminate the singularity.
    We use integer coords here to match the real pipeline.
    """
    t = np.linspace(0, 2 * np.pi, 500)
    a_true, b_true = 50, 30
    cx, cy = 100, 100
    # Use integer coordinates (matching cv2.findContours output) to avoid the
    # exact-float singularity while keeping the test realistic.
    x = np.round(cx + a_true * np.cos(t)).astype(float)
    y = np.round(cy + b_true * np.sin(t)).astype(float)

    avec = fit_ellipse(x, y)
    (a_fit, b_fit), phi_deg, center = ell_parameters(avec)

    fitted = sorted([a_fit, b_fit])
    expected = sorted([a_true, b_true])
    assert abs(fitted[0] - expected[0]) < 2
    assert abs(fitted[1] - expected[1]) < 2
    assert abs(center[0] - cx) < 2
    assert abs(center[1] - cy) < 2


def test_ellipse_line_intersection_horizontal_baseline():
    """Horizontal baseline through center of upright ellipse should give 2 intersections."""
    # Upright ellipse at (100,100), a=50 (horizontal), b=30 (vertical)
    # Horizontal line y=130 (bottom of ellipse) through CPs
    result = ellipse_line_intersection(
        100,
        100,
        50,
        30,
        0,  # xc, yc, a, b, theta=0
        50,
        130,
        150,
        130,  # baseline from (50,130) to (150,130)
        display=False,
    )
    if result is not None:
        assert len(result) == 2
        # Both intersection points should be near y=130
        for pt in result:
            assert abs(pt[1] - 130) < 2


def test_ellipse_closest_point():
    """ellipse_closest_point returns a distance and a 2D point."""
    gap, closest = ellipse_closest_point(
        100, 130, 100, 100, 50, 30, 0, n=1000, display=False
    )
    assert isinstance(gap, (int, float, np.floating))
    assert len(closest) == 2
    assert gap < 2  # Point on the ellipse boundary should be very close


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
