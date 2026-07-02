import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from opendrop_ml.modules.contact_angle.contact_line import (
    ANGLE_SEPARATION_MIN,
    CASE_11_MIN_CPS,
    KEPT_X_RANGE_MIN,
    LOCAL_ANGLE_N_PTS,
    _line_intersection,
    find_first_intercept,
    find_last_intercept,
    fit_contact_line,
    fit_contact_line_with_stats,
    local_contact_angle,
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


# ---------------------------------------------------------------------
# Fix B fixtures + tests (BIMODAL_CP_FIX.md § Production Spec)
# ---------------------------------------------------------------------


def _make_vertical_wall_contour(cp_xy, n_above=20):
    """Synthetic full drop contour: a vertical wall directly above the CP.

    Used for cases where local_contact_angle should resolve to ~90°
    (surface-outward) — the "reflection-confused" tier in MS_APTES.
    """
    cp_x, cp_y = float(cp_xy[0]), float(cp_xy[1])
    xs = np.full(n_above, cp_x)
    ys = np.linspace(cp_y - 1, cp_y - 20, n_above)
    return np.column_stack([xs, ys])


def _make_inward_lean_contour(cp_xy, side, n_above=20, lean=8.0):
    """Synthetic contour leaning toward drop interior — high-CA case.

    For ``side="right"``, the contour above the CP shifts +lean px in
    +x (which is *outward* on the right) so the angle from
    surface-outward is small (e.g. ~50°). Mirrored for left.
    """
    cp_x, cp_y = float(cp_xy[0]), float(cp_xy[1])
    sign = 1.0 if side == "right" else -1.0
    xs = np.linspace(cp_x, cp_x + sign * lean, n_above)
    ys = np.linspace(cp_y - 1, cp_y - 20, n_above)
    return np.column_stack([xs, ys])


def _build_bimodal_fixture(side="right", n_low=16, n_high=11, x_span=27.0):
    """Build a y-bimodal CP fixture mirroring MS_APTES right-side.

    Returns (cps, contours):
        - low-y tier (y≈274): n_low CPs with vertical-wall contours
          (local angle ≈ 90°, "reflection-confused")
        - high-y tier (y≈265): n_high CPs with inward-lean contours
          (local angle far from 90°, "real-contact")

    With these, Fix B should detect the bimodality and pick the
    high-y cluster.
    """
    rng = np.random.default_rng(0)
    xs_low = np.linspace(380.0, 380.0 + x_span, n_low)
    ys_low = 274.0 + rng.normal(0.0, 0.5, n_low)
    xs_high = np.linspace(380.0, 380.0 + x_span, n_high)
    ys_high = 265.0 + rng.normal(0.0, 0.5, n_high)

    cps = np.column_stack([np.r_[xs_low, xs_high], np.r_[ys_low, ys_high]])

    contours = []
    for cp in cps:
        is_low_tier = cp[1] > 270.0
        if is_low_tier:
            contours.append(_make_vertical_wall_contour(cp))
        else:
            contours.append(_make_inward_lean_contour(cp, side=side))
    return cps, contours


def test_fit_contact_line_backward_compat_bit_for_bit():
    """Legacy callers (cps only) get the exact same line as before Fix B."""
    cps, _ = _build_bimodal_fixture()
    legacy = fit_contact_line(cps)
    again = fit_contact_line(cps, contours=None, side=None)
    np.testing.assert_array_equal(legacy, again)


def test_fit_contact_line_case_11_early_return_n3():
    """n=3 < CASE_11_MIN_CPS → Fix B falls through to legacy line."""
    assert CASE_11_MIN_CPS >= 4
    cps = np.array([[0.0, 5.0], [5.0, 6.0], [10.0, 4.0]])
    contours = [_make_vertical_wall_contour(cp) for cp in cps]
    fb_line = fit_contact_line(cps, contours=contours, side="right")
    legacy_line = fit_contact_line(cps)
    np.testing.assert_array_equal(fb_line, legacy_line)


def test_fit_contact_line_fix_b_corrects_bimodal_to_high_y_tier():
    """Synthetic two-tier CPs trigger Fix B and refit lands on high-y tier."""
    cps, contours = _build_bimodal_fixture()
    legacy = fit_contact_line(cps)
    legacy_dy = abs(legacy[1, 1] - legacy[0, 1])
    assert legacy_dy > 1.0, "fixture must be bimodal enough that legacy tilts"

    corrected = fit_contact_line(cps, contours=contours, side="right")
    corrected_dy = abs(corrected[1, 1] - corrected[0, 1])
    assert corrected_dy < legacy_dy / 5.0, "Fix B should produce a near-flat line"
    assert abs(corrected[0, 1] - 265.0) < 1.0, "kept tier median y ≈ 265"
    assert abs(corrected[1, 1] - 265.0) < 1.0


def test_fit_contact_line_unimodal_clean_data_abstains():
    """Sub-pixel scatter on a single horizontal cluster — no trigger."""
    rng = np.random.default_rng(1)
    n = 20
    xs = np.linspace(100.0, 200.0, n)
    ys = 50.0 + rng.normal(0.0, 0.3, n)
    cps = np.column_stack([xs, ys])
    contours = [_make_vertical_wall_contour(cp) for cp in cps]

    legacy = fit_contact_line(cps)
    corrected = fit_contact_line(cps, contours=contours, side="right")
    np.testing.assert_array_equal(legacy, corrected)


def test_fit_contact_line_case_5_x_range_guard_abstains():
    """Artifact cluster bunched at single x → kept_x_range guard abstains."""
    n_artifact = 6
    n_real = 20
    artifact_x = 400.0
    artifact_xs = artifact_x + np.linspace(-1.0, 1.0, n_artifact)  # 2 px span
    artifact_ys = np.full(n_artifact, 200.0)
    real_xs = np.linspace(380.0, 420.0, n_real)
    real_ys = np.full(n_real, 300.0) + np.linspace(-0.3, 0.3, n_real)
    cps = np.column_stack([np.r_[artifact_xs, real_xs], np.r_[artifact_ys, real_ys]])

    contours = []
    for cp in cps:
        if cp[1] < 250.0:
            contours.append(_make_inward_lean_contour(cp, side="right", lean=15.0))
        else:
            contours.append(_make_vertical_wall_contour(cp))

    legacy = fit_contact_line(cps)
    corrected = fit_contact_line(cps, contours=contours, side="right")
    assert KEPT_X_RANGE_MIN > 2.0
    np.testing.assert_array_equal(legacy, corrected)


def test_fit_contact_line_case_9_all_nan_angles_abstains():
    """Cluster with no valid local angles → fails NaN-ratio guard."""
    cps, _ = _build_bimodal_fixture()
    n = len(cps)
    too_short = np.array([[0.0, 0.0], [1.0, -1.0]])  # < LOCAL_ANGLE_N_PTS
    contours = [too_short.copy() for _ in range(n)]
    assert len(too_short) < LOCAL_ANGLE_N_PTS

    legacy = fit_contact_line(cps)
    corrected = fit_contact_line(cps, contours=contours, side="right")
    np.testing.assert_array_equal(legacy, corrected)


def test_fit_contact_line_angle_separation_guard_abstains():
    """Two y-tiers with similar local angles → angle-sep guard abstains."""
    cps, _ = _build_bimodal_fixture()
    contours = [_make_vertical_wall_contour(cp) for cp in cps]

    legacy = fit_contact_line(cps)
    corrected = fit_contact_line(cps, contours=contours, side="right")
    assert ANGLE_SEPARATION_MIN > 0.0
    np.testing.assert_array_equal(legacy, corrected)


def test_fit_contact_line_with_stats_backward_compat_bit_for_bit():
    """Legacy callers of stats variant get the same (line, stats) as before."""
    cps, _ = _build_bimodal_fixture()
    legacy_line, legacy_stats = fit_contact_line_with_stats(cps)
    again_line, again_stats = fit_contact_line_with_stats(cps, contours=None, side=None)
    np.testing.assert_array_equal(legacy_line, again_line)
    assert legacy_stats == again_stats


def test_fit_contact_line_with_stats_recomputes_after_fix_b_correction():
    """After Fix B refits the kept cluster, stats reflect the kept cluster."""
    cps, contours = _build_bimodal_fixture()
    legacy_line, legacy_stats = fit_contact_line_with_stats(cps)
    fixed_line, fixed_stats = fit_contact_line_with_stats(
        cps, contours=contours, side="right"
    )

    assert not np.allclose(legacy_line, fixed_line)
    assert fixed_stats["n_total"] == 11, "stats reflect kept cluster size"
    assert fixed_stats["residual_iqr"] < legacy_stats["residual_iqr"]


def test_fit_contact_line_with_stats_abstain_path_returns_original_stats():
    """Unimodal data → stats variant returns original line + original stats."""
    rng = np.random.default_rng(2)
    n = 20
    xs = np.linspace(100.0, 200.0, n)
    ys = 50.0 + rng.normal(0.0, 0.3, n)
    cps = np.column_stack([xs, ys])
    contours = [_make_vertical_wall_contour(cp) for cp in cps]

    legacy_line, legacy_stats = fit_contact_line_with_stats(cps)
    fixed_line, fixed_stats = fit_contact_line_with_stats(
        cps, contours=contours, side="right"
    )
    np.testing.assert_array_equal(legacy_line, fixed_line)
    assert legacy_stats == fixed_stats


# ---------------------------------------------------------------------
# local_contact_angle helper
# ---------------------------------------------------------------------


def test_local_contact_angle_vertical_wall_returns_about_90_right():
    cp = np.array([400.0, 250.0])
    contour = _make_vertical_wall_contour(cp, n_above=20)
    angle = local_contact_angle(contour, cp, side="right")
    assert abs(angle - 90.0) < 1.0


def test_local_contact_angle_inward_lean_returns_far_from_90_right():
    cp = np.array([400.0, 250.0])
    contour = _make_inward_lean_contour(cp, side="right", lean=15.0)
    angle = local_contact_angle(contour, cp, side="right")
    # Surface-inward / conventional contact-angle convention: a contour
    # leaning toward the drop interior reads as a high CA (>90°).
    assert angle > 110.0
    assert abs(angle - 90.0) > 20.0


def test_local_contact_angle_too_few_points_returns_nan():
    cp = np.array([400.0, 250.0])
    short = _make_vertical_wall_contour(cp, n_above=2)
    assert np.isnan(local_contact_angle(short, cp, side="right"))


def test_local_contact_angle_convention_invariant_dev_from_90():
    """Decision math uses |angle - 90°|, invariant under inward<->outward swap."""
    cp = np.array([400.0, 250.0])
    contour = _make_inward_lean_contour(cp, side="right", lean=15.0)
    a_right = local_contact_angle(contour, cp, side="right")
    a_left = local_contact_angle(contour, cp, side="left")
    # Surface-outward right vs left flips the sign of the reference vector
    # → angle becomes 180° - a_right. |angle - 90°| is the same on both.
    assert abs(abs(a_right - 90.0) - abs(a_left - 90.0)) < 0.01


if __name__ == "__main__":
    pytest.main(["-v", __file__])
