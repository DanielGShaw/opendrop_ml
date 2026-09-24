import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from opendrop_ml.modules.preprocessing.preprocessing import (
    CPID,
    check_for_high_angle,
    iterative_Lfit,
    isolate_contour,
    line_circle_intersection,
)

TRAINING_FILES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "training_files"
)
sys.path.insert(0, TRAINING_FILES_DIR)


# ===== iterative_Lfit sparse-contour fallback =====
#
# A restricted contour with a point count outside {>=110, ==2} used to leave
# `left`/`right` unassigned and crash with UnboundLocalError at `contours =
# [left, right]`. It must now either use the caller-supplied fallback points
# or raise a clear, named error -- never silently fall through.


def _sparse_contour(n=13):
    return np.array([[float(i), float(100 - i)] for i in range(n)])


def test_iterative_lfit_sparse_contour_without_fallback_raises_clear_error():
    with pytest.raises(ValueError, match="iterative_Lfit"):
        iterative_Lfit(_sparse_contour(), n_pts=10, display=False)


def test_iterative_lfit_sparse_contour_with_fallback_returns_fallback_points():
    fallback = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = iterative_Lfit(
        _sparse_contour(), n_pts=10, display=False, fallback_points=fallback
    )
    assert result == [[[1.0, 2.0]], [[3.0, 4.0]]]


def test_iterative_lfit_len_2_pre_split_contours_take_existing_branch():
    # len(contour) == 2: pre-split [left, right] pair -- must still take the
    # existing branch, not the new sparse-contour fallback path. Too few
    # points per side for a real gradient fit, so this is expected to fail
    # deeper in the algorithm (RMSE computation) rather than succeed -- the
    # point of this test is only that it must NOT be UnboundLocalError,
    # which would mean the fallback branch was hit instead.
    left = np.array([[0.0, 10.0], [1.0, 9.0]] * 10)
    right = np.array([[10.0, 10.0], [9.0, 9.0]] * 10)
    try:
        iterative_Lfit([left, right], n_pts=10, display=False)
    except UnboundLocalError:
        pytest.fail("len(contour) == 2 case regressed onto the sparse-contour fallback path")
    except Exception:
        pass


# ===== check_for_high_angle sparse-contour/short-edge errors =====
#
# Same defect shape as iterative_Lfit's fix above -- left/right used to be
# left unassigned (UnboundLocalError) when the contour was neither >110
# points (apex-split) nor exactly 2 (pre-split pair), and `edge[10]` used
# to raise a raw IndexError when a side had <=10 points after filtering.
# Both now raise a clear, named ValueError instead.


def test_check_for_high_angle_wrong_contour_length_raises_clear_error():
    contour = np.array([[float(i), float(100 - i)] for i in range(50)])
    with pytest.raises(ValueError, match="check_for_high_angle"):
        check_for_high_angle(contour, display=False)


def test_check_for_high_angle_short_edge_raises_clear_error():
    # A pre-split [left, right] pair (len(contour) == 2) where each side
    # already starts at its highest point (skips the apex-reorder branch)
    # but has too few points to reach index 10 after
    # find_rightmost_coordinates/find_leftmost_coordinates filtering.
    left = np.array([[0.0, float(y)] for y in range(5)])
    right = np.array([[10.0, float(y)] for y in range(5)])
    with pytest.raises(ValueError, match="check_for_high_angle"):
        check_for_high_angle([left, right], display=False)


# ===== CPID reproduction case =====
#
# angle=160, Bo=0.55, scaler=5.0, roughness=-0.015 is the deliberately-extreme
# cell where CPID's restricted contour collapsed to 13 points and crashed.
# The seed is `matrix_utils.cell_seed(160.0, 0.55, 5.0, -0.015)`, hardcoded
# here so this test doesn't depend on paper_results/synthetic_generation/
# output remaining in place.
_REPRO_SEED = 3075006224


def test_cpid_does_not_crash_on_reproducing_sparse_contour_image():
    from create_contours_args6 import draw_drop

    np.random.seed(_REPRO_SEED)
    img, _ = draw_drop(160.0, 0.55, 5.0, -0.015, 5000, dpi=64)

    _, contour, _, _ = isolate_contour(img, DISPLAY=False, blur=True)
    CPs = CPID(contour, display=False)

    assert CPs.shape == (2, 2)
    h, w = img.shape[:2]
    for x, y in CPs:
        assert 0 <= x < w
        assert 0 <= y < h


# ===== line_circle_intersection integer overflow =====
#
# isolate_contour() passes the Hough circle as uint16 and the Hough baseline
# as platform-default ints. On Windows numpy's default int is 32-bit, so the
# quadratic's integer arithmetic (b**2 is ~1e12) used to wrap, dropping or
# corrupting the drop's chord. These tests force int32 inputs so the overflow
# reproduces on any platform.


def _float_reference_intersections(p1, p2, center, radius):
    p1, p2, center = (np.asarray(v, dtype=float) for v in (p1, p2, center))
    d, f = p2 - p1, p1 - center
    a, b = d @ d, 2 * (f @ d)
    disc = b**2 - 4 * a * (f @ f - float(radius) ** 2)
    ts = [(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)]
    return np.array(sorted((p1 + t * d for t in ts), key=lambda p: p[0]))


@pytest.mark.parametrize("line_dtype", [np.int32, np.int64])
def test_line_circle_intersection_matches_float_reference(line_dtype):
    baseline = np.array([[0, 600], [1023, 601]], dtype=line_dtype)
    center = (np.uint16(500), np.uint16(450))
    radius = np.uint16(300)

    result = line_circle_intersection(baseline[0], baseline[1], center, radius)

    expected = _float_reference_intersections(baseline[0], baseline[1], center, 300)
    np.testing.assert_allclose(result, expected)


def test_line_circle_intersection_int32_products_do_not_overflow():
    # |f|^2 alone exceeds 2**31 here, so int32 arithmetic would wrap.
    p1 = np.array([0, 60000], dtype=np.int32)
    p2 = np.array([60000, 60001], dtype=np.int32)
    center = (np.uint16(30000), np.uint16(60000))

    result = line_circle_intersection(p1, p2, center, np.uint16(20000))

    expected = _float_reference_intersections(p1, p2, center, 20000)
    np.testing.assert_allclose(result, expected)


# ===== iterative_Lfit RMSE-peak ties =====
#
# PS3.BMP's right contact corner is symmetric, so two adjacent fit windows
# have mathematically equal RMSEs that ODR leaves ~1e-10 apart, in an order
# that differed between macOS and Windows/Linux and moved the contact point
# by 1px. CPID must give the same answer under that level of noise.

PS3_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "paper_results", "data", "experimental", "PS3.BMP",
)


@pytest.mark.skipif(not os.path.exists(PS3_PATH), reason="PS3.BMP not available")
def test_cpid_stable_under_odr_convergence_noise(monkeypatch):
    import cv2
    from opendrop_ml.modules.preprocessing import preprocessing

    _, contour, _, _ = isolate_contour(cv2.imread(PS3_PATH), DISPLAY=False, blur=True)
    expected = CPID(contour).tolist()

    real_linear_fit = preprocessing.linear_fit
    for seed in range(10):
        rng = np.random.default_rng(seed)

        def jittered_linear_fit(coords, display=False):
            outputs = real_linear_fit(coords)
            return tuple(
                v * (1 + rng.normal() * 1e-9) if np.isfinite(v) else v for v in outputs
            )

        monkeypatch.setattr(preprocessing, "linear_fit", jittered_linear_fit)
        assert CPID(contour).tolist() == expected
