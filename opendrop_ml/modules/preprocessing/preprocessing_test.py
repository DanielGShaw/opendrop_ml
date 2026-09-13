import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from opendrop_ml.modules.preprocessing.preprocessing import CPID, iterative_Lfit, isolate_contour

TRAINING_FILES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "training_files"
)
sys.path.insert(0, TRAINING_FILES_DIR)


# ===== iterative_Lfit sparse-contour fallback (PART2_REMAINING_WORK.md, 6.2) =====
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


# ===== CPID reproduction case (PART2_REMAINING_WORK.md, 6.2 Reproduction) =====
#
# angle=160, Bo=0.55, scaler=5.0, roughness=-0.015 is the deliberately-extreme
# cell where CPID's restricted contour collapsed to 13 points and crashed.
# The seed is `matrix_utils.cell_seed(160.0, 0.55, 5.0, -0.015)`, hardcoded
# here so this test doesn't depend on paper_results/offgrid/ (the study that
# found it, not shipped code) remaining in place.
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
