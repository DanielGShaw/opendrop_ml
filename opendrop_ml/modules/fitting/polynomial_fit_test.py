from opendrop_ml.modules.fitting.polynomial_fit import (
    polynomial_closest_point,
    polynomial_fit_errors,
    polynomial_fit_img,
    polynomial_fit,
)
from opendrop_ml.modules.preprocessing.preprocessing import extract_edges_cv

from unittest.mock import patch
import numpy as np
import pytest
import sys
import os
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

warnings.filterwarnings("ignore", category=DeprecationWarning)

KNOWN_ANGLE_NPZ = os.path.join(
    current_dir, "test_data", "polynomial_fit_known_angle.npz"
)


@pytest.fixture
def sample_data():
    return np.array([[1, 1], [1.1, 1.1], [0, 5], [0.1, 5.1], [5, 0], [5.1, 0.1]])


@pytest.fixture
def sample_profile():
    return np.array([[0, 0], [1, 1], [2, 2], [3, 1], [4, 0]])


@patch("opendrop_ml.modules.preprocessing.preprocessing.cv2.cvtColor")
@patch("opendrop_ml.modules.preprocessing.preprocessing.cv2.threshold")
@patch("opendrop_ml.modules.preprocessing.preprocessing.cv2.findContours")
def test_extract_edges_cv(mock_findContours, mock_threshold, mock_cvtColor):
    mock_cvtColor.return_value = np.zeros((10, 10))
    mock_threshold.return_value = (None, np.zeros((10, 10)))
    mock_findContours.return_value = ([np.array([[[0, 0]], [[1, 1]]])], None)

    test_img = np.zeros((10, 10, 3))
    result = extract_edges_cv(test_img)
    assert isinstance(result, np.ndarray)
    assert result.ndim in [1, 2]


def test_polynomial_closest_point():
    poly_points = np.array([[0, 0], [1, 1], [2, 2]])
    dist, point = polynomial_closest_point(0.5, 0.5, poly_points)
    assert isinstance(dist, (int, float))
    assert len(point) == 2


def test_polynomial_fit_errors():
    pts1 = np.array([[0, 0], [1, 1]])
    pts2 = np.array([[2, 2], [3, 3]])
    fit_left = np.poly1d([1, 0])  # y = x
    fit_right = np.poly1d([1, 0])  # y = x

    errors = polynomial_fit_errors(pts1, pts2, fit_left, fit_right)
    assert isinstance(errors, dict)
    assert "MAE" in errors
    assert "MSE" in errors
    assert "RMSE" in errors


@patch("opendrop_ml.modules.fitting.polynomial_fit.preprocess")
def test_polynomial_fit_img(mock_preprocess, sample_data, sample_profile):
    CPs_chosen = np.array([sample_profile[0], sample_profile[-1]])
    mock_preprocess.return_value = (
        np.zeros((10, 10, 3)),
        CPs_chosen,
        sample_profile,
    )

    img = np.zeros((10, 10, 3))
    angles, intercepts, errors, timings = polynomial_fit_img(img)

    assert len(angles) == 2
    assert len(intercepts) == 2
    assert isinstance(errors, dict)
    assert isinstance(timings, dict)


def test_polynomial_fit(sample_profile):
    angles, CPs, tangent_lines, errors, timings = polynomial_fit(sample_profile)

    assert len(angles) == 2
    assert len(CPs) == 2
    assert len(tangent_lines) == 2
    assert isinstance(errors, dict)
    assert isinstance(timings, dict)


@pytest.mark.parametrize("expected_ca", [20.0, 60.0, 100.0, 140.0])
def test_polynomial_fit_known_angle(expected_ca):
    """Regression: polynomial_fit returns the correct contact angle for synthetic
    drop profiles with known ground-truth angles.

    The 20 deg case guards against a supplement error where low-angle drops
    could return ~180 deg - theta (~160 deg) instead of ~20 deg.

    Profiles were extracted once from the paper_results/data/synthetic images
    via conan_v2/validation/build_polynomial_fit_npz.py; the npz ships with
    the package so the test runs on every fresh clone.
    """
    data = np.load(KNOWN_ANGLE_NPZ)
    key = "angle_{}".format(int(expected_ca))
    profile = data["{}_profile".format(key)]

    angles, CPs, tangent_lines, errors, timings = polynomial_fit(profile)

    assert angles[0] == pytest.approx(
        expected_ca, abs=5.0
    ), f"Left angle {angles[0]:.1f}° differs from expected {expected_ca}° by more than 5°"
    assert angles[1] == pytest.approx(
        expected_ca, abs=5.0
    ), f"Right angle {angles[1]:.1f}° differs from expected {expected_ca}° by more than 5°"
