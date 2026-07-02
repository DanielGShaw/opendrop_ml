from opendrop_ml.modules.fitting.circular_fit import (
    circular_fit,
    circular_fit_img,
    circle_fit_errors,
)
from opendrop_ml.modules.preprocessing.preprocessing import extract_edges_cv
from unittest.mock import patch
import cv2
import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Generate circular test data


def generate_circle_points(center, radius, num_points=100):
    """Generate a set of circular points for testing"""
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))


@pytest.fixture
def sample_circle():
    """Generate a standard circular dataset as test input"""
    return generate_circle_points(center=(50, 50), radius=20, num_points=100)


@pytest.fixture
def sample_image():
    """Create a test image containing a simulated circular boundary"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (255, 255, 255), thickness=1)
    return img


def test_circular_fit(sample_circle):
    """Test whether circular_fit correctly fits the circular data"""
    CA, center, R, intercepts, errors, timings = circular_fit(sample_circle)

    assert np.isclose(center[0], 50, atol=2), f"Expected center x≈50, got {center[0]}"
    assert np.isclose(center[1], 50, atol=2), f"Expected center y≈50, got {center[1]}"
    assert np.isclose(R, 20, atol=2), f"Expected radius ≈20, got {R}"


@patch("opendrop_ml.modules.fitting.circular_fit.preprocess")
def test_circular_fit_img(mock_preprocess, sample_image, sample_circle):
    """Test whether circular_fit_img can correctly extract edges and fit a circle"""
    CPs_chosen = np.array([sample_circle[0], sample_circle[-1]])
    mock_preprocess.return_value = (sample_image, CPs_chosen, sample_circle)

    CA, center, R, intercepts, errors, timings = circular_fit_img(sample_image)

    assert np.isclose(center[0], 50, atol=2), f"Expected center x≈50, got {center[0]}"
    assert np.isclose(center[1], 50, atol=2), f"Expected center y≈50, got {center[1]}"
    assert np.isclose(R, 20, atol=2), f"Expected radius ≈20, got {R}"


def test_circle_fit_errors(sample_circle):
    """Test whether circle_fit_errors correctly calculates the fitting errors"""
    errors = circle_fit_errors(sample_circle, h=50, k=50, r=20)

    assert np.isclose(errors["MAE"], 0, atol=0.5)
    assert np.isclose(errors["MSE"], 0, atol=0.5)
    assert np.isclose(errors["RMSE"], 0, atol=0.5)
    assert np.isclose(errors["Maximum error"], 0, atol=1)


def test_extract_edges_CV(sample_image):
    """Test whether extract_edges_CV correctly extracts edge points"""
    edges = extract_edges_cv(sample_image)
    assert len(edges) > 20, "Edge detection failed, not enough points detected"


# def test_empty_input():
#     """Test whether an empty input is handled correctly"""
#     with pytest.raises(ValueError):
#         circular_fit(np.array([]))


def test_invalid_shapes():
    """Test cases where the input data format is incorrect"""
    invalid_data = np.array([1, 2, 3])
    with pytest.raises(IndexError):
        circular_fit(invalid_data)


def test_circular_fit_no_intersection():
    """Test circular_fit with a contour whose fitted circle doesn't intersect the baseline.

    Simulates a near-180° drop on a reflective surface where edge detection
    places the contact points far below the actual drop body. The fitted circle
    hovers above the baseline with a gap >> 2px, triggering the early-return guard.
    """
    # Generate a 330° arc (near-complete circle) representing the drop body
    # Center at (50, 30), radius 20, arc from 15° to 345°
    angles = np.linspace(np.deg2rad(15), np.deg2rad(345), 80)
    arc_x = 50 + 20 * np.cos(angles)
    arc_y = 30 + 20 * np.sin(angles)

    # Place endpoints (contact points) far below the circle at y=60,
    # simulating scattered reflection contact points
    left_cp = np.array([30, 60])
    right_cp = np.array([70, 60])

    # Build contour: left_cp -> arc -> right_cp
    contour = np.vstack([left_cp, np.column_stack((arc_x, arc_y)), right_cp])

    CA, center_2, R_2, intercepts, errors, timings = circular_fit(contour)

    # Should return empty angles (not raise TypeError)
    assert CA == [], f"Expected empty CA list, got {CA}"
    # errors should be an error string, not a dict
    assert isinstance(errors, str), f"Expected error string, got {type(errors)}"
    # Circle parameters should still be returned
    assert center_2 is not None
    assert R_2 is not None
