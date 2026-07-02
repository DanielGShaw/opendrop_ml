"""Tests for prepare_experimental.py — normalize_ds and window extraction validation."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from opendrop_ml.modules.ML_model.prepare_experimental import (
    normalize_ds,
    prepare_for_model,
)


# Normalize_ds division by zero
class TestNormalizeDsCurrentBehaviour:
    """Tests for normalize_ds basic behaviour."""

    def test_normal_contour_normalizes_correctly(self):
        """Normal contour: apex should be at z=1 after normalization."""
        # Simple quarter-circle-like contour (in image coords: y increases downward)
        contour = np.array(
            [[0, 10], [1, 8], [2, 5], [3, 3], [4, 2], [5, 1], [6, 0]],
            dtype=float,
        )
        result = normalize_ds(np.array([contour]))

        # After y-flip, max y should be 1.0 (normalized apex)
        assert result.shape == (1, 7, 2)
        assert np.isclose(max(result[0, :, 1]), 1.0), (
            f"Apex should be at z=1 after normalization, got {max(result[0, :, 1])}"
        )

    def test_zero_height_raises_valueerror(self):
        """Zero-height contour should raise ValueError with helpful message."""
        # All y-values identical → height = 0 after y-flip
        flat_contour = np.array([[0, 5], [1, 5], [2, 5], [3, 5]], dtype=float)

        with pytest.raises(ValueError, match=r"(?i)(zero height|20.*180|supported range)"):
            normalize_ds(np.array([flat_contour]))


class TestNormalizeDsDesiredBehaviour:
    """Tests for normalize_ds error handling."""

    def test_zero_height_raises_valueerror_with_message(self):
        """Zero-height contour should raise ValueError with helpful message."""
        flat_contour = np.array([[0, 5], [1, 5], [2, 5], [3, 5]], dtype=float)

        with pytest.raises(ValueError, match=r"(?i)(zero height|20.*180|supported range)"):
            normalize_ds(np.array([flat_contour]))


# Window extraction validation
def _make_mock_contour(n_points, idx_left, idx_right):
    """Create a synthetic contour with CPs at specified indices.

    Returns (contour, cps) where contour is an (n_points, 2) array
    and cps is [left_cp, right_cp].
    """
    # Simple arc-like contour: x goes 0..n_points-1, y is a parabola
    x = np.arange(n_points, dtype=float)
    y = -(((x - n_points / 2) / (n_points / 2)) ** 2) * 50 + 100  # inverted parabola
    contour = np.column_stack([x, y])
    cps = [contour[idx_left].copy(), contour[idx_right].copy()]
    return contour, cps


def _mock_preprocess_factory(contour, cps):
    """Return a mock preprocess function that returns the given contour and CPs."""

    def mock_preprocess(image, display=False, blur=True):
        return image, cps, contour

    return mock_preprocess


def _patch_preprocessing(mock_preprocess):
    """Context manager to patch both preprocess and circular_fit (local imports)."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        with patch(
            "opendrop_ml.modules.preprocessing.preprocessing.preprocess",
            side_effect=mock_preprocess,
        ), patch(
            "opendrop_ml.modules.fitting.circular_fit.circular_fit",
            return_value=((90.0, 90.0), None, None, None, None, None),
        ):
            yield

    return _ctx()


class TestWindowExtractionCurrentBehaviour:
    """Tests for window extraction basic behaviour."""

    def test_normal_extraction_gives_correct_shape(self):
        """Normal case: plenty of contour points, CPs well inside → (2, 50, 2)."""
        contour, cps = _make_mock_contour(200, idx_left=50, idx_right=150)
        mock_preprocess = _mock_preprocess_factory(contour, cps)
        img = np.zeros((100, 200, 3), dtype=np.uint8)

        with _patch_preprocessing(mock_preprocess):
            result = prepare_for_model(img, input_len=50, n_surface=10)

        assert result.shape == (2, 50, 2), f"Expected (2, 50, 2), got {result.shape}"
        assert result.dtype == np.float32

    def test_short_contour_raises_valueerror(self):
        """Contour shorter than input_len should raise ValueError."""
        # 30-point contour, CPs at indices 5 and 25
        contour, cps = _make_mock_contour(30, idx_left=5, idx_right=25)
        mock_preprocess = _mock_preprocess_factory(contour, cps)
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        with _patch_preprocessing(mock_preprocess):
            with pytest.raises(ValueError, match=r"(?i)(expected.*50|length|shape)"):
                prepare_for_model(img, input_len=50, n_surface=10)

    def test_cp_at_edge_raises_valueerror(self):
        """CP at index 2 (only 2 surface points) should raise ValueError."""
        # 200-point contour, left CP at index 2 (only 2 surface points)
        contour, cps = _make_mock_contour(200, idx_left=2, idx_right=150)
        mock_preprocess = _mock_preprocess_factory(contour, cps)
        img = np.zeros((100, 200, 3), dtype=np.uint8)

        with _patch_preprocessing(mock_preprocess):
            with pytest.raises(ValueError, match=r"(?i)(surface points|position|valid)"):
                prepare_for_model(img, input_len=50, n_surface=10)


class TestWindowExtractionDesiredBehaviour:
    """Tests for window extraction validation and error handling."""

    def test_short_contour_raises_valueerror(self):
        """Contour shorter than input_len should raise ValueError."""
        contour, cps = _make_mock_contour(30, idx_left=5, idx_right=25)
        mock_preprocess = _mock_preprocess_factory(contour, cps)
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        with _patch_preprocessing(mock_preprocess):
            with pytest.raises(ValueError, match=r"(?i)(expected.*50|length|shape)"):
                prepare_for_model(img, input_len=50, n_surface=10)

    def test_cp_too_close_to_edge_raises_valueerror(self):
        """CP with < 5 surface points should raise ValueError."""
        contour, cps = _make_mock_contour(200, idx_left=2, idx_right=150)
        mock_preprocess = _mock_preprocess_factory(contour, cps)
        img = np.zeros((100, 200, 3), dtype=np.uint8)

        with _patch_preprocessing(mock_preprocess):
            with pytest.raises(ValueError, match=r"(?i)(surface points|position|valid)"):
                prepare_for_model(img, input_len=50, n_surface=10)

    def test_cp_at_minimum_valid_boundary_ok(self):
        """CP with exactly 5 surface points (boundary) should succeed."""
        contour, cps = _make_mock_contour(200, idx_left=5, idx_right=150)
        mock_preprocess = _mock_preprocess_factory(contour, cps)
        img = np.zeros((100, 200, 3), dtype=np.uint8)

        with _patch_preprocessing(mock_preprocess):
            result = prepare_for_model(img, input_len=50, n_surface=10)
            assert result.shape == (2, 50, 2)
            assert result.dtype == np.float32


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
