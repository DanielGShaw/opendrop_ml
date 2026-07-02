import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from opendrop_ml.modules.image.needle_masking import (
    IGNORE_EDGE_MARGIN,
    detect_vertical_lines,
    extract_edges_two_contours,
    hough_baseline,
    mask_needle,
    remove_points_near_lines,
)


def _make_two_blob_image(size=100):
    """Black background with two bright rectangular blobs.

    Inverted polarity so cv2.findContours (which finds contours of
    non-zero pixels) returns exactly one contour per blob — no image-
    border contour to confuse the length sort.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (35, 35), color=255, thickness=-1)
    cv2.rectangle(img, (60, 60), (90, 90), color=255, thickness=-1)
    return img


def _make_horizontal_edge_image(size=120, y=60, length=None):
    """Binary edge image with a single horizontal line of `255` pixels."""
    img = np.zeros((size, size), dtype=np.uint8)
    x_end = length if length is not None else size
    cv2.line(img, (0, y), (x_end - 1, y), color=255, thickness=1)
    return img


def _make_two_vertical_edge_image(size=200, x_left=50, x_right=150):
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.line(img, (x_left, 10), (x_left, size - 10), color=255, thickness=1)
    cv2.line(img, (x_right, 10), (x_right, size - 10), color=255, thickness=1)
    return img


# ---------------------------------------------------------------------------
# extract_edges_two_contours
# ---------------------------------------------------------------------------


def test_extract_edges_two_contours_returns_two_contours():
    img = _make_two_blob_image()
    c1, c2, thresh = extract_edges_two_contours(img)
    assert c1.ndim == 2 and c1.shape[1] == 2
    assert c2.ndim == 2 and c2.shape[1] == 2
    # Longest contour first.
    assert len(c1) >= len(c2)
    # Otsu can return 0 for purely binary test images — just confirm
    # the threshold is numeric and in the valid uint8 range.
    assert 0 <= thresh <= 255


def test_extract_edges_two_contours_blur_branch_still_returns_two():
    img = _make_two_blob_image()
    c1, c2, _ = extract_edges_two_contours(img, blur=True)
    assert len(c1) > 0 and len(c2) > 0


def test_extract_edges_two_contours_single_blob_raises():
    img = np.zeros((60, 60), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (35, 35), color=255, thickness=-1)
    with pytest.raises(ValueError, match="at least 2 contours"):
        extract_edges_two_contours(img)


# ---------------------------------------------------------------------------
# hough_baseline
# ---------------------------------------------------------------------------


def test_hough_baseline_detects_horizontal_line():
    edges = _make_horizontal_edge_image(size=120, y=60)
    baseline = hough_baseline(edges, threshold=40)
    assert baseline is not None
    # Horizontal line — both endpoints should share roughly the same y.
    assert abs(baseline[0, 1] - baseline[1, 1]) <= 2
    assert abs(baseline[0, 1] - 60) <= 2


def test_hough_baseline_empty_image_returns_none():
    edges = np.zeros((80, 80), dtype=np.uint8)
    assert hough_baseline(edges) is None


def test_hough_baseline_only_vertical_lines_returns_none():
    edges = np.zeros((120, 120), dtype=np.uint8)
    cv2.line(edges, (60, 0), (60, 119), color=255, thickness=1)
    # Vertical line is filtered out by the 10° horizontal tolerance.
    assert hough_baseline(edges, threshold=40) is None


# ---------------------------------------------------------------------------
# detect_vertical_lines
# ---------------------------------------------------------------------------


def test_detect_vertical_lines_happy_case():
    edges = _make_two_vertical_edge_image(size=200, x_left=50, x_right=150)
    left_ext, right_ext, left_seg, right_seg = detect_vertical_lines(
        edges, hough_threshold=30
    )
    assert left_ext is not None and right_ext is not None
    # Extended lines span full image height.
    assert left_ext[0, 1] == 0
    assert left_ext[1, 1] == 199
    # Left line's x ~ 50; right line's x ~ 150.
    assert abs(left_ext[0, 0] - 50) <= 3
    assert abs(right_ext[0, 0] - 150) <= 3
    # Original segments are [x1, y1, x2, y2].
    assert len(left_seg) == 4 and len(right_seg) == 4


def test_detect_vertical_lines_fewer_than_two_returns_none_tuple():
    edges = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(edges, (50, 10), (50, 190), color=255, thickness=1)
    result = detect_vertical_lines(edges, hough_threshold=30)
    assert result == (None, None, None, None)


def test_detect_vertical_lines_empty_image_returns_none_tuple():
    edges = np.zeros((200, 200), dtype=np.uint8)
    assert detect_vertical_lines(edges) == (None, None, None, None)


# ---------------------------------------------------------------------------
# remove_points_near_lines
# ---------------------------------------------------------------------------


def test_remove_points_near_lines_drops_close_points_keeps_far_points():
    # 5 contour points; one sits exactly on the line y=5, the rest are far.
    contour = np.array(
        [
            [0.0, 0.0],
            [10.0, 5.0],  # on the line
            [20.0, 100.0],
            [30.0, 100.0],
            [40.0, 100.0],
        ]
    )
    line = [[-5.0, 5.0], [50.0, 5.0]]  # horizontal at y=5
    filtered = remove_points_near_lines(contour, [line], distance_threshold=1.0)
    # One point dropped.
    assert len(filtered) == 4
    # Point at y=5 is gone.
    assert not np.any(np.all(filtered == [10.0, 5.0], axis=1))


def test_remove_points_near_lines_empty_contour_returns_empty():
    contour = np.empty((0, 2))
    line = [[0.0, 0.0], [10.0, 0.0]]
    out = remove_points_near_lines(contour, [line])
    assert len(out) == 0


def test_remove_points_near_lines_no_lines_returns_original():
    contour = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = remove_points_near_lines(contour, [])
    np.testing.assert_array_equal(out, contour)


def test_remove_points_near_lines_none_lines_skipped():
    contour = np.array([[1.0, 2.0], [3.0, 4.0]])
    # `None` entries must be skipped without error.
    out = remove_points_near_lines(contour, [None], distance_threshold=1.0)
    np.testing.assert_array_equal(out, contour)


# ---------------------------------------------------------------------------
# mask_needle — minimal integration
# ---------------------------------------------------------------------------


def test_mask_needle_removes_points_near_detected_vertical_lines():
    # Build two "contours" that together trace a baseline + two vertical
    # needle lines, so the orchestrator's Hough detection can find them.
    size = 200
    baseline_y = 150

    left_contour = np.array(
        [[x, baseline_y] for x in range(30, 60)]  # baseline strokes
        + [[50, y] for y in range(30, 150)]  # left needle line
    )
    right_contour = np.array(
        [[x, baseline_y] for x in range(140, 170)]
        + [[150, y] for y in range(30, 150)]  # right needle line
    )

    left_iso, right_iso, baseline, left_line, right_line = mask_needle(
        left_contour,
        right_contour,
        (size, size),
        hough_threshold=30,
        distance_threshold=3.0,
    )

    # At least one detected line, and isolated contours shorter than input.
    assert baseline is not None or left_line is not None or right_line is not None
    assert len(left_iso) < len(left_contour)
    assert len(right_iso) < len(right_contour)


# ---------------------------------------------------------------------------
# module constant
# ---------------------------------------------------------------------------


def test_ignore_edge_margin_is_positive_int():
    assert isinstance(IGNORE_EDGE_MARGIN, int)
    assert IGNORE_EDGE_MARGIN >= 1


if __name__ == "__main__":
    pytest.main(["-v", __file__])
