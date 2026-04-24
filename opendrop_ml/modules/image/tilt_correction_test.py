import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

from opendrop_ml.modules.image import tilt_correction as tc
from opendrop_ml.modules.image.tilt_correction import (
    _crop_rotation_borders,
    correct_tilt,
    detect_tilt_angle,
    touches_border,
)


def _image_with_line(angle_deg, size=500):
    """White image (H=W=size) with a thick black line through the centre.

    The drawn line is oriented ``angle_deg`` degrees below the horizontal
    axis (positive = clockwise tilt in image coordinates, matching the
    convention in ``detect_tilt_angle``).
    """
    img = np.full((size, size), 255, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    half = size  # long enough to guarantee > threshold votes
    theta = math.radians(angle_deg)
    dx = int(math.cos(theta) * half)
    dy = int(math.sin(theta) * half)
    cv2.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), color=0, thickness=3)
    return img


def test_detect_tilt_angle_happy_path():
    """A strong +5° line is recovered within ~1° tolerance."""
    img = _image_with_line(5.0)
    angle = detect_tilt_angle(img)
    assert abs(angle - 5.0) < 1.5


def test_detect_tilt_angle_no_line_returns_zero():
    """Uniform image with no detectable lines returns 0.0 fallback."""
    blank = np.full((200, 200), 128, dtype=np.uint8)
    assert detect_tilt_angle(blank) == 0.0


def test_correct_tilt_auto_detects_when_angle_none():
    """Passing tilt_angle=None triggers auto-detection via detect_tilt_angle."""
    img = _image_with_line(5.0)
    corrected, angle = correct_tilt(img, tilt_angle=None, crop=False)
    assert abs(angle - 5.0) < 1.5
    assert corrected.shape == img.shape  # reshape=False keeps dims


def test_correct_tilt_uses_supplied_angle(monkeypatch):
    """A supplied tilt_angle skips detection entirely."""

    def _should_not_be_called(_img):
        raise AssertionError("detect_tilt_angle must not run when angle is given")

    monkeypatch.setattr(tc, "detect_tilt_angle", _should_not_be_called)

    img = np.full((200, 200), 200, dtype=np.uint8)
    corrected, angle = correct_tilt(img, tilt_angle=3.0, crop=False)
    assert angle == 3.0
    assert corrected.shape == img.shape


def test_correct_tilt_crop_shrinks_shape():
    """crop=True removes the black borders and shrinks the frame vs crop=False."""
    img = np.full((400, 400), 200, dtype=np.uint8)
    uncropped, _ = correct_tilt(img, tilt_angle=10.0, crop=False)
    cropped, _ = correct_tilt(img, tilt_angle=10.0, crop=True)
    assert cropped.shape[0] < uncropped.shape[0]
    assert cropped.shape[1] < uncropped.shape[1]


def test_crop_rotation_borders_strips_expected_margin():
    """_crop_rotation_borders strips sin(|angle|)*edge pixels from each side."""
    h, w = 300, 400
    img = np.full((h, w), 200, dtype=np.uint8)
    angle = 10.0
    cropped = _crop_rotation_borders(img, angle)
    sin_a = abs(math.sin(math.radians(angle)))
    expected_h = h - 2 * math.ceil(sin_a * h)
    expected_w = w - 2 * math.ceil(sin_a * w)
    assert cropped.shape == (expected_h, expected_w)


def test_touches_border_no_touch():
    """Contour well inside the frame does not touch any border."""
    contour = np.array([[50, 50], [60, 60], [70, 55]], dtype=np.int32)
    assert touches_border(contour, (200, 200), margin=2) is False


def test_touches_border_left_edge_with_custom_margin():
    """Contour point at x=3 touches the border with margin=5 but not margin=2."""
    contour = np.array([[3, 100], [50, 100], [50, 150]], dtype=np.int32)
    assert touches_border(contour, (200, 200), margin=2) is False
    assert touches_border(contour, (200, 200), margin=5) is True


if __name__ == "__main__":
    pytest.main(["-v", __file__])
