import math

import cv2
import numpy as np
from scipy import ndimage


def detect_tilt_angle(img):
    """Detect substrate tilt angle using Canny + Hough line transform.

    Args:
        img: BGR or grayscale image (np.ndarray).

    Returns:
        Tilt angle in degrees (positive = clockwise tilt in image coords).
        Returns 0.0 if no baseline is detected.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    if lines is None:
        return 0.0

    # Take the strongest (first) detected line
    rho, theta = lines[0][0]

    # Convert to angle: theta is in radians from the vertical axis
    # A horizontal line has theta ~= pi/2
    angle = math.degrees(theta) - 90.0

    return angle


def correct_tilt(img, tilt_angle=None, crop=True):
    """Rotate image to correct for substrate tilt.

    Args:
        img: BGR or grayscale image (np.ndarray).
        tilt_angle: Tilt angle in degrees. If None, auto-detects.
        crop: If True, crop black borders created by rotation.

    Returns:
        (corrected_image, tilt_angle) tuple.
    """
    if tilt_angle is None:
        tilt_angle = detect_tilt_angle(img)

    if abs(tilt_angle) < 0.1:
        return img.copy(), tilt_angle

    # Handle angles near ±90°
    rotate_angle = tilt_angle
    if rotate_angle > 89:
        rotate_angle = -90 + rotate_angle
    elif rotate_angle < -89:
        rotate_angle = 90 + rotate_angle

    rotated = ndimage.rotate(img, rotate_angle, reshape=False)

    if crop:
        rotated = _crop_rotation_borders(rotated, rotate_angle)

    return rotated, tilt_angle


def _crop_rotation_borders(img, angle_deg):
    """Crop black borders introduced by image rotation.

    Args:
        img: Rotated image.
        angle_deg: Rotation angle in degrees.

    Returns:
        Cropped image with black borders removed.
    """
    angle_rad = np.deg2rad(abs(angle_deg))
    h, w = img.shape[:2]

    sin_a = abs(np.sin(angle_rad))

    side = math.ceil(sin_a * w)
    roof = math.ceil(sin_a * h)

    # Ensure we don't over-crop
    if side >= w // 2 or roof >= h // 2:
        return img

    if roof > 0 and side > 0:
        return img[roof:-roof, side:-side]
    elif roof > 0:
        return img[roof:-roof, :]
    elif side > 0:
        return img[:, side:-side]
    return img


def touches_border(contour, img_shape, margin=2):
    """Check if any contour point is within margin pixels of image border.

    Used to detect frames where the drop extends beyond the image edge
    (i.e., the drop is not fully visible).

    Args:
        contour: Nx2 array of [x, y] coordinates.
        img_shape: (height, width, ...) of the image.
        margin: Distance from edge in pixels.

    Returns:
        True if contour touches the border.
    """
    h, w = img_shape[:2]

    x = contour[:, 0]
    y = contour[:, 1]

    if (np.any(x <= margin) or np.any(x >= w - 1 - margin) or
            np.any(y <= margin) or np.any(y >= h - 1 - margin)):
        return True
    return False
