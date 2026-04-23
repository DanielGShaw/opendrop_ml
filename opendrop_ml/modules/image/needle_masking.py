"""Needle detection and masking for needle-in-drop video analysis.

Public API:
    - extract_edges_two_contours: extract the two largest contours
    - hough_baseline: detect horizontal baseline via standard Hough transform
    - detect_vertical_lines: detect two vertical needle lines
    - remove_points_near_lines: remove contour points near detected lines
    - mask_needle: orchestrator combining detection + removal
"""

import numpy as np
import cv2


IGNORE_EDGE_MARGIN = 1  # Margin to ignore edge points near image borders


def extract_edges_two_contours(img, threshold_val=None, blur=False):
    """Extract the two largest contours from an image.

    Uses Otsu thresholding if threshold_val is not provided. Filters
    out points near image borders (IGNORE_EDGE_MARGIN).

    Args:
        img: BGR or grayscale image (np.ndarray).
        threshold_val: Manual threshold value, or None for Otsu.
        blur: If True, apply 3x3 Gaussian blur before thresholding.

    Returns:
        (contour1, contour2, threshold_used) where contour1 is the longest
        and both are Nx2 arrays of [x, y] coordinates.
    """
    img = img.astype("uint8")

    if len(img.shape) == 3:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = img
    else:
        gray = img.copy()

    if blur:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if threshold_val is None:
        thresh_val, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        thresh_val = threshold_val
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(contours) < 2:
        raise ValueError(
            "Expected at least 2 contours, found {}".format(len(contours))
        )

    contours = [c.reshape(-1, 2) for c in contours]

    contours.sort(
        key=lambda c: cv2.arcLength(c.reshape(-1, 1, 2), closed=False),
        reverse=True,
    )

    profiles = contours[:2]

    height, width = img.shape[:2]
    for i, drop_profile in enumerate(profiles):
        if width >= IGNORE_EDGE_MARGIN and height >= IGNORE_EDGE_MARGIN:
            mask = (
                (IGNORE_EDGE_MARGIN < drop_profile[:, 0])
                & (drop_profile[:, 0] < width - IGNORE_EDGE_MARGIN)
                & (IGNORE_EDGE_MARGIN < drop_profile[:, 1])
                & (drop_profile[:, 1] < height - IGNORE_EDGE_MARGIN)
            )
            drop_profile = drop_profile[mask]

        keep = np.ones(len(drop_profile), dtype=bool)
        keep[1:] = np.any(drop_profile[1:] != drop_profile[:-1], axis=1)
        profiles[i] = drop_profile[keep]

    return profiles[0], profiles[1], thresh_val


def hough_baseline(edges_img, threshold=40, vertical_threshold=10):
    """Detect a horizontal baseline using the standard Hough Line Transform.

    Args:
        edges_img: Binary edge image.
        threshold: Accumulator threshold for HoughLines.
        vertical_threshold: Maximum deviation from horizontal (degrees)
            for a line to be considered a baseline candidate.

    Returns:
        Baseline as [[x1, y1], [x2, y2]] array, or None if not found.
    """
    lines = cv2.HoughLines(edges_img, 1, np.pi / 180, threshold)

    if lines is None:
        return None

    filtered_lines = []
    for line in lines:
        rho, theta = line[0]
        if abs(theta * 180 / np.pi - 90) < vertical_threshold:
            filtered_lines.append(line)

    if not filtered_lines:
        return None

    converted = []
    for line in filtered_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - edges_img.shape[1] * (-b))
        y2 = int(y0 - 1000 * a)
        converted.append([[x1, y1], [x2, y2]])

    def gradient_key(line_pts):
        lx1, ly1 = line_pts[0]
        lx2, ly2 = line_pts[1]
        if lx1 == lx2:
            return float("inf")
        return abs((ly2 - ly1) / (lx2 - lx1))

    converted.sort(key=gradient_key)
    return np.array(converted[0])


def detect_vertical_lines(
    edges_img, hough_threshold=100, vertical_angle_threshold=30
):
    """Detect two vertical needle lines using Hough Line Transform.

    Uses merge-count sorting to distinguish real needle lines (many
    overlapping detections) from border artifacts (few detections).

    Args:
        edges_img: Binary edge image.
        hough_threshold: Accumulator threshold for HoughLinesP.
        vertical_angle_threshold: Maximum angle deviation from vertical
            (degrees) to consider a line as a needle candidate.

    Returns:
        (left_ext, right_ext, left_seg, right_seg) where ext are
        [[x1, y1], [x2, y2]] extended to full image height, and seg
        are [x1, y1, x2, y2] original segments. Returns
        (None, None, None, None) if detection fails.
    """
    h, w = edges_img.shape[:2]
    min_line_length = int(h * 0.05)
    max_line_gap = int(w * 0.01)
    merge_threshold = int(w * 0.01)

    lines = cv2.HoughLinesP(
        edges_img,
        1,
        np.pi / 180,
        hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        return None, None, None, None

    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if (
            90 - vertical_angle_threshold
            <= angle
            <= 90 + vertical_angle_threshold
        ):
            vertical_lines.append([x1, y1, x2, y2])

    if len(vertical_lines) < 2:
        return None, None, None, None

    merged_lines = []
    merge_counts = []
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        should_merge = False
        for i, merged_line in enumerate(merged_lines):
            mx1, my1, mx2, my2 = merged_line
            if (
                abs(mx1 - x1) < merge_threshold
                and abs(mx2 - x2) < merge_threshold
            ):
                should_merge = True
                if max(y1, y2) > max(my1, my2):
                    merged_lines[i] = line
                merge_counts[i] += 1
                break
        if not should_merge:
            merged_lines.append(line)
            merge_counts.append(1)

    if len(merged_lines) < 2:
        return None, None, None, None

    sorted_with_counts = sorted(
        zip(merged_lines, merge_counts),
        key=lambda lc: lc[1],
        reverse=True,
    )
    merged_lines = [line for line, count in sorted_with_counts]
    merged_lines = sorted(merged_lines, key=lambda line: line[1])

    using = sorted(merged_lines[:2], key=lambda line: line[0])

    extended_lines = []
    for x1, y1, x2, y2 in using[:2]:
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            new_x1 = int((0 - b) / m) if m != 0 else x1
            new_x2 = int((h - 1 - b) / m) if m != 0 else x1
        else:
            new_x1 = new_x2 = x1
        extended_lines.append([new_x1, 0, new_x2, h - 1])

    if len(extended_lines) < 2:
        return None, None, None, None

    for i, line in enumerate(extended_lines):
        extended_lines[i][0] = max(0, min(w - 1, line[0]))
        extended_lines[i][2] = max(0, min(w - 1, line[2]))

    left_ext = np.array(
        [
            [extended_lines[0][0], extended_lines[0][1]],
            [extended_lines[0][2], extended_lines[0][3]],
        ]
    )
    right_ext = np.array(
        [
            [extended_lines[-1][0], extended_lines[-1][1]],
            [extended_lines[-1][2], extended_lines[-1][3]],
        ]
    )

    return left_ext, right_ext, using[0], using[1]


def remove_points_near_lines(contour, lines, distance_threshold=2.0):
    """Remove contour points within distance_threshold of any line.

    Args:
        contour: Nx2 array of [x, y] coordinates.
        lines: List of [[x1, y1], [x2, y2]] line segments.
        distance_threshold: Maximum distance (pixels) to remove points.

    Returns:
        Filtered contour as Nx2 array.
    """
    if len(contour) == 0 or not lines:
        return contour

    mask = np.ones(len(contour), dtype=bool)

    for line in lines:
        if line is None:
            continue
        p1 = np.array(line[0], dtype=float)
        p2 = np.array(line[1], dtype=float)

        d = p2 - p1
        d_sq = np.dot(d, d)

        if d_sq < 1e-10:
            continue

        pts = contour.astype(float)
        v = pts - p1

        t = np.dot(v, d) / d_sq
        t = np.clip(t, 0, 1)

        closest = p1 + t[:, np.newaxis] * d
        distances = np.sqrt(np.sum((pts - closest) ** 2, axis=1))

        mask &= distances > distance_threshold

    return contour[mask]


def mask_needle(
    left_contour,
    right_contour,
    img_shape,
    hough_threshold=40,
    distance_threshold=5.0,
    display=False,
):
    """Detect needle lines and remove nearby points from drop contours.

    Main orchestrator: creates an edge image from the contours, detects
    vertical needle lines, removes points near those lines (and near
    the baseline), then returns cleaned contours.

    Args:
        left_contour: Nx2 array for left drop edge.
        right_contour: Mx2 array for right drop edge.
        img_shape: (height, width) of the original image.
        hough_threshold: Threshold for Hough line detection.
        distance_threshold: Pixel distance for point removal.
        display: If True, show debug plots.

    Returns:
        (left_isolated, right_isolated, baseline, left_line, right_line)
        where isolated contours have needle/baseline points removed.
    """
    h, w = img_shape[:2]

    edge_img = np.zeros((h, w), dtype=np.uint8)
    for pt in left_contour:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            edge_img[y, x] = 255
    for pt in right_contour:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            edge_img[y, x] = 255

    baseline = hough_baseline(edge_img, threshold=hough_threshold)

    left_line, right_line, _, _ = detect_vertical_lines(
        edge_img, hough_threshold=hough_threshold
    )

    lines_to_remove = []
    if baseline is not None:
        lines_to_remove.append(baseline.tolist())
    if left_line is not None:
        lines_to_remove.append(left_line.tolist())
    if right_line is not None:
        lines_to_remove.append(right_line.tolist())

    left_isolated = remove_points_near_lines(
        left_contour, lines_to_remove, distance_threshold
    )
    right_isolated = remove_points_near_lines(
        right_contour, lines_to_remove, distance_threshold
    )

    if display:
        _plot_debug(
            edge_img,
            left_contour,
            right_contour,
            left_isolated,
            right_isolated,
            baseline,
            left_line,
            right_line,
        )

    return left_isolated, right_isolated, baseline, left_line, right_line


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _plot_debug(
    edge_img,
    left_orig,
    right_orig,
    left_iso,
    right_iso,
    baseline,
    left_line,
    right_line,
):
    """Show debug visualisation of needle masking results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.imshow(edge_img, cmap="gray")
    ax1.set_title("Edge image + detected lines")
    if baseline is not None:
        ax1.plot(
            baseline[:, 0], baseline[:, 1], "g-", linewidth=2, label="Baseline"
        )
    if left_line is not None:
        ax1.plot(
            [left_line[0][0], left_line[1][0]],
            [left_line[0][1], left_line[1][1]],
            "r-",
            linewidth=2,
            label="Left needle",
        )
    if right_line is not None:
        ax1.plot(
            [right_line[0][0], right_line[1][0]],
            [right_line[0][1], right_line[1][1]],
            "b-",
            linewidth=2,
            label="Right needle",
        )
    ax1.legend()

    ax2.set_title("Isolated contours (after removal)")
    if len(left_iso) > 0:
        ax2.plot(
            left_iso[:, 0], left_iso[:, 1], ".", markersize=1, label="Left"
        )
    if len(right_iso) > 0:
        ax2.plot(
            right_iso[:, 0], right_iso[:, 1], ".", markersize=1, label="Right"
        )
    ax2.legend()
    ax2.set_aspect("equal")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.show()
    plt.close()
