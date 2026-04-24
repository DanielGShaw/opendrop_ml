"""Contact-line utilities for the video-mode two-pass pipeline.

`fit_contact_line` fits an ODR line through IQR-filtered contact points
(Pass 1 calibration); `find_first_intercept` and `find_last_intercept`
locate the first/last contour segments that cross a fitted line (Pass 2
stable-CP placement).
"""

import math

import numpy as np

from opendrop_ml.modules.preprocessing.preprocessing import linear_fit


def fit_contact_line(cps):
    """Fit a linear contact line through contact points with IQR filtering.

    Filters contact points to the y-coordinate IQR range (25th-75th
    percentile), then fits a line via ODR (`linear_fit`). Falls back to
    a flat median line if the fit fails or too few points survive
    filtering.

    Args:
        cps: Nx2 array of [x, y] contact points collected across Pass 1.

    Returns:
        2x2 numpy array [[x_min, y_at_x_min], [x_max, y_at_x_max]]
        describing a line segment spanning the original x range.
    """
    cps = np.asarray(cps)
    q1 = np.percentile(cps[:, 1], 25)
    q3 = np.percentile(cps[:, 1], 75)
    cps_inrange = np.array([cp for cp in cps if q1 <= cp[1] <= q3])

    if len(cps_inrange) < 2:
        cps_inrange = cps

    start_x = cps[:, 0].min()
    end_x = cps[:, 0].max()

    try:
        m, c, _ = linear_fit(cps_inrange, display=False)
    except Exception:
        med = np.median(cps, axis=0)
        return np.array([[start_x, med[1]], [end_x, med[1]]])

    return np.array([[start_x, m * start_x + c], [end_x, m * end_x + c]])


def _perp_residuals(points, m, c):
    """Perpendicular distance from each (x, y) in points to y = m*x + c.

    Uses the standard point-line distance formula |m*x - y + c| /
    sqrt(m**2 + 1). Returned as a 1-D float array. A non-finite
    ``m`` or ``c`` (linear_fit degenerated on identical / collinear
    input) yields an all-NaN result so the downstream residual IQR
    becomes NaN and the Pass-2 outlier gate disables itself on that
    side.
    """
    pts = np.asarray(points, dtype=float)
    if not (math.isfinite(m) and math.isfinite(c)):
        return np.full(len(pts), float("nan"))
    return np.abs(m * pts[:, 0] - pts[:, 1] + c) / math.sqrt(m * m + 1.0)


def fit_contact_line_with_stats(cps):
    """Fit a linear contact line and return per-side IQR stats.

    Identical line fit to :func:`fit_contact_line`, plus a ``stats``
    dict the two-pass tilted pipeline's Pass-2 outlier gate uses to
    decide whether a per-frame CPID CP is an outlier against the
    Pass-1 median line. The IQR is computed over perpendicular
    residuals of the post-IQR-filtered CPs — the "well-behaved"
    population that fed the ODR fit.

    Args:
        cps: Nx2 array of [x, y] contact points collected across Pass 1.

    Returns:
        Tuple ``(line, stats)``:
        - ``line``: 2x2 array [[x_min, y_at_x_min], [x_max, y_at_x_max]]
          (same shape as :func:`fit_contact_line`).
        - ``stats``: dict with:
            - ``"residual_iqr"``: IQR (Q3 - Q1) of perpendicular
              residuals of the post-filter CPs from the fitted line.
              ``float("nan")`` when ``linear_fit`` raises or fewer than
              two post-filter points remain.
            - ``"n_filtered"``: count of post-filter CPs used.
            - ``"n_total"``: count of input CPs.
    """
    cps = np.asarray(cps)
    n_total = len(cps)
    q1 = np.percentile(cps[:, 1], 25)
    q3 = np.percentile(cps[:, 1], 75)
    cps_inrange = np.array([cp for cp in cps if q1 <= cp[1] <= q3])

    if len(cps_inrange) < 2:
        cps_inrange = cps

    start_x = cps[:, 0].min()
    end_x = cps[:, 0].max()
    n_filtered = len(cps_inrange)

    try:
        m, c, _ = linear_fit(cps_inrange, display=False)
    except Exception:
        med = np.median(cps, axis=0)
        line = np.array([[start_x, med[1]], [end_x, med[1]]])
        return line, {
            "residual_iqr": float("nan"),
            "n_filtered": 0,
            "n_total": n_total,
        }

    residuals = _perp_residuals(cps_inrange, m, c)
    r_q1 = np.percentile(residuals, 25)
    r_q3 = np.percentile(residuals, 75)
    residual_iqr = float(r_q3 - r_q1)

    line = np.array([[start_x, m * start_x + c], [end_x, m * end_x + c]])
    return line, {
        "residual_iqr": residual_iqr,
        "n_filtered": n_filtered,
        "n_total": n_total,
    }


def find_first_intercept(contour, line_points):
    """Find the index of the first contour-line intersection.

    Args:
        contour: Nx2 array of ordered [x, y] coordinates.
        line_points: [[x1, y1], [x2, y2]] defining the line segment
            (actual segment endpoints; intersection is tested with a
            relaxed parametric tolerance so the line is effectively
            treated as infinite for the contour's x range).

    Returns:
        Index of the first intersecting segment, or -1 if none.
    """
    line_p1, line_p2 = line_points
    n_points = len(contour)

    for i in range(n_points - 1):
        intersection = _line_intersection(
            contour[i], contour[i + 1], line_p1, line_p2
        )
        if intersection is not None:
            return i
    return -1


def find_last_intercept(contour, line_points, max_y=None):
    """Find the index of the last contour-line intersection.

    Iterates backward through the contour.

    Args:
        contour: Nx2 array of ordered [x, y] coordinates.
        line_points: [[x1, y1], [x2, y2]] defining the line.
        max_y: If provided, only return intersections with y <= max_y.

    Returns:
        Index of the last intersecting segment, or -1.
    """
    line_p1, line_p2 = line_points
    n_points = len(contour)

    for i in range(n_points - 2, -1, -1):
        intersection = _line_intersection(
            contour[i], contour[i + 1], line_p1, line_p2
        )
        if intersection is not None:
            if max_y is not None:
                _, y_intercept = intersection
                if y_intercept <= max_y:
                    return i
            else:
                return i
    return -1


def _line_intersection(p1, p2, q1, q2, tol=0.5):
    """Parametric 2D segment intersection with a relaxed tolerance.

    Returns the intersection point as a numpy array, or None if the
    segments are parallel/collinear or don't intersect within tolerance.
    """
    p = np.array(p1, dtype=float)
    r = np.array(p2, dtype=float) - p
    q = np.array(q1, dtype=float)
    s = np.array(q2, dtype=float) - q

    r_cross_s = np.cross(r, s)
    if abs(r_cross_s) < tol:
        return None

    t = np.cross((q - p), s) / r_cross_s
    u = np.cross((q - p), r) / r_cross_s

    if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
        return p + t * r
    return None
