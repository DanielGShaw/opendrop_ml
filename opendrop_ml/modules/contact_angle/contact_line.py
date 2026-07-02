"""Contact-line utilities for the video-mode pipelines.

`fit_contact_line` fits an ODR line through IQR-filtered contact points
(Pass 1 calibration); `find_first_intercept` and `find_last_intercept`
locate the first/last contour segments that cross a fitted line (Pass 2
stable-CP placement).

Both `fit_contact_line` and `fit_contact_line_with_stats` accept
optional ``contours`` and ``side`` arguments; when both are provided
they run **Fix B** (bimodal-CP detection + local-angle-driven cluster
pick + four guards) before returning. Without them, behaviour is
bit-for-bit identical to the legacy IQR + ODR path.

Fix B reference: ``BIMODAL_CP_FIX.md`` § Production Implementation
Specification (validated across 34 sides / 17 experiments).
"""

import math

import numpy as np
from sklearn.cluster import KMeans

from opendrop_ml.modules.preprocessing.preprocessing import linear_fit


# ----------------------------------------------------------------------
# Fix B tunable constants — validated across 34 sides / 17 experiments
# (`BIMODAL_CP_FIX.md` § Cross-dataset validation).
# ----------------------------------------------------------------------

RESIDUAL_STD_THRESHOLD = 2.0  # px — bimodality trigger (signed perp residual std)
LOCAL_ANGLE_N_PTS = 12  # contour points sampled above CP for local tangent
LOCAL_ANGLE_MAX_DY = 25.0  # px — vertical search window above CP
MIN_CLUSTER_MEMBERS = 3  # KMeans cluster minimum size to consider
ANGLE_SEPARATION_MIN = 5.0  # deg — required gap in median |angle-90°|
KEPT_X_RANGE_MIN = 20.0  # px — kept cluster x-span guard (case 5)
CASE_11_MIN_CPS = 6  # below this, skip Fix B entirely (single-pass / static)
CASE_9_NON_NAN_RATIO = 0.5  # min fraction of valid local angles per cluster


def fit_contact_line(cps, contours=None, side=None):
    """Fit a linear contact line through contact points with IQR filtering.

    Filters contact points to the y-coordinate IQR range (25th-75th
    percentile), then fits a line via ODR (`linear_fit`). Falls back to
    a flat median line if the fit fails or too few points survive
    filtering.

    When ``contours`` and ``side`` are both supplied and ``len(cps) >=
    CASE_11_MIN_CPS``, runs Fix B: detects y-bimodality via signed
    perpendicular residual std, KMeans-splits CPs on y, picks the
    cluster whose median ``|local_angle - 90°|`` is largest, refits.
    Abstains (returning the legacy line) on any of:
    - residual std below threshold (clean unimodal data)
    - either cluster has too few members or too few valid local angles
    - cluster median angles are too close (`< ANGLE_SEPARATION_MIN`)
    - kept cluster's x-range is too narrow (`< KEPT_X_RANGE_MIN`)
    - refit returns non-finite slope/intercept

    Args:
        cps: Nx2 array of [x, y] contact points collected across Pass 1.
        contours: optional list of length N. Each entry an Mx2 array of
            contour points (ordered or unordered) for local-angle
            computation. Either side-specific (needle-in) or full drop
            (needle-out / inverse_phase) — geometric helper handles both.
        side: ``"left"`` or ``"right"`` — required when ``contours`` is
            given. Determines the surface-outward reference direction.

    Returns:
        2x2 numpy array [[x_min, y_at_x_min], [x_max, y_at_x_max]]
        describing a line segment. After a Fix B correction the line
        spans the kept cluster's x-range, not the original cps range.
    """
    cps = np.asarray(cps)
    fit_orig = _iqr_trim_odr(cps)

    if contours is None or side is None:
        return fit_orig["line"]

    line, _kept_mask = _apply_fix_b(cps, contours, side, fit_orig)
    return line


def fit_contact_line_with_stats(cps, contours=None, side=None):
    """Fit a linear contact line and return per-side IQR stats.

    Identical line fit to :func:`fit_contact_line`, plus a ``stats``
    dict the two-pass tilted pipeline's Pass-2 outlier gate uses to
    decide whether a per-frame CPID CP is an outlier against the
    Pass-1 median line. The IQR is computed over perpendicular
    residuals of the post-IQR-filtered CPs — the "well-behaved"
    population that fed the ODR fit.

    With ``contours`` and ``side`` provided, runs Fix B (see
    :func:`fit_contact_line`); when Fix B refits through a kept
    cluster, ``stats`` is recomputed against the refit line on that
    kept cluster.

    Args:
        cps: Nx2 array of [x, y] contact points collected across Pass 1.
        contours: optional list — see :func:`fit_contact_line`.
        side: ``"left"`` or ``"right"`` — see :func:`fit_contact_line`.

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
            - ``"n_total"``: count of input CPs (or kept cluster size
              after a Fix B correction).
    """
    cps = np.asarray(cps)
    fit_orig = _iqr_trim_odr(cps)

    if contours is None or side is None:
        return fit_orig["line"], _build_stats(fit_orig)

    line, kept_mask = _apply_fix_b(cps, contours, side, fit_orig)
    if kept_mask is None:
        return line, _build_stats(fit_orig)

    fit_new = _iqr_trim_odr(cps[kept_mask])
    return line, _build_stats(fit_new)


def local_contact_angle(contour, cp, side, n_pts=None, max_dy=None):
    """Local contact angle from CP up the drop edge (surface-inward).

    Operates in **image coordinates** — y increases DOWNWARD, so "above
    the CP" means smaller y, and a tangent pointing into the drop
    interior has a negative-y component.

    Geometric "above + closest" approach — robust to contour traversal
    order, so the same helper works on side-specific ordered contours
    (needle-in) and full unordered drop contours (needle-out, inverse
    phase). Filters contour points to those within ``max_dy`` above the
    CP, takes the ``n_pts`` closest by Euclidean distance, computes the
    average direction from CP to those points, returns the angle
    between that direction and the surface-inward unit vector — the
    conventional contact-angle reference (<90° hydrophilic, >90°
    hydrophobic).

    Args:
        contour: Mx2 array of contour points (image coordinates).
        cp: 1-D length-2 array-like — the contact point [x, y]
            (image coordinates).
        side: ``"left"`` or ``"right"``. Selects the surface-inward
            reference (the substrate direction toward the drop
            interior): ``(+1, 0)`` for left (drop sits right of the
            left CP), ``(-1, 0)`` for right (drop sits left of the
            right CP).
        n_pts: optional override for ``LOCAL_ANGLE_N_PTS``.
        max_dy: optional override for ``LOCAL_ANGLE_MAX_DY``.

    Returns:
        Angle in degrees, conventional contact-angle convention:
        ~53° for a high-CA hydrophobic drop's local edge slope on
        MS_APTES, ~95° for a reflection-confused near-90° detection.
        Returns ``float("nan")`` when too few points qualify or the
        average direction is degenerate.
    """
    if n_pts is None:
        n_pts = LOCAL_ANGLE_N_PTS
    if max_dy is None:
        max_dy = LOCAL_ANGLE_MAX_DY

    contour = np.asarray(contour, dtype=float)
    cp = np.asarray(cp, dtype=float).flatten()
    cp_x, cp_y = float(cp[0]), float(cp[1])

    above_mask = (contour[:, 1] < cp_y) & (contour[:, 1] >= cp_y - max_dy)
    above = contour[above_mask]
    if len(above) < n_pts:
        return float("nan")

    dists = np.hypot(above[:, 0] - cp_x, above[:, 1] - cp_y)
    nearest = above[np.argsort(dists)[:n_pts]]
    direction = nearest.mean(axis=0) - np.array([cp_x, cp_y])
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return float("nan")

    surface_inward = np.array([1.0, 0.0]) if side == "left" else np.array([-1.0, 0.0])
    cos_a = float(np.dot(surface_inward, direction) / norm)
    cos_a = max(-1.0, min(1.0, cos_a))
    return float(np.degrees(np.arccos(cos_a)))


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _iqr_trim_odr(cps):
    """Run the legacy IQR-trim + ODR fit and return everything callers need.

    Returns a dict with:
        - ``line``: 2x2 array of endpoints spanning the input cps' x-range
        - ``m``, ``c``: slope/intercept from ``linear_fit`` (NaN on failure)
        - ``cps_inrange``: post-filter CP subset used for the fit
        - ``n_filtered``: count of post-filter CPs (0 on fit failure)
        - ``n_total``: original cps count
    """
    cps = np.asarray(cps)
    n_total = len(cps)
    q1 = np.percentile(cps[:, 1], 25)
    q3 = np.percentile(cps[:, 1], 75)
    cps_inrange = np.array([cp for cp in cps if q1 <= cp[1] <= q3])

    if len(cps_inrange) < 2:
        cps_inrange = cps

    start_x = float(cps[:, 0].min())
    end_x = float(cps[:, 0].max())

    try:
        m, c, _ = linear_fit(cps_inrange, display=False)
    except Exception:
        med = np.median(cps, axis=0)
        line = np.array([[start_x, med[1]], [end_x, med[1]]])
        return {
            "line": line,
            "m": float("nan"),
            "c": float("nan"),
            "cps_inrange": cps_inrange,
            "n_filtered": 0,
            "n_total": n_total,
        }

    line = np.array([[start_x, m * start_x + c], [end_x, m * end_x + c]])
    return {
        "line": line,
        "m": float(m),
        "c": float(c),
        "cps_inrange": cps_inrange,
        "n_filtered": len(cps_inrange),
        "n_total": n_total,
    }


def _build_stats(fit):
    """Build the (residual_iqr, n_filtered, n_total) stats dict from a fit."""
    if fit["n_filtered"] == 0 or not (
        math.isfinite(fit["m"]) and math.isfinite(fit["c"])
    ):
        return {
            "residual_iqr": float("nan"),
            "n_filtered": 0,
            "n_total": fit["n_total"],
        }
    residuals = np.abs(_perp_residuals(fit["cps_inrange"], fit["m"], fit["c"]))
    r_q1 = np.percentile(residuals, 25)
    r_q3 = np.percentile(residuals, 75)
    return {
        "residual_iqr": float(r_q3 - r_q1),
        "n_filtered": fit["n_filtered"],
        "n_total": fit["n_total"],
    }


def _apply_fix_b(cps, contours, side, fit_orig):
    """Apply Fix B bimodality detection + cluster-pick refit.

    Returns ``(line, kept_mask)``:
        - on abstain: ``(fit_orig["line"], None)``
        - on correction: ``(refit_line, boolean_mask_of_kept_cluster)``
    """
    line_orig = fit_orig["line"]
    m_orig = fit_orig["m"]
    c_orig = fit_orig["c"]

    if len(cps) < CASE_11_MIN_CPS:
        return line_orig, None
    if not (math.isfinite(m_orig) and math.isfinite(c_orig)):
        return line_orig, None

    res_std = float(np.std(_perp_residuals(cps, m_orig, c_orig)))
    if not math.isfinite(res_std) or res_std <= RESIDUAL_STD_THRESHOLD:
        return line_orig, None

    labels = KMeans(
        n_clusters=2,
        n_init=10,
        random_state=0,
    ).fit_predict(cps[:, 1:2].astype(float))

    angles = np.array(
        [local_contact_angle(contours[i], cps[i], side) for i in range(len(cps))],
        dtype=float,
    )

    valid = []
    for cluster_id in (0, 1):
        mask = labels == cluster_id
        n = int(mask.sum())
        if n < MIN_CLUSTER_MEMBERS:
            continue
        cluster_angles = angles[mask]
        cluster_angles_valid = cluster_angles[~np.isnan(cluster_angles)]
        if len(cluster_angles_valid) < CASE_9_NON_NAN_RATIO * n:
            continue
        valid.append(
            {
                "mask": mask,
                "n": n,
                "median_dev": float(np.median(np.abs(cluster_angles_valid - 90.0))),
            }
        )
    if len(valid) < 2:
        return line_orig, None

    valid.sort(key=lambda v: v["median_dev"], reverse=True)
    best, second = valid[0], valid[1]
    if (best["median_dev"] - second["median_dev"]) < ANGLE_SEPARATION_MIN:
        return line_orig, None

    cps_kept = cps[best["mask"]]
    kept_x_range = float(cps_kept[:, 0].max() - cps_kept[:, 0].min())
    if kept_x_range < KEPT_X_RANGE_MIN:
        return line_orig, None

    fit_new = _iqr_trim_odr(cps_kept)
    if not (math.isfinite(fit_new["m"]) and math.isfinite(fit_new["c"])):
        return line_orig, None

    # Reproject the kept-cluster fit onto the original cps' x-range so
    # Pass-2's `find_first_intercept` has the same horizontal reach as
    # the legacy line — drop edges in early/late frames can fall well
    # outside the kept cluster's x-span. Legacy `_iqr_trim_odr` already
    # spans the original cps range; preserve that contract.
    m_new, c_new = fit_new["m"], fit_new["c"]
    start_x = float(cps[:, 0].min())
    end_x = float(cps[:, 0].max())
    line_new = np.array(
        [[start_x, m_new * start_x + c_new], [end_x, m_new * end_x + c_new]]
    )
    if not np.all(np.isfinite(line_new)):
        return line_orig, None

    return line_new, best["mask"]


def _perp_residuals(points, m, c):
    """Signed perpendicular residuals from each (x, y) in points to y = m*x + c.

    Standard point-line distance formula (m*x - y + c) / sqrt(m**2 + 1)
    without an absolute value, so points on opposite sides of the line
    have opposite signs. Callers wanting unsigned distance wrap with
    ``np.abs(...)``. A non-finite ``m`` or ``c`` (linear_fit degenerated
    on identical / collinear input) yields an all-NaN result so
    downstream stats become NaN and the Pass-2 outlier gate disables
    itself on that side.

    Fix B's bimodality trigger relies on the signed form: when the line
    cuts diagonally between two y-tiers, signed residuals split into
    positive and negative populations, inflating their std around the
    line. Unsigned std would collapse that signal.
    """
    pts = np.asarray(points, dtype=float)
    if not (math.isfinite(m) and math.isfinite(c)):
        return np.full(len(pts), float("nan"))
    return (m * pts[:, 0] - pts[:, 1] + c) / math.sqrt(m * m + 1.0)


# ----------------------------------------------------------------------
# Pass-2 contour-vs-line intercept utilities
# ----------------------------------------------------------------------


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
        intersection = _line_intersection(contour[i], contour[i + 1], line_p1, line_p2)
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
        intersection = _line_intersection(contour[i], contour[i + 1], line_p1, line_p2)
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
