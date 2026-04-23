"""Video-mode two-pass CA pipeline (needle-in-drop configuration).

Needle-in imagery (needle visible passing through the drop) produces
two structurally separate contours per frame — left-side (surface +
drop + needle edge) and right-side (mirror). Pass 1 samples every
``video_skip_interval``-th frame, masks the needle, and calibrates a
median contact line per side plus a median needle-intercept point per
side. Pass 2 processes every frame, using those calibrations to place
stable contact points and slice a drop profile, then runs every
enabled fitting method (+ optional ML model) and streams one row per
frame to a ``UnifiedCsvWriter``.

Entry point: :func:`_needle_in_two_pass` (called by
``ca_data_processor.run_video_analysis`` when
``user_input_data.needle_in_drop`` is True).

Ports ``paper_results/evaluate/evaluate_isaac.py`` into the app
package; Isaac-specific constants and multi-stage needle-masking flow
preserved verbatim (see comments at each step).
"""

import math
from typing import Callable, Optional, Tuple, Union

import numpy as np

from opendrop_ml.modules.contact_angle._video_helpers import (
    _frame_filepath,
    _frame_timestamp,
    _load_ml_model,
    _try_circular,
    _try_ellipse,
    _try_polynomial,
    _try_yl,
)
from opendrop_ml.modules.contact_angle.contact_line import (
    find_first_intercept,
    find_last_intercept,
    fit_contact_line,
)
from opendrop_ml.modules.image.frame_loader import get_frame_count, load_frames
from opendrop_ml.modules.image.needle_masking import (
    detect_vertical_lines,
    extract_edges_two_contours,
    hough_baseline,
    remove_points_near_lines,
)
from opendrop_ml.modules.ML_model.prepare_experimental import (
    experimental_pred,
    normalize_ds,
)
from opendrop_ml.modules.preprocessing.preprocessing import (
    CPID,
    cluster_OPTICS,
    optimized_path,
)
from opendrop_ml.utils.config import LEFT_ANGLE, RIGHT_ANGLE
from opendrop_ml.utils.enums import FittingMethod


# Fallback crop when Hough baseline detection fails across every Pass-1
# sample (matches the reference ``evaluate_isaac.py`` config, which used
# 320 for all three Isaac datasets).
_DEFAULT_CROP_VALUE = 320

# Auto-crop derivation: take the median Pass-1 baseline y and add this
# margin (pixels below the baseline) to include enough of the surface
# for Pass 2's Hough re-detection without pulling in the reflection /
# far-below-surface artifacts the reference's fixed 320 explicitly
# excluded.
_USE_AUTO_CROP = True
_CROP_MARGIN_BELOW_BASELINE = 60

# Isaac pipeline tuning — matches the reference configs for all three
# Isaac datasets (``evaluate_isaac.py:74-102``).
_HOUGH_THRESHOLD = 40
_DISTANCE_THRESHOLD_BASE = 5.0
_DISTANCE_THRESHOLD_MAX = 10
_BASELINE_EXTRA_CLEAR = 30  # ``dt + 30`` — not ``dt + 5``; needle-in-specific
_MIN_DROP_TOP_RATIO = 0.2   # Cleaned drop contour's topmost y must be below
                            # ``crop_value * _MIN_DROP_TOP_RATIO``.
_CLUSTER_XI = 0.8           # OPTICS reachability steepness (initial clean).
_CLUSTER_EPS = 2            # DBSCAN radius (isolated re-cluster).

# Polynomial / tangent fit parameters applied to each Pass 2 profile.
POLYNOMIAL_DEGREE_TANGENT = 1
POLYNOMIAL_DEGREE_POLY = 2

# ML window geometry. The reference uses ``[CP-100:CP+10]`` then
# ``[-50:]``; those two slices are equivalent to a direct
# ``[CP-40:CP+10]`` when the contour has enough points either side of
# CP, which it always does for Isaac data (CP sits well inside the
# ordered full contour).
_ML_N_DROP = 40
_ML_N_SURFACE = 10
_ML_INPUT_LEN = 50

# Minimum Pass 1 successful samples before we trust the ODR fit.
_MIN_PASS1_SUCCESSES = 3


ProgressCb = Callable[[int, int, str], None]
CancelCb = Callable[[], bool]


def _needle_in_two_pass(user_input_data, writer, progress: ProgressCb,
                        cancel: CancelCb) -> dict:
    """Run the needle-in two-pass pipeline.

    Reads ``import_files[0]`` (a video file or image directory path)
    and ``video_skip_interval`` off ``user_input_data``. Enabled
    fitting methods are taken from ``analysis_methods_ca``.

    Returns a dict describing the run:

    ``status``
        ``"completed"`` or ``"cancelled"``.
    ``phase``
        Which pass was running when the status was decided.
    ``pass1_total``, ``pass2_written``
        Frame counts for progress/diagnostic use.
    ``left_contact_line``, ``right_contact_line``
        ``[[x1, y1], [x2, y2]]`` lists or ``None`` if Pass 1 was
        cancelled.
    ``pass1_cps``
        ``{"left": [[x, y], …], "right": [[x, y], …]}`` — raw Pass 1
        CPs used to fit the median contact lines, or ``None`` on
        cancel.
    ``left_needle_x``, ``right_needle_x``
        Median needle-intercept x-coordinates (floats) used to build
        the vertical needle lines in Pass 2. ``None`` on cancel. The
        auto (needle-out) pipeline returns the same keys with
        ``None`` values, so Phase I's figure generator can branch on
        presence uniformly.
    """
    source_path = user_input_data.import_files[0]
    skip = max(1, int(user_input_data.video_skip_interval))
    enabled = {m for m, on in user_input_data.analysis_methods_ca.items() if on}

    total_frames = get_frame_count(source_path)
    if total_frames == 0:
        raise ValueError("No frames found in source: {}".format(source_path))
    pass1_total = math.ceil(total_frames / skip)

    pass1 = _pass1_calibrate(source_path, skip, pass1_total, progress, cancel)
    if pass1 is None:
        return {
            "status": "cancelled",
            "phase": "pass_1",
            "pass1_total": pass1_total,
            "pass2_written": 0,
            "left_contact_line": None,
            "right_contact_line": None,
            "pass1_cps": None,
            "left_needle_x": None,
            "right_needle_x": None,
        }
    (left_line, right_line, left_needle_int, right_needle_int,
     left_cps, right_cps, crop_for_pass2) = pass1

    ml_model = _load_ml_model() if FittingMethod.ML_MODEL in enabled else None

    rows_written = _pass2_analyze(
        source_path=source_path,
        left_line=left_line,
        right_line=right_line,
        left_needle_int=left_needle_int,
        right_needle_int=right_needle_int,
        enabled_methods=enabled,
        ml_model=ml_model,
        writer=writer,
        total_frames=total_frames,
        progress=progress,
        cancel=cancel,
        crop_value=crop_for_pass2,
    )

    return {
        "status": "cancelled" if cancel() else "completed",
        "phase": "pass_2",
        "pass1_total": pass1_total,
        "pass2_written": rows_written,
        "left_contact_line": left_line.tolist(),
        "right_contact_line": right_line.tolist(),
        "pass1_cps": {
            "left": left_cps.tolist(),
            "right": right_cps.tolist(),
        },
        "left_needle_x": float(left_needle_int[0]),
        "right_needle_x": float(right_needle_int[0]),
    }


# ---------------------------------------------------------------------------
# Cluster-selection helpers — ``cluster_OPTICS`` returns a dict keyed by
# cluster label; these pick the relevant cluster for each stage. Ndarray
# passthrough lets callers use them uniformly regardless of whether
# clustering was skipped.
# ---------------------------------------------------------------------------

def _largest_cluster(groups: Union[dict, np.ndarray]) -> np.ndarray:
    """Return the cluster with the most points (ndarray passthrough)."""
    if isinstance(groups, np.ndarray):
        return groups
    keys = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)
    return np.array(groups[keys[0]])


def _rightmost_cluster(groups: Union[dict, np.ndarray]) -> np.ndarray:
    """Return the cluster with the largest ``max(x)``.

    Used after isolating the left drop contour: among the re-clustered
    remnants, the one closest to the drop's centerline is the real drop
    edge, and its ``max(x)`` is the largest.
    """
    if isinstance(groups, np.ndarray):
        return groups
    key = max(groups.keys(), key=lambda k: max(groups[k][:, 0]))
    return np.array(groups[key])


def _leftmost_cluster(groups: Union[dict, np.ndarray]) -> np.ndarray:
    """Return the cluster with the smallest ``max(x)``.

    Mirror of ``_rightmost_cluster`` for the right drop contour — the
    cluster closest to the drop centerline has the smallest ``max(x)``.
    """
    if isinstance(groups, np.ndarray):
        return groups
    key = min(groups.keys(), key=lambda k: max(groups[k][:, 0]))
    return np.array(groups[key])


def _create_edge_image(
    left: np.ndarray, right: np.ndarray, shape: tuple
) -> np.ndarray:
    """Rasterise left + right contour points into a binary edge image.

    ``shape`` is ``(h, w)`` or any tuple where the first two entries are
    image height and width. Out-of-bounds points are silently dropped.
    """
    h, w = shape[:2]
    edge_img = np.zeros((h, w), dtype=np.uint8)
    for x, y in left:
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            edge_img[iy, ix] = 255
    for x, y in right:
        ix, iy = int(x), int(y)
        if 0 <= ix < w and 0 <= iy < h:
            edge_img[iy, ix] = 255
    return edge_img


# ---------------------------------------------------------------------------
# Pass 1 — calibrate median contact lines and median needle points
# ---------------------------------------------------------------------------

def _isolate_drop_side(
    side_contour: np.ndarray,
    lines_all: list,
    baseline: Optional[np.ndarray],
    crop_value: int,
) -> Optional[Tuple[np.ndarray, int]]:
    """Clean a single side's raw contour down to the drop proper.

    Mirrors the reference's adaptive-dt loop + extra baseline clear:
    starts at ``_DISTANCE_THRESHOLD_BASE``, grows until the cleaned
    contour's topmost y is below ``crop_value * _MIN_DROP_TOP_RATIO``
    (i.e. well under the needle region), then applies an extra
    ``dt + _BASELINE_EXTRA_CLEAR`` sweep against the baseline alone to
    scrub any surface/baseline residue left at the bottom of the drop.
    Returns ``(cleaned_contour, dt_used)`` or ``None`` if the loop
    exhausts without producing a viable cleaned contour.
    """
    dt = _DISTANCE_THRESHOLD_BASE
    isolated = None
    while dt <= _DISTANCE_THRESHOLD_MAX:
        candidate = remove_points_near_lines(side_contour, lines_all, dt)
        if len(candidate) > 0 and min(candidate[:, 1]) > crop_value * _MIN_DROP_TOP_RATIO:
            isolated = candidate
            break
        dt += 1
    if isolated is None:
        isolated = remove_points_near_lines(
            side_contour, lines_all, _DISTANCE_THRESHOLD_MAX
        )
        dt = _DISTANCE_THRESHOLD_MAX
    if baseline is not None:
        isolated = remove_points_near_lines(
            isolated, [baseline], dt + _BASELINE_EXTRA_CLEAR
        )
    if len(isolated) == 0:
        return None
    return isolated, dt


def process_frame_pass1(
    frame: np.ndarray, crop_value: int = _DEFAULT_CROP_VALUE
) -> Optional[dict]:
    """Process a single frame for Pass 1 calibration.

    Mirrors ``evaluate_isaac.py:process_frame_pass1``. Returns a dict
    with keys ``left_contour`` / ``right_contour`` (ordered full
    contours), ``left_CP`` / ``right_CP`` (CPID output), and
    ``left_needle_int`` / ``right_needle_int`` (topmost points of the
    isolated drop contours), or ``None`` if any stage fails (caller
    counts it as a failed sample and moves on).
    """
    _h, w = frame.shape[:2]
    cropped = frame[:crop_value, :]

    # Stage 1: two largest contours (blur OFF — see plan Risks).
    try:
        contour1, contour2, _thresh = extract_edges_two_contours(cropped)
    except ValueError:
        return None

    # Stage 2: OPTICS clean to drop noise speckle; largest cluster = drop edge.
    try:
        contour1 = _largest_cluster(cluster_OPTICS(contour1, xi=_CLUSTER_XI))
    except Exception:
        pass
    try:
        contour2 = _largest_cluster(cluster_OPTICS(contour2, xi=_CLUSTER_XI))
    except Exception:
        pass
    if len(contour1) < 10 or len(contour2) < 10:
        return None

    # Assign L/R by topmost-point x.
    top1_x = contour1[np.argmin(contour1[:, 1]), 0]
    top2_x = contour2[np.argmin(contour2[:, 1]), 0]
    if top1_x < top2_x:
        left, right = contour1, contour2
    else:
        left, right = contour2, contour1

    # Stage 3: Hough baseline + vertical needle-edge detection.
    edge_img = _create_edge_image(left, right, (crop_value, w))
    try:
        baseline = hough_baseline(edge_img, threshold=_HOUGH_THRESHOLD)
    except Exception:
        baseline = np.array([[0, crop_value], [w, crop_value]])
    left_fiber, right_fiber, _, _ = detect_vertical_lines(
        edge_img, hough_threshold=_HOUGH_THRESHOLD
    )
    if left_fiber is None or right_fiber is None:
        return None

    # Ensure the fiber labelled 'left' really is on the left.
    left_top_x = left_fiber[np.argmin(left_fiber[:, 1])][0]
    right_top_x = right_fiber[np.argmin(right_fiber[:, 1])][0]
    if left_top_x > right_top_x:
        left_fiber, right_fiber = right_fiber, left_fiber

    lines_all = [baseline, left_fiber, right_fiber]

    # Stage 4: isolate drop side per side (adaptive dt + baseline clear).
    left_res = _isolate_drop_side(left, lines_all, baseline, crop_value)
    right_res = _isolate_drop_side(right, lines_all, baseline, crop_value)
    if left_res is None or right_res is None:
        return None
    left_isolated, _ = left_res
    right_isolated, _ = right_res
    if len(left_isolated) < 10 or len(right_isolated) < 10:
        return None

    # Stage 5: re-cluster isolated contours and pick the side-specific cluster.
    try:
        left_clusters = cluster_OPTICS(left_isolated, eps=_CLUSTER_EPS)
        left_isolated = _rightmost_cluster(left_clusters)
    except Exception:
        pass
    try:
        right_clusters = cluster_OPTICS(right_isolated, eps=_CLUSTER_EPS)
        right_isolated = _leftmost_cluster(right_clusters)
    except Exception:
        pass
    if len(left_isolated) < 10 or len(right_isolated) < 10:
        return None

    # Needle intercept = topmost point of the isolated drop contour.
    left_needle_int = left_isolated[np.argmin(left_isolated[:, 1])]
    right_needle_int = right_isolated[np.argmin(right_isolated[:, 1])]

    # Stage 6: sort + optimized_path on the FULL contours. Sort keys are
    # asymmetric: left uses (y, -x) (ties: larger x first = closer to
    # drop centerline); right uses (y, x). Getting this backwards
    # breaks find_first_intercept / find_last_intercept in Pass 2.
    left_sorted = sorted(left.tolist(), key=lambda p: (p[1], -p[0]))
    left_ordered = optimized_path(left_sorted)
    right_sorted = sorted(right.tolist(), key=lambda p: (p[1], p[0]))
    right_ordered = optimized_path(right_sorted)

    # Stage 7: find the needle-intercept index in the ordered full contour.
    left_matches = np.where((left_ordered == left_needle_int).all(axis=1))[0]
    right_matches = np.where((right_ordered == right_needle_int).all(axis=1))[0]
    if len(left_matches) == 0 or len(right_matches) == 0:
        return None
    idx_l_start = left_matches[0]
    idx_r_start = right_matches[0]

    # Stage 8: CPID receives the full contour per side from the needle
    # intercept to the bottom (surface region included — iterative_Lfit
    # needs it to detect the drop-to-surface RMSE transition).
    left_for_cpid = left_ordered[idx_l_start:]
    right_for_cpid = right_ordered[idx_r_start:]
    if len(left_for_cpid) < 10 or len(right_for_cpid) < 10:
        return None

    try:
        cps = CPID([left_for_cpid, right_for_cpid], display=False)
    except Exception:
        return None
    if cps is None or len(cps) < 2:
        return None

    return {
        "left_contour": left_ordered,
        "right_contour": right_ordered,
        "left_CP": cps[0],
        "right_CP": cps[1],
        "left_needle_int": left_needle_int,
        "right_needle_int": right_needle_int,
        "baseline": baseline,
    }


def derive_auto_crop(baselines: list) -> Optional[int]:
    """Return ``median(baseline_y) + _CROP_MARGIN_BELOW_BASELINE``.

    ``baselines`` is a list of ``[[x1, y1], [x2, y2]]`` line endpoints
    returned by ``hough_baseline``. Each baseline's y is taken as the
    mean of its two endpoint y-values (Isaac baselines are ~horizontal,
    so y1 ≈ y2). Returns ``None`` if ``baselines`` is empty.
    """
    if not baselines:
        return None
    y_values = []
    for bl in baselines:
        if bl is None:
            continue
        arr = np.asarray(bl, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        y_values.append(float(np.mean(arr[:2, 1])))
    if not y_values:
        return None
    return int(round(float(np.median(y_values)) + _CROP_MARGIN_BELOW_BASELINE))


def compute_pass1_calibration(
    pass1_results: list,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, Optional[int]]:
    """Reduce Pass-1 per-frame results into median contact lines + needle points.

    ``pass1_results`` is a list of non-``None`` dicts returned by
    :func:`process_frame_pass1` (keys: ``left_CP``, ``right_CP``,
    ``left_needle_int``, ``right_needle_int``, ``baseline``).

    Returns ``(left_line, right_line, left_needle_int, right_needle_int,
    left_CPs_raw, right_CPs_raw, auto_crop)``. ``auto_crop`` is the
    baseline-median-derived crop when ``_USE_AUTO_CROP`` is on and
    derivation succeeds, else ``None`` (the caller chooses a fallback).

    Raises ``RuntimeError`` if fewer than ``_MIN_PASS1_SUCCESSES``
    results were supplied.
    """
    if len(pass1_results) < _MIN_PASS1_SUCCESSES:
        raise RuntimeError(
            "Pass 1 calibration failed: {} successful samples "
            "(need at least {}).".format(
                len(pass1_results), _MIN_PASS1_SUCCESSES
            )
        )

    left_cps_arr = np.array([
        np.asarray(r["left_CP"]) for r in pass1_results
    ])
    right_cps_arr = np.array([
        np.asarray(r["right_CP"]) for r in pass1_results
    ])
    left_ints_arr = np.array([
        np.asarray(r["left_needle_int"]) for r in pass1_results
    ])
    right_ints_arr = np.array([
        np.asarray(r["right_needle_int"]) for r in pass1_results
    ])
    baselines = [r["baseline"] for r in pass1_results]

    left_line = fit_contact_line(left_cps_arr)
    right_line = fit_contact_line(right_cps_arr)

    # Median needle intercept point per side (x, y). The needle is
    # narrow and easily bent; a robust median point with buffer room
    # beats an ODR line fit across a handful of noisy intercepts.
    left_needle_int = np.array([
        float(np.median(left_ints_arr[:, 0])),
        float(np.median(left_ints_arr[:, 1])),
    ])
    right_needle_int = np.array([
        float(np.median(right_ints_arr[:, 0])),
        float(np.median(right_ints_arr[:, 1])),
    ])

    auto_crop = None
    if _USE_AUTO_CROP:
        auto = derive_auto_crop(baselines)
        if auto is not None and auto > 0:
            auto_crop = auto

    return (left_line, right_line, left_needle_int, right_needle_int,
            left_cps_arr, right_cps_arr, auto_crop)


def _pass1_calibrate(
    source_path: str,
    skip: int,
    pass1_total: int,
    progress: ProgressCb,
    cancel: CancelCb,
    crop_value: int = _DEFAULT_CROP_VALUE,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                    np.ndarray, np.ndarray, int]]:
    """Calibrate median contact lines + median needle-intercept points.

    Samples every ``skip``-th frame. Returns
    ``(left_line, right_line, left_needle_int, right_needle_int,
      left_CPs_raw, right_CPs_raw, crop_for_pass2)`` or ``None`` if
    ``cancel()`` fires before calibration completes. Raises
    ``RuntimeError`` if fewer than ``_MIN_PASS1_SUCCESSES`` frames
    processed cleanly.

    ``crop_for_pass2`` is the auto-detected crop when
    ``_USE_AUTO_CROP`` is on (derived from the median Pass-1 baseline
    y + ``_CROP_MARGIN_BELOW_BASELINE``), else ``crop_value`` (the
    caller-supplied fallback).
    """
    results = []
    sample_idx = 0
    for frame, _frame_number, _meta in load_frames(source_path, frame_step=skip):
        if cancel():
            return None
        sample_idx += 1

        result = process_frame_pass1(frame, crop_value=crop_value)
        if result is not None:
            results.append(result)

        progress(sample_idx, pass1_total, "Pass 1")

    (left_line, right_line, left_needle_int, right_needle_int,
     left_cps_arr, right_cps_arr, auto_crop) = compute_pass1_calibration(results)

    crop_for_pass2 = auto_crop if auto_crop is not None else crop_value

    return (left_line, right_line, left_needle_int, right_needle_int,
            left_cps_arr, right_cps_arr, crop_for_pass2)


# ---------------------------------------------------------------------------
# Pass 2 — per-frame analysis
# ---------------------------------------------------------------------------

def _preprocess_pass2(
    frame: np.ndarray, crop_value: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract and order two side contours for Pass 2.

    Returns ``(left_ordered, right_ordered)`` or ``None`` on any stage
    failure (caller writes a NaN row with ``error_status`` set).
    """
    cropped = frame[:crop_value, :]
    try:
        contour1, contour2, _ = extract_edges_two_contours(cropped)
    except ValueError:
        return None

    try:
        contour1 = _largest_cluster(cluster_OPTICS(contour1, xi=_CLUSTER_XI))
    except Exception:
        pass
    try:
        contour2 = _largest_cluster(cluster_OPTICS(contour2, xi=_CLUSTER_XI))
    except Exception:
        pass
    if len(contour1) < 10 or len(contour2) < 10:
        return None

    top1_x = contour1[np.argmin(contour1[:, 1]), 0]
    top2_x = contour2[np.argmin(contour2[:, 1]), 0]
    if top1_x < top2_x:
        left, right = contour1, contour2
    else:
        left, right = contour2, contour1

    left_sorted = sorted(left.tolist(), key=lambda p: (p[1], -p[0]))
    left_ordered = optimized_path(left_sorted)
    right_sorted = sorted(right.tolist(), key=lambda p: (p[1], p[0]))
    right_ordered = optimized_path(right_sorted)
    return left_ordered, right_ordered


def ml_window(contour: np.ndarray, cp_idx: int) -> np.ndarray:
    """Extract ``[CP-_ML_N_DROP : CP+_ML_N_SURFACE]`` with clamping."""
    start = max(0, cp_idx - _ML_N_DROP)
    end = min(len(contour), cp_idx + _ML_N_SURFACE)
    return contour[start:end]


def _predict_ml_needle_in(
    left_contour: np.ndarray,
    right_contour: np.ndarray,
    left_cp_idx: int,
    right_cp_idx: int,
    ml_model,
) -> Optional[Tuple[float, float]]:
    """Run the ML model over the two per-side windows.

    Mirrors ``evaluate_isaac.py:611-629``: x-reflect the left window
    (``ml_left[:, 0] *= -1``), require exactly ``_ML_INPUT_LEN`` points
    per side, normalise, predict side="both". Returns ``None`` if
    either window is short (parity with the reference's strict
    ``len == 50`` check).
    """
    ml_left = ml_window(left_contour, left_cp_idx).astype(np.float64).copy()
    ml_right = ml_window(right_contour, right_cp_idx).astype(np.float64).copy()
    if len(ml_left) != _ML_INPUT_LEN or len(ml_right) != _ML_INPUT_LEN:
        return None
    ml_left[:, 0] *= -1
    pred_ds = normalize_ds(np.array([ml_left, ml_right])).astype(np.float32)
    predictions, _timings = experimental_pred(pred_ds, ml_model, side="both")
    return float(predictions[0]), float(predictions[1])


def _try_ml_needle_in(
    out: dict,
    left_contour: np.ndarray,
    right_contour: np.ndarray,
    left_cp_idx: int,
    right_cp_idx: int,
    ml_model,
) -> None:
    try:
        result = _predict_ml_needle_in(
            left_contour, right_contour, left_cp_idx, right_cp_idx, ml_model
        )
        if result is None:
            return
        left_angle, right_angle = result
        out[FittingMethod.ML_MODEL] = {
            LEFT_ANGLE: float(left_angle),
            RIGHT_ANGLE: float(right_angle),
        }
    except Exception:
        pass


def _pass2_state(
    frame: np.ndarray,
    crop_value: int,
    left_line: np.ndarray,
    right_line: np.ndarray,
    left_needle_int: np.ndarray,
    right_needle_int: np.ndarray,
) -> Tuple[Optional[dict], str]:
    """Build the per-frame Pass-2 state needed by both UI fits and paper extraction.

    Returns ``(state, error_status)``. ``state`` is ``None`` on any
    failure and ``error_status`` identifies the failing stage
    (``preprocess_failed``, ``contour_short``, ``needle_above_cp``,
    ``drop_too_short``). On success, ``state`` keys are: ``profile``,
    ``left_contour``, ``right_contour``, ``left_cp_idx``,
    ``right_cp_idx``, ``left_CP``, ``right_CP``, ``ml_left_window``,
    ``ml_right_window``.
    """
    pre = _preprocess_pass2(frame, crop_value)
    if pre is None:
        return None, "preprocess_failed"
    left, right = pre

    left_cp_idx = find_first_intercept(left, left_line)
    right_cp_idx = find_first_intercept(right, right_line)
    if left_cp_idx < 0 or right_cp_idx < 0:
        return None, "contour_short"

    # Vertical needle lines through the median needle-intercept x, per side.
    left_needle_line = np.array([
        [left_needle_int[0], 0],
        [left_needle_int[0], crop_value],
    ])
    right_needle_line = np.array([
        [right_needle_int[0], 0],
        [right_needle_int[0], crop_value],
    ])

    left_cp_pt = left[left_cp_idx]
    right_cp_pt = right[right_cp_idx]
    # max_y cutoff for find_last_intercept — halfway between the
    # needle-top row and the contact-line row (matches reference
    # line 475-478).
    approx_mid_y = max(left_needle_int[1], right_needle_int[1]) + (
        (max(left_cp_pt[1], right_cp_pt[1])
         - max(left_needle_int[1], right_needle_int[1])) / 2
    )

    left_start_idx = find_last_intercept(
        left, left_needle_line, max_y=approx_mid_y
    )
    right_start_idx = find_last_intercept(
        right, right_needle_line, max_y=approx_mid_y
    )

    # Fallback ladder (per reference): single-side failure borrows from
    # the other; dual failure falls back to frame start.
    if left_start_idx < 0 and right_start_idx >= 0:
        left_start_idx = right_start_idx
    if right_start_idx < 0 and left_start_idx >= 0:
        right_start_idx = left_start_idx
    if left_start_idx < 0:
        left_start_idx = 0
    if right_start_idx < 0:
        right_start_idx = 0

    if left_start_idx >= left_cp_idx or right_start_idx >= right_cp_idx:
        return None, "needle_above_cp"

    left_drop = left[left_start_idx:left_cp_idx + 1]
    right_drop = right[right_start_idx:right_cp_idx + 1]
    if len(left_drop) < 5 or len(right_drop) < 5:
        return None, "drop_too_short"

    # CP-to-CP profile: left goes top→bottom, right goes top→bottom;
    # reversing the left half makes the combined profile run left CP →
    # drop apex → right CP.
    profile = np.vstack([left_drop[::-1], right_drop])

    state = {
        "profile": profile,
        "left_contour": left,
        "right_contour": right,
        "left_cp_idx": left_cp_idx,
        "right_cp_idx": right_cp_idx,
        "left_CP": left[left_cp_idx],
        "right_CP": right[right_cp_idx],
        "ml_left_window": ml_window(left, left_cp_idx),
        "ml_right_window": ml_window(right, right_cp_idx),
    }
    return state, ""


def extract_pass2_frame(
    frame: np.ndarray,
    crop_value: int,
    left_line: np.ndarray,
    right_line: np.ndarray,
    left_needle_int: np.ndarray,
    right_needle_int: np.ndarray,
) -> Optional[dict]:
    """Public Pass-2 extraction: preprocess + CP indices + profile + ML windows.

    Returns the Pass-2 state dict (see :func:`_pass2_state`) or ``None``
    on any failure. Callers that need the failure category should call
    ``_pass2_state`` directly.
    """
    state, _err = _pass2_state(
        frame, crop_value, left_line, right_line,
        left_needle_int, right_needle_int,
    )
    return state


def _analyze_frame_needle_in(
    frame: np.ndarray,
    left_line: np.ndarray,
    right_line: np.ndarray,
    left_needle_int: np.ndarray,
    right_needle_int: np.ndarray,
    enabled_methods,
    ml_model,
    crop_value: int,
) -> Tuple[dict, str]:
    """Per-frame Pass 2 analysis. Returns ``(contact_angles, error_status)``."""
    state, err = _pass2_state(
        frame, crop_value, left_line, right_line,
        left_needle_int, right_needle_int,
    )
    if state is None:
        return {}, err

    profile = state["profile"]
    left = state["left_contour"]
    right = state["right_contour"]
    left_cp_idx = state["left_cp_idx"]
    right_cp_idx = state["right_cp_idx"]

    contact_angles: dict = {}
    if FittingMethod.TANGENT_FIT in enabled_methods:
        _try_polynomial(
            contact_angles, FittingMethod.TANGENT_FIT,
            profile, POLYNOMIAL_DEGREE_TANGENT,
        )
    if FittingMethod.POLYNOMIAL_FIT in enabled_methods:
        _try_polynomial(
            contact_angles, FittingMethod.POLYNOMIAL_FIT,
            profile, POLYNOMIAL_DEGREE_POLY,
        )
    if FittingMethod.CIRCLE_FIT in enabled_methods:
        _try_circular(contact_angles, profile)
    if FittingMethod.ELLIPSE_FIT in enabled_methods:
        _try_ellipse(contact_angles, profile)
    if FittingMethod.YL_FIT in enabled_methods:
        _try_yl(contact_angles, profile)
    if FittingMethod.ML_MODEL in enabled_methods and ml_model is not None:
        _try_ml_needle_in(
            contact_angles, left, right,
            left_cp_idx, right_cp_idx, ml_model,
        )

    return contact_angles, ""


def _pass2_analyze(
    source_path: str,
    left_line: np.ndarray,
    right_line: np.ndarray,
    left_needle_int: np.ndarray,
    right_needle_int: np.ndarray,
    enabled_methods,
    ml_model,
    writer,
    total_frames: int,
    progress: ProgressCb,
    cancel: CancelCb,
    crop_value: int = _DEFAULT_CROP_VALUE,
) -> int:
    """Run Pass 2 over every frame; stream one row per frame to the writer.

    Returns the number of rows written. Respects ``cancel`` — mid-loop
    cancellation leaves a coherent partial CSV (writer writes rows as
    they complete).
    """
    rows = 0
    for frame, frame_number, meta in load_frames(source_path, frame_step=1):
        if cancel():
            break

        filepath = _frame_filepath(source_path, meta)
        timestamp_s = _frame_timestamp(frame_number, meta)

        contact_angles, error_status = _analyze_frame_needle_in(
            frame, left_line, right_line,
            left_needle_int, right_needle_int,
            enabled_methods, ml_model, crop_value,
        )

        writer.write_row(
            frame_number=frame_number + 1,
            timestamp_s=timestamp_s,
            filepath=filepath,
            error_status=error_status,
            contact_angles=contact_angles,
        )
        rows += 1
        progress(rows, total_frames, "Pass 2")

    return rows
