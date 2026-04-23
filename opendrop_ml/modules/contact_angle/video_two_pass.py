"""Video-mode two-pass CA pipeline (standard-sessile, needle-outside).

Pass 1 samples every ``video_skip_interval``-th frame and fits a median
contact line per side via ODR. Pass 2 processes every frame, using the
fitted lines to place stable contact points, then runs every enabled
fitting method (+ optional ML model) on the resulting profile and
streams one row per frame to a ``UnifiedCsvWriter``.

Entry point: :func:`_auto_two_pass` (called by
``ca_data_processor.run_video_analysis``).
"""

import math
from typing import Callable, Dict, Iterable, Optional, Tuple

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
    fit_contact_line,
)
from opendrop_ml.modules.fitting.circular_fit import circular_fit
from opendrop_ml.modules.image.frame_loader import get_frame_count, load_frames
from opendrop_ml.modules.ML_model.prepare_experimental import (
    experimental_pred,
    normalize_ds,
    resize_image,
)
from opendrop_ml.modules.preprocessing.preprocessing import CPID, isolate_contour
from opendrop_ml.utils.config import LEFT_ANGLE, RIGHT_ANGLE
from opendrop_ml.utils.enums import FittingMethod


POLYNOMIAL_DEGREE_TANGENT = 1
POLYNOMIAL_DEGREE_POLY = 2

# Resolution-drop thresholds used when resampling the profile into the
# ML model's expected input size.
_PROFILE_MAX_PLATEAU = 974
_PROFILE_EMPIRICAL_MIN = 565
_RESOLUTION_DROP_SLOPE = 6.383333333333334
_RESOLUTION_DROP_INTERCEPT = 208.0

# ML window geometry: [CP - n_surface : CP + n_drop] on left, mirrored on right.
_ML_N_SURFACE = 10
_ML_N_DROP = 40
_ML_INPUT_LEN = 50

# Minimum Pass 1 successful samples before we trust the ODR fit.
_MIN_PASS1_SUCCESSES = 3


ProgressCb = Callable[[int, int, str], None]
CancelCb = Callable[[], bool]


def _auto_two_pass(user_input_data, writer, progress: ProgressCb, cancel: CancelCb):
    """Run the two-pass video-mode pipeline.

    Reads ``import_files[0]`` (a video file or image directory path) and
    ``video_skip_interval`` off ``user_input_data``. Enabled fitting
    methods are taken from ``analysis_methods_ca``.

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
        ``{"left": [[x, y], …], "right": [[x, y], …]}`` — the raw Pass 1
        CPs used to fit the median lines, or ``None`` on cancel.
    ``left_needle_x``, ``right_needle_x``
        Always ``None`` for the needle-outside pipeline. Emitted so
        the needle-in pipeline (which populates them) can share the
        same Phase I figure generator.
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
    left_line, right_line, left_cps, right_cps = pass1

    ml_model = _load_ml_model() if FittingMethod.ML_MODEL in enabled else None

    rows_written = _pass2_analyze(
        source_path=source_path,
        left_line=left_line,
        right_line=right_line,
        enabled_methods=enabled,
        ml_model=ml_model,
        writer=writer,
        total_frames=total_frames,
        progress=progress,
        cancel=cancel,
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
        "left_needle_x": None,
        "right_needle_x": None,
    }


# ---------------------------------------------------------------------------
# Pass 1 — calibrate median contact lines
# ---------------------------------------------------------------------------

def _pass1_calibrate(
    source_path: str,
    skip: int,
    pass1_total: int,
    progress: ProgressCb,
    cancel: CancelCb,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    left_cps = []
    right_cps = []

    sample_idx = 0
    for frame, _frame_number, _meta in load_frames(source_path, frame_step=skip):
        if cancel():
            return None
        sample_idx += 1

        try:
            _, contour, _, _ = isolate_contour(frame, DISPLAY=False, blur=True)
            cps = CPID(contour, display=False)
        except Exception:
            progress(sample_idx, pass1_total, "Pass 1")
            continue

        if cps is None or len(cps) < 2:
            progress(sample_idx, pass1_total, "Pass 1")
            continue

        if cps[0][0] < cps[1][0]:
            left_cps.append(np.asarray(cps[0]).copy())
            right_cps.append(np.asarray(cps[1]).copy())
        else:
            left_cps.append(np.asarray(cps[1]).copy())
            right_cps.append(np.asarray(cps[0]).copy())

        progress(sample_idx, pass1_total, "Pass 1")

    if len(left_cps) < _MIN_PASS1_SUCCESSES:
        raise RuntimeError(
            "Pass 1 calibration failed: {} successful samples "
            "(need at least {}).".format(len(left_cps), _MIN_PASS1_SUCCESSES)
        )

    left_arr = np.array(left_cps)
    right_arr = np.array(right_cps)
    left_line = fit_contact_line(left_arr)
    right_line = fit_contact_line(right_arr)
    return left_line, right_line, left_arr, right_arr


# ---------------------------------------------------------------------------
# Pass 2 — per-frame analysis
# ---------------------------------------------------------------------------

def _pass2_analyze(
    source_path: str,
    left_line: np.ndarray,
    right_line: np.ndarray,
    enabled_methods: Iterable[FittingMethod],
    ml_model,
    writer,
    total_frames: int,
    progress: ProgressCb,
    cancel: CancelCb,
) -> int:
    rows = 0
    for frame, frame_number, meta in load_frames(source_path, frame_step=1):
        if cancel():
            break

        filepath = _frame_filepath(source_path, meta)
        timestamp_s = _frame_timestamp(frame_number, meta)

        contact_angles, error_status = _analyze_frame(
            frame, left_line, right_line, enabled_methods, ml_model
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


def _analyze_frame(
    frame: np.ndarray,
    left_line: np.ndarray,
    right_line: np.ndarray,
    enabled_methods: Iterable[FittingMethod],
    ml_model,
) -> Tuple[Dict, str]:
    contact_angles: Dict = {}

    try:
        _, contour, _, _ = isolate_contour(frame, DISPLAY=False, blur=True)
    except Exception:
        return contact_angles, "isolate_failed"

    try:
        cps_orig = CPID(contour, display=False)
    except Exception:
        return contact_angles, "cpid_failed"
    if cps_orig is None or len(cps_orig) < 2:
        return contact_angles, "cpid_failed"

    try:
        idx_left_orig = int(np.where((contour == cps_orig[0]).all(axis=1))[0][0])
        idx_right_orig = int(np.where((contour == cps_orig[1]).all(axis=1))[0][0])
    except IndexError:
        return contact_angles, "cpid_index_failed"
    if idx_left_orig > idx_right_orig:
        idx_left_orig, idx_right_orig = idx_right_orig, idx_left_orig

    midpoint = int(idx_left_orig + (idx_right_orig - idx_left_orig) / 2)
    left_half = contour[:midpoint][::-1]
    right_half = contour[midpoint:]

    left_local = find_first_intercept(left_half, left_line)
    right_local = find_first_intercept(right_half, right_line)
    if left_local == -1 or right_local == -1:
        return contact_angles, "contour_short"

    idx_left_cp = midpoint - 1 - left_local
    idx_right_cp = midpoint + right_local
    if idx_left_cp > idx_right_cp:
        idx_left_cp, idx_right_cp = idx_right_cp, idx_left_cp

    profile = contour[idx_left_cp: idx_right_cp + 1]
    if len(profile) < 3:
        return contact_angles, "profile_too_short"

    # --- Per-method fitting. Each is independent; one failing leaves the
    # others populated. Failures manifest as NaN columns in the unified CSV.
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
        _try_ml(
            contact_angles, frame, contour, profile,
            idx_left_cp, idx_right_cp, left_line, right_line, ml_model,
        )

    return contact_angles, ""


def _try_ml(
    out: Dict,
    frame: np.ndarray,
    contour: np.ndarray,
    profile: np.ndarray,
    idx_left_cp: int,
    idx_right_cp: int,
    left_line: np.ndarray,
    right_line: np.ndarray,
    ml_model,
) -> None:
    try:
        left_angle, right_angle = _predict_ml(
            frame, contour, profile,
            idx_left_cp, idx_right_cp,
            left_line, right_line, ml_model,
        )
        out[FittingMethod.ML_MODEL] = {
            LEFT_ANGLE: float(left_angle),
            RIGHT_ANGLE: float(right_angle),
        }
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ML prediction with resolution-drop + window extraction.
# ---------------------------------------------------------------------------

def _predict_ml(
    frame: np.ndarray,
    contour: np.ndarray,
    profile: np.ndarray,
    idx_left_cp: int,
    idx_right_cp: int,
    left_line: np.ndarray,
    right_line: np.ndarray,
    ml_model,
) -> Tuple[float, float]:
    if _needs_resolution_drop(profile):
        contour_ml, idx_l_ml, idx_r_ml = _rerun_preprocess_downsampled(
            frame, left_line, right_line
        )
    else:
        contour_ml, idx_l_ml, idx_r_ml = contour, idx_left_cp, idx_right_cp

    if idx_l_ml < _ML_N_SURFACE:
        left_side = contour_ml[:_ML_INPUT_LEN]
    else:
        left_side = contour_ml[idx_l_ml - _ML_N_SURFACE: idx_l_ml + _ML_N_DROP]

    if idx_r_ml > len(contour_ml) - _ML_N_SURFACE:
        right_side = contour_ml[-_ML_INPUT_LEN:]
    else:
        right_side = contour_ml[idx_r_ml - _ML_N_DROP: idx_r_ml + _ML_N_SURFACE]

    left_side = left_side[::-1].copy()
    left_side[:, 0] = -left_side[:, 0]

    if len(left_side) < 4 or len(right_side) < 4:
        raise ValueError(
            "ML window too short: L={}, R={}".format(len(left_side), len(right_side))
        )

    pred_ds = np.array([left_side, right_side])
    pred_ds = normalize_ds(pred_ds).astype(np.float32)
    predictions, _timings = experimental_pred(pred_ds, ml_model, side="both")
    return float(predictions[0]), float(predictions[1])


def _needs_resolution_drop(profile: np.ndarray) -> bool:
    profile_len = len(profile)
    if profile_len > _PROFILE_MAX_PLATEAU:
        return True
    if _PROFILE_EMPIRICAL_MIN < profile_len < _PROFILE_MAX_PLATEAU:
        try:
            circ_angles, *_ = circular_fit(profile, display=False)
            threshold = (
                _RESOLUTION_DROP_SLOPE * min(circ_angles) + _RESOLUTION_DROP_INTERCEPT
            )
            return profile_len > threshold
        except Exception:
            return False
    return False


def _rerun_preprocess_downsampled(
    frame: np.ndarray,
    left_line: np.ndarray,
    right_line: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    img_small = resize_image(frame, scale=0.5)
    _, contour_ml, _, _ = isolate_contour(img_small, DISPLAY=False, blur=True)
    cps_ml = CPID(contour_ml, display=False)
    if cps_ml is None or len(cps_ml) < 2:
        raise ValueError("CPID failed on resized image")

    idx_l = int(np.where((contour_ml == cps_ml[0]).all(axis=1))[0][0])
    idx_r = int(np.where((contour_ml == cps_ml[1]).all(axis=1))[0][0])
    if idx_l > idx_r:
        idx_l, idx_r = idx_r, idx_l

    mid = int(idx_l + (idx_r - idx_l) / 2)
    left_h = contour_ml[:mid][::-1]
    right_h = contour_ml[mid:]
    ll = find_first_intercept(left_h, left_line * 0.5)
    rr = find_first_intercept(right_h, right_line * 0.5)
    if ll >= 0:
        idx_l = mid - 1 - ll
    if rr >= 0:
        idx_r = mid + rr
    if idx_l > idx_r:
        idx_l, idx_r = idx_r, idx_l
    return contour_ml, idx_l, idx_r


