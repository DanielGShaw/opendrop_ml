"""Video-mode two-pass CA pipeline (tilted-surface, needle-outside).

Pass 1 samples every ``effective_stride``-th frame to derive a single
shared tilt angle (median of per-frame detections) and — in a second
replay over the same samples with the shared tilt applied — fits
per-side median contact lines via ODR, along with the IQR of the
Pass-1 perpendicular residuals that Pass 2's outlier gate uses.

Pass 2 rotates every frame by the shared tilt and, instead of the
pixel-margin ``touches_border`` check the single-pass tilted pipeline
uses, gates each frame with a geometry-aware CP-outlier check: the
CPID CP on each side must lie within
``max(3 x Pass-1 residual IQR, 1.5% x image_height)`` perpendicular
distance of the corresponding Pass-1 median contact line. The floor
is a fraction of image height so the gate is resolution-independent.
When either side fails, the frame is written as a NaN row with
``error_status="drop_off_edge"``. Surviving frames flow through the
same ``find_first_intercept``-based profile extraction, per-method
fits, and ML window as the needle-outside two-pass pipeline.

This pipeline also calls ``isolate_contour(crop=False)`` instead of
the default ``crop=True``: the default Hough-circle bounding-box crop
excises the far-side contact point on partially-rolled drops and
leaves CPID with no choice but to return the drop apex as one CP. The
uncropped longest-OPTICS-cluster contour gives CPID both true baseline
CPs to find, which is what enables geometry-aware recovery on
partially-rolled frames.

Entry point: :func:`_auto_two_pass_with_tilt_correction` (called by
``ca_data_processor.run_video_analysis`` when
``user_input_data.surface_tilted`` is True, ``needle_in_drop`` is
False, and ``video_single_pass_tilted._USE_TWO_PASS_TILTED`` is True).
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
    fit_contact_line_with_stats,
)
from opendrop_ml.modules.contact_angle.video_single_pass_tilted import (
    _PASS1_TARGET_SAMPLES,
    _build_empty_return_dict,
    _derive_shared_tilt,
    _effective_tilt_stride,
)

# ML prediction helpers are pure and independent of the tilted vs
# non-tilted control flow, so this pipeline reuses the non-tilted
# two-pass implementation instead of duplicating ~100 lines of
# window/resize logic.
from opendrop_ml.modules.contact_angle.video_two_pass import _predict_ml
from opendrop_ml.modules.image.frame_loader import get_frame_count, load_frames
from opendrop_ml.modules.image.tilt_correction import correct_tilt
from opendrop_ml.modules.preprocessing.preprocessing import CPID, isolate_contour
from opendrop_ml.utils.config import LEFT_ANGLE, RIGHT_ANGLE
from opendrop_ml.utils.enums import FittingMethod


POLYNOMIAL_DEGREE_TANGENT = 1
POLYNOMIAL_DEGREE_POLY = 2

# Minimum Pass-1 successful samples before we trust the per-side ODR fit.
# Mirrors video_two_pass._MIN_PASS1_SUCCESSES — three CPs is the minimum
# for a defensible linear fit.
_MIN_PASS1_SUCCESSES = 3

# Pass-2 CP-outlier gate threshold components.
# Accept a Pass-2 CPID CP if its perpendicular distance to the Pass-1
# median contact line is <=
#     max(_PASS2_OUTLIER_MULTIPLIER × residual_iqr,
#         _PASS2_OUTLIER_FLOOR_FRAC × image_height_px)
# The multiplier (3×) is the Phase-P-plan value and adapts the gate to
# the observed Pass-1 spread. The floor is expressed as a fraction of
# the (tilt-corrected) image height so the gate is resolution-
# independent: on a well-calibrated baseline the IQR-filtered Pass-1
# CPs sit within ~0.5 px of the ODR line, collapsing the IQR to a
# fraction of a pixel and making a bare 3× IQR gate over-reject
# natural per-frame CP noise (~1-3 px on Kristina). A floor keeps the
# gate's intent — reject CPs geometrically far from the baseline
# (apex / drop-gone cases measure 10-15% of image height in
# perpendicular distance) — without over-rejecting well-measured
# edge-case frames. 1.5% of image height matches the 15 px tuned
# empirically on Kristina (1024 px) and scales predictably to other
# cameras.
_PASS2_OUTLIER_MULTIPLIER = 3.0
_PASS2_OUTLIER_FLOOR_FRAC = 0.015


ProgressCb = Callable[[int, int, str], None]
CancelCb = Callable[[], bool]


def _auto_two_pass_with_tilt_correction(
    user_input_data,
    writer,
    progress: ProgressCb,
    cancel: CancelCb,
) -> dict:
    """Run the two-pass tilted-surface pipeline."""
    source_path = user_input_data.import_files[0]
    enabled = {m for m, on in user_input_data.analysis_methods_ca.items() if on}

    total_frames = get_frame_count(source_path)
    if total_frames == 0:
        raise ValueError("No frames found in source: {}".format(source_path))

    requested_skip = max(1, int(user_input_data.video_skip_interval))
    effective_stride = _effective_tilt_stride(total_frames, requested_skip)
    if effective_stride != requested_skip:
        print(
            "  INFO: Pass 1 tightened stride from {} to {} "
            "to reach the {}-sample floor ({} frames total).".format(
                requested_skip,
                effective_stride,
                _PASS1_TARGET_SAMPLES,
                total_frames,
            )
        )
    total_samples = math.ceil(total_frames / effective_stride)

    pass1 = _pass1_calibrate_with_tilt(
        source_path=source_path,
        stride=effective_stride,
        total_samples=total_samples,
        progress=progress,
        cancel=cancel,
    )
    if pass1 is None:
        out = _build_empty_return_dict(status="cancelled")
        out["phase"] = "two_pass_tilted"
        return out
    (
        shared_tilt,
        left_line,
        right_line,
        left_stats,
        right_stats,
        left_cps,
        right_cps,
        rotated_first_frame,
    ) = pass1

    ml_model = _load_ml_model() if FittingMethod.ML_MODEL in enabled else None

    rows_written, frames_skipped = _pass2_analyze_with_tilt(
        source_path=source_path,
        shared_tilt=shared_tilt,
        left_line=left_line,
        right_line=right_line,
        left_stats=left_stats,
        right_stats=right_stats,
        enabled_methods=enabled,
        ml_model=ml_model,
        writer=writer,
        total_frames=total_frames,
        progress=progress,
        cancel=cancel,
    )

    return {
        "status": "cancelled" if cancel() else "completed",
        "phase": "two_pass_tilted",
        "frames_processed": rows_written - frames_skipped,
        "frames_skipped": frames_skipped,
        "median_tilt_angle": float(shared_tilt),
        "rotated_first_frame": rotated_first_frame,
        "left_contact_line": left_line.tolist(),
        "right_contact_line": right_line.tolist(),
        "pass1_cps": {
            "left": np.asarray(left_cps).tolist() if len(left_cps) else [],
            "right": np.asarray(right_cps).tolist() if len(right_cps) else [],
        },
        "left_needle_x": None,
        "right_needle_x": None,
    }


# ---------------------------------------------------------------------------
# Pass 1 — shared tilt + per-side median contact lines + IQR stats
# ---------------------------------------------------------------------------


def _pass1_calibrate_with_tilt(
    source_path: str,
    stride: int,
    total_samples: int,
    progress: ProgressCb,
    cancel: CancelCb,
) -> Optional[
    Tuple[
        float,
        np.ndarray,
        np.ndarray,
        Dict,
        Dict,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
    ]
]:
    """Derive shared tilt + median contact lines + residual IQR stats.

    Two-stage: (1a) sample every ``stride``-th frame via
    ``_derive_shared_tilt`` to get one shared tilt angle; (1b) replay
    those same samples with the shared tilt applied and collect per-side
    CPID CPs for the ODR fit.

    Returns ``None`` on cancel. Raises ``RuntimeError`` on unrecoverable
    calibration failure (no valid tilt samples, or too few Pass-1 CPs
    to fit a contact line).
    """
    shared_tilt, _sample_angles, success_rate = _derive_shared_tilt(
        source_path=source_path,
        skip=stride,
        total_samples=total_samples,
        progress=progress,
        cancel=cancel,
        label="Pass 1",
    )
    if cancel():
        return None
    if success_rate == 0.0 or not np.isfinite(shared_tilt):
        raise RuntimeError(
            "Pass 1 tilt calibration failed: no usable tilt samples "
            "(every sampled frame returned 0.0). Re-run with the "
            "single-pass tilted pipeline (_USE_TWO_PASS_TILTED=False) "
            "or the single-pass once-per-video tilt fallback "
            "(_USE_PER_VIDEO_TILT=True)."
        )

    left_cps: list = []
    right_cps: list = []
    contours_per_frame: list = []
    rotated_first_frame: Optional[np.ndarray] = None
    sample_idx = 0
    for frame, _frame_number, _meta in load_frames(
        source_path,
        frame_step=stride,
    ):
        if cancel():
            return None
        sample_idx += 1

        try:
            corrected, _ = correct_tilt(
                frame,
                tilt_angle=shared_tilt,
                crop=True,
            )
        except Exception:
            progress(sample_idx, total_samples, "Pass 1")
            continue

        if rotated_first_frame is None:
            rotated_first_frame = corrected.copy()

        # crop=False takes the longest OPTICS cluster directly, skipping
        # isolate_contour's Hough-circle bounding-box crop. That crop
        # excises the left (or right) half of the drop on frames where
        # the drop has rolled partially off the image — Hough fits a
        # circle to the visible half and the resulting bbox excludes the
        # far-side CP region. crop=False keeps the full drop silhouette,
        # giving CPID both true baseline CPs to find.
        try:
            contour, _threshold = isolate_contour(
                corrected,
                crop=False,
                DISPLAY=False,
                blur=True,
            )
        except Exception:
            progress(sample_idx, total_samples, "Pass 1")
            continue

        try:
            cps = CPID(contour, display=False)
        except Exception:
            progress(sample_idx, total_samples, "Pass 1")
            continue
        if cps is None or len(cps) < 2:
            progress(sample_idx, total_samples, "Pass 1")
            continue

        if cps[0][0] < cps[1][0]:
            left_cps.append(np.asarray(cps[0]).copy())
            right_cps.append(np.asarray(cps[1]).copy())
        else:
            left_cps.append(np.asarray(cps[1]).copy())
            right_cps.append(np.asarray(cps[0]).copy())
        contours_per_frame.append(np.asarray(contour))

        progress(sample_idx, total_samples, "Pass 1")

    if len(left_cps) < _MIN_PASS1_SUCCESSES:
        raise RuntimeError(
            "Pass 1 CP collection failed: {} successful samples "
            "(need at least {}).".format(
                len(left_cps),
                _MIN_PASS1_SUCCESSES,
            )
        )

    left_arr = np.array(left_cps)
    right_arr = np.array(right_cps)
    left_line, left_stats = fit_contact_line_with_stats(
        left_arr,
        contours=contours_per_frame,
        side="left",
    )
    right_line, right_stats = fit_contact_line_with_stats(
        right_arr,
        contours=contours_per_frame,
        side="right",
    )

    return (
        float(shared_tilt),
        left_line,
        right_line,
        left_stats,
        right_stats,
        left_arr,
        right_arr,
        rotated_first_frame,
    )


# ---------------------------------------------------------------------------
# Pass 2 — per-frame analysis with shared-tilt rotation + CP-outlier gate
# ---------------------------------------------------------------------------


def _pass2_analyze_with_tilt(
    source_path: str,
    shared_tilt: float,
    left_line: np.ndarray,
    right_line: np.ndarray,
    left_stats: Dict,
    right_stats: Dict,
    enabled_methods: Iterable[FittingMethod],
    ml_model,
    writer,
    total_frames: int,
    progress: ProgressCb,
    cancel: CancelCb,
) -> Tuple[int, int]:
    """Run Pass 2 over every frame. Returns (rows_written, frames_skipped)."""
    rows = 0
    skipped = 0
    for frame, frame_number, meta in load_frames(source_path, frame_step=1):
        if cancel():
            break

        filepath = _frame_filepath(source_path, meta)
        timestamp_s = _frame_timestamp(frame_number, meta)

        contact_angles, error_status = _analyze_frame_tilted(
            frame=frame,
            shared_tilt=shared_tilt,
            left_line=left_line,
            right_line=right_line,
            left_stats=left_stats,
            right_stats=right_stats,
            enabled_methods=enabled_methods,
            ml_model=ml_model,
        )

        writer.write_row(
            frame_number=frame_number + 1,
            timestamp_s=timestamp_s,
            filepath=filepath,
            error_status=error_status,
            contact_angles=contact_angles,
            detected_tilt_angle=shared_tilt,
        )
        rows += 1
        if error_status:
            skipped += 1
        progress(rows, total_frames, "Pass 2")

    return rows, skipped


def _analyze_frame_tilted(
    frame: np.ndarray,
    shared_tilt: float,
    left_line: np.ndarray,
    right_line: np.ndarray,
    left_stats: Dict,
    right_stats: Dict,
    enabled_methods: Iterable[FittingMethod],
    ml_model,
) -> Tuple[Dict, str]:
    contact_angles: Dict = {}

    try:
        corrected, _ = correct_tilt(frame, tilt_angle=shared_tilt, crop=True)
    except Exception:
        return contact_angles, "tilt_failed"

    # crop=False — see the Pass-1 rationale in _pass1_calibrate_with_tilt.
    # Identical motivation: the Hough-circle bounding-box crop hides the
    # far-side contact point on partially-rolled drops.
    try:
        contour, _threshold = isolate_contour(
            corrected,
            crop=False,
            DISPLAY=False,
            blur=True,
        )
    except Exception:
        return contact_angles, "isolate_failed"

    try:
        cps_orig = CPID(contour, display=False)
    except Exception:
        return contact_angles, "cpid_failed"
    if cps_orig is None or len(cps_orig) < 2:
        return contact_angles, "cpid_failed"

    # Order CPs left-then-right so the outlier gate and the
    # midpoint split both match the Pass-1 per-side fit geometry.
    if cps_orig[0][0] <= cps_orig[1][0]:
        left_cp, right_cp = cps_orig[0], cps_orig[1]
    else:
        left_cp, right_cp = cps_orig[1], cps_orig[0]

    # CP-outlier gate — geometry-aware replacement for the 2-pixel
    # touches_border margin used by the single-pass tilted pipeline.
    # NaN IQR (Pass-1 fallback fit) disables the IQR term on that side;
    # the floor still applies.
    image_height_px = corrected.shape[0]
    if _cp_is_outlier(left_cp, left_line, left_stats, image_height_px):
        return contact_angles, "drop_off_edge"
    if _cp_is_outlier(right_cp, right_line, right_stats, image_height_px):
        return contact_angles, "drop_off_edge"

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

    profile = contour[idx_left_cp : idx_right_cp + 1]
    if len(profile) < 3:
        return contact_angles, "profile_too_short"

    if FittingMethod.TANGENT_FIT in enabled_methods:
        _try_polynomial(
            contact_angles,
            FittingMethod.TANGENT_FIT,
            profile,
            POLYNOMIAL_DEGREE_TANGENT,
        )
    if FittingMethod.POLYNOMIAL_FIT in enabled_methods:
        _try_polynomial(
            contact_angles,
            FittingMethod.POLYNOMIAL_FIT,
            profile,
            POLYNOMIAL_DEGREE_POLY,
        )
    if FittingMethod.CIRCLE_FIT in enabled_methods:
        _try_circular(contact_angles, profile)
    if FittingMethod.ELLIPSE_FIT in enabled_methods:
        _try_ellipse(contact_angles, profile)
    if FittingMethod.YL_FIT in enabled_methods:
        _try_yl(contact_angles, profile)

    if FittingMethod.ML_MODEL in enabled_methods and ml_model is not None:
        try:
            left_angle, right_angle = _predict_ml(
                corrected,
                contour,
                profile,
                idx_left_cp,
                idx_right_cp,
                left_line,
                right_line,
                ml_model,
            )
            contact_angles[FittingMethod.ML_MODEL] = {
                LEFT_ANGLE: float(left_angle),
                RIGHT_ANGLE: float(right_angle),
            }
        except Exception:
            pass

    return contact_angles, ""


def _cp_is_outlier(
    cp: np.ndarray,
    line: np.ndarray,
    stats: Dict,
    image_height_px: int,
) -> bool:
    """True if the CP is an outlier vs the Pass-1 median contact line.

    Threshold = max(3× Pass-1 residual IQR, 1.5% × image_height_px).
    Falls back to the floor alone when the IQR is NaN / non-positive
    (Pass-1 fallback fit) — avoids rejecting every frame when Pass 1
    couldn't characterise the residual spread.
    """
    floor = _PASS2_OUTLIER_FLOOR_FRAC * float(image_height_px)
    iqr = stats.get("residual_iqr", float("nan"))
    if np.isfinite(iqr) and iqr > 0:
        threshold = max(_PASS2_OUTLIER_MULTIPLIER * iqr, floor)
    else:
        threshold = floor
    distance = _perp_distance_point_to_line(cp, line)
    return distance > threshold


def _perp_distance_point_to_line(point, line_points) -> float:
    """Perpendicular distance from ``point`` to the line through
    ``line_points[0]`` and ``line_points[1]``."""
    (x1, y1), (x2, y2) = line_points[0], line_points[1]
    px, py = float(point[0]), float(point[1])
    dx = float(x2) - float(x1)
    dy = float(y2) - float(y1)
    denom = math.hypot(dx, dy)
    if denom == 0:
        return float("nan")
    return (
        abs(dy * px - dx * py + float(x2) * float(y1) - float(y2) * float(x1)) / denom
    )
