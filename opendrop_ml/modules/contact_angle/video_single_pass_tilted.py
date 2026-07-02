"""Video-mode single-pass CA pipeline (tilted-surface, needle-outside).

Each frame is independent: detect substrate tilt, rotate the image to
level, then run the standard isolate/CPID/fit/ML pipeline on the
corrected image. Frames where the drop touches the image edge are
written as NaN rows with ``error_status="drop_off_edge"`` and excluded
from the fit pass.

Module-level flag ``_USE_TWO_PASS_TILTED`` selects between this
single-pass path and the two-pass variant in
``video_two_pass_tilted.py`` (two-pass variant with a shared median
tilt angle is the default since it recovers more frames on
partially-rolled drops).

Entry point: :func:`_single_pass_with_tilt` (called by
``ca_data_processor.run_video_analysis`` when
``user_input_data.surface_tilted`` is True and
``user_input_data.needle_in_drop`` is False).
"""

import math
from typing import Callable, Dict, Optional, Tuple

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
from opendrop_ml.modules.contact_angle.contact_line import fit_contact_line
from opendrop_ml.modules.image.frame_loader import get_frame_count, load_frames
from opendrop_ml.modules.image.tilt_correction import (
    correct_tilt,
    detect_tilt_angle,
    touches_border,
)
from opendrop_ml.modules.ML_model.prepare_experimental import (
    experimental_pred,
    prepare_for_model,
)
from opendrop_ml.modules.preprocessing.preprocessing import CPID, isolate_contour
from opendrop_ml.utils.config import LEFT_ANGLE, RIGHT_ANGLE
from opendrop_ml.utils.enums import FittingMethod


POLYNOMIAL_DEGREE_TANGENT = 1
POLYNOMIAL_DEGREE_POLY = 2

# ML preprocessing parameters — match the reference call in evaluate_kristina.
_ML_INPUT_LEN = 50
_ML_N_SURFACE = 10

# Minimum collected per-frame CPs before fit_contact_line is called for
# the cp_scatter overlay; below this, the contact lines stay None.
_MIN_CPS_FOR_LINE_FIT = 3

# When True, detect tilt once per video (median of sampled frames)
# instead of per-frame. Default False preserves per-frame behaviour.
_USE_PER_VIDEO_TILT = False

# When True, the (surface_tilted=True, needle_in_drop=False) dispatch
# branch delegates to ``_auto_two_pass_with_tilt_correction`` (two-pass
# with shared tilt + CP-outlier gate) instead of
# ``_single_pass_with_tilt``. The two-pass variant is the default
# because it recovers more frames on partially-rolled drops (shared
# tilt + ``crop=False`` sidesteps two failure modes of the single-pass
# path — see ``video_two_pass_tilted.py``).
_USE_TWO_PASS_TILTED = True

# Minimum Stage-1 sample count for the per-video tilt median. When the
# user's ``video_skip_interval`` would yield fewer samples than this,
# the stride tightens until the floor is met. Protects short / low-FPS
# videos where the raw stride could collapse to 3-4 samples. The
# two-pass tilted variant reuses the same floor for its Pass-1 ODR line
# fit (where >=10 CPs give a stable median per side).
_PASS1_TARGET_SAMPLES = 10


ProgressCb = Callable[[int, int, str], None]
CancelCb = Callable[[], bool]


def _build_empty_return_dict(status: str) -> dict:
    """Return dict matching the main-loop shape for zero-work exits
    (e.g. Stage 1 cancel). Keys mirror the completed-run return so
    downstream consumers see a uniform contract."""
    return {
        "status": status,
        "phase": "single_pass",
        "frames_processed": 0,
        "frames_skipped": 0,
        "median_tilt_angle": float("nan"),
        "rotated_first_frame": None,
        "left_contact_line": None,
        "right_contact_line": None,
        "pass1_cps": {"left": [], "right": []},
        "left_needle_x": None,
        "right_needle_x": None,
    }


def _effective_tilt_stride(
    total_frames: int,
    requested_skip: int,
    target_samples: int = _PASS1_TARGET_SAMPLES,
) -> int:
    """Return the stride to use for Stage 1 tilt sampling.

    When ``requested_skip`` would already yield at least ``target_samples``
    samples, returns ``requested_skip`` unchanged. When fewer, tightens
    the stride so at least ``target_samples`` are sampled (floored at 1).
    Very short videos (``total_frames <= target_samples``) fall through to
    stride 1 (every frame).

    Examples (target=10):
        total=55,  skip=10 → 5 (6→11 samples)
        total=110, skip=10 → 10 (unchanged; 11 samples)
        total=440, skip=10 → 10 (unchanged; 44 samples)
        total=8,   skip=10 → 1 (8 samples)
    """
    if total_frames <= target_samples:
        return 1
    cap = max(1, total_frames // target_samples)
    return min(max(1, requested_skip), cap)


def _derive_shared_tilt(
    source_path: str,
    skip: int,
    total_samples: int,
    progress: ProgressCb,
    cancel: CancelCb,
    label: str = "Tilt calibration",
) -> Tuple[float, list, float]:
    """Sample every ``skip``-th frame and derive a single tilt angle.

    Runs ``detect_tilt_angle`` on each sampled frame and returns the
    median of the non-zero results. Shared between the single-pass
    once-per-video tilt path and the two-pass tilted pipeline (which
    structurally requires cross-frame coordinate alignment).

    Args:
        source_path: Video file or image directory.
        skip: ``load_frames`` ``frame_step`` — sample every Nth frame.
        total_samples: Expected sample count, for the progress denom.
        progress: UI progress callback; invoked per sample with
            ``label`` as the stage name.
        cancel: Polled before each sample; on True, returns with the
            partial results collected so far.
        label: Progress-callback stage label. Defaults to
            ``"Tilt calibration"``. The two-pass tilted caller passes
            ``"Pass 1"`` to keep the two-pass UI contract unchanged.

    Returns:
        ``(shared_tilt, sample_angles, success_rate)``:
        - ``shared_tilt``: ``np.median`` of non-zero samples, or
          ``float("nan")`` if every sample returned ``0.0`` (or none
          were collected).
        - ``sample_angles``: full list of per-sample angles including
          zeros, ordered by sample iteration.
        - ``success_rate``: ``len(non_zero) / len(sample_angles)`` in
          ``[0.0, 1.0]``; ``0.0`` when no samples were collected.
    """
    sample_angles: list = []
    sample_idx = 0
    for frame, _frame_number, _meta in load_frames(
        source_path,
        frame_step=skip,
    ):
        if cancel():
            break
        sample_idx += 1
        angle = detect_tilt_angle(frame)
        sample_angles.append(float(angle))
        progress(sample_idx, total_samples, label)

    if not sample_angles:
        return float("nan"), sample_angles, 0.0

    non_zero = [a for a in sample_angles if a != 0.0]
    success_rate = len(non_zero) / len(sample_angles)
    if not non_zero:
        return float("nan"), sample_angles, success_rate
    return float(np.median(non_zero)), sample_angles, success_rate


def _single_pass_with_tilt(
    user_input_data, writer, progress: ProgressCb, cancel: CancelCb
) -> dict:
    """Run the single-pass tilted-surface pipeline.

    Reads ``import_files[0]`` (a video file or image directory) off
    ``user_input_data``. Enabled fitting methods are taken from
    ``analysis_methods_ca``.

    Returns a dict describing the run:

    ``status``
        ``"completed"`` or ``"cancelled"``.
    ``phase``
        Always ``"single_pass"`` (single-pass pipeline has no pass split).
    ``frames_processed``, ``frames_skipped``
        Counts of usable vs. drop-off-edge frames.
    ``median_tilt_angle``
        ``np.median`` of per-frame detected tilt angles (NaN if none).
    ``rotated_first_frame``
        Copy of the first successfully-corrected frame (or ``None``).
        Consumed by the cp_scatter renderer as the background image.
    ``left_contact_line``, ``right_contact_line``
        ``[[x1, y1], [x2, y2]]`` ODR fits over the IQR-filtered per-frame
        CPs, or ``None`` if fewer than three usable frames were collected.
        Used only for the cp_scatter overlay — not for the fits themselves.
    ``pass1_cps``
        ``{"left": [...], "right": [...]}`` — the raw per-frame CPs.
        Same shape as the two-pass pipelines so figure code can stay shared.
    ``left_needle_x``, ``right_needle_x``
        Always ``None`` for the needle-outside pipeline (kept for
        return-dict back-compat with the needle-in pipeline).
    """
    source_path = user_input_data.import_files[0]
    enabled = {m for m, on in user_input_data.analysis_methods_ca.items() if on}

    total_frames = get_frame_count(source_path)
    if total_frames == 0:
        raise ValueError("No frames found in source: {}".format(source_path))

    # Stage 1 — optional once-per-video tilt derivation.
    shared_tilt: Optional[float] = None
    if _USE_PER_VIDEO_TILT:
        requested_skip = max(1, int(user_input_data.video_skip_interval))
        skip = _effective_tilt_stride(total_frames, requested_skip)
        if skip != requested_skip:
            print(
                "  INFO: tilt-sample stride tightened from {} to {} "
                "to reach the {}-sample floor ({} frames total).".format(
                    requested_skip,
                    skip,
                    _PASS1_TARGET_SAMPLES,
                    total_frames,
                )
            )
        total_samples = math.ceil(total_frames / skip)
        shared_tilt, _samples, success_rate = _derive_shared_tilt(
            source_path,
            skip,
            total_samples,
            progress,
            cancel,
        )
        if cancel():
            return _build_empty_return_dict(status="cancelled")
        if success_rate == 0.0 or not np.isfinite(shared_tilt):
            print(
                "  WARNING: per-video tilt detection produced zero "
                "successful samples; falling back to per-frame "
                "detection for this run."
            )
            shared_tilt = None

    ml_model = _load_ml_model() if FittingMethod.ML_MODEL in enabled else None

    rows_written = 0
    frames_skipped = 0
    tilt_angles: list = []
    left_cps: list = []
    right_cps: list = []
    rotated_first_frame: Optional[np.ndarray] = None

    for frame, frame_number, meta in load_frames(source_path, frame_step=1):
        if cancel():
            break

        filepath = _frame_filepath(source_path, meta)
        timestamp_s = _frame_timestamp(frame_number, meta)

        # 1. Tilt correction — per-frame by default; shared median when
        #    ``_USE_PER_VIDEO_TILT`` is active.
        try:
            corrected, tilt_angle = correct_tilt(
                frame,
                tilt_angle=shared_tilt,
                crop=True,
            )
        except Exception:
            writer.write_row(
                frame_number=frame_number + 1,
                timestamp_s=timestamp_s,
                filepath=filepath,
                error_status="tilt_failed",
                contact_angles={},
                detected_tilt_angle=math.nan,
            )
            rows_written += 1
            progress(rows_written, total_frames, "Processing")
            continue

        if rotated_first_frame is None:
            rotated_first_frame = corrected.copy()

        # 2. Preprocess on the corrected image.
        try:
            _, contour, _, _ = isolate_contour(corrected, DISPLAY=False, blur=True)
        except Exception:
            writer.write_row(
                frame_number=frame_number + 1,
                timestamp_s=timestamp_s,
                filepath=filepath,
                error_status="isolate_failed",
                contact_angles={},
                detected_tilt_angle=tilt_angle,
            )
            rows_written += 1
            progress(rows_written, total_frames, "Processing")
            continue

        # 3. Drop-off-edge skip — matches reference exactly.
        if touches_border(contour, corrected.shape, margin=2):
            writer.write_row(
                frame_number=frame_number + 1,
                timestamp_s=timestamp_s,
                filepath=filepath,
                error_status="drop_off_edge",
                contact_angles={},
                detected_tilt_angle=tilt_angle,
            )
            rows_written += 1
            frames_skipped += 1
            tilt_angles.append(tilt_angle)
            progress(rows_written, total_frames, "Processing")
            continue

        # 4. Contact points.
        try:
            cps = CPID(contour, display=False)
        except Exception:
            writer.write_row(
                frame_number=frame_number + 1,
                timestamp_s=timestamp_s,
                filepath=filepath,
                error_status="cpid_failed",
                contact_angles={},
                detected_tilt_angle=tilt_angle,
            )
            rows_written += 1
            progress(rows_written, total_frames, "Processing")
            continue
        if cps is None or len(cps) < 2:
            writer.write_row(
                frame_number=frame_number + 1,
                timestamp_s=timestamp_s,
                filepath=filepath,
                error_status="cpid_failed",
                contact_angles={},
                detected_tilt_angle=tilt_angle,
            )
            rows_written += 1
            progress(rows_written, total_frames, "Processing")
            continue

        # 5. Profile extraction — direct slice between CPID-returned CPs
        #    (no Pass-1 contact-line intercepts in single-pass).
        profile, _idx_l, _idx_r = _extract_profile_between_cps(contour, cps)
        if profile is None or len(profile) < 3:
            writer.write_row(
                frame_number=frame_number + 1,
                timestamp_s=timestamp_s,
                filepath=filepath,
                error_status="profile_too_short",
                contact_angles={},
                detected_tilt_angle=tilt_angle,
            )
            rows_written += 1
            progress(rows_written, total_frames, "Processing")
            continue

        # 6. Order CPs left-then-right and stash for the post-loop line fit.
        if cps[0][0] < cps[1][0]:
            left_cps.append(np.asarray(cps[0]).copy())
            right_cps.append(np.asarray(cps[1]).copy())
        else:
            left_cps.append(np.asarray(cps[1]).copy())
            right_cps.append(np.asarray(cps[0]).copy())

        # 7. Per-method fits — independent failure isolation.
        contact_angles: Dict = {}
        if FittingMethod.TANGENT_FIT in enabled:
            _try_polynomial(
                contact_angles,
                FittingMethod.TANGENT_FIT,
                profile,
                POLYNOMIAL_DEGREE_TANGENT,
            )
        if FittingMethod.POLYNOMIAL_FIT in enabled:
            _try_polynomial(
                contact_angles,
                FittingMethod.POLYNOMIAL_FIT,
                profile,
                POLYNOMIAL_DEGREE_POLY,
            )
        if FittingMethod.CIRCLE_FIT in enabled:
            _try_circular(contact_angles, profile)
        if FittingMethod.ELLIPSE_FIT in enabled:
            _try_ellipse(contact_angles, profile)
        if FittingMethod.YL_FIT in enabled:
            _try_yl(contact_angles, profile)
        if FittingMethod.ML_MODEL in enabled and ml_model is not None:
            _try_ml_single_pass(contact_angles, corrected, ml_model)

        tilt_angles.append(tilt_angle)
        writer.write_row(
            frame_number=frame_number + 1,
            timestamp_s=timestamp_s,
            filepath=filepath,
            error_status="",
            contact_angles=contact_angles,
            detected_tilt_angle=tilt_angle,
        )
        rows_written += 1
        progress(rows_written, total_frames, "Processing")

    left_line = _safe_fit_contact_line(left_cps)
    right_line = _safe_fit_contact_line(right_cps)
    median_tilt = float(np.median(tilt_angles)) if tilt_angles else float("nan")

    return {
        "status": "cancelled" if cancel() else "completed",
        "phase": "single_pass",
        "frames_processed": rows_written - frames_skipped,
        "frames_skipped": frames_skipped,
        "median_tilt_angle": median_tilt,
        "rotated_first_frame": rotated_first_frame,
        "left_contact_line": (left_line.tolist() if left_line is not None else None),
        "right_contact_line": (right_line.tolist() if right_line is not None else None),
        "pass1_cps": {
            "left": np.asarray(left_cps).tolist() if left_cps else [],
            "right": np.asarray(right_cps).tolist() if right_cps else [],
        },
        "left_needle_x": None,
        "right_needle_x": None,
    }


# ---------------------------------------------------------------------------
# Profile extraction — direct slice between CPID-returned CPs. Distinct
# from contact_line.find_first_intercept (which intercepts against an
# external Pass-1 line, not direct CPID indices).
# ---------------------------------------------------------------------------


def _extract_profile_between_cps(
    contour: np.ndarray,
    cps: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[int]]:
    try:
        idx_l = int(np.where((contour == cps[0]).all(axis=1))[0][0])
        idx_r = int(np.where((contour == cps[1]).all(axis=1))[0][0])
    except IndexError:
        return None, None, None
    if idx_l > idx_r:
        idx_l, idx_r = idx_r, idx_l
    return contour[idx_l : idx_r + 1], idx_l, idx_r


# ---------------------------------------------------------------------------
# ML predict — single-pass uses the reference's prepare_for_model path
# (operates directly on the corrected image, no line-intercept windows).
# ---------------------------------------------------------------------------


def _try_ml_single_pass(out: Dict, corrected: np.ndarray, ml_model) -> None:
    try:
        pred_ds = prepare_for_model(
            corrected,
            input_len=_ML_INPUT_LEN,
            n_surface=_ML_N_SURFACE,
            display=False,
            blur=True,
        )
        predictions, _timings = experimental_pred(pred_ds, ml_model, side="both")
        out[FittingMethod.ML_MODEL] = {
            LEFT_ANGLE: float(predictions[0]),
            RIGHT_ANGLE: float(predictions[1]),
        }
    except Exception:
        pass


def _safe_fit_contact_line(cps: list) -> Optional[np.ndarray]:
    """Fit a median contact line via IQR + ODR. Returns None if too few
    CPs were collected to produce a stable line."""
    if len(cps) < _MIN_CPS_FOR_LINE_FIT:
        return None
    try:
        return fit_contact_line(np.asarray(cps))
    except Exception:
        return None
