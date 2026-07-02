#!/usr/bin/env python
"""Evaluate contact angles on the Kristina PEEK tilted-surface dataset.

This script drives the app's two-pass tilted-surface pipeline via its
mid-level primitives (middle-path pattern; cf. ``evaluate_isaac.py``):

  * Pass 1: call ``_pass1_calibrate_with_tilt`` once to derive a shared
    tilt angle, per-side median contact lines, and residual-IQR stats.
  * Pass 2: rotate each frame by the shared tilt, run
    ``isolate_contour(crop=False)`` + ``CPID``, gate the CPs with
    ``_cp_is_outlier``, extract the profile via ``find_first_intercept``,
    then run each fitting method inline (so researchers can swap
    parameters or add methods without editing the app module).

Run from OpenDrop_OP/:
    python paper_results/evaluate/evaluate_kristina.py
"""

import sys
import os
import math
import re
import time
import traceback

import numpy as np
import pandas as pd
import cv2

# Ensure imports work regardless of working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_RESULTS_DIR = os.path.dirname(SCRIPT_DIR)
OPENDROP_DIR = os.path.dirname(PAPER_RESULTS_DIR)
sys.path.insert(0, OPENDROP_DIR)
sys.path.insert(0, PAPER_RESULTS_DIR)

from config import (
    KRISTINA_DIR,
    RESULTS_DIR,
    ML_MODEL_DIR,
    N_TANGENT_POINTS,
    POLYNOMIAL_DEGREE_TANGENT,
    IMAGE_EXTENSIONS,
)

from opendrop_ml.modules.preprocessing.preprocessing import isolate_contour, CPID
from opendrop_ml.modules.fitting.polynomial_fit import polynomial_fit

from opendrop_ml.modules.image.tilt_correction import correct_tilt
from opendrop_ml.modules.image.frame_loader import get_frame_count

from opendrop_ml.modules.contact_angle.contact_line import find_first_intercept
from opendrop_ml.modules.contact_angle.video_single_pass_tilted import (
    _effective_tilt_stride,
)
from opendrop_ml.modules.contact_angle.video_two_pass_tilted import (
    _pass1_calibrate_with_tilt,
    _cp_is_outlier,
)
from opendrop_ml.modules.contact_angle.video_two_pass import _predict_ml

import tensorflow as tf


# Stride for Pass 1 tilt calibration. Matches the app default
# ExperimentalSetup.video_skip_interval. _effective_tilt_stride tightens
# this to reach the 10-sample floor on short sequences like Kristina (55
# frames → effective stride 5).
PASS1_REQUESTED_STRIDE = 10


# --- Kristina dataset configurations ---
KRISTINA_DATASETS = {
    "Water_air_PEEK_60degrees": {
        "expected_tilt": 60.0,
        "subdir": "Water_air_PEEK_60degrees",
    },
}


def natural_sort_key(s):
    """Sort key that handles embedded numbers naturally."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(s))]


def _safe_error(errors, key, fallback_key=None):
    """Get error metric from dict, handling missing keys."""
    if not isinstance(errors, dict):
        return np.nan
    val = errors.get(key, np.nan)
    if val is np.nan and fallback_key:
        val = errors.get(fallback_key, np.nan)
    return val


def evaluate_image(
    img_path,
    frame_number,
    shared_tilt,
    left_line,
    right_line,
    left_stats,
    right_stats,
    model,
):
    """Evaluate a single tilted-surface image via the two-pass pipeline.

    Mirrors ``video_two_pass_tilted._analyze_frame_tilted`` step-for-step
    so the paper-schema output is bit-for-bit identical to the app's
    per-frame output. Returns a dict of paper-schema columns; skipped
    frames carry ``skipped=True`` + ``skip_reason=<error_status>`` and
    NaN angle columns.
    """
    result = {
        "image_name": os.path.basename(img_path),
        "frame_number": frame_number,
        "detected_tilt_angle": shared_tilt,
    }

    # --- Load + shared-tilt rotation (Pass 2 step 1) ---
    img = cv2.imread(img_path)
    if img is None:
        result["skipped"] = True
        result["skip_reason"] = "load_failed"
        return result
    result["image_shape"] = str(img.shape)

    try:
        corrected, _ = correct_tilt(img, tilt_angle=shared_tilt, crop=True)
    except Exception:
        result["skipped"] = True
        result["skip_reason"] = "tilt_failed"
        return result

    # --- Preprocessing (crop=False keeps the full drop silhouette) ---
    t0 = time.time()
    try:
        contour, threshold = isolate_contour(
            corrected,
            crop=False,
            DISPLAY=False,
            blur=True,
        )
    except Exception:
        result["skipped"] = True
        result["skip_reason"] = "isolate_failed"
        return result

    try:
        cps_orig = CPID(contour, display=False)
    except Exception:
        result["skipped"] = True
        result["skip_reason"] = "cpid_failed"
        return result
    if cps_orig is None or len(cps_orig) < 2:
        result["skipped"] = True
        result["skip_reason"] = "cpid_failed"
        return result

    # Order CPs left-then-right (matches app's ordering convention).
    if cps_orig[0][0] <= cps_orig[1][0]:
        left_cp, right_cp = cps_orig[0], cps_orig[1]
    else:
        left_cp, right_cp = cps_orig[1], cps_orig[0]

    # --- CP-outlier gate (geometry-aware replacement for touches_border) ---
    image_height_px = corrected.shape[0]
    if _cp_is_outlier(left_cp, left_line, left_stats, image_height_px):
        result["skipped"] = True
        result["skip_reason"] = "drop_off_edge"
        return result
    if _cp_is_outlier(right_cp, right_line, right_stats, image_height_px):
        result["skipped"] = True
        result["skip_reason"] = "drop_off_edge"
        return result

    # --- Profile extraction via find_first_intercept against Pass-1 lines ---
    try:
        idx_left_orig = int(np.where((contour == cps_orig[0]).all(axis=1))[0][0])
        idx_right_orig = int(np.where((contour == cps_orig[1]).all(axis=1))[0][0])
    except IndexError:
        result["skipped"] = True
        result["skip_reason"] = "cpid_index_failed"
        return result
    if idx_left_orig > idx_right_orig:
        idx_left_orig, idx_right_orig = idx_right_orig, idx_left_orig

    midpoint = int(idx_left_orig + (idx_right_orig - idx_left_orig) / 2)
    left_half = contour[:midpoint][::-1]
    right_half = contour[midpoint:]

    left_local = find_first_intercept(left_half, left_line)
    right_local = find_first_intercept(right_half, right_line)
    if left_local == -1 or right_local == -1:
        result["skipped"] = True
        result["skip_reason"] = "contour_short"
        return result

    idx_left_cp = midpoint - 1 - left_local
    idx_right_cp = midpoint + right_local
    if idx_left_cp > idx_right_cp:
        idx_left_cp, idx_right_cp = idx_right_cp, idx_left_cp

    profile = contour[idx_left_cp : idx_right_cp + 1]
    if len(profile) < 3:
        result["skipped"] = True
        result["skip_reason"] = "profile_too_short"
        return result

    preprocess_time = time.time() - t0

    # Frame survived all gates — populate processed-frame metadata.
    result["skipped"] = False
    result["preprocessing_time"] = preprocess_time
    result["threshold_value"] = threshold
    result["contour_length"] = len(contour)
    result["left_CP_x"] = int(contour[idx_left_cp, 0])
    result["left_CP_y"] = int(contour[idx_left_cp, 1])
    result["right_CP_x"] = int(contour[idx_right_cp, 0])
    result["right_CP_y"] = int(contour[idx_right_cp, 1])
    result["profile_length"] = len(profile)
    result["diameter"] = float(
        np.linalg.norm(contour[idx_right_cp] - contour[idx_left_cp])
    )

    # --- Tangent fit ---
    try:
        angles, _, _, errors, timings = polynomial_fit(
            profile,
            num_points=N_TANGENT_POINTS,
            polynomial_degree=POLYNOMIAL_DEGREE_TANGENT,
            display=False,
        )
        result["tangent_left_angle"] = angles[0]
        result["tangent_right_angle"] = angles[1]
        result["tangent_fit_time"] = timings["fit time"]
        result["tangent_RMSE"] = _safe_error(errors, "RMSE")
    except Exception as e:
        result["tangent_error"] = str(e)

    # --- ML model (uses app's _predict_ml for bit-for-bit parity) ---
    if model is not None:
        try:
            ml_t0 = time.time()
            ml_left, ml_right = _predict_ml(
                corrected,
                contour,
                profile,
                idx_left_cp,
                idx_right_cp,
                left_line,
                right_line,
                model,
            )
            result["ml_left_angle"] = ml_left
            result["ml_right_angle"] = ml_right
            result["ml_prediction_time"] = time.time() - ml_t0
        except Exception as e:
            result["ml_error"] = str(e)

    return result


def process_dataset(dataset_name, dataset_config, model):
    """Process a single Kristina dataset.

    Args:
        dataset_name: Display name for the dataset.
        dataset_config: Config dict with 'subdir' and 'expected_tilt'.
        model: TF model.

    Returns:
        DataFrame of results.
    """
    subdir = dataset_config["subdir"]
    image_dir = os.path.join(KRISTINA_DIR, subdir)

    if not os.path.isdir(image_dir):
        print("  ERROR: Directory not found: {}".format(image_dir))
        return None

    # Prefer nested subdirectory if it exists and contains images
    nested = os.path.join(image_dir, subdir)
    if os.path.isdir(nested):
        image_dir = nested

    print()
    print("-" * 60)
    print("Processing: {} ({})".format(dataset_name, image_dir))
    print("-" * 60)

    # Collect images
    images = sorted(
        (f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTENSIONS)),
        key=natural_sort_key,
    )
    n_images = len(images)
    print("  Found {} images".format(n_images))

    if n_images == 0:
        return None

    # --- Pass 1: shared tilt + per-side median contact lines + residual IQRs
    total_frames = get_frame_count(image_dir)
    effective_stride = _effective_tilt_stride(
        total_frames,
        requested_skip=PASS1_REQUESTED_STRIDE,
    )
    total_samples = math.ceil(total_frames / effective_stride)
    print(
        "  Pass 1: stride={} (requested {}), {} samples".format(
            effective_stride, PASS1_REQUESTED_STRIDE, total_samples
        )
    )

    t0 = time.time()
    pass1 = _pass1_calibrate_with_tilt(
        source_path=image_dir,
        stride=effective_stride,
        total_samples=total_samples,
        progress=lambda *a: None,
        cancel=lambda: False,
    )
    (
        shared_tilt,
        left_line,
        right_line,
        left_stats,
        right_stats,
        _left_cps,
        _right_cps,
        _rotated_first_frame,
    ) = pass1
    print(
        "  Pass 1 done in {:.1f}s: shared_tilt={:.3f}°, "
        "left_IQR={:.2f}, right_IQR={:.2f}".format(
            time.time() - t0,
            shared_tilt,
            left_stats.get("residual_iqr", float("nan")),
            right_stats.get("residual_iqr", float("nan")),
        )
    )

    # --- Pass 2: per-frame analysis (mirrors _analyze_frame_tilted) ---
    results = []
    skipped = 0
    failures = 0
    t_start = time.time()

    for i, img_name in enumerate(images):
        img_path = os.path.join(image_dir, img_name)

        elapsed = time.time() - t_start
        eta_str = ""
        if i > 0:
            eta = elapsed / i * (n_images - i)
            eta_str = "  ETA: {:.1f}s".format(eta)
        print("  [{}/{}] {}{}".format(i + 1, n_images, img_name, eta_str))

        try:
            result = evaluate_image(
                img_path,
                frame_number=i + 1,
                shared_tilt=shared_tilt,
                left_line=left_line,
                right_line=right_line,
                left_stats=left_stats,
                right_stats=right_stats,
                model=model,
            )
        except Exception:
            result = {
                "image_name": img_name,
                "frame_number": i + 1,
                "error": traceback.format_exc()[:200],
            }

        if result.get("skipped"):
            skipped += 1
            print("    SKIPPED: {}".format(result.get("skip_reason", "")))
        elif "error" in result:
            failures += 1
            print("    ERROR: {}".format(str(result["error"])[:120]))

        results.append(result)

    total_time = time.time() - t_start
    df = pd.DataFrame(results)

    # Summary
    usable = n_images - skipped - failures
    print()
    print("  Done in {:.1f}s".format(total_time))
    print(
        "  Total: {}, Usable: {}, Skipped: {}, Failures: {}".format(
            n_images, usable, skipped, failures
        )
    )

    return df


def main():
    print("=" * 60)
    print("Kristina PEEK Dataset Evaluation (Tilted Surface)")
    print("=" * 60)

    # Suppress TF warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load model
    print("Loading ML model...")
    model = tf.keras.models.load_model(ML_MODEL_DIR)
    print("Model loaded.")

    if not os.path.isdir(KRISTINA_DIR):
        print("ERROR: Kristina data directory not found: {}".format(KRISTINA_DIR))
        print("Please copy data to paper_results/data/video_analysis/kristina/")
        sys.exit(1)

    # Process each dataset
    for dataset_name, config in KRISTINA_DATASETS.items():
        df = process_dataset(dataset_name, config, model)

        if df is not None:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            output_path = os.path.join(
                RESULTS_DIR,
                "kristina_{}_results.csv".format(
                    dataset_name.lower().replace(" ", "_")
                ),
            )
            df.to_csv(output_path, index=False)
            print("  Saved: {}".format(output_path))

            # Print tilt angle summary for usable frames
            usable = df[df.get("skipped", False) != True]
            if "detected_tilt_angle" in usable.columns:
                angles = usable["detected_tilt_angle"].dropna()
                if len(angles) > 0:
                    print(
                        "  Tilt angle: mean={:.1f}, std={:.1f}, range=[{:.1f}, {:.1f}]".format(
                            angles.mean(), angles.std(), angles.min(), angles.max()
                        )
                    )

            # Print method summary
            for method in ["tangent", "ml"]:
                left_col = "{}_left_angle".format(method)
                if left_col in usable.columns:
                    valid = usable[left_col].notna().sum()
                    print("  {}: {}/{} successful".format(method, valid, len(usable)))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
