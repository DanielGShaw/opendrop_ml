#!/usr/bin/env python
"""Two-pass evaluation of the inverse phase image dataset.

Implements the same median-contact-line approach as the reference
Inverse_phase_analysis_executable.py:
  Pass 1: Sample frames to determine median contact points & fit contact lines
  Pass 2: Use median contact lines to find stable CPs for all frames

Run from OpenDrop_OP/:
    python paper_results/evaluate/evaluate_inverse_phase.py
"""

import sys
import os
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
    INVERSE_PHASE_DIR,
    RESULTS_DIR,
    ML_MODEL_DIR,
    CONAN_MODEL_REPO_DIR,
    CONAN_MODEL_DIR,
    N_TANGENT_POINTS,
    POLYNOMIAL_DEGREE_TANGENT,
    POLYNOMIAL_DEGREE_POLY,
    IMAGE_EXTENSIONS,
)

from opendrop_ml.modules.preprocessing.preprocessing import (
    isolate_contour,
    CPID,
    linear_fit,
)
from opendrop_ml.modules.ML_model.prepare_experimental import (
    normalize_ds,
    experimental_pred,
    resize_image,
)
from opendrop_ml.modules.fitting.polynomial_fit import polynomial_fit
from opendrop_ml.modules.fitting.circular_fit import circular_fit
from opendrop_ml.modules.fitting.ellipse_fit import ellipse_fit
from opendrop_ml.modules.fitting.BA_fit import yl_fit
from evaluate.needle_masking import find_first_intercept

import tensorflow as tf

# --- v03 model imports (from cloned conan-ML repo) ---
V03_MODULE_DIR = os.path.join(CONAN_MODEL_REPO_DIR, "modules", "ML_model")
sys.path.insert(0, V03_MODULE_DIR)
from prepare_experimental import (
    prepare4model_v03_img,
    experimental_pred as v03_experimental_pred,
)


# --- Resolution thresholds (from Inverse_phase_analysis_executable.py) ---
PROFILE_MAX_PLATEAU = 974  # Max plateau of profile length in training data
PROFILE_EMPIRICAL_MIN = 565  # Lowest max profile length in training data
PROFILE_MIN_160DEG = 234  # Min length of 160 deg drop profile
RESOLUTION_DROP_SLOPE = 6.383333333333334
RESOLUTION_DROP_INTERCEPT = 208.0

# --- Two-pass calibration settings ---
MEDIAN_STEP = 10  # Sample every Nth image for Pass 1


def natural_sort_key(s):
    """Sort key that handles embedded numbers naturally."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(s))]


def _safe_error(errors, key, fallback_key=None):
    """Get error metric from dict, handling missing keys and non-dict errors."""
    if not isinstance(errors, dict):
        return np.nan
    val = errors.get(key, np.nan)
    if val is np.nan and fallback_key:
        val = errors.get(fallback_key, np.nan)
    return val


def _fit_contact_line(CPs):
    """Fit a linear contact line through contact points with IQR filtering.

    Matches the reference Inverse_phase_analysis_executable.py approach:
    filter CPs to y-coordinate IQR range, then fit a line.
    """
    q1 = np.percentile(CPs[:, 1], 25)
    q3 = np.percentile(CPs[:, 1], 75)
    CPs_inrange = np.array([cp for cp in CPs if q1 <= cp[1] <= q3])

    if len(CPs_inrange) < 2:
        CPs_inrange = CPs

    try:
        m, c, _ = linear_fit(CPs_inrange, display=False)
    except Exception:
        med = np.median(CPs, axis=0)
        return np.array([[CPs[:, 0].min(), med[1]], [CPs[:, 0].max(), med[1]]])

    start_x = CPs[:, 0].min()
    end_x = CPs[:, 0].max()
    start_y = m * start_x + c
    end_y = m * end_x + c

    return np.array([[start_x, start_y], [end_x, end_y]])



def _check_resolution_drop(profile):
    """Check if image resolution needs dropping for ML model."""
    profile_len = len(profile)

    if profile_len > PROFILE_MAX_PLATEAU:
        return True

    if PROFILE_EMPIRICAL_MIN < profile_len < PROFILE_MAX_PLATEAU:
        try:
            circ_angles, _, _, _, _, _ = circular_fit(profile, display=False)
            threshold = RESOLUTION_DROP_SLOPE * min(circ_angles) + RESOLUTION_DROP_INTERCEPT
            if profile_len > threshold:
                return True
        except Exception:
            pass

    return False


# -----------------------------------------------------------------------
# Pass 1: Calibration — collect CPs to compute median contact lines
# -----------------------------------------------------------------------

def run_pass1(images, step=MEDIAN_STEP):
    """Run Pass 1: sample images, collect CPs for median contact line fitting.

    Returns:
        (left_contact_line, right_contact_line) — each is [[x1,y1],[x2,y2]]
    """
    left_CPs = []
    right_CPs = []
    successes = 0
    failures = 0

    for i in range(0, len(images), step):
        img_path = os.path.join(INVERSE_PHASE_DIR, images[i])
        try:
            img = cv2.imread(img_path)
            if img is None:
                failures += 1
                continue
            _, contour, _, _ = isolate_contour(img, DISPLAY=False, blur=True)
            CPs = CPID(contour, display=False)
            if CPs is None or len(CPs) < 2:
                failures += 1
                continue

            # Ensure left CP has smaller x
            if CPs[0][0] < CPs[1][0]:
                left_CPs.append(CPs[0].copy())
                right_CPs.append(CPs[1].copy())
            else:
                left_CPs.append(CPs[1].copy())
                right_CPs.append(CPs[0].copy())
            successes += 1
        except Exception:
            failures += 1

    print("  Pass 1: {} sampled, {} succeeded, {} failed".format(
        successes + failures, successes, failures))

    if successes < 3:
        raise RuntimeError("Pass 1 failed: too few successful frames ({})".format(
            successes))

    left_CPs_arr = np.array(left_CPs)
    right_CPs_arr = np.array(right_CPs)

    left_line = _fit_contact_line(left_CPs_arr)
    right_line = _fit_contact_line(right_CPs_arr)

    left_median = np.median(left_CPs_arr, axis=0)
    right_median = np.median(right_CPs_arr, axis=0)
    print("  Left median CP:  ({:.1f}, {:.1f})".format(*left_median))
    print("  Right median CP: ({:.1f}, {:.1f})".format(*right_median))

    return left_line, right_line


# -----------------------------------------------------------------------
# Pass 2: Full analysis using median contact lines
# -----------------------------------------------------------------------

def evaluate_image(img_path, model, frame_number,
                   left_contact_line, right_contact_line,
                   model_v03=None):
    """Process a single image using median contact lines for stable CPs.

    Follows the reference Inverse_phase_analysis_executable.py:
    1. isolate_contour + CPID (for approximate midpoint only)
    2. Split contour at midpoint into left/right halves
    3. find_first_intercept with median contact lines → stable CPs
    4. If contour doesn't reach contact line, skip frame (NaN)
    5. Run all fitting methods + ML prediction (v05 and optionally v03)
    """
    result = {
        "image_name": os.path.basename(img_path),
        "frame_number": frame_number,
    }

    # --- Preprocessing ---
    try:
        t0 = time.time()
        img = cv2.imread(img_path)
        if img is None:
            result["error"] = "Failed to load image"
            return result
        result["image_shape"] = str(img.shape)

        img_processed, contour, drop_guess, info = isolate_contour(
            img, DISPLAY=False, blur=True
        )
        CPs_orig = CPID(contour, display=False)
        preprocess_time = time.time() - t0
        result["preprocessing_time"] = preprocess_time
        result["threshold_value"] = info.get("threshold", np.nan)
        result["contour_length"] = len(contour)

        if CPs_orig is None or len(CPs_orig) < 2:
            result["error"] = "CPID failed"
            return result

        # Record raw CPID contact points (before median-line stabilization)
        # Ensure left CP has smaller x
        if CPs_orig[0][0] <= CPs_orig[1][0]:
            result["cpid_left_x"] = int(CPs_orig[0][0])
            result["cpid_left_y"] = int(CPs_orig[0][1])
            result["cpid_right_x"] = int(CPs_orig[1][0])
            result["cpid_right_y"] = int(CPs_orig[1][1])
        else:
            result["cpid_left_x"] = int(CPs_orig[1][0])
            result["cpid_left_y"] = int(CPs_orig[1][1])
            result["cpid_right_x"] = int(CPs_orig[0][0])
            result["cpid_right_y"] = int(CPs_orig[0][1])

        # Get original CP indices for midpoint calculation
        idx_left_orig = np.where((contour == CPs_orig[0]).all(axis=1))[0][0]
        idx_right_orig = np.where((contour == CPs_orig[1]).all(axis=1))[0][0]
        if idx_left_orig > idx_right_orig:
            idx_left_orig, idx_right_orig = idx_right_orig, idx_left_orig

        # Split contour at midpoint (matching reference executable)
        midpoint_idx = int(
            idx_left_orig + (idx_right_orig - idx_left_orig) / 2
        )
        left_half = contour[:midpoint_idx][::-1]  # reversed: apex → surface
        right_half = contour[midpoint_idx:]

        # Find new CPs using median contact lines.
        left_CP_local = find_first_intercept(left_half, left_contact_line)
        right_CP_local = find_first_intercept(right_half, right_contact_line)

        # When find_first_intercept returns -1, the contour doesn't reach
        # the contact line (edge detection stopped short). These frames have
        # unreliable geometry — skip analysis and record as NaN.
        if left_CP_local == -1 or right_CP_local == -1:
            result["skipped"] = "contour_short"
            result["left_CP_fallback"] = "skipped_short_contour"
            result["right_CP_fallback"] = "skipped_short_contour"
            return result

        # Map local indices back to original contour.
        # left_half = contour[:midpoint][::-1], so local i → original (midpoint-1-i)
        idx_leftCP = midpoint_idx - 1 - left_CP_local
        # right_half = contour[midpoint:], so local i → original (midpoint+i)
        idx_rightCP = midpoint_idx + right_CP_local

        if idx_leftCP > idx_rightCP:
            idx_leftCP, idx_rightCP = idx_rightCP, idx_leftCP

        # Record the stable CPs
        result["left_CP_x"] = int(contour[idx_leftCP, 0])
        result["left_CP_y"] = int(contour[idx_leftCP, 1])
        result["right_CP_x"] = int(contour[idx_rightCP, 0])
        result["right_CP_y"] = int(contour[idx_rightCP, 1])

        # Extract profile between stable CPs
        profile = contour[idx_leftCP: idx_rightCP + 1]
        result["profile_length"] = len(profile)

        # Diameter
        distance = np.linalg.norm(
            contour[idx_rightCP] - contour[idx_leftCP]
        )
        result["diameter"] = distance

    except Exception as e:
        result["error"] = "Preprocessing failed: {}".format(e)
        return result

    # --- Tangent fit (polynomial degree=1, num_points=10) ---
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
        result["tangent_analysis_time"] = timings["analysis time"]
        result["tangent_RMSE"] = _safe_error(errors, "RMSE")
        result["tangent_max_error"] = _safe_error(errors, "Maximum error")
    except Exception as e:
        result["tangent_error"] = str(e)

    # --- Polynomial fit (degree=2, num_points=10) ---
    try:
        angles, _, _, errors, timings = polynomial_fit(
            profile,
            num_points=N_TANGENT_POINTS,
            polynomial_degree=POLYNOMIAL_DEGREE_POLY,
            display=False,
        )
        result["polynomial_left_angle"] = angles[0]
        result["polynomial_right_angle"] = angles[1]
        result["polynomial_fit_time"] = timings["fit time"]
        result["polynomial_analysis_time"] = timings["analysis time"]
        result["polynomial_RMSE"] = _safe_error(errors, "RMSE")
        result["polynomial_max_error"] = _safe_error(errors, "Maximum error")
    except Exception as e:
        result["polynomial_error"] = str(e)

    # --- Circular fit ---
    try:
        angles, center, radius, intercepts, errors, timings = circular_fit(
            profile, display=False
        )
        result["circular_left_angle"] = angles[0]
        result["circular_right_angle"] = angles[1]
        result["circular_fit_time"] = timings["fit time"]
        result["circular_analysis_time"] = timings["analysis time"]
        result["circular_RMSE"] = _safe_error(errors, "RMSE")
        result["circular_max_error"] = _safe_error(errors, "Maximum error")
    except Exception as e:
        result["circular_error"] = str(e)

    # --- Ellipse fit ---
    try:
        angles, intercepts, center, axes, phi, errors, timings = ellipse_fit(
            profile, display=False
        )
        if angles:
            result["ellipse_left_angle"] = angles[0]
            result["ellipse_right_angle"] = angles[1]
        else:
            result["ellipse_left_angle"] = np.nan
            result["ellipse_right_angle"] = np.nan
        result["ellipse_fit_time"] = timings["fit time"]
        result["ellipse_analysis_time"] = timings["analysis time"]
        result["ellipse_RMSE"] = _safe_error(errors, "RMSE")
        result["ellipse_max_error"] = _safe_error(errors, "Maximum error")
    except Exception as e:
        result["ellipse_error"] = str(e)

    # --- Young-Laplace fit ---
    try:
        yl_angles, bond, baseline_width, volume, fit_contour, baseline, errors, sym_errors, timings = yl_fit(
            profile, display=False
        )
        result["yl_left_angle"] = yl_angles[0]
        result["yl_right_angle"] = yl_angles[1]
        result["yl_bond_number"] = bond
        result["yl_fit_time"] = timings["fit time"]
        result["yl_analysis_time"] = timings["analysis time"]
        result["yl_RMSE"] = _safe_error(errors, "RMSE")
        result["yl_max_error"] = _safe_error(errors, "Maximum error")
        result["drop_symmetry_RMSE"] = _safe_error(sym_errors, "RMSE")
    except Exception as e:
        result["yl_error"] = str(e)

    # --- ML model (using median-line-derived CPs for window extraction) ---
    try:
        ml_t0 = time.time()

        # Resolution check
        needs_drop = _check_resolution_drop(profile)
        if needs_drop:
            result["resolution_dropped_for_ml"] = True
            img_small = resize_image(img, scale=0.5)
            _, contour_ml, _, _ = isolate_contour(
                img_small, DISPLAY=False, blur=True
            )
            CPs_ml = CPID(contour_ml, display=False)
            if CPs_ml is None or len(CPs_ml) < 2:
                raise ValueError("CPID failed on resized image")
            idx_l_ml = np.where((contour_ml == CPs_ml[0]).all(axis=1))[0][0]
            idx_r_ml = np.where((contour_ml == CPs_ml[1]).all(axis=1))[0][0]
            if idx_l_ml > idx_r_ml:
                idx_l_ml, idx_r_ml = idx_r_ml, idx_l_ml
            # Re-apply median contact lines on resized contour
            mid_ml = int(idx_l_ml + (idx_r_ml - idx_l_ml) / 2)
            left_h = contour_ml[:mid_ml][::-1]
            right_h = contour_ml[mid_ml:]
            ll = find_first_intercept(left_h, left_contact_line * 0.5)
            rr = find_first_intercept(right_h, right_contact_line * 0.5)
            if ll >= 0:
                idx_l_ml = mid_ml - 1 - ll
            if rr >= 0:
                idx_r_ml = mid_ml + rr
            if idx_l_ml > idx_r_ml:
                idx_l_ml, idx_r_ml = idx_r_ml, idx_l_ml
            contour_for_ml = contour_ml
            idx_leftCP_ml = idx_l_ml
            idx_rightCP_ml = idx_r_ml
        else:
            result["resolution_dropped_for_ml"] = False
            contour_for_ml = contour
            idx_leftCP_ml = idx_leftCP
            idx_rightCP_ml = idx_rightCP

        # Extract ML windows (matching prepare_for_model logic)
        n_surface = 10
        n_drop = 40
        input_len = 50

        if idx_leftCP_ml < n_surface:
            left_side = contour_for_ml[:input_len]
        else:
            left_side = contour_for_ml[
                idx_leftCP_ml - n_surface: idx_leftCP_ml + n_drop
            ]

        if idx_rightCP_ml > len(contour_for_ml) - n_surface:
            right_side = contour_for_ml[-input_len:]
        else:
            right_side = contour_for_ml[
                idx_rightCP_ml - n_drop: idx_rightCP_ml + n_surface
            ]

        # Reverse and reflect left side (training convention)
        left_side = left_side[::-1].copy()
        left_side[:, 0] = -left_side[:, 0]

        ml_preprocess_time = time.time() - ml_t0

        if len(left_side) >= 4 and len(right_side) >= 4:
            pred_ds = np.array([left_side, right_side])
            pred_ds = normalize_ds(pred_ds).astype(np.float32)
            predictions, ml_timings = experimental_pred(
                pred_ds, model, side="both"
            )
            result["ml_left_angle"] = float(predictions[0])
            result["ml_right_angle"] = float(predictions[1])
            result["ml_preprocessing_time"] = ml_preprocess_time
            result["ml_prediction_time"] = ml_timings["fit time"]
            result["ml_analysis_time"] = ml_timings["analysis time"]
        else:
            result["ml_error"] = "ML window too short: L={}, R={}".format(
                len(left_side), len(right_side))
    except Exception as e:
        result["ml_error"] = str(e)

    # --- v03 ML model (independent preprocessing, no median contact lines) ---
    if model_v03 is not None:
        try:
            t0 = time.time()
            pred_ds_v03 = prepare4model_v03_img(
                img, cluster=True, input_len=1223, display=False
            )
            v03_preprocess_time = time.time() - t0

            t0 = time.time()
            predictions_v03, timings_v03 = v03_experimental_pred(
                pred_ds_v03, model_v03, side="both"
            )
            v03_prediction_time = time.time() - t0

            result["v03_ml_left_angle"] = float(predictions_v03[0])
            result["v03_ml_right_angle"] = float(predictions_v03[1])
            result["v03_preprocess_time"] = v03_preprocess_time
            result["v03_prediction_time"] = v03_prediction_time
        except Exception as e:
            result["v03_error"] = str(e)

    return result


def main():
    print("=" * 60)
    print("Inverse Phase Dataset Evaluation (Two-Pass)")
    print("=" * 60)

    # Suppress TF warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load models
    print("Loading ML models...")
    model = tf.keras.models.load_model(ML_MODEL_DIR)
    print("  v05 loaded: input shape {}".format(model.input_shape))
    model_v03 = tf.keras.models.load_model(CONAN_MODEL_DIR)
    print("  v03 loaded: input shape {}".format(model_v03.input_shape))

    # Collect images
    if not os.path.isdir(INVERSE_PHASE_DIR):
        print("ERROR: Inverse phase data directory not found: {}".format(
            INVERSE_PHASE_DIR))
        print("Please copy data to paper_results/data/video_analysis/inverse_phase/")
        sys.exit(1)

    images = sorted(
        (f for f in os.listdir(INVERSE_PHASE_DIR)
         if f.lower().endswith(IMAGE_EXTENSIONS)),
        key=natural_sort_key,
    )
    n_images = len(images)
    print("Found {} images in {}".format(n_images, INVERSE_PHASE_DIR))

    if n_images == 0:
        print("ERROR: No images found.")
        sys.exit(1)

    # ===================================================================
    # PASS 1: Calibration — compute median contact lines
    # ===================================================================
    print()
    print("PASS 1: Calibration (every {}th image)...".format(MEDIAN_STEP))
    t0 = time.time()
    left_contact_line, right_contact_line = run_pass1(images, step=MEDIAN_STEP)
    print("  Pass 1 completed in {:.1f}s".format(time.time() - t0))

    # ===================================================================
    # PASS 2: Full analysis using median contact lines
    # ===================================================================
    print()
    print("PASS 2: Full analysis ({} images)...".format(n_images))

    results = []
    failures = 0
    t_start = time.time()

    for i, img_name in enumerate(images):
        img_path = os.path.join(INVERSE_PHASE_DIR, img_name)

        elapsed = time.time() - t_start
        eta_str = ""
        if i > 0:
            eta = elapsed / i * (n_images - i)
            eta_str = "  ETA: {:.1f} min".format(eta / 60)
        print("  [{}/{}] {}{}".format(i + 1, n_images, img_name, eta_str))

        try:
            result = evaluate_image(
                img_path, model, frame_number=i + 1,
                left_contact_line=left_contact_line,
                right_contact_line=right_contact_line,
                model_v03=model_v03,
            )
        except Exception:
            result = {
                "image_name": img_name,
                "frame_number": i + 1,
                "error": traceback.format_exc(),
            }

        if "error" in result:
            failures += 1
            print("    ERROR: {}".format(str(result["error"])[:120]))

        results.append(result)

        # Save incrementally every 10 images
        if (i + 1) % 10 == 0 or (i + 1) == n_images:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            output_path = os.path.join(RESULTS_DIR, "inverse_phase_results.csv")
            pd.DataFrame(results).to_csv(output_path, index=False)

    total_time = time.time() - t_start

    # Final save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "inverse_phase_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    # Summary
    print()
    print("Done in {:.1f} minutes.".format(total_time / 60))
    print("Results: {} images, {} failures".format(n_images, failures))
    print("Saved to: {}".format(output_path))
    print()

    for method in ["tangent", "polynomial", "circular", "ellipse", "yl", "ml", "v03_ml"]:
        left_col = "{}_left_angle".format(method)
        if left_col in df.columns:
            valid = df[left_col].notna().sum()
            print("  {}: {}/{} successful".format(method, valid, n_images))

    return df


if __name__ == "__main__":
    main()
