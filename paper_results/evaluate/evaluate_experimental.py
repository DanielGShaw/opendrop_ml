#!/usr/bin/env python
"""Evaluate all fitting methods on the 169 experimental images.

Run from OpenDrop_OP/:
    python paper_results/evaluate/evaluate_experimental.py
"""

import sys
import os
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
    EXPERIMENTAL_DATA_DIR,
    RESULTS_DIR,
    ML_MODEL_DIR,
    N_TANGENT_POINTS,
    POLYNOMIAL_DEGREE_TANGENT,
    POLYNOMIAL_DEGREE_POLY,
    IMAGE_EXTENSIONS,
)

from opendrop_ml.modules.preprocessing.preprocessing import isolate_contour, CPID
from opendrop_ml.modules.ML_model.prepare_experimental import (
    prepare_for_model,
    experimental_pred,
)
from opendrop_ml.modules.fitting.polynomial_fit import polynomial_fit
from opendrop_ml.modules.fitting.circular_fit import circular_fit
from opendrop_ml.modules.fitting.ellipse_fit import ellipse_fit
from opendrop_ml.modules.fitting.BA_fit import yl_fit

import tensorflow as tf


def extract_profile(contour, CPs):
    """Extract drop profile between contact points."""
    idx_left = np.where((contour == CPs[0]).all(axis=1))[0][0]
    idx_right = np.where((contour == CPs[1]).all(axis=1))[0][0]
    if idx_left > idx_right:
        idx_left, idx_right = idx_right, idx_left
    return contour[idx_left : idx_right + 1]


def _safe_error(errors, key, fallback_key=None):
    """Get error metric from dict, handling missing keys and non-dict errors."""
    if not isinstance(errors, dict):
        return np.nan
    val = errors.get(key, np.nan)
    if val is np.nan and fallback_key:
        val = errors.get(fallback_key, np.nan)
    return val


def evaluate_image(img_path, model):
    """Run all fitting methods on a single image.

    Returns a dict of results. Missing values are np.nan.
    """
    result = {"image_name": os.path.basename(img_path)}

    # --- Preprocessing (imread + isolate_contour + CPID) ---
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
        CPs = CPID(contour, display=False)
        preprocess_time = time.time() - t0

        result["preprocessing_time"] = preprocess_time
        result["threshold_value"] = info.get("threshold", np.nan)
        result["contour_length"] = len(contour)
        result["left_CP_x"] = int(CPs[0, 0])
        result["left_CP_y"] = int(CPs[0, 1])
        result["right_CP_x"] = int(CPs[1, 0])
        result["right_CP_y"] = int(CPs[1, 1])

        profile = extract_profile(contour, CPs)
        result["profile_length"] = len(profile)
    except Exception as e:
        result["error"] = f"Preprocessing failed: {e}"
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
        if angles:  # returns [] if invalid
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
        (
            yl_angles,
            bond,
            baseline_width,
            volume,
            fit_contour,
            baseline,
            errors,
            sym_errors,
            timings,
        ) = yl_fit(profile, display=False)
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

    # --- ML model ---
    try:
        ml_t0 = time.time()
        pred_ds = prepare_for_model(
            img, input_len=50, n_surface=10, display=False, blur=True
        )
        ml_preprocess_time = time.time() - ml_t0

        predictions, ml_timings = experimental_pred(pred_ds, model, side="both")

        result["ml_left_angle"] = float(predictions[0])
        result["ml_right_angle"] = float(predictions[1])
        result["ml_preprocessing_time"] = ml_preprocess_time
        result["ml_prediction_time"] = ml_timings["fit time"]
        result["ml_analysis_time"] = ml_timings["analysis time"]
    except Exception as e:
        result["ml_error"] = str(e)

    return result


def main():
    print("=" * 60)
    print("Experimental Dataset Evaluation")
    print("=" * 60)

    # Suppress TF warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load model
    print("Loading ML model...")
    model = tf.keras.models.load_model(ML_MODEL_DIR)
    print("Model loaded.")

    # Collect images
    images = sorted(
        f
        for f in os.listdir(EXPERIMENTAL_DATA_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    n_images = len(images)
    print(f"Found {n_images} images in {EXPERIMENTAL_DATA_DIR}")

    if n_images == 0:
        print("ERROR: No images found. Check EXPERIMENTAL_DATA_DIR in config.py")
        sys.exit(1)

    # Evaluate all images
    results = []
    failures = 0
    t_start = time.time()

    for i, img_name in enumerate(images):
        img_path = os.path.join(EXPERIMENTAL_DATA_DIR, img_name)

        elapsed = time.time() - t_start
        eta_str = ""
        if i > 0:
            eta = elapsed / i * (n_images - i)
            eta_str = f"  ETA: {eta / 60:.1f} min"
        print(f"  [{i + 1}/{n_images}] {img_name}{eta_str}")

        try:
            result = evaluate_image(img_path, model)
        except Exception:
            result = {
                "image_name": img_name,
                "error": traceback.format_exc(),
            }

        if "error" in result:
            failures += 1
            print(f"    ERROR: {result['error'][:120]}")

        results.append(result)

        # Save incrementally every 10 images (fault tolerance)
        if (i + 1) % 10 == 0 or (i + 1) == n_images:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            output_path = os.path.join(RESULTS_DIR, "experimental_results.csv")
            pd.DataFrame(results).to_csv(output_path, index=False)

    total_time = time.time() - t_start

    # Final save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "experimental_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    # Summary
    print()
    print(f"Done in {total_time / 60:.1f} minutes.")
    print(f"Results: {n_images} images, {failures} failures")
    print(f"Saved to: {output_path}")
    print()

    for method in ["tangent", "polynomial", "circular", "ellipse", "yl", "ml"]:
        left_col = f"{method}_left_angle"
        if left_col in df.columns:
            valid = df[left_col].notna().sum()
            print(f"  {method}: {valid}/{n_images} successful")

    return df


if __name__ == "__main__":
    main()
