#!/usr/bin/env python
"""Evaluate contact angles on the Kristina PEEK tilted-surface dataset.

Processing per image:
1. Load JPEG image
2. Detect tilt angle via Hough baseline
3. Rotate image to level
4. Standard pipeline: isolate_contour + CPID + fitting methods
5. Skip frames where drop extends beyond image edge

Run from OpenDrop_OP/:
    python paper_results/evaluate/evaluate_kristina.py
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
    KRISTINA_DIR,
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

from evaluate.tilt_correction import detect_tilt_angle, correct_tilt, touches_border

import tensorflow as tf


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


def extract_profile(contour, CPs):
    """Extract drop profile between contact points."""
    idx_left = np.where((contour == CPs[0]).all(axis=1))[0][0]
    idx_right = np.where((contour == CPs[1]).all(axis=1))[0][0]
    if idx_left > idx_right:
        idx_left, idx_right = idx_right, idx_left
    return contour[idx_left : idx_right + 1]


def _safe_error(errors, key, fallback_key=None):
    """Get error metric from dict, handling missing keys."""
    if not isinstance(errors, dict):
        return np.nan
    val = errors.get(key, np.nan)
    if val is np.nan and fallback_key:
        val = errors.get(fallback_key, np.nan)
    return val


def evaluate_image(img_path, model, frame_number):
    """Evaluate a single tilted-surface image.

    Returns a dict of results. Missing values are np.nan.
    """
    result = {
        "image_name": os.path.basename(img_path),
        "frame_number": frame_number,
    }

    # --- Load and correct tilt ---
    try:
        img = cv2.imread(img_path)
        if img is None:
            result["error"] = "Failed to load image"
            return result
        result["image_shape"] = str(img.shape)

        # Detect and correct tilt
        corrected, tilt_angle = correct_tilt(img, crop=True)
        result["detected_tilt_angle"] = tilt_angle
    except Exception as e:
        result["error"] = "Tilt correction failed: {}".format(e)
        return result

    # --- Preprocessing on corrected image ---
    try:
        t0 = time.time()
        img_processed, contour, drop_guess, info = isolate_contour(
            corrected, DISPLAY=False, blur=True
        )

        # Check if drop is fully visible (contour doesn't touch border)
        if touches_border(contour, corrected.shape, margin=2):
            result["skipped"] = True
            result["skip_reason"] = "Drop extends beyond image edge"
            return result

        CPs = CPID(contour, display=False)
        preprocess_time = time.time() - t0

        result["skipped"] = False
        result["preprocessing_time"] = preprocess_time
        result["threshold_value"] = info.get("threshold", np.nan)
        result["contour_length"] = len(contour)
        result["left_CP_x"] = int(CPs[0, 0])
        result["left_CP_y"] = int(CPs[0, 1])
        result["right_CP_x"] = int(CPs[1, 0])
        result["right_CP_y"] = int(CPs[1, 1])

        profile = extract_profile(contour, CPs)
        result["profile_length"] = len(profile)

        diameter = np.linalg.norm(CPs[1] - CPs[0])
        result["diameter"] = diameter
    except Exception as e:
        result["error"] = "Preprocessing failed: {}".format(e)
        return result

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

    # --- Polynomial fit ---
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
        result["polynomial_RMSE"] = _safe_error(errors, "RMSE")
    except Exception as e:
        result["polynomial_error"] = str(e)

    # --- Circular fit ---
    try:
        angles, center, radius, intercepts, errors, timings = circular_fit(
            profile, display=False
        )
        result["circular_left_angle"] = angles[0]
        result["circular_right_angle"] = angles[1]
        result["circular_RMSE"] = _safe_error(errors, "RMSE")
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
        result["ellipse_RMSE"] = _safe_error(errors, "RMSE")
    except Exception as e:
        result["ellipse_error"] = str(e)

    # --- Young-Laplace fit ---
    try:
        yl_angles, bond, bw, vol, fc, bl, errors, sym_errors, timings = yl_fit(
            profile, display=False
        )
        result["yl_left_angle"] = yl_angles[0]
        result["yl_right_angle"] = yl_angles[1]
        result["yl_bond_number"] = bond
        result["yl_RMSE"] = _safe_error(errors, "RMSE")
        result["drop_symmetry_RMSE"] = _safe_error(sym_errors, "RMSE")
    except Exception as e:
        result["yl_error"] = str(e)

    # --- ML model ---
    try:
        ml_t0 = time.time()
        pred_ds = prepare_for_model(
            corrected, input_len=50, n_surface=10, display=False, blur=True
        )
        ml_preprocess_time = time.time() - ml_t0

        predictions, ml_timings = experimental_pred(pred_ds, model, side="both")

        result["ml_left_angle"] = float(predictions[0])
        result["ml_right_angle"] = float(predictions[1])
        result["ml_preprocessing_time"] = ml_preprocess_time
        result["ml_prediction_time"] = ml_timings["fit time"]
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
            result = evaluate_image(img_path, model, frame_number=i + 1)
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
    print("  Total: {}, Usable: {}, Skipped: {}, Failures: {}".format(
        n_images, usable, skipped, failures))

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
                    print("  Tilt angle: mean={:.1f}, std={:.1f}, range=[{:.1f}, {:.1f}]".format(
                        angles.mean(), angles.std(), angles.min(), angles.max()))

            # Print method summary
            for method in ["tangent", "polynomial", "circular", "ellipse", "yl", "ml"]:
                left_col = "{}_left_angle".format(method)
                if left_col in usable.columns:
                    valid = usable[left_col].notna().sum()
                    print("  {}: {}/{} successful".format(
                        method, valid, len(usable)))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
