#!/usr/bin/env python
"""Evaluate v03 and v05.05 models on the Conan sensitivity dataset (120 images).

The sensitivity dataset is the original evaluation dataset from the jdber1/conan-ML
repository. It contains 120 synthetic images across 8 GT angles (110-180°),
5 Bond numbers (0-2), and 3 scales (1, 3.5, 6), all with roughness=0.

Filename format: {angle}_{bond}_{scale}_{roughness}_.png

Run from OpenDrop_OP/:
    python paper_results/evaluate/evaluate_sensitivity.py
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
    CONAN_MODEL_REPO_DIR,
    CONAN_MODEL_DIR,
    RESULTS_DIR,
    ML_MODEL_DIR,
    IMAGE_EXTENSIONS,
)

# v05.05 imports (current model)
from opendrop_ml.modules.ML_model.prepare_experimental import (
    prepare_for_model,
    experimental_pred,
)

# v03 imports — from the cloned conan-ML repo
V03_MODULE_DIR = os.path.join(CONAN_MODEL_REPO_DIR, "modules", "ML_model")
sys.path.insert(0, V03_MODULE_DIR)
from prepare_experimental import (
    prepare4model_v03_img,
    experimental_pred as v03_experimental_pred,
)

import tensorflow as tf

# --- Config ---
SENSITIVITY_DIR = os.path.join(CONAN_MODEL_REPO_DIR, "sensitivity data set")


def parse_sensitivity_filename(filename):
    """Parse ground truth from sensitivity dataset filename.

    Format: {angle}_{bond}_{scale}_{roughness}_.png
    Example: 110.0_0.0_1.0_0.0_.png
    """
    name = os.path.splitext(filename)[0]  # strip .png
    if name.endswith("_"):
        name = name[:-1]  # strip trailing underscore

    parts = name.split("_")
    if len(parts) != 4:
        raise ValueError(
            "Expected 4 parameters in filename, got {}: {}".format(len(parts), filename)
        )

    return {
        "gt_angle": float(parts[0]),
        "gt_bond": float(parts[1]),
        "gt_scale": float(parts[2]),
        "gt_roughness": float(parts[3]),
    }


def evaluate_image(img_path, model_v05, model_v03, gt):
    """Run both v05.05 and v03 models on a single image.

    Returns a dict of results.
    """
    result = {"image_name": os.path.basename(img_path)}
    result.update(gt)

    img = cv2.imread(img_path)
    if img is None:
        result["error"] = "Failed to load image"
        return result
    result["image_shape"] = str(img.shape)

    # --- v05.05 (current all-angle model) ---
    try:
        t0 = time.time()
        pred_ds = prepare_for_model(
            img, input_len=50, n_surface=10, display=False, blur=True
        )
        v05_preprocess_time = time.time() - t0

        predictions, timings = experimental_pred(pred_ds, model_v05, side="both")

        result["v05_ml_left"] = float(predictions[0])
        result["v05_ml_right"] = float(predictions[1])
        result["v05_preprocess_time"] = v05_preprocess_time
        result["v05_prediction_time"] = timings["fit time"]
    except Exception as e:
        result["v05_error"] = str(e)

    # --- v03 (old conan-ML model) ---
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

        result["v03_ml_left"] = float(predictions_v03[0])
        result["v03_ml_right"] = float(predictions_v03[1])
        result["v03_preprocess_time"] = v03_preprocess_time
        result["v03_prediction_time"] = v03_prediction_time
    except Exception as e:
        result["v03_error"] = str(e)

    # --- Deltas ---
    if "v05_ml_left" in result and "v03_ml_left" in result:
        result["delta_left"] = result["v05_ml_left"] - result["v03_ml_left"]
        result["delta_right"] = result["v05_ml_right"] - result["v03_ml_right"]

    return result


def report(df):
    """Print comparison summary."""
    print()
    print("=" * 70)
    print("SENSITIVITY DATASET: v05.05 vs v03 ({} images)".format(len(df)))
    print("=" * 70)

    # Check for errors
    for tag in ["v05", "v03"]:
        err_col = "{}_error".format(tag)
        if err_col in df.columns:
            n_err = df[err_col].notna().sum()
            if n_err > 0:
                print("  WARNING: {} {} errors".format(n_err, tag))

    # Filter to images where both models succeeded
    has_both = df["v05_ml_left"].notna() & df["v03_ml_left"].notna()
    valid = df[has_both].copy()
    print("  Images with both models: {}/{}".format(len(valid), len(df)))

    # Overall accuracy vs GT
    for tag, left_col, right_col in [
        ("v05.05", "v05_ml_left", "v05_ml_right"),
        ("v03", "v03_ml_left", "v03_ml_right"),
    ]:
        err_l = (valid[left_col] - valid["gt_angle"]).abs()
        err_r = (valid[right_col] - valid["gt_angle"]).abs()
        err_avg = (err_l + err_r) / 2.0
        print()
        print("  {} vs GT (overall):".format(tag))
        print("    MAE (L):   {:.2f}".format(err_l.mean()))
        print("    MAE (R):   {:.2f}".format(err_r.mean()))
        print("    MAE (avg): {:.2f}".format(err_avg.mean()))
        print("    Max err:   {:.2f}".format(err_avg.max()))

    # Per GT-angle breakdown
    print()
    print("  Per GT-angle MAE (avg of L+R):")
    print("  {:<10} {:>10} {:>10} {:>6}".format("GT angle", "v05.05", "v03", "n"))
    print("  " + "-" * 40)
    for gt in sorted(valid["gt_angle"].unique()):
        sub = valid[valid["gt_angle"] == gt]
        v05_err = (
            (sub["v05_ml_left"] - gt).abs() + (sub["v05_ml_right"] - gt).abs()
        ) / 2.0
        v03_err = (
            (sub["v03_ml_left"] - gt).abs() + (sub["v03_ml_right"] - gt).abs()
        ) / 2.0
        print(
            "  {:<10} {:10.2f} {:10.2f} {:6d}".format(
                int(gt), v05_err.mean(), v03_err.mean(), len(sub)
            )
        )

    # Per Bond number breakdown
    print()
    print("  Per Bond number MAE (avg of L+R):")
    print("  {:<10} {:>10} {:>10} {:>6}".format("Bond", "v05.05", "v03", "n"))
    print("  " + "-" * 40)
    for bond in sorted(valid["gt_bond"].unique()):
        sub = valid[valid["gt_bond"] == bond]
        v05_err = (
            (sub["v05_ml_left"] - sub["gt_angle"]).abs()
            + (sub["v05_ml_right"] - sub["gt_angle"]).abs()
        ) / 2.0
        v03_err = (
            (sub["v03_ml_left"] - sub["gt_angle"]).abs()
            + (sub["v03_ml_right"] - sub["gt_angle"]).abs()
        ) / 2.0
        print(
            "  {:<10} {:10.2f} {:10.2f} {:6d}".format(
                bond, v05_err.mean(), v03_err.mean(), len(sub)
            )
        )

    # Per Scale breakdown
    print()
    print("  Per Scale MAE (avg of L+R):")
    print("  {:<10} {:>10} {:>10} {:>6}".format("Scale", "v05.05", "v03", "n"))
    print("  " + "-" * 40)
    for scale in sorted(valid["gt_scale"].unique()):
        sub = valid[valid["gt_scale"] == scale]
        v05_err = (
            (sub["v05_ml_left"] - sub["gt_angle"]).abs()
            + (sub["v05_ml_right"] - sub["gt_angle"]).abs()
        ) / 2.0
        v03_err = (
            (sub["v03_ml_left"] - sub["gt_angle"]).abs()
            + (sub["v03_ml_right"] - sub["gt_angle"]).abs()
        ) / 2.0
        print(
            "  {:<10} {:10.2f} {:10.2f} {:6d}".format(
                scale, v05_err.mean(), v03_err.mean(), len(sub)
            )
        )

    # Direct v05 vs v03 deltas
    delta_l = valid["delta_left"].abs()
    delta_r = valid["delta_right"].abs()
    print()
    print("  v05.05 - v03 deltas:")
    print("    MAE (L):   {:.2f}".format(delta_l.mean()))
    print("    MAE (R):   {:.2f}".format(delta_r.mean()))
    print("    Max |delta| (L): {:.2f}".format(delta_l.max()))
    print("    Max |delta| (R): {:.2f}".format(delta_r.max()))


def main():
    print("=" * 70)
    print("Sensitivity Dataset Evaluation: v05.05 vs v03")
    print("=" * 70)

    # Suppress TF warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load v05.05 model
    print("Loading v05.05 model...")
    model_v05 = tf.keras.models.load_model(ML_MODEL_DIR)
    print("  v05.05 loaded: input shape {}".format(model_v05.input_shape))

    # Load v03 model
    print("Loading v03 model...")
    model_v03 = tf.keras.models.load_model(CONAN_MODEL_DIR)
    print("  v03 loaded: input shape {}".format(model_v03.input_shape))

    # Collect images
    images = sorted(
        f for f in os.listdir(SENSITIVITY_DIR) if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    n_images = len(images)
    print("Found {} images in {}".format(n_images, SENSITIVITY_DIR))

    if n_images == 0:
        print("ERROR: No images found.")
        sys.exit(1)

    # Evaluate all images
    results = []
    failures = 0
    t_start = time.time()

    for i, img_name in enumerate(images):
        img_path = os.path.join(SENSITIVITY_DIR, img_name)

        elapsed = time.time() - t_start
        eta_str = ""
        if i > 0:
            eta = elapsed / i * (n_images - i)
            eta_str = "  ETA: {:.1f} min".format(eta / 60)
        print("  [{}/{}] {}{}".format(i + 1, n_images, img_name, eta_str))

        # Parse GT from filename
        try:
            gt = parse_sensitivity_filename(img_name)
        except ValueError as e:
            result = {"image_name": img_name, "error": str(e)}
            results.append(result)
            failures += 1
            print("    ERROR: {}".format(e))
            continue

        try:
            result = evaluate_image(img_path, model_v05, model_v03, gt)
        except Exception:
            result = {"image_name": img_name, "error": traceback.format_exc()}
            result.update(gt)

        if "error" in result or "v05_error" in result or "v03_error" in result:
            err_msg = result.get(
                "error", result.get("v05_error", result.get("v03_error", ""))
            )
            failures += 1
            print("    ERROR: {}".format(str(err_msg)[:120]))

        results.append(result)

        # Save incrementally every 10 images
        if (i + 1) % 10 == 0 or (i + 1) == n_images:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            output_path = os.path.join(RESULTS_DIR, "sensitivity_results.csv")
            pd.DataFrame(results).to_csv(output_path, index=False)

    total_time = time.time() - t_start

    # Final save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "sensitivity_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    # Summary
    print()
    print("Done in {:.1f} minutes.".format(total_time / 60))
    print("Results: {} images, {} with errors".format(n_images, failures))
    print("Saved to: {}".format(output_path))

    report(df)

    print()
    print("=" * 70)
    print("Sensitivity evaluation complete.")


if __name__ == "__main__":
    main()
