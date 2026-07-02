#!/usr/bin/env python
"""Two-pass needle-in video analysis for Isaac datasets.

Thin driver over the app's needle-in pipeline
(``opendrop_ml.modules.contact_angle.video_two_pass_needle_in``). This
script owns the per-dataset video configs and the paper-specific CSV
schema (adds ``yl_bond_number``, ``*_RMSE``, CP coords, diameter, and
per-method ``*_error`` strings that the UI's ``UnifiedCsvWriter``
doesn't carry).

Pass 1 samples every ``median_step``-th frame via
``process_frame_pass1``; ``compute_pass1_calibration`` reduces the
samples to median contact lines, median needle-intercept points, and
the auto-derived crop. Pass 2 runs ``extract_pass2_frame`` per frame
and then this script's :func:`evaluate_frame` over the extracted
profile.

Run from OpenDrop_OP/:
    python paper_results/evaluate/evaluate_isaac.py
"""

import sys
import os
import time
import traceback

import numpy as np
import pandas as pd

# Ensure imports work regardless of working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_RESULTS_DIR = os.path.dirname(SCRIPT_DIR)
OPENDROP_DIR = os.path.dirname(PAPER_RESULTS_DIR)
sys.path.insert(0, OPENDROP_DIR)
sys.path.insert(0, PAPER_RESULTS_DIR)

from config import (
    ISAAC_DIR,
    RESULTS_DIR,
    ML_MODEL_DIR,
    N_TANGENT_POINTS,
    POLYNOMIAL_DEGREE_TANGENT,
    VIDEO_EXTENSIONS,
)

from opendrop_ml.modules.contact_angle.video_two_pass_needle_in import (
    process_frame_pass1,
    compute_pass1_calibration,
    extract_pass2_frame,
)
from opendrop_ml.modules.fitting.polynomial_fit import polynomial_fit
from opendrop_ml.modules.fitting.circular_fit import circular_fit
from opendrop_ml.modules.fitting.ellipse_fit import ellipse_fit
from opendrop_ml.modules.fitting.BA_fit import yl_fit
from opendrop_ml.modules.ML_model.prepare_experimental import (
    normalize_ds,
    experimental_pred,
)
from opendrop_ml.modules.image.frame_loader import load_frames

import tensorflow as tf


# --- Isaac dataset configurations ---
# Only video-level knobs live here; crop/hough/distance thresholds are
# owned by the app module (``video_two_pass_needle_in``).
ISAAC_DATASETS = {
    "0822_10k_Bb5": {
        "median_step": 10,
        "start_sec": 0,
        "end_sec": 11,
        "frame_step": 1,
    },
    "MS_APTES_S2_0001": {
        "median_step": 10,
        "start_sec": 0,
        "end_sec": -1,
        "frame_step": 1,
    },
    "SC69_1": {
        "median_step": 10,
        "start_sec": 0,
        "end_sec": 42,
        "frame_step": 1,
    },
}


def _safe_error(errors, key, fallback_key=None):
    """Get error metric from dict, handling missing keys and non-dict errors."""
    if not isinstance(errors, dict):
        return np.nan
    val = errors.get(key, np.nan)
    if val is np.nan and fallback_key:
        val = errors.get(fallback_key, np.nan)
    return val


def find_video_file(dataset_dir):
    """Find the video file in a dataset directory (checks raw_data/ too)."""
    for f in os.listdir(dataset_dir):
        if f.lower().endswith(VIDEO_EXTENSIONS):
            return os.path.join(dataset_dir, f)
    raw_dir = os.path.join(dataset_dir, "raw_data")
    if os.path.isdir(raw_dir):
        for f in os.listdir(raw_dir):
            if f.lower().endswith(VIDEO_EXTENSIONS):
                return os.path.join(raw_dir, f)
    return None


def evaluate_frame(frame_data, profile, img, model):
    """Run fitting methods on an extracted profile.

    Args:
        frame_data: dict from ``extract_pass2_frame``
        profile: Nx2 drop profile array
        img: Original BGR frame (for ML model)
        model: TF model

    Returns:
        dict of results
    """
    result = {}

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
    except Exception as e:
        result["yl_error"] = str(e)

    # --- ML model (using pre-extracted contour windows, per notebook) ---
    try:
        ml_left = frame_data["ml_left_window"].astype(np.float64).copy()
        ml_right = frame_data["ml_right_window"].astype(np.float64).copy()
        # Reflect x for left side (notebook: ML_left[:, 0] *= -1)
        ml_left[:, 0] *= -1
        # Take last 50 points (matches reference [-50:] slice).
        ml_left = ml_left[-50:]
        ml_right = ml_right[-50:]
        if len(ml_left) == 50 and len(ml_right) == 50:
            pred_ds = np.array([ml_left, ml_right])
            pred_ds = normalize_ds(pred_ds)
            predictions, ml_timings = experimental_pred(pred_ds, model, side="both")
            result["ml_left_angle"] = float(predictions[0])
            result["ml_right_angle"] = float(predictions[1])
        else:
            result["ml_error"] = "Insufficient ML window points: L={}, R={}".format(
                len(ml_left), len(ml_right)
            )
    except Exception as e:
        result["ml_error"] = str(e)

    return result


def process_dataset(dataset_name, dataset_dir, config, model):
    """Process a single Isaac dataset end-to-end.

    Pass 1 → compute_pass1_calibration → Pass 2 via app functions.
    """
    print()
    print("-" * 60)
    print("Processing dataset: {}".format(dataset_name))
    print("-" * 60)

    video_path = find_video_file(dataset_dir)
    if video_path is None:
        print("  ERROR: No video file found in {}".format(dataset_dir))
        return None

    print("  Video: {}".format(video_path))

    median_step = config["median_step"]
    start_sec = config["start_sec"]
    end_sec = config["end_sec"]
    frame_step = config["frame_step"]

    # -----------------------------------------------------------------
    # PASS 1: sample frames for contact-point / needle calibration
    # -----------------------------------------------------------------
    print()
    print("  PASS 1: Calibration (every {}th frame)...".format(median_step))
    t0 = time.time()

    pass1_results = []
    failures_p1 = 0

    for frame, frame_num, meta in load_frames(
        video_path, start_sec, end_sec, median_step
    ):
        try:
            result = process_frame_pass1(frame)
            if result is not None:
                pass1_results.append(result)
            else:
                failures_p1 += 1
        except Exception as e:
            failures_p1 += 1
            if len(pass1_results) == 0:
                print("    Pass 1 frame {} error: {}".format(frame_num, e))

    print(
        "  Pass 1: {} frames processed, {} failures".format(
            len(pass1_results), failures_p1
        )
    )

    if len(pass1_results) < 3:
        print("  ERROR: Too few successful frames in Pass 1")
        return None

    (
        left_line,
        right_line,
        left_needle_int,
        right_needle_int,
        left_cps,
        right_cps,
        auto_crop,
    ) = compute_pass1_calibration(pass1_results)

    if auto_crop is None:
        print("  ERROR: auto-crop derivation failed; Isaac runs require it.")
        return None
    crop_for_pass2 = auto_crop

    left_median_CP = np.array([np.median(left_cps[:, 0]), np.median(left_cps[:, 1])])
    right_median_CP = np.array([np.median(right_cps[:, 0]), np.median(right_cps[:, 1])])
    print("  Left median CP:  ({:.1f}, {:.1f})".format(*left_median_CP))
    print("  Right median CP: ({:.1f}, {:.1f})".format(*right_median_CP))
    print("  Left median needle intercept:  ({:.1f}, {:.1f})".format(*left_needle_int))
    print("  Right median needle intercept: ({:.1f}, {:.1f})".format(*right_needle_int))
    print("  Auto-crop for Pass 2: {}".format(crop_for_pass2))
    print("  Pass 1 completed in {:.1f}s".format(time.time() - t0))

    # -----------------------------------------------------------------
    # PASS 2: full analysis
    # -----------------------------------------------------------------
    print()
    print("  PASS 2: Full analysis (every {}th frame)...".format(frame_step))
    t0 = time.time()

    results = []
    failures_p2 = 0
    frame_count = 0

    for frame, frame_num, meta in load_frames(
        video_path, start_sec, end_sec, frame_step
    ):
        frame_count += 1
        if frame_count % 50 == 0:
            elapsed = time.time() - t0
            print(
                "    Frame {} processed ({:.0f}s elapsed)...".format(
                    frame_count, elapsed
                )
            )

        row = {
            "dataset": dataset_name,
            "frame_number": frame_num,
        }

        try:
            frame_data = extract_pass2_frame(
                frame,
                crop_for_pass2,
                left_line,
                right_line,
                left_needle_int,
                right_needle_int,
            )

            if frame_data is None:
                row["error"] = "Pass 2 processing failed"
                failures_p2 += 1
                results.append(row)
                continue

            profile = frame_data["profile"]
            row["profile_length"] = len(profile)
            row["left_CP_x"] = float(frame_data["left_CP"][0])
            row["left_CP_y"] = float(frame_data["left_CP"][1])
            row["right_CP_x"] = float(frame_data["right_CP"][0])
            row["right_CP_y"] = float(frame_data["right_CP"][1])
            row["diameter"] = float(
                np.linalg.norm(frame_data["right_CP"] - frame_data["left_CP"])
            )

            fit_results = evaluate_frame(frame_data, profile, frame, model)
            row.update(fit_results)

        except Exception:
            row["error"] = traceback.format_exc()[:200]
            failures_p2 += 1

        results.append(row)

    total_time = time.time() - t0
    print(
        "  Pass 2: {} frames, {} failures, {:.1f}s".format(
            frame_count, failures_p2, total_time
        )
    )

    df = pd.DataFrame(results)
    return df


def main():
    print("=" * 60)
    print("Isaac Dataset Evaluation (Needle-In Video Analysis)")
    print("=" * 60)

    # Suppress TF warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    print("Loading ML model...")
    model = tf.keras.models.load_model(ML_MODEL_DIR)
    print("Model loaded.")

    if not os.path.isdir(ISAAC_DIR):
        print("ERROR: Isaac data directory not found: {}".format(ISAAC_DIR))
        print("Please copy data to paper_results/data/video_analysis/isaac/")
        sys.exit(1)

    all_results = []

    for dataset_name, config in ISAAC_DATASETS.items():
        dataset_dir = os.path.join(ISAAC_DIR, dataset_name)

        if not os.path.isdir(dataset_dir):
            print("  WARNING: Dataset directory not found: {}".format(dataset_dir))
            print("  Skipping {}".format(dataset_name))
            continue

        df = process_dataset(dataset_name, dataset_dir, config, model)
        if df is not None:
            all_results.append(df)

            os.makedirs(RESULTS_DIR, exist_ok=True)
            output_path = os.path.join(
                RESULTS_DIR, "isaac_{}_results.csv".format(dataset_name)
            )
            df.to_csv(output_path, index=False)
            print("  Saved: {}".format(output_path))

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = os.path.join(RESULTS_DIR, "isaac_combined_results.csv")
        combined.to_csv(output_path, index=False)
        print()
        print("Combined results saved to: {}".format(output_path))
        print("Total frames: {}".format(len(combined)))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
