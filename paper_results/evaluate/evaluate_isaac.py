#!/usr/bin/env python
"""Two-pass needle-in video analysis for Isaac datasets.

Implements the two-pass approach from needle_in_clean.ipynb:
  Pass 1: Sample frames to determine median contact points & needle intercepts
  Pass 2: Use median contact lines to extract precise drop profiles for all frames

Each Isaac dataset is an AVI video with a needle-in-drop configuration.

Run from OpenDrop_OP/:
    python paper_results/evaluate/evaluate_isaac.py
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
    ISAAC_DIR,
    RESULTS_DIR,
    ML_MODEL_DIR,
    N_TANGENT_POINTS,
    POLYNOMIAL_DEGREE_TANGENT,
    VIDEO_EXTENSIONS,
)

from opendrop_ml.modules.preprocessing.preprocessing import (
    CPID,
    cluster_OPTICS,
    optimized_path,
    linear_fit,
)
from opendrop_ml.modules.ML_model.prepare_experimental import (
    prepare_for_model,
    normalize_ds,
    experimental_pred,
)
from opendrop_ml.modules.fitting.polynomial_fit import polynomial_fit
from opendrop_ml.modules.fitting.circular_fit import circular_fit
from opendrop_ml.modules.fitting.ellipse_fit import ellipse_fit
from opendrop_ml.modules.fitting.BA_fit import yl_fit

from evaluate.frame_loader import load_frames, get_middle_frame
from evaluate.needle_masking import (
    extract_edges_two_contours,
    hough_baseline,
    detect_vertical_lines,
    remove_points_near_lines,
    find_first_intercept,
    find_last_intercept,
)

import tensorflow as tf


# --- Isaac dataset configurations ---
ISAAC_DATASETS = {
    "0822_10k_Bb5": {
        "crop_value": 320,
        "median_step": 10,
        "start_sec": 0,
        "end_sec": 11,
        "frame_step": 1,
        "hough_threshold": 40,
        "distance_threshold": 5.0,
    },
    "MS_APTES_S2_0001": {
        "crop_value": 320,
        "median_step": 10,
        "start_sec": 0,
        "end_sec": -1,
        "frame_step": 1,
        "hough_threshold": 40,
        "distance_threshold": 5.0,
    },
    "SC69_1": {
        "crop_value": 320,
        "median_step": 10,
        "start_sec": 0,
        "end_sec": 42,
        "frame_step": 1,
        "hough_threshold": 40,
        "distance_threshold": 5.0,
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
    # Check raw_data subdirectory
    raw_dir = os.path.join(dataset_dir, "raw_data")
    if os.path.isdir(raw_dir):
        for f in os.listdir(raw_dir):
            if f.lower().endswith(VIDEO_EXTENSIONS):
                return os.path.join(raw_dir, f)
    return None


def _largest_cluster(groups):
    """Extract the largest cluster from a cluster_OPTICS dict return."""
    if isinstance(groups, np.ndarray):
        return groups
    keys = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)
    return np.array(groups[keys[0]])


def _rightmost_cluster(groups):
    """Extract the right-most cluster (highest max x). Used for left contour."""
    if isinstance(groups, np.ndarray):
        return groups
    key = max(groups.keys(), key=lambda k: max(groups[k][:, 0]))
    return np.array(groups[key])


def _leftmost_cluster(groups):
    """Extract the left-most cluster (lowest max x). Used for right contour."""
    if isinstance(groups, np.ndarray):
        return groups
    key = min(groups.keys(), key=lambda k: max(groups[k][:, 0]))
    return np.array(groups[key])


def _create_edge_image(left, right, shape):
    """Create binary edge image from contour points."""
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


def process_frame_pass1(frame, crop_value, hough_threshold=40,
                        distance_threshold=5.0):
    """Process a single frame for Pass 1 (contact point calibration).

    Matches the reference notebook Cell 14 pipeline:
    1. extract_edges_two_contours (with border filtering)
    2. cluster_OPTICS, assign left/right by topmost point x
    3. Hough baseline + detect_vertical_lines
    4. Adaptive distance threshold for point removal (5..10)
    5. Extra baseline clearing (distance + 30)
    6. Re-cluster: right-most for left, left-most for right
    7. Find needle intercept (topmost point of isolated contour)
    8. optimized_path on FULL contours
    9. CPID on [left[needle_idx:], right[needle_idx:]]
       (full contour from needle to bottom, INCLUDING surface region)

    Returns:
        dict with contact points, contours, and needle intercepts,
        or None if processing fails.
    """
    h, w = frame.shape[:2]
    cropped = frame[:crop_value, :]

    try:
        contour1, contour2, thresh = extract_edges_two_contours(cropped)
    except ValueError:
        return None

    # Cluster each contour with OPTICS to clean noise
    try:
        contour1 = _largest_cluster(cluster_OPTICS(contour1, xi=0.8))
    except Exception:
        pass
    try:
        contour2 = _largest_cluster(cluster_OPTICS(contour2, xi=0.8))
    except Exception:
        pass

    if len(contour1) < 10 or len(contour2) < 10:
        return None

    # Assign left/right by topmost-point x-coordinate
    top1_x = contour1[np.argmin(contour1[:, 1]), 0]
    top2_x = contour2[np.argmin(contour2[:, 1]), 0]

    if top1_x < top2_x:
        left, right = contour1, contour2
    else:
        left, right = contour2, contour1

    # Create edge image for Hough detection
    edge_img = _create_edge_image(left, right, (crop_value, w))

    # Detect baseline
    try:
        baseline = hough_baseline(edge_img, threshold=hough_threshold)
    except Exception:
        baseline = np.array([[0, crop_value], [w, crop_value]])

    # Detect needle lines
    left_fiber, right_fiber, _, _ = detect_vertical_lines(
        edge_img, hough_threshold=hough_threshold
    )

    if left_fiber is None or right_fiber is None:
        return None

    # Ensure left_fiber is actually on the left
    if left_fiber[np.argmin(left_fiber[:, 1])][0] > right_fiber[np.argmin(right_fiber[:, 1])][0]:
        left_fiber, right_fiber = right_fiber, left_fiber

    # Adaptive distance threshold: increase until the highest point of
    # the cleaned contour is below 20% of the image height
    lines_all = [baseline, left_fiber, right_fiber]

    # Left contour
    dt = distance_threshold
    left_isolated = None
    while dt <= 10:
        candidate = remove_points_near_lines(left, lines_all, dt)
        if len(candidate) > 0 and min(candidate[:, 1]) > crop_value * 0.2:
            left_isolated = candidate
            break
        dt += 1
    if left_isolated is None:
        left_isolated = remove_points_near_lines(left, lines_all, 10)
        dt = 10
    # Extra baseline clearing (dt+30, matching notebook Cell 14)
    if baseline is not None:
        left_isolated = remove_points_near_lines(
            left_isolated, [baseline], dt + 30
        )

    # Right contour
    dt = distance_threshold
    right_isolated = None
    while dt <= 10:
        candidate = remove_points_near_lines(right, lines_all, dt)
        if len(candidate) > 0 and min(candidate[:, 1]) > crop_value * 0.2:
            right_isolated = candidate
            break
        dt += 1
    if right_isolated is None:
        right_isolated = remove_points_near_lines(right, lines_all, 10)
        dt = 10
    # Extra baseline clearing (dt+30, matching notebook Cell 14)
    if baseline is not None:
        right_isolated = remove_points_near_lines(
            right_isolated, [baseline], dt + 30
        )

    if len(left_isolated) < 10 or len(right_isolated) < 10:
        return None

    # Re-cluster isolated contours
    # Left contour: take right-most cluster (closest to drop center)
    # Right contour: take left-most cluster (closest to drop center)
    try:
        left_clusters = cluster_OPTICS(left_isolated, eps=2)
        left_isolated = _rightmost_cluster(left_clusters)
    except Exception:
        pass

    try:
        right_clusters = cluster_OPTICS(right_isolated, eps=2)
        right_isolated = _leftmost_cluster(right_clusters)
    except Exception:
        pass

    if len(left_isolated) < 10 or len(right_isolated) < 10:
        return None

    # Needle intercept: topmost point of each isolated contour
    left_isolated_start = left_isolated[np.argmin(left_isolated[:, 1])]
    right_isolated_start = right_isolated[np.argmin(right_isolated[:, 1])]

    # Sort + optimized_path on the FULL contours (not isolated).
    # This produces a continuous top-to-bottom ordering that CPID expects.
    left_sorted = sorted(left.tolist(), key=lambda x: (x[1], -x[0]))
    left = optimized_path(left_sorted)
    right_sorted = sorted(right.tolist(), key=lambda x: (x[1], x[0]))
    right = optimized_path(right_sorted)

    # Find needle intercept index in the full ordered contour
    left_start_matches = np.where(
        (left == left_isolated_start).all(axis=1)
    )[0]
    right_start_matches = np.where(
        (right == right_isolated_start).all(axis=1)
    )[0]

    if len(left_start_matches) == 0 or len(right_start_matches) == 0:
        return None

    idx_left_start = left_start_matches[0]
    idx_right_start = right_start_matches[0]

    # CPID on full contour from needle intercept to bottom (includes surface).
    # This is critical: CPID needs the surface region to detect the
    # drop-to-surface transition via iterative_Lfit.
    left_for_cpid = left[idx_left_start:]
    right_for_cpid = right[idx_right_start:]

    if len(left_for_cpid) < 10 or len(right_for_cpid) < 10:
        return None

    try:
        CPs = CPID([left_for_cpid, right_for_cpid], display=False)
    except Exception:
        return None

    if CPs is None or len(CPs) < 2:
        return None

    return {
        "left_contour": left,
        "right_contour": right,
        "left_isolated": left_isolated,
        "right_isolated": right_isolated,
        "left_CP": CPs[0],
        "right_CP": CPs[1],
        "left_needle_intercept": left_isolated_start,
        "right_needle_intercept": right_isolated_start,
        "baseline": baseline,
    }


def compute_median_contact_lines(pass1_results):
    """Compute median contact points and fit contact lines from Pass 1 results.

    Args:
        pass1_results: dict of {frame_number: pass1_result_dict}

    Returns:
        (left_contact_line_path, right_contact_line_path,
         left_median_CP, right_median_CP,
         left_median_int, right_median_int)
    """
    left_CPs = np.array([r["left_CP"] for r in pass1_results.values()])
    right_CPs = np.array([r["right_CP"] for r in pass1_results.values()])
    left_ints = np.array([r["left_needle_intercept"] for r in pass1_results.values()])
    right_ints = np.array([r["right_needle_intercept"] for r in pass1_results.values()])

    # Median values
    left_median_CP = np.array([
        np.median(left_CPs[:, 0]), np.median(left_CPs[:, 1])
    ])
    right_median_CP = np.array([
        np.median(right_CPs[:, 0]), np.median(right_CPs[:, 1])
    ])
    left_median_int = np.array([
        np.median(left_ints[:, 0]), np.median(left_ints[:, 1])
    ])
    right_median_int = np.array([
        np.median(right_ints[:, 0]), np.median(right_ints[:, 1])
    ])

    # Fit linear contact lines through CPs with IQR filtering
    left_line = _fit_contact_line(left_CPs)
    right_line = _fit_contact_line(right_CPs)

    return (left_line, right_line,
            left_median_CP, right_median_CP,
            left_median_int, right_median_int)


def _fit_contact_line(CPs):
    """Fit a linear contact line through contact points with IQR filtering."""
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


def process_frame_pass2(frame, crop_value, left_contact_line, right_contact_line,
                        left_median_int, right_median_int,
                        left_median_CP, right_median_CP,
                        hough_threshold=40, distance_threshold=5.0):
    """Process a single frame for Pass 2 (full analysis).

    Uses median contact lines from Pass 1 to find precise CP and needle
    intercept indices. Follows the notebook's Pass 2 flow.

    Returns:
        dict with 'profile', 'left_drop', 'right_drop',
        'left_CP', 'right_CP', or None if processing fails.
    """
    h, w = frame.shape[:2]
    cropped = frame[:crop_value, :]

    try:
        contour1, contour2, thresh = extract_edges_two_contours(cropped)
    except ValueError:
        return None

    # Cluster and take largest
    try:
        contour1 = _largest_cluster(cluster_OPTICS(contour1, xi=0.8))
    except Exception:
        pass
    try:
        contour2 = _largest_cluster(cluster_OPTICS(contour2, xi=0.8))
    except Exception:
        pass

    if len(contour1) < 10 or len(contour2) < 10:
        return None

    # Assign left/right by topmost-point x-coordinate
    top1_x = contour1[np.argmin(contour1[:, 1]), 0]
    top2_x = contour2[np.argmin(contour2[:, 1]), 0]

    if top1_x < top2_x:
        left, right = contour1, contour2
    else:
        left, right = contour2, contour1

    # Sort and order contours continuously top to bottom
    left_sorted = sorted(left.tolist(), key=lambda x: (x[1], -x[0]))
    left = optimized_path(left_sorted)
    right_sorted = sorted(right.tolist(), key=lambda x: (x[1], x[0]))
    right = optimized_path(right_sorted)

    # Find CP index using median contact line
    left_CP_idx = find_first_intercept(left, left_contact_line)
    right_CP_idx = find_first_intercept(right, right_contact_line)

    if left_CP_idx < 0 or right_CP_idx < 0:
        return None

    # Find needle intercept using median needle position (vertical line)
    # Use max_y limit: halfway between median needle intercept and median CP
    approx_mid_y = max(left_median_int[1], right_median_int[1]) + (
        (max(left_median_CP[1], right_median_CP[1])
         - max(left_median_int[1], right_median_int[1])) / 2
    )

    left_needle_line = [
        [left_median_int[0], 0],
        [left_median_int[0], crop_value],
    ]
    right_needle_line = [
        [right_median_int[0], 0],
        [right_median_int[0], crop_value],
    ]

    left_start_idx = find_last_intercept(left, left_needle_line, max_y=approx_mid_y)
    right_start_idx = find_last_intercept(right, right_needle_line, max_y=approx_mid_y)

    # Fallback: if one fails but the other succeeds, use the other
    if left_start_idx < 0 and right_start_idx >= 0:
        left_start_idx = right_start_idx
    if right_start_idx < 0 and left_start_idx >= 0:
        right_start_idx = left_start_idx

    # Last fallback: use first point
    if left_start_idx < 0:
        left_start_idx = 0
    if right_start_idx < 0:
        right_start_idx = 0

    # Ensure valid index ordering
    if left_start_idx >= left_CP_idx:
        return None
    if right_start_idx >= right_CP_idx:
        return None

    # Extract isolated drop contours
    left_drop = left[left_start_idx:left_CP_idx + 1]
    right_drop = right[right_start_idx:right_CP_idx + 1]

    if len(left_drop) < 5 or len(right_drop) < 5:
        return None

    # Combine for profile (CP to CP)
    left_drop_reversed = left_drop[::-1]
    profile = np.vstack([left_drop_reversed, right_drop])

    # Extract ML input windows (points around CP)
    # Notebook: left[CP-100:CP+10] and right[CP-100:CP+10]
    # Contours go top-to-bottom, so points before CP index are on the drop,
    # points after are on the surface. Last 50 = 40 drop + 10 surface.
    ml_left_start = max(0, left_CP_idx - 100)
    ml_left_end = min(len(left), left_CP_idx + 10)
    ml_left = left[ml_left_start:ml_left_end]

    ml_right_start = max(0, right_CP_idx - 100)
    ml_right_end = min(len(right), right_CP_idx + 10)
    ml_right = right[ml_right_start:ml_right_end]

    return {
        "profile": profile,
        "left_drop": left_drop,
        "right_drop": right_drop,
        "ml_left_window": ml_left,
        "ml_right_window": ml_right,
        "left_CP": left[left_CP_idx],
        "right_CP": right[right_CP_idx],
    }


def evaluate_frame(frame_data, profile, img, model):
    """Run fitting methods on an extracted profile.

    Args:
        frame_data: dict from process_frame_pass2
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
        ml_left = frame_data["ml_left_window"].copy()
        ml_right = frame_data["ml_right_window"].copy()
        # Reflect x for left side (notebook: ML_left[:, 0] *= -1)
        ml_left[:, 0] *= -1
        # Take last 50 points
        ml_left = ml_left[-50:]
        ml_right = ml_right[-50:]
        if len(ml_left) == 50 and len(ml_right) == 50:
            pred_ds = np.array([ml_left, ml_right])
            pred_ds = normalize_ds(pred_ds)
            predictions, ml_timings = experimental_pred(
                pred_ds, model, side="both"
            )
            result["ml_left_angle"] = float(predictions[0])
            result["ml_right_angle"] = float(predictions[1])
        else:
            result["ml_error"] = "Insufficient ML window points: L={}, R={}".format(
                len(ml_left), len(ml_right))
    except Exception as e:
        result["ml_error"] = str(e)

    return result


def process_dataset(dataset_name, dataset_dir, config, model):
    """Process a single Isaac dataset end-to-end.

    Args:
        dataset_name: Name of the dataset (e.g., 'MS_APTES_S2_0001')
        dataset_dir: Path to dataset directory containing the video
        config: dict of processing parameters
        model: TF model

    Returns:
        DataFrame of results
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

    crop_value = config["crop_value"]
    median_step = config["median_step"]
    start_sec = config["start_sec"]
    end_sec = config["end_sec"]
    frame_step = config["frame_step"]
    hough_threshold = config["hough_threshold"]
    distance_threshold = config["distance_threshold"]

    # ===================================================================
    # PASS 1: Contact Point & Needle Intercept Calibration
    # ===================================================================
    print()
    print("  PASS 1: Calibration (every {}th frame)...".format(median_step))
    t0 = time.time()

    pass1_results = {}
    failures_p1 = 0

    for frame, frame_num, meta in load_frames(
        video_path, start_sec, end_sec, median_step
    ):
        try:
            result = process_frame_pass1(
                frame, crop_value, hough_threshold, distance_threshold
            )
            if result is not None:
                pass1_results[frame_num] = result
            else:
                failures_p1 += 1
        except Exception as e:
            failures_p1 += 1
            if len(pass1_results) == 0:
                # Print first failure for diagnostics
                print("    Pass 1 frame {} error: {}".format(frame_num, e))

    print("  Pass 1: {} frames processed, {} failures".format(
        len(pass1_results), failures_p1))

    if len(pass1_results) < 3:
        print("  ERROR: Too few successful frames in Pass 1")
        return None

    # Compute median contact lines
    (left_contact_line, right_contact_line,
     left_median_CP, right_median_CP,
     left_median_int, right_median_int) = compute_median_contact_lines(pass1_results)

    print("  Left median CP:  ({:.1f}, {:.1f})".format(*left_median_CP))
    print("  Right median CP: ({:.1f}, {:.1f})".format(*right_median_CP))
    print("  Left median needle intercept:  ({:.1f}, {:.1f})".format(*left_median_int))
    print("  Right median needle intercept: ({:.1f}, {:.1f})".format(*right_median_int))
    print("  Pass 1 completed in {:.1f}s".format(time.time() - t0))

    # ===================================================================
    # PASS 2: Full Analysis
    # ===================================================================
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
            print("    Frame {} processed ({:.0f}s elapsed)...".format(
                frame_count, elapsed))

        row = {
            "dataset": dataset_name,
            "frame_number": frame_num,
        }

        try:
            frame_data = process_frame_pass2(
                frame, crop_value,
                left_contact_line, right_contact_line,
                left_median_int, right_median_int,
                left_median_CP, right_median_CP,
                hough_threshold, distance_threshold,
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
            row["diameter"] = float(np.linalg.norm(
                frame_data["right_CP"] - frame_data["left_CP"]
            ))

            # Run fitting methods
            fit_results = evaluate_frame(frame_data, profile, frame, model)
            row.update(fit_results)

        except Exception:
            row["error"] = traceback.format_exc()[:200]
            failures_p2 += 1

        results.append(row)

    total_time = time.time() - t0
    print("  Pass 2: {} frames, {} failures, {:.1f}s".format(
        frame_count, failures_p2, total_time))

    df = pd.DataFrame(results)
    return df


def main():
    print("=" * 60)
    print("Isaac Dataset Evaluation (Needle-In Video Analysis)")
    print("=" * 60)

    # Suppress TF warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load model
    print("Loading ML model...")
    model = tf.keras.models.load_model(ML_MODEL_DIR)
    print("Model loaded.")

    if not os.path.isdir(ISAAC_DIR):
        print("ERROR: Isaac data directory not found: {}".format(ISAAC_DIR))
        print("Please copy data to paper_results/data/video_analysis/isaac/")
        sys.exit(1)

    # Process each dataset
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

            # Save per-dataset results
            os.makedirs(RESULTS_DIR, exist_ok=True)
            output_path = os.path.join(
                RESULTS_DIR, "isaac_{}_results.csv".format(dataset_name)
            )
            df.to_csv(output_path, index=False)
            print("  Saved: {}".format(output_path))

    # Save combined results
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
