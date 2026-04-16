#!/usr/bin/env python
"""Compare ellipse fit eigenvalue selection: argmax(E) vs argmax(abs(E)).

Runs ellipse_fit on 20 synthetic images (spread across 20°–180°) using both
approaches and compares against ground truth angles.

Run from OpenDrop_OP/:
    python paper_results/evaluate/compare_eigenvalue_selection.py
"""

import sys
import os
import re

import numpy as np

# Ensure imports work regardless of working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_RESULTS_DIR = os.path.dirname(SCRIPT_DIR)
OPENDROP_DIR = os.path.dirname(PAPER_RESULTS_DIR)
sys.path.insert(0, OPENDROP_DIR)
sys.path.insert(0, PAPER_RESULTS_DIR)

import cv2
from config import SYNTHETIC_IMAGES_DIR

from opendrop_ml.modules.preprocessing.preprocessing import isolate_contour, CPID
from opendrop_ml.modules.fitting.ellipse_fit import (
    ellipse_fit,
    fit_ellipse,
    normalize_data,
    denormalize_ellipse,
)


def parse_gt_angle(filename):
    """Extract ground truth angle from filename."""
    name = os.path.splitext(filename)[0]
    return float(name.split("_")[0])


def fit_ellipse_abs(x, y):
    """fit_ellipse but using argmax(abs(E)) — the OLD approach."""
    x, y, x_center, y_center, x_scale, y_scale = normalize_data(x, y)
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = +2
    C[2, 0] = +2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))  # <-- OLD approach
    avec = V[:, n]
    return denormalize_ellipse(avec, x_center, y_center, x_scale, y_scale)


def extract_profile(contour, CPs):
    """Extract drop profile between contact points."""
    idx_left = np.where((contour == CPs[0]).all(axis=1))[0][0]
    idx_right = np.where((contour == CPs[1]).all(axis=1))[0][0]
    if idx_left > idx_right:
        idx_left, idx_right = idx_right, idx_left
    return contour[idx_left : idx_right + 1]


def select_images(image_dir, n_per_angle=2):
    """Select ~2 images per angle bucket, varied Bond/roughness."""
    all_files = sorted(
        f for f in os.listdir(image_dir)
        if f.endswith(".png")
    )

    # Group by angle
    by_angle = {}
    for f in all_files:
        angle = parse_gt_angle(f)
        by_angle.setdefault(angle, []).append(f)

    selected = []
    for angle in sorted(by_angle.keys()):
        files = by_angle[angle]
        # Pick first n_per_angle with different parameters
        # Prefer scale=1.0, varied Bond and roughness
        scale1 = [f for f in files if "_1.0_" in f.split("CPL")[0]]
        chosen = scale1[:n_per_angle] if len(scale1) >= n_per_angle else files[:n_per_angle]
        selected.extend(chosen)

    # Also add 2 scale=4.0 images for variety
    scale4 = [f for f in all_files if "_4.0_" in f.split("CPL")[0]]
    for angle in [80.0, 140.0]:
        for f in scale4:
            if f.startswith(f"{angle}_"):
                selected.append(f)
                break

    return selected


def run_comparison(image_dir, selected_files):
    """Run both eigenvalue approaches on selected images."""
    import opendrop_ml.modules.fitting.ellipse_fit as ell_module

    results = []

    for fname in selected_files:
        img_path = os.path.join(image_dir, fname)
        gt_angle = parse_gt_angle(fname)

        try:
            img = cv2.imread(img_path)
            img_processed, contour, drop_guess, info = isolate_contour(
                img, DISPLAY=False, blur=True
            )
            CPs = CPID(contour, display=False)
            profile = extract_profile(contour, CPs)
        except Exception as e:
            print(f"  SKIP {fname}: preprocessing failed: {e}")
            continue

        # --- Current approach: argmax(E) ---
        try:
            angles_new, _, _, _, _, errors_new, _ = ellipse_fit(profile, display=False)
            left_new = angles_new[0] if angles_new else np.nan
            right_new = angles_new[1] if angles_new else np.nan
        except Exception as e:
            left_new, right_new = np.nan, np.nan

        # --- Old approach: argmax(abs(E)) ---
        # Temporarily monkey-patch fit_ellipse
        original_fit = ell_module.fit_ellipse
        ell_module.fit_ellipse = fit_ellipse_abs
        try:
            angles_old, _, _, _, _, errors_old, _ = ellipse_fit(profile, display=False)
            left_old = angles_old[0] if angles_old else np.nan
            right_old = angles_old[1] if angles_old else np.nan
        except Exception as e:
            left_old, right_old = np.nan, np.nan
        finally:
            ell_module.fit_ellipse = original_fit

        avg_new = (left_new + right_new) / 2
        avg_old = (left_old + right_old) / 2
        err_new = abs(avg_new - gt_angle)
        err_old = abs(avg_old - gt_angle)

        results.append({
            "image": fname,
            "gt": gt_angle,
            "new_L": left_new,
            "new_R": right_new,
            "new_avg": avg_new,
            "new_err": err_new,
            "old_L": left_old,
            "old_R": right_old,
            "old_avg": avg_old,
            "old_err": err_old,
            "same": np.isclose(avg_new, avg_old, atol=0.01),
        })

    return results


def print_results(results):
    """Print comparison table and summary statistics."""
    print()
    print(f"{'GT':>6}  {'argmax(E)':>12}  {'argmax(|E|)':>12}  {'err_new':>8}  {'err_old':>8}  {'better':>7}")
    print("-" * 70)

    n_better_new = 0
    n_better_old = 0
    n_same = 0
    errs_new = []
    errs_old = []

    for r in results:
        if np.isnan(r["new_err"]) or np.isnan(r["old_err"]):
            tag = "N/A"
        elif r["same"]:
            tag = "same"
            n_same += 1
        elif r["new_err"] < r["old_err"]:
            tag = "NEW"
            n_better_new += 1
        else:
            tag = "OLD"
            n_better_old += 1

        errs_new.append(r["new_err"])
        errs_old.append(r["old_err"])

        print(
            f"{r['gt']:6.1f}  "
            f"{r['new_avg']:12.2f}  "
            f"{r['old_avg']:12.2f}  "
            f"{r['new_err']:8.2f}  "
            f"{r['old_err']:8.2f}  "
            f"{tag:>7}"
        )

    errs_new = np.array(errs_new)
    errs_old = np.array(errs_old)
    valid = ~np.isnan(errs_new) & ~np.isnan(errs_old)

    print("-" * 70)
    print(f"MAE  argmax(E):    {np.nanmean(errs_new[valid]):.3f}°")
    print(f"MAE  argmax(|E|):  {np.nanmean(errs_old[valid]):.3f}°")
    print(f"Max  argmax(E):    {np.nanmax(errs_new[valid]):.3f}°")
    print(f"Max  argmax(|E|):  {np.nanmax(errs_old[valid]):.3f}°")
    print(f"\nargmax(E) better: {n_better_new}  |  argmax(|E|) better: {n_better_old}  |  same: {n_same}")
    print(f"Total images: {len(results)} ({valid.sum()} with valid results)")


if __name__ == "__main__":
    print("Selecting 20 synthetic images across angle range...")
    selected = select_images(SYNTHETIC_IMAGES_DIR, n_per_angle=2)
    print(f"Selected {len(selected)} images")

    print("\nRunning ellipse fit with both eigenvalue selection methods...")
    results = run_comparison(SYNTHETIC_IMAGES_DIR, selected)

    print_results(results)
