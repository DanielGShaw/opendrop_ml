#!/usr/bin/env python
# coding=utf-8

from opendrop_ml.modules.core.classes import ExperimentalDrop, ExperimentalSetup
from opendrop_ml.modules.image.select_regions_geom import (
    initial_baseline_seed,
    longest_contour_arc_above_baseline,
    points_above_baseline,
    rotate_baseline,
)
from opendrop_ml.modules.preprocessing.preprocessing import (
    cluster_OPTICS,
    preprocess,
)
from opendrop_ml.utils.geometry import Rect2
from opendrop_ml.utils.enums import FittingMethod, RegionSelect, ThresholdSelect

# from opendrop_ml.utils.keymap import *

from datetime import datetime
from typing import List, Tuple
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import ctypes
import platform
import warnings
if platform.system() == "Windows":
    try:
        import win32gui
        import win32con
    except ImportError:
        print("pywin32 not available. Close button will not be disabled.")


# from scipy import optimize  # DS 7/6/21 - for least squares fit
# import tensorflow as tf  # DS 9/6/21 - for loading ML model


# import os

MAX_IMAGE_TO_SCREEN_RATIO = 0.8


def get_ift_regions(
    img: np.ndarray,
    padding: int = 5,
    area_thresh: int = 1,
    drop_frac: float = 1,
    scharr_block: int = 5,
    canny1: float = 80,
    canny2: float = 160,
) -> Tuple[np.ndarray, tuple]:
    """
    Automatically detects the drop and needle regions in an image.
    """

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    # Applying 7x7 Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 1) rough mask via inverted Otsu + largest CC
    threshold = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    analysis = cv2.connectedComponentsWithStats(threshold, 8)
    (totalLabels, label_ids, stats, centroids) = analysis
    if totalLabels <= 1:
        print("drops found.")
        return original, None

    # Filter for components entering from the top and find the largest among them
    top_entering_blobs_indices = []
    for i in range(1, totalLabels):  # Iterate through all components (0 is background)
        if stats[i, cv2.CC_STAT_TOP] == 0:
            top_entering_blobs_indices.append(i)

    if not top_entering_blobs_indices:
        print("No blobs found entering from the top of the frame.")
        return original, None

    # Select the largest blob among those entering from the top
    blob = -1
    max_area_found = -1
    for idx in top_entering_blobs_indices:
        area = stats[idx, cv2.CC_STAT_AREA]
        if area > max_area_found:
            max_area_found = area
            blob = idx

    if blob == -1:  # Should not happen if top_entering_blobs_indices was populated
        print("Error selecting largest top-entering blob.")
        return original, None

    if stats[blob, cv2.CC_STAT_AREA] < area_thresh:
        print(
            f"Largest top-entering blob (ID: {blob}, Area: {stats[blob, cv2.CC_STAT_AREA]}) too small (threshold: {area_thresh})."
        )
        return original, None

    x, y, bw, bh, _ = stats[blob]
    # print(f"Blob ID: {blob}, Area: {stats[blob, cv2.CC_STAT_AREA]}, Bounding box: (x= {x}, y= {y}) - (w= {bw}, h= {bh})")
    roi = gray[y: y + bh, x: x + bw]
    roi_mask = (label_ids[y: y + bh, x: x + bw] == blob).astype(np.uint8) * 255

    # 2) Scharr→adaptiveThresh→Canny inside ROI
    blur = cv2.GaussianBlur(roi, (scharr_block, scharr_block), 0)
    dx = cv2.Scharr(blur, cv2.CV_16S, 1, 0)
    dy = cv2.Scharr(blur, cv2.CV_16S, 0, 1)
    grad = dx.astype(float) ** 2 + dy.astype(float) ** 2
    grad = np.uint8(255 * (grad / grad.max()))
    cv2.adaptiveThreshold(
        grad,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=scharr_block,
        C=0,
        dst=grad,
    )
    grad = cv2.bitwise_and(grad, grad, mask=roi_mask)
    edges = cv2.Canny(grad, canny1, canny2)

    # 3) collect edge points *only* in the bottom (1-drop_frac) of ROI:
    pts = np.column_stack(np.nonzero(edges))  # (row, col)
    if len(pts) < 20:
        print("Too few edges.")
        return original, None
    # Calculate width of the blob at each row within the ROI
    blob_widths_at_each_row = np.sum(roi_mask // 255, axis=1)
    # Define the upper part of the blob to estimate needle width
    # Consider top 15% of blob height, but at least 3px and at most 20px
    initial_needle_scan_height = max(3, min(int(bh * 0.15), 500))
    # Ensure scan height is less than blob height for sensible indexing
    if initial_needle_scan_height >= bh and bh > 0:
        initial_needle_scan_height = bh - 1
    elif bh == 0:  # Should not happen if blob found
        initial_needle_scan_height = 0
    avg_needle_width = 0
    if initial_needle_scan_height > 0:
        candidate_needle_widths = blob_widths_at_each_row[:initial_needle_scan_height]
        if np.any(candidate_needle_widths > 0):
            avg_needle_width = np.median(
                candidate_needle_widths[candidate_needle_widths > 0]
            )

    if (
        avg_needle_width <= 1
    ):  # If needle width is too small or not found, use a small portion of top widths
        fallback_scan_height = max(1, int(bh * 0.05))
        if bh > 0 and np.any(blob_widths_at_each_row[:fallback_scan_height] > 0):
            avg_needle_width = np.mean(
                blob_widths_at_each_row[:fallback_scan_height][
                    blob_widths_at_each_row[:fallback_scan_height] > 0
                ]
            )
        if avg_needle_width <= 0:  # Absolute fallback if still no valid width
            avg_needle_width = 1
    expansion_factor = 1  # Drop should be significantly wider than needle
    search_start_row = initial_needle_scan_height

    found_expansion_point = False
    calculated_row_thresh = 0
    for current_row in range(search_start_row, bh):
        # Add small absolute diff
        if (
            blob_widths_at_each_row[current_row] > expansion_factor *
                avg_needle_width
            and blob_widths_at_each_row[current_row] > avg_needle_width + 2
        ):
            calculated_row_thresh = current_row
            found_expansion_point = True
            # print(f"Dynamic row_thresh: Initial needle width ~{avg_needle_width:.1f}px. Expansion at row {current_row} (width {blob_widths_at_each_row[current_row]}px).")
            break

    if not found_expansion_point:
        print(
            f"No clear expansion point found. Initial needle width ~{avg_needle_width:.1f}px. Using drop_frac based fallback: {calculated_row_thresh}."
        )
        pass  # Already defaulted to drop_frac based
    row_thresh = calculated_row_thresh

    # print(f"Row threshold for drop detection: {row_thresh} (calculated)")
    # --- End of dynamic row_thresh determination ---
    def create_bounds_array(x1, y1, x2, y2):
        # Ensure coordinates are integers for array creation if they are not None
        if any(v is None for v in [x1, y1, x2, y2]):
            return None
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        return np.array(
            [
                [[x1, y1, x2, y1]],  # top edge
                [[x2, y1, x2, y2]],  # right edge
                [[x2, y2, x1, y2]],  # bottom edge
                [[x1, y2, x1, y1]],  # left edge
            ],
            dtype=np.int32,
        )

    # Create the bounding box for the entire ROI
    x1_roi = x
    y1_roi = y
    x2_roi = x + bw
    y2_roi = y + bh
    # Create the absolute bounding box for the entire ROI
    abs_bounds = create_bounds_array(x1_roi, y1_roi, x2_roi, y2_roi)
    # Create the cropped bounding box for the entire ROI
    crop_bounds = create_bounds_array(x1_roi, y1_roi, x2_roi, y2_roi)
    crop_roi = original[y1_roi:y2_roi, x1_roi:x2_roi]
    if not found_expansion_point:
        needle_pts = pts[pts[:, 0] < row_thresh]
        pts_xy_needle = needle_pts[:, ::-1]  # (x, y) coordinates
        if pts_xy_needle.shape[0] > 0:  # Ensure there are points to process
            # These are min/max coordinates of the needle points, *relative to the ROI's top-left corner*
            needle_x1_roi = np.min(pts_xy_needle[:, 0])
            needle_x2_roi = np.max(pts_xy_needle[:, 0])
            needle_y1_roi = np.min(pts_xy_needle[:, 1])
            needle_y2_roi = np.max(pts_xy_needle[:, 1])
            needle_abs_bounds = create_bounds_array(
                x + needle_x1_roi - padding,
                y + needle_y1_roi,
                x + needle_x2_roi + padding,
                y + needle_y2_roi - padding,
            )
            needle_crop_bounds = create_bounds_array(
                needle_x1_roi - padding,
                needle_y1_roi,
                needle_x2_roi + padding,
                needle_y2_roi - padding,
            )
            # Crop the original image using the final absolute bounding box coordinates
            crop = original[
                needle_y1_roi:needle_y2_roi, x + needle_x1_roi: x + needle_x2_roi
            ]
            needle_rect = Rect2(
                (x + needle_x1_roi, y + needle_y1_roi),
                (x + needle_x2_roi, y + needle_y2_roi),
            )
            return needle_rect
    else:
        needle_pts = pts[pts[:, 0] < row_thresh]
        pts_xy_needle = needle_pts[:, ::-1]  # (x, y) coordinates
        if pts_xy_needle.shape[0] > 0:  # Ensure there are points to process
            # These are min/max coordinates of the needle points, *relative to the ROI's top-left corner*
            needle_x1_roi = np.min(pts_xy_needle[:, 0]) - padding
            needle_x2_roi = np.max(pts_xy_needle[:, 0]) + padding
            needle_y1_roi = np.min(pts_xy_needle[:, 1])
            needle_y2_roi = np.max(pts_xy_needle[:, 1]) - padding
            needle_abs_bounds = create_bounds_array(
                x + needle_x1_roi,
                y + needle_y1_roi,
                x + needle_x2_roi,
                y + needle_y2_roi,
            )
            needle_crop_bounds = create_bounds_array(
                needle_x1_roi - padding, needle_y1_roi, needle_x2_roi, needle_y2_roi
            )
            needle_rect = Rect2(
                (x + needle_x1_roi, y + needle_y1_roi),
                (x + needle_x2_roi, y + needle_y2_roi),
            )

        drop_pts = pts[pts[:, 0] >= row_thresh]
        pts_xy_drop = drop_pts[:, ::-1]  # (x, y) coordinates
        if pts_xy_drop.shape[0] > 0:  # Ensure there are points to process
            # These are min/max coordinates of the drop points, *relative to the ROI's top-left corner*
            drop_x1_roi = np.min(pts_xy_drop[:, 0]) - padding
            drop_x2_roi = np.max(pts_xy_drop[:, 0]) + padding
            drop_y1_roi = np.min(pts_xy_drop[:, 1])  # - padding
            drop_y2_roi = np.max(pts_xy_drop[:, 1]) + padding
            drop_abs_bounds = create_bounds_array(
                x + drop_x1_roi, y + drop_y1_roi, x + drop_x2_roi, y + drop_y2_roi
            )
            drop_crop_bounds = create_bounds_array(
                drop_x1_roi, drop_y1_roi, drop_x2_roi, drop_y2_roi
            )
            drop_rect = Rect2(
                (x + drop_x1_roi, y + drop_y1_roi), (x + drop_x2_roi, y + drop_y2_roi)
            )

        crop = original[drop_y1_roi:drop_y2_roi,
                        x + drop_x1_roi: x + drop_x2_roi]
        return drop_rect, needle_rect


def set_drop_region(
    experimental_drop: ExperimentalDrop,
    experimental_setup: ExperimentalSetup,
    index: int = 0,
) -> None:
    # select the drop and needle regions in the image
    screen_size = experimental_setup.screen_resolution
    image_size = experimental_drop.image.shape
    scale: float = set_scale(image_size, screen_size)
    screen_position = set_screen_position(screen_size)

    if experimental_setup.drop_id_method == RegionSelect.AUTOMATED:
        from opendrop_ml.modules.preprocessing.preprocessing import isolate_contour

        experimental_drop.cropped_image, _, _, _info = isolate_contour(
            experimental_drop.image
        )
        top, bottom, left, right = _info["bounds"]
        # isolate_contour returns the *full* (tilt-corrected) image as its
        # first value but stashes the actually-computed crop bounds in
        # info["bounds"]. Apply them here so cropped_image is the tight crop
        # downstream code expects — extract_drop_profile then runs Canny on
        # this smaller frame, which is what keeps halo gradients out of the
        # contour on diffuse-light images like image_original (59).png.
        experimental_drop.cropped_image = experimental_drop.cropped_image[
            top:bottom, left:right
        ]
        # print("experimental_drop.cropped_image",experimental_drop.cropped_image is None)
        if experimental_setup.original_boole:  # show found drop

            # Clear all existing figures to avoid conflicts with residual plots
            plt.close("all")
            fig = plt.figure()  # Explicitly create a new figure window
            plt.title(f"Original image {index}")
            plt.imshow(experimental_drop.image)
            plt.show()  # Block execution until the window is manually closed by the user
            plt.close(fig)  # clean up and close the figure after it's shown

        if experimental_setup.cropped_boole:
            # Clear all existing figures to avoid conflicts with residual plots
            plt.close("all")
            fig = plt.figure()  # Explicitly create a new figure window
            plt.title(f"Cropped image {index}")
            plt.imshow(experimental_drop.cropped_image)
            plt.show()
            plt.close(fig)
        experimental_setup.drop_region = [(left, top), (right, bottom)]
    elif experimental_setup.drop_id_method == RegionSelect.USER_SELECTED:
        experimental_setup.drop_region, experimental_drop.cropped_image = (
            user_select_region(
                experimental_drop.image,
                f"Select drop region for Image {index}",
                scale,
                screen_position,
            )
        )


def find_image_edge(img: np.ndarray, low=50, high=150, apertureSize=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high, apertureSize=apertureSize)
    return edges


def user_select_region(
    image: np.ndarray, title: str, scale: float, screen_position
) -> Tuple[Rect2, np.ndarray]:
    """
    Function to manually select a region of interest (ROI) in an image using mouse events.
    User can click and drag to create a rectangle, and press Enter to confirm the selection.
    Returns:
        - A Rect2 object representing the selected region.
        - The cropped image based on the selected region.
    """
    region = user_roi(image, title, scale, screen_position)
    cropped_image = image_crop(image, region)
    return (region, cropped_image)


def set_needle_region(
    experimental_drop: ExperimentalDrop,
    experimental_setup: ExperimentalSetup,
    center_x: float = None,
    index: int = 0,
) -> None:
    screen_size = experimental_setup.screen_resolution
    image_size = experimental_drop.image.shape
    scale = set_scale(image_size, screen_size)
    screen_position = set_screen_position(screen_size)

    if experimental_setup.needle_region_method == RegionSelect.AUTOMATED:
        img = experimental_drop.image

        # find edges in the image
        edges = find_image_edge(img)

        height, width = img.shape[:2]
        cx = center_x if center_x is not None else width // 2

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        vertical_lines = []

        # Collect near-vertical lines in upper-middle region
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                deg = np.rad2deg(theta)

                if deg < 5 or deg > 175:  # very vertical
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))

                    y_min = min(y1, y2)
                    y_max = max(y1, y2)
                    x_avg = (x1 + x2) // 2

                    # Must be near center and top
                    if abs(x_avg - cx) < width * 0.4 and y_min < height * 0.2:
                        vertical_lines.append(
                            {
                                "x": x_avg,
                                "theta": theta,
                                "y_min": y_min,
                                "y_max": y_max,
                                "pts": ((x1, y1), (x2, y2)),
                            }
                        )

        # If exactly 2 vertical lines were found, just use them
        if len(vertical_lines) == 2:
            l1, l2 = vertical_lines[0], vertical_lines[1]
            best_pair = (l1, l2)
        else:
            # Try to find a good pair
            best_pair = None
            best_score = float("inf")

            for i in range(len(vertical_lines)):
                for j in range(i + 1, len(vertical_lines)):
                    l1 = vertical_lines[i]
                    l2 = vertical_lines[j]

                    # Height similarity
                    height_diff = abs(
                        (l1["y_max"] - l1["y_min"]) -
                        (l2["y_max"] - l2["y_min"])
                    )
                    if height_diff > 20:
                        continue

                    # Theta similarity
                    angle_diff = abs(l1["theta"] - l2["theta"])
                    if angle_diff > np.deg2rad(1):  # ~1 degree
                        continue

                    # Prefer taller + closer lines
                    avg_height = (
                        l1["y_max"] - l1["y_min"] + l2["y_max"] - l2["y_min"]
                    ) / 2
                    score = angle_diff + height_diff / 10 - avg_height / 50

                    if score < best_score:
                        best_score = score
                        best_pair = (l1, l2)

        if best_pair:
            x1 = min(best_pair[0]["x"], best_pair[1]["x"])
            x2 = max(best_pair[0]["x"], best_pair[1]["x"])
            y_top = min(best_pair[0]["y_min"], best_pair[1]["y_min"])
            y_bot = max(best_pair[0]["y_max"], best_pair[1]["y_max"])

            width = x2 - x1
            height = y_bot - y_top

            x_padding = int(0.2 * width)
            y_padding = int(0.2 * height)

            x1 = max(x1 - x_padding, 0)
            x2 = x2 + x_padding
            y_top = y_top + y_padding  # subtract padding from top = move it down
            # subtract padding from bottom = move it up
            y_bot = max(y_bot - y_padding, y_top)

            # Optional: draw
            # cv2.rectangle(img, (x1, y_top), (x2, y_bot), (0, 255, 0), 2)

            experimental_setup.needle_region = Rect2((x1, y_top), (x2, y_bot))
            return

        print("error: can't find needle. Please use user-selected method")

    elif experimental_setup.needle_region_method == RegionSelect.USER_SELECTED:
        experimental_setup.needle_region, experimental_drop.cropped_image = (
            user_select_region(
                experimental_drop.image,
                f"Select drop region for Image {index}",
                scale,
                screen_position,
            )
        )


def crop_needle(img: np.ndarray) -> np.ndarray:
    padding = 10
    angle_tolerance = 15
    original_img = img.copy()
    # adjustable parameters
    vertical_pct = 0.8  # fraction of height to keep from top
    horizontal_pct = 0.5  # fraction of width to keep, centered
    # initial crop of the image
    h, w = img.shape[:2]
    top = int(h * vertical_pct)
    side_margin = int((w * (1 - horizontal_pct)) / 2)
    img = img[:top, side_margin: w - side_margin]

    # a veriable used to eliminate the vertical lines that exist in the bottom half of the image
    bottom_verticals_threshold = 0.7
    # 1. Preprocess: grayscale & edges
    # if img.ndim == 3 else img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 180, apertureSize=3)

    # 2. Detect line segments via probabilistic Hough
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=45, minLineLength=1, maxLineGap=50
    )
    if lines is None or len(lines) < 2:
        print("Insufficient line segments detected.")
        return original_img

    # 3. Optimized vertical filter
    dx = lines[:, 0, 2] - lines[:, 0, 0]
    dy = lines[:, 0, 3] - lines[:, 0, 1]
    raw_angles = np.degrees(np.arctan2(dy, dx))  # -180..+180
    # 0 = vertical up, 180 = vertical down
    orient = (raw_angles + 90) % 180
    mask = (orient <= angle_tolerance) | (orient >= 180 - angle_tolerance)
    vertical = lines[mask]
    if vertical.shape[0] < 2:
        print("Could not find two vertical lines within tolerance.")
        return original_img

    # Create a mask for lines where both y-coordinates are in the top half
    # A line (x1,y1,x2,y2) is in the top half if both y1 and y2 are less than y_midpoint

    y_midpoint = abs(h * bottom_verticals_threshold)
    top_half_mask = (vertical[:, 0, 1] < y_midpoint) & (
        vertical[:, 0, 3] < y_midpoint)
    vertical = vertical[top_half_mask]

    if vertical.shape[0] < 2:
        print(
            "Could not find two vertical lines in the top half of the image within tolerance."
        )
        return original_img
    # Extract endpoints
    vertical = vertical[:, 0]  # shape (N,4)

    # min_dist = int(w * min_dist_pct)

    # 4. Select two longest vertical lines

    def length_sq(l):
        return (l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2

    sorted_lines = sorted(vertical, key=length_sq, reverse=True)
    min_dist = 10  # your minimum x-distance in pixels

    # find the longest pair whose centers are >= min_dist apart
    for i, l1 in enumerate(sorted_lines):
        x1 = (l1[0] + l1[2]) // 2
        for l2 in sorted_lines[i + 1:]:
            x2 = (l2[0] + l2[2]) // 2
            if abs(x2 - x1) >= min_dist:
                # we found a valid pair
                break
        else:
            # no valid partner for l1, continue with next
            continue
        # break out of outer loop as well
        break
    else:
        print(f"Could not find two vertical lines ≥ {min_dist}px apart.")
        return original_img
    # 5. Compute bounding coords
    x1 = (l1[0] + l1[2]) // 2
    x2 = (l2[0] + l2[2]) // 2
    left_x, right_x = int(min(x1, x2)), int(max(x1, x2))
    ys = [l1[1], l1[3], l2[1], l2[3]]
    top_y, bottom_y = int(min(ys)), int(max(ys))

    # 6. Pad and clamp
    h, w = gray.shape[:2]
    left = max(0, left_x - padding)
    right = min(w, right_x + padding)
    top = max(0, top_y - padding)
    bottom = min(h, bottom_y + padding)

    # 8. Crop and return
    img = img[top:bottom, left:right]

    return img


def image_crop(image: np.ndarray, points):
    # return image[min(y):max(y), min(x),max(x)]
    return image[
        int(points[0][1]): int(points[1][1]), int(points[0][0]): int(points[1][0])
    ]


def set_surface_line(
    experimental_drop: ExperimentalDrop, experimental_setup: ExperimentalSetup
) -> None:
    # message = []

    #
    # if experimental_drop.cropped_image is None:
    #     if experimental_setup.drop_id_method == "User-selected":
    #         msgbox.showwarning("Warning", "Please select the drop region")
    #         set_drop_region(experimental_drop, experimental_setup)
    #         return
    #     # autuomatic
    #     else:
    #         set_drop_region(experimental_drop, experimental_setup)

    # if experimental_setup.threshold_method == "User-selected":
    #     if experimental_setup.threshold_val is None:
    #         threshold = simpledialog.askinteger("Input Required", "Enter the threshold value:")
    #         if threshold is None:  # User pressed "Cancel"
    #             msgbox.showwarning("Warning", "Threshold is required to continue.")
    #             return
    #         experimental_setup.threshold_val = threshold

    # extract_drop_profile(experimental_drop, experimental_setup)

    if experimental_setup.baseline_method == ThresholdSelect.AUTOMATED:
        img_processed, CPs_chosen, longest_contour = preprocess(
            experimental_drop.image, display=False
        )
        if CPs_chosen is None or len(CPs_chosen) < 2:
            raise ValueError("CPID failed to identify contact points")
        idx_leftCP = np.where((longest_contour == CPs_chosen[0]).all(axis=1))[0][0]
        idx_rightCP = np.where((longest_contour == CPs_chosen[1]).all(axis=1))[0][0]
        if idx_leftCP > idx_rightCP:
            # The raw contour traverses right-to-left between the CPs; reverse the
            # slice so profile[0] is always the left CP and profile[-1] is the right CP.
            experimental_drop.drop_contour = longest_contour[idx_rightCP:idx_leftCP + 1][::-1]
        else:
            experimental_drop.drop_contour = longest_contour[idx_leftCP:idx_rightCP + 1]
        experimental_drop.contact_points = CPs_chosen
    elif experimental_setup.baseline_method == ThresholdSelect.USER_SELECTED:
        user_line(experimental_drop, experimental_setup)


def set_scale(image_size: Tuple[int, int, int], screen_size: List[int]) -> float:
    x_ratio = image_size[1] / float(screen_size[0])
    y_ratio = image_size[0] / float(screen_size[1])
    max_ratio = max(x_ratio, y_ratio)
    scale = 1
    if max_ratio > MAX_IMAGE_TO_SCREEN_RATIO:
        scale = MAX_IMAGE_TO_SCREEN_RATIO / max_ratio
    return scale


def set_screen_position(screen_size: List[int]) -> List[int]:
    # percentage room free
    percent_free_space = 0.5 * (1 - MAX_IMAGE_TO_SCREEN_RATIO)
    x_position = int(percent_free_space * screen_size[0])
    # 0.5 moves window a little bit higher
    y_position = int(0.5 * percent_free_space * screen_size[1])
    return [x_position, y_position]


def disable_close_button(window_title: str):
    if platform.system() == "Windows":
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd:
            style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
            win32gui.SetWindowLong(
                hwnd, win32con.GWL_STYLE, style & ~win32con.WS_SYSMENU)


def user_roi(
    raw_image: np.ndarray, title: str, scale: float, screen_position
) -> List[Tuple[float, float]]:
    """
    Function to select a region of interest (ROI) in an image using mouse events.
    User can click and drag to create a rectangle, and press Enter to confirm the selection.
    """
    global drawing
    global ix, iy
    global fx, fy
    global image_TEMP
    global img
    # raw_image = raw_image2
    # raw_image = np.flipud(cv2.cvtColor(raw_image2,cv2.COLOR_GRAY2BGR))
    # raw_image = np.flipud(raw_image2)
    drawing = False  # true if mouse is pressed
    ix, iy = -1, -1
    fx, fy = -1, -1

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(title, screen_position[0], screen_position[1])
    disable_close_button(title)  # Disable the close button on Windows
    cv2.setMouseCallback(title, draw_rectangle)
    # scale =1
    image_TEMP = cv2.resize(raw_image, (0, 0), fx=scale, fy=scale)

    img = image_TEMP.copy()

    while 1:
        # Add message to the image before showing
        img_display = img.copy()
        msg = "Drag to select a region. Press Enter to confirm."
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        # Position: bottom left, 10px from left, 10px from bottom
        text_size, _ = cv2.getTextSize(msg, font, font_scale, thickness)
        text_x = 10
        text_y = img_display.shape[0] - 10
        cv2.putText(
            img_display,
            msg,
            (text_x, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        cv2.imshow(title, img_display)

        k = cv2.waitKey(1) & 0xFF
        if k != 255:
            if (k == 13) or (k == 32):
                # either 'return' or 'space' pressed
                # break
                if ((fx - ix) * (fy - iy)) != 0:  # ensure there is an enclosed region
                    break
            if k == 27:
                # 'esc'
                kill()

    cv2.destroyAllWindows()
    min_x = min(ix, fx) / scale
    max_x = max(ix, fx) / scale
    min_y = min(iy, fy) / scale
    max_y = max(iy, fy) / scale
    return [(min_x, min_y), (max_x, max_y)]


# Method enum, on-screen label, and contact_angles dict key for each fit
# rendered in user_line. Order is the render order (top → bottom).
_USER_LINE_METHODS = (
    (FittingMethod.TANGENT_FIT, "tangent", "tangent fit"),
    (FittingMethod.POLYNOMIAL_FIT, "polynomial", "polynomial fit"),
    (FittingMethod.CIRCLE_FIT, "circle", "circle fit"),
    (FittingMethod.ELLIPSE_FIT, "ellipse", "ellipse fit"),
)

_USER_LINE_DEBUG_DIR = "/tmp/user_line_debug"


def _user_line_compute_overlays(
    experimental_drop, drop_data, ix, iy, fx, fy, methods_boole
):
    """Filter contour, set drop_contour/contact_points, run fits, gather overlays.

    Returns ``(overlays, failure_msg)``: ``overlays`` is a dict consumed by
    ``_draw_user_line_overlays`` (or ``None`` when no fit ran), ``failure_msg``
    is a string for the bottom-left banner (``None`` on success).
    """
    # Longest contiguous arc above the baseline (filters surface spikes that
    # poke above it) is the actual drop subset; drop[0] and drop[-1] are the
    # implicit contact points where that arc meets the baseline.
    drop = longest_contour_arc_above_baseline(drop_data, ix, iy, fx, fy)

    # Always update drop_contour/contact_points so downstream sees fresh state
    # even when fits are skipped.
    experimental_drop.drop_contour = drop
    if len(drop) > 0:
        experimental_drop.contact_points = {0: drop[0], 1: drop[-1]}

    if len(drop) < 10:
        return None, "no fit available — adjust baseline"

    any_method = any(
        methods_boole[m] for m, _, _ in _USER_LINE_METHODS
    )
    if not any_method:
        return {"angles": []}, None

    from opendrop_ml.modules.fitting.fits import perform_fits

    # Reset before each call so an overlay from a previously-successful method
    # doesn't render after that method fails this iteration.
    experimental_drop.contact_angles = {}
    # Polyfit floods stderr with "RankWarning: Polyfit may be poorly conditioned"
    # when the drop subset is short or near-vertical — every keypress while the
    # user nudges the baseline. The warning is non-actionable for an interactive
    # tool: the fit either succeeds (overlay shows) or raises (banner shows).
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            perform_fits(
                experimental_drop,
                tangent=methods_boole[FittingMethod.TANGENT_FIT],
                polynomial=methods_boole[FittingMethod.POLYNOMIAL_FIT],
                circle=methods_boole[FittingMethod.CIRCLE_FIT],
                ellipse=methods_boole[FittingMethod.ELLIPSE_FIT],
            )
    except Exception as fit_err:
        return None, f"fit failed: {type(fit_err).__name__}"

    if not experimental_drop.contact_angles:
        return None, "no fit available — adjust baseline"

    angles = experimental_drop.contact_angles
    overlays = {"angles": []}

    if methods_boole[FittingMethod.TANGENT_FIT] and "tangent fit" in angles:
        overlays["tangent_lines"] = tuple(angles["tangent fit"]["tangent lines"])
    if (
        methods_boole[FittingMethod.POLYNOMIAL_FIT]
        and "polynomial fit" in angles
        # Polynomial overlay would visually duplicate tangent — show only when
        # tangent isn't drawn this frame.
        and not (methods_boole[FittingMethod.TANGENT_FIT] and "tangent fit" in angles)
    ):
        overlays["polynomial_lines"] = tuple(angles["polynomial fit"]["tangent lines"])
    if methods_boole[FittingMethod.CIRCLE_FIT] and "circle fit" in angles:
        xc, yc = angles["circle fit"]["circle center"]
        r = angles["circle fit"]["circle radius"]
        overlays["circle"] = (int(xc), int(yc), int(r))
    if methods_boole[FittingMethod.ELLIPSE_FIT] and "ellipse fit" in angles:
        center = angles["ellipse fit"]["ellipse center"]
        axes = angles["ellipse fit"]["ellipse a and b"]
        phi = angles["ellipse fit"]["ellipse rotation"]
        overlays["ellipse"] = (
            (int(center[0]), int(center[1])),
            (int(axes[0]), int(axes[1])),
            phi,
        )

    for method_enum, label, key in _USER_LINE_METHODS:
        if not methods_boole[method_enum]:
            continue
        method_angles = angles.get(key)
        if method_angles is None:
            overlays["angles"].append((label, "L --   R --"))
            continue
        l_val = method_angles.get("left angle")
        r_val = method_angles.get("right angle")
        # Hershey stroke font (cv2's only built-in) is ASCII-only — degree
        # sign renders as "?". Use "deg" instead.
        l_str = f"{l_val:.0f}deg" if isinstance(l_val, (int, float)) else "--"
        r_str = f"{r_val:.0f}deg" if isinstance(r_val, (int, float)) else "--"
        overlays["angles"].append((label, f"L {l_str:>6}  R {r_str:>6}"))

    return overlays, None


def _draw_user_line_overlays(img, overlays):
    """Draw fit overlays (tangent/poly lines, circle, ellipse) onto ``img``."""
    if "tangent_lines" in overlays:
        lines = overlays["tangent_lines"]
        cv2.line(
            img,
            (int(lines[0][0][0]), int(lines[0][0][1])),
            (int(lines[0][1][0]), int(lines[0][1][1])),
            (0, 0, 255), 2,
        )
        cv2.line(
            img,
            (int(lines[1][0][0]), int(lines[1][0][1])),
            (int(lines[1][1][0]), int(lines[1][1][1])),
            (0, 0, 255), 2,
        )
    if "polynomial_lines" in overlays:
        lines = overlays["polynomial_lines"]
        cv2.line(img, lines[0][0], lines[0][1], (0, 0, 255), 2)
        cv2.line(img, lines[1][0], lines[1][1], (0, 0, 255), 2)
    if "circle" in overlays:
        xc, yc, r = overlays["circle"]
        cv2.circle(img, (xc, yc), r, (255, 150, 0), 1)
    if "ellipse" in overlays:
        center, axes, phi = overlays["ellipse"]
        cv2.ellipse(img, center, axes, phi, 0, 360, (0, 88, 255), 1)


def _draw_user_line_text_block(img, lines, anchor):
    """Render a list of strings stacked at ``anchor`` ('tl', 'tr', or 'bl')."""
    if not lines:
        return
    canvas_h, canvas_w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    line_h = 16
    if anchor == "tl":
        x0, y0 = 10, 18
    elif anchor == "tr":
        widths = [cv2.getTextSize(s, font, scale, thickness)[0][0] for s in lines]
        x0 = canvas_w - max(widths) - 10
        y0 = 18
    elif anchor == "bl":
        x0 = 10
        y0 = canvas_h - line_h * (len(lines) - 1) - 8
    else:
        return
    for i, s in enumerate(lines):
        cv2.putText(
            img, s, (x0, y0 + i * line_h),
            font, scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )


def _draw_user_line_help(img):
    # Two lines so the help string fits comfortably in narrow cropped canvases.
    # Movement keys on the upper line, meta keys on the lower line.
    canvas_h = img.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        "W/S move  A/D rot 1deg  Q/E rot 0.1deg  O reset",
        (10, canvas_h - 21),
        font, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "V debug  ENTER confirm  ESC cancel",
        (10, canvas_h - 5),
        font, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )


def _draw_user_line_failure(img, msg):
    canvas_h = img.shape[0]
    # canvas_h - 41 sits clear of both help-text lines (canvas_h - 21 and -5).
    cv2.putText(
        img, msg, (10, canvas_h - 41),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
    )


def _draw_user_line_debug_panel(
    img, ix, iy, fx, fy, drop_data, drop_arc, overlays, frame_idx
):
    """Top-left semi-transparent live-state panel toggled by the V key."""
    n_total = len(drop_data)
    n_above = len(points_above_baseline(drop_data, ix, iy, fx, fy))
    n_arc = len(drop_arc) if drop_arc is not None else 0
    dx, dy = fx - ix, fy - iy
    slope_deg = float(np.degrees(np.arctan2(dy, dx))) if (dx or dy) else 0.0
    lines = [
        f"frame {frame_idx}",
        f"baseline ({ix},{iy})-({fx},{fy})  slope {slope_deg:+.1f}deg",
        # `above` is every contour point with a positive cross product;
        # `arc` is the longest contiguous run (== drop_contour). The gap
        # between them is what the spike-tip filter removed.
        f"above {n_above}/{n_total}  arc {n_arc}",
    ]
    if n_arc > 0:
        cp0 = drop_arc[0]
        cp1 = drop_arc[-1]
        lines.append(f"CP0 ({cp0[0]:.0f},{cp0[1]:.0f})  CP1 ({cp1[0]:.0f},{cp1[1]:.0f})")
    if overlays:
        for label, txt in overlays.get("angles", []):
            lines.append(f"{label}: {txt}")
    panel_h = 18 * (len(lines) + 1)
    panel_w = 320
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)
    _draw_user_line_text_block(img, lines, anchor="tl")


def _dump_user_line_state_log(
    experimental_drop, experimental_setup, ix, iy, fx, fy
):
    """On confirm, write a JSON snapshot of the final baseline state."""
    payload = {
        "baseline": [int(ix), int(iy), int(fx), int(fy)],
        "drop_contour_length": (
            int(len(experimental_drop.drop_contour))
            if experimental_drop.drop_contour is not None else 0
        ),
        "contact_points": (
            {
                "0": [float(c) for c in experimental_drop.contact_points[0]],
                "1": [float(c) for c in experimental_drop.contact_points[1]],
            } if experimental_drop.contact_points else None
        ),
        "source_image": getattr(experimental_setup, "filename", None),
        "angles": {},
    }
    for _, _, key in _USER_LINE_METHODS:
        method_data = experimental_drop.contact_angles.get(key)
        if method_data is None:
            continue
        payload["angles"][key] = {
            "left angle": method_data.get("left angle"),
            "right angle": method_data.get("right angle"),
        }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_USER_LINE_DEBUG_DIR, f"session_{timestamp}.json")
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except OSError:
        pass


def user_line(
    experimental_drop: ExperimentalDrop, experimental_setup: ExperimentalSetup
) -> None:
    """Interactive USER_SELECTED baseline definition.

    Manual baseline → manual contact points by intersecting the baseline with
    the drop contour. CPID / Hough are intentionally not called here — the user
    is the contact-point detector, and the live tangent overlay exists as a
    visual sanity check the user judges with their eyes.
    """
    screen_position = set_screen_position(experimental_setup.screen_resolution)
    raw_image = experimental_drop.cropped_image
    drop_data = experimental_drop.contour.astype(float)
    canvas_h, canvas_w = raw_image.shape[:2]
    title = "Define surface line"
    methods_boole = experimental_setup.analysis_methods_ca

    debug_enabled = bool(os.environ.get("OPENDROP_DEBUG_USER_LINE"))
    if debug_enabled:
        os.makedirs(_USER_LINE_DEBUG_DIR, exist_ok=True)

    # Filter the needle out of the contour before any user interaction.
    # AUTOMATED gets this for free via preprocess(); USER_SELECTED was feeding
    # raw extract_edges_cv output (needle still attached) into the loop, so
    # the needle's edge contaminated every fit. cluster_OPTICS is too slow
    # for the per-keypress loop — run it exactly once on entry and keep the
    # largest cluster.
    try:
        groups = cluster_OPTICS(drop_data, xi=0.8)
        if isinstance(groups, dict) and groups:
            largest_label = max(groups.keys(), key=lambda k: len(groups[k]))
            drop_data = np.array(groups[largest_label]).astype(float)
    except Exception as cluster_err:
        print(
            f"cluster_OPTICS failed during USER_SELECTED setup "
            f"({type(cluster_err).__name__}: {cluster_err}); "
            "continuing with the unfiltered contour."
        )

    # Horizontal baseline 1 px below the lowest contour point, clamped inside
    # the canvas. Replaces the polyfit-through-endpoints seed that produced
    # near-vertical baselines on tilted-surface images.
    #
    # State is stored as float so Q/E sub-pixel rotations (0.1°) actually
    # accumulate across keypresses; the cv2 draw calls int-cast at the
    # render boundary.
    _ix0, _iy0, _fx0, _fy0 = initial_baseline_seed(drop_data, canvas_w, canvas_h)
    ix0, iy0, fx0, fy0 = float(_ix0), float(_iy0), float(_fx0), float(_fy0)
    ix, iy, fx, fy = ix0, iy0, fx0, fy0

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 500, 400)
    cv2.moveWindow(title, screen_position[0], screen_position[1])
    try:
        cv2.setWindowProperty(title, 5, 0)  # Lock size where supported
    except Exception:
        pass

    # Sentinel forces first-iteration fit before the user touches a key.
    last_state = None
    cached_overlays = None
    failure_msg = None
    show_debug_panel = False
    frame_idx = 0

    while True:
        state = (ix, iy, fx, fy)
        if state != last_state:
            cached_overlays, failure_msg = _user_line_compute_overlays(
                experimental_drop, drop_data, ix, iy, fx, fy, methods_boole,
            )
            last_state = state

        # Draw onto a fresh raw_image copy each iteration. cv2.resizeWindow
        # handles on-screen scaling, so every draw call uses native
        # cropped_image coordinates — same frame as drop_data and every fit
        # overlay.
        img = raw_image.copy()
        cv2.line(img, (int(ix), int(iy)), (int(fx), int(fy)), (0, 255, 0), 2)
        if cached_overlays is not None:
            # Highlight only the longest-arc subset (== experimental_drop
            # .drop_contour after _user_line_compute_overlays ran). Visible
            # feedback that surface spikes were excluded from the drop.
            arc = experimental_drop.drop_contour
            if arc is not None:
                for cx, cy in arc:
                    cv2.circle(img, (int(cx), int(cy)), 2, (255, 255, 255), 1)
            _draw_user_line_overlays(img, cached_overlays)
            angle_lines = [
                f"{label}: {txt}" for label, txt in cached_overlays.get("angles", [])
            ]
            _draw_user_line_text_block(img, angle_lines, anchor="tr")
        if failure_msg is not None:
            _draw_user_line_failure(img, failure_msg)
        if show_debug_panel:
            _draw_user_line_debug_panel(
                img, ix, iy, fx, fy, drop_data,
                experimental_drop.drop_contour, cached_overlays, frame_idx,
            )
        _draw_user_line_help(img)

        cv2.imshow(title, img)

        if debug_enabled:
            cv2.imwrite(
                os.path.join(_USER_LINE_DEBUG_DIR, f"{frame_idx:04d}.png"), img,
            )
        frame_idx += 1

        # On macOS, clicking the cv2 window's red close button destroys the
        # underlying NSWindow; the next waitKey then crashes with
        # ``Fatal Python error: PyEval_RestoreThread: NULL tstate``. Detect
        # the dead window first and abort gracefully instead.
        try:
            visible = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0
        if visible < 1:
            cv2.destroyAllWindows()
            kill()

        k = cv2.waitKey(1) & 0xFF
        if k == 13 or k == 32:  # ENTER / SPACE
            if (ix, iy) == (fx, fy):
                # Degenerate baseline; surface in the banner instead of breaking
                # downstream silently.
                failure_msg = "baseline has zero length — adjust before confirming"
                continue
            if debug_enabled:
                _dump_user_line_state_log(
                    experimental_drop, experimental_setup, ix, iy, fx, fy,
                )
            break
        elif k == 27:  # ESC
            kill()
        elif k == ord("w"):
            iy -= 1.0
            fy -= 1.0
        elif k == ord("s"):
            iy += 1.0
            fy += 1.0
        elif k == ord("a"):
            ix, iy, fx, fy = rotate_baseline(ix, iy, fx, fy, -1.0)
        elif k == ord("d"):
            ix, iy, fx, fy = rotate_baseline(ix, iy, fx, fy, +1.0)
        elif k == ord("q"):
            ix, iy, fx, fy = rotate_baseline(ix, iy, fx, fy, -0.1)
        elif k == ord("e"):
            ix, iy, fx, fy = rotate_baseline(ix, iy, fx, fy, +0.1)
        elif k == ord("o"):
            ix, iy, fx, fy = ix0, iy0, fx0, fy0
        elif k == ord("v"):
            show_debug_panel = not show_debug_panel

    cv2.destroyAllWindows()


def run_set_surface_line(
    experimental_drop: ExperimentalDrop,
    experimental_setup: ExperimentalSetup,
    result_queue,
):

    set_surface_line(experimental_drop, experimental_setup)
    result_queue.put(experimental_drop.contact_angles)


# mouse callback function


def draw_rectangle(event: int, x, y, flags, param):
    global ix, iy, drawing
    global fx, fy
    global image_TEMP
    global img

    if event == cv2.EVENT_LBUTTONDOWN:
        img = image_TEMP.copy()
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img = image_TEMP.copy()
            # line_colour,line_thickness)
            cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        img = image_TEMP.copy()
        drawing = False
        fx, fy = x, y
        # line_colour,line_thickness)
        cv2.rectangle(img, (ix, iy), (fx, fy), (0, 255, 0), 2)


# mouse callback function


def draw_line(event: int, x, y, flags, param):
    global ix, iy, drawing
    global fx, fy
    global image_TEMP
    global img

    if event == cv2.EVENT_LBUTTONDOWN:
        img = image_TEMP.copy()
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img = image_TEMP.copy()
            # line_colour,line_thickness)
            cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        img = image_TEMP.copy()
        drawing = False
        fx, fy = x, y
        # line_colour,line_thickness)
        cv2.line(img, (ix, iy), (fx, fy), (0, 255, 0), 2)


def kill():
    sys.exit()


def distance(p1, p2):
    """This function computes the distance between 2 points defined by
    P1 = (x1,y1) and P2 = (x2,y2)"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def optimized_path(coords, start=None):
    """This function finds the nearest point to a point
    coords should be a list in this format coords = [ [x1, y1], [x2, y2] , ...]
    https://stackoverflow.com/questions/45829155/sort-points-in-order-to-have-a-continuous-curve-using-python
    """
    if isinstance(coords, list) == False:
        coords = coords.tolist()
    if 0:
        if isinstance(start, list) == False:
            try:
                start = start.tolist()
            except:
                start = list(start)
    if start is None:
        start = coords[0]
    pass_by = coords
    path = [start]
    pass_by.remove(start)
    while pass_by:
        nearest = min(pass_by, key=lambda x: distance(path[-1], x))
        path.append(nearest)
        pass_by.remove(nearest)
    path = np.array(path)
    return path


def intersection(center, radius, p1, p2):
    """find the two points where a secant intersects a circle"""

    dx, dy = p2[0] - p1[0], p2[1] - p1[1]

    a = dx**2 + dy**2
    b = 2 * (dx * (p1[0] - center[0]) + dy * (p1[1] - center[1]))
    c = (p1[0] - center[0]) ** 2 + (p1[1] - center[1]) ** 2 - radius**2

    discriminant = b**2 - 4 * a * c
    assert discriminant > 0, "Not a secant!"

    t1 = (-b + discriminant**0.5) / (2 * a)
    t2 = (-b - discriminant**0.5) / (2 * a)

    return (dx * t1 + p1[0], dy * t1 + p1[1]), (dx * t2 + p1[0], dy * t2 + p1[1])


