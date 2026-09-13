"""Scatter-plot the 50-point model window in true image pixel coordinates,
for the bad (folded) vs. good (clean) outlier scaler, so the fold is visible
directly against the rendered image's coordinate frame (not the normalized
apex-relative window prepare_contour/normalize_ds ultimately feeds the model).

Run:
    python plot_outlier_window.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "training_files"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from create_contours_args6 import draw_drop, extract_edges_CV
from matrix_utils import cell_seed

ANGLE, BOND, ROUGHNESS = 20, 0.6206391777630103, -0.007207227008199237
WINDOW_SIZE = 50
SURFACE_POINTS = 10


def raw_right_side_in_image_coords(angle, bond, scaler, roughness):
    """Reimplements prepare_contour's right-side selection (x > xapex) but
    keeps ORIGINAL image-pixel coordinates (no y-flip, no re-centering), so
    the result can be plotted directly against the rendered PNG's frame."""
    seed = cell_seed(angle, bond, scaler, roughness)
    np.random.seed(seed)
    img, CPs_img = draw_drop(angle, bond, scaler, roughness, 5000, dpi=64)
    contour = extract_edges_CV(img)  # raw image-pixel coords, y increases downward

    # apex x-position, per prepare_contour: midpoint of the two CPs' x
    xapex = (CPs_img[0][0] + CPs_img[1][0]) / 2.0
    right_mask = contour[:, 0] > xapex
    r_drop_raw = contour[right_mask]  # same relative order as prepare_contour's r_drop

    cp_raw = np.array(CPs_img[1])  # image-pixel coords of the (unsnapped) right CP
    # nearest raw point to the analytic CP -> same snap prepare_contour performs
    d = np.sum((r_drop_raw - cp_raw) ** 2, axis=1)
    cp_index = int(np.argmin(d))

    return img, r_drop_raw, cp_index


def extract_window(r_drop_raw, cp_index, window_size=WINDOW_SIZE, surface_points=SURFACE_POINTS):
    no_shift_start = window_size - surface_points
    start = cp_index - no_shift_start
    end = cp_index - no_shift_start + window_size
    return r_drop_raw[start:end], cp_index - start  # window slice, CP's position within it


def plot_case(ax, img, window, cp_pos_in_window, title):
    n = len(window)
    colors = plt.get_cmap("jet")(np.linspace(0, 1, n))
    ax.scatter(window[:, 0], window[:, 1], c=colors, s=40, zorder=3)
    ax.plot(window[:, 0], window[:, 1], "-", color="gray", linewidth=0.8, zorder=2, alpha=0.6)
    # mark the CP distinctly
    ax.scatter(
        [window[cp_pos_in_window, 0]], [window[cp_pos_in_window, 1]],
        s=200, facecolors="none", edgecolors="black", linewidths=2, zorder=4, label="Contact point"
    )
    # annotate order with a few index labels so the fold direction is legible
    for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
        ax.annotate(str(i), (window[i, 0], window[i, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.set_title(title)
    ax.set_xlabel("image x (px)")
    ax.set_ylabel("image y (px)")
    ax.invert_yaxis()  # image convention: y increases downward
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, scaler, label in [
        (axes[0], 0.9403283632516403, "BAD: scaler=0.9403283632516403\n(folded / self-crossing)"),
        (axes[1], 0.940, "GOOD: scaler=0.940\n(clean, monotonic)"),
    ]:
        img, r_drop_raw, cp_index = raw_right_side_in_image_coords(ANGLE, BOND, scaler, ROUGHNESS)
        window, cp_pos = extract_window(r_drop_raw, cp_index)
        plot_case(ax, img, window, cp_pos, label)
        print(f"{label.splitlines()[0]}: window shape={window.shape}, CP at position {cp_pos}")

    fig.suptitle(
        f"50-point model window in image pixel coordinates\n"
        f"angle={ANGLE}, bond={BOND:.4f}, roughness={ROUGHNESS:.4f} "
        f"(colour = order along window, black circle = contact point)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outlier_window_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
