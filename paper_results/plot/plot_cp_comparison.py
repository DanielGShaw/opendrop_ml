#!/usr/bin/env python
"""Generate CP-comparison subfigure components (SVG) for the paper.

For each curated image, produces three SVGs that compare the YL fit and
ML prediction tangent lines at the contact points:

  - {stem}_right.svg    Right contact-point zoom (6x6)
  - {stem}_left.svg     Left contact-point zoom (6x6)
  - {stem}_overview.svg Full drop with YL fit curve and red CP zoom boxes (10x10)

Output: paper_results/figures/CA_tangent_plots/{experimental,synthetic}/

Image selections are user-curated and hard-coded below. Edit the
EXPERIMENTAL_IMAGES / SYNTHETIC_IMAGES lists to change selections.

Run as part of `run_all.py`, or standalone:

    python -m paper_results.plot.plot_cp_comparison
"""

import os
import re
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_RESULTS_DIR = os.path.dirname(SCRIPT_DIR)
OPENDROP_DIR = os.path.dirname(PAPER_RESULTS_DIR)
sys.path.insert(0, OPENDROP_DIR)
sys.path.insert(0, PAPER_RESULTS_DIR)
sys.path.insert(0, SCRIPT_DIR)

from config import (
    EXPERIMENTAL_DATA_DIR,
    SYNTHETIC_DATA_DIR,
    FIGURES_DIR,
    ML_MODEL_DIR,
)

from opendrop_ml.modules.preprocessing.preprocessing import isolate_contour, CPID
from opendrop_ml.modules.ML_model.prepare_experimental import (
    prepare_for_model,
    experimental_pred,
)
from opendrop_ml.modules.fitting.BA_fit import yl_fit

import tensorflow as tf


# ---------------------------------------------------------------------------
# Curated image lists
# ---------------------------------------------------------------------------
# File names (with extension) relative to paper_results/data/experimental/
EXPERIMENTAL_IMAGES = [
    "image_original (67).png",
    "10-1.bmp",
    "steelball.jpg",
    "14-1.bmp",
    "122.094841003417.bmp",
    "20230404_water_solvent_2a_0_100_10_3.jpg",
]

# File names (with extension) relative to paper_results/data/synthetic/
SYNTHETIC_IMAGES = [
    "20.0_0.6000000000000001_4.0_0.006_CPL_302_461_CPR_442_461_.png",
    "100.0_0.8_4.0_0.0_CPL_303_461_CPR_441_461_.png",
    "180.0_0.0_1.0_0.006_CPL_372_461_CPR_372_461_.png",
]

# Panel labels overlaid on the top-left of each *_overview.svg (aligned to the
# image lists above; set to None for no label). Experimental labels mirror the
# i)–vi) markers on the agreement-panel extrema in the plan.
SYNTHETIC_LABELS = ["i)", "ii)", "iii)"]
EXPERIMENTAL_LABELS = ["i)", "ii)", "iii)", "iv)", "v)", "vi)"]


# ---------------------------------------------------------------------------
# Rendering constants (match legacy reference exactly)
# ---------------------------------------------------------------------------
LINE_LENGTH = 40
LINE_WIDTH = 5
BELOW = False
# figsize chosen to match the content aspect ratio so the axes fills the figure
# without whitespace. Zoom crops use width/height = 1.4; overview uses
# width/height = 1/0.83 ≈ 1.205. A small vertical allowance is added for the title.
ZOOM_FIGSIZE = (7.0, 5.5)
OVERVIEW_FIGSIZE = (12.0, 10.5)
ZOOM_SUBPLOT = dict(left=0.02, right=0.98, bottom=0.02, top=0.98)
OVERVIEW_SUBPLOT = dict(left=0.02, right=0.98, bottom=0.02, top=0.98)

# Fraction of the tangent-span added as padding on each side of the crop
# (controls how much empty context sits around the tangent lines). The other
# axis is derived from this to preserve the 1.4 zoom aspect ratio.
CROP_PADDING = 0.4

OUTPUT_ROOT = os.path.join(FIGURES_DIR, "CA_tangent_plots")


# ---------------------------------------------------------------------------
# Tangent line helper (ported verbatim from reference, minor renames only)
# ---------------------------------------------------------------------------
def return_tangent_line(CPs, left_angle, right_angle, line_length=50, below=False):
    """Return array of shape (2, 2, 2): [[left_p1, left_p2], [right_p1, right_p2]].

    Ported from Other figures needed.py:872. Image coordinates have y
    increasing downward, hence the sign flips on y components.
    """
    CP_L = CPs[0]
    CP_R = CPs[1]

    # Right tangent
    if below is False:
        right_point1 = CP_R
    else:
        r_x1 = CP_R[0] - (line_length * np.cos((180 - right_angle) * (np.pi / 180)))
        r_y1 = CP_R[1] + (line_length * np.sin((180 - right_angle) * (np.pi / 180)))
        right_point1 = [r_x1, r_y1]
    r_x2 = line_length * np.cos((180 - right_angle) * (np.pi / 180)) + CP_R[0]
    r_y2 = CP_R[1] - (line_length * np.sin((180 - right_angle) * (np.pi / 180)))
    right_point2 = [r_x2, r_y2]

    # Left tangent
    if below is False:
        left_point1 = CP_L
    else:
        l_x1 = CP_L[0] + (line_length * np.cos((180 - left_angle) * (np.pi / 180)))
        l_y1 = CP_L[1] + (line_length * np.sin((180 - left_angle) * (np.pi / 180)))
        left_point1 = [l_x1, l_y1]
    l_x2 = CP_L[0] - (line_length * np.cos((180 - left_angle) * (np.pi / 180)))
    l_y2 = CP_L[1] - (line_length * np.sin((180 - left_angle) * (np.pi / 180)))
    left_point2 = [l_x2, l_y2]

    return np.array([[left_point1, left_point2], [right_point1, right_point2]])


# ---------------------------------------------------------------------------
# Per-image compute
# ---------------------------------------------------------------------------
def extract_profile(contour, CPs):
    """Extract drop profile between the two contact points."""
    idx_left = np.where((contour == CPs[0]).all(axis=1))[0][0]
    idx_right = np.where((contour == CPs[1]).all(axis=1))[0][0]
    if idx_left > idx_right:
        idx_left, idx_right = idx_right, idx_left
    return contour[idx_left : idx_right + 1]


def compute_for_image(img_path, model):
    """Run preprocessing + YL fit + ML prediction for one image."""
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Failed to load {img_path}")

    _, contour, _, _ = isolate_contour(img, DISPLAY=False, blur=True)
    CPs = CPID(contour, display=False)
    profile = extract_profile(contour, CPs)

    (
        yl_angles,
        Bo,
        _baseline_width,
        _volume,
        fit_YL,
        baseline,
        yl_errors,
        _sym_errors,
        _timings,
    ) = yl_fit(profile, display=False)
    YL_intercepts = np.array(
        [[baseline[0, 0], baseline[0, 1]], [baseline[-1, 0], baseline[-1, 1]]]
    )
    rmse = yl_errors.get("RMSE", np.nan) if isinstance(yl_errors, dict) else np.nan

    pred_ds = prepare_for_model(img, input_len=50, n_surface=10, display=False, blur=True)
    predictions, _ = experimental_pred(pred_ds, model, side="both")
    ml_angles = [float(predictions[0]), float(predictions[1])]

    return {
        "img": img,
        "contour": contour,
        "CPs": CPs,
        "YL_angles": [float(yl_angles[0]), float(yl_angles[1])],
        "YL_intercepts": YL_intercepts,
        "fit_YL": np.asarray(fit_YL),
        "Bo": float(Bo),
        "rmse": float(rmse) if rmse == rmse else rmse,
        "ML_angles": ml_angles,
    }


# ---------------------------------------------------------------------------
# Crop bound computation (ported from reference, width/height = 1.4 aspect)
# ---------------------------------------------------------------------------
def _crop_bounds(yl_tangent_side, ml_tangent_side):
    """Return (xlims, ylims) for an auto-zoomed view with 25% padding.

    Aspect ratio is forced to 1.4 (width/height), matching the reference.
    """
    leftmost = min(yl_tangent_side[:, 0].min(), ml_tangent_side[:, 0].min())
    rightmost = max(yl_tangent_side[:, 0].max(), ml_tangent_side[:, 0].max())
    highest = max(yl_tangent_side[:, 1].max(), ml_tangent_side[:, 1].max())
    lowest = min(yl_tangent_side[:, 1].min(), ml_tangent_side[:, 1].min())

    width = rightmost - leftmost
    height = highest - lowest
    pad = CROP_PADDING
    if width > height:
        xlims = [leftmost - width * pad, rightmost + width * pad]
        width = xlims[1] - xlims[0]
        height = width / 1.4
        ylims = [highest + height / 6, highest - height * 5 / 6]
    else:
        ylims = [highest + height * pad, lowest - height * pad]
        height = max(ylims) - min(ylims)
        width = 1.4 * height
        xlims = [leftmost - width * pad, rightmost + width * pad]
    return xlims, ylims


def _compute_tangents(data):
    """Return (yl_tangent, ml_tangent), each shape (2, 2, 2)."""
    yl_tangent = return_tangent_line(
        data["YL_intercepts"],
        data["YL_angles"][0],
        data["YL_angles"][1],
        LINE_LENGTH,
        BELOW,
    )
    ml_tangent = return_tangent_line(
        data["CPs"],
        data["ML_angles"][0],
        data["ML_angles"][1],
        LINE_LENGTH,
        BELOW,
    )
    return yl_tangent, ml_tangent


def _pad_width(xlims, ylims, target_width, side, img_width):
    """Widen xlims to target_width by extending outward on `side` (the side of
    the drop the zoom shows). If the outward extension would run past the
    image edge (xlims < 0 or xlims > img_width), spill the overflow to the
    opposite side so the crop stays inside the image.

    ylims are rescaled to preserve the 1.4 aspect ratio, centered on the
    existing y-midpoint so the contact point stays in roughly the same
    visual position.
    """
    current_width = xlims[1] - xlims[0]
    extra = target_width - current_width
    if extra <= 1e-9:
        return list(xlims), list(ylims)

    max_left_ext = max(0.0, xlims[0])
    max_right_ext = max(0.0, img_width - xlims[1])

    if side == "left":
        left_ext = min(extra, max_left_ext)
        right_ext = extra - left_ext
    else:
        right_ext = min(extra, max_right_ext)
        left_ext = extra - right_ext
    new_xlims = [xlims[0] - left_ext, xlims[1] + right_ext]

    new_height = target_width / 1.4
    y_mid = (ylims[0] + ylims[1]) / 2.0
    # Preserve convention: ylims[0] is the larger (bottom) value, ylims[1] smaller (top)
    new_ylims = [y_mid + new_height / 2.0, y_mid - new_height / 2.0]
    return new_xlims, new_ylims


def _draw_cp_zoom(ax, data, side, xlims, ylims, yl_tangent, ml_tangent, legend_fontsize=18):
    """Render the CP detail onto an existing axes."""
    img = data["img"]
    contour = data["contour"]
    fit_YL = data["fit_YL"]
    YL_angles = data["YL_angles"]
    ML_angles = data["ML_angles"]

    idx = 1 if side == "right" else 0

    ax.imshow(img)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(
        yl_tangent[idx, :, 0],
        yl_tangent[idx, :, 1],
        "-",
        color="cyan",
        linewidth=LINE_WIDTH,
        label="YL fit - " + "{:.2f}°".format(YL_angles[idx]),
    )
    ax.plot(fit_YL[:, 0], fit_YL[:, 1], "-", color="cyan")
    ax.plot(
        ml_tangent[idx, :, 0],
        ml_tangent[idx, :, 1],
        "--",
        color="red",
        linewidth=LINE_WIDTH,
        label="ML prediction - " + "{:.2f}°".format(ML_angles[idx]),
    )
    ax.plot(
        contour[:, 0], contour[:, 1], ".", color="magenta", label="Detected edge"
    )
    ax.legend(
        loc="upper right" if side == "right" else "upper left",
        fontsize=legend_fontsize,
        markerscale=3,
        handlelength=3,
        handleheight=1.2,
    )




def _parse_synthetic_stem(stem):
    """Parse ground-truth params from a synthetic-image stem.

    Returns dict with gt_angle, gt_bond, gt_scale, gt_roughness — or None if
    the stem does not follow the synthetic-image naming convention
    `{angle}_{bond}_{scale}_{roughness}_CPL_x_y_CPR_x_y_`.
    """
    if stem is None:
        return None
    name = stem.rstrip("_")
    m = re.match(r"^(.+?)_CPL_\d+_\d+_CPR_\d+_\d+$", name)
    if not m:
        return None
    parts = m.group(1).split("_")
    if len(parts) != 4:
        return None
    try:
        return {
            "gt_angle": float(parts[0]),
            "gt_bond": float(parts[1]),
            "gt_scale": float(parts[2]),
            "gt_roughness": float(parts[3]),
        }
    except ValueError:
        return None


def _draw_overview(ax, data, xlims_left, ylims_left, xlims_right, ylims_right,
                   stem=None, panel_label=None, overlay_fontsize=None):
    """Render the overview onto an existing axes."""
    img = data["img"]
    contour = data["contour"]
    fit_YL = data["fit_YL"]
    Bo = data["Bo"]
    rmse = data["rmse"]

    left_square = [
        [xlims_left[0], ylims_left[0]],
        [xlims_left[0], ylims_left[1]],
        [xlims_left[1], ylims_left[1]],
        [xlims_left[1], ylims_left[0]],
        [xlims_left[0], ylims_left[0]],
    ]
    right_square = [
        [xlims_right[0], ylims_right[0]],
        [xlims_right[0], ylims_right[1]],
        [xlims_right[1], ylims_right[1]],
        [xlims_right[1], ylims_right[0]],
        [xlims_right[0], ylims_right[0]],
    ]
    x_left, y_left = zip(*left_square)
    x_right, y_right = zip(*right_square)

    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(
        contour[:, 0], contour[:, 1], ",", color="magenta", label="Detected edge"
    )
    ax.plot(fit_YL[:, 0], fit_YL[:, 1], "-", color="cyan")
    ax.plot(x_left, y_left, "r")
    ax.plot(x_right, y_right, "r")

    xlims = [contour[:, 0].min(), contour[:, 0].max()]
    height = (xlims[1] - xlims[0]) * 0.83
    # Clamp the bottom to the image height so the image's bottom edge sits flush
    # against the axes bottom (no blank band below the photograph).
    bottom = min(contour[:, 1].max() + height * 0.16, img.shape[0])
    ylims = [bottom, bottom - height]
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    info_lines = [
        f"YL Bond number: {Bo:.3g}",
        f"YL RMSE: {rmse:.3g} px",
    ]
    gt = _parse_synthetic_stem(stem)
    if gt is not None:
        info_lines.append(r"$\theta_{True}$: " + f"{gt['gt_angle']:.3g}°")
        info_lines.append(f"True Bond number: {gt['gt_bond']:.3g}")

    if overlay_fontsize is None:
        overlay_fontsize = 18 * (OVERVIEW_FIGSIZE[0] / ZOOM_FIGSIZE[0])
    overlay_bbox = dict(facecolor="0.85", edgecolor="none", alpha=0.8, pad=6)
    ax.text(
        0.5, 0.97,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=overlay_fontsize, color="black",
        bbox=overlay_bbox,
    )
    if panel_label:
        ax.text(
            0.03, 0.97,
            panel_label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=overlay_fontsize, color="black",
            bbox=overlay_bbox,
        )




# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def prepare_entry(img_path, model):
    """Compute everything needed to render one image (no padding applied).

    Returns a dict: stem, data, yl_tangent, ml_tangent, xlims_left, ylims_left,
    xlims_right, ylims_right.
    """
    stem = os.path.splitext(os.path.basename(img_path))[0]
    print(f"  Processing {os.path.basename(img_path)}")
    data = compute_for_image(img_path, model)
    print(
        "    YL = ({:.2f}, {:.2f})   ML = ({:.2f}, {:.2f})   Bo = {:.3g}   RMSE = {:.3g}".format(
            data["YL_angles"][0],
            data["YL_angles"][1],
            data["ML_angles"][0],
            data["ML_angles"][1],
            data["Bo"],
            data["rmse"],
        )
    )
    yl_tangent, ml_tangent = _compute_tangents(data)
    xlims_left, ylims_left = _crop_bounds(yl_tangent[0], ml_tangent[0])
    xlims_right, ylims_right = _crop_bounds(yl_tangent[1], ml_tangent[1])
    return {
        "stem": stem,
        "data": data,
        "yl_tangent": yl_tangent,
        "ml_tangent": ml_tangent,
        "xlims_left": xlims_left,
        "ylims_left": ylims_left,
        "xlims_right": xlims_right,
        "ylims_right": ylims_right,
    }


def prepare_batch(image_names, data_dir, model, labels=None):
    """Compute data + padded crop bounds for every image in the batch.

    `labels`, if provided, must be aligned to `image_names` and supplies the
    panel label ('i)', 'ii)', ...) rendered in the top-left of each overview.
    Returns a list of entry dicts ready for `compose_figure`.
    """
    entries = []
    for idx, fname in enumerate(image_names):
        img_path = os.path.join(data_dir, fname)
        if not os.path.exists(img_path):
            print(f"  SKIP (missing): {img_path}")
            continue
        try:
            entry = prepare_entry(img_path, model)
            if labels is not None and idx < len(labels):
                entry["panel_label"] = labels[idx]
            entries.append(entry)
        except Exception as e:
            print(f"  FAILED {fname}: {e}")

    if not entries:
        return entries

    # Find the widest zoom across all (left + right) in the batch and pad the
    # narrower ones outward so every zoom cell has the same xlim range.
    max_width = 0.0
    for e in entries:
        for key in ("xlims_left", "xlims_right"):
            w = e[key][1] - e[key][0]
            if w > max_width:
                max_width = w
    print(f"  Batch max zoom width: {max_width:.1f}px — padding narrower zooms")
    for e in entries:
        img_width = e["data"]["img"].shape[1]
        e["xlims_left"], e["ylims_left"] = _pad_width(
            e["xlims_left"], e["ylims_left"], max_width, side="left",
            img_width=img_width,
        )
        e["xlims_right"], e["ylims_right"] = _pad_width(
            e["xlims_right"], e["ylims_right"], max_width, side="right",
            img_width=img_width,
        )
    return entries


def compose_figure(entries, out_path, col_width=6.0, row_height=None):
    """Assemble a 3-column composite (left, overview, right) with one row per
    entry, saved as a high-dpi PNG. All subplots share column widths so the
    panels align cleanly even though the content aspect ratios differ.
    """
    n_rows = len(entries)
    if row_height is None:
        # Match the zoom aspect (1.4) so left/right panels fill their cell.
        row_height = col_width / 1.4
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(3 * col_width, n_rows * row_height),
        squeeze=False,
    )
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995,
                        wspace=0.02, hspace=0.02)

    for row, e in enumerate(entries):
        ax_left, ax_over, ax_right = axes[row]
        _draw_cp_zoom(
            ax_left, e["data"], "left",
            e["xlims_left"], e["ylims_left"],
            e["yl_tangent"], e["ml_tangent"],
        )
        _draw_cp_zoom(
            ax_right, e["data"], "right",
            e["xlims_right"], e["ylims_right"],
            e["yl_tangent"], e["ml_tangent"],
        )
        # Overview cell is the same figure-width as zoom cells, so the overlay
        # text should be the same size as the zoom legends (both at fontsize=18).
        _draw_overview(
            ax_over, e["data"],
            e["xlims_left"], e["ylims_left"],
            e["xlims_right"], e["ylims_right"],
            stem=e["stem"],
            panel_label=e.get("panel_label"),
            overlay_fontsize=18,
        )

    fig.savefig(out_path, format="png", dpi=600)
    plt.close(fig)
    print(f"  Wrote composite {out_path}")


def main():
    plt.rc("font", size=14)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("Loading ML model from", ML_MODEL_DIR)
    model = tf.keras.models.load_model(ML_MODEL_DIR)

    print("\nSynthetic images")
    syn_entries = prepare_batch(
        SYNTHETIC_IMAGES, SYNTHETIC_DATA_DIR, model, labels=SYNTHETIC_LABELS,
    )
    if syn_entries:
        compose_figure(
            syn_entries, os.path.join(OUTPUT_ROOT, "synthetic_composite.png"),
        )

    print("\nExperimental images")
    exp_entries = prepare_batch(
        EXPERIMENTAL_IMAGES, EXPERIMENTAL_DATA_DIR, model,
        labels=EXPERIMENTAL_LABELS,
    )
    if exp_entries:
        compose_figure(
            exp_entries, os.path.join(OUTPUT_ROOT, "experimental_composite.png"),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
