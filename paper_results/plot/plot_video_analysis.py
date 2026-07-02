#!/usr/bin/env python
"""Generate validation figures for video analysis datasets.

Produces:
  - figures/video_analysis/inverse_phase_timeseries.png
  - figures/video_analysis/isaac_{dataset}_timeseries.png (per dataset)
  - figures/video_analysis/isaac_cp_scatter.png
  - figures/video_analysis/kristina_peek_results.png

Run from OpenDrop_OP/:
    python paper_results/plot/plot_video_analysis.py
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_RESULTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PAPER_RESULTS_DIR)
sys.path.insert(0, SCRIPT_DIR)

from config import (
    RESULTS_DIR,
    ISAAC_DIR,
    INVERSE_PHASE_DIR,
    VIDEO_EXTENSIONS,
    IMAGE_EXTENSIONS,
)
from plot_utils import (
    setup_matplotlib,
    save_figure,
    add_agreement_line,
    scatter_lr,
    COLORS,
    MARKERS,
    LABELS,
    MARKER_SIZE,
)

SUBDIR = "video_analysis"


def _find_video_file(dataset_dir):
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


def _get_first_frame(dataset_name):
    """Load the first frame from an Isaac dataset video."""
    dataset_dir = os.path.join(ISAAC_DIR, dataset_name)
    if not os.path.isdir(dataset_dir):
        return None
    video_path = _find_video_file(dataset_dir)
    if video_path is None:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return frame


# -----------------------------------------------------------------------
# Inverse Phase: time-series of ML and tangent angles vs frame number
# -----------------------------------------------------------------------


def plot_inverse_phase_timeseries():
    """ML predicted angle vs frame number with tangent fit overlay."""
    results_path = os.path.join(RESULTS_DIR, "inverse_phase_results.csv")
    if not os.path.isfile(results_path):
        print("  Inverse phase results not found, skipping.")
        return

    df = pd.read_csv(results_path)
    frames = df["frame_number"].values

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel a: ML angles vs frame number
    ax = axes[0]
    if "ml_left_angle" in df.columns:
        ax.plot(
            frames,
            df["ml_left_angle"],
            "*-",
            color=COLORS["ml"],
            label="ML left",
            markersize=4,
            linewidth=0.8,
        )
        ax.plot(
            frames,
            df["ml_right_angle"],
            "*-",
            color=COLORS["ml"] + "50",
            label="ML right",
            markersize=4,
            linewidth=0.8,
        )
    ax.set_ylabel("$\\theta$ (\u00B0)")
    ax.set_title("Inverse Phase — ML Model Predictions")
    ax.legend()
    ax.text(0.01, 1.02, "a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

    # Panel b: Tangent angles vs frame number
    ax = axes[1]
    if "tangent_left_angle" in df.columns:
        ax.plot(
            frames,
            df["tangent_left_angle"],
            "^-",
            color=COLORS["tangent"],
            label="Tangent left",
            markersize=4,
            linewidth=0.8,
        )
        ax.plot(
            frames,
            df["tangent_right_angle"],
            "^-",
            color=COLORS["tangent"] + "50",
            label="Tangent right",
            markersize=4,
            linewidth=0.8,
        )
    ax.set_xlabel("Frame number")
    ax.set_ylabel("$\\theta$ (\u00B0)")
    ax.set_title("Inverse Phase — Tangent Fit")
    ax.legend()
    ax.text(0.01, 1.02, "b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

    fig.tight_layout()
    save_figure(fig, SUBDIR, "inverse_phase_timeseries.png")


# -----------------------------------------------------------------------
# Inverse Phase: CP scatter overlaid on first image
# -----------------------------------------------------------------------


def plot_inverse_phase_cp_scatter():
    """Contact point positions overlaid on first inverse phase image,
    with marginal box-whisker plots for x and y distributions.

    Matches the style of inverse_phase_contact_lines2.png from the
    reference notebook (Inverse_phase_analysis.ipynb).
    """
    results_path = os.path.join(RESULTS_DIR, "inverse_phase_results.csv")
    if not os.path.isfile(results_path):
        print("  Inverse phase results not found, skipping CP scatter.")
        return

    df = pd.read_csv(results_path)

    # Use raw CPID contact points (per-frame, before median-line stabilization)
    # to show the scatter that motivates the median contact line approach.
    # Fall back to median-line CPs if raw CPID columns not available.
    if "cpid_left_x" in df.columns:
        cp_cols = ["cpid_left_x", "cpid_left_y", "cpid_right_x", "cpid_right_y"]
    else:
        cp_cols = ["left_CP_x", "left_CP_y", "right_CP_x", "right_CP_y"]
    valid = df.dropna(subset=cp_cols)

    if len(valid) == 0:
        print("  No valid CPs in inverse phase results, skipping CP scatter.")
        return

    # Load first image for background
    bg_img = None
    if os.path.isdir(INVERSE_PHASE_DIR):
        import re

        images = sorted(
            (
                f
                for f in os.listdir(INVERSE_PHASE_DIR)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ),
            key=lambda s: [
                int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(s))
            ],
        )
        if images:
            first_img = cv2.imread(os.path.join(INVERSE_PHASE_DIR, images[0]))
            if first_img is not None:
                bg_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)

    # Data arrays
    left_x = valid[cp_cols[0]].values
    left_y = valid[cp_cols[1]].values
    right_x = valid[cp_cols[2]].values
    right_y = valid[cp_cols[3]].values

    # --- Layout: 4x4 grid (matching reference notebook) ---
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(4, 4, hspace=0.5, wspace=0.5)

    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_xbox = fig.add_subplot(gs[0, 0:3])
    ax_ybox = fig.add_subplot(gs[1:4, 3])

    # --- Main scatter panel ---
    if bg_img is not None:
        ax_scatter.imshow(bg_img)

    ax_scatter.scatter(
        left_x,
        left_y,
        alpha=0.6,
        s=50,
        zorder=6,
        label="Left Contact Points",
        color="C0",
    )
    ax_scatter.scatter(
        right_x,
        right_y,
        alpha=0.6,
        s=50,
        zorder=6,
        label="Right Contact Points",
        color="C1",
    )

    # Fitted contact lines (red)
    left_pts = np.column_stack([left_x, left_y])
    right_pts = np.column_stack([right_x, right_y])

    left_line = _fit_cp_line(left_pts)
    right_line = _fit_cp_line(right_pts)

    if left_line is not None:
        ax_scatter.plot(
            left_line[:, 0], left_line[:, 1], "-", color="red", linewidth=2, zorder=5
        )
    if right_line is not None:
        ax_scatter.plot(
            right_line[:, 0],
            right_line[:, 1],
            "-",
            color="red",
            linewidth=2,
            zorder=5,
            label="Contact line movement",
        )

    ax_scatter.set_title("Contact Point Scatter Plot", fontsize=14)
    ax_scatter.set_xlabel("x-axis (px)", fontsize=12)
    ax_scatter.set_ylabel("y-axis (px)", fontsize=12)
    ax_scatter.legend(fontsize=10)

    # Capture scatter axis limits (set by imshow) for syncing marginals
    xlims = ax_scatter.get_xlim()
    ylims = ax_scatter.get_ylim()
    pos_scatter = ax_scatter.get_position()

    # --- Top panel: x-distribution box-whisker ---
    scatter_labels = ["Left Contact Points", "Right Contact Points"]
    ax_xbox.boxplot([left_x, right_x], vert=False, labels=scatter_labels)
    ax_xbox.set_title("x Distributions", fontsize=13)
    ax_xbox.tick_params(axis="y", labelsize=10)

    # Median annotations
    for i, x_vals in enumerate([left_x, right_x]):
        median_x = np.median(x_vals)
        ax_xbox.annotate(
            "{:.0f}".format(median_x),
            xy=(median_x, i + 1.2),
            va="center",
            ha="left",
            fontsize=9,
            color="black",
        )
    ax_xbox.set_xlim(xlims)
    ax_xbox.set_xlabel("x-axis (px)", fontsize=11)

    # --- Right panel: y-distribution box-whisker ---
    # Offset positions and manual outlier plotting (matching reference)
    positions = [0.9, 1.9]
    box_y = ax_ybox.boxplot(
        [left_y, right_y],
        vert=True,
        positions=positions,
        labels=scatter_labels,
        patch_artist=False,
        showfliers=False,
    )

    # Plot outliers manually, offset to the right of the box
    for i, y_vals in enumerate([left_y, right_y]):
        q1, q3 = np.percentile(y_vals, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [v for v in y_vals if v < lower_bound or v > upper_bound]
        x_outliers = [positions[i] + 0.2] * len(outliers)
        ax_ybox.scatter(x_outliers, outliers, color="black", alpha=0.3, s=15)

    ax_ybox.set_title("y Distributions", fontsize=13)
    ax_ybox.set_xticks(positions)
    ax_ybox.set_xticklabels(scatter_labels, rotation=45, ha="right", fontsize=10)
    ax_ybox.tick_params(axis="x", labelsize=10)
    ax_ybox.invert_yaxis()
    ax_ybox.set_ylim(ylims)
    ax_ybox.set_ylabel("y-axis (px)", fontsize=11)

    # Sync y-box vertical extent with scatter panel
    pos_ybox = ax_ybox.get_position()
    ax_ybox.set_position(
        [pos_ybox.x0, pos_scatter.y0, pos_ybox.width, pos_scatter.height]
    )

    save_figure(fig, SUBDIR, "inverse_phase_cp_scatter.png")


# -----------------------------------------------------------------------
# Isaac: time-series per dataset, overlaid with commercial reference
# -----------------------------------------------------------------------


def plot_isaac_timeseries():
    """ML predicted angle vs frame number for each Isaac dataset."""
    datasets = ["MS_APTES_S2_0001", "0822_10k_Bb5", "SC69_1"]

    for dataset in datasets:
        results_path = os.path.join(RESULTS_DIR, "isaac_{}_results.csv".format(dataset))
        if not os.path.isfile(results_path):
            print("  Isaac {} results not found, skipping.".format(dataset))
            continue

        df = pd.read_csv(results_path)
        frames = df["frame_number"].values

        fig, ax = plt.subplots(figsize=(14, 6))

        # ML predictions
        if "ml_left_angle" in df.columns:
            ml_left = df["ml_left_angle"].values
            ml_right = df["ml_right_angle"].values
            ax.plot(
                frames,
                ml_left,
                "*-",
                color=COLORS["ml"],
                label="ML left",
                markersize=3,
                linewidth=0.5,
            )
            ax.plot(
                frames,
                ml_right,
                "*-",
                color=COLORS["ml"] + "50",
                label="ML right",
                markersize=3,
                linewidth=0.5,
            )

        # Tangent fit
        if "tangent_left_angle" in df.columns:
            ax.plot(
                frames,
                df["tangent_left_angle"],
                "^-",
                color=COLORS["tangent"],
                label="Tangent left",
                markersize=3,
                linewidth=0.5,
                alpha=0.6,
            )
            ax.plot(
                frames,
                df["tangent_right_angle"],
                "^-",
                color=COLORS["tangent"] + "50",
                label="Tangent right",
                markersize=3,
                linewidth=0.5,
                alpha=0.6,
            )

        ax.set_xlabel("Frame number")
        ax.set_ylabel("$\\theta$ (\u00B0)")
        ax.set_title("Isaac — {} — Contact Angle Time Series".format(dataset))
        ax.legend(loc="best")

        fig.tight_layout()
        save_figure(fig, SUBDIR, "isaac_{}_timeseries.png".format(dataset))


def _fit_cp_line(CPs):
    """Fit a line through contact points for visualization.

    Returns [[x1,y1],[x2,y2]] endpoints of the fitted line segment,
    or None if fitting fails.
    """
    if len(CPs) < 3:
        return None
    q1 = np.percentile(CPs[:, 1], 25)
    q3 = np.percentile(CPs[:, 1], 75)
    inrange = CPs[(CPs[:, 1] >= q1) & (CPs[:, 1] <= q3)]
    if len(inrange) < 2:
        inrange = CPs
    # Simple OLS fit: y = mx + c
    x, y = inrange[:, 0], inrange[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    try:
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception:
        return None
    x_min, x_max = CPs[:, 0].min(), CPs[:, 0].max()
    return np.array([[x_min, m * x_min + c], [x_max, m * x_max + c]])


def plot_isaac_cp_scatter():
    """Contact point positions overlaid on first video frame for each Isaac dataset."""
    datasets = ["MS_APTES_S2_0001", "0822_10k_Bb5", "SC69_1"]
    available = []

    for dataset in datasets:
        results_path = os.path.join(RESULTS_DIR, "isaac_{}_results.csv".format(dataset))
        if os.path.isfile(results_path):
            df = pd.read_csv(results_path)
            if "left_CP_x" in df.columns:
                available.append((dataset, df))

    if not available:
        print("  No Isaac results with CP data found, skipping CP scatter.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 8), squeeze=False)

    for i, (dataset, df) in enumerate(available):
        ax = axes[0, i]
        valid = df.dropna(subset=["left_CP_x", "left_CP_y", "right_CP_x", "right_CP_y"])

        # Load first frame as background
        first_frame = _get_first_frame(dataset)
        if first_frame is not None:
            frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            ax.imshow(frame_rgb, aspect="auto")
        else:
            ax.set_facecolor("#f0f0f0")

        # Scatter CP positions (semi-transparent so background visible)
        ax.scatter(
            valid["left_CP_x"],
            valid["left_CP_y"],
            alpha=0.35,
            s=15,
            label="Left CP",
            color="C0",
            edgecolors="none",
            zorder=3,
        )
        ax.scatter(
            valid["right_CP_x"],
            valid["right_CP_y"],
            alpha=0.35,
            s=15,
            label="Right CP",
            color="C1",
            edgecolors="none",
            zorder=3,
        )

        # Fitted contact lines (red) through each CP cluster
        left_pts = valid[["left_CP_x", "left_CP_y"]].values
        right_pts = valid[["right_CP_x", "right_CP_y"]].values

        left_line = _fit_cp_line(left_pts)
        right_line = _fit_cp_line(right_pts)

        line_plotted = False
        if left_line is not None:
            ax.plot(
                left_line[:, 0],
                left_line[:, 1],
                "-",
                color="red",
                linewidth=2,
                zorder=5,
                label="Contact line" if not line_plotted else None,
            )
            line_plotted = True
        if right_line is not None:
            ax.plot(
                right_line[:, 0],
                right_line[:, 1],
                "-",
                color="red",
                linewidth=2,
                zorder=5,
                label="Contact line" if not line_plotted else None,
            )

        ax.set_title(dataset, fontsize=13)
        ax.set_xlabel("x (px)")
        if i == 0:
            ax.set_ylabel("y (px)")
        ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    save_figure(fig, SUBDIR, "isaac_cp_scatter.png")


# -----------------------------------------------------------------------
# Kristina PEEK: angle per usable frame, left vs right
# -----------------------------------------------------------------------


def plot_kristina_results():
    """ML predicted angle per usable frame for Kristina PEEK dataset."""
    results_path = os.path.join(
        RESULTS_DIR, "kristina_water_air_peek_60degrees_results.csv"
    )
    if not os.path.isfile(results_path):
        print("  Kristina results not found, skipping.")
        return

    df = pd.read_csv(results_path)

    # Filter to usable frames
    if "skipped" in df.columns:
        usable = df[df["skipped"] != True].copy()
    else:
        usable = df.copy()

    if len(usable) == 0:
        print("  No usable Kristina frames.")
        return

    frames = usable["frame_number"].values

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Panel a: ML predictions
    ax = axes[0]
    if "ml_left_angle" in usable.columns:
        ax.plot(
            frames,
            usable["ml_left_angle"],
            "*-",
            color=COLORS["ml"],
            label="ML left",
            markersize=6,
        )
        ax.plot(
            frames,
            usable["ml_right_angle"],
            "*-",
            color=COLORS["ml"] + "50",
            label="ML right",
            markersize=6,
        )
    ax.set_ylabel("$\\theta$ (\u00B0)")
    ax.set_title("Kristina PEEK 60° — ML Model")
    ax.legend()
    ax.text(0.01, 1.02, "a)", transform=ax.transAxes, fontsize=14, fontweight="bold")

    # Panel b: Left vs Right comparison (all methods)
    ax = axes[1]
    methods_to_plot = ["tangent", "circular", "ml"]
    for method in methods_to_plot:
        left_col = "{}_left_angle".format(method)
        right_col = "{}_right_angle".format(method)
        if left_col in usable.columns and right_col in usable.columns:
            left_vals = usable[left_col].dropna()
            right_vals = usable[right_col].dropna()
            color = COLORS.get(method, "gray")
            marker = MARKERS.get(method, "o")
            label = LABELS.get(method, method)
            ax.plot(
                frames[: len(left_vals)],
                left_vals,
                marker + "-",
                color=color,
                label="{} L".format(label),
                markersize=5,
                linewidth=0.8,
            )
            ax.plot(
                frames[: len(right_vals)],
                right_vals,
                marker + "-",
                color=color + "50",
                label="{} R".format(label),
                markersize=5,
                linewidth=0.8,
            )

    ax.set_xlabel("Frame number")
    ax.set_ylabel("$\\theta$ (\u00B0)")
    ax.set_title("Kristina PEEK 60° — Method Comparison")
    ax.legend(ncol=2, fontsize=8)
    ax.text(0.01, 1.02, "b)", transform=ax.transAxes, fontsize=14, fontweight="bold")

    fig.tight_layout()
    save_figure(fig, SUBDIR, "kristina_peek_results.png")


# -----------------------------------------------------------------------
# Inverse Phase: 5-panel agreement figure (method vs YL)
# -----------------------------------------------------------------------

# ML model styling — both use red stars
COLORS_V03 = COLORS["ml"]  # red (same as v05)
MARKERS_V03 = MARKERS["ml"]  # star (same as v05)
LABELS_V03 = "Conan 2024"


def plot_inverse_phase_agreement():
    """5-panel (3x2, last empty) agreement figure: method angle vs YL angle.

    Panels: a) Tangent, b) Circle, c) Ellipse, d) ML v03, e) ML v05.
    YL fit is the x-axis reference for all panels.
    """
    results_path = os.path.join(RESULTS_DIR, "inverse_phase_results.csv")
    if not os.path.isfile(results_path):
        print("  Inverse phase results not found, skipping agreement figure.")
        return

    df = pd.read_csv(results_path)

    # Need YL angles as reference
    if "yl_left_angle" not in df.columns:
        print("  YL angles not found in results, skipping agreement figure.")
        return

    # Filter to rows with valid YL angles
    valid_yl = df["yl_left_angle"].notna() & df["yl_right_angle"].notna()

    # Define panels: (key, left_col, right_col, color, marker, label)
    panels = [
        (
            "tangent",
            "tangent_left_angle",
            "tangent_right_angle",
            COLORS["tangent"],
            MARKERS["tangent"],
            LABELS["tangent"],
        ),
        (
            "circular",
            "circular_left_angle",
            "circular_right_angle",
            COLORS["circular"],
            MARKERS["circular"],
            LABELS["circular"],
        ),
        (
            "ellipse",
            "ellipse_left_angle",
            "ellipse_right_angle",
            COLORS["ellipse"],
            MARKERS["ellipse"],
            LABELS["ellipse"],
        ),
        (
            "v03",
            "v03_ml_left_angle",
            "v03_ml_right_angle",
            COLORS_V03,
            MARKERS_V03,
            LABELS_V03,
        ),
        (
            "ml",
            "ml_left_angle",
            "ml_right_angle",
            COLORS["ml"],
            MARKERS["ml"],
            "This work",
        ),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10, 14), constrained_layout=True)

    ml_keys = {"v03", "ml"}
    # Short subscripts for the y-axis label, matching plot_synthetic.py's
    # agreement plots (Tangent, Circle, Ellipse) — no " fit" suffix.
    ylabel_subscripts = {
        "tangent": "Tangent",
        "circular": "Circle",
        "ellipse": "Ellipse",
    }
    for idx, (key, lcol, rcol, color, marker, label) in enumerate(panels):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        # Filter to rows with valid data for this method
        if lcol not in df.columns or rcol not in df.columns:
            ax.text(
                0.5,
                0.5,
                "{}\n(not available)".format(label),
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
                color="gray",
            )
            continue

        mask = valid_yl & df[lcol].notna() & df[rcol].notna()
        yl_left = df.loc[mask, "yl_left_angle"].values
        yl_right = df.loc[mask, "yl_right_angle"].values
        m_left = df.loc[mask, lcol].values
        m_right = df.loc[mask, rcol].values

        # Scatter: left (transparent fill) and right
        ax.scatter(
            yl_left,
            m_left,
            marker=marker,
            s=MARKER_SIZE,
            facecolor=color + "00",
            edgecolor=color,
            label=label,
        )
        ax.scatter(
            yl_right,
            m_right,
            marker=marker,
            s=MARKER_SIZE,
            facecolor=color + "00",
            edgecolor=color,
        )

        add_agreement_line(ax, lims=(80, 180))
        ax.set_xlim(80, 180)
        ax.set_ylim(80, 180)
        ax.set_aspect("equal")
        ax.set_xlabel("$\\theta_{YL}$ (\u00B0)")
        if key in ml_keys:
            ax.set_ylabel("$\\theta_{ML}$ (\u00B0)")
        else:
            ax.set_ylabel("$\\theta_{%s}$ (\u00B0)" % ylabel_subscripts.get(key, key))
        ax.legend(loc="upper left")

    # Hide the empty 6th panel (row 2, col 1)
    axes[2, 1].set_visible(False)

    save_figure(fig, SUBDIR, "inverse_phase_agreement_5panel.png")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    setup_matplotlib()

    print("=" * 60)
    print("Video Analysis Figures")
    print("=" * 60)

    plot_inverse_phase_timeseries()
    plot_inverse_phase_cp_scatter()
    plot_inverse_phase_agreement()
    plot_isaac_timeseries()
    plot_isaac_cp_scatter()
    plot_kristina_results()

    print()
    print("All video analysis figures generated.")


if __name__ == "__main__":
    main()
