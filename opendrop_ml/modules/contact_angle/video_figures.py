"""Diagnostic figure generation for video-mode CA analysis.

Produces two PNGs saved alongside the per-run CSV:

  {name}_timeseries.png  -- left/right CA vs frame number for all methods
  {name}_cp_scatter.png  -- Pass 1 CPs overlaid on the first frame with
                            fitted median contact lines and marginal
                            box-whisker distributions.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

from opendrop_ml.modules.image.frame_loader import get_first_frame


def generate_timeseries_figure(
    csv_path,
    output_path,
    median_tilt_angle: Optional[float] = None,
):
    """Plot all enabled methods' left/right angles vs frame number.

    Reads the unified CSV written by ``UnifiedCsvWriter`` and produces a
    two-panel figure (top = left angles, bottom = right angles) with one
    line per method.

    Args:
        csv_path: Path to the unified results CSV.
        output_path: Where to save the PNG.
        median_tilt_angle: Optional median surface-tilt angle (degrees)
            emitted by ``_single_pass_with_tilt``. When provided and
            finite, the top-panel title is suffixed with the detected
            tilt for at-a-glance inspection. Non-tilted pipelines pass
            ``None`` and the suffix is omitted.
    """
    df = pd.read_csv(csv_path)
    if "frame_number" not in df.columns:
        raise ValueError("CSV missing 'frame_number' column")

    frames = df["frame_number"].values

    left_cols = [c for c in df.columns if c.endswith(" left angle")]
    right_cols = [c for c in df.columns if c.endswith(" right angle")]

    methods = []
    for lc in left_cols:
        method_name = lc.replace(" left angle", "")
        rc = method_name + " right angle"
        if rc in df.columns:
            methods.append((method_name, lc, rc))

    if not methods:
        raise ValueError("No angle columns found in CSV")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for method_name, lc, rc in methods:
        left_vals = df[lc].values
        right_vals = df[rc].values

        mask = np.isfinite(left_vals)
        axes[0].plot(
            frames[mask],
            left_vals[mask],
            "o-",
            label=method_name,
            markersize=3,
            linewidth=0.8,
        )

        mask = np.isfinite(right_vals)
        axes[1].plot(
            frames[mask],
            right_vals[mask],
            "o-",
            label=method_name,
            markersize=3,
            linewidth=0.8,
        )

    title = "Contact Angle Time Series"
    if median_tilt_angle is not None and not math.isnan(float(median_tilt_angle)):
        title += "  (median tilt: {:.1f}\u00b0)".format(float(median_tilt_angle))

    axes[0].set_ylabel("Left \u03b8 (\u00b0)")
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Frame number")
    axes[1].set_ylabel("Right \u03b8 (\u00b0)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_cp_scatter_figure(
    source_path,
    pass1_cps,
    contact_lines,
    output_path,
    left_needle_x=None,
    right_needle_x=None,
    rotated_first_frame=None,
):
    """Overlay Pass 1 contact points on the first frame with fitted lines.

    Produces a 4-quadrant figure: main scatter panel with the first frame
    as background, top marginal for x distributions, right marginal for y
    distributions.

    Args:
        source_path: Path to the video file or image directory (for loading
            the first frame as background).
        pass1_cps: Dict ``{"left": [[x,y],...], "right": [[x,y],...]}``.
        contact_lines: Dict ``{"left": [[x1,y1],[x2,y2]], "right": ...}``.
        output_path: Where to save the PNG.
        left_needle_x: Optional x-coordinate of the median left needle
            intercept. Rendered as a full-height vertical line on the
            scatter panel when provided. Emitted by ``_needle_in_two_pass``;
            ``_auto_two_pass`` passes ``None`` and the line is omitted.
        right_needle_x: As ``left_needle_x`` but for the right side.
        rotated_first_frame: Optional BGR frame already tilt-corrected by
            ``_single_pass_with_tilt``. When provided, used as the background
            so the CPs (which are in rotated-frame coordinates) align with
            the image. Non-tilted pipelines pass ``None`` and the original
            first frame is loaded via ``source_path``.
    """
    if pass1_cps is None:
        raise ValueError("No Pass 1 contact point data available")

    left_pts = np.asarray(pass1_cps["left"])
    right_pts = np.asarray(pass1_cps["right"])

    if len(left_pts) == 0 and len(right_pts) == 0:
        raise ValueError("Pass 1 produced no contact points")

    bg_img = None
    if rotated_first_frame is not None:
        bg_img = cv2.cvtColor(rotated_first_frame, cv2.COLOR_BGR2RGB)
    else:
        try:
            frame, _, _ = get_first_frame(source_path)
            bg_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(4, 4, hspace=0.5, wspace=0.5)

    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_xbox = fig.add_subplot(gs[0, 0:3])
    ax_ybox = fig.add_subplot(gs[1:4, 3])

    if bg_img is not None:
        ax_scatter.imshow(bg_img)

    if len(left_pts) > 0:
        ax_scatter.scatter(
            left_pts[:, 0],
            left_pts[:, 1],
            alpha=0.6,
            s=50,
            zorder=6,
            label="Left CPs",
            color="C0",
        )
    if len(right_pts) > 0:
        ax_scatter.scatter(
            right_pts[:, 0],
            right_pts[:, 1],
            alpha=0.6,
            s=50,
            zorder=6,
            label="Right CPs",
            color="C1",
        )

    if contact_lines:
        left_line = contact_lines.get("left")
        right_line = contact_lines.get("right")
        if left_line is not None:
            ll = np.asarray(left_line)
            ax_scatter.plot(
                ll[:, 0],
                ll[:, 1],
                "-",
                color="red",
                linewidth=2,
                zorder=7,
            )
        if right_line is not None:
            rl = np.asarray(right_line)
            ax_scatter.plot(
                rl[:, 0],
                rl[:, 1],
                "-",
                color="red",
                linewidth=2,
                zorder=7,
                label="Median contact line",
            )

    needle_xs = [v for v in (left_needle_x, right_needle_x) if v is not None]
    for i, nx in enumerate(needle_xs):
        ax_scatter.axvline(
            nx,
            linestyle="--",
            color="magenta",
            linewidth=1.5,
            alpha=0.8,
            zorder=4,
            label="Median needle line" if i == 0 else None,
        )

    ax_scatter.set_title("Contact Point Scatter Plot", fontsize=14)
    ax_scatter.set_xlabel("x (px)", fontsize=12)
    ax_scatter.set_ylabel("y (px)", fontsize=12)
    ax_scatter.legend(fontsize=10)

    xlims = ax_scatter.get_xlim()
    ylims = ax_scatter.get_ylim()
    pos_scatter = ax_scatter.get_position()

    all_x = []
    all_y = []
    box_labels = []
    if len(left_pts) > 0:
        all_x.append(left_pts[:, 0])
        all_y.append(left_pts[:, 1])
        box_labels.append("Left CPs")
    if len(right_pts) > 0:
        all_x.append(right_pts[:, 0])
        all_y.append(right_pts[:, 1])
        box_labels.append("Right CPs")

    if all_x:
        ax_xbox.boxplot(all_x, vert=False, labels=box_labels)
        ax_xbox.set_title("x Distributions", fontsize=13)
        ax_xbox.set_xlim(xlims)
        ax_xbox.set_xlabel("x (px)", fontsize=11)

        positions = list(range(1, len(all_y) + 1))
        ax_ybox.boxplot(
            all_y,
            vert=True,
            positions=positions,
            labels=box_labels,
            patch_artist=False,
            showfliers=True,
        )
        ax_ybox.set_title("y Distributions", fontsize=13)
        ax_ybox.set_xticklabels(box_labels, rotation=45, ha="right", fontsize=10)
        ax_ybox.invert_yaxis()
        ax_ybox.set_ylim(ylims)
        ax_ybox.set_ylabel("y (px)", fontsize=11)

        pos_ybox = ax_ybox.get_position()
        ax_ybox.set_position(
            [
                pos_ybox.x0,
                pos_scatter.y0,
                pos_ybox.width,
                pos_scatter.height,
            ]
        )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
