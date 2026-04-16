"""Shared plotting configuration for paper figures.

Defines colors, markers, figure sizes, and helper functions used
by all figure-generation scripts. Matches the legacy styling from
files_from_old_computer/figure_generation_notebooks/.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_RESULTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PAPER_RESULTS_DIR)

from config import FIGURES_DIR, ANGLE_BUCKETS, ANGLE_BUCKET_LABELS

# ---------------------------------------------------------------------------
# Method colors (hex) — matches legacy scripts exactly
# ---------------------------------------------------------------------------
COLORS = {
    "tangent": "#1b9e77",   # green
    "circular": "#7570b3",  # purple
    "ellipse": "#585858",   # gray
    "yl": "#000000",        # black
    "ml": "#e41a1c",        # red
}

# Transparent versions for left/right split in agreement scatter plots
COLORS_TRANSPARENT = {k: v + "00" for k, v in COLORS.items()}

# ---------------------------------------------------------------------------
# Method markers — matches legacy scripts
# ---------------------------------------------------------------------------
MARKERS = {
    "tangent": "^",     # triangle up
    "circular": "o",    # circle
    "ellipse": "d",     # diamond
    "yl": "X",          # cross
    "ml": "*",          # star
}

# ---------------------------------------------------------------------------
# Display labels for methods
# ---------------------------------------------------------------------------
LABELS = {
    "tangent": "Tangent fit",
    "circular": "Circle fit",
    "ellipse": "Ellipse fit",
    "yl": "YL fit",
    "ml": "ML model",
}

# ---------------------------------------------------------------------------
# Standard marker size
# ---------------------------------------------------------------------------
MARKER_SIZE = 100

# ---------------------------------------------------------------------------
# Standard figure sizes
# ---------------------------------------------------------------------------
FIGSIZE_SINGLE = (8, 8)       # Single agreement plot
FIGSIZE_COMBINED = (6, 6)     # Combined multi-method agreement
FIGSIZE_SPEED = (14, 8)       # 2x2 speed grid
FIGSIZE_BAR = (10, 6)         # Single bar chart
FIGSIZE_SCATTER = (8, 6)      # Single scatter/error plot

# ---------------------------------------------------------------------------
# Matplotlib global settings — matches legacy
# ---------------------------------------------------------------------------
def setup_matplotlib():
    """Apply global matplotlib style settings."""
    plt.rc("font", size=14)
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


# ---------------------------------------------------------------------------
# Methods to show in figures (excludes polynomial — see plan Phase 0.5)
# ---------------------------------------------------------------------------
FIGURE_METHODS = ["tangent", "circular", "ellipse", "yl", "ml"]

# For experimental agreement, YL is the reference (x-axis), not a method
EXPERIMENTAL_AGREEMENT_METHODS = ["tangent", "circular", "ellipse", "ml"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def angle_bucket(angle):
    """Classify angle into bucket label."""
    for (lo, hi), label in zip(ANGLE_BUCKETS, ANGLE_BUCKET_LABELS):
        if lo <= angle < hi:
            return label
    return ">110"


def save_figure(fig, subdir, filename):
    """Save figure as PNG at 300 dpi into figures/<subdir>/."""
    out_dir = os.path.join(FIGURES_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(out_path))
    return out_path


def add_agreement_line(ax, lims=(0, 180)):
    """Add perfect agreement diagonal and ±5% bands."""
    lo, hi = lims
    ax.plot([lo, hi], [lo, hi], "-", color=COLORS["yl"],
            label="Perfect Agreement", zorder=0)
    ax.plot([0, hi * 0.95], [0, hi], "--", color=COLORS["yl"],
            label=u"\u00B1 5%", zorder=0)
    ax.plot([0, hi * 1.05], [0, hi], "--", color=COLORS["yl"], zorder=0)


def scatter_lr(ax, x_left, y_left, x_right, y_right, method, label=None):
    """Plot left and right angle pairs as separate scatter sets.

    Left points use transparent fill, right points use solid fill,
    matching the legacy convention.
    """
    color = COLORS[method]
    marker = MARKERS[method]
    lbl = label if label else LABELS[method]

    # Left angles — transparent facecolor
    ax.scatter(x_left, y_left, marker=marker, s=MARKER_SIZE,
               facecolor=color + "00", edgecolor=color, label=lbl)
    # Right angles — no label (avoid legend duplicate)
    ax.scatter(x_right, y_right, marker=marker, s=MARKER_SIZE,
               facecolor=color + "00", edgecolor=color)
