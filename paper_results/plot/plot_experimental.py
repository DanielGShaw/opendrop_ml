#!/usr/bin/env python
"""Generate experimental dataset figures.

Produces:
  - figures/experimental/experimental_agreement_6panel.png     (F1b: 3x2 RMSE split)
  - figures/experimental/experimental_yl_characterization.png  (E3)


Run from OpenDrop_OP/:
    python paper_results/plot/plot_experimental.py
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_RESULTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PAPER_RESULTS_DIR)
sys.path.insert(0, SCRIPT_DIR)

from config import RESULTS_DIR
from plot_utils import (
    setup_matplotlib,
    save_figure,
    add_agreement_line,
    COLORS,
    MARKERS,
    LABELS,
    MARKER_SIZE,
    FIGSIZE_BAR,
)

SUBDIR = "experimental"


def load_data():
    """Load experimental results CSV."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "experimental_results.csv"))
    return df


# -----------------------------------------------------------------------
# F1b: 3x2 Agreement/Disagreement split by YL RMSE
# -----------------------------------------------------------------------


def plot_agreement_6panel(df):
    """3x2 figure: agreement (left) and disagreement (right) for
    all images, YL RMSE < 1 px, and YL RMSE >= 1 px.

    Matches legacy pred_vs_true_plots_experimental.py In[64].
    """
    yl_left = df["yl_left_angle"].values
    yl_right = df["yl_right_angle"].values
    ml_left = df["ml_left_angle"].values
    ml_right = df["ml_right_angle"].values
    yl_rmse = df["yl_RMSE"].values

    # Build concatenated L+R arrays for each subset
    subsets = [
        ("All Experimental Images", np.ones(len(df), dtype=bool)),
        ("YL fit RMSE < 1 px", yl_rmse < 1),
        ("YL fit RMSE \u2265 1 px", yl_rmse >= 1),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10, 14), constrained_layout=True)
    panel_labels = ["a)", "b)", "c)", "d)", "e)", "f)"]

    for row, (title, mask) in enumerate(subsets):
        yl_vals = np.concatenate([yl_left[mask], yl_right[mask]])
        ml_vals = np.concatenate([ml_left[mask], ml_right[mask]])
        err_vals = ml_vals - yl_vals

        # --- Left column: agreement scatter ---
        ax_agree = axes[row, 0]
        mid = len(yl_vals) // 2
        ax_agree.scatter(
            yl_vals[:mid],
            ml_vals[:mid],
            marker=MARKERS["ml"],
            s=MARKER_SIZE,
            facecolor=COLORS["ml"] + "00",
            edgecolor=COLORS["ml"],
            label="ML model",
        )
        ax_agree.scatter(
            yl_vals[mid:],
            ml_vals[mid:],
            marker=MARKERS["ml"],
            s=MARKER_SIZE,
            facecolor=COLORS["ml"] + "00",
            edgecolor=COLORS["ml"],
        )
        add_agreement_line(ax_agree)
        ax_agree.set_xlim(0, 180)
        ax_agree.set_ylim(0, 180)
        ax_agree.set_aspect("equal")
        ax_agree.set_xlabel("$\\theta_{YL}$ (\u00B0)")
        ax_agree.set_ylabel("$\\theta_{ML}$ (\u00B0)")
        ax_agree.set_title(title)
        ax_agree.text(
            0.01,
            1.02,
            panel_labels[row * 2],
            transform=ax_agree.transAxes,
            fontsize=16,
            fontweight="bold",
        )
        ax_agree.legend(loc="upper left")

        # --- Right column: disagreement (error) ---
        ax_err = axes[row, 1]
        ax_err.scatter(
            yl_vals[:mid],
            err_vals[:mid],
            marker=MARKERS["ml"],
            s=MARKER_SIZE,
            facecolor=COLORS["ml"] + "00",
            edgecolor=COLORS["ml"],
            label="ML model",
        )
        ax_err.scatter(
            yl_vals[mid:],
            err_vals[mid:],
            marker=MARKERS["ml"],
            s=MARKER_SIZE,
            facecolor=COLORS["ml"] + "00",
            edgecolor=COLORS["ml"],
        )
        ax_err.set_xlim(10, 185)
        ax_err.set_ylim(-20, 10)
        ax_err.set_xlabel("$\\theta_{YL}$ (\u00B0)")
        ax_err.set_ylabel("$\\theta_{ML} - \\theta_{YL}$ (\u00B0)")
        ax_err.set_title(title)
        ax_err.text(
            0.01,
            1.02,
            panel_labels[row * 2 + 1],
            transform=ax_err.transAxes,
            fontsize=16,
            fontweight="bold",
        )
        ax_err.legend(loc="lower left")

    # "i)"-"vi)" annotations on right-column panels (from reference In[64])
    _annot_kw = dict(fontsize=16, fontweight="bold")
    # Panel b) — row 0, right
    for txt, xy in [
        ("i)", (0.04, 0.92)),
        ("ii)", (0.32, 0.91)),
        ("iii)", (0.81, 0.91)),
        ("iv)", (0.05, 0.24)),
        ("v)", (0.56, 0.05)),
        ("vi)", (0.89, 0.1)),
    ]:
        axes[0, 1].text(xy[0], xy[1], txt, transform=axes[0, 1].transAxes, **_annot_kw)
    # Panel d) — row 1, right
    for txt, xy in [("iii)", (0.81, 0.91)), ("v)", (0.56, 0.05))]:
        axes[1, 1].text(xy[0], xy[1], txt, transform=axes[1, 1].transAxes, **_annot_kw)
    # Panel f) — row 2, right
    for txt, xy in [
        ("i)", (0.04, 0.91)),
        ("ii)", (0.32, 0.92)),
        ("iv)", (0.05, 0.24)),
        ("vi)", (0.89, 0.1)),
    ]:
        axes[2, 1].text(xy[0], xy[1], txt, transform=axes[2, 1].transAxes, **_annot_kw)

    save_figure(fig, SUBDIR, "experimental_agreement_6panel.png")


# -----------------------------------------------------------------------
# E3: YL Characterization
# -----------------------------------------------------------------------


def plot_yl_characterization(df):
    """YL fit RMSE vs angle and Bond number distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    yl_avg = (df["yl_left_angle"] + df["yl_right_angle"]) / 2.0

    # RMSE vs angle
    ax1.scatter(
        yl_avg,
        df["yl_RMSE"],
        marker="o",
        s=40,
        facecolor="#00000040",
        edgecolor="black",
    )
    ax1.set_xlabel("$\\theta_{YL}$ (\u00B0)")
    ax1.set_ylabel("YL fit RMSE (px)")
    ax1.set_title("YL Fit Quality")

    # Bond number distribution
    ax2.scatter(
        yl_avg,
        df["yl_bond_number"],
        marker="o",
        s=40,
        facecolor="#00000040",
        edgecolor="black",
    )
    ax2.set_yscale("log")
    ax2.set_xlabel("$\\theta_{YL}$ (\u00B0)")
    ax2.set_ylabel("Bond number")
    ax2.set_title("Bond Number Distribution")

    fig.tight_layout()
    save_figure(fig, SUBDIR, "experimental_yl_characterization.png")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    setup_matplotlib()

    print("=" * 60)
    print("Experimental Figures")
    print("=" * 60)

    df = load_data()
    print("Loaded {} images".format(len(df)))

    plot_agreement_6panel(df)
    plot_yl_characterization(df)

    print()
    print("All experimental figures generated.")


if __name__ == "__main__":
    main()
