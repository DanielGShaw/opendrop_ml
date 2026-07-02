#!/usr/bin/env python
"""Generate synthetic dataset figures.

Produces:
  - figures/synthetic/synthetic_agreement_5panel.png (3x2 multi-panel F2)
  - figures/synthetic/synthetic_bias_and_disagreement.png (S5)

Run from OpenDrop_OP/:
    python paper_results/plot/plot_synthetic.py
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
    setup_matplotlib, save_figure, add_agreement_line, scatter_lr,
    COLORS, MARKERS, LABELS, MARKER_SIZE,
    FIGURE_METHODS,
)

SUBDIR = "synthetic"


def load_data():
    """Load synthetic results CSV."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "synthetic_results.csv"))
    return df


def plot_agreement_5panel(df):
    """3x2 multi-panel agreement scatter (5 methods, 6th panel removed).

    Matches legacy plot_synthetic_results.py In[13] layout:
    3 rows x 2 cols, figsize (10, 14), each subplot shows GT angle (x)
    vs method-predicted angle (y) with perfect agreement line and +/-5% bands.
    """
    gt = df["gt_angle"].values
    lims = [0, 180]

    # LaTeX y-axis labels per method (matching legacy)
    ylabel_latex = {
        "tangent": u"$\\theta_{Tangent}$ (\u00B0)",
        "circular": u"$\\theta_{Circle}$ (\u00B0)",
        "ellipse": u"$\\theta_{Ellipse}$ (\u00B0)",
        "yl": u"$\\theta_{YL}$ (\u00B0)",
        "ml": u"$\\theta_{ML}$ (\u00B0)",
    }

    fig, axes = plt.subplots(3, 2, figsize=(10, 14),
                             constrained_layout=True)
    axes = axes.flatten()

    for idx, method in enumerate(FIGURE_METHODS):
        ax = axes[idx]
        left = df["{}_left_angle".format(method)].values
        right = df["{}_right_angle".format(method)].values

        scatter_lr(ax, gt, left, gt, right, method)
        add_agreement_line(ax, lims=tuple(lims))

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(u"$\\theta_{True}$ (\u00B0)")
        ax.set_ylabel(ylabel_latex[method])
        ax.legend(loc="upper left")

    # Remove unused 6th subplot
    for idx in range(len(FIGURE_METHODS), len(axes)):
        fig.delaxes(axes[idx])

    save_figure(fig, SUBDIR, "synthetic_agreement_5panel.png")


def plot_bias_and_disagreement(df):
    """Two-panel figure: bias vs true angle (YL + ML) and ML-YL disagreement.

    Reproduces legacy pred_vs_true_plots_experimental.py lines 1031-1053
    (originally saved as 'synthetic data bias and disagreement angle.png').

    Left panel: bias (measured - true) vs GT angle for YL and ML.
    Right panel: (ML - YL) vs GT angle.
    Both left and right contact angles are plotted as separate points.
    """
    gt = df["gt_angle"].values
    gt_doubled = np.concatenate([gt, gt])

    # YL measured angles (x-axis for right panel)
    yl_x = np.concatenate([
        df["yl_left_angle"].values,
        df["yl_right_angle"].values,
    ])

    # Bias = measured - true, for both left and right sides
    yl_bias = np.concatenate([
        df["yl_left_angle"].values - gt,
        df["yl_right_angle"].values - gt,
    ])
    ml_bias = np.concatenate([
        df["ml_left_angle"].values - gt,
        df["ml_right_angle"].values - gt,
    ])

    # Disagreement = ML - YL, for both left and right sides
    disagree = np.concatenate([
        df["ml_left_angle"].values - df["yl_left_angle"].values,
        df["ml_right_angle"].values - df["yl_right_angle"].values,
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Left panel: bias vs true angle
    axes[0].scatter(gt_doubled, yl_bias, edgecolor=COLORS["yl"],
                    facecolor="None", marker=MARKERS["yl"],
                    s=MARKER_SIZE, label=LABELS["yl"])
    axes[0].scatter(gt_doubled, ml_bias, edgecolor=COLORS["ml"],
                    facecolor="None", marker=MARKERS["ml"],
                    s=MARKER_SIZE, label=LABELS["ml"])
    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[0].set_xlabel(u"$\\theta_{True}$ (\u00B0)")
    axes[0].set_ylabel(u"Bias (\u00B0)")
    axes[0].text(0.01, 1.02, "a)", transform=axes[0].transAxes,
                 fontsize=16, fontweight="bold")
    axes[0].legend()

    # Right panel: ML - YL disagreement vs YL angle
    axes[1].scatter(yl_x, disagree, edgecolor=COLORS["ml"],
                    facecolor="None", marker=MARKERS["ml"],
                    s=MARKER_SIZE,
                    label=u"$\\theta_{ML} - \\theta_{YL}$")
    axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[1].set_xlabel(u"$\\theta_{YL}$ (\u00B0)")
    axes[1].set_ylabel(u"$\\theta_{ML} - \\theta_{YL}$ (\u00B0)")
    axes[1].text(0.01, 1.02, "b)", transform=axes[1].transAxes,
                 fontsize=16, fontweight="bold")
    axes[1].legend()

    # i)-iv) annotations linking to the curated systems shown in
    # figures/CA_tangent_plots/synthetic_composite.png
    composite_systems = [
        ("i)", "20.0_0.6000000000000001_4.0_0.006_CPL_302_461_CPR_442_461_.png",
         "left", (0, 10)),
        ("ii)", "100.0_0.8_4.0_0.0_CPL_303_461_CPR_441_461_.png",
         "left", (-6, -26)),
        ("iii)", "160.0_0.8_4.0_-0.03_CPL_331_447_CPR_413_447_.png",
         "left", (-6, 10)),
        ("iv)", "180.0_1.0_1.0_0.006_CPL_234_461_CPR_510_461_.png",
         "left", (12, -14)),
    ]
    for label, image_name, side, offset_b in composite_systems:
        row = df.loc[df["image_name"] == image_name]
        if row.empty:
            continue
        row = row.iloc[0]
        yl_angle = row[f"yl_{side}_angle"]
        ml_angle = row[f"ml_{side}_angle"]

        # Panel b): (theta_ML - theta_YL) vs theta_YL
        axes[1].annotate(label, xy=(yl_angle, ml_angle - yl_angle),
                          xytext=offset_b, textcoords="offset points",
                          ha="center", fontsize=12, fontweight="bold")

    save_figure(fig, SUBDIR, "synthetic_bias_and_disagreement.png")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    setup_matplotlib()

    print("=" * 60)
    print("Synthetic Figures")
    print("=" * 60)

    df = load_data()
    print("Loaded {} images".format(len(df)))

    plot_agreement_5panel(df)
    plot_bias_and_disagreement(df)

    print()
    print("All synthetic figures generated.")


if __name__ == "__main__":
    main()
