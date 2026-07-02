#!/usr/bin/env python
"""Generate figures comparing v05.05 vs v03 on the Conan sensitivity dataset.

Produces (in figures/comparison/):
  - sensitivity_agreement.png      (agreement scatter + delta plot)

Run from OpenDrop_OP/:
    python paper_results/plot/plot_sensitivity.py
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
    MARKER_SIZE,
)

SUBDIR = "comparison"

# Colors for the two models
COLOR_V03 = "#999999"  # gray for old model
COLOR_V05 = COLORS["ml"]  # red for new model


def load_data():
    """Load sensitivity results CSV."""
    path = os.path.join(RESULTS_DIR, "sensitivity_results.csv")
    df = pd.read_csv(path)
    # Compute average errors
    df["v05_err"] = (
        (df["v05_ml_left"] - df["gt_angle"]).abs()
        + (df["v05_ml_right"] - df["gt_angle"]).abs()
    ) / 2.0
    df["v03_err"] = (
        (df["v03_ml_left"] - df["gt_angle"]).abs()
        + (df["v03_ml_right"] - df["gt_angle"]).abs()
    ) / 2.0
    return df


# -----------------------------------------------------------------------
# Figure 1: Side-by-side agreement scatter
# -----------------------------------------------------------------------


def plot_agreement(df):
    """2 row figure: top row is 1x2 agreement scatter (v03, v05.05),
    bottom row is a wide delta plot (v05 - v03 vs GT)."""
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[2, 1],
        hspace=0.35,
        wspace=0.3,
        left=0.08,
        right=0.97,
        top=0.95,
        bottom=0.06,
    )

    # --- Top row: agreement scatter ---
    lims = (100, 190)
    color = COLOR_V05
    marker = "*"

    configs = [
        (0, "v03_ml_left", "v03_ml_right", "Conan 2024"),
        (1, "v05_ml_left", "v05_ml_right", "This work"),
    ]

    for col_idx, y_l_col, y_r_col, model_name in configs:
        ax = fig.add_subplot(gs[0, col_idx])
        gt = df["gt_angle"].values

        # Plot left and right as single series (no distinction)
        x_all = np.concatenate([gt, gt])
        y_all = np.concatenate([df[y_l_col].values, df[y_r_col].values])
        ax.scatter(
            x_all,
            y_all,
            marker=marker,
            s=MARKER_SIZE,
            facecolor="none",
            edgecolor=color,
            label=model_name,
        )

        add_agreement_line(ax, lims=lims)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel("$\\theta_{True}$ (\u00B0)")
        ax.set_ylabel("$\\theta_{ML}$ (\u00B0)")
        ax.legend(loc="upper left")

        label_char = chr(ord("a") + col_idx)
        ax.text(
            0.01,
            1.02,
            "{})".format(label_char),
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
        )

    # --- Bottom row: delta plot spanning both columns ---
    ax_c = fig.add_subplot(gs[1, :])
    gt = df["gt_angle"].values
    delta_l = df["v05_ml_left"].values - df["v03_ml_left"].values
    delta_r = df["v05_ml_right"].values - df["v03_ml_right"].values

    x_all = np.concatenate([gt, gt])
    delta_all = np.concatenate([delta_l, delta_r])
    ax_c.scatter(
        x_all,
        delta_all,
        marker=marker,
        s=MARKER_SIZE,
        facecolor="none",
        edgecolor=color,
    )
    ax_c.axhline(0, color="black", linewidth=0.8, zorder=0)
    ax_c.set_xlabel("$\\theta_{True}$ (\u00B0)")
    ax_c.set_ylabel("$\\theta_{This\\ work}$ \u2212 $\\theta_{Conan\\ 2024}$ (\u00B0)")
    ax_c.set_xlim(lims)
    ax_c.text(
        0.005,
        1.05,
        "c)",
        transform=ax_c.transAxes,
        fontsize=16,
        fontweight="bold",
    )

    save_figure(fig, SUBDIR, "sensitivity_agreement.png")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    setup_matplotlib()

    print("=" * 60)
    print("Sensitivity Dataset Figures")
    print("=" * 60)

    df = load_data()
    print("Loaded {} images".format(len(df)))

    plot_agreement(df)

    print()
    print("All sensitivity figure generated.")


if __name__ == "__main__":
    main()
