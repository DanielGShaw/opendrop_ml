#!/usr/bin/env python
"""Generate speed/timing figures for the experimental dataset.

Produces:
  - figures/experimental/experimental_speed.png  (F3: 2x2 grid)

Run from OpenDrop_OP/:
    python paper_results/plot/plot_speed.py
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
    setup_matplotlib, save_figure,
    COLORS, MARKERS, LABELS, FIGSIZE_SPEED,
)

SUBDIR = "experimental"

# Methods in display order for bar charts
BAR_METHODS = ["tangent", "circular", "ellipse", "yl", "ml"]
BAR_COLORS = [COLORS[m] for m in BAR_METHODS]
BAR_LABELS = [LABELS[m] for m in BAR_METHODS]


def load_data():
    """Load experimental results CSV."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, "experimental_results.csv"))
    return df


def plot_speed_composite(df):
    """2x2 speed figure matching legacy layout.

    (a) Preprocessing time: classical vs ML
    (b) Total analysis time per method
    (c) Fit/prediction time per method (log scale)
    (d) Fit time vs profile length (scatter, log y)
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_SPEED)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # --- (a) Preprocessing time ---
    preproc_classical = df["preprocessing_time"].values
    preproc_ml = df["ml_preprocessing_time"].values

    categories_a = ["Existing methods", "ML model"]
    means_a = [preproc_classical.mean(), preproc_ml.mean()]
    sems_a = [preproc_classical.std() / np.sqrt(len(preproc_classical)),
              preproc_ml.std() / np.sqrt(len(preproc_ml))]
    colors_a = ["white", COLORS["ml"]]

    ax_a.bar(categories_a, means_a, yerr=sems_a, capsize=5,
             color=colors_a, edgecolor="black", width=0.5)
    ax_a.set_ylabel("Time (s)")
    ax_a.text(0.02, 0.96, "a)", transform=ax_a.transAxes,
              fontsize=12, fontweight="bold", va="top")

    # --- (b) Total analysis time per method ---
    # Total = preprocessing + fit time (for classical) or preprocessing + prediction (for ML)
    totals = {}
    for method in BAR_METHODS:
        if method == "ml":
            totals[method] = (df["ml_preprocessing_time"]
                              + df["ml_prediction_time"]).values
        else:
            totals[method] = (df["preprocessing_time"]
                              + df["{}_fit_time".format(method)]).values

    means_b = [totals[m].mean() for m in BAR_METHODS]
    sems_b = [totals[m].std() / np.sqrt(len(totals[m])) for m in BAR_METHODS]

    ax_b.bar(BAR_LABELS, means_b, yerr=sems_b, capsize=5,
             color=BAR_COLORS, edgecolor="black", width=0.5)
    ax_b.set_ylabel("Time (s)")
    ax_b.tick_params(axis="x", rotation=30)
    ax_b.text(0.02, 0.96, "b)", transform=ax_b.transAxes,
              fontsize=12, fontweight="bold", va="top")

    # --- (c) Fit/prediction time per method (log scale) ---
    fit_times = {}
    for method in BAR_METHODS:
        if method == "ml":
            fit_times[method] = df["ml_prediction_time"].values
        else:
            fit_times[method] = df["{}_fit_time".format(method)].values

    means_c = [fit_times[m].mean() for m in BAR_METHODS]
    sems_c = [fit_times[m].std() / np.sqrt(len(fit_times[m]))
              for m in BAR_METHODS]

    ax_c.bar(BAR_LABELS, means_c, yerr=sems_c, capsize=5,
             color=BAR_COLORS, edgecolor="black", width=0.5)
    ax_c.set_yscale("log")
    ax_c.set_ylabel("Time (s)")
    ax_c.tick_params(axis="x", rotation=30)
    ax_c.text(0.02, 0.96, "c)", transform=ax_c.transAxes,
              fontsize=12, fontweight="bold", va="top")

    # --- (d) Fit time vs profile length (scatter, log y) ---
    profile_len = df["profile_length"].values

    for method in BAR_METHODS:
        if method == "ml":
            times = df["ml_prediction_time"].values
        else:
            times = df["{}_fit_time".format(method)].values

        ax_d.scatter(profile_len, times,
                     marker=MARKERS[method], s=50,
                     facecolor=COLORS[method] + "40",
                     edgecolor=COLORS[method],
                     label=LABELS[method])

    ax_d.set_yscale("log")
    ax_d.set_xlabel("Profile length (px)")
    ax_d.set_ylabel("Time (s)")
    ax_d.legend(fontsize=10, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax_d.text(0.02, 0.96, "d)", transform=ax_d.transAxes,
              fontsize=12, fontweight="bold", va="top")

    fig.tight_layout()
    save_figure(fig, SUBDIR, "experimental_speed.png")


def main():
    setup_matplotlib()

    print("=" * 60)
    print("Speed Figures")
    print("=" * 60)

    df = load_data()
    print("Loaded {} images".format(len(df)))

    plot_speed_composite(df)

    print()
    print("Speed figure generated.")


if __name__ == "__main__":
    main()
