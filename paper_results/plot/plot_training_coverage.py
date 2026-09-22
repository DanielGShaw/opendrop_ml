#!/usr/bin/env python
"""Generate the training-domain-coverage figure for the experimental set.

Two-panel SI/main figure comparing the experimental dataset's operating
domain against the ML model's training domain:

  (a) Drop profile length vs YL angle — per-angle training min/max bounds,
      rendered directly from the training grid's scaler/Bond endpoints
      (results/training_domain_bounds.csv), drawn over the
      experimental population.
  (b) YL Bond number vs YL angle (RMSE < 1 px) — training Bo bounds
      [0, 1] drawn over the experimental population.

Output:
  figures/experimental/training_domain_coverage.png

Run from OpenDrop_OP/:
    python paper_results/plot/plot_training_coverage.py
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_RESULTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PAPER_RESULTS_DIR)
sys.path.insert(0, SCRIPT_DIR)

from config import RESULTS_DIR
from plot_utils import setup_matplotlib, save_figure

SUBDIR = "experimental"
OUT_FILENAME = "training_domain_coverage.png"

# Filter threshold on YL RMSE for panel (b) — matches the legacy figure.
BOND_RMSE_FILTER_PX = 1.0

# Training Bo range — synthetic training grid spans Bo ∈ [0, 1] inclusive.
TRAIN_BO_MIN = 0.0
TRAIN_BO_MAX = 1.0


def load_data():
    """Return (df_exp, df_bounds) — both sourced from paper_results/results/.

    `df_bounds` is the training envelope (results/training_domain_bounds.csv):
    scaler in {1, 4} x Bond in {0, 1} x roughness 0, rendered
    directly from the training grid's own endpoints rather than derived from
    the evaluation set, whose scalers (0.95-5.0) extend beyond the training
    grid's [1, 4] and would otherwise widen the band past what was trained on.
    """
    df_exp = pd.read_csv(os.path.join(RESULTS_DIR, "experimental_results.csv"))
    df_bounds = pd.read_csv(os.path.join(RESULTS_DIR, "training_domain_bounds.csv"))
    return df_exp, df_bounds


def training_profile_length_bounds(df_bounds):
    """Return (angles, min_lengths, max_lengths) grouped by angle."""
    grouped = (
        df_bounds.groupby("angle")["profile_length"].agg(["min", "max"]).sort_index()
    )
    angles = grouped.index.tolist()
    min_lengths = grouped["min"].tolist()
    max_lengths = grouped["max"].tolist()
    return angles, min_lengths, max_lengths


def plot_training_coverage(df_exp, df_bounds):
    """Build the 1x2 training-coverage figure and return the Figure."""
    angles, min_lens, max_lens = training_profile_length_bounds(df_bounds)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ----- (a) Profile length vs angle -----
    ax = axes[0]
    ax.scatter(
        df_exp["yl_left_angle"],
        df_exp["profile_length"],
        facecolor="none",
        edgecolor="black",
        label=r"All experimental data ($\theta_{YL}$)",
    )
    ax.scatter(
        df_exp["yl_right_angle"],
        df_exp["profile_length"],
        facecolor="none",
        edgecolor="black",
    )
    ax.plot(angles, min_lens, "r", label=r"Training bounds ($\theta_{True}$)")
    ax.plot(angles, max_lens, "r")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Drop profile length")
    ax.legend()

    # ----- (b) Bond number vs angle (RMSE filter) -----
    ax = axes[1]
    filt = df_exp[df_exp["yl_RMSE"] < BOND_RMSE_FILTER_PX]
    ax.scatter(
        filt["yl_left_angle"],
        filt["yl_bond_number"],
        facecolor="none",
        edgecolor="black",
        label=rf"YL fit RMSE < {BOND_RMSE_FILTER_PX:g} px ($\theta_{{YL}}$)",
    )
    ax.scatter(
        filt["yl_right_angle"],
        filt["yl_bond_number"],
        facecolor="none",
        edgecolor="black",
    )
    ax.plot(
        angles,
        [TRAIN_BO_MIN] * len(angles),
        "r",
        label=r"Training bounds ($\theta_{True}$)",
    )
    ax.plot(angles, [TRAIN_BO_MAX] * len(angles), "r")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Bond number")
    ax.legend()

    fig.tight_layout()
    return fig


def main():
    setup_matplotlib()

    print("Loading experimental results and training-domain bounds from paper_results/results/")
    df_exp, df_bounds = load_data()
    print(f"  Experimental rows: {len(df_exp)}   Training-bounds rows: {len(df_bounds)}")

    fig = plot_training_coverage(df_exp, df_bounds)
    save_figure(fig, SUBDIR, OUT_FILENAME)


if __name__ == "__main__":
    main()
