#!/usr/bin/env python
"""Compare video analysis results against reference CSVs.

Produces:
  - comparison/inverse_phase_comparison.csv
  - comparison/isaac_comparison.csv
  - comparison/kristina_comparison.csv
  - Console report with per-method statistics

Run from OpenDrop_OP/:
    python paper_results/compare/compare_video_analysis.py
"""

import sys
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_RESULTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PAPER_RESULTS_DIR)

from config import (
    RESULTS_DIR,
    COMPARISON_DIR,
    INVERSE_PHASE_REF_DIR,
    ISAAC_REF_DIR,
    KRISTINA_REF_DIR,
)


def compare_inverse_phase():
    """Compare inverse phase results against reference CSV."""
    print()
    print("=" * 70)
    print("Inverse Phase Comparison")
    print("=" * 70)

    results_path = os.path.join(RESULTS_DIR, "inverse_phase_results.csv")
    if not os.path.isfile(results_path):
        print("  Results not found: {}".format(results_path))
        return

    new = pd.read_csv(results_path)
    print("  New results: {} rows".format(len(new)))

    # Look for reference CSV
    ref_path = os.path.join(
        INVERSE_PHASE_REF_DIR, "outputs", "results_each_image.csv"
    )
    if not os.path.isfile(ref_path):
        # Try alternative location
        ref_path = os.path.join(
            INVERSE_PHASE_REF_DIR, "PNipam2020 rotated", "outputs",
            "results_each_image.csv"
        )

    if not os.path.isfile(ref_path):
        print("  Reference CSV not found. Skipping comparison.")
        print("  Expected at: {}".format(ref_path))
        _print_summary(new)
        return

    ref = pd.read_csv(ref_path)
    ref["basename"] = ref["image path"].apply(lambda x: os.path.basename(str(x)))
    print("  Reference: {} rows".format(len(ref)))

    merged = new.merge(ref, left_on="image_name", right_on="basename", how="inner")
    print("  Merged: {} rows".format(len(merged)))

    # Compare ML predictions
    delta_df = pd.DataFrame()
    delta_df["image_name"] = merged["image_name"]
    delta_df["frame_number"] = merged["frame_number"]

    _compare_column(merged, delta_df, "ml_left_angle",
                    "model_v05.05 predicted left angle", "delta_ml_left")
    _compare_column(merged, delta_df, "ml_right_angle",
                    "model_v05.05 predicted right angle", "delta_ml_right")

    # Save
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    out_path = os.path.join(COMPARISON_DIR, "inverse_phase_comparison.csv")
    delta_df.to_csv(out_path, index=False)
    print("  Saved: {}".format(out_path))

    # Print report
    _print_delta_report(delta_df, ["delta_ml_left", "delta_ml_right"])


def compare_isaac():
    """Compare Isaac results against processed_data/results.csv for each dataset."""
    print()
    print("=" * 70)
    print("Isaac Comparison")
    print("=" * 70)

    datasets = ["MS_APTES_S2_0001", "0822_10k_Bb5", "SC69_1"]

    for dataset in datasets:
        results_path = os.path.join(
            RESULTS_DIR, "isaac_{}_results.csv".format(dataset)
        )
        if not os.path.isfile(results_path):
            print("  {} — results not found".format(dataset))
            continue

        new = pd.read_csv(results_path)
        print()
        print("  --- {} ---".format(dataset))
        print("  New results: {} rows".format(len(new)))

        # Look for reference
        ref_path = os.path.join(
            ISAAC_REF_DIR, dataset, "processed_data", "results.csv"
        )
        if os.path.isfile(ref_path):
            ref = pd.read_csv(ref_path)
            print("  Reference: {} rows".format(len(ref)))
            # Column names vary by dataset — print available columns
            print("  Reference columns: {}".format(list(ref.columns)[:10]))
        else:
            print("  Reference CSV not found at: {}".format(ref_path))

        # Look for commercial system data
        raw_dir = os.path.join(ISAAC_REF_DIR, dataset, "raw_data")
        if os.path.isdir(raw_dir):
            txt_files = [f for f in os.listdir(raw_dir) if f.endswith(".txt")]
            if txt_files:
                print("  Commercial data files: {}".format(txt_files))
                for txt_file in txt_files[:1]:  # Show first file columns
                    try:
                        raw = pd.read_csv(
                            os.path.join(raw_dir, txt_file), sep="\t",
                            encoding="latin-1"
                        )
                        ca_cols = [c for c in raw.columns if "CA" in c.upper()]
                        print("  Commercial CA columns: {}".format(ca_cols))
                    except Exception as e:
                        print("  Could not parse {}: {}".format(txt_file, e))

        _print_summary(new)


def compare_kristina():
    """Compare Kristina results against reference results.csv."""
    print()
    print("=" * 70)
    print("Kristina Comparison")
    print("=" * 70)

    results_path = os.path.join(
        RESULTS_DIR, "kristina_water_air_peek_60degrees_results.csv"
    )
    if not os.path.isfile(results_path):
        print("  Results not found: {}".format(results_path))
        return

    new = pd.read_csv(results_path)
    print("  New results: {} rows".format(len(new)))

    # Filter to usable frames
    if "skipped" in new.columns:
        usable = new[new["skipped"] != True]
        print("  Usable frames: {}".format(len(usable)))
    else:
        usable = new

    # Reference
    ref_path = os.path.join(
        KRISTINA_REF_DIR, "Water_air_PEEK_60degrees", "results.csv"
    )
    if os.path.isfile(ref_path):
        ref = pd.read_csv(ref_path)
        print("  Reference: {} rows".format(len(ref)))
        print("  Reference columns: {}".format(list(ref.columns)[:10]))
    else:
        print("  Reference CSV not found at: {}".format(ref_path))

    _print_summary(usable)

    # Tilt angle check
    if "detected_tilt_angle" in usable.columns:
        angles = usable["detected_tilt_angle"].dropna()
        if len(angles) > 0:
            print()
            print("  Tilt angle statistics:")
            print("    Mean:  {:.2f}".format(angles.mean()))
            print("    Std:   {:.2f}".format(angles.std()))
            print("    Range: [{:.2f}, {:.2f}]".format(angles.min(), angles.max()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compare_column(merged, delta_df, new_col, old_col, delta_name):
    """Compare a single column between new and reference."""
    if new_col in merged.columns and old_col in merged.columns:
        delta_df[delta_name] = merged[new_col] - merged[old_col]


def _print_delta_report(delta_df, delta_cols):
    """Print MAE and max delta for delta columns."""
    print()
    print("  {:<30} {:>10} {:>12} {:>6}".format("Column", "MAE", "Max|delta|", "n"))
    print("  " + "-" * 60)
    for col in delta_cols:
        if col not in delta_df.columns:
            continue
        vals = delta_df[col].dropna()
        if len(vals) == 0:
            continue
        mae = vals.abs().mean()
        maxd = vals.abs().max()
        print("  {:<30} {:10.4f} {:12.4f} {:6d}".format(col, mae, maxd, len(vals)))


def _print_summary(df):
    """Print a quick summary of angle results."""
    print()
    methods = ["tangent", "polynomial", "circular", "ellipse", "yl", "ml"]
    for method in methods:
        left_col = "{}_left_angle".format(method)
        right_col = "{}_right_angle".format(method)
        if left_col in df.columns:
            left_vals = df[left_col].dropna()
            right_vals = df[right_col].dropna() if right_col in df.columns else left_vals
            if len(left_vals) > 0:
                all_vals = pd.concat([left_vals, right_vals])
                print("  {:<12}: n={}, mean={:.1f}, std={:.1f}, range=[{:.1f}, {:.1f}]".format(
                    method, len(left_vals), all_vals.mean(), all_vals.std(),
                    all_vals.min(), all_vals.max()))


def main():
    print("=" * 70)
    print("Video Analysis Comparison")
    print("=" * 70)

    compare_inverse_phase()
    compare_isaac()
    compare_kristina()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
