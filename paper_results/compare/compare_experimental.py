#!/usr/bin/env python
"""Compare new experimental results against the reference CSV (list_full.csv).

Produces:
  - comparison/experimental_comparison.csv  (per-image deltas)
  - Console report with per-method and per-bucket statistics

Run from OpenDrop_OP/:
    python paper_results/compare/compare_experimental.py
"""

import sys
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_RESULTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PAPER_RESULTS_DIR)

from config import (
    EXPERIMENTAL_REF_CSV,
    RESULTS_DIR,
    COMPARISON_DIR,
    ANGLE_BUCKETS,
    ANGLE_BUCKET_LABELS,
)

# Column mapping: (new_prefix, old_column_pattern, sides)
# "sides" is a list of (new_suffix, old_suffix) pairs
METHOD_MAP = [
    ("tangent", "Tangent fit", [
        ("left_angle", "left angle"),
        ("right_angle", "right angle"),
        ("fit_time", "fit time"),
        ("RMSE", "RMSE"),
        ("max_error", "max error"),
    ]),
    ("polynomial", "Polynomial fit", [
        ("left_angle", "left angle"),
        ("right_angle", "right angle"),
        ("fit_time", "fit time"),
        ("RMSE", "RMSE"),
        ("max_error", "max error"),
    ]),
    ("circular", "Circular fit", [
        ("left_angle", "left angle"),
        ("right_angle", "right angle"),
        ("fit_time", "fit time"),
        ("RMSE", "RMSE"),
        ("max_error", "max error"),
    ]),
    ("ellipse", "Ellipse fit", [
        ("left_angle", "left angle"),
        ("right_angle", "right angle"),
        ("fit_time", "fit time"),
        ("RMSE", "RMSE"),
        ("max_error", "max error"),
    ]),
    ("yl", "YL fit", [
        ("left_angle", "left angle"),
        ("right_angle", "right angle"),
        ("fit_time", "fit time"),
        ("RMSE", "RMSE"),
        ("max_error", "max error"),
        ("bond_number", "Bond number"),
    ]),
]

ML_MAP = [
    ("ml", "model_v05.05", [
        ("left_angle", "predicted left angle"),
        ("right_angle", "predicted right angle"),
    ]),
    ("ml", "model_v05_conan", [
        ("left_angle", "predicted left angle"),
        ("right_angle", "predicted right angle"),
    ]),
]


def angle_bucket(angle):
    """Classify angle into bucket label."""
    for (lo, hi), label in zip(ANGLE_BUCKETS, ANGLE_BUCKET_LABELS):
        if lo <= angle < hi:
            return label
    return ">110"


def load_and_merge():
    """Load new results and reference CSV, merge on basename."""
    new = pd.read_csv(os.path.join(RESULTS_DIR, "experimental_results.csv"))

    ref = pd.read_csv(EXPERIMENTAL_REF_CSV)
    ref["basename"] = ref["image path"].apply(lambda x: os.path.basename(str(x)))

    # Deduplicate: keep row with model_v05_conan values populated
    ref_dedup = (
        ref.sort_values("model_v05_conan predicted left angle", na_position="last")
        .drop_duplicates(subset="basename", keep="first")
    )

    merged = new.merge(ref_dedup, left_on="image_name", right_on="basename", how="inner")
    return merged, len(ref), len(ref_dedup)


def compute_deltas(merged):
    """Compute per-image deltas for all method pairs. Returns delta DataFrame."""
    delta_df = pd.DataFrame()
    delta_df["image_name"] = merged["image_name"]
    delta_df["yl_left_angle"] = merged["yl_left_angle"]
    delta_df["yl_right_angle"] = merged["yl_right_angle"]

    # Classical methods
    for new_pfx, old_pfx, cols in METHOD_MAP:
        for new_suf, old_suf in cols:
            new_col = "{}_{}".format(new_pfx, new_suf)
            old_col = "{} {}".format(old_pfx, old_suf)
            if new_col in merged.columns and old_col in merged.columns:
                delta_col = "delta_{}_{}".format(new_pfx, new_suf)
                delta_df[delta_col] = merged[new_col] - merged[old_col]

    # ML comparisons
    for new_pfx, old_pfx, cols in ML_MAP:
        for new_suf, old_suf in cols:
            new_col = "{}_{}".format(new_pfx, new_suf)
            old_col = "{} {}".format(old_pfx, old_suf)
            if new_col in merged.columns and old_col in merged.columns:
                tag = old_pfx.replace(".", "").replace("model_", "")
                delta_col = "delta_{}_vs_{}_{}".format(new_pfx, tag, new_suf)
                delta_df[delta_col] = merged[new_col] - merged[old_col]

    return delta_df


def print_angle_report(merged, delta_df):
    """Print per-method and per-bucket MAE for angle columns."""
    # Use YL angle as the bucket reference (best available for experimental)
    avg_yl = (merged["yl_left_angle"] + merged["yl_right_angle"]) / 2.0
    buckets = avg_yl.apply(angle_bucket)

    print()
    print("=" * 90)
    print("PER-METHOD ANGLE DELTAS (new - old)")
    print("=" * 90)

    all_comparisons = []

    # Classical methods — angle columns only
    for new_pfx, old_pfx, cols in METHOD_MAP:
        for new_suf, old_suf in cols:
            if "angle" not in new_suf:
                continue
            new_col = "{}_{}".format(new_pfx, new_suf)
            old_col = "{} {}".format(old_pfx, old_suf)
            if new_col not in merged.columns or old_col not in merged.columns:
                continue
            valid = merged[[new_col, old_col]].dropna()
            if valid.empty:
                continue
            deltas = (valid[new_col] - valid[old_col]).abs()
            label = "{} {}".format(new_pfx, new_suf)
            all_comparisons.append((label, deltas.mean(), deltas.max(), len(valid)))

    # ML comparisons
    for new_pfx, old_pfx, cols in ML_MAP:
        for new_suf, old_suf in cols:
            new_col = "{}_{}".format(new_pfx, new_suf)
            old_col = "{} {}".format(old_pfx, old_suf)
            if new_col not in merged.columns or old_col not in merged.columns:
                continue
            valid = merged[[new_col, old_col]].dropna()
            if valid.empty:
                continue
            deltas = (valid[new_col] - valid[old_col]).abs()
            label = "{} vs {} {}".format(new_pfx, old_pfx, new_suf)
            all_comparisons.append((label, deltas.mean(), deltas.max(), len(valid)))

    print()
    print("  {:<35} {:>10} {:>12} {:>6}".format("Comparison", "MAE", "Max|delta|", "n"))
    print("  " + "-" * 65)
    for label, mae, maxd, n in all_comparisons:
        flag = " ***" if maxd > 1.0 else ""
        print("  {:<35} {:10.4f} {:12.4f} {:6d}{}".format(label, mae, maxd, n, flag))

    # Per-bucket breakdown for key comparisons
    print()
    print("=" * 90)
    print("PER-BUCKET BREAKDOWN (buckets by avg YL angle)")
    print("=" * 90)

    key_pairs = [
        ("tangent", "left_angle", "Tangent fit", "left angle"),
        ("circular", "left_angle", "Circular fit", "left angle"),
        ("ellipse", "left_angle", "Ellipse fit", "left angle"),
        ("yl", "left_angle", "YL fit", "left angle"),
        ("ml", "left_angle", "model_v05.05", "predicted left angle"),
    ]

    for new_pfx, new_suf, old_pfx, old_suf in key_pairs:
        new_col = "{}_{}".format(new_pfx, new_suf)
        old_col = "{} {}".format(old_pfx, old_suf)
        if new_col not in merged.columns or old_col not in merged.columns:
            continue
        label = "{} vs {}".format(new_pfx, old_pfx)
        print()
        print("  {}:".format(label))
        print("    {:<10} {:>10} {:>12} {:>6}".format("Bucket", "MAE", "Max|delta|", "n"))
        print("    " + "-" * 40)
        for blabel in ANGLE_BUCKET_LABELS:
            mask = buckets == blabel
            subset = merged.loc[mask, [new_col, old_col]].dropna()
            if subset.empty:
                continue
            d = (subset[new_col] - subset[old_col]).abs()
            print("    {:<10} {:10.4f} {:12.4f} {:6d}".format(blabel, d.mean(), d.max(), len(subset)))


def flag_large_deviations(merged, delta_df, threshold=1.0):
    """Print images where any angle delta exceeds threshold."""
    angle_delta_cols = [c for c in delta_df.columns if "angle" in c and c.startswith("delta_")]

    flagged = []
    for _, row in delta_df.iterrows():
        for col in angle_delta_cols:
            val = row.get(col, np.nan)
            if pd.notna(val) and abs(val) > threshold:
                flagged.append((row["image_name"], col, val))

    print()
    print("=" * 90)
    print("IMAGES WITH |delta| > {:.1f} degrees".format(threshold))
    print("=" * 90)

    if not flagged:
        print("  None.")
    else:
        print("  {:<40} {:<40} {:>10}".format("Image", "Column", "Delta"))
        print("  " + "-" * 90)
        for img, col, val in sorted(flagged, key=lambda x: abs(x[2]), reverse=True):
            print("  {:<40} {:<40} {:10.4f}".format(img, col, val))
    print()
    print("  Total flagged: {} entries across {} unique images".format(
        len(flagged), len(set(f[0] for f in flagged))))


def main():
    print("=" * 90)
    print("Experimental Comparison: New Pipeline vs Reference CSV")
    print("=" * 90)

    merged, n_ref, n_dedup = load_and_merge()
    print("Reference CSV: {} rows -> {} after dedup".format(n_ref, n_dedup))
    print("Merged: {} rows".format(len(merged)))

    delta_df = compute_deltas(merged)

    # Save per-image deltas
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    out_path = os.path.join(COMPARISON_DIR, "experimental_comparison.csv")
    delta_df.to_csv(out_path, index=False)
    print("Saved per-image deltas to: {}".format(out_path))

    print_angle_report(merged, delta_df)
    flag_large_deviations(merged, delta_df, threshold=1.0)


if __name__ == "__main__":
    main()
