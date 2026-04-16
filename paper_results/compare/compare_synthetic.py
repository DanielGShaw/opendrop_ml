#!/usr/bin/env python
"""Compare new synthetic results against the reference CSV (list_synthetic_dpi64.csv).

Produces:
  - comparison/synthetic_comparison.csv  (per-image deltas)
  - Console report with per-method and per-bucket statistics

Run from OpenDrop_OP/:
    python paper_results/compare/compare_synthetic.py
"""

import sys
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_RESULTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PAPER_RESULTS_DIR)

from config import (
    SYNTHETIC_REF_CSV,
    RESULTS_DIR,
    COMPARISON_DIR,
    ANGLE_BUCKETS,
    ANGLE_BUCKET_LABELS,
)

# Column mapping: (new_prefix, old_column_pattern, fields)
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
    new = pd.read_csv(os.path.join(RESULTS_DIR, "synthetic_results.csv"))

    ref = pd.read_csv(SYNTHETIC_REF_CSV)
    ref["basename"] = ref["image path"].apply(lambda x: os.path.basename(str(x)))

    merged = new.merge(ref, left_on="image_name", right_on="basename", how="inner")
    return merged


def compute_deltas(merged):
    """Compute per-image deltas for all method pairs. Returns delta DataFrame."""
    delta_df = pd.DataFrame()
    delta_df["image_name"] = merged["image_name"]
    delta_df["gt_angle"] = merged["gt_angle"]
    delta_df["gt_bond"] = merged["gt_bond"]

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

    # Also include new-vs-GT errors for convenience
    for method in ["tangent", "circular", "ellipse", "yl", "ml"]:
        for side in ["left_angle", "right_angle"]:
            col = "{}_{}".format(method, side)
            if col in merged.columns:
                delta_df["error_{}_{}".format(method, side)] = merged[col] - merged["gt_angle"]

    return delta_df


def print_old_vs_new_report(merged):
    """Print per-method MAE of new vs old predictions."""
    gt = merged["gt_angle"]
    buckets = gt.apply(angle_bucket)

    print()
    print("=" * 90)
    print("PART 1: NEW vs OLD — Per-method angle deltas (new pipeline - old pipeline)")
    print("=" * 90)

    all_comparisons = []

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
    print("  {:<40} {:>10} {:>12} {:>6}".format("Comparison", "MAE", "Max|delta|", "n"))
    print("  " + "-" * 70)
    for label, mae, maxd, n in all_comparisons:
        print("  {:<40} {:10.4f} {:12.4f} {:6d}".format(label, mae, maxd, n))

    # Per-bucket breakdown (new vs old) for key comparisons
    print()
    print("  Per-bucket breakdown (buckets by GT angle):")

    key_pairs = [
        ("tangent", "left_angle", "Tangent fit", "left angle"),
        ("circular", "left_angle", "Circular fit", "left angle"),
        ("ellipse", "left_angle", "Ellipse fit", "left angle"),
        ("yl", "left_angle", "YL fit", "left angle"),
        ("ml", "left_angle", "model_v05.05", "predicted left angle"),
        ("ml", "left_angle", "model_v05_conan", "predicted left angle"),
    ]

    for new_pfx, new_suf, old_pfx, old_suf in key_pairs:
        new_col = "{}_{}".format(new_pfx, new_suf)
        old_col = "{} {}".format(old_pfx, old_suf)
        if new_col not in merged.columns or old_col not in merged.columns:
            continue
        label = "{} vs {}".format(new_pfx, old_pfx)
        print()
        print("    {}:".format(label))
        print("      {:<10} {:>10} {:>12} {:>6}".format("Bucket", "MAE", "Max|delta|", "n"))
        print("      " + "-" * 40)
        for blabel in ANGLE_BUCKET_LABELS:
            mask = buckets == blabel
            subset = merged.loc[mask, [new_col, old_col]].dropna()
            if subset.empty:
                continue
            d = (subset[new_col] - subset[old_col]).abs()
            print("      {:<10} {:10.4f} {:12.4f} {:6d}".format(
                blabel, d.mean(), d.max(), len(subset)))


def print_vs_gt_report(merged):
    """Print per-method MAE vs ground truth for both new and old pipelines."""
    gt = merged["gt_angle"]
    buckets = gt.apply(angle_bucket)

    print()
    print("=" * 90)
    print("PART 2: ACCURACY vs GROUND TRUTH — New pipeline vs old pipeline")
    print("=" * 90)

    # For each method, compare MAE vs GT for new and old
    comparisons = [
        ("Tangent", "tangent", "Tangent fit"),
        ("Circular", "circular", "Circular fit"),
        ("Ellipse", "ellipse", "Ellipse fit"),
        ("YL", "yl", "YL fit"),
        ("ML (new)", "ml", None),
        ("ML (v05.05)", None, "model_v05.05"),
        ("ML (conan)", None, "model_v05_conan"),
    ]

    print()
    print("  {:<18} {:>12} {:>12} {:>12} {:>12}".format(
        "Method", "New L MAE", "Old L MAE", "New R MAE", "Old R MAE"))
    print("  " + "-" * 68)

    for label, new_pfx, old_pfx in comparisons:
        vals = {}
        for side, side_old in [("left", "left"), ("right", "right")]:
            if new_pfx:
                new_col = "{}_{}_{}" .format(new_pfx, side, "angle")
                if new_col in merged.columns:
                    v = merged[new_col].dropna()
                    g = gt.loc[v.index]
                    vals["new_" + side] = (v - g).abs().mean()
            if old_pfx:
                if "model" in old_pfx:
                    old_col = "{} predicted {} angle".format(old_pfx, side_old)
                else:
                    old_col = "{} {} angle".format(old_pfx, side_old)
                if old_col in merged.columns:
                    v = merged[old_col].dropna()
                    g = gt.loc[v.index]
                    vals["old_" + side] = (v - g).abs().mean()

        nl = vals.get("new_left", np.nan)
        ol = vals.get("old_left", np.nan)
        nr = vals.get("new_right", np.nan)
        orr = vals.get("old_right", np.nan)

        def fmt(v):
            return "{:12.4f}".format(v) if pd.notna(v) else "{:>12}".format("—")
        print("  {:<18} {} {} {} {}".format(label, fmt(nl), fmt(ol), fmt(nr), fmt(orr)))

    # Per-bucket for ML new vs GT
    print()
    print("  ML (new pipeline) MAE vs GT by bucket:")
    print("    {:<10} {:>12} {:>12}".format("Bucket", "Left MAE", "Right MAE"))
    print("    " + "-" * 36)
    for blabel in ANGLE_BUCKET_LABELS:
        mask = buckets == blabel
        for side in ["left", "right"]:
            col = "ml_{}_angle".format(side)
            if col in merged.columns:
                subset = merged.loc[mask, [col, "gt_angle"]].dropna()
                if not subset.empty:
                    mae = (subset[col] - subset["gt_angle"]).abs().mean()
                    if side == "left":
                        left_mae = mae
                        n = len(subset)
                    else:
                        right_mae = mae
        print("    {:<10} {:12.4f} {:12.4f}  (n={})".format(blabel, left_mae, right_mae, n))


def flag_large_deviations(delta_df, threshold=1.0):
    """Print images where any old-vs-new angle delta exceeds threshold."""
    angle_delta_cols = [
        c for c in delta_df.columns
        if c.startswith("delta_") and "angle" in c
    ]

    flagged = []
    for _, row in delta_df.iterrows():
        for col in angle_delta_cols:
            val = row.get(col, np.nan)
            if pd.notna(val) and abs(val) > threshold:
                flagged.append((row["image_name"], row["gt_angle"], col, val))

    print()
    print("=" * 90)
    print("IMAGES WITH |old-vs-new delta| > {:.1f} degrees".format(threshold))
    print("=" * 90)

    if not flagged:
        print("  None.")
    else:
        # Summarize by image
        from collections import defaultdict
        by_image = defaultdict(list)
        for img, gt, col, val in flagged:
            by_image[(img, gt)].append((col, val))

        print("  {} flagged entries across {} unique images".format(
            len(flagged), len(by_image)))
        print()

        # Show top 10 by worst delta
        flagged_sorted = sorted(flagged, key=lambda x: abs(x[3]), reverse=True)
        show = min(20, len(flagged_sorted))
        print("  Top {} largest deviations:".format(show))
        print("  {:<55} {:>6} {:<30} {:>10}".format("Image", "GT", "Column", "Delta"))
        print("  " + "-" * 105)
        for img, gt, col, val in flagged_sorted[:show]:
            print("  {:<55} {:6.0f} {:<30} {:10.4f}".format(img, gt, col, val))

    # Count per method
    print()
    print("  Flagged count per comparison (|delta| > {:.1f}):".format(threshold))
    method_counts = {}
    for _, _, col, _ in flagged:
        method_counts[col] = method_counts.get(col, 0) + 1
    for col, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print("    {:<45} {:>5}".format(col, count))


def main():
    print("=" * 90)
    print("Synthetic Comparison: New Pipeline vs Reference CSV")
    print("=" * 90)

    merged = load_and_merge()
    print("Merged: {} rows".format(len(merged)))

    delta_df = compute_deltas(merged)

    # Save per-image deltas
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    out_path = os.path.join(COMPARISON_DIR, "synthetic_comparison.csv")
    delta_df.to_csv(out_path, index=False)
    print("Saved per-image deltas to: {}".format(out_path))

    print_old_vs_new_report(merged)
    print_vs_gt_report(merged)
    flag_large_deviations(delta_df, threshold=1.0)


if __name__ == "__main__":
    main()
