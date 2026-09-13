"""Phase 7.1: diff the post-6.1 off-grid / on-grid-control result sets against
the pre-6.1 snapshots in results/pre_6.1/ (PART2_REMAINING_WORK.md, 7.1).

Same comparison rules as phase6_regression.py: contact points must match
exactly, angles are flagged when they move by more than ANGLE_TOLERANCE, and
each changed image is bucketed as better / worse by its mean-of-both-sides
ML absolute error against the filename ground truth. Timing columns are
ignored -- they never reproduce.

Run from OpenDrop_OP/paper_results/offgrid/:
    python phase7_offgrid_diff.py
Writes results/offgrid/phase7_offgrid_diff.csv (one row per changed image) and prints
a per-dataset summary.
"""

import os
import sys

import numpy as np
import pandas as pd

_PAPER_RESULTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PAPER_RESULTS_DIR)  # paper_results/, for config
from config import OFFGRID_RESULTS_DIR as RESULTS_DIR  # noqa: E402

PRE_DIR = os.path.join(RESULTS_DIR, "pre_6.1")
DATASETS = [
    "synthetic_offgrid_results.csv",
    "synthetic_ongrid_control_results.csv",
] + [f"synthetic_ongrid_control_replicate_{i}_results.csv" for i in range(5)]

ANGLE_TOLERANCE = 0.01
METHODS = ["tangent", "polynomial", "circular", "ellipse", "yl", "ml"]
CP_COLS = ["left_CP_x", "left_CP_y", "right_CP_x", "right_CP_y"]
ANGLE_COLS = [f"{m}_{s}_angle" for m in METHODS for s in ("left", "right")]


def ml_abs_error(row):
    if pd.isna(row.get("ml_left_angle")) or pd.isna(row.get("ml_right_angle")):
        return np.nan
    return (abs(row["ml_left_angle"] - row["gt_angle"]) + abs(row["ml_right_angle"] - row["gt_angle"])) / 2


def cp_error_px(row):
    try:
        return (np.hypot(row["left_CP_x"] - row["gt_cpl_x"], row["left_CP_y"] - row["gt_cpl_y"])
                + np.hypot(row["right_CP_x"] - row["gt_cpr_x"], row["right_CP_y"] - row["gt_cpr_y"])) / 2
    except (KeyError, TypeError):
        return np.nan


def diff_dataset(name):
    pre = pd.read_csv(os.path.join(PRE_DIR, name)).set_index("image_name")
    post = pd.read_csv(os.path.join(RESULTS_DIR, name)).set_index("image_name")
    label = name.replace("_results.csv", "")
    rows = []
    n_unchanged = n_better = n_worse = n_recovered = n_newly_failed = 0
    for img in sorted(set(pre.index) | set(post.index)):
        if img not in post.index:
            rows.append({"dataset": label, "image_name": img, "status": "missing_post"})
            continue
        if img not in pre.index:
            rows.append({"dataset": label, "image_name": img, "status": "missing_pre"})
            continue
        a, b = pre.loc[img], post.loc[img]
        a_err = a.get("error") if "error" in pre.columns else np.nan
        b_err = b.get("error") if "error" in post.columns else np.nan
        if pd.notna(a_err) and pd.isna(b_err):
            n_recovered += 1
            rows.append({"dataset": label, "image_name": img, "status": "recovered",
                         "pre_error": a_err, "post_ml_abs_error": ml_abs_error(b),
                         "post_cp_error_px": cp_error_px(b)})
            continue
        if pd.isna(a_err) and pd.notna(b_err):
            n_newly_failed += 1
            rows.append({"dataset": label, "image_name": img, "status": "newly_failed", "post_error": b_err})
            continue
        changed_cols = []
        for c in CP_COLS:
            if not (pd.isna(a[c]) and pd.isna(b[c])) and (pd.isna(a[c]) or pd.isna(b[c]) or int(a[c]) != int(b[c])):
                changed_cols.append(c)
        for c in ANGLE_COLS:
            if c not in pre.columns or c not in post.columns:
                continue
            if pd.isna(a[c]) and pd.isna(b[c]):
                continue
            if pd.isna(a[c]) or pd.isna(b[c]) or abs(float(a[c]) - float(b[c])) > ANGLE_TOLERANCE:
                changed_cols.append(c)
        if not changed_cols:
            n_unchanged += 1
            continue
        pre_e, post_e = ml_abs_error(a), ml_abs_error(b)
        better = bool(post_e < pre_e) if pd.notna(pre_e) and pd.notna(post_e) else None
        if better is True:
            n_better += 1
        elif better is False:
            n_worse += 1
        rows.append({
            "dataset": label, "image_name": img,
            "status": "better" if better is True else ("worse" if better is False else "changed"),
            "gt_angle": a["gt_angle"], "gt_scale": a["gt_scale"], "gt_roughness": a["gt_roughness"],
            "pre_ml_abs_error": pre_e, "post_ml_abs_error": post_e,
            "pre_cp_error_px": cp_error_px(a), "post_cp_error_px": cp_error_px(b),
            "changed_cols": ";".join(changed_cols),
        })
    print(f"{label:45s} unchanged={n_unchanged:4d} better={n_better:3d} worse={n_worse:3d} "
          f"recovered={n_recovered} newly_failed={n_newly_failed}")
    return rows


def main():
    all_rows = []
    for name in DATASETS:
        all_rows += diff_dataset(name)
    out = os.path.join(RESULTS_DIR, "phase7_offgrid_diff.csv")
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\n{len(all_rows)} rows written to {out}")
    if all_rows:
        df = pd.DataFrame(all_rows)
        cols = [c for c in ["dataset", "image_name", "status", "gt_angle", "gt_scale", "gt_roughness",
                            "pre_ml_abs_error", "post_ml_abs_error", "pre_cp_error_px", "post_cp_error_px"]
                if c in df.columns]
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
