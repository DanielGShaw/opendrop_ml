"""Phase 6.4 regression harness (PART2_REMAINING_WORK.md, section 6.4).

Re-runs evaluate_image() over the 324 on-grid synthetic images and the 171
experimental images and compares the result against a frozen baseline.
Must be built and verified clean (0 changes) against unmodified code before
any change to preprocessing.py -- that is what proves the harness itself is
sound, not just that the code happens to be unchanged.

Baseline capture (do this once, before making any code change):
    python phase6_regression.py --capture-baseline

Then, after any change to preprocessing.py:
    python phase6_regression.py

The baseline is captured as a frozen copy under synthetic_reanalysis/results/
rather than read live from paper_results/results/, because Phase 7 later
overwrites paper_results/results/*.csv with improved numbers (per the ground
rules) -- the frozen copy keeps this harness meaningful after that happens.

Run from OpenDrop_OP/synthetic_reanalysis/:
    python phase6_regression.py --capture-baseline
    python phase6_regression.py
    python phase6_regression.py synthetic       # just one dataset
"""

import argparse
import os
import shutil
import sys
import time

import cv2
import numpy as np
import pandas as pd

from config import PAPER_RESULTS_DIR, RESULTS_DIR

# --- Baseline (frozen snapshot of paper_results/results at capture time) ---
BASELINE_SYNTHETIC_CSV = os.path.join(RESULTS_DIR, "phase6_baseline_synthetic.csv")
BASELINE_EXPERIMENTAL_CSV = os.path.join(RESULTS_DIR, "phase6_baseline_experimental.csv")

STORED_SYNTHETIC_CSV = os.path.join(PAPER_RESULTS_DIR, "results", "synthetic_results.csv")
STORED_EXPERIMENTAL_CSV = os.path.join(PAPER_RESULTS_DIR, "results", "experimental_results.csv")

# --- This run's checkpointed output (accumulate + atomic rewrite, per the
# known trap: header-less to_csv(mode="a") silently misaligns columns on a
# short error-row) ---
RUN_SYNTHETIC_CSV = os.path.join(RESULTS_DIR, "phase6_regression_synthetic_run.csv")
RUN_EXPERIMENTAL_CSV = os.path.join(RESULTS_DIR, "phase6_regression_experimental_run.csv")

CP_COLUMNS = ["left_CP_x", "left_CP_y", "right_CP_x", "right_CP_y"]
METHOD_PREFIXES = ["tangent", "polynomial", "circular", "ellipse", "yl", "ml"]
ANGLE_COLUMNS = [f"{p}_{side}_angle" for p in METHOD_PREFIXES for side in ("left", "right")]
ANGLE_TOLERANCE = 0.01  # degrees; flag any change strictly greater than this

GT_FOR_CP = {
    "left_CP_x": "gt_cpl_x", "left_CP_y": "gt_cpl_y",
    "right_CP_x": "gt_cpr_x", "right_CP_y": "gt_cpr_y",
}
GT_FOR_ANGLE_SIDE = {"left": "gt_angle", "right": "gt_angle"}


def _import_evaluate_module(module_name):
    """Import a paper_results/evaluate/*.py module.

    Both synthetic_reanalysis/config.py and paper_results/config.py are
    imported as bare `config`. If ours is already cached in sys.modules
    (it is -- this file imports it above), that stale cache would satisfy
    paper_results modules' `from config import (...)` regardless of
    sys.path order. Evict it for the duration of this one import, then
    restore it. (Same workaround as spotcheck_pipeline._import_evaluate_synthetic.)
    """
    saved_config = sys.modules.pop("config", None)
    eval_dir = os.path.join(PAPER_RESULTS_DIR, "evaluate")
    sys.path.insert(0, PAPER_RESULTS_DIR)
    sys.path.insert(0, eval_dir)
    try:
        module = __import__(module_name)
    finally:
        sys.modules.pop("config", None)
        if saved_config is not None:
            sys.modules["config"] = saved_config
    return module


def capture_baseline():
    for src, dst in [
        (STORED_SYNTHETIC_CSV, BASELINE_SYNTHETIC_CSV),
        (STORED_EXPERIMENTAL_CSV, BASELINE_EXPERIMENTAL_CSV),
    ]:
        if not os.path.exists(src):
            print(f"SKIP -- not found: {src}")
            continue
        shutil.copy2(src, dst)
        print(f"Captured baseline: {src} -> {dst}")


def _run_dataset(label, image_dir, run_csv_path, evaluate_fn, is_synthetic, es_module=None):
    """Run evaluate_fn over every image in image_dir, checkpointed."""
    if not os.path.isdir(image_dir):
        print(f"[{label}] SKIP -- directory not found: {image_dir}")
        return None

    images = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif"))
    )
    if not images:
        print(f"[{label}] SKIP -- no images found in {image_dir}")
        return None

    all_rows = []
    done = set()
    if os.path.exists(run_csv_path):
        existing = pd.read_csv(run_csv_path)
        all_rows = existing.to_dict("records")
        done = set(existing["image_name"].astype(str))
        print(f"[{label}] resuming: {len(done)}/{len(images)} already done")

    t_start = time.time()
    n_done_this_run = 0
    n_remaining = len(images) - len(done)

    for i, img_name in enumerate(images):
        if img_name in done:
            continue

        img_path = os.path.join(image_dir, img_name)
        elapsed = time.time() - t_start
        eta_str = ""
        if n_done_this_run > 0:
            eta = elapsed / n_done_this_run * (n_remaining - n_done_this_run)
            eta_str = f"  ETA: {eta / 60:.1f} min"
        print(f"  [{label} {i + 1}/{len(images)}]{eta_str} {img_name}")

        if is_synthetic:
            try:
                gt = es_module.parse_synthetic_filename(img_name)
            except ValueError as e:
                result = {"image_name": img_name, "error": f"Filename parse failed: {e}"}
            else:
                try:
                    result = evaluate_fn(img_path, gt)
                except Exception as e:
                    result = {"image_name": img_name, "error": f"evaluate_image raised: {e}"}
                    result.update(gt)
        else:
            try:
                result = evaluate_fn(img_path)
            except Exception as e:
                result = {"image_name": img_name, "error": f"evaluate_image raised: {e}"}

        all_rows.append(result)
        tmp_path = run_csv_path + ".tmp"
        pd.DataFrame(all_rows).to_csv(tmp_path, index=False)
        os.replace(tmp_path, run_csv_path)
        n_done_this_run += 1

    total_time = time.time() - t_start
    print(f"[{label}] done: {n_done_this_run} newly evaluated in {total_time / 60:.1f} min "
          f"(total {len(images)} in dataset)")
    return pd.DataFrame(all_rows)


def _compare(baseline_df, run_df, has_gt, label):
    """Per-image, per-column comparison. Returns (records, any_regression)."""
    baseline_df = baseline_df.set_index("image_name")
    run_df = run_df.set_index("image_name")

    common = sorted(set(baseline_df.index) & set(run_df.index))
    missing_from_run = sorted(set(baseline_df.index) - set(run_df.index))
    new_in_run = sorted(set(run_df.index) - set(baseline_df.index))
    if missing_from_run:
        print(f"[{label}] WARNING: {len(missing_from_run)} baseline images missing from this run")
    if new_in_run:
        print(f"[{label}] WARNING: {len(new_in_run)} images in this run not present in baseline")

    columns_to_check = [c for c in (CP_COLUMNS + ANGLE_COLUMNS) if c in baseline_df.columns]

    records = []
    any_regression = False

    for img in common:
        b_row = baseline_df.loc[img]
        r_row = run_df.loc[img]

        for col in columns_to_check:
            b_val = b_row[col] if col in b_row.index else np.nan
            r_val = r_row[col] if col in r_row.index else np.nan

            b_nan = pd.isna(b_val)
            r_nan = pd.isna(r_val)

            if b_nan and r_nan:
                status = "unchanged"
            elif b_nan != r_nan:
                status = "changed"
            elif col in CP_COLUMNS:
                status = "unchanged" if int(b_val) == int(r_val) else "changed"
            else:  # angle column
                status = "unchanged" if abs(float(b_val) - float(r_val)) <= ANGLE_TOLERANCE else "changed"

            if status == "unchanged":
                continue

            bucket = "changed"
            if has_gt and not (b_nan or r_nan):
                gt_col = GT_FOR_CP.get(col)
                if gt_col is None:
                    side = "left" if col.startswith(tuple(f"{p}_left" for p in METHOD_PREFIXES)) else "right"
                    gt_col = GT_FOR_ANGLE_SIDE[side]
                if gt_col in b_row.index and not pd.isna(b_row.get(gt_col, np.nan)):
                    gt_val = float(b_row[gt_col])
                    b_err = abs(float(b_val) - gt_val)
                    r_err = abs(float(r_val) - gt_val)
                    if r_err < b_err:
                        bucket = "changed-and-better"
                    elif r_err > b_err:
                        bucket = "changed-and-worse"
                    else:
                        bucket = "changed-neutral"

            if bucket in ("changed-and-worse", "changed") :
                # Without GT (experimental) we cannot prove a change is safe,
                # so treat any change there as a regression risk too.
                any_regression = True
            if bucket == "changed" and not has_gt:
                bucket = "changed-unverifiable"

            records.append({
                "dataset": label, "image_name": img, "column": col,
                "baseline": b_val, "run": r_val, "bucket": bucket,
            })

    return records, any_regression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-baseline", action="store_true")
    parser.add_argument("datasets", nargs="*", default=None,
                         help="subset to run: synthetic, experimental")
    args = parser.parse_args()

    if args.capture_baseline:
        capture_baseline()
        return

    run_synthetic = not args.datasets or "synthetic" in args.datasets
    run_experimental = not args.datasets or "experimental" in args.datasets

    os.makedirs(RESULTS_DIR, exist_ok=True)

    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    all_records = []
    any_regression = False
    t_start = time.time()

    if run_synthetic:
        if not os.path.exists(BASELINE_SYNTHETIC_CSV):
            print(f"ERROR: no baseline captured at {BASELINE_SYNTHETIC_CSV}. "
                  f"Run with --capture-baseline first.")
            sys.exit(2)
        es = _import_evaluate_module("evaluate_synthetic")
        print("Loading ML model...")
        model = tf.keras.models.load_model(es.ML_MODEL_DIR)
        print(f"\n{'=' * 60}\nsynthetic\n{'=' * 60}")
        run_df = _run_dataset(
            "synthetic", es.SYNTHETIC_DATA_DIR, RUN_SYNTHETIC_CSV,
            lambda p, gt: es.evaluate_image(p, model, gt),
            is_synthetic=True, es_module=es,
        )
        baseline_df = pd.read_csv(BASELINE_SYNTHETIC_CSV)
        records, reg = _compare(baseline_df, run_df, has_gt=True, label="synthetic")
        all_records += records
        any_regression = any_regression or reg

    if run_experimental:
        if not os.path.exists(BASELINE_EXPERIMENTAL_CSV):
            print(f"ERROR: no baseline captured at {BASELINE_EXPERIMENTAL_CSV}. "
                  f"Run with --capture-baseline first.")
            sys.exit(2)
        ee = _import_evaluate_module("evaluate_experimental")
        print("Loading ML model...")
        model = tf.keras.models.load_model(ee.ML_MODEL_DIR)
        print(f"\n{'=' * 60}\nexperimental\n{'=' * 60}")
        run_df = _run_dataset(
            "experimental", ee.EXPERIMENTAL_DATA_DIR, RUN_EXPERIMENTAL_CSV,
            lambda p: ee.evaluate_image(p, model),
            is_synthetic=False,
        )
        baseline_df = pd.read_csv(BASELINE_EXPERIMENTAL_CSV)
        records, reg = _compare(baseline_df, run_df, has_gt=False, label="experimental")
        all_records += records
        any_regression = any_regression or reg

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time / 60:.1f} min")

    diff_df = pd.DataFrame(all_records)
    diff_csv = os.path.join(RESULTS_DIR, "phase6_regression_diff.csv")
    diff_df.to_csv(diff_csv, index=False)
    print(f"Diff written: {diff_csv} ({len(diff_df)} changed cells)")

    print("\n--- Summary ---")
    if diff_df.empty:
        print("0 changes. Harness clean.")
    else:
        print(diff_df.groupby(["dataset", "bucket"]).size())
        n_images_changed = diff_df.groupby("dataset")["image_name"].nunique()
        print("\nDistinct images with >=1 changed column:")
        print(n_images_changed)

    if any_regression:
        print("\nREGRESSION DETECTED.")
        sys.exit(1)
    else:
        print("\nNo regressions.")
        sys.exit(0)


if __name__ == "__main__":
    main()
