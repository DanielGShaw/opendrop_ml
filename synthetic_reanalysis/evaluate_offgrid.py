"""Phase 3 full image-level evaluation (SYNTHETIC_DATA_REANALYSIS.md, Phase 3 + A.8).

Thin wrapper reusing paper_results/evaluate/evaluate_synthetic.py's
evaluate_image / parse_synthetic_filename against the off-grid,
on-grid-control, and replicate datasets built in Phase 1 -- no duplicated
fitting/ML logic. Checkpointed (append-mode CSV, flush after every image,
skips image_names already present on restart) so a crash partway through
doesn't cost the whole run, and provenance-stamped per A.8.

Run from OpenDrop_OP/synthetic_reanalysis/:
    python evaluate_offgrid.py                # all datasets
    python evaluate_offgrid.py offgrid         # just one (see DATASETS below)
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd

from config import (
    IMAGE_EXTENSIONS,
    OPENDROP_DIR,
    RESULTS_DIR,
    SYNTHETIC_OFFGRID_DATA_DIR,
    SYNTHETIC_ONGRID_CONTROL_DIR,
)
from spotcheck_pipeline import _import_evaluate_synthetic

SAVED_MODEL_PB = os.path.join(OPENDROP_DIR, "opendrop_ml", "modules", "ML_model", "saved_model.pb")


def _dataset_specs():
    """(label, image_dir, results_csv_name) for every dataset Phase 3 covers."""
    specs = [
        ("offgrid", SYNTHETIC_OFFGRID_DATA_DIR, "synthetic_offgrid_results.csv"),
        ("ongrid_control", SYNTHETIC_ONGRID_CONTROL_DIR, "synthetic_ongrid_control_results.csv"),
    ]
    for i in range(5):
        rep_dir = os.path.join(SYNTHETIC_ONGRID_CONTROL_DIR, f"replicate_{i}")
        specs.append((
            f"ongrid_control_replicate_{i}",
            rep_dir,
            f"synthetic_ongrid_control_replicate_{i}_results.csv",
        ))
    return specs


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(results_csv_path):
    import tensorflow as tf

    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=OPENDROP_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_head = None

    provenance = {
        "saved_model_pb_md5": _md5(SAVED_MODEL_PB) if os.path.exists(SAVED_MODEL_PB) else None,
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "cv2_version": cv2.__version__,
        "seed_scheme": (
            "matrix_utils.cell_seed(angle, bond, scaler, roughness) via "
            "np.random.seed() immediately before each draw_drop() call; "
            "replicates use cell_seed(...) + replicate_index (see A.3)"
        ),
        "git_head": git_head,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    sidecar_path = results_csv_path.replace(".csv", "_provenance.json")
    with open(sidecar_path, "w") as f:
        json.dump(provenance, f, indent=2)
    return sidecar_path


def run_dataset(es, model, label, image_dir, results_csv_name):
    results_path = os.path.join(RESULTS_DIR, results_csv_name)

    if not os.path.isdir(image_dir):
        print(f"[{label}] SKIP -- directory not found: {image_dir}")
        return

    images = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not images:
        print(f"[{label}] SKIP -- no images found in {image_dir}")
        return

    # Results are accumulated in memory and the *entire* CSV is rewritten
    # (via a temp file + atomic os.replace) after every image, rather than
    # appended row-by-row. A row-by-row append with header=False writes
    # values positionally, not by column name -- a failed image produces a
    # dict with only a handful of keys (image_name/gt_*/error), and
    # appending that short row under a 55-column header silently shifts
    # every value after the missing columns into the wrong column (found
    # in practice: an error string ended up in the `preprocessing_time`
    # column). Rebuilding the DataFrame from all accumulated dicts lets
    # pandas union the columns and pad missing ones with NaN correctly.
    all_rows = []
    done = set()
    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        all_rows = existing.to_dict("records")
        done = set(existing["image_name"].astype(str))
        print(f"[{label}] resuming: {len(done)}/{len(images)} already done in {results_path}")

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

        try:
            gt = es.parse_synthetic_filename(img_name)
        except ValueError as e:
            result = {"image_name": img_name, "error": f"Filename parse failed: {e}"}
        else:
            try:
                result = es.evaluate_image(img_path, model, gt)
            except Exception as e:
                result = {"image_name": img_name, "error": f"evaluate_image raised: {e}"}
                result.update(gt)

        all_rows.append(result)
        tmp_path = results_path + ".tmp"
        pd.DataFrame(all_rows).to_csv(tmp_path, index=False)
        os.replace(tmp_path, results_path)
        n_done_this_run += 1

    total_time = time.time() - t_start
    print(f"[{label}] done: {n_done_this_run} newly evaluated in {total_time / 60:.1f} min "
          f"(total {len(images)} in dataset). Results: {results_path}")

    prov_path = write_provenance(results_path)
    print(f"[{label}] provenance written: {prov_path}")


def main():
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser()
    all_labels = [s[0] for s in _dataset_specs()]
    parser.add_argument(
        "datasets", nargs="*", default=None,
        help=f"subset of datasets to run (default: all). Choices: {all_labels}",
    )
    args = parser.parse_args()

    specs = _dataset_specs()
    if args.datasets:
        specs = [s for s in specs if s[0] in args.datasets]
        missing = set(args.datasets) - {s[0] for s in specs}
        if missing:
            print(f"Unknown dataset(s): {missing}. Choices: {all_labels}")
            sys.exit(1)

    es = _import_evaluate_synthetic()

    print("Loading ML model...")
    model = tf.keras.models.load_model(es.ML_MODEL_DIR)
    print("Model loaded.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for label, image_dir, results_csv_name in specs:
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        run_dataset(es, model, label, image_dir, results_csv_name)


if __name__ == "__main__":
    main()
