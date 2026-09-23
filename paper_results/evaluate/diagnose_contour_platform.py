#!/usr/bin/env python
"""Diagnostic-only: instrument every cv2.findContours() call made while running
isolate_contour() over the experimental dataset, to pin down exactly where
macOS/Windows results diverge (see ML_MODEL_INTEGRATION_PLAN.md Phase 7.3).

Disabling cv2's optimized/IPP code paths (OPENDROP_DISABLE_CV2_OPTIMIZATION=1
in evaluate_experimental.py) made zero difference to any output column, so
IPP is ruled out. This script instead monkeypatches cv2.findContours to
record, for every call made during isolate_contour() (there are normally two:
once on the raw image, once again after tilt-correction/cropping), the raw
contour count, the arc lengths of the top 3 candidates, and the bounding box
/ border-touch status of whichever one gets selected (sorted by arc length,
descending -- see extract_edges_CV in modules/preprocessing/preprocessing.py).

Run from OpenDrop_OP/:
    python paper_results/evaluate/diagnose_contour_platform.py

Output: paper_results/results/contour_diag_<platform>.csv, one row per
findContours() call (image_name, call_index, n_contours, top-3 arc lengths,
chosen contour's point count / bbox / border-touch flag). Compare the macOS
and Windows CSVs offline to see at which call, for which images, the raw
contour set itself first differs.
"""

import csv
import hashlib
import os
import platform
import sys

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_RESULTS_DIR = os.path.dirname(SCRIPT_DIR)
OPENDROP_DIR = os.path.dirname(PAPER_RESULTS_DIR)
sys.path.insert(0, OPENDROP_DIR)
sys.path.insert(0, PAPER_RESULTS_DIR)

from config import EXPERIMENTAL_DATA_DIR, RESULTS_DIR, IMAGE_EXTENSIONS

_records = []
_current_image = {"name": None, "call_index": 0, "optics_call_index": 0}

_real_findContours = cv2.findContours


def _instrumented_findContours(image, mode, method, *args, **kwargs):
    contours, hierarchy = _real_findContours(image, mode, method, *args, **kwargs)

    height, width = image.shape[:2]
    lengths = [cv2.arcLength(c, False) for c in contours]
    order = sorted(range(len(contours)), key=lambda i: lengths[i], reverse=True)

    top3 = [(i, round(lengths[i], 3)) for i in order[:3]]

    record = {
        "image_name": _current_image["name"],
        "call_index": _current_image["call_index"],
        "n_contours": len(contours),
        "image_width": width,
        "image_height": height,
    }
    for rank, (idx, length) in enumerate(top3):
        c = contours[idx].reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(contours[idx])
        touches_border = x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height
        # Two hashes to separate "different points" from "same points, different
        # trace order/direction": order_hash is sensitive to point order, the
        # sorted set_hash is not.
        order_hash = hashlib.md5(c.astype(np.int32).tobytes()).hexdigest()[:12]
        sorted_pts = c[np.lexsort((c[:, 1], c[:, 0]))]
        set_hash = hashlib.md5(sorted_pts.astype(np.int32).tobytes()).hexdigest()[:12]
        record[f"rank{rank}_arclength"] = length
        record[f"rank{rank}_npoints"] = len(c)
        record[f"rank{rank}_bbox"] = f"({x},{y})-({x + w},{y + h})"
        record[f"rank{rank}_touches_border"] = touches_border
        record[f"rank{rank}_order_hash"] = order_hash
        record[f"rank{rank}_set_hash"] = set_hash

    _records.append(record)
    _current_image["call_index"] += 1

    return contours, hierarchy


_optics_records = []


def _make_instrumented_cluster_OPTICS(real_cluster_OPTICS):
    def _instrumented(sample, out_style="coords", xi=None, eps=None, verbose=0):
        groups = real_cluster_OPTICS(sample, out_style, xi, eps, verbose)
        sizes = {k: len(v) for k, v in groups.items()}
        maxkey = max(sizes, key=sizes.get)
        chosen = np.array(groups[maxkey])
        sorted_pts = chosen[np.lexsort((chosen[:, 1], chosen[:, 0]))]
        set_hash = hashlib.md5(sorted_pts.astype(np.int32).tobytes()).hexdigest()[:12]
        _optics_records.append(
            {
                "image_name": _current_image["name"],
                "call_index": _current_image["optics_call_index"],
                "n_input_points": len(sample),
                "n_groups": len(groups),
                "chosen_group_size": sizes[maxkey],
                "chosen_group_set_hash": set_hash,
            }
        )
        _current_image["optics_call_index"] += 1
        return groups

    return _instrumented


def main():
    cv2.findContours = _instrumented_findContours

    import opendrop_ml.modules.preprocessing.preprocessing as preprocessing

    preprocessing.cluster_OPTICS = _make_instrumented_cluster_OPTICS(
        preprocessing.cluster_OPTICS
    )
    isolate_contour = preprocessing.isolate_contour

    images = sorted(
        f
        for f in os.listdir(EXPERIMENTAL_DATA_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    print(f"Found {len(images)} images in {EXPERIMENTAL_DATA_DIR}")

    for i, img_name in enumerate(images):
        img_path = os.path.join(EXPERIMENTAL_DATA_DIR, img_name)
        _current_image["name"] = img_name
        _current_image["call_index"] = 0
        _current_image["optics_call_index"] = 0
        print(f"  [{i + 1}/{len(images)}] {img_name}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"    Failed to load image, skipping")
            continue

        try:
            isolate_contour(img, DISPLAY=False, blur=True)
        except Exception as e:
            print(f"    isolate_contour raised: {e}")

    plat = "windows" if os.name == "nt" else platform.system().lower()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"contour_diag_{plat}.csv")

    fieldnames = [
        "image_name",
        "call_index",
        "n_contours",
        "image_width",
        "image_height",
    ]
    for rank in range(3):
        fieldnames += [
            f"rank{rank}_arclength",
            f"rank{rank}_npoints",
            f"rank{rank}_bbox",
            f"rank{rank}_touches_border",
            f"rank{rank}_order_hash",
            f"rank{rank}_set_hash",
        ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in _records:
            writer.writerow(record)

    print(f"Wrote {len(_records)} findContours() call records to {out_path}")

    optics_out_path = os.path.join(RESULTS_DIR, f"optics_diag_{plat}.csv")
    optics_fieldnames = [
        "image_name",
        "call_index",
        "n_input_points",
        "n_groups",
        "chosen_group_size",
        "chosen_group_set_hash",
    ]
    with open(optics_out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=optics_fieldnames)
        writer.writeheader()
        for record in _optics_records:
            writer.writerow(record)

    print(f"Wrote {len(_optics_records)} cluster_OPTICS() call records to {optics_out_path}")


if __name__ == "__main__":
    main()
