#!/usr/bin/env python
"""Diagnostic-only: fingerprint isolate_contour()'s intermediate steps over the
experimental dataset, so results from two platforms can be compared offline
to find the first step at which they diverge.

It monkeypatches cv2.findContours, cluster_OPTICS, cv2.HoughLines /
HoughCircles, hough_baseline / hough_circle and cv2.threshold to record
call-level counts and hashes, and sets preprocessing._ISOLATE_CONTOUR_DEBUG_HOOK
to log each crop-boundary decision inside isolate_contour(). This is how the
Windows int32 overflow in line_circle_intersection() was located.

Run from OpenDrop_OP/ (the smoke-test workflow runs it with diagnostics=true):
    python paper_results/evaluate/diagnose_contour_platform.py

Output, all in paper_results/results/ (gitignored):
    contour_diag_<platform>.csv, optics_diag_<platform>.csv,
    hough_diag_<platform>.csv, hough_selected_diag_<platform>.csv,
    threshold_diag_<platform>.csv  -- one row per instrumented call
    bounds_diag_<platform>.jsonl   -- one line per isolate_contour() decision
"""

import csv
import hashlib
import json
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
_current_image = {
    "name": None,
    "call_index": 0,
    "optics_call_index": 0,
    "hough_call_index": 0,
    "threshold_call_index": 0,
}

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


_hough_records = []
_real_HoughLines = cv2.HoughLines
_real_HoughCircles = cv2.HoughCircles


def _instrumented_HoughLines(image, *args, **kwargs):
    lines = _real_HoughLines(image, *args, **kwargs)
    n = 0 if lines is None else len(lines)
    first = None
    full_hash = None
    if lines is not None and n > 0:
        rho, theta = lines[0][0]
        first = f"rho={rho:.4f},theta={theta:.6f}"
        # Round before hashing: rho/theta are float32, and we only care
        # whether the *set* of detected lines differs meaningfully, not
        # about noise below Hough's own quantization.
        rounded = np.round(lines.reshape(-1, 2).astype(np.float64), 4)
        full_hash = hashlib.md5(rounded.tobytes()).hexdigest()[:12]
    _hough_records.append(
        {
            "image_name": _current_image["name"],
            "call_index": _current_image["hough_call_index"],
            "kind": "HoughLines",
            "n_results": n,
            "first_result": first,
            "full_hash": full_hash,
        }
    )
    _current_image["hough_call_index"] += 1
    return lines


def _instrumented_HoughCircles(image, *args, **kwargs):
    circles = _real_HoughCircles(image, *args, **kwargs)
    n = 0 if circles is None else circles.shape[1]
    first = None
    full_hash = None
    if circles is not None and n > 0:
        x, y, r = circles[0, 0]
        first = f"x={x:.4f},y={y:.4f},r={r:.4f}"
        rounded = np.round(circles.reshape(-1, 3).astype(np.float64), 4)
        full_hash = hashlib.md5(rounded.tobytes()).hexdigest()[:12]
    _hough_records.append(
        {
            "image_name": _current_image["name"],
            "call_index": _current_image["hough_call_index"],
            "kind": "HoughCircles",
            "n_results": n,
            "first_result": first,
            "full_hash": full_hash,
        }
    )
    _current_image["hough_call_index"] += 1
    return circles


_threshold_records = []
_real_threshold = cv2.threshold


def _instrumented_threshold(src, thresh, maxval, type_, *args, **kwargs):
    ret, dst = _real_threshold(src, thresh, maxval, type_, *args, **kwargs)
    is_otsu = bool(type_ & cv2.THRESH_OTSU)
    dst_hash = hashlib.md5(dst.tobytes()).hexdigest()[:12]
    _threshold_records.append(
        {
            "image_name": _current_image["name"],
            "call_index": _current_image["threshold_call_index"],
            "is_otsu": is_otsu,
            "input_shape": str(src.shape),
            "ret": ret,
            "output_hash": dst_hash,
        }
    )
    _current_image["threshold_call_index"] += 1
    return ret, dst


_real_selected = {}


def _make_instrumented_hough_baseline(real_hough_baseline):
    def _instrumented(edges, *args, **kwargs):
        result = real_hough_baseline(edges, *args, **kwargs)
        line = np.asarray(result).flatten()
        _hough_selected_records.append(
            {
                "image_name": _current_image["name"],
                "kind": "hough_baseline",
                "value": ",".join(f"{v:.4f}" for v in line),
            }
        )
        return result

    return _instrumented


def _make_instrumented_hough_circle(real_hough_circle):
    def _instrumented(edges, display):
        center, radius = real_hough_circle(edges, display)
        _hough_selected_records.append(
            {
                "image_name": _current_image["name"],
                "kind": "hough_circle",
                "value": f"center={center},radius={radius}",
            }
        )
        return center, radius

    return _instrumented


_hough_selected_records = []


def _jsonify(value):
    """Make a value from isolate_contour()'s debug hook JSON-serializable,
    summarizing large point arrays down to a hash + count instead of
    dumping every point."""
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.shape[1] == 2 and len(value) > 8:
            sorted_pts = value[np.lexsort((value[:, 1], value[:, 0]))]
            return {
                "n_points": len(value),
                "set_hash": hashlib.md5(
                    sorted_pts.astype(np.int32).tobytes()
                ).hexdigest()[:12],
            }
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


_bounds_debug_records = []


def _bounds_debug_hook(event, kwargs):
    record = {"image_name": _current_image["name"], "event": event}
    record.update({k: _jsonify(v) for k, v in kwargs.items()})
    _bounds_debug_records.append(record)


def main():
    cv2.findContours = _instrumented_findContours
    cv2.HoughLines = _instrumented_HoughLines
    cv2.HoughCircles = _instrumented_HoughCircles
    cv2.threshold = _instrumented_threshold

    import opendrop_ml.modules.preprocessing.preprocessing as preprocessing

    preprocessing.cluster_OPTICS = _make_instrumented_cluster_OPTICS(
        preprocessing.cluster_OPTICS
    )
    preprocessing.hough_baseline = _make_instrumented_hough_baseline(
        preprocessing.hough_baseline
    )
    preprocessing.hough_circle = _make_instrumented_hough_circle(
        preprocessing.hough_circle
    )
    preprocessing._ISOLATE_CONTOUR_DEBUG_HOOK = _bounds_debug_hook
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
        _current_image["hough_call_index"] = 0
        _current_image["threshold_call_index"] = 0
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

    hough_out_path = os.path.join(RESULTS_DIR, f"hough_diag_{plat}.csv")
    hough_fieldnames = [
        "image_name",
        "call_index",
        "kind",
        "n_results",
        "first_result",
        "full_hash",
    ]
    with open(hough_out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=hough_fieldnames)
        writer.writeheader()
        for record in _hough_records:
            writer.writerow(record)

    print(f"Wrote {len(_hough_records)} Hough call records to {hough_out_path}")

    hough_selected_out_path = os.path.join(RESULTS_DIR, f"hough_selected_diag_{plat}.csv")
    hough_selected_fieldnames = ["image_name", "kind", "value"]
    with open(hough_selected_out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=hough_selected_fieldnames)
        writer.writeheader()
        for record in _hough_selected_records:
            writer.writerow(record)

    print(
        f"Wrote {len(_hough_selected_records)} selected-baseline/circle records "
        f"to {hough_selected_out_path}"
    )

    threshold_out_path = os.path.join(RESULTS_DIR, f"threshold_diag_{plat}.csv")
    threshold_fieldnames = [
        "image_name",
        "call_index",
        "is_otsu",
        "input_shape",
        "ret",
        "output_hash",
    ]
    with open(threshold_out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=threshold_fieldnames)
        writer.writeheader()
        for record in _threshold_records:
            writer.writerow(record)

    print(f"Wrote {len(_threshold_records)} threshold() call records to {threshold_out_path}")

    bounds_out_path = os.path.join(RESULTS_DIR, f"bounds_diag_{plat}.jsonl")
    with open(bounds_out_path, "w") as f:
        for record in _bounds_debug_records:
            f.write(json.dumps(record) + "\n")

    print(
        f"Wrote {len(_bounds_debug_records)} isolate_contour() internal debug events "
        f"to {bounds_out_path}"
    )


if __name__ == "__main__":
    main()
