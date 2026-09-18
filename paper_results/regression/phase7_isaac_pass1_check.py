"""Phase 7.1 follow-up: why did isaac_MS_APTES_S2_0001's right contact line
move by 1 px when the paper pipeline was regenerated after the 6.1 fix?

The Isaac two-pass pipeline takes its contact lines from the *median* of
CPID's per-frame output over the Pass-1 sample (every ``median_step``-th
frame). Re-runs Pass 1 on one Isaac dataset with the reflective kink branch
enabled (deployed) and disabled (pre-6.1 behaviour, via the module constant
``KINK_MIN_DEG`` -- the same switch phase6_figures.py uses), and reports
per-frame CP differences, which branch fired, and the resulting medians.

Run from OpenDrop_OP/:
    python paper_results/regression/phase7_isaac_pass1_check.py [dataset]
Writes paper_results/regression/phase7_isaac_pass1_<dataset>.csv.
"""

import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

_PAPER_RESULTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PAPER_RESULTS_DIR)  # paper_results/, for config
from config import ISAAC_DIR, REGRESSION_DIR as RESULTS_DIR, OPENDROP_DIR  # noqa: E402

sys.path.insert(0, OPENDROP_DIR)
sys.path.insert(0, os.path.join(_PAPER_RESULTS_DIR, "evaluate"))
import opendrop_ml.modules.preprocessing.preprocessing as pp  # noqa: E402
from opendrop_ml.modules.contact_angle.video_two_pass_needle_in import (  # noqa: E402
    compute_pass1_calibration,
    process_frame_pass1,
)
from opendrop_ml.modules.image.frame_loader import load_frames  # noqa: E402
from evaluate_isaac import ISAAC_DATASETS, find_video_file  # noqa: E402

BRANCH_STRINGS = {
    "kink at widest point - reflective": "reflective_kink",
    "large angles under widest points": "low_angle_reflective",
    "looks reflective but is high angle": "high_angle_via_79",
    "maybe high angle": "maybe_high_lfit",
}


def _branch(stdout):
    for needle, label in BRANCH_STRINGS.items():
        if needle in stdout:
            return label
    return "none"


def run_pass1(video_path, cfg, kink_min_deg):
    saved = pp.KINK_MIN_DEG
    pp.KINK_MIN_DEG = kink_min_deg
    rows, results = [], []
    try:
        for frame, frame_num, _meta in load_frames(video_path, cfg["start_sec"], cfg["end_sec"], cfg["median_step"]):
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    r = process_frame_pass1(frame)
            except Exception as e:  # counted as a failed sample, as in the pipeline
                rows.append({"frame": frame_num, "status": f"error: {e}"})
                continue
            if r is None:
                rows.append({"frame": frame_num, "status": "none"})
                continue
            results.append(r)
            rows.append({
                "frame": frame_num, "status": "ok", "branch": _branch(buf.getvalue()),
                "left_x": r["left_CP"][0], "left_y": r["left_CP"][1],
                "right_x": r["right_CP"][0], "right_y": r["right_CP"][1],
            })
    finally:
        pp.KINK_MIN_DEG = saved
    calib = compute_pass1_calibration(results) if len(results) >= 3 else None
    return pd.DataFrame(rows), calib, results


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "MS_APTES_S2_0001"
    cfg = ISAAC_DATASETS[name]
    video_path = find_video_file(os.path.join(ISAAC_DIR, name))
    print(f"{name}: {video_path}")

    after, calib_after, res_after = run_pass1(video_path, cfg, pp.KINK_MIN_DEG)
    before, calib_before, res_before = run_pass1(video_path, cfg, 1e9)

    df = after.merge(before, on="frame", suffixes=("_after", "_before"))
    for side in ("left", "right"):
        df[f"{side}_dx"] = df[f"{side}_x_after"] - df[f"{side}_x_before"]
        df[f"{side}_dy"] = df[f"{side}_y_after"] - df[f"{side}_y_before"]
    out = os.path.join(RESULTS_DIR, f"phase7_isaac_pass1_{name}.csv")
    df.to_csv(out, index=False)

    ok = df[df["status_after"] == "ok"]
    changed = ok[(ok["left_dx"] != 0) | (ok["left_dy"] != 0) | (ok["right_dx"] != 0) | (ok["right_dy"] != 0)]
    print(f"\nPass-1 frames: {len(df)} sampled, {len(ok)} ok; {len(changed)} with a CP change")
    print("branch counts, after :", ok["branch_after"].value_counts().to_dict())
    print("branch counts, before:", ok["branch_before"].value_counts().to_dict())
    if len(changed):
        cols = ["frame", "branch_before", "branch_after", "left_x_before", "left_y_before", "left_x_after", "left_y_after",
                "right_x_before", "right_y_before", "right_x_after", "right_y_after"]
        with pd.option_context("display.width", 220):
            print(changed[cols].to_string(index=False))
    for label, res in (("before", res_before), ("after", res_after)):
        l = np.array([r["left_CP"] for r in res]); rr = np.array([r["right_CP"] for r in res])
        print(f"{label}: median left CP ({np.median(l[:,0]):.1f}, {np.median(l[:,1]):.1f}); "
              f"median right CP ({np.median(rr[:,0]):.1f}, {np.median(rr[:,1]):.1f}); "
              f"right_y values: {sorted(pd.Series(rr[:,1]).value_counts().to_dict().items())}")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
