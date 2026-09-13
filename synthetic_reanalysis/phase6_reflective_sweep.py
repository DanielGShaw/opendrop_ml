"""Phase 6.1 Step 1 -- map the reflective-branch failure region
(PART2_REMAINING_WORK.md, section 6.1 Step 1).

Measurement only. Does NOT modify preprocessing.py and does not choose or
implement a fix -- it produces the map the fix is chosen from.

Sweep: roughness in [-0.080, -0.005] step 0.005 (16 values) x angle in
[50, 110] step 5 (13 values) x scaler in {1.0, 2.83, 4.0} (3 values),
bond fixed at 0.55. 624 renders.

For each cell: render with draw_drop(dpi=64), seeded via matrix_utils.cell_seed
(same scheme as generate_offgrid_synthetic.py), run isolate_contour + CPID
with display=True (stdout captured to identify which branch fired), and the
full evaluate_image ML path for ml_left_angle/ml_right_angle.

Checkpointed with the accumulate + atomic-rewrite pattern from
evaluate_offgrid.py (not a header-less to_csv(mode="a") -- see Known traps #1
in PART2_REMAINING_WORK.md).

Run from OpenDrop_OP/synthetic_reanalysis/:
    python phase6_reflective_sweep.py
"""

import contextlib
import io
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # must precede any other matplotlib import -- CPID(display=True)
# below triggers plt.show() calls throughout preprocessing.py's display-mode branches,
# and draw_drop's own figure-to-buffer rendering also goes through pyplot; Agg keeps
# both headless instead of popping up a window that blocks the sweep on every cell.

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_files"))

from create_contours_args6 import draw_drop  # noqa: E402

from config import DPI, ML_MODEL_DIR, N_PTS_DROP, OPENDROP_DIR, RESULTS_DIR  # noqa: E402
from matrix_utils import cell_seed  # noqa: E402

sys.path.insert(0, OPENDROP_DIR)

from opendrop_ml.modules.preprocessing.preprocessing import isolate_contour, CPID  # noqa: E402
from opendrop_ml.modules.ML_model.prepare_experimental import (  # noqa: E402
    prepare_for_model,
    experimental_pred,
)

OUT_CSV = os.path.join(RESULTS_DIR, "phase6_reflective_sweep.csv")

BOND = 0.55
ROUGHNESS_VALUES = [round(-0.005 - 0.005 * i, 4) for i in range(16)]  # -0.005 .. -0.080
ANGLE_VALUES = list(range(50, 111, 5))  # 50 .. 110
SCALER_VALUES = [1.0, 2.83, 4.0]


def _euclid(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _branch_from_stdout(text):
    lines = [l for l in text.split("\n") if "reflect" in l.lower() or "high angle" in l.lower()]
    return lines[0] if lines else "none"


def evaluate_cell(angle, bond, scaler, roughness, model):
    row = {"angle": angle, "bond": bond, "scaler": scaler, "roughness": roughness}
    seed = cell_seed(angle, bond, scaler, roughness)
    row["seed"] = seed

    try:
        np.random.seed(seed)  # immediately before draw_drop, per A.3 / Known trap context
        img, CPs_img = draw_drop(angle, bond, scaler, roughness, N_PTS_DROP, dpi=DPI)
        (gl_x, gl_y), (gr_x, gr_y) = CPs_img
        row["gt_cpl_x"], row["gt_cpl_y"] = gl_x, gl_y
        row["gt_cpr_x"], row["gt_cpr_y"] = gr_x, gr_y

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, contour, _, _ = isolate_contour(img, DISPLAY=False, blur=True)
            CPs = CPID(contour, display=True)
        row["branch"] = _branch_from_stdout(buf.getvalue())

        cpid_l = (int(CPs[0, 0]), int(CPs[0, 1]))
        cpid_r = (int(CPs[1, 0]), int(CPs[1, 1]))
        row["cpid_cpl_x"], row["cpid_cpl_y"] = cpid_l
        row["cpid_cpr_x"], row["cpid_cpr_y"] = cpid_r
        row["cp_error_left_px"] = _euclid(cpid_l, (gl_x, gl_y))
        row["cp_error_right_px"] = _euclid(cpid_r, (gr_x, gr_y))

        try:
            pred_ds = prepare_for_model(img, input_len=50, n_surface=10, display=False, blur=True)
            predictions, _ = experimental_pred(pred_ds, model, side="both")
            row["ml_left_angle"] = float(predictions[0])
            row["ml_right_angle"] = float(predictions[1])
        except Exception as e:
            row["ml_error"] = str(e)

        row["status"] = "ok"
    except Exception as e:
        row["status"] = "error"
        row["error"] = str(e)

    return row


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cells = [
        (angle, BOND, scaler, roughness)
        for roughness in ROUGHNESS_VALUES
        for angle in ANGLE_VALUES
        for scaler in SCALER_VALUES
    ]
    print(f"Sweep size: {len(cells)} cells")

    all_rows = []
    done = set()
    if os.path.exists(OUT_CSV):
        existing = pd.read_csv(OUT_CSV)
        all_rows = existing.to_dict("records")
        done = {(r["angle"], r["bond"], r["scaler"], r["roughness"]) for r in all_rows}
        print(f"Resuming: {len(done)}/{len(cells)} already done")

    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    print("Loading ML model...")
    model = tf.keras.models.load_model(ML_MODEL_DIR)
    print("Model loaded.")

    t_start = time.time()
    n_done_this_run = 0
    n_remaining = len(cells) - len(done)

    for i, (angle, bond, scaler, roughness) in enumerate(cells):
        key = (angle, bond, scaler, roughness)
        if key in done:
            continue

        elapsed = time.time() - t_start
        eta_str = ""
        if n_done_this_run > 0:
            eta = elapsed / n_done_this_run * (n_remaining - n_done_this_run)
            eta_str = f"  ETA: {eta / 60:.1f} min"
        print(f"  [{i + 1}/{len(cells)}]{eta_str} angle={angle} scaler={scaler} roughness={roughness}")

        row = evaluate_cell(angle, bond, scaler, roughness, model)
        all_rows.append(row)

        tmp_path = OUT_CSV + ".tmp"
        pd.DataFrame(all_rows).to_csv(tmp_path, index=False)
        os.replace(tmp_path, OUT_CSV)
        n_done_this_run += 1

    total_time = time.time() - t_start
    print(f"Done: {n_done_this_run} newly evaluated in {total_time / 60:.1f} min "
          f"(total {len(cells)} in sweep). Results: {OUT_CSV}")

    df = pd.DataFrame(all_rows)
    ok = df[df["status"] == "ok"].copy()
    if not ok.empty:
        ok["branch"] = ok["branch"].fillna("none")
        print("\n--- branch x roughness x angle (scaler=2.83) ---")
        sub = ok[ok["scaler"] == 2.83]
        pivot = sub.pivot_table(
            index="roughness", columns="angle", values="branch",
            aggfunc=lambda s: s.iloc[0] if len(s) else "?",
        )
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(pivot)

        print("\n--- mean CP error (px, left+right avg) by branch ---")
        ok["cp_error_mean_px"] = ok[["cp_error_left_px", "cp_error_right_px"]].mean(axis=1)
        print(ok.groupby("branch")["cp_error_mean_px"].agg(["count", "mean", "max"]))
    else:
        print("No successful cells.")


if __name__ == "__main__":
    main()
