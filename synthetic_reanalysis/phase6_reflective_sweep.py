"""Phase 6.1 Step 1b -- map the reflective-branch failure region with the
discriminants recorded (PART2_REMAINING_WORK.md, section 6.1 Step 1b).

Measurement only. Does NOT modify preprocessing.py and does not choose or
implement a fix -- it produces the table the fix is chosen from offline.

Sweep (Step 1b ranges):
  angle     20 .. 180 step 5                          (33 values)
  roughness -0.005 .. -0.080 step 0.005, plus 0.0,
            plus +0.002 .. +0.010 step 0.002         (22 values)
  scaler    {1.0, 2.83, 4.0}                          (3 values)
  bond      0.55 fixed
  = 2178 cells.  ~1 s/cell measured (rev. 14).

For each cell: render with draw_drop(dpi=64), seeded via matrix_utils.cell_seed
(same scheme as generate_offgrid_synthetic.py), run isolate_contour + CPID with
display=True (stdout captured to identify which branch fired), and the full
ML path for ml_left_angle/ml_right_angle.

What Step 1b adds over the rev. 5 run:
  * the four cascade discriminants (widest{1,2}_{top,bottom}_angle), computed
    directly from the contour by mirroring CPID's own code, and cross-checked
    against the values CPID prints in display mode (`disc_parse_ok`);
  * widest_points and CP_guesses coordinates, and each one's CP error vs
    ground truth -- so for every cell you can read off which choice the
    cascade *should* have made, without re-running anything;
  * size covariates for a scale-normalised threshold (option a'):
    gt_width_px, widest_width_px, apex_y, drop_height_px, contour_len
    (gt_* and drop_height_px use the GT baseline and are NOT deployable);
  * a complete branch classifier (rev. 5 missed "low angle", "assumed
    hydrophobic" and the Lfit path);
  * `ml_error` renamed `error_message` (Step 1a item 4).

Output: results/phase6_reflective_sweep_1b.csv (the rev. 5 map at
phase6_reflective_sweep.csv is left untouched for reference).

Checkpointed with the accumulate + atomic-rewrite pattern from
evaluate_offgrid.py (not a header-less to_csv(mode="a") -- see Known traps #1
in PART2_REMAINING_WORK.md).

Run from OpenDrop_OP/synthetic_reanalysis/:
    python phase6_reflective_sweep.py            # full sweep, resumable
    python phase6_reflective_sweep.py --smoke    # 4 cells -> results/phase6_reflective_sweep_smoke.csv
    python phase6_reflective_sweep.py --out phase6_reflective_sweep_postfix.csv   # after a preprocessing.py change
"""

import contextlib
import io
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")  # must precede any other matplotlib import -- CPID(display=True)
# below triggers plt.show() calls throughout preprocessing.py's display-mode branches,
# and draw_drop's own figure-to-buffer rendering also goes through pyplot; Agg keeps
# both headless instead of popping up a window that blocks the sweep on every cell.

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_files"))

from create_contours_args6 import draw_drop  # noqa: E402

from config import DPI, ML_MODEL_DIR, N_PTS_DROP, OPENDROP_DIR, RESULTS_DIR  # noqa: E402
from matrix_utils import cell_seed  # noqa: E402

sys.path.insert(0, OPENDROP_DIR)

from opendrop_ml.modules.preprocessing.preprocessing import (  # noqa: E402
    CPID,
    _find_point_index,
    calculate_contact_angle,
    check_for_high_angle,
    isolate_contour,
    linear_fit,
)
from opendrop_ml.modules.ML_model.prepare_experimental import (  # noqa: E402
    prepare_for_model,
    experimental_pred,
)

_out_name = "phase6_reflective_sweep_1b.csv"
if "--out" in sys.argv:  # e.g. --out phase6_reflective_sweep_postfix.csv (post-fix re-run)
    _out_name = sys.argv[sys.argv.index("--out") + 1]
OUT_CSV = os.path.join(RESULTS_DIR, _out_name)
SMOKE_CSV = os.path.join(RESULTS_DIR, "phase6_reflective_sweep_smoke.csv")

BOND = 0.55
ROUGHNESS_VALUES = (
    [round(-0.005 - 0.005 * i, 4) for i in range(16)]  # -0.005 .. -0.080 (reflective)
    + [0.0]
    + [round(0.002 * i, 4) for i in range(1, 6)]       # +0.002 .. +0.010 (jitter)
)
ANGLE_VALUES = list(range(20, 181, 5))  # 20 .. 180 (rev. 17: widened from 50)
SCALER_VALUES = [1.0, 2.83, 4.0]

# Every distinct print string the cascade in CPID() can emit, in the order
# they appear in preprocessing.py. First match wins. "lfit_coincident" is the
# path taken before the cascade when a CP guess equals a widest point -- CPID
# prints nothing branch-specific there, so it is identified by absence.
BRANCH_PATTERNS = [
    ("large angles under widest points - low angle reflective", "reflective_large_bottom"),
    ("angles on either side of widest points are similar - reflective", "reflective_similar"),
    ("kink at widest point - reflective", "reflective_kink"),
    ("looks reflective but is high angle", "high_angle_cp_guesses"),
    ("assumed low angle reflective", "reflective_lfit"),
    ("low angle - angles preceding", "low_angle"),
    ("maybe high angle, Lfit", "maybe_high_lfit"),
    ("assumed hydrophobic", "hydrophobic_cp_guesses"),
]

_PRECEDING_RE = re.compile(r"angles preceding widest point are ([-\d.e+nan]+), ([-\d.e+nan]+)")
_PROCEEDING_RE = re.compile(r"angles proceeding widest point are ([-\d.e+nan]+), ([-\d.e+nan]+)")

N_PTS = 10  # CPID's local-fit window (preprocessing.py `n_pts = 10`)


def _euclid(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _branch_from_stdout(text):
    for needle, label in BRANCH_PATTERNS:
        if needle in text:
            return label
    return "lfit_coincident"


def _parsed_discriminants(text):
    """The four angles as CPID printed them, or None if the branch printed none."""
    top = _PRECEDING_RE.search(text)
    bot = _PROCEEDING_RE.search(text)
    if not top:
        return None
    out = {"p_w1_top": float(top.group(1)), "p_w2_top": float(top.group(2))}
    if bot:
        out["p_w1_bot"] = float(bot.group(1))
        out["p_w2_bot"] = float(bot.group(2))
    return out


def compute_discriminants(contour):
    """Mirror of CPID()'s discriminant computation (preprocessing.py ~1724-1823),
    single-contour case only. Returns the values the cascade tests, plus the two
    candidate CP sets it chooses between."""
    CP_guesses, widest_points = check_for_high_angle(contour, display=False)
    out = {
        "wp_l_x": int(widest_points[0][0]), "wp_l_y": int(widest_points[0][1]),
        "wp_r_x": int(widest_points[1][0]), "wp_r_y": int(widest_points[1][1]),
        "cpg_l_x": int(CP_guesses[0][0]), "cpg_l_y": int(CP_guesses[0][1]),
        "cpg_r_x": int(CP_guesses[1][0]), "cpg_r_y": int(CP_guesses[1][1]),
        "cpg_equals_wp": bool(np.any(np.all(CP_guesses[:, None] == widest_points, axis=2))),
    }
    if len(contour) <= 110:
        return out  # two-contour case; not expected for these renders

    i1 = _find_point_index(contour, widest_points[0], "sweep: widest_points[0]")
    i2 = _find_point_index(contour, widest_points[1], "sweep: widest_points[1]")
    out["widest_index1"], out["widest_index2"] = int(i1), int(i2)

    w1_bot_m, _, _ = linear_fit(contour[i1 - N_PTS : i1])
    w1_top_m, _, _ = linear_fit(contour[i1 : i1 + N_PTS])
    w2_bot_m, _, _ = linear_fit(contour[i2 : i2 + N_PTS])
    w2_top_m, _, _ = linear_fit(contour[i2 - N_PTS : i2])

    out["widest1_bottom_angle"] = float(calculate_contact_angle(w1_bot_m, "left"))
    out["widest1_top_angle"] = float(calculate_contact_angle(w1_top_m, "left"))
    out["widest2_bottom_angle"] = float(calculate_contact_angle(w2_bot_m, "right"))
    out["widest2_top_angle"] = float(calculate_contact_angle(w2_top_m, "right"))
    # The `>79` test at preprocessing.py:1866 fires on either side, i.e. on the max.
    out["disc_left"] = 180.0 - out["widest1_bottom_angle"]
    out["disc_right"] = 180.0 - out["widest2_bottom_angle"]
    out["disc_max"] = max(out["disc_left"], out["disc_right"])
    return out


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
        row["gt_width_px"] = float(gr_x - gl_x)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, contour, _, _ = isolate_contour(img, DISPLAY=False, blur=True)
            CPs = CPID(contour, display=True)
        text = buf.getvalue()
        row["branch"] = _branch_from_stdout(text)
        row["contour_len"] = int(len(contour))
        row["apex_y"] = float(np.min(contour[:, 1]))
        row["drop_height_px"] = float(gl_y - row["apex_y"])  # GT-based; not deployable

        cpid_l = (int(CPs[0, 0]), int(CPs[0, 1]))
        cpid_r = (int(CPs[1, 0]), int(CPs[1, 1]))
        row["cpid_cpl_x"], row["cpid_cpl_y"] = cpid_l
        row["cpid_cpr_x"], row["cpid_cpr_y"] = cpid_r
        row["cp_error_left_px"] = _euclid(cpid_l, (gl_x, gl_y))
        row["cp_error_right_px"] = _euclid(cpid_r, (gr_x, gr_y))

        disc = compute_discriminants(contour)
        row.update(disc)
        row["widest_width_px"] = float(disc["wp_r_x"] - disc["wp_l_x"])
        row["wp_error_left_px"] = _euclid((disc["wp_l_x"], disc["wp_l_y"]), (gl_x, gl_y))
        row["wp_error_right_px"] = _euclid((disc["wp_r_x"], disc["wp_r_y"]), (gr_x, gr_y))
        row["cpg_error_left_px"] = _euclid((disc["cpg_l_x"], disc["cpg_l_y"]), (gl_x, gl_y))
        row["cpg_error_right_px"] = _euclid((disc["cpg_r_x"], disc["cpg_r_y"]), (gr_x, gr_y))

        parsed = _parsed_discriminants(text)
        if parsed is None or "widest1_top_angle" not in disc:
            row["disc_parse_ok"] = np.nan  # branch printed nothing to check against
        else:
            pairs = [
                (parsed["p_w1_top"], disc["widest1_top_angle"]),
                (parsed["p_w2_top"], disc["widest2_top_angle"]),
            ]
            if "p_w1_bot" in parsed:
                pairs += [
                    (parsed["p_w1_bot"], disc["widest1_bottom_angle"]),
                    (parsed["p_w2_bot"], disc["widest2_bottom_angle"]),
                ]
            row["disc_parse_ok"] = all(
                (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-6 for a, b in pairs
            )

        try:
            pred_ds = prepare_for_model(img, input_len=50, n_surface=10, display=False, blur=True)
            predictions, _ = experimental_pred(pred_ds, model, side="both")
            row["ml_left_angle"] = float(predictions[0])
            row["ml_right_angle"] = float(predictions[1])
        except Exception as e:
            row["error_message"] = str(e)

        row["status"] = "ok"
    except Exception as e:
        row["status"] = "error"
        row["error"] = str(e)

    return row


def _load_model():
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    print("Loading ML model...")
    model = tf.keras.models.load_model(ML_MODEL_DIR)
    print("Model loaded.")
    return model


def _summarise(df):
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("No successful cells.")
        return
    ok["cp_error_mean_px"] = ok[["cp_error_left_px", "cp_error_right_px"]].mean(axis=1)
    ok["reflective"] = ok["roughness"] < 0
    ok["high_angle"] = ok["angle"] >= 90

    print("\n--- cells by branch x (reflective, high_angle) ---")
    print(pd.crosstab(ok["branch"], [ok["reflective"], ok["high_angle"]]))

    print("\n--- CP error (px, left+right avg) by branch ---")
    print(ok.groupby("branch")["cp_error_mean_px"].agg(["count", "mean", "max"]).round(1))

    print("\n--- cells with CP error > 10 px, by branch ---")
    print(ok[ok["cp_error_mean_px"] > 10].groupby("branch").size())

    if "disc_parse_ok" in ok.columns:
        checked = ok["disc_parse_ok"].dropna()
        print(f"\ndiscriminant cross-check: {int(checked.sum())}/{len(checked)} cells match "
              f"CPID's printed values ({len(ok) - len(checked)} cells printed none)")
    print("\nNext: python phase6_separability.py")


def main():
    smoke = "--smoke" in sys.argv
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if smoke:
        # the three 6.1 reproduction cases plus one genuine high-angle non-reflective drop
        cells = [(80, BOND, 2.83, -0.045), (80, BOND, 2.83, -0.015),
                 (60, BOND, 2.83, -0.045), (150, BOND, 2.83, 0.0)]
    else:
        cells = [
            (angle, BOND, scaler, roughness)
            for roughness in ROUGHNESS_VALUES
            for angle in ANGLE_VALUES
            for scaler in SCALER_VALUES
        ]
    print(f"Sweep size: {len(cells)} cells")

    all_rows = []
    done = set()
    if not smoke and os.path.exists(OUT_CSV):
        existing = pd.read_csv(OUT_CSV)
        all_rows = existing.to_dict("records")
        done = {(r["angle"], r["bond"], r["scaler"], r["roughness"]) for r in all_rows}
        print(f"Resuming: {len(done)}/{len(cells)} already done")

    model = _load_model()

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
        n_done_this_run += 1

        if not smoke:
            tmp_path = OUT_CSV + ".tmp"
            pd.DataFrame(all_rows).to_csv(tmp_path, index=False)
            os.replace(tmp_path, OUT_CSV)

    total_time = time.time() - t_start
    print(f"Done: {n_done_this_run} newly evaluated in {total_time / 60:.1f} min "
          f"(total {len(cells)} in sweep). Results: {SMOKE_CSV if smoke else OUT_CSV}")

    df = pd.DataFrame(all_rows)
    if smoke:
        df.to_csv(SMOKE_CSV, index=False)
        cols = ["angle", "roughness", "branch", "cp_error_left_px", "wp_error_left_px",
                "cpg_error_left_px", "disc_max", "gt_width_px", "disc_parse_ok", "status"]
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(df[[c for c in cols if c in df.columns]].round(2))
        if "error" in df.columns:
            print(df["error"].dropna())
    _summarise(df)


if __name__ == "__main__":
    main()
