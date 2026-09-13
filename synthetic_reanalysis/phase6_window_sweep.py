"""Phase 6.1 Step 2 prototyping -- does a scale-aware fit window (or an
upper-vs-lower mirror score) separate reflective sub-90 drops from genuine
high-angle drops where the cascade's fixed 10-point fits cannot?

Session 1 found the cascade's `180 - widest_bottom_angle` statistic saturated
at exactly 90 in 923/1272 decision cells: linear_fit() returns a vertical line
whenever the x-spread is < 1e-6, and the widest point is by construction the
x-extremum, so 10 rows around it at theta ~ 80 deg span ~2 px horizontally and
quantise to identical x. This script recomputes the same top/bottom angles with
longer windows -- fixed lengths and fractions of the widest-point span -- and
adds a per-side mirror score about the widest-point row, then reports which
window/feature best separates the two classes.

Measurement only; preprocessing.py is not modified.

Images: loaded from disk where the cell exists in paper_results/data/synthetic
or synthetic_reanalysis/data/synthetic_offgrid (same PNGs the deployed pipeline
reads), rendered via draw_drop otherwise (bit-equivalent -- both come from the
same seeded generator, the PNGs are lossless).

Run from OpenDrop_OP/synthetic_reanalysis/:
    python phase6_window_sweep.py
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_files"))
from create_contours_args6 import draw_drop  # noqa: E402
from config import (  # noqa: E402
    DPI, N_PTS_DROP, OPENDROP_DIR, RESULTS_DIR,
    SYNTHETIC_OFFGRID_DATA_DIR, SYNTHETIC_ONGRID_DATA_DIR,
)
from matrix_utils import cell_seed  # noqa: E402
sys.path.insert(0, OPENDROP_DIR)
from opendrop_ml.modules.preprocessing.preprocessing import (  # noqa: E402
    _find_point_index, calculate_contact_angle, check_for_high_angle, isolate_contour, linear_fit,
)
from phase6_separability import best_threshold  # noqa: E402

OUT_CSV = os.path.join(RESULTS_DIR, "phase6_window_sweep.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "phase6_window_sweep_summary.csv")

BOND = 0.55
ANGLES = [70, 75, 80, 85, 90, 95, 100, 110, 130, 150]
SCALERS = [1.0, 2.83, 4.0]
ROUGHNESS = [-0.06, -0.045, -0.03, -0.015, 0.0, 0.006]
PREFER_MARGIN_PX = 3.0

FIXED_WINDOWS = [10, 15, 20, 30, 40, 60]
FRAC_WINDOWS = [0.025, 0.05, 0.10, 0.15, 0.20, 0.30]  # of the widest-to-widest index span
MIRROR_BANDS = [0.10, 0.20, 0.30]  # rows each side of the widest-point row, as frac of widest width


def _find_saved(angle, bond, scaler, roughness):
    for d in (SYNTHETIC_ONGRID_DATA_DIR, SYNTHETIC_OFFGRID_DATA_DIR):
        for pat in (f"{angle:.1f}_{bond:.4f}_{scaler:.4f}_{roughness:.4f}_*",
                    f"{angle}_{bond}_{scaler}_{roughness}_*", f"{angle:.1f}_{bond}_{scaler}_{roughness}_*"):
            hits = glob.glob(os.path.join(d, pat))
            if hits:
                return hits[0]
    return None


def _gt_from_name(path):
    parts = os.path.basename(path).split("_")
    i = parts.index("CPL")
    return (int(parts[i + 1]), int(parts[i + 2])), (int(parts[i + 4]), int(parts[i + 5]))


def load_or_render(angle, bond, scaler, roughness):
    path = _find_saved(angle, bond, scaler, roughness)
    if path:
        return cv2.imread(path), _gt_from_name(path), "disk"
    np.random.seed(cell_seed(angle, bond, scaler, roughness))
    img, cps = draw_drop(angle, bond, scaler, roughness, N_PTS_DROP, dpi=DPI)
    return img, (tuple(cps[0]), tuple(cps[1])), "render"


def angles_at(contour, i1, i2, n):
    n = max(3, int(n))
    w1_bot, _, _ = linear_fit(contour[max(0, i1 - n): i1])
    w1_top, _, _ = linear_fit(contour[i1: i1 + n])
    w2_bot, _, _ = linear_fit(contour[i2: i2 + n])
    w2_top, _, _ = linear_fit(contour[max(0, i2 - n): i2])
    return (calculate_contact_angle(w1_top, "left"), calculate_contact_angle(w1_bot, "left"),
            calculate_contact_angle(w2_top, "right"), calculate_contact_angle(w2_bot, "right"))


def mirror_score(contour, wp, band_px, side):
    """Mean |x_upper(y0 - d) - x_lower(y0 + d)| / band over d in [1, band], for one side.
    Low => the contour below the widest-point row mirrors the contour above it."""
    x0, y0 = wp
    pts = contour[np.abs(contour[:, 0] - x0) < band_px * 3]  # this side only
    rows = {}
    for x, y in pts:
        d = int(y - y0)
        if abs(d) > band_px or d == 0:
            continue
        # per row keep the outermost point (mirrors find_rightmost/leftmost)
        if d not in rows or (x < rows[d] if side == "left" else x > rows[d]):
            rows[d] = x
    diffs = [abs(rows[d] - rows[-d]) for d in range(1, int(band_px) + 1) if d in rows and -d in rows]
    return float(np.mean(diffs) / band_px) if len(diffs) >= 3 else np.nan


def evaluate_cell(angle, bond, scaler, roughness):
    row = {"angle": angle, "bond": bond, "scaler": scaler, "roughness": roughness}
    img, (gl, gr), src = load_or_render(angle, bond, scaler, roughness)
    row["source"] = src
    _, contour, _, _ = isolate_contour(img, DISPLAY=False, blur=True)
    cpg, wp = check_for_high_angle(contour, display=False)
    i1 = _find_point_index(contour, wp[0], "window sweep")
    i2 = _find_point_index(contour, wp[1], "window sweep")
    span = i2 - i1
    width = float(wp[1][0] - wp[0][0])
    row.update({"widest_width_px": width, "index_span": int(span)})

    e = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))
    row["wp_err"] = (e(wp[0], gl) + e(wp[1], gr)) / 2
    row["cpg_err"] = (e(cpg[0], gl) + e(cpg[1], gr)) / 2
    row["prefer"] = ("widest_points" if row["wp_err"] + PREFER_MARGIN_PX < row["cpg_err"]
                     else "CP_guesses" if row["cpg_err"] + PREFER_MARGIN_PX < row["wp_err"] else "either")

    for n in FIXED_WINDOWS:
        t1, b1, t2, b2 = angles_at(contour, i1, i2, n)
        row[f"bot_max_n{n}"] = max(b1, b2)
        row[f"kink_max_n{n}"] = max(b1 - t1, b2 - t2)  # reflective kink: bottom exceeds top by ~2(90-theta)
    for f in FRAC_WINDOWS:
        n = span * f
        t1, b1, t2, b2 = angles_at(contour, i1, i2, n)
        row[f"bot_max_f{f}"] = max(b1, b2)
        row[f"kink_max_f{f}"] = max(b1 - t1, b2 - t2)
    for f in MIRROR_BANDS:
        band = width * f
        row[f"mirror_f{f}"] = np.nanmean([mirror_score(contour, wp[0], band, "left"),
                                          mirror_score(contour, wp[1], band, "right")])
    return row


def main():
    cells = [(a, BOND, s, r) for r in ROUGHNESS for a in ANGLES for s in SCALERS]
    print(f"{len(cells)} cells")
    rows = []
    for i, c in enumerate(cells):
        try:
            rows.append(evaluate_cell(*c))
        except Exception as ex:
            rows.append({"angle": c[0], "bond": c[1], "scaler": c[2], "roughness": c[3], "error": str(ex)})
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(cells)}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    ok = df[df.get("error").isna()] if "error" in df.columns else df
    print(f"{len(ok)} ok ({(df['source'] == 'disk').sum()} from disk, "
          f"{(df['source'] == 'render').sum()} rendered)")

    lab = ok[ok["prefer"] != "either"]
    print(f"\nlabelled cells: {len(lab)}  (want widest_points: {(lab.prefer == 'widest_points').sum()}, "
          f"want CP_guesses: {(lab.prefer == 'CP_guesses').sum()})")

    feats = [c for c in ok.columns if c.startswith(("bot_max_", "kink_max_", "mirror_"))]
    out = []
    for feat in feats:
        sat = float((np.abs(lab[feat] - 90) < 1e-6).mean()) if feat.startswith("bot_max") else np.nan
        for sc in ["pooled"] + SCALERS:
            sub = lab if sc == "pooled" else lab[lab.scaler == sc]
            sub = sub.dropna(subset=[feat])
            r = best_threshold(sub[feat], sub["prefer"])
            r.update({"feature": feat, "scaler": sc, "n": len(sub), "frac_saturated_90": sat})
            out.append(r)
    summary = pd.DataFrame(out)[["feature", "scaler", "n", "n_wp", "n_cpg", "rule", "best_T",
                                 "min_errors", "separable", "frac_saturated_90"]]
    summary.to_csv(SUMMARY_CSV, index=False)

    print("\n--- pooled: misassignments by feature (0 = clean separation) ---")
    pooled = summary[summary.scaler == "pooled"].sort_values("min_errors")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(pooled[["feature", "n", "rule", "best_T", "min_errors", "frac_saturated_90"]]
              .round(3).to_string(index=False))
    print("\n--- per scaler, features that separate cleanly ---")
    clean = summary[(summary.scaler != "pooled") & summary.separable]
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(clean[["feature", "scaler", "rule", "best_T"]].round(3).to_string(index=False)
              if len(clean) else "none")
    print(f"\nWrote {OUT_CSV}\n      {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
