"""Figures explaining the 6.1 reflective-branch fix (PART2 rev. 17-18).

Broad:
  figures/phase6_fig1_cp_error_heatmaps.png   CP error over (angle x roughness) per
                                              scaler: before, after, and the change
  figures/phase6_fig2_decision_space.png      the decision statistic (tangent step at
                                              the widest point) vs drop width, both
                                              classes + experimental drops, with the
                                              accepted region drawn

Worked examples (overview + left-CP zoom, before/after), styled after
paper_results/plot/plot_cp_comparison.py and CPID's display-mode plots:
  figures/phase6_ex_*.png

"Before" is reproduced by disabling the new branch through its module
constants, so the comparison is against the identical code path minus the
fix. Run from OpenDrop_OP/synthetic_reanalysis/:
    python phase6_figures.py
"""

import contextlib
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import cv2  # noqa: E402

sys.path.insert(0, ".")
from config import FIGURES_DIR, ML_MODEL_DIR, OPENDROP_DIR, PAPER_RESULTS_DIR, RESULTS_DIR  # noqa: E402
from phase6_window_sweep import load_or_render  # noqa: E402
from phase6_reflective_sweep import compute_discriminants, _branch_from_stdout  # noqa: E402
sys.path.insert(0, OPENDROP_DIR)
import opendrop_ml.modules.preprocessing.preprocessing as pp  # noqa: E402
from opendrop_ml.modules.ML_model.prepare_experimental import prepare_for_model, experimental_pred  # noqa: E402

plt.rc("font", size=12)
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

PRE = pd.read_csv(os.path.join(RESULTS_DIR, "phase6_reflective_sweep_1b.csv"))
POST = pd.read_csv(os.path.join(RESULTS_DIR, "phase6_reflective_sweep_postfix.csv"))
EXP = pd.read_csv(os.path.join(RESULTS_DIR, "phase6_experimental_kink.csv"))
for _d in (PRE, POST):
    _d["cp_err"] = _d[["cp_error_left_px", "cp_error_right_px"]].mean(axis=1)

C_WP, C_CPG, C_GT, C_PRE, C_POST, C_EDGE = "tab:blue", "tab:orange", "lime", "red", "cyan", "magenta"


# --------------------------------------------------------------------------
# Broad figure 1 -- CP error heatmaps
# --------------------------------------------------------------------------
def fig1_heatmaps():
    m = PRE.merge(POST, on=["angle", "scaler", "roughness"], suffixes=("_pre", "_post"))
    m = m[(m.angle >= 60) & (m.angle <= 120) & (m.roughness < 0)]
    scalers = [1.0, 2.83, 4.0]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    for r, sc in enumerate(scalers):
        sub = m[m.scaler == sc]
        width = sub.widest_width_px_post.median()
        pre = sub.pivot(index="roughness", columns="angle", values="cp_err_pre")
        post = sub.pivot(index="roughness", columns="angle", values="cp_err_post")
        for c, (data, title, cmap, vmin, vmax) in enumerate([
            (pre, "before", "magma_r", 0, 40), (post, "after", "magma_r", 0, 40),
            (post - pre, "change (after - before)", "RdBu", -30, 30)]):
            ax = axes[r, c]
            im = ax.imshow(data.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xticks(range(len(data.columns))); ax.set_xticklabels([f"{int(a)}" for a in data.columns], fontsize=9)
            ax.set_yticks(range(len(data.index))); ax.set_yticklabels([f"{v:.3f}" for v in data.index], fontsize=8)
            if r == 2: ax.set_xlabel("true contact angle (deg)")
            if c == 0: ax.set_ylabel("roughness (negative = reflective)")
            ax.set_title(f"scaler {sc} (~{width:.0f} px wide) -- CP error, {title}", fontsize=11)
            ax.axvline(list(data.columns).index(90) - 0.5, color="k", lw=0.8, ls=":")
            ax.axvline(list(data.columns).index(90) + 0.5, color="k", lw=0.8, ls=":")
            fig.colorbar(im, ax=ax, fraction=0.04, label="px" if c < 2 else "delta px")
    fig.suptitle("Mean contact-point error (px) on reflective substrates, 60-120 deg, before and after the kink branch\n"
                 "dotted lines mark 90 deg; the residual band at 85-95 deg is the static-image ambiguity", fontsize=13)
    out = os.path.join(FIGURES_DIR, "phase6_fig1_cp_error_heatmaps.png"); fig.savefig(out); plt.close(fig); print(out)


# --------------------------------------------------------------------------
# Broad figure 2 -- the decision space
# --------------------------------------------------------------------------
def fig2_decision_space():
    d = PRE.copy()
    d["kink"] = np.maximum(d.widest1_bottom_angle - d.widest1_top_angle, d.widest2_bottom_angle - d.widest2_top_angle)
    d["wp_err"] = d[["wp_error_left_px", "wp_error_right_px"]].mean(axis=1)
    d["cpg_err"] = d[["cpg_error_left_px", "cpg_error_right_px"]].mean(axis=1)
    dec = d[d.branch.isin(["high_angle_cp_guesses", "reflective_lfit"])]
    want_wp = dec[dec.wp_err + 3 < dec.cpg_err]
    want_cpg = dec[dec.cpg_err + 3 < dec.wp_err]
    ex = EXP[EXP.branch.isin(["high_angle_cp_guesses", "reflective_kink"])].dropna(subset=["kink"])

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(want_cpg.widest_width_px, want_cpg.kink, s=14, c="0.6", label=f"synthetic: contact point is the CP guess (n={len(want_cpg)}, 95-180 deg)")
    sc = ax.scatter(want_wp.widest_width_px, want_wp.kink, s=34, c=want_wp.angle, cmap="viridis", vmin=75, vmax=95,
                    edgecolor="k", linewidth=0.4, label=f"synthetic: contact point is the widest point (n={len(want_wp)}, 75-95 deg)")
    ax.scatter(ex.width, ex.kink, s=40, marker="x", c="k", label=f"experimental drops reaching this decision (n={len(ex)}, no GT, ML 98-156 deg)")
    w = np.linspace(80, 1000, 200)
    arc = np.degrees(10 / (w / 2))
    ax.plot(w, 3 * arc, "r--", lw=1.2, label="ratio threshold: step = 3 x expected arc over 10 rows")
    ax.axhline(14, color="r", lw=1.2, ls="-", label="noise floor: step = 14 deg (experimental max 11.8 deg)")
    ax.fill_between(w, np.maximum(3 * arc, 14), 60, color="tab:blue", alpha=0.08, label="routed to widest point (reflective)")
    ax.set_xlim(80, 1000); ax.set_ylim(-5, 45)
    ax.set_xlabel("drop width at the widest points (px)"); ax.set_ylabel("tangent step at the widest point, bottom - top (deg, 10-row fits)")
    ax.set_title("The decision statistic: a reflection's contact line is a tangent step of ~2(90 - theta);\n"
                 "a genuine high-angle equator is smooth. Real-image edge noise reaches ~12 deg, so the 85 deg cells at high resolution sit under the floor.")
    fig.colorbar(sc, ax=ax, label="true angle of the reflective sub-90 cells (deg)")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.annotate("85 deg, 530 px: step ~10 deg -- below the floor,\nstill misrouted to the CP guess on the substrate",
                xy=(531, 10.5), xytext=(560, 30), fontsize=9, arrowprops=dict(arrowstyle="->", color="k"))
    ax.annotate("the six real high-angle drops the ratio-only\nrule misrouted (step 8.7-10.5 deg)", xy=(480, 9.5), xytext=(120, 38),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="k"))
    out = os.path.join(FIGURES_DIR, "phase6_fig2_decision_space.png"); fig.savefig(out); plt.close(fig); print(out)


# --------------------------------------------------------------------------
# Worked examples
# --------------------------------------------------------------------------
def _run_variant(img, model, min_deg):
    saved = pp.KINK_MIN_DEG
    pp.KINK_MIN_DEG = min_deg
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _, contour, _, _ = pp.isolate_contour(img, DISPLAY=False, blur=True)
            cps = pp.CPID(contour, display=True)
            pred_ds = prepare_for_model(img, input_len=50, n_surface=10, display=False, blur=True)
            ml, _ = experimental_pred(pred_ds, model, side="both")
        return contour, np.array(cps, dtype=float), (float(ml[0]), float(ml[1])), _branch_from_stdout(buf.getvalue())
    finally:
        pp.KINK_MIN_DEG = saved


def _tangent(cp, angle_deg, side, length):
    a = np.radians(angle_deg)
    dx = np.cos(a) * length * (1 if side == "left" else -1)
    dy = -np.sin(a) * length
    return np.array([[cp[0] - dx * 0.0, cp[1]], [cp[0] + dx, cp[1] + dy]])


def _fit_line(pts):
    m, c, _ = pp.linear_fit(pts)
    ys = np.array([pts[:, 1].min() - 3, pts[:, 1].max() + 3])
    if np.isinf(m):
        return np.array([[c, ys[0]], [c, ys[1]]])
    xs = (ys - c) / m if abs(m) > 1e-9 else np.array([pts[:, 0].min(), pts[:, 0].max()])
    return np.column_stack([xs, ys])


def example(name, img, gt, model, title, variants=(("before", 1e9), ("after", pp.KINK_MIN_DEG)), extra_note=""):
    results = {lab: _run_variant(img, model, v) for lab, v in variants}
    contour = results[variants[0][0]][0]
    disc = compute_discriminants(contour)
    wp = np.array([[disc["wp_l_x"], disc["wp_l_y"]], [disc["wp_r_x"], disc["wp_r_y"]]], float)
    cpg = np.array([[disc["cpg_l_x"], disc["cpg_l_y"]], [disc["cpg_r_x"], disc["cpg_r_y"]]], float)
    i1 = disc["widest_index1"]
    width = wp[1, 0] - wp[0, 0]
    arc = np.degrees(10 / (width / 2))
    kink = max(disc["widest1_bottom_angle"] - disc["widest1_top_angle"], disc["widest2_bottom_angle"] - disc["widest2_top_angle"])

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
    # -- overview --
    ax = axes[0]
    ax.imshow(img)
    ax.plot(contour[:, 0], contour[:, 1], ",", color=C_EDGE, label="detected edge")
    ax.plot(wp[:, 0], wp[:, 1], "o", mfc="none", mec=C_WP, ms=11, mew=2, label="widest points")
    ax.plot(cpg[:, 0], cpg[:, 1], "s", mfc="none", mec=C_CPG, ms=11, mew=2, label="CP guesses")
    if gt is not None:
        ax.plot([gt[0][0], gt[1][0]], [gt[0][1], gt[1][1]], "+", color=C_GT, ms=14, mew=2, label="true contact points")
    for lab, col in ((variants[0][0], C_PRE), (variants[1][0], C_POST)):
        cps = results[lab][1]
        ax.plot(cps[:, 0], cps[:, 1], "x", color=col, ms=12, mew=2.5, label=f"CPID {lab}: {results[lab][3]}")
    x0, x1 = contour[:, 0].min(), contour[:, 0].max(); h = (x1 - x0) * 0.83
    bottom = min(contour[:, 1].max() + h * 0.16, img.shape[0]); ax.set_xlim(x0 - 10, x1 + 10); ax.set_ylim(bottom, bottom - h)
    z = max(18, int(width * 0.12)); zx, zy = wp[0]
    ax.add_patch(Rectangle((zx - z, zy - z), 2 * z, 2 * z, fill=False, ec="r", lw=1.2))
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8, loc="upper right"); ax.set_title("overview", fontsize=11)

    # -- zoom on the left widest point with the two 10-row fits --
    ax = axes[1]
    ax.imshow(img); ax.set_xlim(zx - z, zx + z); ax.set_ylim(zy + z, zy - z)
    ax.plot(contour[:, 0], contour[:, 1], ".", color=C_EDGE, ms=4, label="detected edge")
    top_pts, bot_pts = contour[i1:i1 + 10], contour[i1 - 10:i1]
    tl, bl = _fit_line(top_pts), _fit_line(bot_pts)
    ax.plot(top_pts[:, 0], top_pts[:, 1], "o", color="gold", ms=5, label=f"10 rows above: top angle {disc['widest1_top_angle']:.1f} deg")
    ax.plot(bot_pts[:, 0], bot_pts[:, 1], "o", color="deepskyblue", ms=5, label=f"10 rows below: bottom angle {disc['widest1_bottom_angle']:.1f} deg")
    ax.plot(tl[:, 0], tl[:, 1], "-", color="gold", lw=1.5); ax.plot(bl[:, 0], bl[:, 1], "-", color="deepskyblue", lw=1.5)
    ax.plot(*wp[0], "o", mfc="none", mec=C_WP, ms=13, mew=2); ax.plot(*cpg[0], "s", mfc="none", mec=C_CPG, ms=13, mew=2)
    if gt is not None: ax.plot(*gt[0], "+", color=C_GT, ms=16, mew=2.5)
    for lab, col in ((variants[0][0], C_PRE), (variants[1][0], C_POST)):
        ax.plot(*results[lab][1][0], "x", color=col, ms=14, mew=2.5)
    fired = kink > pp.KINK_MIN_DEG and kink / arc > pp.KINK_RATIO_THRESHOLD
    ax.text(0.03, 0.03, f"step (bottom - top) = {kink:.1f} deg\nexpected arc over 10 rows = {arc:.1f} deg\nratio = {kink / arc:.2f}  (threshold {pp.KINK_RATIO_THRESHOLD})\n"
            f"floor {pp.KINK_MIN_DEG:.0f} deg  ->  reflective branch {'FIRES' if fired else 'does not fire'}",
            transform=ax.transAxes, fontsize=9.5, va="bottom", bbox=dict(facecolor="0.9", edgecolor="none", alpha=0.9, pad=5))
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8, loc="upper right"); ax.set_title("left widest point: the two fits the cascade compares", fontsize=11)

    # -- ML result before/after at the left CP --
    ax = axes[2]
    ax.imshow(img); ax.set_xlim(zx - z * 2.2, zx + z * 2.2); ax.set_ylim(zy + z * 1.6, zy - z * 2.8)
    ax.plot(contour[:, 0], contour[:, 1], ".", color=C_EDGE, ms=3, label="detected edge")
    for lab, col, ls in ((variants[0][0], C_PRE, "--"), (variants[1][0], C_POST, "-")):
        cps, ml = results[lab][1], results[lab][2]
        t = _tangent(cps[0], ml[0], "left", z * 2)
        ax.plot(t[:, 0], t[:, 1], ls, color=col, lw=2, label=f"ML {lab}: {ml[0]:.1f} deg" + (f"  (true {gt_angle:.0f})" if (gt_angle := _gt_angle(name)) else ""))
        ax.plot(*cps[0], "x", color=col, ms=12, mew=2.5)
    if gt is not None: ax.plot(*gt[0], "+", color=C_GT, ms=16, mew=2.5, label="true contact point")
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=9, loc="upper right"); ax.set_title("ML prediction from each contact point", fontsize=11)

    fig.suptitle(title + (f"\n{extra_note}" if extra_note else ""), fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, f"phase6_ex_{name}.png"); fig.savefig(out); plt.close(fig); print(out)


def _gt_angle(name):
    try:
        return float(name.split("deg")[0].split("_")[-1])
    except Exception:
        return None


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig1_heatmaps()
    fig2_decision_space()

    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    model = tf.keras.models.load_model(ML_MODEL_DIR)

    cases = [
        ("fixed_80deg_s2.83", (80, 0.55, 2.83, -0.045),
         "FIXED -- 80 deg, reflective (roughness -0.045), scaler 2.83 (~200 px): the 21 px misplacement the study found",
         "before: routed to the CP guess sitting on the substrate strip inside the reflection; after: the step (20 deg) clears both thresholds"),
        ("unfixed_85deg_s1.0", (85, 0.55, 1.0, -0.045),
         "NOT FIXED -- 85 deg, reflective, scaler 1.0 (~560 px, the highest resolution): the case to be suspicious of",
         "the step is ~10 deg = 2(90-85), under the 14 deg experimental noise floor; the CP guess lands on the substrate 30+ px below and ML reads ~110 deg"),
        ("lowres_85deg_s4.0", (85, 0.55, 4.0, -0.045),
         "ALREADY OK -- the same 85 deg drop at scaler 4.0 (~140 px): caught by the older `bottom angle > 100` branch",
         "at low resolution 10 rows span enough arc that the bottom angle exceeds 100 -- correct answer, resolution-dependent reason"),
        ("protected_150deg", (150, 0.55, 2.83, 0.0),
         "PROTECTED -- 150 deg, no reflection: the class the branch must not touch",
         "smooth equator: step ~ arc only; unchanged before/after"),
    ]
    for name, (a, b, s, r), title, note in cases:
        img, gt, _ = load_or_render(a, b, s, r)
        example(f"{name}_{a}deg", img, gt, model, title, extra_note=note)

    # the experimental drop the ratio-only rule broke: show what that variant would have done
    p = os.path.join(PAPER_RESULTS_DIR, "data", "experimental", "b0044.bmp")
    img = cv2.imread(p)
    example("experimental_b0044_ratio_only_vs_final", img, None, model,
            "WHY THE NOISE FLOOR -- experimental b0044.bmp (genuine ~145 deg drop, ML and YL agree): the ratio-only rule misrouted it",
            variants=(("ratio-only", 0.0), ("final", pp.KINK_MIN_DEG)),
            extra_note="real-edge noise gives an 8.7 deg step at a smooth equator; on a 435 px drop the expected arc is only 2.6 deg, so ratio-only fires. The 14 deg floor stops it.")


if __name__ == "__main__":
    main()
