"""Phase 6.1 Step 1b analysis -- is the reflective-vs-high-angle decision
separable by a one-branch threshold? (PART2_REMAINING_WORK.md, 6.1 Step 1b,
"Analysis to produce".)

Reads results/offgrid/phase6_reflective_sweep_1b.csv (from phase6_reflective_sweep.py)
and answers, offline, the question Step 2 needs:

  Among the cells that reach the `>79` test at preprocessing.py:1866, does any
  single scalar -- the current discriminant, a scale-normalised version of it,
  or another quantity CPID already has in hand -- cleanly separate the cells
  where `widest_points` is the right answer from those where `CP_guesses` is?

The "right answer" per cell is read directly from the sweep table: whichever
of widest_points / CP_guesses lands nearer the ground-truth contact point
(margin PREFER_MARGIN_PX; cells inside the margin are "either" and ignored).
No re-running, no model.

Caveat (stated in Step 1b): cells that fall *below* the `>79` threshold go to
iterative_Lfit, not widest_points, so the widest_points error is a proxy for
that side. Session 2 re-runs the sweep with the real code path.

Outputs:
  figures/offgrid/phase6_separability_<feature>.png   histograms per class, faceted by scaler
  results/offgrid/phase6_separability_summary.csv     best threshold + residual errors per feature/scaler
  stdout: the separability verdict

Run from OpenDrop_OP/paper_results/offgrid/:
    python phase6_separability.py            # full table
    python phase6_separability.py --smoke    # the 4-cell smoke CSV, to check plumbing
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_PAPER_RESULTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PAPER_RESULTS_DIR)  # paper_results/, for config
from config import OFFGRID_FIGURES_DIR as FIGURES_DIR, OFFGRID_RESULTS_DIR as RESULTS_DIR  # noqa: E402

PREFER_MARGIN_PX = 3.0
CURRENT_THRESHOLD = 79.0  # preprocessing.py:1866  `180 - widest_bottom_angle > 79`

# Branches whose cells reached the `>79` test (passed the first `if`, failed the
# two reflective `elif`s). Only these are affected by moving that threshold.
DECISION_BRANCHES = {"high_angle_cp_guesses", "reflective_lfit", "reflective_kink"}

# Candidate one-branch discriminants. Every one except aspect_gt is computable
# inside CPID() from what it already has (contour, widest_points, CP_guesses).
# aspect_gt uses the ground-truth baseline and is kept only to show whether the
# classes are separable *given* the baseline. Direction is searched both ways.
FEATURES = {
    "disc_max": "current statistic: max(180 - widest_bottom_angle)",
    "disc_min": "min(180 - widest_bottom_angle)",
    "disc_max_over_width": "disc_max / widest_width_px  (option a': scale-normalised)",
    "top_angle_max": "max(widest_top_angle)",
    "top_angle_min": "min(widest_top_angle)",
    "top_minus_bottom_max": "max(widest_top_angle - widest_bottom_angle)",
    "cpg_below_wp_px": "mean vertical gap CP_guesses below widest_points, px",
    "cpg_below_wp_frac": "that gap / widest_width_px",
    "aspect_wp": "(widest_points y - apex y) / widest_width_px",
    "aspect_cpg": "(CP_guesses y - apex y) / widest_width_px",
    "cpg_width_frac": "CP_guesses width / widest_width_px",
    "aspect_gt": "REFERENCE ONLY, uses the GT baseline: drop_height_px / widest_width_px",
}
NON_DEPLOYABLE = {"aspect_gt"}


def load(path):
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["wp_err"] = df[["wp_error_left_px", "wp_error_right_px"]].mean(axis=1)
    df["cpg_err"] = df[["cpg_error_left_px", "cpg_error_right_px"]].mean(axis=1)
    df["cpid_err"] = df[["cp_error_left_px", "cp_error_right_px"]].mean(axis=1)
    df["prefer"] = np.select(
        [df["wp_err"] + PREFER_MARGIN_PX < df["cpg_err"],
         df["cpg_err"] + PREFER_MARGIN_PX < df["wp_err"]],
        ["widest_points", "CP_guesses"], default="either",
    )
    df["reflective"] = df["roughness"] < 0
    df["high_angle"] = df["angle"] >= 90

    df["disc_min"] = df[["disc_left", "disc_right"]].min(axis=1)
    df["disc_max_over_width"] = df["disc_max"] / df["widest_width_px"]
    df["top_angle_max"] = df[["widest1_top_angle", "widest2_top_angle"]].max(axis=1)
    df["top_angle_min"] = df[["widest1_top_angle", "widest2_top_angle"]].min(axis=1)
    df["top_minus_bottom_max"] = np.maximum(
        df["widest1_top_angle"] - df["widest1_bottom_angle"],
        df["widest2_top_angle"] - df["widest2_bottom_angle"],
    )
    df["cpg_below_wp_px"] = ((df["cpg_l_y"] - df["wp_l_y"]) + (df["cpg_r_y"] - df["wp_r_y"])) / 2
    df["cpg_below_wp_frac"] = df["cpg_below_wp_px"] / df["widest_width_px"]
    if "apex_y" not in df.columns:  # rev. 13 CSV predates the column; recover it
        df["apex_y"] = df["gt_cpl_y"] - df["drop_height_px"]
    wp_y = (df["wp_l_y"] + df["wp_r_y"]) / 2
    cpg_y = (df["cpg_l_y"] + df["cpg_r_y"]) / 2
    df["aspect_wp"] = (wp_y - df["apex_y"]) / df["widest_width_px"]
    df["aspect_cpg"] = (cpg_y - df["apex_y"]) / df["widest_width_px"]
    df["cpg_width_frac"] = (df["cpg_r_x"] - df["cpg_l_x"]) / df["widest_width_px"]
    df["aspect_gt"] = df["drop_height_px"] / df["widest_width_px"]
    return df


def best_threshold(x, prefer):
    """Smallest misassignment count over all thresholds and both directions.
    Rule A: x > T -> CP_guesses.  Rule B: x > T -> widest_points."""
    x = np.asarray(x, dtype=float)
    want_cpg = np.asarray(prefer == "CP_guesses")
    want_wp = np.asarray(prefer == "widest_points")
    if want_cpg.sum() == 0 or want_wp.sum() == 0:
        return {"n_cpg": int(want_cpg.sum()), "n_wp": int(want_wp.sum()),
                "best_T": np.nan, "rule": "degenerate", "min_errors": 0, "separable": True}
    cands = np.unique(np.concatenate([x, [-np.inf]]))
    best = None
    for T in cands:
        above = x > T
        errs_a = int((above & want_wp).sum() + (~above & want_cpg).sum())
        errs_b = int((above & want_cpg).sum() + (~above & want_wp).sum())
        for rule, e in (("x>T -> CP_guesses", errs_a), ("x>T -> widest_points", errs_b)):
            if best is None or e < best["min_errors"]:
                best = {"best_T": float(T), "rule": rule, "min_errors": e}
    best.update({"n_cpg": int(want_cpg.sum()), "n_wp": int(want_wp.sum()),
                 "separable": best["min_errors"] == 0})
    return best


def plot_feature(dec, feat, desc, out_path):
    scalers = sorted(dec["scaler"].unique())
    fig, axes = plt.subplots(1, len(scalers), figsize=(5 * len(scalers), 3.8), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scalers):
        sub = dec[dec["scaler"] == sc]
        for label, colour in (("widest_points", "tab:blue"), ("CP_guesses", "tab:red")):
            vals = sub.loc[sub["prefer"] == label, feat].dropna()
            if len(vals):
                ax.hist(vals, bins=30, alpha=0.55, color=colour, label=f"should be {label} (n={len(vals)})")
        if feat == "disc_max":
            ax.axvline(CURRENT_THRESHOLD, color="k", ls="--", lw=1, label="current 79")
        ax.set_title(f"scaler = {sc}")
        ax.set_xlabel(feat)
        ax.legend(fontsize=7)
    fig.suptitle(f"{feat}: {desc}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    smoke = "--smoke" in sys.argv
    name = "phase6_reflective_sweep_1b.csv"
    if "--in" in sys.argv:
        name = sys.argv[sys.argv.index("--in") + 1]
    path = os.path.join(RESULTS_DIR, "phase6_reflective_sweep_smoke.csv" if smoke else name)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = load(path)
    print(f"{len(df)} ok cells from {os.path.basename(path)}")

    # ---- 1. Who reaches the decision, and what should they have got? ----
    dec = df[df["branch"].isin(DECISION_BRANCHES)].copy()
    print(f"\n{len(dec)} cells reach the `>79` decision "
          f"({', '.join(sorted(DECISION_BRANCHES))})")
    print(pd.crosstab([dec["reflective"], dec["high_angle"]], dec["prefer"]))

    # ---- 2. Is the current statistic even informative? ----
    sat = (dec["disc_max"] >= 90 - 1e-6)
    print(f"\ndisc_max saturated at 90 (vertical fit under widest point): "
          f"{int(sat.sum())}/{len(dec)} decision cells")
    if len(dec):
        print(pd.crosstab(sat, dec["prefer"], rownames=["disc_max==90"]))
    print("If both classes are mostly saturated, no threshold on disc_max can separate them "
          "-- fixed (a) and scale-normalised (a') are both ruled out on this statistic, "
          "and the fix needs a different one-branch discriminant (see the other rows below).")

    # ---- 3. Current-code damage, for reference ----
    cur = dec[dec["prefer"] != "either"]
    if len(cur):
        wrong = ((cur["prefer"] == "widest_points") & (cur["branch"] == "high_angle_cp_guesses"))
        print(f"\nCurrent code: {int(wrong.sum())}/{len(cur)} labelled decision cells take CP_guesses "
              f"when widest_points was nearer GT; their mean CPID error = "
              f"{cur.loc[wrong, 'cpid_err'].mean():.1f} px")

    # ---- 4. Separability per feature, per scaler and pooled ----
    rows = []
    for feat, desc in FEATURES.items():
        if feat not in dec.columns:
            continue
        for sc in ["pooled"] + sorted(dec["scaler"].unique().tolist()):
            sub = dec if sc == "pooled" else dec[dec["scaler"] == sc]
            sub = sub[sub["prefer"] != "either"].dropna(subset=[feat])
            if sub.empty:
                continue
            r = best_threshold(sub[feat], sub["prefer"])
            r.update({"feature": feat, "scaler": sc, "n": len(sub)})
            rows.append(r)
        plot_feature(dec, feat, desc, os.path.join(FIGURES_DIR, f"phase6_separability_{feat}.png"))

    summary = pd.DataFrame(rows)[["feature", "scaler", "n", "n_wp", "n_cpg", "rule",
                                  "best_T", "min_errors", "separable"]]
    summary.to_csv(os.path.join(RESULTS_DIR, "phase6_separability_summary.csv"), index=False)

    print("\n--- best single threshold per feature (min misassignments) ---")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(summary.round(3).to_string(index=False))

    # ---- 5. Verdict ----
    ref = summary[(summary["feature"].isin(NON_DEPLOYABLE)) & (summary["scaler"] != "pooled")]
    if len(ref):
        print(f"\nReference (not deployable): aspect_gt separable per scaler = "
              f"{bool(ref['separable'].all())}, thresholds "
              f"{ref['best_T'].round(3).tolist()} -- i.e. the classes ARE separable given the true "
              f"baseline; the difficulty is entirely in estimating that baseline.")
    summary_dep = summary[~summary["feature"].isin(NON_DEPLOYABLE)]
    pooled = summary_dep[summary_dep["scaler"] == "pooled"].sort_values("min_errors")
    print("\n=== Step 1b verdict ===")
    if pooled.empty:
        print("No labelled decision cells -- nothing to separate.")
        return
    top = pooled.iloc[0]
    if top["separable"]:
        print(f"SEPARABLE (pooled): feature `{top['feature']}`, rule `{top['rule']}`, "
              f"T = {top['best_T']:.3f}, 0 misassignments on n={top['n']}.")
        print("-> go for Session 2: option (a) if feature is disc_max, otherwise a one-branch "
              "swap of the discriminant. Regression-guard with the 6.4 harness.")
    else:
        per_scaler = summary_dep[(summary_dep["scaler"] != "pooled") & summary_dep["separable"]]
        print(f"NOT cleanly separable pooled. Best: `{top['feature']}` with "
              f"{int(top['min_errors'])} misassignments on n={top['n']}.")
        if len(per_scaler):
            print("Separable per scaler for:")
            print(per_scaler[["feature", "scaler", "rule", "best_T"]].to_string(index=False))
            print("-> a scale-aware threshold on that feature may work; check whether best_T "
                  "moves monotonically with scaler before committing to (a').")
        else:
            print("-> no candidate separates even per scaler. Record this in the changelog, "
                  "stop, and decide deliberately between (b') and documenting the map.")
    print(f"\nFigures: {FIGURES_DIR}/phase6_separability_*.png")
    print(f"Summary: {RESULTS_DIR}/phase6_separability_summary.csv")


if __name__ == "__main__":
    main()
