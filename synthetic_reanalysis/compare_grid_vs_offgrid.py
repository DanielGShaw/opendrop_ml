"""Phase 4 comparative analysis (SYNTHETIC_DATA_REANALYSIS.md, Phase 4 + A.6).

Joins the Phase 3 image-level result sets (off-grid, on-grid-control, 5
replicates) with the Phase 1 matrix/manifest CSVs to attach grid_distance,
then computes:

  - Primary: image-level |ML error| vs. grid_distance slope (OLS +
    Theil-Sen, image-level bootstrap CI), with each classical method's
    slope as a null-slope control.
  - Paired nearest-on-grid-control matching: signed bias and |bias|,
    correlation of the paired difference with grid_distance.
  - CP-localisation mediation: CP pixel error vs. ML angle error.
  - Beyond-MAE tail metrics (median/p90/max/failure rate), split by
    in-domain/out-of-domain and angle bucket.

Does NOT re-derive the render-noise band or contour-level primary endpoint
-- those are Phase 1 / Phase 2A outputs, read from existing result files.

Run from OpenDrop_OP/synthetic_reanalysis/:
    python compare_grid_vs_offgrid.py
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from config import BOOTSTRAP_SEED, RESULTS_DIR
from matrix_utils import grid_distance as grid_distance_fn
from matrix_utils import roughness_regime

OFFGRID_RESULTS = os.path.join(RESULTS_DIR, "synthetic_offgrid_results.csv")
ONGRID_CONTROL_RESULTS = os.path.join(RESULTS_DIR, "synthetic_ongrid_control_results.csv")
OFFGRID_MATRIX = os.path.join(RESULTS_DIR, "offgrid_matrix.csv")
ONGRID_MANIFEST = os.path.join(RESULTS_DIR, "ongrid_control_manifest.csv")
REPLICATE_RESULTS = [
    os.path.join(RESULTS_DIR, f"synthetic_ongrid_control_replicate_{i}_results.csv") for i in range(5)
]

MEASUREMENTS_CSV = os.path.join(RESULTS_DIR, "phase4_measurements.csv")
SUMMARY_MD = os.path.join(RESULTS_DIR, "phase4_summary.md")

METHODS = ["tangent", "polynomial", "circular", "ellipse", "yl", "ml"]
ANGLE_BUCKETS = [(0, 90), (90, 110), (110, 181)]
ANGLE_BUCKET_LABELS = ["<90", "90-110", ">=110"]

IMAGE_LEVEL_BAR = 1.4  # deg; Phase 4 practical bar (Significance criteria)
N_BOOTSTRAP_OLS = 10000
N_BOOTSTRAP_TS = 2000  # Theil-Sen is O(n^2) per resample; smaller n here (image-level, ~650 images) keeps this tractable


def angle_bucket(angle):
    for (lo, hi), label in zip(ANGLE_BUCKETS, ANGLE_BUCKET_LABELS):
        if lo <= angle < hi:
            return label
    return ANGLE_BUCKET_LABELS[-1]


def build_measurements():
    """Long-format table: one row per (image, side), with grid_distance and
    per-method absolute error attached."""
    offgrid = pd.read_csv(OFFGRID_RESULTS)
    ongrid = pd.read_csv(ONGRID_CONTROL_RESULTS)
    matrix = pd.read_csv(OFFGRID_MATRIX)
    manifest = pd.read_csv(ONGRID_MANIFEST)

    offgrid = offgrid.merge(
        matrix[["filename", "grid_distance", "in_domain", "ood_axis", "roughness_regime", "cell_id"]],
        left_on="image_name", right_on="filename", how="left",
    )
    offgrid["set"] = "offgrid"

    manifest = manifest.copy()
    manifest["grid_distance"] = manifest.apply(
        lambda r: grid_distance_fn(r["bond"], r["scaler"], r["roughness"]), axis=1
    )
    manifest["in_domain"] = True
    manifest["ood_axis"] = np.nan
    manifest["roughness_regime"] = manifest["roughness"].apply(roughness_regime)
    ongrid = ongrid.merge(
        manifest[["filename", "grid_distance", "in_domain", "ood_axis", "roughness_regime"]],
        left_on="image_name", right_on="filename", how="left",
    )
    ongrid["set"] = "ongrid_control"

    both = pd.concat([offgrid, ongrid], ignore_index=True, sort=False)
    both = both[both["error"].isna() if "error" in both.columns else slice(None)].copy()

    rows = []
    for _, r in both.iterrows():
        for side, cp_pred_x, cp_pred_y, cp_gt_x, cp_gt_y in [
            ("left", "left_CP_x", "left_CP_y", "gt_cpl_x", "gt_cpl_y"),
            ("right", "right_CP_x", "right_CP_y", "gt_cpr_x", "gt_cpr_y"),
        ]:
            row = {
                "image_name": r["image_name"],
                "set": r["set"],
                "side": side,
                "angle": r["gt_angle"],
                "bond": r["gt_bond"],
                "scaler": r["gt_scale"],
                "roughness": r["gt_roughness"],
                "roughness_regime": r.get("roughness_regime"),
                "grid_distance": r.get("grid_distance"),
                "in_domain": r.get("in_domain"),
                "ood_axis": r.get("ood_axis"),
                "profile_length": r.get("profile_length"),
            }
            gt_angle = r["gt_angle"]
            for method in METHODS:
                pred_col = f"{method}_{side}_angle"
                row[f"{method}_error"] = (
                    abs(r[pred_col] - gt_angle) if pred_col in r and pd.notna(r[pred_col]) else np.nan
                )
            if pd.notna(r.get(cp_pred_x)) and pd.notna(r.get(cp_gt_x)):
                row["cp_error_px"] = float(
                    np.hypot(r[cp_pred_x] - r[cp_gt_x], r[cp_pred_y] - r[cp_gt_y])
                )
            else:
                row["cp_error_px"] = np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def fit_slopes(distances, errors):
    ols_slope, ols_intercept = np.polyfit(distances, errors, 1)
    ts_slope, ts_intercept, ts_lo, ts_hi = stats.theilslopes(errors, distances, alpha=0.95)
    return {
        "ols_slope": float(ols_slope),
        "ols_intercept": float(ols_intercept),
        "theilsen_slope": float(ts_slope),
        "theilsen_intercept": float(ts_intercept),
        "theilsen_ci95": (float(ts_lo), float(ts_hi)),
    }


def bootstrap_slope_ci_by_image(df, error_col, n_boot, seed, method="ols"):
    """Resampling unit is the image (A.6) -- resample unique image_names
    with replacement, then take both sides' rows for the resampled images,
    so left/right correlation within an image is respected."""
    d = df.dropna(subset=[error_col, "grid_distance"])
    images = d["image_name"].unique()
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot)
    grouped = {name: g[["grid_distance", error_col]].to_numpy() for name, g in d.groupby("image_name")}
    for i in range(n_boot):
        sampled_images = rng.choice(images, size=len(images), replace=True)
        stacked = np.concatenate([grouped[name] for name in sampled_images], axis=0)
        distances, errors = stacked[:, 0], stacked[:, 1]
        if method == "ols":
            s, _ = np.polyfit(distances, errors, 1)
        else:
            s, _, _, _ = stats.theilslopes(errors, distances)
        slopes[i] = s
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(lo), float(hi)


def primary_endpoint(df):
    lines = ["## Primary endpoint: image-level |error| vs. grid_distance slope", ""]
    lines.append("| Method | OLS slope | Theil-Sen slope | Theil-Sen 95% CI (analytical) |")
    lines.append("|---|---|---|---|")

    ml_result = None
    for method in METHODS:
        col = f"{method}_error"
        d = df.dropna(subset=[col, "grid_distance"])
        fits = fit_slopes(d["grid_distance"].to_numpy(), d[col].to_numpy())
        lines.append(
            f"| {method} | {fits['ols_slope']:.4f} | {fits['theilsen_slope']:.4f} | "
            f"[{fits['theilsen_ci95'][0]:.4f}, {fits['theilsen_ci95'][1]:.4f}] |"
        )
        if method == "ml":
            ml_result = fits

    d_ml = df.dropna(subset=["ml_error", "grid_distance"])
    print("  Bootstrapping ML OLS slope CI (image-level resampling)...")
    ols_lo, ols_hi = bootstrap_slope_ci_by_image(df, "ml_error", N_BOOTSTRAP_OLS, BOOTSTRAP_SEED, "ols")
    disagreement_ratio = ml_result["ols_slope"] / ml_result["theilsen_slope"] if ml_result["theilsen_slope"] else np.nan
    material_disagreement = abs(disagreement_ratio - 1) > 0.2

    lines.append("")
    lines.append(f"ML OLS slope 95% bootstrap CI (image-level, n={N_BOOTSTRAP_OLS}): [{ols_lo:.4f}, {ols_hi:.4f}]")
    lines.append(f"OLS/Theil-Sen ratio: {disagreement_ratio:.2f} "
                 f"({'material disagreement -> Theil-Sen is the primary estimate' if material_disagreement else 'agree closely -> OLS is fine as primary'})")

    ts_lo_boot = ts_hi_boot = None
    if material_disagreement:
        print(f"  Bootstrapping ML Theil-Sen slope CI (image-level resampling, n={N_BOOTSTRAP_TS})...")
        ts_lo_boot, ts_hi_boot = bootstrap_slope_ci_by_image(
            df, "ml_error", N_BOOTSTRAP_TS, BOOTSTRAP_SEED, "theilsen"
        )
        lines.append(f"ML Theil-Sen slope 95% bootstrap CI (image-level, n={N_BOOTSTRAP_TS}): [{ts_lo_boot:.4f}, {ts_hi_boot:.4f}]")

    primary_slope = ml_result["theilsen_slope"] if material_disagreement else ml_result["ols_slope"]
    primary_ci = (ts_lo_boot, ts_hi_boot) if material_disagreement else (ols_lo, ols_hi)
    max_gd = df["grid_distance"].max()
    predicted_increase = primary_slope * max_gd

    lines.append("")
    lines.append(
        f"Primary estimate: {'Theil-Sen' if material_disagreement else 'OLS'} slope = {primary_slope:.4f} deg/unit, "
        f"95% CI [{primary_ci[0]:.4f}, {primary_ci[1]:.4f}]"
    )
    ci_excludes_zero = primary_ci[0] > 0
    lines.append(f"CI excludes zero: {ci_excludes_zero}")
    lines.append(
        f"Predicted |error| increase at max observed grid_distance ({max_gd:.2f}): {predicted_increase:.4f} deg "
        f"vs. image-level practical bar {IMAGE_LEVEL_BAR} deg"
    )
    if ci_excludes_zero:
        materiality = "MATERIAL" if predicted_increase >= IMAGE_LEVEL_BAR else "statistically real but below the practical bar"
        lines.append(f"Verdict: {materiality}")
    else:
        lines.append(f"Verdict: BOUNDED NULL -- any grid-proximity effect is below {primary_ci[1]:.4f} deg/unit (95% CI)")

    return "\n".join(lines), {
        "ml_ols_slope": ml_result["ols_slope"],
        "ml_theilsen_slope": ml_result["theilsen_slope"],
        "primary_slope": primary_slope,
        "primary_ci_lo": primary_ci[0],
        "primary_ci_hi": primary_ci[1],
        "predicted_increase_at_max": predicted_increase,
        "material_disagreement": material_disagreement,
    }


def paired_matching(df):
    """Match each off-grid cell to its nearest on-grid-control cell (same
    angle, same roughness sign, minimum per-axis-normalised parameter
    distance), and compare per-image mean-of-both-sides signed error.

    `df` (the long per-side table) isn't used directly here since signed
    (not absolute) error is needed -- recomputed from the raw wide result
    CSVs instead."""
    lines = ["## Paired nearest-on-grid-control matching", ""]

    offgrid = pd.read_csv(OFFGRID_RESULTS)
    ongrid = pd.read_csv(ONGRID_CONTROL_RESULTS)
    matrix = pd.read_csv(OFFGRID_MATRIX)
    manifest = pd.read_csv(ONGRID_MANIFEST)

    def signed_avg(row):
        if pd.isna(row.get("ml_left_angle")) or pd.isna(row.get("ml_right_angle")):
            return np.nan
        return ((row["ml_left_angle"] + row["ml_right_angle"]) / 2.0) - row["gt_angle"]

    offgrid = offgrid.merge(matrix, left_on="image_name", right_on="filename", how="left", suffixes=("", "_m"))
    offgrid["signed_error"] = offgrid.apply(signed_avg, axis=1)

    manifest = manifest.copy()
    manifest["grid_distance"] = manifest.apply(
        lambda r: grid_distance_fn(r["bond"], r["scaler"], r["roughness"]), axis=1
    )
    ongrid = ongrid.merge(manifest, left_on="image_name", right_on="filename", how="left", suffixes=("", "_m"))
    ongrid["signed_error"] = ongrid.apply(signed_avg, axis=1)

    def axis_distance(b1, s1, r1, b2, s2, r2):
        d_bo = abs(b1 - b2) / 0.10
        d_sc = abs(np.log(s1) - np.log(s2)) / (np.log(4) / 6)
        step_r = 0.002 if r1 >= 0 else 0.01
        d_r = abs(r1 - r2) / step_r
        return np.sqrt(d_bo**2 + d_sc**2 + d_r**2)

    matches = []
    for _, off_row in offgrid.dropna(subset=["signed_error"]).iterrows():
        candidates = ongrid[ongrid["gt_angle"] == off_row["gt_angle"]]
        # same-sign roughness filter (never match across the reflective/jitter boundary, A.2)
        same_sign = candidates[
            (candidates["gt_roughness"] >= 0) == (off_row["gt_roughness"] >= 0)
        ]
        same_sign = same_sign.dropna(subset=["signed_error"])
        if same_sign.empty:
            continue
        dists = same_sign.apply(
            lambda r: axis_distance(
                off_row["gt_bond"], off_row["gt_scale"], off_row["gt_roughness"],
                r["gt_bond"], r["gt_scale"], r["gt_roughness"],
            ),
            axis=1,
        )
        best_idx = dists.idxmin()
        best = same_sign.loc[best_idx]
        matches.append({
            "offgrid_image": off_row["image_name"],
            "offgrid_signed_error": off_row["signed_error"],
            "offgrid_grid_distance": off_row["grid_distance"],
            "ongrid_image": best["image_name"],
            "ongrid_signed_error": best["signed_error"],
            "pair_distance": dists.loc[best_idx],
            "in_domain": off_row["in_domain"],
        })

    match_df = pd.DataFrame(matches)
    match_df["delta"] = match_df["offgrid_signed_error"] - match_df["ongrid_signed_error"]
    match_df["abs_delta"] = match_df["delta"].abs()

    bias = match_df["delta"].mean()
    abs_bias = match_df["abs_delta"].mean()
    corr = match_df["abs_delta"].corr(match_df["offgrid_grid_distance"])

    lines.append(f"n pairs: {len(match_df)}")
    lines.append(f"Mean signed difference (bias, off-grid minus matched on-grid): {bias:.4f} deg")
    lines.append(f"Mean |difference|: {abs_bias:.4f} deg")
    lines.append(f"Correlation(|Δ|, off-grid grid_distance): r={corr:.3f}")

    ood_only = match_df[match_df["in_domain"] == False]  # noqa: E712
    if len(ood_only):
        lines.append(
            f"Out-of-domain subset (n={len(ood_only)}): mean bias {ood_only['delta'].mean():.4f} deg, "
            f"mean |Δ| {ood_only['abs_delta'].mean():.4f} deg"
        )

    match_df.to_csv(os.path.join(RESULTS_DIR, "phase4_paired_matches.csv"), index=False)
    return "\n".join(lines)


def cp_mediation(df):
    lines = ["## CP-localisation mediation", ""]
    for set_name in ["offgrid", "ongrid_control"]:
        d = df[df["set"] == set_name].dropna(subset=["cp_error_px", "ml_error"])
        if len(d) < 2:
            continue
        r = d["cp_error_px"].corr(d["ml_error"])
        lines.append(
            f"{set_name}: n={len(d)}, mean CP error {d['cp_error_px'].mean():.2f} px, "
            f"corr(CP error px, ML angle error) = {r:.3f}"
        )
    d_all = df.dropna(subset=["cp_error_px", "ml_error"])
    r_all = d_all["cp_error_px"].corr(d_all["ml_error"])
    lines.append(f"pooled: n={len(d_all)}, corr = {r_all:.3f}")
    return "\n".join(lines)


def tail_metrics(df):
    lines = ["## Beyond-MAE tail metrics", ""]
    lines.append("| set | method | n | MAE | median | p90 | max | failure rate |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for set_name in ["offgrid", "ongrid_control"]:
        d_set = df[df["set"] == set_name]
        n_total = d_set["image_name"].nunique()
        for method in METHODS:
            col = f"{method}_error"
            d = d_set.dropna(subset=[col])
            n_fail = n_total - d["image_name"].nunique()
            if len(d) == 0:
                lines.append(f"| {set_name} | {method} | 0 | - | - | - | - | {n_fail}/{n_total} |")
                continue
            lines.append(
                f"| {set_name} | {method} | {len(d)} | {d[col].mean():.3f} | {d[col].median():.3f} | "
                f"{d[col].quantile(0.9):.3f} | {d[col].max():.3f} | {n_fail}/{n_total} |"
            )

    lines.append("")
    lines.append("### ML error by in-domain / out-of-domain (off-grid only)")
    lines.append("| group | n | MAE | median | max |")
    lines.append("|---|---|---|---|---|")
    off = df[df["set"] == "offgrid"]
    for label, sub in [("in_domain", off[off["in_domain"] == True]), ("out_of_domain", off[off["in_domain"] == False])]:  # noqa: E712
        d = sub.dropna(subset=["ml_error"])
        if len(d):
            lines.append(f"| {label} | {len(d)} | {d['ml_error'].mean():.3f} | {d['ml_error'].median():.3f} | {d['ml_error'].max():.3f} |")

    lines.append("")
    lines.append("### ML error by out-of-domain axis (off-grid only)")
    lines.append("| ood_axis | n | MAE | median | max |")
    lines.append("|---|---|---|---|---|")
    for axis, sub in off[off["in_domain"] == False].groupby("ood_axis"):  # noqa: E712
        d = sub.dropna(subset=["ml_error"])
        if len(d):
            lines.append(f"| {axis} | {len(d)} | {d['ml_error'].mean():.3f} | {d['ml_error'].median():.3f} | {d['ml_error'].max():.3f} |")

    lines.append("")
    lines.append("### ML error by angle bucket")
    lines.append("| set | bucket | n | MAE | median |")
    lines.append("|---|---|---|---|---|")
    df = df.copy()
    df["angle_bucket"] = df["angle"].apply(angle_bucket)
    for set_name in ["offgrid", "ongrid_control"]:
        for label in ANGLE_BUCKET_LABELS:
            d = df[(df["set"] == set_name) & (df["angle_bucket"] == label)].dropna(subset=["ml_error"])
            if len(d):
                lines.append(f"| {set_name} | {label} | {len(d)} | {d['ml_error'].mean():.3f} | {d['ml_error'].median():.3f} |")

    return "\n".join(lines)


def render_noise_context():
    lines = ["## Render-noise null band (context, from Phase 1/3 replicates)", ""]
    maes = []
    for i, path in enumerate(REPLICATE_RESULTS):
        rdf = pd.read_csv(path)
        valid = rdf.dropna(subset=["ml_left_angle", "ml_right_angle", "gt_angle"])
        err = pd.concat([
            (valid["ml_left_angle"] - valid["gt_angle"]).abs(),
            (valid["ml_right_angle"] - valid["gt_angle"]).abs(),
        ])
        maes.append(err.mean())
        lines.append(f"replicate_{i}: MAE={err.mean():.4f}")
    lines.append(f"Band: [{min(maes):.4f}, {max(maes):.4f}], SD={np.std(maes):.4f}")
    return "\n".join(lines), (min(maes), max(maes))


def main():
    print("Building merged per-measurement table...")
    df = build_measurements()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(MEASUREMENTS_CSV, index=False)
    print(f"Wrote {len(df)} measurement rows to {MEASUREMENTS_CSV}")

    print("Render-noise band...")
    noise_text, noise_band = render_noise_context()

    print("Primary endpoint (grid-distance slope + bootstrap)...")
    primary_text, primary_stats = primary_endpoint(df)

    print("Paired matching...")
    paired_text = paired_matching(df)

    print("CP mediation...")
    cp_text = cp_mediation(df)

    print("Tail metrics...")
    tail_text = tail_metrics(df)

    md = "\n\n".join([
        "# Phase 4 -- comparative analysis",
        "",
        f"Pooled off-grid (n_images={df[df['set']=='offgrid']['image_name'].nunique()}) + "
        f"on-grid-control (n_images={df[df['set']=='ongrid_control']['image_name'].nunique()}).",
        noise_text,
        primary_text,
        paired_text,
        cp_text,
        tail_text,
    ])
    with open(SUMMARY_MD, "w") as f:
        f.write(md)

    print("\n" + "=" * 60)
    print(md)
    print("=" * 60)
    print(f"\nSummary written to {SUMMARY_MD}")
    print(f"Paired matches written to {os.path.join(RESULTS_DIR, 'phase4_paired_matches.csv')}")


if __name__ == "__main__":
    main()
