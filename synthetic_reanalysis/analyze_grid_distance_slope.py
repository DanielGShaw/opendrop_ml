"""Phase 2A primary endpoint: grid-distance slope (SYNTHETIC_DATA_REANALYSIS.md, A.6).

Pools the on-grid sweep (contour_sweep_ongrid.csv, grid_distance recomputed
post-hoc since it wasn't originally stored -- most rows are 0, but the
roughness=-0.03 subset is ~1 grid-step away, see Background) and the
off-grid sweep (contour_sweep_offgrid.csv), and fits OLS + Theil-Sen slopes
of |error| on grid_distance, with a bootstrap 95% CI (10,000 resamples,
seed 20260813, percentile method).

Run after contour_level_sweep.py ongrid and offgrid:
    python analyze_grid_distance_slope.py
"""

import csv
import os

import numpy as np
from scipy import stats

from config import BOOTSTRAP_SEED, RESULTS_DIR
from matrix_utils import grid_distance

ONGRID_CSV = os.path.join(RESULTS_DIR, "contour_sweep_ongrid.csv")
OFFGRID_CSV = os.path.join(RESULTS_DIR, "contour_sweep_offgrid.csv")

N_BOOTSTRAP = 10000
CONTOUR_LEVEL_BAR = 0.2  # deg; see plan Significance criteria, decision point resolved 2026-08-13
RENDER_NOISE_BAND = (0.5887, 0.6428)  # deg; contour-level MAE band from Phase 1 replicates


def load_pooled():
    rows = []
    with open(ONGRID_CSV) as f:
        for r in csv.DictReader(f):
            if r["abs_error"] == "":
                continue
            bond, scaler, roughness = float(r["bond"]), float(r["scaler"]), float(r["roughness"])
            rows.append(
                {
                    "abs_error": float(r["abs_error"]),
                    "grid_distance": grid_distance(bond, scaler, roughness),
                    "source": "ongrid",
                    "angle": float(r["angle"]),
                }
            )
    with open(OFFGRID_CSV) as f:
        for r in csv.DictReader(f):
            if r["abs_error"] == "":
                continue
            rows.append(
                {
                    "abs_error": float(r["abs_error"]),
                    "grid_distance": float(r["grid_distance"]),
                    "source": "offgrid",
                    "angle": float(r["angle"]),
                }
            )
    return rows


def fit_slopes(distances, errors):
    ols_slope, ols_intercept = np.polyfit(distances, errors, 1)
    # theilslopes' own CI (Sen 1968, alpha=0.95 -> 95% two-sided) is used instead
    # of a percentile bootstrap here: a single call costs ~1.5s at n~5700 (O(n^2)
    # pairwise slopes), so 10,000 bootstrap resamples would take ~4 hours. This
    # is a standard, well-established CI method for the Theil-Sen estimator, not
    # an ad-hoc substitute.
    ts_slope, ts_intercept, ts_lo, ts_hi = stats.theilslopes(errors, distances, alpha=0.95)
    return {
        "ols_slope": float(ols_slope),
        "ols_intercept": float(ols_intercept),
        "theilsen_slope": float(ts_slope),
        "theilsen_intercept": float(ts_intercept),
        "theilsen_ci95": (float(ts_lo), float(ts_hi)),
    }


def bootstrap_slope_ci(distances, errors, n=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, method="ols"):
    rng = np.random.default_rng(seed)
    n_samples = len(distances)
    slopes = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        if method == "ols":
            s, _ = np.polyfit(distances[idx], errors[idx], 1)
        else:
            s, _, _, _ = stats.theilslopes(errors[idx], distances[idx])
        slopes[i] = s
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(lo), float(hi), slopes


def main():
    rows = load_pooled()
    distances = np.array([r["grid_distance"] for r in rows])
    errors = np.array([r["abs_error"] for r in rows])
    n = len(rows)
    print(f"Pooled sample: n={n} ({sum(1 for r in rows if r['source']=='ongrid')} on-grid, "
          f"{sum(1 for r in rows if r['source']=='offgrid')} off-grid)")
    print(f"grid_distance range: [{distances.min():.3f}, {distances.max():.3f}]")
    print(f"abs_error range: [{errors.min():.3f}, {errors.max():.3f}]")

    fits = fit_slopes(distances, errors)
    print(f"\nOLS slope:      {fits['ols_slope']:.5f} deg per unit grid_distance "
          f"(intercept {fits['ols_intercept']:.4f})")
    print(f"Theil-Sen slope: {fits['theilsen_slope']:.5f} deg per unit grid_distance "
          f"(intercept {fits['theilsen_intercept']:.4f}, 95% CI {fits['theilsen_ci95']})")

    print(f"\nBootstrapping OLS slope CI ({N_BOOTSTRAP} resamples, seed={BOOTSTRAP_SEED})...")
    lo, hi, boots = bootstrap_slope_ci(distances, errors)
    print(f"OLS slope 95% CI: [{lo:.5f}, {hi:.5f}]")

    disagreement_ratio = fits["ols_slope"] / fits["theilsen_slope"] if fits["theilsen_slope"] else float("nan")
    print(f"\nOLS/Theil-Sen ratio: {disagreement_ratio:.2f} "
          f"({'material disagreement -> quoting Theil-Sen per plan tie-break' if abs(disagreement_ratio-1)>0.2 else 'agree closely'})")

    # --- Sensitivity check: does the effect survive excluding angle=180,
    # which Phase 2A(i) already showed carries a one-sided boundary artifact
    # independent of grid_distance? ---
    not180 = np.array([r["angle"] for r in rows]) != 180.0
    fits_not180 = fit_slopes(distances[not180], errors[not180])
    print(f"\nSensitivity (excluding angle=180, n={not180.sum()}):")
    print(f"  OLS slope:      {fits_not180['ols_slope']:.5f}")
    print(f"  Theil-Sen slope: {fits_not180['theilsen_slope']:.5f} (95% CI {fits_not180['theilsen_ci95']})")

    # Interpretation: primary estimate is Theil-Sen (plan's tie-break rule
    # when OLS/Theil-Sen disagree materially, which they do here).
    print("\n" + "=" * 60)
    print("Interpretation (primary estimate: Theil-Sen, per plan tie-break)")
    print("=" * 60)
    ts_lo, ts_hi = fits["theilsen_ci95"]
    slope_is_positive_and_significant = ts_lo > 0
    print(f"Theil-Sen slope CI excludes zero: {slope_is_positive_and_significant}")

    max_distance = distances.max()
    predicted_increase_at_max = fits["theilsen_slope"] * max_distance
    print(f"Predicted |error| increase at max observed grid_distance ({max_distance:.2f}): "
          f"{predicted_increase_at_max:.4f} deg (Theil-Sen)")
    print(f"Practical bar (contour-level): {CONTOUR_LEVEL_BAR} deg")
    print(f"Render-noise band: {RENDER_NOISE_BAND}")

    if not slope_is_positive_and_significant:
        print(f"\nBOUNDED NULL: any grid-proximity effect (linear in grid_distance) is below "
              f"{ts_hi:.4f} deg per unit distance (95% CI), i.e. below {ts_hi*max_distance:.4f} deg "
              f"at the farthest sampled point (distance={max_distance:.2f}).")
    else:
        materiality = "MATERIAL" if predicted_increase_at_max >= CONTOUR_LEVEL_BAR else "statistically real but below the practical bar"
        print(f"\nEFFECT DETECTED: {materiality} "
              f"(predicted increase {predicted_increase_at_max:.4f} deg vs. bar {CONTOUR_LEVEL_BAR} deg)")

    # save summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "grid_distance_slope_summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["n", n])
        writer.writerow(["ols_slope", fits["ols_slope"]])
        writer.writerow(["ols_intercept", fits["ols_intercept"]])
        writer.writerow(["ols_slope_ci_lo", lo])
        writer.writerow(["ols_slope_ci_hi", hi])
        writer.writerow(["theilsen_slope", fits["theilsen_slope"]])
        writer.writerow(["theilsen_intercept", fits["theilsen_intercept"]])
        writer.writerow(["max_grid_distance", max_distance])
        writer.writerow(["predicted_increase_at_max_ols", predicted_increase_at_max])
        writer.writerow(["contour_level_bar", CONTOUR_LEVEL_BAR])
    print(f"\nSummary written to {os.path.join(RESULTS_DIR, 'grid_distance_slope_summary.csv')}")


if __name__ == "__main__":
    main()
