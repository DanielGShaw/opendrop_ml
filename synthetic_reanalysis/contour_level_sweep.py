"""Phase 2A contour-level sweep (SYNTHETIC_DATA_REANALYSIS.md, Phase 2A + A.6/A.7).

Bypasses the image pipeline entirely (no evaluate_image, no classical
fits, no CPID) -- exact-CP, training-format predictions only. This is the
plan's primary/decisive test of the model's shape->angle mapping, cheap
enough to sweep thousands of points rather than the ~324 of the image-level
sets.

Run order is load-bearing (see plan): run `ongrid` first, inspect the
printed decision-point inputs, finalise the practical bar, THEN run
`offgrid`.

    python contour_level_sweep.py ongrid    # 2A(i): on-grid MAE/RMSE + render-noise band
    python contour_level_sweep.py offgrid   # 2A(ii): LHS sweep + grid-distance slope (run after the decision point)
"""

import argparse
import csv
import math
import os
import time

import numpy as np

from config import LHS_SEED, RESULTS_DIR
from contour_level import get_model, predict_contour_level
from generate_offgrid_synthetic import (
    N_REPLICATES,
    REPLICATE_ANGLES,
    REPLICATE_CELLS,
    parse_ongrid_params,
)
from matrix_utils import cell_seed, grid_distance

ONGRID_CSV = os.path.join(RESULTS_DIR, "contour_sweep_ongrid.csv")
RENDER_NOISE_CSV = os.path.join(RESULTS_DIR, "contour_sweep_render_noise.csv")
OFFGRID_CSV = os.path.join(RESULTS_DIR, "contour_sweep_offgrid.csv")

VALIDATION_RMSE_BENCHMARK = 0.65  # deg; from the model's own training run (see plan Background)

ANGLES = [20, 40, 60, 80, 100, 120, 140, 160, 180]
N_IN_DOMAIN_PER_ANGLE = 500
N_OUT_DOMAIN_PER_ANGLE = 100


def mae_rmse(errors):
    errors = np.asarray(errors, dtype=float)
    return float(np.mean(errors)), float(np.sqrt(np.mean(errors**2)))


# ---------------------------------------------------------------------------
# 2A(i): on-grid portion + render-noise band
# ---------------------------------------------------------------------------

def run_ongrid():
    model = get_model()
    params = parse_ongrid_params()
    print(f"On-grid contour-level sweep: {len(params)} parameter tuples")

    rows = []
    t0 = time.time()
    for i, (angle, bond, scaler, roughness) in enumerate(params):
        try:
            pred = predict_contour_level(angle, bond, scaler, roughness, dpi=64, model=model)
            err = abs(pred - angle)
            rows.append(
                {"angle": angle, "bond": bond, "scaler": scaler, "roughness": roughness,
                 "pred": pred, "abs_error": err, "error": ""}
            )
        except Exception as e:
            rows.append(
                {"angle": angle, "bond": bond, "scaler": scaler, "roughness": roughness,
                 "pred": "", "abs_error": "", "error": f"{type(e).__name__}: {e}"}
            )
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(params)}  ({time.time() - t0:.0f}s)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(ONGRID_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["angle", "bond", "scaler", "roughness", "pred", "abs_error", "error"])
        writer.writeheader()
        writer.writerows(rows)

    errors = [r["abs_error"] for r in rows if r["abs_error"] != ""]
    n_failed = len(rows) - len(errors)
    mae, rmse = mae_rmse(errors)
    print(f"\nOn-grid contour-level: n={len(errors)} (failed={n_failed})  MAE={mae:.4f}  RMSE={rmse:.4f} deg")
    print(f"Model's own validation RMSE benchmark: {VALIDATION_RMSE_BENCHMARK} deg")
    print(f"Written to {ONGRID_CSV}")
    return mae, rmse


def run_render_noise():
    model = get_model()
    replicate_params = [
        (angle, bond, scaler, roughness)
        for (bond, scaler, roughness) in REPLICATE_CELLS
        for angle in REPLICATE_ANGLES
    ]
    print(f"Render-noise band: {N_REPLICATES} replicates x {len(replicate_params)} cells")

    rows = []
    replicate_maes = []
    for r in range(N_REPLICATES):
        errs = []
        for angle, bond, scaler, roughness in replicate_params:
            seed = cell_seed(angle, bond, scaler, roughness) + r
            try:
                pred = predict_contour_level(angle, bond, scaler, roughness, dpi=64, seed=seed, model=model)
                err = abs(pred - angle)
                errs.append(err)
                rows.append(
                    {"replicate": r, "angle": angle, "bond": bond, "scaler": scaler,
                     "roughness": roughness, "pred": pred, "abs_error": err, "error": ""}
                )
            except Exception as e:
                rows.append(
                    {"replicate": r, "angle": angle, "bond": bond, "scaler": scaler,
                     "roughness": roughness, "pred": "", "abs_error": "",
                     "error": f"{type(e).__name__}: {e}"}
                )
        mae_r, _ = mae_rmse(errs)
        replicate_maes.append(mae_r)
        print(f"  replicate {r}: MAE={mae_r:.4f} deg (n={len(errs)})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RENDER_NOISE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["replicate", "angle", "bond", "scaler", "roughness", "pred", "abs_error", "error"]
        )
        writer.writeheader()
        writer.writerows(rows)

    band = (min(replicate_maes), max(replicate_maes))
    sd = float(np.std(replicate_maes, ddof=1))
    print(f"\nRender-noise band (contour-level MAE across {N_REPLICATES} replicates): "
          f"[{band[0]:.4f}, {band[1]:.4f}] deg, SD={sd:.4f}")
    print(f"Written to {RENDER_NOISE_CSV}")
    return band, sd, replicate_maes


def cmd_ongrid():
    print("=" * 60)
    print("Phase 2A(i): on-grid contour-level sweep")
    print("=" * 60)
    mae, rmse = run_ongrid()

    print()
    print("=" * 60)
    print("Render-noise band (contour-level, from Phase-1 replicates)")
    print("=" * 60)
    band, sd, replicate_maes = run_render_noise()

    print()
    print("=" * 60)
    print("Decision-point inputs (see plan doc: finalise the practical bar before running `offgrid`)")
    print("=" * 60)
    print(f"  On-grid contour-level MAE:  {mae:.4f} deg")
    print(f"  On-grid contour-level RMSE: {rmse:.4f} deg")
    print(f"  Validation RMSE benchmark:  {VALIDATION_RMSE_BENCHMARK} deg")
    print(f"  Render-noise band:          [{band[0]:.4f}, {band[1]:.4f}] deg (SD={sd:.4f})")
    print(f"  Replicate MAEs:             {[round(x, 4) for x in replicate_maes]}")


# ---------------------------------------------------------------------------
# 2A(ii): off-grid LHS sweep
# ---------------------------------------------------------------------------

def lhs_unit(rng, n, d):
    """Standard Latin Hypercube sample in [0,1)^d: n stratified, shuffled samples per dim."""
    result = np.empty((n, d))
    for j in range(d):
        cut = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(cut)
        result[:, j] = cut
    return result


def sample_in_domain(rng, n):
    """LHS over Bo in [0,1], log-scaler in [log1, log4], |roughness| within a
    50/50-chosen regime. Returns list of (bond, scaler, roughness) dicts."""
    u = lhs_unit(rng, n, 3)
    bonds = u[:, 0]  # [0, 1]
    scalers = np.exp(u[:, 1] * (math.log(4.0) - math.log(1.0)) + math.log(1.0))  # [1, 4]
    signs = rng.integers(0, 2, size=n)  # 0=positive regime, 1=negative regime
    roughnesses = np.where(
        signs == 0,
        u[:, 2] * 0.01,           # positive regime: [0, 0.01]
        -(u[:, 2] * 0.06 + 0.0),  # negative regime: [-0.06, 0] -- reuse same u col, fine for LHS-ish coverage
    )
    return list(zip(bonds, scalers, roughnesses, [False] * n, ["in_domain"] * n))


def sample_out_domain(rng, n):
    """Uniform draw, one axis out-of-domain at a time (matching the 36-cell
    matrix's design), other two axes drawn in-domain uniformly."""
    axes = rng.choice(
        ["bond", "scaler_high", "scaler_low", "roughness_pos", "roughness_neg"], size=n
    )
    out = []
    for axis in axes:
        bond = rng.uniform(0.0, 1.0)
        scaler = math.exp(rng.uniform(math.log(1.0), math.log(4.0)))
        sign = rng.integers(0, 2)
        roughness = rng.uniform(0.0, 0.01) if sign == 0 else -rng.uniform(0.0, 0.06)

        if axis == "bond":
            bond = rng.uniform(1.0, 1.5)
        elif axis == "scaler_high":
            scaler = rng.uniform(4.0, 6.0)
        elif axis == "scaler_low":
            scaler = rng.uniform(0.9, 1.0)  # 0.9 = empirically-confirmed safe floor, see Phase 1 findings
        elif axis == "roughness_pos":
            roughness = rng.uniform(0.01, 0.02)
        elif axis == "roughness_neg":
            roughness = -rng.uniform(0.06, 0.08)

        out.append((bond, scaler, roughness, True, axis))
    return out


def build_offgrid_sweep():
    rng = np.random.default_rng(LHS_SEED)
    samples = []
    for angle in ANGLES:
        in_dom = sample_in_domain(rng, N_IN_DOMAIN_PER_ANGLE)
        out_dom = sample_out_domain(rng, N_OUT_DOMAIN_PER_ANGLE)
        for bond, scaler, roughness, out_of_domain, tag in in_dom + out_dom:
            samples.append(
                {
                    "angle": angle, "bond": bond, "scaler": scaler, "roughness": roughness,
                    "out_of_domain": out_of_domain, "ood_tag": tag,
                    "grid_distance": grid_distance(bond, scaler, roughness),
                }
            )
    return samples


def run_offgrid():
    model = get_model()
    samples = build_offgrid_sweep()
    print(f"Off-grid contour-level sweep: {len(samples)} samples "
          f"({N_IN_DOMAIN_PER_ANGLE} in-domain + {N_OUT_DOMAIN_PER_ANGLE} out-of-domain per angle x {len(ANGLES)} angles)")

    rows = []
    t0 = time.time()
    n_failed = 0
    for i, s in enumerate(samples):
        angle, bond, scaler, roughness = s["angle"], s["bond"], s["scaler"], s["roughness"]
        try:
            pred = predict_contour_level(angle, bond, scaler, roughness, dpi=64, model=model)
            err = abs(pred - angle)
            rows.append({**s, "pred": pred, "abs_error": err, "error": ""})
        except Exception as e:
            n_failed += 1
            rows.append({**s, "pred": "", "abs_error": "", "error": f"{type(e).__name__}: {e}"})

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(samples) - i - 1)
            print(f"  {i + 1}/{len(samples)}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fieldnames = ["angle", "bond", "scaler", "roughness", "out_of_domain", "ood_tag",
                  "grid_distance", "pred", "abs_error", "error"]
    with open(OFFGRID_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone in {time.time() - t0:.0f}s. {n_failed} failed. Written to {OFFGRID_CSV}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["ongrid", "offgrid"])
    args = parser.parse_args()
    if args.phase == "ongrid":
        cmd_ongrid()
    else:
        run_offgrid()
