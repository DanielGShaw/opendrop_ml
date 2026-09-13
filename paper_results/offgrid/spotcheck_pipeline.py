"""Phase 2B image-pipeline smoke test (SYNTHETIC_DATA_REANALYSIS.md, Phase 2B).

Runs the full `evaluate_image()` pipeline (isolate_contour + CPID + all
classical fits + ML) on a small subset of off-grid images spanning the
matrix (interpolation + extrapolation, several angles), plus a matched
subset from the on-grid control. This is explicitly a smoke test and a
large-effect detector, not a confirmation -- Phase 2A (contour-level, 5400
samples) is the statistically powered result. The point here is to check
for CPID / edge-detection failures at extreme roughness or scaler, sane
profile lengths, and no frame-clipping, at image-pipeline resolution.

Run from OpenDrop_OP/paper_results/offgrid/:
    python spotcheck_pipeline.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd

_PAPER_RESULTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PAPER_RESULTS_DIR)  # paper_results/, for config
from config import (  # noqa: E402
    OFFGRID_RESULTS_DIR as RESULTS_DIR,
    OPENDROP_DIR,
    PAPER_RESULTS_DIR,
    SYNTHETIC_OFFGRID_DATA_DIR,
    SYNTHETIC_ONGRID_CONTROL_DIR,
)

MATRIX_CSV = os.path.join(RESULTS_DIR, "offgrid_matrix.csv")
ONGRID_MANIFEST_CSV = os.path.join(RESULTS_DIR, "ongrid_control_manifest.csv")
SUMMARY_MD = os.path.join(RESULTS_DIR, "phase2b_smoke_test.md")

# Cells per ood_axis group to draw into the off-grid subset (sums to 17,
# inside the plan's ~15-20 range) -- every group represented, in_domain
# weighted slightly higher since it covers 24 cells vs 1-4 per OOD group.
GROUP_COUNTS = {
    "in_domain": 4,
    "bond": 3,
    "scaler": 3,
    "roughness": 3,
    "compound_bond_scaler": 2,
    "compound_scaler_roughness": 2,
}


def _import_evaluate_synthetic():
    """Import paper_results/evaluate/evaluate_synthetic.py.

    It does `from config import (...)` at import time; since this package
    now lives under paper_results/ and shares that config module, the
    only thing needed is paper_results/evaluate/ on sys.path.
    """
    sys.path.insert(0, os.path.join(PAPER_RESULTS_DIR, "evaluate"))
    import evaluate_synthetic as es
    return es


def select_offgrid_subset(matrix):
    matrix = matrix[matrix["status"] == "ok"].copy()
    matrix["group"] = matrix["ood_axis"].fillna("in_domain")

    picked = []
    for group, n in GROUP_COUNTS.items():
        sub = matrix[matrix["group"] == group].sort_values("angle")
        if sub.empty:
            continue
        idx = np.unique(np.linspace(0, len(sub) - 1, n).round().astype(int))
        picked.append(sub.iloc[idx])
    subset = pd.concat(picked).drop_duplicates(subset="filename").reset_index(drop=True)
    return subset


def select_ongrid_subset(ongrid_manifest, offgrid_subset):
    ongrid_manifest = ongrid_manifest[ongrid_manifest["status"] == "ok"].copy()
    combos = ongrid_manifest[["bond", "scaler", "roughness"]].drop_duplicates().reset_index(drop=True)

    rows = []
    seen = set()
    for i, angle in enumerate(offgrid_subset["angle"].tolist()):
        # Cycle through distinct on-grid (bond, scaler, roughness) combos so
        # the matched subset also spans the on-grid parameter space, not
        # just one corner of it repeated at different angles.
        for offset in range(len(combos)):
            combo = combos.iloc[(i + offset) % len(combos)]
            match = ongrid_manifest[
                (ongrid_manifest["angle"] == angle)
                & (ongrid_manifest["bond"] == combo["bond"])
                & (ongrid_manifest["scaler"] == combo["scaler"])
                & (ongrid_manifest["roughness"] == combo["roughness"])
            ]
            if not match.empty and match.iloc[0]["filename"] not in seen:
                rows.append(match.iloc[0])
                seen.add(match.iloc[0]["filename"])
                break
    return pd.DataFrame(rows).reset_index(drop=True)


def _parse_image_shape(shape_str):
    if not isinstance(shape_str, str):
        return None
    m = re.match(r"\((\d+),\s*(\d+)", shape_str)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))  # (h, w)


def check_frame_clipping(row, edge_margin=3):
    """Heuristic frame-clipping flag: any recorded CP within `edge_margin`
    px of the image border. evaluate_image() doesn't expose the full
    contour, so this uses the CP coordinates it does record as a proxy --
    consistent with a smoke test's role as a large-effect detector."""
    shape = _parse_image_shape(row.get("image_shape"))
    if shape is None:
        return None
    h, w = shape
    xs = [row.get("left_CP_x"), row.get("right_CP_x")]
    ys = [row.get("left_CP_y"), row.get("right_CP_y")]
    for x in xs:
        if pd.notna(x) and (x <= edge_margin or x >= w - edge_margin):
            return True
    for y in ys:
        if pd.notna(y) and (y <= edge_margin or y >= h - edge_margin):
            return True
    return False


def run_subset(es, model, data_dir, subset_df, label):
    results = []
    n = len(subset_df)
    for i, row in enumerate(subset_df.itertuples(index=False)):
        fname = row.filename
        img_path = os.path.join(data_dir, fname)
        print(f"  [{label} {i + 1}/{n}] {fname}")
        try:
            gt = es.parse_synthetic_filename(fname)
        except ValueError as e:
            results.append({"image_name": fname, "error": f"Filename parse failed: {e}"})
            continue
        try:
            result = es.evaluate_image(img_path, model, gt)
        except Exception as e:
            result = {"image_name": fname, "error": f"evaluate_image raised: {e}"}
            result.update(gt)
        result["set"] = label
        result["frame_clipping"] = check_frame_clipping(result)
        results.append(result)
    return pd.DataFrame(results)


def summarize(df, label):
    lines = [f"### {label} (n={len(df)})"]

    hard_failures = df[df["error"].notna()] if "error" in df.columns else pd.DataFrame()
    lines.append(f"- Hard failures (preprocessing/evaluate_image exception): {len(hard_failures)}/{len(df)}")
    for _, row in hard_failures.iterrows():
        lines.append(f"    - `{row['image_name']}`: {str(row['error'])[:160]}")

    for method in ["tangent", "polynomial", "circular", "ellipse", "yl", "ml"]:
        err_col = f"{method}_error"
        if err_col in df.columns:
            n_fail = df[err_col].notna().sum()
            if n_fail:
                lines.append(f"- `{method}` fit failures: {n_fail}/{len(df)}")

    if "profile_length" in df.columns:
        pl = df["profile_length"].dropna()
        if len(pl):
            lines.append(
                f"- profile_length: min={pl.min():.0f}, median={pl.median():.0f}, "
                f"max={pl.max():.0f} "
                f"(reference thresholds from prepare_experimental.py: "
                f"<234 low-res warning, 565-974 conditional downsample, >974 auto-downsample)"
            )
            n_low = (pl < 234).sum()
            n_high = (pl > 974).sum()
            if n_low:
                lines.append(f"    - {n_low} image(s) below the 234-point low-resolution warning threshold")
            if n_high:
                lines.append(f"    - {n_high} image(s) above the 974-point auto-downsample threshold")

    if "frame_clipping" in df.columns:
        n_clip = (df["frame_clipping"] == True).sum()  # noqa: E712
        if n_clip:
            lines.append(f"- Possible frame-clipping (CP within 3px of image edge): {n_clip}/{len(df)}")
            for _, row in df[df["frame_clipping"] == True].iterrows():  # noqa: E712
                lines.append(f"    - `{row['image_name']}`")

    if "ml_left_angle" in df.columns and "ml_right_angle" in df.columns and "gt_angle" in df.columns:
        valid = df.dropna(subset=["ml_left_angle", "ml_right_angle", "gt_angle"])
        if len(valid):
            ml_avg = (valid["ml_left_angle"] + valid["ml_right_angle"]) / 2.0
            ml_err = (ml_avg - valid["gt_angle"]).abs()
            lines.append(
                f"- ML angle MAE (avg of L+R vs. gt_angle): {ml_err.mean():.2f} deg "
                f"(median {ml_err.median():.2f}, max {ml_err.max():.2f}, n={len(valid)})"
            )

    return "\n".join(lines)


def main():
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    es = _import_evaluate_synthetic()

    matrix = pd.read_csv(MATRIX_CSV)
    ongrid_manifest = pd.read_csv(ONGRID_MANIFEST_CSV)

    offgrid_subset = select_offgrid_subset(matrix)
    ongrid_subset = select_ongrid_subset(ongrid_manifest, offgrid_subset)

    print(f"Off-grid subset: {len(offgrid_subset)} images across groups "
          f"{sorted(offgrid_subset['group'].unique().tolist())}, "
          f"angles {sorted(offgrid_subset['angle'].unique().tolist())}")
    print(f"On-grid control subset: {len(ongrid_subset)} images, "
          f"angles {sorted(ongrid_subset['angle'].unique().tolist())}")

    print("Loading ML model...")
    model = tf.keras.models.load_model(es.ML_MODEL_DIR)
    print("Model loaded.")

    print("\nRunning off-grid subset through evaluate_image()...")
    offgrid_results = run_subset(es, model, SYNTHETIC_OFFGRID_DATA_DIR, offgrid_subset, "offgrid")

    print("\nRunning on-grid control subset through evaluate_image()...")
    ongrid_results = run_subset(es, model, SYNTHETIC_ONGRID_CONTROL_DIR, ongrid_subset, "ongrid_control")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    md = [
        "# Phase 2B — image-pipeline smoke test",
        "",
        "Smoke test / large-effect detector only (n~17-18 per set) -- see "
        "Phase 2A (contour-level, 5400 samples) for the statistically "
        "powered primary endpoint. Go/no-go: renderer or preprocessing "
        "failures here mean fix the generator or trim the matrix before "
        "Phase 3, not chase a formal comparison at this n.",
        "",
        summarize(offgrid_results, "Off-grid subset"),
        "",
        summarize(ongrid_results, "On-grid control subset"),
        "",
    ]
    summary_text = "\n".join(md)
    with open(SUMMARY_MD, "w") as f:
        f.write(summary_text)

    print("\n" + "=" * 60)
    print(summary_text)
    print("=" * 60)
    print(f"\nSummary written to {SUMMARY_MD}")


if __name__ == "__main__":
    main()
