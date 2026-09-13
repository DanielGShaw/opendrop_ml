"""Phase 1 generator (SYNTHETIC_DATA_REANALYSIS.md, Phase 1 + Appendix A.1-A.5).

Renders:
  - paper_results/data/synthetic_offgrid/       — the 324-cell off-grid matrix (built by
                                     offgrid_matrix.py)
  - paper_results/data/synthetic_ongrid_control/ — a fresh re-render of the *existing*
                                     on-grid parameter combinations (parsed
                                     from paper_results/data/synthetic/
                                     filenames), for the render-noise band
  - paper_results/data/synthetic_ongrid_control/replicate_{0..4}/ — 5 replicates of an
    8-cell x 9-angle subset (72 images each), for the render-noise band

Run (after `python offgrid_matrix.py`, and only after acceptance_gate.py
has passed):
    python generate_offgrid_synthetic.py
"""

import csv
import os
import re
import sys

_PAPER_RESULTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PAPER_RESULTS_DIR)  # paper_results/, for config
from config import (  # noqa: E402
    OFFGRID_DPI as DPI,
    OFFGRID_MATRIX_CSV as MATRIX_CSV,
    OFFGRID_N_PTS_DROP as N_PTS_DROP,
    OFFGRID_RESULTS_DIR as RESULTS_DIR,
    SYNTHETIC_DATA_DIR as SYNTHETIC_ONGRID_DATA_DIR,
    SYNTHETIC_OFFGRID_DATA_DIR,
    SYNTHETIC_ONGRID_CONTROL_DIR,
    TRAINING_FILES_DIR,
)

sys.path.insert(0, TRAINING_FILES_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from create_contours_args6 import draw_drop, extract_edges_CV, prepare_contour  # noqa: E402
from matrix_utils import cell_seed
from offgrid_matrix import MATRIX_COLUMNS

RESOLUTION_FLOOR_PX = 2  # degenerate-render guard only, not a resolution filter — see
# SYNTHETIC_DATA_REANALYSIS.md Findings: the scaler axis exists to simulate camera-
# resolution differences, so filtering *out* low drop_height_px would remove exactly
# the real-world-relevant regime under test. drop_height_px is instead recorded on
# every row (pass or fail) as a covariate for Phase 4 to stratify/weight by. With a
# minimum angle of 20 deg, drop_height_px should never legitimately hit single digits;
# this floor only catches genuinely degenerate renders (e.g. a rendering bug), not
# "just small" drops.
WINDOW_DROP_POINTS = 40  # window_size(50) - surface_points(10); see A.4 note 4
FILENAME_RE = re.compile(
    r"^(.+?)_CPL_(\d+)_(\d+)_CPR_(\d+)_(\d+)_?$"
)


class RenderCheckFailed(Exception):
    """Carries drop_height_px (when computable) alongside the failure
    message, so it can be recorded as a covariate even for dropped cells."""

    def __init__(self, msg, drop_height_px=None):
        super().__init__(msg)
        self.drop_height_px = drop_height_px


def render_and_check(angle, bond, scaler, roughness, seed):
    """Render one image and run the A.4 asserts. Returns (img, CPs_img,
    filename, drop_height_px) on success, or raises RenderCheckFailed
    (.drop_height_px set when computable) identifying which check failed
    (caller catches and records it). drop_height_px is a covariate, not
    (beyond RESOLUTION_FLOOR_PX) a filter — see the RESOLUTION_FLOOR_PX
    comment above."""
    np.random.seed(seed)  # immediately before draw_drop — see A.3
    img, CPs_img = draw_drop(angle, bond, scaler, roughness, N_PTS_DROP, dpi=DPI)

    contour = extract_edges_CV(img)
    h, w = img.shape[:2]
    (lx, ly), (rx, ry) = CPs_img

    # Computed unconditionally, before any assert, so it's available as a
    # covariate even when a later check fails.
    drop_height_px = max(ly, ry) - contour[:, 1].min()

    # 2. Contact points inside the frame (checked before 1 — needed to
    #    define the drop-only mask below).
    if not (0 < lx < w and 0 < ly < h):
        raise RenderCheckFailed("left CP outside frame", drop_height_px)
    if not (0 < rx < w and 0 < ry < h):
        raise RenderCheckFailed("right CP outside frame", drop_height_px)

    # 1. Frame geometry — the DROP must not touch any frame edge. The
    #    baseline strip legitimately spans the full frame width by
    #    construction (draw_drop fills it edge-to-edge every render), so
    #    checking the whole contour's x-extent against the frame (as
    #    Appendix A.4 literally states) rejects every image, including
    #    known-good ones from the existing paper dataset (verified against
    #    the acceptance-gate reference: contour x range is [2, 742] on a
    #    744px-wide frame). Restrict the x-edge check to points above the
    #    baseline row (smaller pixel y = higher in the image) — both CPs
    #    sit at the baseline by construction, so their y is the threshold.
    #    (IGNORE_EDGE_MARGIN in extract_edges_CV silently truncates rather
    #    than raising, so this must be checked explicitly.)
    baseline_row = min(ly, ry)
    drop_mask = contour[:, 1] < baseline_row - 2
    if not drop_mask.any():
        raise RenderCheckFailed("no drop-only contour points above baseline", drop_height_px)
    drop_contour = contour[drop_mask]
    if not (drop_contour[:, 0].min() > 1 and drop_contour[:, 0].max() < w - 2):
        raise RenderCheckFailed("drop touches x-edge", drop_height_px)
    if not (drop_contour[:, 1].min() > 1):
        raise RenderCheckFailed("drop touches y-edge (top)", drop_height_px)

    # 3. Degenerate-render guard (not a resolution filter — see comment on
    #    RESOLUTION_FLOOR_PX above).
    if not (drop_height_px >= RESOLUTION_FLOOR_PX):
        raise RenderCheckFailed(
            f"drop_height_px={drop_height_px:.1f} < {RESOLUTION_FLOOR_PX} (degenerate render)",
            drop_height_px,
        )

    # 4. Window feasibility — right-side contour must have >= 40 points
    #    before the CP (sampling() wraps around silently otherwise; see A.7).
    CPs_for_prep = [list(CPs_img[0]), list(CPs_img[1])]
    right, CPs_prepared = prepare_contour(contour, CPs_for_prep, given_input_len=None, right_only=True)
    cp_match = np.where(np.all(right == np.array(CPs_prepared[1]), axis=1))[0]
    if not len(cp_match) > 0:
        raise RenderCheckFailed("right-side CP not found in prepared contour", drop_height_px)
    cp_index = int(cp_match[0])
    if not (cp_index >= WINDOW_DROP_POINTS):
        raise RenderCheckFailed(f"cp_index={cp_index} < {WINDOW_DROP_POINTS}", drop_height_px)

    filename = (
        f"{angle:.1f}_{bond:.4f}_{scaler:.4f}_{roughness:.4f}"
        f"_CPL_{lx}_{ly}_CPR_{rx}_{ry}_.png"
    )
    return img, CPs_img, filename, drop_height_px


def generate_matrix(matrix_rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    seen_filenames = set()
    n_ok, n_dropped = 0, 0

    for row in matrix_rows:
        angle, bond, scaler, roughness = row["angle"], row["bond"], row["scaler"], row["roughness"]
        seed = int(row["seed"])
        try:
            img, CPs_img, filename, drop_height_px = render_and_check(
                angle, bond, scaler, roughness, seed
            )
        except RenderCheckFailed as e:
            row["status"] = "dropped"
            row["drop_reason"] = str(e)
            row["drop_height_px"] = e.drop_height_px if e.drop_height_px is not None else ""
            n_dropped += 1
            continue
        except Exception as e:
            # prepare_contour() can raise IndexError on degenerate geometry
            # (e.g. all contour points falling on one side of the computed
            # apex x-position) rather than a clean check failure — this is
            # itself a finding (a pre-existing bug in the shared training
            # pipeline exposed by unusual parameter combinations), not
            # something to let crash the whole batch.
            row["status"] = "dropped"
            row["drop_reason"] = f"{type(e).__name__}: {e}"
            row["drop_height_px"] = ""
            n_dropped += 1
            continue

        assert filename not in seen_filenames, f"filename collision: {filename}"
        seen_filenames.add(filename)

        cv2.imwrite(os.path.join(out_dir, filename), img)
        row["filename"] = filename
        row["status"] = "ok"
        row["drop_height_px"] = drop_height_px
        n_ok += 1

    return n_ok, n_dropped


def write_updated_matrix(rows, path=MATRIX_CSV):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATRIX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_ongrid_params():
    """Parse (angle, bond, scaler, roughness) tuples out of the existing
    on-grid set's filenames — the parameters, not the CP labels (those get
    re-derived from the fresh render)."""
    params = []
    for fname in sorted(os.listdir(SYNTHETIC_ONGRID_DATA_DIR)):
        if not fname.lower().endswith(".png"):
            continue
        name = fname[:-4]
        m = FILENAME_RE.match(name)
        if not m:
            continue
        parts = m.group(1).split("_")
        assert len(parts) == 4, f"unexpected on-grid filename: {fname}"
        angle, bond, scaler, roughness = (float(p) for p in parts)
        params.append((angle, bond, scaler, roughness))
    return params


def generate_ongrid_control(params, out_dir, replicate_index=None):
    """Render the given (angle, bond, scaler, roughness) tuples fresh.
    replicate_index, if given, offsets the seed (A.3) so replicates differ
    but the whole set still regenerates identically."""
    os.makedirs(out_dir, exist_ok=True)
    n_ok, n_dropped = 0, 0
    rows = []
    for angle, bond, scaler, roughness in params:
        seed = cell_seed(angle, bond, scaler, roughness)
        if replicate_index is not None:
            seed += replicate_index
        try:
            img, CPs_img, filename, drop_height_px = render_and_check(
                angle, bond, scaler, roughness, seed
            )
        except RenderCheckFailed as e:
            n_dropped += 1
            rows.append(
                {"angle": angle, "bond": bond, "scaler": scaler, "roughness": roughness,
                 "seed": seed, "filename": "", "status": "dropped",
                 "drop_reason": str(e),
                 "drop_height_px": e.drop_height_px if e.drop_height_px is not None else ""}
            )
            continue
        except Exception as e:
            n_dropped += 1
            rows.append(
                {"angle": angle, "bond": bond, "scaler": scaler, "roughness": roughness,
                 "seed": seed, "filename": "", "status": "dropped",
                 "drop_reason": f"{type(e).__name__}: {e}", "drop_height_px": ""}
            )
            continue
        cv2.imwrite(os.path.join(out_dir, filename), img)
        n_ok += 1
        rows.append(
            {"angle": angle, "bond": bond, "scaler": scaler, "roughness": roughness,
             "seed": seed, "filename": filename, "status": "ok", "drop_reason": "",
             "drop_height_px": drop_height_px}
        )
    return rows, n_ok, n_dropped


# 8 on-grid (bond, scaler, roughness) cells x all 9 angles, for the
# render-noise-band replicates (A.5). Chosen to span the existing set's
# Bond/scaler/roughness values.
REPLICATE_CELLS = [
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.006),
    (0.2, 4.0, -0.03),
    (0.4, 1.0, 0.006),
    (0.6, 4.0, 0.0),
    (0.8, 1.0, -0.03),
    (1.0, 4.0, 0.006),
    (1.0, 1.0, -0.03),
]
REPLICATE_ANGLES = [20, 40, 60, 80, 100, 120, 140, 160, 180]
N_REPLICATES = 5


def main():
    from offgrid_matrix import build_matrix, write_matrix_csv

    print("=" * 60)
    print("Phase 1: off-grid matrix")
    print("=" * 60)
    matrix_rows = build_matrix()
    n_ok, n_dropped = generate_matrix(matrix_rows, SYNTHETIC_OFFGRID_DATA_DIR)
    write_updated_matrix(matrix_rows)
    print(f"Off-grid: {n_ok} rendered, {n_dropped} dropped (see {MATRIX_CSV})")

    print()
    print("=" * 60)
    print("On-grid control (fresh render, existing parameters)")
    print("=" * 60)
    ongrid_params = parse_ongrid_params()
    print(f"Parsed {len(ongrid_params)} parameter tuples from {SYNTHETIC_ONGRID_DATA_DIR}")
    control_rows, n_ok, n_dropped = generate_ongrid_control(
        ongrid_params, SYNTHETIC_ONGRID_CONTROL_DIR
    )
    with open(os.path.join(RESULTS_DIR, "ongrid_control_manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["angle", "bond", "scaler", "roughness", "seed", "filename", "status", "drop_reason", "drop_height_px"]
        )
        writer.writeheader()
        writer.writerows(control_rows)
    print(f"Control: {n_ok} rendered, {n_dropped} dropped")

    print()
    print("=" * 60)
    print(f"Render-noise replicates ({N_REPLICATES}x, 8 cells x 9 angles = 72 images each)")
    print("=" * 60)
    replicate_params = [
        (angle, bond, scaler, roughness)
        for (bond, scaler, roughness) in REPLICATE_CELLS
        for angle in REPLICATE_ANGLES
    ]
    for r in range(N_REPLICATES):
        rep_dir = os.path.join(SYNTHETIC_ONGRID_CONTROL_DIR, f"replicate_{r}")
        rows, n_ok, n_dropped = generate_ongrid_control(replicate_params, rep_dir, replicate_index=r)
        with open(os.path.join(RESULTS_DIR, f"replicate_{r}_manifest.csv"), "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["angle", "bond", "scaler", "roughness", "seed", "filename", "status", "drop_reason", "drop_height_px"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"  replicate {r}: {n_ok} rendered, {n_dropped} dropped")


if __name__ == "__main__":
    main()
