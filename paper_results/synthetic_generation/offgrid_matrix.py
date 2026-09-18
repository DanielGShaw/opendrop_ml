"""Build the off-grid parameter matrix (SYNTHETIC_DATA_REANALYSIS.md, Appendix A.1).

36 (Bo, scaler, roughness) cells x 9 fixed angles = up to 324 images,
matching the existing on-grid set's size. Cells are dropped (not
substituted) if they fail Phase-1 asserts in generate_offgrid_synthetic.py;
this module only builds the intended matrix.

Run standalone to (re)write results/synthetic_matrix.csv:
    python offgrid_matrix.py
"""

import csv
import os
import sys

_PAPER_RESULTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PAPER_RESULTS_DIR)  # paper_results/, for config
from config import SYNTHETIC_MATRIX_CSV as MATRIX_CSV  # noqa: E402
from matrix_utils import cell_seed, grid_distance, roughness_regime

ANGLES = [20, 40, 60, 80, 100, 120, 140, 160, 180]

# --- Interpolation block: 3 Bo x 2 scaler x 4 roughness = 24 cells, full factorial ---
INTERP_BO = [0.15, 0.55, 0.95]
INTERP_SCALER = [1.41, 2.83]
INTERP_ROUGHNESS = [0.003, 0.009, -0.015, -0.045]

# --- Extrapolation block: 12 cells, one axis out-of-domain at a time + 2 compound corners ---
# (bond, scaler, roughness, ood_axis)
EXTRAP_CELLS = [
    (1.2, 1.41, 0.003, "bond"),
    (1.5, 1.41, 0.003, "bond"),
    (1.2, 1.41, -0.015, "bond"),
    (1.5, 1.41, -0.015, "bond"),
    (0.55, 5.0, 0.003, "scaler"),
    (0.55, 0.95, 0.003, "scaler"),
    (0.55, 5.0, -0.015, "scaler"),
    (0.55, 0.95, -0.015, "scaler"),
    (0.55, 1.41, 0.02, "roughness"),
    (0.55, 1.41, -0.08, "roughness"),
    (1.5, 5.0, 0.003, "compound_bond_scaler"),
    (0.55, 0.95, -0.08, "compound_scaler_roughness"),
]


def build_cells():
    """Return the 36 (bond, scaler, roughness, in_domain, ood_axis) cells."""
    cells = []
    for bo in INTERP_BO:
        for sc in INTERP_SCALER:
            for r in INTERP_ROUGHNESS:
                cells.append(
                    {"bond": bo, "scaler": sc, "roughness": r, "in_domain": True, "ood_axis": ""}
                )
    for bo, sc, r, axis in EXTRAP_CELLS:
        cells.append(
            {"bond": bo, "scaler": sc, "roughness": r, "in_domain": False, "ood_axis": axis}
        )
    assert len(cells) == 36, f"expected 36 cells, got {len(cells)}"
    return cells


def build_matrix():
    """Cross the 36 cells with the 9 fixed angles -> up to 324 rows.

    Returns a list of dicts with the columns specified in the plan
    (Appendix A.1): cell_id, angle, bond, scaler, roughness,
    roughness_regime, in_domain, ood_axis, grid_distance, seed, filename,
    status, drop_reason. `filename`/`status`/`drop_reason` are filled in by
    generate_offgrid_synthetic.py once images are actually rendered — here
    they start as "" / "pending" / "".
    """
    cells = build_cells()
    rows = []
    for cell_id, cell in enumerate(cells):
        bond, scaler, roughness = cell["bond"], cell["scaler"], cell["roughness"]
        for angle in ANGLES:
            rows.append(
                {
                    "cell_id": cell_id,
                    "angle": angle,
                    "bond": bond,
                    "scaler": scaler,
                    "roughness": roughness,
                    "roughness_regime": roughness_regime(roughness),
                    "in_domain": cell["in_domain"],
                    "ood_axis": cell["ood_axis"],
                    "grid_distance": grid_distance(bond, scaler, roughness),
                    "seed": cell_seed(angle, bond, scaler, roughness),
                    "filename": "",
                    "status": "pending",
                    "drop_reason": "",
                    "drop_height_px": "",
                }
            )
    return rows


MATRIX_COLUMNS = [
    "cell_id",
    "angle",
    "bond",
    "scaler",
    "roughness",
    "roughness_regime",
    "in_domain",
    "ood_axis",
    "grid_distance",
    "seed",
    "filename",
    "status",
    "drop_reason",
    "drop_height_px",
]


def write_matrix_csv(rows, path=MATRIX_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATRIX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    rows = build_matrix()
    n_in_domain = sum(1 for r in rows if r["in_domain"])
    n_out_domain = len(rows) - n_in_domain
    print(f"Built {len(rows)} matrix rows ({n_in_domain} in-domain, {n_out_domain} out-of-domain)")
    path = write_matrix_csv(rows)
    print(f"Written to: {path}")


if __name__ == "__main__":
    main()
