#!/usr/bin/env python
"""Render the training envelope for the coverage figure (PART2 §7.3.2a).

The synthetic evaluation set's scalers (0.95, 1.41, 2.83, 5.0) do not match
the training grid's scaler range (geomspace(1, 4, 7), i.e. [1, 4] -- see
training_files/create_contours_args6.py:750). Deriving the coverage figure's
"training bounds" band from the evaluation CSV therefore mixes in cells
outside the training domain. This script instead renders the training
envelope directly: scaler in {1, 4} x Bond in {0, 1} x roughness 0, at the
nine display angles (36 renders), through the same isolate_contour + CPID
preprocessing evaluate_synthetic.py uses -- no TensorFlow, since only
profile_length is needed.

Writes results/training_domain_bounds.csv (angle, scaler, bond, profile_length).
plot_training_coverage.py takes its per-angle min/max from this file, not
from the evaluation set, so the figure no longer depends on scalers the
model was never trained on.

Run from OpenDrop_OP/:
    python paper_results/synthetic_generation/render_training_bounds.py
"""

import os
import sys

import numpy as np
import pandas as pd

_PAPER_RESULTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OPENDROP_DIR = os.path.dirname(_PAPER_RESULTS_DIR)
sys.path.insert(0, _PAPER_RESULTS_DIR)  # paper_results/, for config
sys.path.insert(0, _OPENDROP_DIR)

from config import OFFGRID_DPI as DPI, OFFGRID_N_PTS_DROP as N_PTS_DROP, RESULTS_DIR, TRAINING_FILES_DIR  # noqa: E402

sys.path.insert(0, TRAINING_FILES_DIR)
from create_contours_args6 import draw_drop  # noqa: E402
from matrix_utils import cell_seed  # noqa: E402

from opendrop_ml.modules.preprocessing.preprocessing import isolate_contour, CPID  # noqa: E402

ANGLES = [20, 40, 60, 80, 100, 120, 140, 160, 180]
TRAIN_SCALERS = [1.0, 4.0]  # geomspace(1, 4, 7) endpoints
TRAIN_BONDS = [0.0, 1.0]  # linspace(0, 1, 11) endpoints
ROUGHNESS = 0.0

OUT_CSV = os.path.join(RESULTS_DIR, "training_domain_bounds.csv")


def extract_profile_length(contour, CPs):
    idx_left = np.where((contour == CPs[0]).all(axis=1))[0][0]
    idx_right = np.where((contour == CPs[1]).all(axis=1))[0][0]
    if idx_left > idx_right:
        idx_left, idx_right = idx_right, idx_left
    return idx_right - idx_left + 1


def main():
    rows = []
    for scaler in TRAIN_SCALERS:
        for bond in TRAIN_BONDS:
            for angle in ANGLES:
                seed = cell_seed(angle, bond, scaler, ROUGHNESS)
                np.random.seed(seed)
                img, _ = draw_drop(angle, bond, scaler, ROUGHNESS, N_PTS_DROP, dpi=DPI)
                img_processed, contour, drop_guess, info = isolate_contour(
                    img, DISPLAY=False, blur=True
                )
                CPs = CPID(contour, display=False)
                profile_length = extract_profile_length(contour, CPs)
                rows.append(
                    {
                        "angle": angle,
                        "scaler": scaler,
                        "bond": bond,
                        "roughness": ROUGHNESS,
                        "profile_length": profile_length,
                    }
                )
                print(f"  angle={angle} scaler={scaler} bond={bond}: profile_length={profile_length}")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
