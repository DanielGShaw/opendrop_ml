"""Configuration for the synthetic-reanalysis pipeline (SYNTHETIC_DATA_REANALYSIS.md).

Isolated staging area — see the plan doc's "Where this work lives" section.
All paths are computed relative to this file's location so the pipeline
works regardless of working directory.
"""

import os

# --- Directory layout ---
SYNTHETIC_REANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
OPENDROP_DIR = os.path.dirname(SYNTHETIC_REANALYSIS_DIR)  # OpenDrop_OP/
PAPER_RESULTS_DIR = os.path.join(OPENDROP_DIR, "paper_results")
TRAINING_FILES_DIR = os.path.join(OPENDROP_DIR, "training_files")

# --- Existing (on-grid) synthetic set — read-only reference ---
SYNTHETIC_ONGRID_DATA_DIR = os.path.join(PAPER_RESULTS_DIR, "data", "synthetic")
SYNTHETIC_ONGRID_RESULTS_CSV = os.path.join(
    PAPER_RESULTS_DIR, "results", "synthetic_results.csv"
)

# --- New datasets produced by this pipeline ---
DATA_DIR = os.path.join(SYNTHETIC_REANALYSIS_DIR, "data")
SYNTHETIC_OFFGRID_DATA_DIR = os.path.join(DATA_DIR, "synthetic_offgrid")
SYNTHETIC_ONGRID_CONTROL_DIR = os.path.join(DATA_DIR, "synthetic_ongrid_control")

# --- Output directories ---
RESULTS_DIR = os.path.join(SYNTHETIC_REANALYSIS_DIR, "results")
FIGURES_DIR = os.path.join(SYNTHETIC_REANALYSIS_DIR, "figures")

MATRIX_CSV = os.path.join(RESULTS_DIR, "offgrid_matrix.csv")

# --- ML model ---
ML_MODEL_DIR = os.path.join(OPENDROP_DIR, "opendrop_ml", "modules", "ML_model")

# --- Image file extensions (matches paper_results/config.py) ---
IMAGE_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif")

# --- Rendering ---
DPI = 64  # matches the existing synthetic set / create_contours_args6.py's create_contour()
N_PTS_DROP = 5000  # matches create_contour()'s call to draw_drop()

# --- Reproducibility ---
BOOTSTRAP_SEED = 20260813
LHS_SEED = 20260813
