"""Configuration for paper_results pipeline.

All paths are computed relative to this file's location so the pipeline
works regardless of working directory.
"""

import os

# --- Directory layout ---
PAPER_RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
OPENDROP_DIR = os.path.dirname(PAPER_RESULTS_DIR)  # OpenDrop_OP/

# --- Dataset paths (all in-tree under paper_results/data/) ---
EXPERIMENTAL_DATA_DIR = os.path.join(PAPER_RESULTS_DIR, "data", "experimental")
SYNTHETIC_DATA_DIR = os.path.join(PAPER_RESULTS_DIR, "data", "synthetic")

# --- Output directories ---
RESULTS_DIR = os.path.join(PAPER_RESULTS_DIR, "results")
FIGURES_DIR = os.path.join(PAPER_RESULTS_DIR, "figures")

# --- ML model ---
ML_MODEL_DIR = os.path.join(OPENDROP_DIR, "opendrop_ml", "modules", "ML_model")

# --- Fitting constants ---
N_TANGENT_POINTS = 10  # num_points for tangent / polynomial fit
POLYNOMIAL_DEGREE_TANGENT = 1  # degree=1 for tangent fit
POLYNOMIAL_DEGREE_POLY = 2  # degree=2 for polynomial fit

# --- Angle buckets for per-bucket analysis ---
ANGLE_BUCKETS = [(0, 90), (90, 110), (110, 180)]
ANGLE_BUCKET_LABELS = ["<90", "90-110", ">110"]

# --- Old model (v03) — cloned separately, see paper_results/README.md ---
CONAN_MODEL_REPO_DIR = os.path.join(PAPER_RESULTS_DIR, "conan_model_repo")
CONAN_MODEL_DIR = os.path.join(PAPER_RESULTS_DIR, "conan_model")

# --- Video analysis datasets ---
VIDEO_ANALYSIS_DIR = os.path.join(PAPER_RESULTS_DIR, "data", "video_analysis")
INVERSE_PHASE_DIR = os.path.join(VIDEO_ANALYSIS_DIR, "inverse_phase")
ISAAC_DIR = os.path.join(VIDEO_ANALYSIS_DIR, "isaac")
KRISTINA_DIR = os.path.join(VIDEO_ANALYSIS_DIR, "kristina")
ISAAC_REF_DIR = ISAAC_DIR
KRISTINA_REF_DIR = KRISTINA_DIR

# --- Image file extensions ---
IMAGE_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif")

# --- Off-grid synthetic study (paper_results/offgrid/, see its README) ---
TRAINING_FILES_DIR = os.path.join(OPENDROP_DIR, "training_files")
SYNTHETIC_OFFGRID_DATA_DIR = os.path.join(PAPER_RESULTS_DIR, "data", "synthetic_offgrid")
SYNTHETIC_ONGRID_CONTROL_DIR = os.path.join(PAPER_RESULTS_DIR, "data", "synthetic_ongrid_control")
OFFGRID_RESULTS_DIR = os.path.join(RESULTS_DIR, "offgrid")
OFFGRID_FIGURES_DIR = os.path.join(FIGURES_DIR, "offgrid")
OFFGRID_MATRIX_CSV = os.path.join(OFFGRID_RESULTS_DIR, "offgrid_matrix.csv")
OFFGRID_DPI = 64  # matches the on-grid set / create_contours_args6.py's create_contour()
OFFGRID_N_PTS_DROP = 5000  # matches create_contour()'s call to draw_drop()
OFFGRID_BOOTSTRAP_SEED = 20260813
OFFGRID_LHS_SEED = 20260813

# --- Video file extensions ---
VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov")
