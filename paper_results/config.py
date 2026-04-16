"""Configuration for paper_results pipeline.

All paths are computed relative to this file's location so the pipeline
works regardless of working directory.
"""

import os

# --- Directory layout ---
PAPER_RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
OPENDROP_DIR = os.path.dirname(PAPER_RESULTS_DIR)        # OpenDrop_OP/
REPO_DIR = os.path.dirname(OPENDROP_DIR)                 # conan_v2/

# --- Dataset paths ---
EXPERIMENTAL_IMAGES_DIR = os.path.join(
    REPO_DIR, "files_from_old_computer", "experimental_dataset",
    "experimental_dataset"
)
EXPERIMENTAL_REF_CSV = os.path.join(
    REPO_DIR, "files_from_old_computer", "experimental_dataset", "list_full.csv"
)
SYNTHETIC_IMAGES_DIR = os.path.join(
    REPO_DIR, "files_from_old_computer", "synthetic_dataset",
    "synthetic_evaluation_data_set_dpi64_images"
)
SYNTHETIC_REF_CSV = os.path.join(
    REPO_DIR, "files_from_old_computer", "synthetic_dataset",
    "list_synthetic_dpi64.csv"
)

# --- Output directories ---
RESULTS_DIR = os.path.join(PAPER_RESULTS_DIR, "results")
FIGURES_DIR = os.path.join(PAPER_RESULTS_DIR, "figures")
COMPARISON_DIR = os.path.join(PAPER_RESULTS_DIR, "comparison")

# --- ML model ---
ML_MODEL_DIR = os.path.join(OPENDROP_DIR, "opendrop_ml", "modules", "ML_model")

# --- Fitting constants ---
N_TANGENT_POINTS = 10          # num_points for tangent / polynomial fit
POLYNOMIAL_DEGREE_TANGENT = 1  # degree=1 for tangent fit
POLYNOMIAL_DEGREE_POLY = 2    # degree=2 for polynomial fit

# --- Angle buckets for per-bucket analysis ---
ANGLE_BUCKETS = [(0, 90), (90, 110), (110, 180)]
ANGLE_BUCKET_LABELS = ["<90", "90-110", ">110"]

# --- Data directories (images copied to paper_results/data/ for reproducibility) ---
EXPERIMENTAL_DATA_DIR = os.path.join(PAPER_RESULTS_DIR, "data", "experimental")
SYNTHETIC_DATA_DIR = os.path.join(PAPER_RESULTS_DIR, "data", "synthetic")

# --- Comparison reference CSVs (old model v03 predictions) ---
COMPARISON_NOTEBOOKS_DIR = os.path.join(
    REPO_DIR, "files_from_old_computer", "comparison_notebooks"
)
EXPERIMENTAL_COMPARE_REF_CSV = os.path.join(
    COMPARISON_NOTEBOOKS_DIR, "list_compare.csv"
)
SYNTHETIC_COMPARE_REF_CSV = os.path.join(
    COMPARISON_NOTEBOOKS_DIR, "list_synthetic_dpi64_compare.csv"
)

# --- Old model (v03) ---
CONAN_MODEL_REPO_DIR = os.path.join(PAPER_RESULTS_DIR, "conan_model_repo")
CONAN_MODEL_DIR = os.path.join(PAPER_RESULTS_DIR, "conan_model")

# --- Video analysis datasets ---
VIDEO_ANALYSIS_DIR = os.path.join(PAPER_RESULTS_DIR, "data", "video_analysis")
INVERSE_PHASE_DIR = os.path.join(VIDEO_ANALYSIS_DIR, "inverse_phase")
ISAAC_DIR = os.path.join(VIDEO_ANALYSIS_DIR, "isaac")
KRISTINA_DIR = os.path.join(VIDEO_ANALYSIS_DIR, "kristina")

# References for comparison (inverse_phase still in files_from_old_computer;
# Isaac and Kristina were moved entirely into paper_results/data/)
INVERSE_PHASE_REF_DIR = os.path.join(
    REPO_DIR, "files_from_old_computer", "video_analysis", "inverse_phase"
)
ISAAC_REF_DIR = ISAAC_DIR
KRISTINA_REF_DIR = KRISTINA_DIR

# --- Image file extensions ---
IMAGE_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tiff", ".tif")

# --- Video file extensions ---
VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov")
