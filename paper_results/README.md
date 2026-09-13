# Paper Results Regeneration

Regenerates all paper results and figures using the integrated `opendrop_ml` pipeline.

## Prerequisites

```bash
conda activate opendrop_op   # Python 3.8.10

# Build Cython extensions (required for YL fit)
cd OpenDrop_OP
python setup.py build_ext --inplace
```

### v03 baseline repo (required for sensitivity + inverse-phase evaluation)

The v03 Conan-ML baseline isn't vendored here — clone it alongside this
package. `evaluate_sensitivity.py` and `evaluate_inverse_phase.py` import
from it directly, and the sensitivity dataset lives inside the clone.

```bash
# From OpenDrop_OP/paper_results/
git clone https://github.com/jdber1/conan-ML.git conan_model_repo
```

The expected path is `paper_results/conan_model_repo/` (matches
`CONAN_MODEL_REPO_DIR` in `paper_results/config.py`). The directory is
gitignored — each reproducer clones it locally.

## Dataset Paths

Paths are configured in `config.py`. All datasets live in-tree under `paper_results/`:

- **Experimental images**: `paper_results/data/experimental/` (171 images)
- **Synthetic images**: `paper_results/data/synthetic/` (324 images)
- **Off-grid synthetic images**: `paper_results/data/synthetic_offgrid/` (324) and `paper_results/data/synthetic_ongrid_control/` (324 + 5 × 72 replicates) — the generalisation study in `paper_results/offgrid/`, see its README
- **Sensitivity images**: `paper_results/conan_model_repo/sensitivity data set/` (120 images — in the v03 clone, see Prerequisites)
- **Video analysis datasets** (Isaac, Kristina, inverse-phase): `paper_results/data/video_analysis/` (gitignored, see deferred data-hosting item in `ML_MODEL_INTEGRATION_PLAN.md` Phase 7.4)

## Usage

### Run everything
```bash
cd OpenDrop_OP
python paper_results/run_all.py
```

### Run individual steps
```bash
# Generate results
python paper_results/evaluate/evaluate_experimental.py   # → results/experimental_results.csv
python paper_results/evaluate/evaluate_synthetic.py      # → results/synthetic_results.csv

# Generate figures (requires evaluation output)
python paper_results/plot/plot_experimental.py            # → figures/experimental/*.png
python paper_results/plot/plot_synthetic.py               # → figures/synthetic/*.png
python paper_results/plot/plot_speed.py                   # → figures/experimental/experimental_speed.png

# Sensitivity dataset — v03 vs v05.05 (requires evaluation output)
python paper_results/evaluate/evaluate_sensitivity.py     # → results/sensitivity_results.csv
python paper_results/plot/plot_sensitivity.py              # → figures/comparison/sensitivity_agreement.png

# Off-grid synthetic study (see paper_results/offgrid/README.md)
python paper_results/offgrid/evaluate_offgrid.py          # → results/offgrid/synthetic_offgrid_results.csv, ...
python paper_results/offgrid/compare_grid_vs_offgrid.py   # → results/offgrid/phase4_summary.md
```

## Output

### Results (`results/`)
- `experimental_results.csv` — 169 rows, all fitting method predictions + timings
- `synthetic_results.csv` — 324 rows, all fitting method predictions + ground truth
- `sensitivity_results.csv` — 120 rows, v03 and v05.05 predictions on sensitivity dataset (GT 110-180°)
- `offgrid/` — the off-grid synthetic study: per-image results for the off-grid set, the on-grid control and its replicates, the Phase 4 comparison (`phase4_summary.md`), the CPID regression-harness baselines and the Phase 6 sweep outputs. `offgrid/pre_6.1/` is the pre-fix snapshot.

### Figures (`figures/`)

**Experimental** (`figures/experimental/`):
- `experimental_agreement_6panel.png` — 3x2 grid: ML vs YL agreement split by YL RMSE quality
- `experimental_yl_characterization.png` — YL fit RMSE vs angle, Bond number distribution
- `experimental_speed.png` — 2x2 timing breakdown (preprocessing, total, fit time, fit vs contour length)

**Synthetic** (`figures/synthetic/`):
- `synthetic_agreement_5panel.png` — 3x2 grid: per-method agreement scatter (tangent, circle, ellipse, YL, ML)
- `synthetic_bias_and_disagreement.png` — Bias vs true angle and ML-YL disagreement

**Comparison** (`figures/comparison/`):
- `sensitivity_agreement.png` — 3-panel: v03 and v05.05 agreement + delta on sensitivity dataset

**Off-grid study** (`figures/offgrid/`):
- `phase6_fig1_cp_error_heatmaps.png`, `phase6_fig2_decision_space.png`, `phase6_ex_*.png` — the reflective-substrate CPID fix: before/after contact-point error, the decision statistic, worked examples
- `phase6_separability_*.png` — per-feature class histograms from the separability analysis

## Methods Evaluated

| Method | Function | Parameters |
|--------|----------|------------|
| Tangent | `polynomial_fit` | degree=1, num_points=10 |
| Polynomial | `polynomial_fit` | degree=2, num_points=10 |
| Circle | `circular_fit` | — |
| Ellipse | `ellipse_fit` | — |
| Young-Laplace | `yl_fit` | Bashforth-Adams |
| ML (Conan v05.05) | `prepare_for_model` + `experimental_pred` | input_len=50 |

## Runtime Estimates

- Experimental evaluation: ~8 minutes (169 images)
- Synthetic evaluation: ~11 minutes (324 images)
- Sensitivity evaluation: ~3 minutes (120 images, both models)
- Figures: ~1 minute
- Video-analysis evaluations (inverse phase, Isaac, Kristina): see each script
- Off-grid synthetic evaluation: ~55 minutes (1008 images); comparison ~5 minutes
- **Total**: ~25 minutes for the experimental + synthetic + sensitivity core; the off-grid step roughly doubles it
