# Paper Results Regeneration

Regenerates all paper results and figures using the integrated `opendrop_ml` pipeline.

## Prerequisites

```bash
conda activate opendrop_op   # Python 3.8.10

# Build Cython extensions (required for YL fit)
cd OpenDrop_OP
python setup.py build_ext --inplace
```

## Dataset Paths

Paths are configured in `config.py`. Default locations:

- **Experimental images**: `paper_results/data/experimental/` (171 images)
- **Synthetic images**: `files_from_old_computer/synthetic_dataset/synthetic_evaluation_data_set_dpi64_images/` (324 images)
- **Sensitivity images**: `paper_results/conan_model_repo/sensitivity data set/` (120 images)
- **Reference CSVs**: `files_from_old_computer/experimental_dataset/list_full.csv`, `files_from_old_computer/synthetic_dataset/list_synthetic_dpi64.csv`
- **Model comparison CSVs**: `files_from_old_computer/comparison_notebooks/list_compare.csv`, `list_synthetic_dpi64_compare.csv`

## Usage

### Run everything
```bash
cd OpenDrop_OP
python paper_results/run_all.py
```

### Run individual phases
```bash
# Phase 1: Generate results
python paper_results/evaluate/evaluate_experimental.py   # → results/experimental_results.csv
python paper_results/evaluate/evaluate_synthetic.py      # → results/synthetic_results.csv

# Phase 2: Compare against prior results (requires Phase 1 output)
python paper_results/compare/compare_experimental.py     # → comparison/experimental_comparison.csv
python paper_results/compare/compare_synthetic.py        # → comparison/synthetic_comparison.csv

# Phase 3: Generate figures (requires Phase 1 output)
python paper_results/plot/plot_experimental.py            # → figures/experimental/*.png
python paper_results/plot/plot_synthetic.py               # → figures/synthetic/*.png
python paper_results/plot/plot_speed.py                   # → figures/experimental/experimental_speed.png

# Phase 3.5: Sensitivity dataset — v03 vs v05.05 (requires Phase 1 output)
python paper_results/evaluate/evaluate_sensitivity.py     # → results/sensitivity_results.csv
python paper_results/plot/plot_sensitivity.py              # → figures/comparison/sensitivity_agreement.png
```

## Output

### Results (`results/`)
- `experimental_results.csv` — 169 rows, all fitting method predictions + timings
- `synthetic_results.csv` — 324 rows, all fitting method predictions + ground truth
- `sensitivity_results.csv` — 120 rows, v03 and v05.05 predictions on sensitivity dataset (GT 110-180°)

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

### Comparison (`comparison/`)
- `experimental_comparison.csv` — Per-image deltas: new vs prior pipeline
- `synthetic_comparison.csv` — Per-image deltas: new vs prior pipeline

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
- Comparison + figures: ~1 minute
- **Total**: ~25 minutes
