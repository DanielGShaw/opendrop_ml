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
- **Synthetic images**: `paper_results/data/synthetic/` (324 images) — 36 (Bond, scaler, roughness) cells x 9 angles, placed off the training grid (24 cells between training-grid points, 12 beyond it) so accuracy is measured between and beyond the values the model was trained on, not at them. Built by `paper_results/synthetic_generation/` (see its README); `results/synthetic_matrix.csv` records each cell's design and its `in_domain`/`ood_axis` label.
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
```

The regression harness that gates any `preprocessing.py` change lives in
`paper_results/regression/` (see its own docstring) — not part of the
figure-regeneration pipeline above.

## Output

### Results (`results/`)
- `experimental_results.csv` — 169 rows, all fitting method predictions + timings
- `synthetic_results.csv` — 324 rows, all fitting method predictions + ground truth
- `sensitivity_results.csv` — 120 rows, v03 and v05.05 predictions on sensitivity dataset (GT 110-180°)
- `synthetic_matrix.csv` — the synthetic set's design: per-cell (Bond, scaler, roughness), `in_domain`/`ood_axis` labels, and the seed/filename each cell rendered to
- `synthetic_domain_split_mae.csv` — per-method MAE (all-angle and excluding 180°) computed separately for the in-domain and out-of-domain subsets (see figures below)
- `training_domain_bounds.csv` — the training grid's own scaler/Bond envelope (36 rendered cells), used by the training-domain-coverage figure

### Figures (`figures/`)

**Experimental** (`figures/experimental/`):
- `experimental_agreement_6panel.png` — 3x2 grid: ML vs YL agreement split by YL RMSE quality
- `experimental_yl_characterization.png` — YL fit RMSE vs angle, Bond number distribution
- `experimental_speed.png` — 2x2 timing breakdown (preprocessing, total, fit time, fit vs contour length)

**Synthetic** (`figures/synthetic/`):
- `synthetic_agreement_5panel.png`, `synthetic_bias_and_disagreement.png` — main-text figures, in-domain cells only (n=216): 3x2 per-method agreement scatter, and bias vs true angle / ML-YL disagreement
- `synthetic_agreement_5panel_si_outdomain.png`, `synthetic_bias_and_disagreement_si_outdomain.png` — supplementary figures, out-of-domain cells only (n=108, beyond the training grid) — same layout, reported separately rather than marked on the main-text panel (PART2 §7.3.3)

**Comparison** (`figures/comparison/`):
- `sensitivity_agreement.png` — 3-panel: v03 and v05.05 agreement + delta on sensitivity dataset

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
- **Total**: ~25 minutes for the experimental + synthetic + sensitivity core
