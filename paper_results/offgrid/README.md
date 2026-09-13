# Off-grid synthetic study

Does the Conan-ML contact-angle model get worse as a drop's rendering parameters
move away from the training grid? The paper's synthetic evaluation set
(`paper_results/data/synthetic/`, 324 images) samples parameter values that sit
*on* the grid the model was trained on, so it can't answer that. This study
renders a second 324-image set at *off-grid* values, evaluates every method on
it, and regresses error against distance from the nearest training-grid point.

**Answer: a bounded null.** The image-level ML error slope against grid distance
is −0.004°/unit, 95% CI [−0.071, 0.066] (Theil–Sen, image-level bootstrap over
the pooled off-grid and on-grid-control sets, 1296 measurements). The off-grid ML
MAE is 2.02°, below the on-grid control's 2.37° and its render-noise band
[2.17°, 2.28°]. There is no graded penalty for being off-grid.

The study also found, and fixed, a real defect: `CPID`'s reflective-substrate
branch placed the contact point ~20 px inside the reflection for 75–95° drops at
roughness values the on-grid set never samples. The fix (an arc-normalised
tangent-step test at the widest point) changes nothing on the paper's 324 + 171
images and removes the failure everywhere except |θ − 90°| ≲ 5°, which is a
physical limit of single-frame geometry at deployed resolution. See
`PART2_REMAINING_WORK.md` §6.1 for the full trail.

The two long documents are the study's record:

- `PART1_ANALYSIS_AND_FINDINGS.md` — the working log: design, every phase's
  measurements, the post-hoc review and revised verdict. Read its "How to read
  this document" preamble first; the rest is in chronological order and earlier
  sections are superseded by later ones.
- `PART2_REMAINING_WORK.md` — the prescriptive record of the fixes (Phase 6) and
  the fold into `paper_results/` (Phase 7), with a changelog and a list of traps.

## Datasets

Both sets are rendered with `training_files/create_contours_args6.py`'s
`draw_drop()` at `dpi=64` (744 × 492 px), the same renderer and settings that
produced the on-grid set. Every image is seeded deterministically from its
parameters (`matrix_utils.cell_seed`), so a re-render is bit-identical for
non-positive roughness and statistically identical for positive roughness (which
draws random surface jitter).

Filenames encode the ground truth:
`{angle}_{bond}_{scaler}_{roughness}_CPL_{x}_{y}_CPR_{x}_{y}_.png`.

| directory | images | what it is |
|---|---|---|
| `paper_results/data/synthetic_offgrid/` | 324 | 36 (Bo, scaler, roughness) cells × 9 angles (20–180° step 20). 24 cells interpolate between grid points; 12 extrapolate past the grid: bond 1.2/1.5, scaler 0.95/5.0, roughness +0.02/−0.08, and two compound corners (bond × scaler, scaler × roughness). Design and per-cell grid distance in `results/offgrid/offgrid_matrix.csv`. |
| `paper_results/data/synthetic_ongrid_control/` | 324 | A fresh render of the on-grid set's parameter combinations. Used for paired matching against the off-grid cells and as a same-code control. Manifest: `results/offgrid/ongrid_control_manifest.csv`. |
| `paper_results/data/synthetic_ongrid_control/replicate_{0..4}/` | 5 × 72 | Five re-renders of an 8-cell × 9-angle subset with different seeds. Their MAE spread is the render-noise band that any claimed effect must exceed. Manifests: `results/offgrid/replicate_{i}_manifest.csv`. |

Grid distance is Euclidean over per-axis grid-step units (bond step 0.10, scaler
step ln 4 / 6 in log space, roughness step 0.002 positive / 0.01 negative);
angle is excluded because the training grid steps by 0.05°.

## Scripts

Run everything from `OpenDrop_OP/` with the `opendrop_op` environment. Paths are
resolved from each script's own location, so the working directory doesn't
matter. Constants live in `paper_results/config.py` under "Off-grid synthetic
study".

**Pipeline, in order**

| step | script | writes | runtime |
|---|---|---|---|
| 1. matrix | `offgrid_matrix.py` | `results/offgrid/offgrid_matrix.csv` (status `pending`) | seconds |
| 1. gate | `acceptance_gate.py` | — (asserts the renderer reproduces a known on-grid image) | seconds |
| 1. render | `generate_offgrid_synthetic.py` | the three data directories above; fills in the matrix and manifests | minutes |
| 2A. contour-level sweep | `contour_level_sweep.py`, `analyze_grid_distance_slope.py` | `results/offgrid/contour_sweep_*.csv`, `grid_distance_slope_summary.csv` | long (5400 contour samples through the model) |
| 2B. smoke test | `spotcheck_pipeline.py` | `results/offgrid/phase2b_smoke_test.md` | minutes |
| 3. image-level evaluation | `evaluate_offgrid.py` | `results/offgrid/synthetic_offgrid_results.csv`, `synthetic_ongrid_control*_results.csv`, one `*_provenance.json` per CSV | ~55 min (1008 images) |
| 4. comparison | `compare_grid_vs_offgrid.py` | `results/offgrid/phase4_summary.md`, `phase4_measurements.csv`, `phase4_paired_matches.csv` | ~5 min |

Steps 3 and 4 are registered in `paper_results/run_all.py`. Steps 1 and 2 are
not: the rendered data is committed, and 2A is a contour-level check that is
faithful to *training* preprocessing rather than the deployed pipeline, so it
doesn't move when `preprocessing.py` changes.

**Resume behaviour.** `evaluate_offgrid.py` and `phase6_regression.py` checkpoint
after every image and skip images already present in their output CSV. After any
change to `preprocessing.py`, move the old CSVs aside first (for the harness:
`results/offgrid/phase6_regression_*_run.csv`) or the run evaluates nothing and
reports the old numbers.

**Regression harness and the CPID fix (Phase 6)**

| script | purpose |
|---|---|
| `phase6_regression.py` | Re-evaluates the paper's 324 synthetic + 171 experimental images and diffs them against frozen baselines (`results/offgrid/phase6_baseline_*.csv`). Contact points must match exactly; angles within 0.01°. **Run this before and after any change to `preprocessing.py`.** ~20 min. |
| `phase6_reflective_sweep.py` | 2178-cell map (angle 20–180 × roughness × scaler) of which `CPID` branch fires and the resulting contact-point error; records the branch's discriminant values. `--smoke` runs four cells. ~30 min. |
| `phase6_separability.py` | Offline: can any deployable scalar separate reflective sub-90° drops from genuine high-angle drops? (No, on the original statistic.) |
| `phase6_window_sweep.py`, `phase6_kink_refine.py`, `phase6_window_vs_noise.py` | The levers that were measured and declined: longer or width-scaled fit windows, curvature-removed kink, mirror symmetry. Each writes a `results/offgrid/phase6_*.csv`. |
| `phase6_figures.py` | The explanatory figures in `figures/offgrid/phase6_*.png`: before/after CP-error heatmaps, the decision space, and worked examples. |
| `phase7_offgrid_diff.py` | Diffs the current step-3 CSVs against the pre-fix snapshots in `results/offgrid/pre_6.1/`. |
| `phase7_isaac_pass1_check.py` | Video-side gate the harness lacks: re-runs one Isaac dataset's Pass-1 CPID sample with the kink branch on and off and reports per-frame CP changes and the resulting medians (~2 min). |

## Results

`results/offgrid/phase4_summary.md` is the headline output. Post-fix numbers
(13 September 2026):

| quantity | value |
|---|---|
| ML Theil–Sen slope vs grid distance | −0.004°/unit, 95% CI [−0.071, 0.066] |
| off-grid ML MAE / median / max | 2.02° / 0.84° / 21.3° (n = 648) |
| on-grid control ML MAE | 2.37° |
| render-noise band (5 replicates) | [2.17°, 2.28°] |
| in-domain / out-of-domain ML MAE | 1.71° / 2.65° |
| hardest out-of-domain axes | scaler (0.95/5.0 pooled, 3.3°), bond × scaler corner (6.4°) |

`results/offgrid/pre_6.1/` holds the same outputs from before the CPID fix, for
the before/after comparison in PART2 §7.1. The only images that changed are 11
at 80° on reflective roughness (all improved) and one recovered crash.
