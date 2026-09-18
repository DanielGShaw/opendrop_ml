# Synthetic set generator

Generates the paper's synthetic evaluation set: `paper_results/data/synthetic/`
(324 images), the set now used directly by `evaluate/evaluate_synthetic.py` and
`plot/plot_synthetic.py` via `config.SYNTHETIC_DATA_DIR`.

**Why this set is placed off the training grid.** The model was trained on a
regular (Bond, scaler, roughness) grid. A synthetic evaluation set drawn from
the same grid points can't tell you whether accuracy degrades between or
beyond those values — only at them. This set instead samples 36 (Bond,
scaler, roughness) cells × 9 angles (20–180°, step 20°): 24 cells interpolate
between training-grid points, 12 extrapolate past it (bond 1.2/1.5, scaler
0.95/5.0, roughness +0.02/−0.08, and two compound corners). Design and each
cell's `in_domain`/`ood_axis` label are in `results/synthetic_matrix.csv`.

This originated as a standalone "off-grid vs on-grid" generalisation study
(PART2_REMAINING_WORK.md, outside this repo). The study's finding — a bounded
null, no graded penalty for being off-grid (image-level ML MAE 2.02° off-grid
vs 2.37° on-grid control, Theil–Sen slope −0.004°/unit CI [−0.071, 0.066]) —
led to the off-grid set replacing the on-grid set as the paper's synthetic
evaluation entirely (PART2 §7.3.2), so the on-grid set, its control replicates
and the comparison/sweep machinery were pruned; only the generator and the
regression harness (now `paper_results/regression/`) survive in-tree. The
study also found and fixed a real `CPID` defect along the way — see
`0c1adcd8` / `fa10d758` — unrelated to the off-grid/on-grid question itself.

## Datasets

Rendered with `training_files/create_contours_args6.py`'s `draw_drop()` at
`dpi=64` (744 × 492 px). Every image is seeded deterministically from its
parameters (`matrix_utils.cell_seed`): a re-render is bit-identical for
non-positive roughness and statistically identical for positive roughness
(random surface jitter). Filenames encode the ground truth:
`{angle}_{bond}_{scaler}_{roughness}_CPL_{x}_{y}_CPR_{x}_{y}_.png`.

Grid distance (used for the `in_domain`/`ood_axis` labels) is Euclidean over
per-axis grid-step units (bond step 0.10, scaler step ln 4 / 6 in log space,
roughness step 0.002 positive / 0.01 negative); angle is excluded because the
training grid steps by 0.05°.

`config.SYNTHETIC_ONGRID_CONTROL_DIR` and the on-grid-control code path in
`generate_offgrid_synthetic.py` are kept only as a record of how the (now
pruned) control set and its 5 render-noise replicates were produced; they are
not runnable as-is since they parse parameters out of the pruned on-grid
images, which are only recoverable from git history before the PART2 §7.3.2
commit.

## Scripts

Run from `OpenDrop_OP/` with the `opendrop_op` environment; paths resolve
from each script's own location.

| script | writes |
|---|---|
| `offgrid_matrix.py` | `results/synthetic_matrix.csv` (design + labels; status `pending` until rendered) |
| `generate_offgrid_synthetic.py` | `paper_results/data/synthetic/`; fills in the matrix's `filename`/`status` columns |

The set is already rendered and committed — these scripts are for
reproducing the generation from scratch, not part of `run_all.py`.
