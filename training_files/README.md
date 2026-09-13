# Training pipeline for the Conan-ML all-angle model

This directory holds the actual data-generation and training scripts that
produced the TensorFlow SavedModel shipped at
`opendrop_ml/modules/ML_model/` (verified by MD5 match against
`saved_model.pb` in the original training run's output directory).

**Provenance note (2026-08):** an earlier version of this directory
contained a *different, superseded* pipeline (`create_contours_args4.py`,
`combine.py`, `zeropad.py`, `train_model.py`) that had been copied over
from `paper_results/conan_model_repo/training files/` — the repo for an
earlier iteration of the model. That pipeline generated angle 110°–180°,
Bond number 0–2, and zero-padded full contours to a global max length. It
does **not** match the input format or training domain of the model
actually shipped in this repo, and has been removed. The files below were
recovered from `files_from_old_computer/_mse_0.4218772351741791/` (a
directory named for the shipped model's validation MSE) and confirmed via
checksum to be the real pipeline.

This directory is a **curated subset** of that archive, not a full
mirror. Four one-off development scripts (`check_imports.py`,
`clean_output_file.py`, `data_check.py`, `shuffle_test.py` — an import
smoke-test, a log reformatter, and two ad-hoc sanity checks) were left
out because they played no part in generating the data or training the
model. The archive remains the untouched reference copy.

## Training domain

Unlike the superseded pipeline, this one covers the **full angle range**:

- **Contact angle:** 20°–180°, swept in 0.05° steps (`seq 20 0.05 180`)
- **Bond number:** 0–1, 11 values (`np.linspace(0, 1, 11)`)
- **Scaler** (image-scale proxy): 1–4, 7 values, **log-spaced**
  (`np.geomspace(1, 4, 7)`)
- **Roughness:** discrete set `{0, 0.002, 0.004, 0.006, 0.008, 0.01,
  -0.01, -0.02, -0.03, -0.04, -0.05, -0.06}` (positive = random surface
  jitter, negative = reflective baseline)

There is **no zero-padding**. Each generated contour is cut down to a
fixed **50-point window** around the contact point (10 surface-side
points + 40 drop-side points, via `sampling()` in
`create_contours_args6.py`), and during generation the window is also
shifted ±5 points relative to the contact point
(`window_shift_start=-5, window_shift_end=5`) to augment the model
against small errors in contact-point localisation at inference time.
This is why the model's runtime input shape is fixed at `(50, 2)` — see
`opendrop_ml/modules/ML_model/prepare_experimental.py`.

If you're checking whether some other dataset overlaps this training
grid, these are the exact bounds/steps to check against.

## Create the training data set

- `create_dataset_parallel.sh` — PBS/HPC job script that fans the
  contour-generation calls out across CPUs with `xargs -P`. Run this to
  regenerate the dataset. **Paths and PBS directives (`-P oe97`,
  `/scratch/oe97/...`) are specific to the original NCI Gadi cluster job
  and need editing for any other environment.**
- `create_contours_args6.py` — solves the Young-Laplace equation
  (`draw_drop`/`yl_eqs`) for the given angle/Bond/scale/roughness,
  extracts the contour, and writes sampled 50-point windows to a `.pkl`
  per angle/roughness combination. Called by `create_dataset_parallel.sh`.
- `create_env.sh` — sets up the Python venv used on the HPC cluster.

Unlike the old pipeline there's no separate `combine.py`/`zeropad.py`
step — each `.pkl` already contains normalised, fixed-length samples
ready to train on directly.

## Train the model

- `gpu_train.sh` — PBS/HPC GPU job script. Its active line runs
  `finish_optuna.py`; `train_optuna.py` and `load_optuna.py` are earlier
  stages of the same hyperparameter search, kept commented out in the
  script as a record of how the search was staged.
- `train_optuna.py` — first-stage Optuna hyperparameter search (Conv1D
  model over the `(50, 2)` windows).
- `load_optuna.py` — resumes/continues a hyperparameter search from an
  existing Optuna study.
- `finish_optuna.py` — the stage that actually produced the shipped
  model: seeds a specific parameter set as a trial (the best-found
  hyperparameters) and runs it to completion.
- `hyperparameter_study.db` — the Optuna SQLite study database from the
  original run; `examine_db.py` / `load_optuna.py` read from it. Useful
  for inspecting what hyperparameters were tried and which one won.
- `examine_db.py` — prints all trials in a `.db` study, sorted by score.
- `test_evaluation.py` / `test_model.sh` — post-training step: loads the
  finished model from its `_mse_…` output directory and scores it on the
  held-out test split. `test_model.sh` is the PBS wrapper that runs it.

## Where the runtime model lives

The TensorFlow SavedModel this pipeline produced is **not** in this
directory — see `opendrop_ml/modules/ML_model/`. `trained_model/` here
holds diagnostic training artifacts only (loss curves, model summary);
see `trained_model/README.md`.
