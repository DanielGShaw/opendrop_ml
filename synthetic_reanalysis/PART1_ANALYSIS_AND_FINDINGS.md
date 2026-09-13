# Off-Grid Synthetic Reanalysis — Part 1: Analysis and Findings

**Project**: Build a training-grid-independent synthetic evaluation set and quantify how much of the reported ML-model accuracy is grid coincidence vs. genuine generalization
**Date**: 2026-08-14 (rev. 26)
**Developer**: dgshaw
**Status**: **Complete.** This part is the finished record of Phases 1–5 — design, execution, findings, and the post-hoc review. It is history: read it, don't edit it except to correct an error.

> **Remaining work is in [PART2_REMAINING_WORK.md](PART2_REMAINING_WORK.md)** — Phase 6 (fix the three bugs this study surfaced) and Phase 7 (graduation into `paper_results/`). That is the document to work from.

## How to read this document

It is revision-layered: conclusions were corrected in place by later revisions rather than rewritten, so the changelog is the audit trail and **later always supersedes earlier**. Two consequences:

1. **Do not quote a conclusion from the middle of the Findings section without checking whether a later revision corrected it.** Revisions 13 and 17 both over-generalised from a single test point and were later corrected; the rev. 24 post-hoc review corrects several more.
2. **The authoritative result is the "Revised verdict" at the end of the rev. 24 section.** The Phase 5 verdict above it is superseded in part and is marked as such.

If you only need the answer: read the Background's baseline numbers, then the rev. 24 post-hoc review, then stop.

---

## Background

While auditing `OpenDrop_OP/training_files/` against `OpenDrop_OP/paper_results/data/synthetic/` (the 324-image synthetic evaluation set), two things were found and the first has already been fixed:

1. **`training_files/` contained the wrong pipeline.** It held a reformatted copy of the superseded `paper_results/conan_model_repo/training files/` pipeline (angle 110°–180° only, Bond number 0–2, scaler 1–6, contours zero-padded to a global max length). This does not match the model actually shipped at `opendrop_ml/modules/ML_model/`.
   **Fixed**: the real pipeline was recovered from `files_from_old_computer/_mse_0.4218772351741791/training_files/` (confirmed via MD5 match on `saved_model.pb` against the shipped model) and now lives in `training_files/`. See `training_files/README.md` for the corrected pipeline description.

2. **The real training grid overlaps the synthetic evaluation set almost completely**, which this document addresses. The real generator (`create_contours_args6.py`, via `create_dataset_parallel.sh`) sweeps:

   | Parameter | Training grid | `paper_results/data/synthetic/` values | Overlap |
   |---|---|---|---|
   | Contact angle | 20°–180°, step **0.05°** (continuous, ~3200 points) | 20, 40, 60, …, 180 (9 values) | Every synthetic angle is within 0.05° of a training point — angle is too densely swept to construct a meaningful "off-grid" angle |
   | Bond number | `linspace(0, 1, 11)` → 0, 0.1, 0.2, …, 1.0 | 0, 0.2, 0.4, 0.6, 0.8, 1.0 | **Exact match** on every value |
   | Scaler | `geomspace(1, 4, 7)` → 1, ~1.26, …, 4.0 (log-spaced) | 1.0, 4.0 | **Exact match** on both (the grid's endpoints) |
   | Roughness | `{0, .002, .004, .006, .008, .01, -.01, -.02, -.03, -.04, -.05, -.06}` | -0.03, 0.0, 0.006 | **Exact match** on all three |

   So, with the corrected training-grid knowledge, essentially all 324 images in the current synthetic evaluation set sit on training grid points for Bond number, scaler, and roughness.

   **What this set does and doesn't test.** Training is *not* purely analytical: `create_contour()` (`create_contours_args6.py:388`) renders the drop with `draw_drop(..., dpi=64)` and extracts the contour from the rendered PNG via `extract_edges_CV()`. The rendering path is therefore **shared** between training and evaluation. The two genuine differences are narrower:

   - **Contact point.** Training uses the exact CP returned by `draw_drop`, snapped to the nearest contour point (`prepare_contour`, lines 348–357). Evaluation uses `CPID`'s *estimate*. This is the main untested path.
   - **Edge extractor.** Training uses a fixed `threshValue=50` + `findContours` with no blur (`extract_edges_CV`, line 286). Evaluation uses `isolate_contour(..., blur=True)`. Different contours from the same image.

   The underlying shape → angle mapping is not independently tested by the current set at all.

   **Important caveat, not to overstate this**: this is proximity to the training *grid*, not literal duplication of training *samples*. Roughness perturbation uses unseeded `np.random.uniform`, so even an on-grid image is a different random realization each render. Training also augments each contact point with a ±5-point window shift (`window_shift_start=-5, window_shift_end=5` in `create_contour_dataset`), whereas evaluation uses whatever window CPID happens to land on. So this isn't strict train/test leakage in the sense of a duplicated row — it's closer to "evaluating at exactly the coordinates the model was interpolation-trained on," which is expected to look better than true extrapolation and shouldn't be read as a held-out generalization test.

### Baseline numbers (measured 2026-08-13, from existing results)

Over 40–160° (both sides pooled, n=504 angle measurements from the 324 on-grid images):

| Quantity | Value |
|---|---|
| On-grid synthetic **ML** MAE | **1.397°** (median 0.826°, p90 3.02°) |
| On-grid synthetic YL MAE (next-best method) | 1.823° |
| On-grid synthetic tangent / polynomial / circular / ellipse MAE | 5.96° / 7.24° / 10.03° / 4.68° |
| Model's own **validation RMSE** | **~0.65°** (val MSE 0.4219, from the `_mse_0.4218772351741791` run directory; `finish_optuna.py:273`) |
| Experimental \|ML − YL\| disagreement, 40–160° | 3.02° (median 2.27°, n=297) |

Two consequences:

- **The image pipeline already costs ~0.75° of the error budget.** ML error on rendered on-grid images (1.40°) is roughly double the model's own validation RMSE (0.65°) on training-format windows. CPID + `isolate_contour` account for about half the current error. Any off-grid effect at the image level will be partly buried in that noise — which is precisely why the contour-level arm (Phase 2A) is the decisive test of the mapping.
- **There is no experimental ground truth.** `evaluate_experimental.py` does not parse any reference angle; `plot_experimental.py` compares ML against YL. The paper's experimental evidence is inter-method *agreement*, not accuracy. There is therefore no experimental MAE from which to derive a significance threshold — see "Significance criteria" below for the substitute.

## Goal

Build a **sister synthetic dataset**, roughly the same size as the existing one (~324 images), whose Bond number, scaler, and roughness values deliberately avoid the training grid — plus a subset that goes outside the trained domain bounds — then run the same evaluation machinery against it and compare against the existing on-grid set. Separately and more cheaply, sweep the same off-grid space at the **contour level**, bypassing the image pipeline, to test the shape → angle mapping in isolation.

### Non-goals
- Not retraining or modifying the model.
- Not touching the angle dimension's sampling — it's already continuous/dense, so no angle value is meaningfully "off-grid". **The primary comparison reuses the existing set's 9 angles exactly** (see "Design", point 1).
- Not replacing the existing `paper_results/data/synthetic/` set during the investigation — it stays as-is. (Superseding it in the paper is a possible *outcome*; see Phase 5.)

## Where this work lives

Diagnostic scripts are built in an isolated staging area so the existing `paper_results/` outputs stay reproducible while this is in flux:

```
OpenDrop_OP/synthetic_reanalysis/          # isolated staging area
├── PART1_ANALYSIS_AND_FINDINGS.md         # this document — the completed record
├── PART2_REMAINING_WORK.md                # Phases 6-7 — the document to work from
├── config.py                              # own paths; imports opendrop_ml + training_files directly
├── matrix_utils.py                        # grid-distance metric + seeding helpers
├── offgrid_matrix.py                      # parameter-matrix construction
├── contour_level.py                       # shared contour-level prediction helper
├── generate_offgrid_synthetic.py          # Phase 1
├── acceptance_gate.py                     # Phase 1 acceptance gate
├── contour_level_sweep.py                 # Phase 2A — no image pipeline
├── spotcheck_pipeline.py                  # Phase 2B — smoke test
├── evaluate_offgrid.py                    # Phase 3 (thin wrapper reusing paper_results/evaluate/evaluate_synthetic.py)
├── analyze_grid_distance_slope.py         # Phase 2A slope analysis
├── compare_grid_vs_offgrid.py             # Phase 4
├── data/
│   ├── synthetic_offgrid/                 # 324 off-grid images
│   └── synthetic_ongrid_control/          # matched on-grid re-render, fresh roughness draws
├── results/                               # all result CSVs + provenance sidecars
├── scratch/
└── figures/
```

It imports from `opendrop_ml` and reuses `paper_results/evaluate/evaluate_synthetic.py` (`parse_synthetic_filename`, `evaluate_image`) and `training_files/create_contours_args6.py` (`draw_drop`, `create_contour`, `sampling`, `normalize_ds`) via `sys.path` insertion, exactly as the existing `paper_results/` scripts already do — so isolation costs nothing in import complexity.

**Dataset quality**: the off-grid dataset itself is generated to **paper quality from the start** — final size, stable naming, provenance stamps, its own README — because it is likely to graduate into `paper_results/` rather than be discarded. Only the analysis scripts are treated as throwaway.

## Design of the off-grid parameter matrix

### 1. Angles are held fixed

The primary comparison uses the **same 9 angles as the existing set** (20, 40, …, 180). Error is strongly angle-dependent for every method, so changing angles between the two sets would confound "off-grid" with "different angle distribution" — and angle is not the variable under test. Off-multiple-of-20 angles may be added later as a separate secondary set; they are not part of the primary comparison.

### 2. Roughness is two regimes, not one axis

Positive roughness = random surface jitter; negative = a reflective baseline (`draw_drop`, lines 113–147 vs 154–186). These are different rendering modes — there is nothing "between" +0.006 and −0.03. Roughness is therefore treated as **(sign = categorical regime, magnitude = continuous coordinate)**, and off-grid/nearest-neighbour matching happens *within* a regime, never across.

Note also that roughness perturbs only `baselinex`/`baselinez`. The drop contour `X, Z` from the ODE solve is never touched, so **the angle label remains exact at every roughness value** — roughness is feature noise in the 10 surface-side points of the model window, not label noise. (The code additionally guards CP occlusion: `dpZ ~ uniform(-r·h, 0)` only moves the surface down, and the two baseline points nearest each CP get half jitter and no x-jitter.) The thing to verify at out-of-domain roughness is whether the *drop-side* contour is still extracted cleanly, not whether the CP is buried.

### 3. Scaler is bounded below by frame geometry

Larger scaler ⇒ larger frame extent ⇒ **smaller drop in the image** (fewer pixels; it is a proxy-resolution parameter). The frame carries a fixed 4/3 margin factor (`img_width = 2·max(X)·4/3·scaler`, line 92, and the height branch likewise), so the drop **exactly fills the frame at scaler = 0.75**. Below that it is clipped by construction, and `IGNORE_EDGE_MARGIN` in both edge extractors will silently truncate the contour rather than raise. Accordingly:

- The originally proposed 0.7 is **dropped** — it is past the geometric limit, not near it.
- Low-side extrapolation sits at **0.95**, with a hard assert that the rendered contour touches no frame edge.
- **High scaler is the primary extrapolation arm** (5.0, optionally 6–8): geometrically safe, and the physically relevant direction — small drop, few pixels, which is the real-experiment failure mode.

Ground-truth discretisation fidelity was originally going to be measured by a dpi=64-vs-dpi=256 model-prediction comparison per scaler; that approach was retired (see A.5) as measuring the wrong thing — dpi isn't the resolution-mimicking axis here, scaler is, by design. Resolution adequacy is instead assessed via `profile_length` against the real pipeline's own thresholds once images run through evaluation (Phase 3/4), not as a generation-time exclusion.

### 4. Reflective-roughness / low-scaler interaction

In the negative-roughness branch the drop is raised to make room for the reflection (`Z = Z + shift + (max(ref) − min(ref))`, line 175) — but `ylim` was already fixed above that line. The raise is bounded by roughly `|roughness|·img_height` against ~25% headroom from the 4/3 factor: fine alone at −0.08, but it **compounds with low scaler**, and scaler 0.875 × roughness −0.08 is exactly a corner this matrix wanted — and, empirically (see Findings below), broke: below scaler≈0.9 at this roughness, `extract_edges_CV`'s contour ends up entirely on one side of the computed apex x-position, crashing `prepare_contour`. Resolved by moving the low-scaler extrapolation value to 0.95 (still below the training grid's minimum of 1.0), confirmed clear at all 9 angles.

### 5. Proposed values

| Parameter | Training grid points | **Interpolation** (mid-grid, in domain) | **Extrapolation** (out of domain) |
|---|---|---|---|
| Bond number | 0, 0.1, …, 1.0 | 0.05, 0.15, 0.35, 0.55, 0.75, 0.95 | 1.2, 1.5 |
| Scaler | 1, 1.26, 1.59, 2.0, 2.52, 3.17, 4.0 (geomspace) | ~1.13, ~1.41, ~1.78, ~2.24, ~2.83, ~3.56 | **5.0** (primary), 0.95 (secondary, frame-asserted) |
| Roughness (+) | 0, .002, .004, .006, .008, .01 | 0.003, 0.005, 0.009 | 0.02 |
| Roughness (−) | −.01 … −.06 | −0.015, −0.025, −0.045 | −0.08 |
| Angle | continuous, 20–180° step 0.05° | **20, 40, …, 180 — identical to the existing set** | n/a |

Note the asymmetry to state explicitly in the write-up: the existing set uses scaler ∈ {1.0, 4.0}, the grid's **endpoints**. The interpolation midpoints are *interior*, i.e. easier, not harder. Combined with the window-local normalisation (below), **scaler is the weakest discriminating dimension and a null there is expected** — this is predicted in advance so a null is not later read as evidence of anything.

### 6. Window-local normalisation limits what scaler can do

`normalize_ds` divides each window by `max(contour[:,1])` of the **50-point window**, not the drop apex — the normalisation is window-local. (Both the training and inference docstrings claimed apex normalisation; both have been corrected as part of this work. The apex-relative version is the disabled `if 0:` block at `create_contours_args6.py:362`.) Scaler therefore survives into the model input essentially only as contour discretisation granularity.

### 7. Exact combinations

Finalised by `offgrid_matrix.py` rather than hand-enumerated, following the existing combinatorial structure (angles × Bond × scaler × roughness), sized near 324 images, with an explicit programmatic trim rule (not trimmed by eye) so in-domain / out-of-domain cells stay balanced. Each combination carries an `in_domain` / `out_of_domain` flag and its grid-distance value (below) in an accompanying CSV.

### 8. On-grid control with replicates

`data/synthetic_ongrid_control/`: a re-render using the **existing** on-grid parameter combinations with fresh roughness draws. This separates "off-grid parameters changed the result" from "re-rendering with different random noise changed the result".

A single control render gives a point estimate with no variance. Render **3–5 replicates of a subset** (roughness is unseeded, so replicates are free) and use that spread as the **render-noise null band** against which the off-grid Δ is judged.

### 9. Grid-distance metric

For nearest-neighbour matching and for the primary endpoint, distance from a parameter point to the nearest training grid point is computed as:

- **Angle**: matched exactly, not distance-weighted.
- **Roughness sign**: categorical — matching only within regime.
- **Scaler**: distance in **log** space (the grid is `geomspace`).
- **Bond, |roughness|**: standardised by their respective grid steps so no dimension dominates.

## Significance criteria

- **Statistical**: the off-grid vs. on-grid paired ΔMAE is *real* if its bootstrap 95% CI excludes the render-noise band measured from the Phase-1 replicates. This one is data-determined — nothing to pre-commit.
- **Practical**: it is *material* if ΔMAE ≥ **1.4° (PROVISIONAL — see below)** — a doubling of the current on-grid ML MAE (1.397°).
  *This is a substitute.* The original intent was 1.5× the experimental MAE over 40–160°, but the experimental set has no ground truth (see Background), so no experimental accuracy figure exists. The 1.4° bar is cross-referenced against the 3.02° \|ML − YL\| disagreement on real data: a gap smaller than the spread between the two best methods on real drops will not change a user's conclusion.
- **Primary endpoint**: the slope of per-measurement \|ML error\| regressed on grid-distance, from the Phase-2A contour-level sweep. One number with a CI, using every sample rather than discarding information into buckets, with the classical methods supplying the null slope at image level. Everything else is reported as secondary.
- **Reporting a null**: if no gap is detected, it is reported as **bounded** — "any grid-proximity effect is below X° (95% CI)", where X comes from achieved power — never as simply absent.

### 🔶 DECISION POINT — finalise the practical bar (after Phase 2A on-grid, before any off-grid comparison)

The 1.4° figure is a placeholder chosen before any noise scale was known. It **is expected to be revised**, and the revision happens at one specific place in the plan:

**When**: after the Phase-1 render-noise band exists (A.5) and after the Phase-2A sweep's **on-grid** portion has been run — but **before** computing any off-grid delta, slope, or comparison.

**Why there**: at that moment two of the three unknowns are resolved (render-noise scale, and contour-level on-grid error against the 0.65° validation benchmark), while the off-grid result is still unseen. Setting the bar from inputs that are independent of the answer keeps it a genuine threshold. Setting it *after* seeing the off-grid delta would make Phase 5's fork circular — the failure mode the rest of this section exists to prevent.

**Inputs to use**: the render-noise band width; contour-level on-grid MAE/RMSE vs. the 0.65° validation RMSE; the achieved CI width at the realised sample size (which sets the smallest bar that is even detectable — a bar below the CI half-width is unusable).

**What to record**: both the provisional 1.4° and the final value, the date, and the reasoning, in the Findings section. Do not silently overwrite the number above — strike it through and note the replacement, so the sequence stays auditable.

> **RESOLVED (2026-08-13).** ~~1.4°~~ **→ two bars, scoped by which noise regime the comparison lives in:**
> - **Contour-level primary endpoint (Phase 2A): 0.2°.** Measured inputs: render-noise band [0.589°, 0.643°] (SD 0.020°) from the 5 Phase-1 replicates; on-grid contour-level MAE 0.674° / RMSE 1.103° against the 0.65° validation-RMSE benchmark (excluding the 36 angle=180° points, which carry a known one-sided boundary-extrapolation artifact — see Findings — MAE 0.472°/RMSE 0.696°, matching the benchmark closely); bootstrap 95% CI half-width 0.094° at n=324, 0.023° at the planned off-grid n=5400. All of these cluster in 0.02-0.1°. The original 1.4° (2x the *image-level* on-grid ML MAE) was calibrated for a much noisier regime — the Background section already shows the image pipeline adds ~0.75° of its own noise beyond the contour-level mapping — so applying it unchanged to the contour-level endpoint would be a bar effects could clear "significantly" without being remotely material. 0.2° sits at ~10x the render-noise SD and ~9x the achieved CI half-width at n=5400: comfortably above the noise floor, far more sensitive than 1.4° would allow, matching Phase 2A's role as the decisive, cheap test.
> - **Image-level comparisons (Phase 4): 1.4° retained**, since that's the regime it was actually derived from (2x the image-level on-grid ML MAE of 1.397°, cross-referenced against the 3.02° real-data \|ML-YL\| disagreement).

**Knock-on**: the dpi-based GT-fidelity threshold this note originally referred to has been retired (see A.5) — resolution adequacy is now assessed via `profile_length` against `prepare_experimental.py`'s existing thresholds, not a bar-derived cutoff, so there is nothing here that moves with the practical bar.

## Implementation phases

### Phase 1 — Generator + parameter matrix
- `generate_offgrid_synthetic.py` imports `draw_drop` / `create_contour` from `training_files/create_contours_args6.py` (dpi=64, matching the existing set) rather than reimplementing the renderer. `draw_drop` defaults to `dpi=256` — 64 must be passed explicitly.
- **Seed the roughness draw per image**, derived from the parameter tuple, so the dataset regenerates identically; keep an explicit unseeded mode for the replicate control.
- **Format parameters at fixed decimals** and assert filename uniqueness (`str(np.float64)` on geomspace midpoints yields `1.4142135623730951`; parseable but collision-prone under later rounding).
- Save PNGs to `data/synthetic_offgrid/` using the **same filename convention** as the existing set (`{angle}_{Bond}_{scale}_{roughness}_CPL_{x}_{y}_CPR_{x}_{y}_.png`) so `parse_synthetic_filename()` works unmodified.
- **Per-image asserts**: contour touches no frame edge (apex-in-frame); CP within the contour; profile length sane.
- **Measure GT discretisation fidelity per scaler** (fit the analytic contour near the CP, compare to nominal); drop cells whose label error approaches the significance threshold.
- Render `data/synthetic_ongrid_control/` plus 3–5 replicates of a subset.
- Write the matrix CSV: parameters, `in_domain` flag, grid-distance, seed.

**Phase 1 acceptance gate**: re-render one existing on-grid combination and confirm image dimensions and CP coordinates match that image's filename values to within a pixel. This validates dpi/bbox settings before generating hundreds of images.

### Phase 2A — Contour-level sweep (the decisive, cheap test)
Bypasses the image pipeline entirely, isolating the shape → angle mapping:

`create_contour` → `sampling(window_size=50, shift 0, surface_points=10)` → `normalize_ds` → direct `model.signatures["serving_default"]` call, using the **exact** CP — reproducing training preprocessing precisely. No `evaluate_image`, no classical fits, milliseconds per sample.

- Sweep thousands of off-grid (Bo, scaler, roughness) points — this is where the statistical power is, not in ~324 images.
- **Benchmark against the model's own validation RMSE (~0.65°).** If off-grid contour-level RMSE lands near 0.65°, there is no mapping-level grid effect and the question is settled without the image pipeline. If it is much worse, the problem is localised to the mapping before any image work. Caveat to state: that validation split was itself drawn from the on-grid sweep, so 0.65° is an interpolation number — the right *ceiling* to compare against, not evidence of generalization.
- Produces the primary endpoint (error vs. grid-distance slope).
- **This is the early go/no-go on effect size**, and it runs in the time the Phase-2B subset takes.

**Run order within this phase is load-bearing.** Run the on-grid portion of the sweep first and stop. At that point, take the 🔶 decision point above and finalise the practical bar. Only then run the off-grid portion and compute deltas and slopes. Splitting the phase this way costs nothing — it is the same sweep, executed in two halves — and it is what keeps the threshold independent of the result it will be applied to.

### Phase 2B — Image-pipeline smoke test
- ~15–20 off-grid images spanning the matrix (interpolation + extrapolation, several angles) plus a matched ~15–20 from the on-grid control.
- Run the full `evaluate_image()` pipeline on just this subset.
- Check for: CPID / edge-detection failures at extreme roughness or scaler, sane profile lengths, no frame-clipping.
- **Explicitly a smoke test and a large-effect detector, not a confirmation.** At n≈20 a sub-degree gap is indistinguishable from render noise; early certainty about a *slight* difference comes from Phase 2A, not here.
- Go/no-go: if the renderer or preprocessing breaks on some corner of the matrix, fix the generator or trim the matrix here — cheaply.
- Output: a short printed/markdown summary, not a formal CSV.

### Phase 3 — Full image-level evaluation
- `config.py` defines `SYNTHETIC_OFFGRID_DATA_DIR` and `SYNTHETIC_ONGRID_CONTROL_DIR`.
- `evaluate_offgrid.py` reuses `evaluate_synthetic.py`'s `evaluate_image` / `parse_synthetic_filename` against both new directories — no duplicated fitting/ML logic.
- **Checkpoint the results CSV** (append per image). Runs include YL fits; a crash at image 300 must not cost the run. Estimate wall clock from the timing columns already in `synthetic_results.csv` before starting.
- **Stamp provenance** into each results CSV: `saved_model.pb` MD5, TF version, code commit, generator seed. (`results/synthetic_results.csv` is dated Jul 3 10:55 and `preprocessing.py` Jul 2 18:38, so the existing CSV does postdate the `isolate_contour` reflective-surface fix — but pin this explicitly rather than relying on mtimes, since roughness −0.03 images are in that set and are exactly what that fix touched.)
- Output: `results/synthetic_offgrid_results.csv`, `results/synthetic_ongrid_control_results.csv`, plus replicate CSVs.

### Phase 4 — Comparative analysis
`compare_grid_vs_offgrid.py` joins the contour-level sweep and the three image-level result sets (original on-grid, fresh on-grid control, off-grid).

- **Primary**: error vs. grid-distance slope with bootstrap CI (Phase 2A), and the same regression at image level with classical methods as the null-slope control.
- **Paired matching**: each off-grid point matched to its nearest on-grid point under the Section-9 metric, comparing **signed bias** as well as absolute error. Also report whether Δ correlates with distance-to-nearest-grid-point — that correlation is the cleanest single piece of evidence, since each nearest neighbour is by construction only ~half a grid step away and paired Δ alone conflates "off-grid" with "slightly different physical drop".
- **Render-noise null band** from the Phase-1 replicates, as the reference against which any Δ is judged.
- **Control interpretation, with a caveat**: classical fits have no training grid, so flat classical error + rising ML error isolates an ML-specific gap. But circular and ellipse fits degrade with Bond number by construction — a high-Bo drop is not a spherical cap — and the extrapolation cells (Bo 1.2, 1.5) are exactly where they are worst. So "both rise" is *uninformative*; lean on the matched-difficulty pairing rather than on classical-fits-as-null.
- **CP localisation as a mediator**: `parse_synthetic_filename` already returns `gt_cpl_x/y`, `gt_cpr_x/y` and `evaluate_image` already records `left_CP_x` etc. Report CP error in px and its correlation with per-image ML angle error. If the off-grid gap tracks CP error, it is a CPID story, not a generalization story — and given that CPID + `isolate_contour` already account for ~half the current error budget, this is a live possibility. CPID is an algorithm, not a trained model, so it cannot have a grid-proximity effect of its own: it is the noise floor. Flat contour-level error with rising image-level error ⇒ the difference is CPID noise.
- **Beyond MAE**: median, p90, max, and **failure rate** (exceptions / NaN counts) per method. Grid-proximity effects, if real, are likelier to appear in the tail than the mean.
- Split all of the above by in-domain / out-of-domain and by angle bucket.

### Phase 5 — Findings write-up + reflection

Write-up only. No code moves, no paper edits, no bug fixes — those are Phases 6 and 7.

- Summarize results in a "Findings" section appended to this document, including the bounded-null statement if applicable.
- State the **overall verdict once**, in language that can be lifted into the paper, with its bounds attached (parameter ranges covered, metric used, what was *not* tested).
- Record every real defect surfaced along the way, with reproduction, so none of it lives only in conversation. These become Phase 6's input; do not attempt fixes here — the measurement must not move while the study is being written up.
- **Reflect on the method, not just the result**: which parts of the design earned their place, which were dead weight, and which conclusions rest on thinner evidence than their phrasing implies. The rev. 24 post-hoc review is part of this phase and is where most of that reflection landed — including three cases where an earlier revision's conclusion had over-generalised from a single test point.
- Note explicitly what the study *cannot* answer, so the paper doesn't overclaim: the training-data-quality question (rev. 17), the per-axis estimates that were never bootstrapped, and any regime outside the tested ranges.

**Deliverables**: the Findings section of this document, the revised verdict, the Phase 6 bug inventory, and draft caveat text for the paper (drafted, not applied — applying it is Phase 7).

> **Phases 6 and 7 live in [PART2_REMAINING_WORK.md](PART2_REMAINING_WORK.md)**, because both were scoped from what the Findings and the rev. 24 post-hoc review turned up rather than planned up front. **Phase 6** fixes the three defects the study surfaced; **Phase 7** is graduation into `paper_results/` and the paper-output updates, and must run after Phase 6.

---

# Appendix A — Implementation contract

Everything below is **prescriptive**. Where a value is stated, use that value; where an assert is stated, write it. This section exists so the plan can be implemented without further design decisions.

## A.1 The parameter matrix (`offgrid_matrix.py`)

Full factorial over the Section-5 values would be 9 × 8 × 8 × 8 = 4,608 images. Instead build **36 (Bo, scaler, roughness) cells**, crossed with the 9 fixed angles, for exactly **324 images** — matching the existing set's size.

**Interpolation block — 24 cells** (full factorial, all in domain):

```
Bo        ∈ {0.15, 0.55, 0.95}          (3)
scaler    ∈ {1.41, 2.83}                 (2)
roughness ∈ {0.003, 0.009, -0.015, -0.045}  (4)
→ 3 × 2 × 4 = 24 cells, in_domain = True
```

**Extrapolation block — 12 cells.** One axis out of domain at a time (clean attribution), with the other axes held at interpolation values, plus two compound corners:

| # | Bo | scaler | roughness | Out-of-domain axis |
|---|---|---|---|---|
| 1–2 | **1.2**, **1.5** | 1.41 | 0.003 | Bo |
| 3–4 | **1.2**, **1.5** | 1.41 | −0.015 | Bo |
| 5–6 | 0.55 | **5.0**, **0.95** | 0.003 | scaler |
| 7–8 | 0.55 | **5.0**, **0.95** | −0.015 | scaler |
| 9 | 0.55 | 1.41 | **0.02** | roughness (+) |
| 10 | 0.55 | 1.41 | **−0.08** | roughness (−) |
| 11 | **1.5** | **5.0** | 0.003 | compound (Bo × scaler) |
| 12 | 0.55 | **0.95** | **−0.08** | compound (scaler × roughness) — the frame-headroom corner of Section 4 |

Angles: `[20, 40, 60, 80, 100, 120, 140, 160, 180]` for every cell.

**Trim rule**: if a cell fails any Phase-1 assert (A.4), **drop it and record the reason in the matrix CSV; do not substitute another cell.** The set is allowed to come in under 324. Never trim by eye.

**Matrix CSV columns**: `cell_id, angle, bond, scaler, roughness, roughness_regime, in_domain, ood_axis, grid_distance, seed, filename, status, drop_reason`.

## A.2 Grid-distance metric

Per-axis distance to the nearest training grid point, in units of that axis's grid step, combined in quadrature. Angle contributes 0 (matched exactly).

```python
BO_GRID     = np.linspace(0, 1, 11)          # step 0.10
SCALER_GRID = np.geomspace(1, 4, 7)          # log step ln(4)/6 = 0.23105
ROUGH_POS   = np.array([0, .002, .004, .006, .008, .01])   # step 0.002
ROUGH_NEG   = np.array([-.01, -.02, -.03, -.04, -.05, -.06])  # step 0.01

def grid_distance(bond, scaler, roughness):
    d_bo   = np.min(np.abs(BO_GRID - bond)) / 0.10
    d_sc   = np.min(np.abs(np.log(SCALER_GRID) - np.log(scaler))) / (np.log(4) / 6)
    grid_r = ROUGH_POS if roughness >= 0 else ROUGH_NEG
    step_r = 0.002     if roughness >= 0 else 0.01
    d_r    = np.min(np.abs(grid_r - roughness)) / step_r
    return float(np.sqrt(d_bo**2 + d_sc**2 + d_r**2))
```

Notes that must be honoured:
- **Roughness sign partitions.** Never compute distance across the sign boundary — positive jitter and negative reflection are different rendering modes. `grid_distance` above enforces this by selecting the grid; nearest-neighbour *matching* in Phase 4 must likewise filter to same-sign candidates before searching.
- Out-of-domain values fall out naturally as distance to the nearest endpoint (e.g. Bo 1.5 → `d_bo = 5.0`).
- On-grid points give distance 0, which is what makes the pooled regression in A.6 work.

## A.3 Seeding and filenames

`draw_drop` perturbs roughness with **global** `np.random.uniform`, so seeding must use `np.random.seed()` immediately before each call — not a `Generator`.

```python
import hashlib

def cell_seed(angle, bond, scaler, roughness):
    key = f"{angle:.4f}|{bond:.4f}|{scaler:.4f}|{roughness:.4f}"
    return int(hashlib.blake2b(key.encode(), digest_size=4).hexdigest(), 16)

np.random.seed(cell_seed(angle, bond, scaler, roughness))   # immediately before
img, CPs_img = draw_drop(angle, bond, scaler, roughness, 5000, dpi=64)
```

For the **replicate control** (A.5), use `np.random.seed(cell_seed(...) + replicate_index)` so replicates differ but the whole set still regenerates identically.

**Filename format** — 1 decimal for angle (matching the existing set), 4 for the rest:

```python
fname = (f"{angle:.1f}_{bond:.4f}_{scaler:.4f}_{roughness:.4f}"
         f"_CPL_{lx}_{ly}_CPR_{rx}_{ry}_.png")
```

`parse_synthetic_filename` calls `float()` on each part, so decimal counts need not match the existing set — **all joins between result sets must be on parsed float values, never on filename strings.** Assert filenames are unique before writing; a collision means two cells rounded together and the matrix is wrong.

## A.4 Phase 1 asserts

Run per rendered image; on failure, drop the cell and record `drop_reason`.

```python
# 1. Frame geometry — the drop must not touch any frame edge.
#    (IGNORE_EDGE_MARGIN in both edge extractors silently truncates rather
#     than raising, so this must be checked explicitly.)
contour = extract_edges_CV(img)
h, w = img.shape[:2]
assert contour[:, 0].min() > 1 and contour[:, 0].max() < w - 2
assert contour[:, 1].min() > 1 and contour[:, 1].max() < h - 2

# 2. Contact points are inside the frame and on the contour's row range.
assert 0 < lx < w and 0 < ly < h and 0 < rx < w and 0 < ry < h

# 3. Resolution floor — reject drops too small for meaningful discretisation.
drop_height_px = ly - contour[:, 1].min()
assert drop_height_px >= 40

# 4. Window feasibility (the failure mode described in A.7).
#    The right-side contour must have >= 40 points before the CP.
assert cp_index_in_right_contour >= 40
```

**Acceptance gate before generating the full set**: re-render one existing on-grid combination (e.g. `angle=100.0, Bo=0.0, scaler=1.0, roughness=-0.03`) and assert

- `img.shape == (492, 744, 3)` — all 324 existing images are exactly this shape, so any deviation means the dpi/bbox settings are wrong;
- returned CPs within 1 px of the values in that image's filename (`CPL_97_447_CPR_647_447`).

## A.5 GT fidelity (retired) and the render-noise band

**GT fidelity (per scaler) — retired, 2026-08-13.** The originally specified check (render one cell at `dpi=64` and the same cell at `dpi=256`, compare the A.7 contour-level model prediction on both, drop scalers where `|pred_64 - pred_256|` exceeds a threshold) was implemented, run, and found to be measuring the wrong thing — for two independent reasons, the second more fundamental than the first:

1. **Windowing artifact.** `sampling()`'s 50-point model window is defined in point *count*, not physical distance, and contour point density scales with `dpi` (measured: 466 points at dpi=64 vs. 1872 at dpi=256 for the same cell — ~4x). So the "same" 50-point window covers ~4x less physical arc length at dpi=256, meaning the two model calls aren't looking at the same neighbourhood of the contact point at all. This alone produced nonsensical results (mean `|pred_64-pred_256|` of 3-12 deg, max up to 76 deg, across all 4 distinct scalers) that look like a massive resolution problem but are actually a comparison-construction bug.
2. **Wrong axis, more fundamentally.** `dpi` isn't the resolution-mimicking parameter in this generator — `scaler` is, *by design*: roughness is computed as `roughness * img_height`, a data-space (physical) quantity that scales with `scaler` but not with `dpi`. This is deliberate so that roughness stays a fixed fraction of drop size (mimicking a real camera's field-of-view/magnification changes) rather than a fixed pixel count. Varying `dpi` instead just rasterizes the *same* physical roughness at a different pixel density — it doesn't correspond to "a higher-resolution photo of the same setup," and the model, which only ever trained on dpi=64 renders, has no basis to be evaluated at dpi=256 in the first place. There is no meaningful "true" high-fidelity render to compare dpi=64 against this way.

**Substitute**: the deployed pipeline already has a real resolution-adequacy check — `opendrop_ml/modules/ML_model/prepare_experimental.py`'s `prepare_for_model()`, lines ~222-248 — operating on `profile_length` (contour points between the two contact points, the same quantity `evaluate_image()` in `evaluate_synthetic.py` already records per image): `profile_len > 974` triggers an automatic 0.5x downsample; `565 < profile_len < 974` conditionally downsamples via an angle-dependent slope check (`6.383 * min(circ_angle) + 208`); `profile_len < 234` prints a low-resolution warning but proceeds regardless. Rather than inventing a synthetic-label-fidelity proxy, Phase 3/4 will cross-reference each cell's `profile_length` (produced for free once images run through the real pipeline) against these exact thresholds — so "is this cell's resolution reasonable" is answered by the same safeguard a real low-resolution photo would hit, not a bespoke dpi experiment. No cells are excluded on this basis at generation time; `drop_height_px` (already recorded per cell) remains the generation-time resolution covariate.

**Render-noise band.** Render `R = 5` replicates of a **subset of 8 on-grid cells × all 9 angles** (72 images per replicate) with seeds `cell_seed + 0..4`. Compute ML MAE per replicate, giving 5 values. The band is `[min, max]` of those 5, reported alongside their SD. A ΔMAE is *real* only if its bootstrap CI lies entirely outside this band. (Implemented and run in Phase 1 — see Status tracking.)

## A.6 Statistical definitions

- **Resampling unit is the image**, not the individual angle measurement. Left and right sides of the same image share a render, a threshold, and a CPID call, so they are correlated; resampling measurements independently would understate CI width.
- **Bootstrap**: 10,000 resamples with replacement over images, percentile method, 95% CI. Fixed seed (`np.random.default_rng(20260813)`) so CIs are reproducible.
- **Primary endpoint — grid-distance slope**: OLS of per-measurement `|ML error|` on `grid_distance`, fitted on the pooled contour-level sweep (on-grid points contribute distance 0). Report **both** OLS and Theil–Sen slopes; if they disagree materially, the tail is driving the result and Theil–Sen is the one to quote. CI by the image-level bootstrap above.
- **Null slope**: the same regression run on each classical method's error at image level. These have no training grid, so their slopes estimate "off-grid cells are simply harder to render/segment."
- **Paired comparison**: match each off-grid measurement to its nearest on-grid measurement — same angle, same roughness sign, minimum `grid_distance` between the two parameter points — and report the mean **signed** difference (bias) as well as the mean absolute difference, both with CIs.
- **Bounded null**: if the CI includes zero, report the bound as the CI's upper limit: "any grid-proximity effect is below X° (95% CI)."

## A.7 Phase 2A skeleton (contour-level sweep)

The fiddliest phase; this is the reference implementation.

```python
import numpy as np, tensorflow as tf
from create_contours_args6 import create_contour, sampling, normalize_ds

model = tf.saved_model.load(ML_MODEL_DIR)

def predict_contour_level(angle, bond, scaler, roughness):
    """Exact-CP, training-format prediction. No image pipeline."""
    np.random.seed(cell_seed(angle, bond, scaler, roughness))

    # right-hand half-drop only, with the EXACT contact point
    right, CPs = create_contour(angle, bond, scaler, roughness,
                                input_len=None, display=False)

    # sampling() re-parses its dict keys with eval(), so the key format
    # is load-bearing: angle_Bo_scaler_roughness_CPx_CPy_0_0
    key = (f"{angle}_{bond}_{scaler}_{roughness}_"
           f"{CPs[1][0]}_{CPs[1][1]}_0_0")

    windows = sampling({key: right}, window_size=50,
                       window_shift_start=0, window_shift_end=0,
                       surface_points=10, display=False)
    ds = normalize_ds(windows)

    window = np.array(list(ds.values())[0], dtype=np.float32)
    assert window.shape == (50, 2), f"bad window shape {window.shape}"

    out = model.signatures["serving_default"](
        conv1d_input=tf.convert_to_tensor(window[None, ...]))   # (1, 50, 2)
    return float(list(out.values())[0].numpy().ravel()[0])
```

Gotchas that will otherwise cost hours:

- **`prepare_contour` mutates its arguments in place** (`coords[:,1] = -coords[:,1]`, `CPs[0][1] = -CPs[0][1]`). `create_contour` calls it, so never reuse a contour or CP array across calls — always re-derive.
- **`sampling` can slice with a negative start.** `bound_start = index - 40 + shift`; if the CP index is under 40, `contour[bound_start:bound_end]` wraps from the end of the array and returns a wrong (or empty) window **without raising**. `sampling` only `print`s a warning. Hence assert 4 in A.4 and the shape assert above.
- **`window_shift_start=0, window_shift_end=0`** gives exactly one window per contour — the shift augmentation is a *training* device and must not be used at evaluation.
- **Right side only.** This arm skips left-side mirroring entirely; that is deliberate, since training used `right_only=True`.
- **Output is a single scalar** for a `(1, 50, 2)` input — unlike `experimental_pred(side="both")`, which passes `(2, 50, 2)` and returns two.
- **No clipping.** `experimental_pred` clamps to [0, 180]; this arm does not, so raw extrapolation behaviour at the 20°/180° endpoints stays visible. Record the unclipped value.

**Sweep size and sampling scheme.** For each of the 9 fixed angles, draw **500 parameter triples** by Latin hypercube over the in-domain ranges (Bo ∈ [0, 1], log-scaler ∈ [log 1, log 4], |roughness| within regime, sign assigned 50/50), plus **100 out-of-domain triples** per angle drawn uniformly from the extrapolation ranges in Section 5 — **5,400 samples total**. At contour-level speed this is minutes, not hours, and it is where the statistical power lives. Use `np.random.default_rng(20260813)` for the LHS draw so the sweep is reproducible.

## A.8 Phase 3 mechanics

- **Checkpointing**: **as implemented (superseding the original append-mode spec below), rewrite the whole results CSV (accumulated rows in memory -> temp file -> atomic `os.replace`) after every image.** The original plan of appending each row with `mode="a", header=False` was tried first and found to have a real bug: pandas' header-less append writes column values *positionally*, so a row with fewer keys than the established header (e.g. an early-exit failure dict with ~11 keys vs. 55+ for a full success) silently shifts every subsequent value into the wrong column instead of raising or padding correctly. Caught in the Phase-3 run (see Findings) after one image legitimately failed preprocessing. Rewriting the full accumulated table each checkpoint lets pandas union columns by name and pad missing ones with NaN, at the cost of an O(n) rewrite per image — cheap at this n (a few hundred rows). On restart, still reads the existing CSV and skips `image_name`s already present, per the original intent.
- **Provenance row**: write a sidecar `<results>_provenance.json` per run containing `saved_model.pb` MD5, `tensorflow.__version__`, `numpy.__version__`, `cv2.__version__`, the generator seed base, and the output of `git rev-parse HEAD` if available.
- **Runtime estimate**: before launching, sum the `*_fit_time` columns in `paper_results/results/synthetic_results.csv` to project wall clock. YL dominates; expect the off-grid run to be slower at high Bond number.

---

## Notes

- Training uses `right_only=True` (`create_contour`, line 398); the left side is mirrored at inference. If any left/right asymmetry appears in ML error, that is the likely cause. Keep sides separate throughout, as `evaluate_synthetic.py` already does.
- `experimental_pred` clips predictions to [0, 180], so error at the 20° and 180° endpoints is one-sided. Relevant when reading the extreme angle buckets.
- Roughness jitter is scaled by `img_height` (line 138), so **roughness and scaler are coupled** — roughness 0.02 at scaler 5.0 is a much larger perturbation than at 0.95. The matrix treats them as independent; the analysis should not.

## Findings so far

**Frame-geometry assert (A.4, check 1) needed correction.** As literally specified, the check (`contour[:, 0].min() > 1 and contour[:, 0].max() < w - 2` on the *whole* contour) rejected 100% of renders, including the acceptance-gate reference image itself — because `draw_drop`'s baseline strip legitimately spans the full frame width by construction on every render (verified: the accepted reference image's contour x-range is `[2, 742]` on a 744px frame). Fixed by restricting the x-edge check to contour points above the baseline row (using the CP y-coordinate as the threshold, without needing to pull `shift` out of `draw_drop`'s internals) — see `generate_offgrid_synthetic.py::render_and_check`.

**The 40px resolution floor (A.4, check 3) was withdrawn as a hard filter after investigation.** Re-rendering the flagged on-grid images (e.g. `20.0_0.0_4.0_-0.03_...`, drop_height_px=13) confirmed they are correctly rendered, not corrupted — a visual sweep at scaler=4.0 (13px @ 20° → 58px @ 80°) showed the shape becoming a recognizably smooth dome around 40px, so the number itself wasn't wrong as a "coarse vs. resolvable" line. But applying it as an exclusion would have removed 48/324 on-grid and 51/324 off-grid cells, **all clustered at low angle x high scaler** — i.e. exactly the regime the scaler parameter exists to simulate (lower camera resolution), and exactly the corner Section 3 already flagged as "the primary extrapolation arm... the physically relevant direction... the real-experiment failure mode." Excluding it would have removed data specifically where a generalization gap is most likely to show up, biasing the study toward a false null. **Resolution**: `RESOLUTION_FLOOR_PX` lowered to 2 (a degenerate-render guard only — with a 20° minimum angle, single-digit heights should never occur legitimately), and `drop_height_px` is now recorded as a column on every row (pass or fail) so Phase 4 can stratify/weight by resolution instead of the dataset silently excluding it. Result: off-grid 321/324 rendered (vs. 270/324 under the withdrawn floor), on-grid control 324/324 (vs. 276/324), all 5 replicate sets 72/72.

**A real bug was found (not fixed in the shared code, worked around in the matrix), at exactly the cell predicted to be risky.** The 3 off-grid drops on the first pass (`IndexError: too many indices for array`, `l_drop[:,0]` on a 1-D empty array) were all matrix cell #12 — `(bond=0.55, scaler=0.875, roughness=-0.08)`, the compound "scaler x roughness" corner Section 4 explicitly named as risky in advance — at angles 140/160/180 specifically (120 and below survived). Root cause, confirmed by a scaler sweep at the same (bond, roughness): `extract_edges_CV`'s contour lands entirely on one side of the apex x-position for scaler ≤ 0.875, and is cleanly split for scaler ≥ 0.9, at every angle tested. This is a pre-existing degenerate-geometry edge case in the shared training pipeline (`training_files/create_contours_args6.py`'s `prepare_contour`), not something introduced by this analysis — it's a real bug, worth fixing in that shared file at some point, but out of scope here since the matrix has full control over which scaler values it probes. **Fix applied**: moved the matrix's low-scaler extrapolation value from 0.875 to 0.95 (Section 3/5 and Appendix A.1 updated to match) — still below the training grid's minimum (1.0), still a genuine extrapolation point, confirmed clear at all 9 angles. Off-grid now renders **324/324**, no drops.

**Phase 2A(ii) — off-grid sweep: a real, material grid-proximity effect, surviving robustness checks.** 5400 samples (500 in-domain LHS + 100 out-of-domain per angle x 9 angles), 0 hard failures. (18 `sampling()` internal warnings fired during the run — all at CP index 44-49, i.e. above the actual wraparound threshold of 40; verified as `sampling()`'s print condition being more conservative than the real danger zone, not data corruption.)

Pooled with the 324 on-grid points (grid_distance recomputed post-hoc for those — most are 0, but the roughness=-0.03 subset sits ~1 grid-step away, per Background), regressing `|error|` on `grid_distance`:

| Fit | Slope (deg / unit distance) | 95% CI |
|---|---|---|
| OLS | 0.198 | [0.149, 0.252] (bootstrap, 10,000 resamples) |
| Theil-Sen | 0.135 | [0.101, 0.171] (Sen 1968 analytical CI — a 10,000-resample bootstrap would cost ~4h at this n due to Theil-Sen's O(n²) pairwise-slope cost) |

OLS/Theil-Sen ratio 1.46 — a material disagreement, so per the plan's tie-break rule **Theil-Sen (0.135°/unit) is the quoted primary estimate**; OLS is being pulled up by tail cases (max single-point error 39.5°, at `scaler=0.94` — right at the edge of the safe zone found during the Phase-1 bug investigation, angle=20 — an extreme-case worth noting under "beyond MAE" tail reporting even though it doesn't drive the robust estimate).

**Sensitivity check**: angle=180° carries a one-sided boundary artifact independent of grid_distance (Phase 2A(i) finding, confirmed compounding here too — off-grid angle=180° MAE 1.86° vs. 0.47-1.1° at other angles). Excluding it entirely, the slope **survives and slightly strengthens**: Theil-Sen 0.144 [0.111, 0.179]. So the effect is not an artifact of the boundary confound.

**Verdict against the decision-point bar (0.2°, contour-level) — statistically real, but revise "material" downward after checking the binned data.** CI excludes zero (statistically real, not noise), and the linear Theil-Sen fit projected to the farthest sampled point predicts a 0.68-0.72° increase, formally clearing the 0.2° bar. But the linear projection overstates the practical picture; binning `|error|` by `grid_distance` directly tells a more modest story:

| grid_distance | n | MAE | median |
|---|---|---|---|
| [0, 0.5) | 2413 | 0.726° | 0.478° |
| [0.5, 1) | 2463 | 0.781° | 0.517° |
| [1, 1.5) | 177 | 1.780° | 0.953° |
| [1.5, 2) | 128 | 1.986° | 0.999° |
| [2, 3) | 79 | 0.912° | 0.612° |
| [3, 6) | 140 | 1.028° | 0.688° |

Two corrections to the earlier framing: (1) **the absolute magnitude stays in the same ballpark as the model's existing on-grid error** — near-grid MAE (0.73°) is already close to the render-noise band (~0.6°) and validation RMSE (0.65°), and the farthest bucket (~1.0°) is roughly a 1.4x increase over that, not an order-of-magnitude failure; (2) **it isn't even monotonic** — the worst MAE (1.78-1.99°) sits in the *middle* distance range (1-2), and the farthest bucket (3-6) comes back down, inconsistent with clean "further = worse" degradation (one gd=5.0 point, bond=1.5/scaler=3.15, has error of just 0.14°).

**The non-monotonicity is not noise — it's two specific axes.** Breaking down the `grid_distance >= 1` region by which axis is out-of-domain (`ood_tag`) resolves it cleanly:

| OOD axis (grid_distance >= 1) | n | MAE | median | max |
|---|---|---|---|---|
| roughness_neg (reflective, extreme) | 101 | **2.739°** | 2.118° | 10.40° |
| scaler_high (very low apparent resolution) | 108 | **1.944°** | 1.237° | 12.99° |
| in_domain (interior LHS, gd>=1 tail) | 30 | 1.405° | 0.454° | 7.45° |
| bond (extreme Bond number) | 146 | 0.935° | 0.598° | 7.54° |
| roughness_pos (extreme random jitter) | 139 | 0.863° | 0.664° | 4.61° |

The generic quadrature-combined `grid_distance` metric (A.2) treats a unit of distance the same regardless of which axis it comes from, but the axes clearly aren't equally hard: **extreme reflective (negative) roughness and very-high scaler are disproportionately difficult even at modest nominal distance**, while Bond-number and positive-roughness extrapolation stay comparatively mild even further out — which is exactly why the pooled distance bins looked non-monotonic (the farthest bucket, 3-6, is dominated by bond=1.5 points, one of the milder axes). Small-n caveat: these are point estimates on 30-146 points each, not bootstrapped per-axis — worth doing before this goes in the paper. Note also `CLAUDE.md`'s existing best-practice line singles out Bond number ("Bond numbers <=1... should be verified"), but Bond turns out to be one of the *milder* extrapolation axes here relative to roughness/scaler — worth reconciling if this analysis is folded back into that guidance.

**The single 39.5° outlier: examined, and the "bug" conclusion below was wrong — corrected in rev. 13.** At `angle=20, bond=0.62, scaler=0.94, roughness=-0.007`, ground truth is 20°, the model predicted 59.52°. The rendered *image* geometry is correct (height/half-width ratio 0.185 vs. the theoretical 20° value of 0.176 — an initial visual impression that the drop "looked too tall/round" was a mismeasurement, corrected by measuring CP-to-CP width instead of full-contour width, which is dominated by the baseline strip). The 50-point model window's X-coordinates climb 0->2.0, dip back to 1.42, then climb again to 2.9 — initially misread as a malformed/corrupted window. **This is wrong: for a reflective surface with contact angle below ~90°, that fold is the expected, correct shape** (the contour legitimately runs down-and-right along the drop side, then folds down-and-left crossing into the mirrored reflection below the baseline) — not a defect. The sign-flip "isolation" test (negative roughness -> fold + 59.52° prediction; positive roughness of the same magnitude -> clean window + 20.80° prediction) also doesn't isolate a single variable the way it was framed: negative vs. positive roughness aren't the same physical perturbation mirrored, they're two different surface-feature types by design (reflective vs. random jitter), so of course they render and predict differently — that alone doesn't demonstrate either one is broken.

**Neighborhood sweep (rev. 14) — found the real mechanism, refining rev. 13.** Swept scaler, bond, roughness magnitude, and angle individually around the outlier point to check whether it's an isolated fluke or systematic:

- **Bond sweep (0.0-1.0, else fixed): prediction is bit-identical (59.52083...°) at every value.** The extracted contour is literally identical too (same length, same CP, same first 5 points). At dpi=64 and this shallow angle, Bond number's effect on drop shape is below the pixel-quantization threshold here — not a bug, just coarse resolution swallowing a small physical effect.
- **Scaler sweep (0.80-2.00, else fixed): sharply bimodal, not smooth.** 0.85/0.90/0.92/0.94(rounded)/1.00/1.41 all predict well (~0.3-0.7° error); 0.9403283632516403(exact)/0.95/0.96/0.98/1.05/2.00 all fail badly (38-47° error) — including a change only in the *4th decimal place* of scaler (0.9403283632516403 vs. 0.940) flipping a 39.5° error to a 0.4° one. This is a discrete flip, not a gradient.
- **Mechanism, confirmed**: the raw extracted contour is exactly **384 points for every "bad" scaler and 386 for every "good" one** (0.85 through 1.41). Inspecting the extra/missing points shows the "bad" contours have a **long flat run of identical Y-values sitting right through the contact-point index** (18+ consecutive `[x, 0]` points spanning the CP) — the window carries essentially no real curvature information right where it matters most. The "good" contours instead show genuine Y variation through the same region. This is a `cv2.findContours` connectivity instability: at these borderline scalers, sub-pixel rendering differences flip which of two slightly different contour topologies gets traced (a 2-point discrepancy), and one of them happens to land the CP in a degenerate flat segment.
- **This isn't confined to off-grid scalers.** `scaler=2.0` — inside the trained 1-4 range — also fails badly (47.1° error) with the *same* flat-run-through-the-CP signature, even though its total contour length (386) matches the "good" pattern this time. So the generalizable diagnostic is "is the contour locally flat at the CP index," not contour length or distance from the grid.
- **Angle sweep (else fixed): only angle=20 (the exact outlier) fails.** 25-100° all predict well (errors 0.4-2.8°). So this is specific to this shallow-angle/reflective/low-Bond corner, not a broad angle-dependent effect.
- **Roughness-magnitude sweep (negative, else fixed): mostly fine, one more failure.** -0.001 through -0.06 predict reasonably (0.5-1.3° error) except -0.005 (10.1° error) and -0.0072 (the exact outlier, 39.5°) — consistent with the same sharp, non-smooth instability rather than a magnitude-dependent trend.

**Conclusion**: this is a genuine, if narrow, **rendering/contour-extraction instability** — not the general "reflective fold is expected for <90°" point from rev. 13 (which still stands as a separate, correct point about the *shape* of reflective contours), and not a smooth model-generalization gap either. It's a discrete, hard-to-predict connectivity flip in `extract_edges_CV`'s contour tracing that occasionally lands the contact point inside a degenerate flat run, feeding the model a window with no real local curvature — and the model, reasonably, does not handle that gracefully. It appears at both off-grid and (at least once, scaler=2.0) in-domain scalers, so it isn't purely an extrapolation-boundary issue; it may simply be rare enough in the on-grid training sweep's exact scaler values not to have shown up before. Given the narrowness (isolated to this shallow-angle/reflective/low-Bond corner in this spot-check), it's flagged here rather than chased to a code-level fix — but it does mean the single 39.5° point, and possibly others like it, should be treated as **rendering-instability outliers, not generalization-gap evidence**, which nuances (without reversing) the `roughness_neg` axis finding: that axis's elevated MAE (2.74°) likely reflects a mix of genuine reflective-extrapolation difficulty *and* an unknown-sized contribution from this kind of instability. Worth a systematic check (per-point local-flatness diagnostic) before quoting that number precisely in the paper.

**Two follow-up checks (rev. 15) pin down the mechanism further and suggest a concrete fix.**

1. **The "bond has no effect" finding does not generalize, and is not a bug in the Young-Laplace solver.** Repeating the bond sweep with *positive* roughness (jitter branch, same magnitude 0.0072) at the same angle/scaler: contour length varies (370 vs. 372) and predictions vary meaningfully (21.07/21.02/20.86/20.86°) across bond=0.0-1.0 — not identical. Same with negative roughness at a different magnitude (-0.02): predictions vary (20.67/20.59/20.67/20.59°) even where contour length happens to match. Confirmed further by solving the raw ODE (`yl_eqs`/`odeint`) directly at angle=20 across Bo=0.0-1.0 *before* rasterization: `max(X)` genuinely shifts from 0.342020 (Bo=0) to 0.337106 (Bo=1.0), a real ~1.4% analytic shape change, with the normalized aspect ratio shifting too (5.671->5.714). So Bond number **does** affect the analytic solution correctly — there is no bug in the physics/solver — and "Bond has zero effect" at the exact outlier point is a **rasterization/quantization artifact**: a small (~1%) analytic difference falling below one pixel of resolution at dpi=64 for that specific scaler, consistent with the same sharp scaler-sensitivity found above. Not a code bug, and specifically not something that would require regenerating training data or retraining to address — but also, on its own, it doesn't explain the 59.5° misprediction; that's the separate CP-placement mechanism below.

2. **CP placement, traced precisely — refined again in rev. 16 after a valid objection.** The *analytic* (intended, pre-snap) contact point for the outlier is `(296, 2)` in the prepared coordinate frame — 2px above the flat baseline row. No contour pixel exists exactly there, so `prepare_contour`'s nearest-neighbour snap falls back to `(296, 0)`, on the flat run. **Objection raised and correct**: "avoid flat runs" is the wrong generalization — for a reflective surface, a monotonic run that's shallow (near-0deg systems) or steep/near-vertical (near-90deg systems) is legitimate, angle-dependent geometry, not a defect, and a fix that penalizes flatness/verticality generically would misfire on those valid cases. **Refined signature (rev. 16)**: comparing the 15 raw contour points leading into the CP for the bad vs. good scaler, the bad case's X-coordinate is not just flat, it **reverses direction and comes back** (`...293,292,291,290,289,288,287,288,289,...296` — down then back up) while Y is pinned at 0; the good case is strictly monotonic in X with Y still smoothly descending (not yet flat) at the CP. So the actual defect is a **local fold / self-crossing in the traced contour**, not flatness or steepness — those are legitimate monotonic features at their respective angle regimes, a fold is not a legitimate feature of any single-valued contact-angle geometry. A better-targeted check would flag local non-monotonicity in X near the candidate CP, not flat-Y runs. Not implemented (touches the shared `training_files/create_contours_args6.py`/`prepare_contour`, and the deployed `CPID` may or may not share the failure mode — worth checking, but out of scope given resourcing) — added to Phase 5's checklist with this corrected description. Note also: this still doesn't explain *why* the model outputs specifically ~60 deg on this corrupted input rather than some other wrong value — that remains unexplained, and should be described as "the model's response to a genuinely out-of-distribution input" rather than attributed to any specific mechanism.

**Addendum (rev. 18) — plausible explanation for the specific ~60deg value.** The two-point secant angle between the true curvature point (window index 24, `(667,457)`) and the mislocated snapped CP (window index 40, `(669,460)`) is `atan2(3, 2) = 56.31deg` — close to, and suggestively near, the model's actual output (59.52deg vs. ground truth 20deg). This suggests the model isn't outputting an arbitrary/unconstrained value on this corrupted input — it plausibly reads off something like the slope of the spurious secant line from the last real curvature point to the mislocated flat-run CP, rather than the true local tangent at the actual contact point. Not proven mechanistically (the model is a black box; this is a two-point geometric coincidence, not a traced computation), but it's a satisfying, consistent answer to the "why ~60, not some other wrong value" question left open in rev. 16.

**Decisive check (rev. 17): CPID vs. the training-time CP — and this changes the diagnosis substantially.** Everything above (Phase 2A, the whole contour-level sweep, this outlier) uses the *training-time* contact point: `draw_drop`'s analytic CP, nearest-neighbour-snapped onto the extracted contour by `prepare_contour` — because Phase 2A is deliberately built to reproduce training preprocessing exactly (A.7). This is **not** the same as `CPID`, the independent, pixel-based contact-point detector actually used at evaluation/deployment time (`opendrop_ml.modules.preprocessing.preprocessing.isolate_contour` + `CPID`) — training never calls CPID. Running the real CPID on this exact rendered image:

- Training-time (analytic-snap) right CP: `(669, 458)` (unsnapped) -> `(669, 460)` (snapped, inside the flat/folded run)
- **CPID's independently-detected right CP: `(667, 457)`** — which is *pixel-identical* to the point flagged by inspection as "the correct contact point based on the shape of the drop" (window index 24 in the earlier scatter plot).

Rebuilding the 50-point window around CPID's point instead (same window/surface-point sizes, same flip+normalize transform) and feeding it to the model: **prediction = 21.08°** (ground truth 20°) — essentially correct, versus 59.52° from the training-time-anchored window. Confirmed directly, not just plausible: **the training-time CP-snap is measurably less accurate than CPID at this point, and that inaccuracy, not the model, is the proximate cause of the 39.5° outlier.**

This has two important, partly opposing implications:

1. **Real-world deployment risk from this specific failure mode is likely lower than Phase 2A implied.** Deployment always uses CPID, never the analytic-snap. If CPID reliably avoids the corrupted-window scenario the way it did here, then a real (or synthetic-rendered) image landing in this same geometric corner would likely be evaluated correctly in practice, even though Phase 2A's training-preprocessing-faithful sweep shows a large error there. This means the `roughness_neg` axis's elevated contour-level MAE (2.74°) may **overstate** real deployed risk to an unknown degree — Phase 2B/3 (which use the real image pipeline, CPID included) are the ones that can actually answer this, and now carry more evidentiary weight on this specific question than originally assumed.
2. **But it raises a training-data-quality question Phase 2A can't answer.** If `prepare_contour`'s analytic-CP-snap is measurably less accurate than CPID at this corner, and if this corner (or others like it) was ever sampled during the *actual* training sweep (continuous in angle, so not implausible), some fraction of training windows could carry a similar CP-placement inaccuracy — feeding the model quietly-corrupted (if correctly-labelled-by-angle) inputs during training itself. That's a different, harder-to-check concern than "will a user's image be evaluated correctly" — it's "was the model trained on some fraction of contaminated windows" — and this analysis doesn't have the data to resolve it (would require auditing the actual training generation run, not available here).

**Revised primary finding, updated**: the pooled grid-distance slope (0.135°/unit, CI excludes zero) is still statistically real, and the axis breakdown (reflective-roughness and high-scaler as the hardest regimes) still stands as a pattern. But **at least part of what Phase 2A measures as "model difficulty" is actually training-time CP-labelling inaccuracy, not a limitation of the learned mapping itself** — and that fraction likely doesn't survive into real deployment, where CPID is used instead. The clean, defensible framing for the paper is now: **Phase 2A's contour-level results describe the training-preprocessing-faithful mapping, which is a useful but not deployment-representative quantity; Phase 2B/3's image-level results (which use CPID, matching real usage) are the ones that speak to actual deployed accuracy, and should be weighted more heavily for that claim than originally planned.** The Phase 5 checklist item on `prepare_contour`'s CP-snap (rev. 14-16) is superseded by this: the actionable fix is not "improve `prepare_contour`'s snap heuristic" so much as **"consider whether training's CP determination should match CPID's approach**, or at minimum audit how often they'd disagree on the real training grid" — a training-data-quality question, not a training-time-only quirk, and one this analysis flags but cannot resolve without access to a full retraining run.

**Phase 2A(i) — on-grid contour-level sweep + render-noise band.** Ran clean, n=324, 0 failures. MAE=0.674°, RMSE=1.103° against the 0.65° validation-RMSE benchmark. The RMSE gap is entirely explained by a one-sided boundary artifact at angle=180°: those 36 points alone have MAE=2.288° (predictions systematically *undershoot* 180°, e.g. 176.9-179.9°, never overshoot — nothing here clips to [0,180] the way the deployed `experimental_pred` does), while the other 288 points have MAE=0.473°/RMSE=0.696° — matching the benchmark closely. This is consistent with, and confirms, the plan's existing Notes-section caveat about one-sided endpoint error. Render-noise band (contour-level MAE across the 5 Phase-1 replicates): [0.589°, 0.643°], SD=0.020°.

**Phase 2B — image-pipeline smoke test: clean pass, no go/no-go issues.** `spotcheck_pipeline.py` ran `evaluate_image()` (isolate_contour + CPID + all four classical fits + ML) on 17 off-grid images (spanning all 6 matrix groups — in-domain plus bond/scaler/roughness/both compound extrapolation axes — across 5 of the 9 angles, including the boundary-artifact angle 180) and 17 matched on-grid-control images (same angles, cycled across the 36 on-grid parameter combos).

- **0/17 hard failures on either side** (no preprocessing exceptions, no classical-fit exceptions, no ML exceptions). `isolate_contour`/`CPID` did not choke anywhere in the matrix, including the two compound corners (`scaler=0.95, roughness=-0.08` and `bond=1.5, scaler=5.0`) that Phase 1 had already flagged as the riskiest cells.
- **No frame-clipping detected** (CP-near-edge heuristic; see script docstring for its limits as a proxy rather than a full contour-edge check).
- **`profile_length` behaves as expected, not as a red flag.** Off-grid: min 112 / median 482 / max 1076, with 3 images below the 234-point low-res-warning threshold and 2 above the 974-point auto-downsample threshold. On-grid control: min 138 / median 556 / max 1013, 5 below / 3 above the same thresholds — essentially the same shape, not worse for off-grid. This is expected, not a defect: the subset was deliberately drawn to include the high-scaler and low-scaler extrapolation cells (few-pixel and many-pixel drops respectively), so hitting both `prepare_experimental.py` thresholds is the matrix doing its job, not the pipeline misbehaving.
- **ML MAE at this n is uninformative by design and matches that framing**: off-grid 4.48° (median 1.83°, max 21.23°) vs. on-grid-control 4.53° (median 2.32°, max 13.86°) — off-grid is not worse here, but n=17 with the 180° boundary artifact and both extreme-corner cells included is far too small and non-uniform a sample to read as a comparison; it is not the primary endpoint (that's Phase 2A) and Phase 4 is where a properly powered image-level comparison happens.

**Go decision**: no renderer or preprocessing failures anywhere in the sampled matrix, so nothing to fix in the generator and nothing to trim before Phase 3. Full markdown output: `synthetic_reanalysis/results/phase2b_smoke_test.md`.

**Phase 3 — full image-level evaluation: complete, one real bug found in the checkpointer and one real bug found in the deployed pipeline.**

- **Checkpointing bug, found and fixed.** `evaluate_offgrid.py`'s first version appended each image's result as a one-row `DataFrame.to_csv(mode="a", header=False)`. Pandas append-with-no-header writes values *positionally*, not by column name — so when `evaluate_image()` returns early on a preprocessing exception (a short dict: `image_name`, `gt_*`, `error` — 11 keys, vs. 55+ for a full success), that short row's 11 values land under the first 11 header columns regardless of name. Concretely: on the one image that failed (see below), the error text ended up stored under the `preprocessing_time` column, and every other column silently read back as NaN with no signal that the row was misaligned rather than genuinely all-missing. Caught by an explicit ragged-row check (`csv.reader` field-count vs. header) run against all 7 result CSVs after the run — only this one row was affected, everywhere else the per-run column set had stayed constant so positional writes happened to stay aligned. **Fixed** by switching the checkpoint strategy: accumulate all results for a dataset in memory and rewrite the *entire* CSV (temp file + `os.replace` for atomicity) after every image, so pandas builds the column union and pads missing fields with NaN correctly on every checkpoint, instead of relying on positional alignment holding across differently-shaped rows. Re-ran the one affected image after deleting its corrupted row; it reproduced the same failure, now correctly recorded.
- **A real, reproducible bug in the deployed preprocessing pipeline** (not the training pipeline this time — c.f. the Phase-1 `prepare_contour` IndexError, which was in `training_files/create_contours_args6.py`). `160.0_0.5500_5.0000_-0.0150` (angle=160, bond=0.55, **scaler=5.0**, roughness=-0.015) throws `UnboundLocalError: local variable 'left' referenced before assignment` during preprocessing, preceded by the printed warning `Number of contours is not correct for find_CP_static_Lft, contour.shape[0] == 13` — i.e. at this corner the extracted contour is so sparse (13 points; this is the primary high-scaler extrapolation arm, deliberately a few-pixel drop) that `find_CP_static_Lft` (in `opendrop_ml/modules/preprocessing/preprocessing.py`) falls through a branch that never assigns `left` before using it. Reproduced deterministically (failed identically on both the original run and the re-run). This is the deployed `CPID`/`isolate_contour` path, so it's a genuine finding about real evaluation-time behavior at extreme scaler, not a training-preprocessing-only artifact like the Phase-1 bug was. **Not fixed here** (out of scope, same reasoning as the Phase-1 bug: this analysis has full control over which cells it probes and the point is already made) — flagged for the Phase 5 regeneration checklist. Net effect on this run: 1/324 off-grid images hard-failed (0/324 on-grid-control, 0/360 replicates) — dropped from Phase 4's analysis with reason recorded, per the established trim rule (A.1): drop and record, don't substitute.
- **All 7 datasets evaluated clean otherwise**: off-grid 323/324 usable (the one failure above), on-grid-control 324/324, all 5 replicates 72/72 — zero classical-fit exceptions anywhere. Provenance sidecars (`saved_model.pb` MD5, TF/numpy/cv2 versions, seed scheme, git HEAD, timestamp) written for all 7. Total wall clock ~34 min for the 648 new offgrid+ongrid_control images plus ~11 min for the 4 remaining replicates (replicate_0 had already been run as the Phase-3-mechanics validation step).
- **Preliminary, unweighted snapshot** [superseded by the formal Phase 4 result below — kept for the historical run-order record] (not the formal endpoint — Phase 4 does the bootstrap/paired/matched version): pooled per-measurement ML MAE, off-grid 2.453° vs. on-grid-control 2.365° (Δ≈0.09°), both well inside the 5-replicate image-level render-noise band [2.168°, 2.283°]... except off-grid sits just *above* that band's upper end (2.453° > 2.283°), a ~0.17-0.28° gap depending on which control point it's read against — small, and far under the 1.4° image-level practical bar, but worth Phase 4 formally testing with a bootstrap CI rather than eyeballing it here, since this pooled number doesn't yet account for angle-bucket composition or the matched-pair structure the plan specifies.

**Phase 4 — comparative analysis: image-level bounded null, with strong evidence the deployed pipeline's answer differs qualitatively from Phase 2A's contour-level answer, for the reason the plan predicted in advance.** `compare_grid_vs_offgrid.py` pooled the 646 off-grid + 648 on-grid-control per-side measurements (323/324 and 324/324 images respectively, after dropping the one Phase 3 hard failure) and ran the full A.6 program.

- **Primary endpoint — image-level grid-distance slope: BOUNDED NULL.** ML: OLS slope 0.104, Theil-Sen 0.011 (ratio 9.2 -> material disagreement, Theil-Sen is primary per the tie-break rule). Theil-Sen 95% CI (image-level bootstrap, resampling by image per A.6, n=2000): **[-0.054, 0.091] — includes zero.** Predicted increase at the farthest observed grid_distance (5.12) is 0.057°, far under the 1.4° image-level practical bar even before the CI is considered. **Contrast with Phase 2A**: contour-level Theil-Sen was 0.135°/unit with a CI that *excluded* zero (statistically real). At image level the same underlying effect is not detectable — exactly what the plan's Background section predicted ("the image pipeline already costs ~0.75° of the error budget... any off-grid effect at the image level will be partly buried in that noise — which is precisely why the contour-level arm is the decisive test"). This is not a contradiction between phases; it's the two phases doing their designed jobs — Phase 2A isolates the mapping and finds a small real effect, Phase 4 shows that effect doesn't survive contact with the deployed pipeline's own noise floor.
- **Null-slope controls (classical methods) are *larger* than ML's, and several are themselves statistically significant.** Theil-Sen slopes: tangent 0.353 [0.041, 0.657], polynomial 0.613 [0.219, 1.031], circular 0.618 [0.212, 1.140], ellipse 0.281 [0.091, 0.489], yl 0.191 [0.074, 0.319] — all five have CIs excluding zero, vs. ML's CI straddling zero. Per the plan's stated caveat, circular/ellipse degrading with Bond number is expected and uninformative on its own (extrapolation cells are exactly where Bo is highest) — but YL and tangent/polynomial degrading too, on methods with no training grid, means "off-grid cells are simply harder to render/segment" (echoing the plan's framing) is real and roughly as large an effect as anything ML shows. **ML is not the outlier here — if anything it is the most robust method to grid-distance at image level.**
- **Paired nearest-on-grid-control matching**: n=323 pairs (angle- and roughness-sign-matched, minimum per-axis parameter distance). Mean signed bias (off-grid minus matched on-grid) +0.74° overall but **-0.07° for the out-of-domain subset alone** (n=107, mean |Δ| 1.41°) — i.e. off-grid points don't show a consistent directional pull once restricted to the actual extrapolation cells. Correlation(|Δ|, off-grid grid_distance) = **-0.02**, essentially zero — "the cleanest single piece of evidence" the plan called this out to be turns out to show no relationship at image level, consistent with the bounded-null primary result rather than contradicting it.
- **CP-localisation mediation confirms the CPID-noise story.** Correlation(CP pixel error, ML angle error): offgrid r=0.785, on-grid-control r=0.629, pooled r=0.710 — a strong, consistent relationship in both sets. The single worst tail case (44.7° error, `80.0_0.95_2.83_-0.045`, side=left) is an **in-domain interpolation cell** (grid_distance=0.86, not an extrapolation point) with a 21.8px CP error — i.e. the tail is being driven by CPID mislocalisation events that can occur anywhere in the matrix, on-grid or off, not by a grid-distance effect. This directly supports the Phase-5 checklist's existing framing (rev. 17): image-level noise is dominated by CPID, and CPID has no grid-proximity effect of its own since it's an algorithm, not a trained model.
- **Tail metrics**: off-grid ML MAE 2.453° vs. on-grid-control 2.365° (medians nearly identical: 0.859° vs 0.873°, off-grid's is *lower*), but off-grid's max is much higher (44.7° vs 18.7°) — a heavier tail, as the plan anticipated ("grid-proximity effects, if real, are likelier to appear in the tail than the mean"), though the tail-driving case above turns out to be CPID-mediated rather than grid-distance-driven. In-domain vs. out-of-domain (off-grid only): MAE 2.346° vs. 2.668° — a small, expected-direction gap. By `ood_axis` (small-n, unbootstrapped point estimates, n=18-72 each): `compound_bond_scaler` stands out as hardest (MAE 6.35°, n=18), `scaler` next (MAE 3.23°, n=70), with `bond`/`roughness`/`compound_scaler_roughness` all much milder (~1.8-1.9°) — a different axis ranking than Phase 2A's contour-level result (which flagged `roughness_neg` and `scaler_high` as hardest), consistent with image-level noise reshuffling which axis looks hardest rather than preserving the contour-level pattern. Angle-bucket MAE is inconsistent in direction between sets (off-grid higher in `<90`, on-grid-control higher in `90-110`), which is more evidence of noise dominating over a directional grid-distance signal at this sample size.
- **Overall verdict**: the deployment-relevant question Phase 2B/3 were built to answer (per rev. 17's reframing) comes back **negative for a material grid-proximity effect** — the off-grid gap detected at contour level does not survive into image-level, CPID-based evaluation at a magnitude anyone would notice in practice. Combined with rev. 17's finding that CPID also rescues the specific training-time CP-snap failure mode, the overall picture is: **the shape->angle mapping does show a small, real off-grid penalty in isolation (Phase 2A), but the deployed pipeline's own preprocessing noise (dominated by CPID) is larger than that penalty and swamps it**, so users are not expected to see a detectable accuracy drop from evaluating drops with parameters between/beyond the training grid, within the ranges tested here (Bo up to 1.5, scaler 0.95-5.0, roughness to ±0.08).

**Phase 5 — Findings written up; graduation deliberately deferred pending review (2026-08-13).**

> ⚠️ **The verdict below was written before the rev. 24 post-hoc data review and is superseded in part.** The bounded-null result for a *graded* grid-distance effect stands. The claim that users should not expect a measurable accuracy drop does **not** — a reproducible deployed failure at off-grid reflective roughness was found afterwards. Read this paragraph together with "Post-hoc review of the completed analysis (rev. 24)" below, and use the **revised verdict** at the end of that section for any paper text.

**Overall verdict, stated once for the paper** *(superseded — see rev. 24)*: the reported synthetic-evaluation accuracy is **not** primarily a training-grid coincidence. A real, modest grid-proximity effect exists in the pure shape->angle mapping (Phase 2A: Theil-Sen 0.135°/unit, 95% CI [0.101, 0.171], concentrated in the reflective-roughness and high-scaler axes), but it does not survive into image-level, CPID-based evaluation — the regime users actually experience (Phase 4: Theil-Sen 0.011°/unit, 95% CI [-0.054, 0.091], a bounded null, with classical non-trainable methods showing *larger* grid-distance slopes than ML). The deployed pipeline's own CP-localisation noise (CPID + `isolate_contour`) is larger than the mapping-level penalty and swamps it. This is the "gap is negligible" branch of this section (image-level, deployment-relevant result), so the existing `paper_results/` on-grid results and headline 1.40° ML MAE figure **stand** — no regeneration checklist is triggered, only a documented caveat.

**This finding is bounded, not absolute**: it covers Bo up to 1.5, scaler 0.95-5.0, roughness to ±0.08, angles on the existing 9-point grid, and it is specific to the accuracy metric (MAE-style angle error) — it does not clear the two real bugs found along the way (below), which are correctness issues independent of the grid-distance question.

**Two real, unfixed bugs surfaced by this analysis** (both flagged for the checklist above, neither fixed here — out of scope, but should not be lost):
1. `training_files/create_contours_args6.py`'s `prepare_contour` — `IndexError` at low scaler + strongly negative (reflective) roughness (Phase 1, rev. 6), and separately, its CP-snap can land inside a folded/degenerate contour region during training preprocessing (rev. 14-17) — though rev. 17 shows CPID avoids this at evaluation time, so the risk is specifically about *training-data quality*, not deployed accuracy.
2. `opendrop_ml/modules/preprocessing/preprocessing.py`'s `find_CP_static_Lft` — `UnboundLocalError` on `left` when the extracted contour is very sparse (13 points), reproduced at the scaler=5.0 extrapolation corner (Phase 3). This one **is** in the deployed path, so it can affect real high-scaler (low-apparent-resolution) user images.

**Proposed caveat text for `plot_training_coverage.py`'s figure/caption** (drafted here, not yet applied to the script or the paper — pending review):

> *Training-domain coverage shown above (Bond number, profile length vs. angle) reflects the parameter grid used for training data generation. A dedicated off-grid/out-of-domain evaluation (contour-level, n=5400; image-level, n=1294 measurements across 647 images) found a small, statistically real accuracy penalty when evaluating strictly on the shape-to-angle mapping in isolation (~0.01-0.1° per unit of standardised grid-distance, concentrated in reflective-roughness and high-scaler regimes), but this penalty was not detectable at the image level once the deployed contact-point detection pipeline (CPID) is included (bounded at <0.09°/unit, 95% CI). In practice, users evaluating drops with parameters between or beyond the training grid (tested here up to Bond number 1.5, scaler 0.95-5.0, and roughness magnitude 0.08) should not expect a measurable accuracy drop from grid-proximity alone within these ranges.*

---

## Post-hoc review of the completed analysis (rev. 24, 2026-08-13)

An independent pass over the **result CSVs** (rather than the write-up) after Phase 5. Six items; the first changes the headline conclusion.

### 1. A reproducible deployed failure at off-grid reflective roughness — previously misattributed

Phase 4 described its worst tail case as "CPID mislocalisation events that can occur **anywhere** in the matrix, on-grid or off." That is wrong. The failures cluster at one geometric corner and reproduce deterministically. At `angle=80, scaler=2.83, roughness=-0.045`, across all three bond values in the matrix:

| bond | CPID left CP | GT left CP | CP error | ML | YL | GT |
|---|---|---|---|---|---|---|
| 0.15 | (279, **459**) | (273, **439**) | 21.5 px | 120.7° | 104.1° | 80° |
| 0.55 | (279, **460**) | (273, 439) | 21.5 px | 122.7° | 108.0° | 80° |
| 0.95 | (279, **460**) | (273, 439) | 21.5 px | 124.7° | 108.3° | 80° |

CPID descends ~21 px into the reflection. Both sides fail; **every method fails**, YL included — so this is a preprocessing failure feeding all consumers, not a model failure. Bond number barely matters, consistent with the rev. 15/16 finding that bond can fall below pixel resolution at some corners.

**It is roughness-magnitude specific, and the on-grid value is the one that works** (off-grid set, negative-roughness axis, mean per-angle):

| angle | −0.080 | −0.045 | **−0.030 (on-grid)** | −0.015 |
|---|---|---|---|---|
| ML error | 0.67° | **21.72°** | **0.18–2.78°** | 3.18° |
| CP error | 1.8 px | **11.9 px** | **1.6–3.0 px** | 5.7 px |

**Root cause, traced to a specific branch.** Running `CPID(display=True)` on the rendered images and capturing which classification branch fires:

- `roughness=-0.03`, angle 80 (all 12 on-grid images, both scalers): `"large angles under widest points - low angle reflective"` → `preprocessing.py:2371` → `final_CPs = widest_points` → **correct**, CP error 1–3 px.
- `roughness=-0.045` and `-0.015`, angle 80: `"looks reflective but is high angle"` → `preprocessing.py:2403` (`elif 180 - widest1_bottom_angle > 79 ...`) → `final_CPs = CP_guesses` → **lands in the reflection**.
- `roughness=-0.045`, angle 60: does not enter the reflective cascade at all → correct.

So an 80° reflective drop is misclassified as a high-angle system, and the branch selection **flips between the on-grid roughness value and its immediate off-grid neighbours**. The reflective-CP fix recorded as working "for roughness=−0.03 near 90°" holds at exactly that grid value and fails either side of it. The thresholds in this cascade (`error_margin = 18`, `> 100`, `< 8`, `> 99`, `> 79`, `< 65`) are hand-tuned constants; nothing about them is roughness-magnitude aware.

**This is the study's own thesis, appearing in the deployed preprocessing rather than in the model**: the existing 324-image evaluation set cannot see this failure mode, because it samples the single roughness value at which the tuned branch happens to fire correctly.

**It also partly reverses rev. 17.** Rev. 17 concluded from **one** test point that CPID independently finds the correct contact point and therefore that Phase 2A overstated deployed risk. Here CPID demonstrably fails in the same reflective/sub-90° family, on 6 measurements across 3 images, with 21 px errors. Rev. 17's specific observation stands; the generalisation drawn from it — and the resulting decision to down-weight the `roughness_neg` finding — does not.

### 2. The headline verdict needs qualifying in both directions

Excluding the 32 angle-80 reflective measurements from the off-grid set:

| | ML MAE |
|---|---|
| off-grid, all | 2.453° |
| off-grid, excluding angle-80 reflective | **2.069°** |
| the excluded 32 measurements alone | **9.817°** |
| on-grid control | 2.365° |

Off-grid is **better** than the on-grid control everywhere except that one corner. So the correct statement is simultaneously stronger and weaker than what was written: there is no graded accuracy penalty from grid distance (the bounded null is real and well-supported), **and** there is a discrete, reproducible failure corner that only off-grid sampling exposes. The regression missed it because the failing cells sit at modest `grid_distance` (0.74–0.87) and are flagged `in_domain` — so neither the linear fit nor the in/out-of-domain split could surface it. A linear model in a single scalar distance cannot represent "one branch of a threshold cascade misfires in a small region."

### 3. Two-thirds of the render-noise control is deterministic

Only *positive* roughness draws random numbers. The negative (reflective) branch and `roughness=0` contain no `np.random` call, so **216/324 control images are bit-identical to the originals** — verified: predictions match to <1e-9 for `roughness ∈ {0.0, -0.03}`, and differ (max 13.36°) only for `+0.006`. Replicate manifests are 27 stochastic / 45 deterministic per 72 images.

The band `[2.1685, 2.2832]` is therefore diluted by images that cannot vary and understates true render noise. The bias direction makes effects look *more* real, so it does not threaten the null — but it does undercut the 🔶 decision-point reasoning that "0.2° sits at ~10× the render-noise SD," since that SD is computed over a mostly-frozen set. Any future use of the band should compute it on the stochastic subset only.

**Silver lining**: 216 bit-identical images is a perfect code-drift check between the original run and now.

### 4. The paired analysis is uninformative, not supporting evidence

`pair_distance > offgrid_grid_distance` in **323/323 pairs** (median 2.18 vs 0.86). Every matched on-grid neighbour differs from its off-grid partner by more than the off-grid partner differs from the training grid — the pairing noise exceeds the signal by ~2.5×. Root cause: the control has only 36 parameter combinations, so nearest neighbours are necessarily coarse.

So `r = -0.02` means "this design cannot measure the quantity," not "there is no effect." Phase 4 cites it as corroborating the bounded null; it should be reported as a **failed instrument**. Relatedly, the +0.74° overall bias is driven by 3 images — the angle-80 cluster from item 1. Removing them: overall bias +0.345°, in-domain bias 1.144° → 0.555°.

### 5. Two reporting issues

- **Failure rate prints `0/323` for off-grid** — computed *after* dropping the image that failed. The true rate is **1/324 off-grid vs 0/324 control**, and the failure is at the scaler=5.0 extrapolation corner. That is informative missingness and one of the metrics the plan explicitly asked to track; reporting it as 0 hides an off-grid-specific hard failure.
- **"Swamped by noise" is the wrong mechanism.** The image-level Theil-Sen CI is [−0.054, 0.091], whose upper bound is *below* the contour-level point estimate of 0.135. Noise would widen the CI around a similar point estimate; this is **attenuation** — the deployed path removes part of the effect rather than masking it. That is a stronger and better-mechanised claim, and consistent with rev. 17's CPID story for the cases where CPID does succeed.

### 6. The three-way comparison specified in Phase 4 was never run

`compare_grid_vs_offgrid.py` never reads `paper_results/results/synthetic_results.csv`; Phase 4 was two-way (off-grid vs. fresh control). Run in this review:

| set | n | MAE | median | max |
|---|---|---|---|---|
| original (paper) | 648 | 2.399° | 0.904° | 18.662° |
| fresh on-grid control | 648 | 2.365° | 0.873° | 18.662° |

It passes cleanly — rendering variance and code drift are not confounds, which is what this control existed to establish (and see item 3: for 2/3 of the set the agreement is exact by construction, making it an unusually strong code-drift check).

### Revised verdict (supersedes the Phase 5 verdict above)

The reported synthetic-evaluation accuracy is **not** primarily a training-grid coincidence, and there is **no graded accuracy penalty** as a function of distance from the training grid at the deployed image level (Phase 4 bounded null, Theil-Sen 0.011°/unit, 95% CI [−0.054, 0.091]; classical non-trainable methods show *larger* slopes than ML). Away from the failure corner in item 1, off-grid accuracy is slightly *better* than the matched on-grid control (2.069° vs 2.365°).

**However**, off-grid sampling exposed a reproducible, deployment-path failure that the existing on-grid evaluation set is structurally unable to detect: `CPID`'s reflective-surface classification cascade selects the correct branch at the on-grid roughness value (−0.03) and the wrong one at neighbouring off-grid values (−0.015, −0.045) for drops near 80°, producing ~20 px contact-point errors and ~40° angle errors across every fitting method. This is grid coincidence in the **preprocessing**, not the model — and it is the clearest single justification for the off-grid set's existence.

Consequences for the paper: the existing `paper_results/` figures and the 1.40° headline MAE remain valid **for the parameter regime they cover**, but they cannot be described as demonstrating robustness to reflective substrates, and the reflective-surface limitation should be stated explicitly. Phase 6 addresses the underlying bugs.

---

**Not done in this pass, by design** (per user direction 2026-08-13): moving `data/synthetic_offgrid/` / `data/synthetic_ongrid_control/` into `paper_results/data/`, moving `evaluate_offgrid.py` / `compare_grid_vs_offgrid.py` / `contour_level_sweep.py` into `paper_results/evaluate/`, editing `paper_results/config.py` / `run_all.py` / `plot_training_coverage.py`, and deleting `OpenDrop_OP/synthetic_reanalysis/`. All of `synthetic_reanalysis/` (code, data, results) is left in place for review before any of that happens.

---

## Status tracking — completed work
- [x] Phase 1 — matrix + generator, seeded, asserts + control + replicates rendered — **324/324 off-grid, 324/324 control, 360/360 replicates**, zero drops. GT-fidelity check retired (A.5) in favour of the real pipeline's `profile_length` thresholds, assessed in Phase 3/4. Phase 1 complete.
- [x] Phase 2A(i) — contour-level sweep, **on-grid portion only**; on-grid error vs. the 0.65° validation benchmark — matches closely once the angle=180° boundary artifact is accounted for (see Findings).
- [x] 🔶 **Decision point** — practical bar finalised: **0.2°** for the Phase 2A contour-level primary endpoint, **1.4° retained** for Phase 4 image-level comparisons. Reasoning recorded above (Significance criteria section).
- [x] Phase 2A(ii) — off-grid portion (5400 samples); primary endpoint computed — **Theil-Sen slope 0.135°/unit distance, 95% CI [0.101, 0.171], statistically real (excludes zero).** Effect is modest and concentrated in two axes: reflective/negative roughness (MAE 2.74°) and high scaler (MAE 1.94°) at gd>=1, vs. bond (0.94°) and positive roughness (0.86°) which stay comparatively mild — see Findings for the full breakdown.
- [x] **Phase 2A closed.** Primary endpoint, decision point, axis breakdown, and the outlier deep-dive (rev. 9-17) are all done and recorded in Findings. **One open thread carried into Phase 2B/3, not resolved here**: how much of the `roughness_neg` axis's contour-level MAE (2.74°) is genuine model difficulty vs. training-time CP-snap inaccuracy (rev. 17) is unquantified beyond the single outlier — resolving it with more contour-level work is possible but Phase 2B/3 (which use the real CPID-based image pipeline) will answer the *deployment-relevant* version of this question directly and more efficiently, so proceeding to Phase 2B without further contour-level analysis is the recommended next step, not a gap.
- [x] Phase 2B — image-pipeline smoke test passes: 0/17 hard failures on both off-grid and matched on-grid-control subsets (17 each, spanning all 6 matrix groups and 5/9 angles), no frame-clipping, `profile_length` distributions comparable between the two sets. Go — nothing to fix in the generator, nothing to trim. See Findings for detail.
- [x] Phase 3 — full evaluation, checkpointed, provenance-stamped, all result CSVs produced. 323/324 off-grid usable (1 hard failure, a real bug in the deployed `find_CP_static_Lft` at the scaler=5.0 corner — see Findings), 324/324 on-grid-control, 360/360 replicates. Found and fixed a checkpointing bug (positional CSV append silently misaligned columns on short/failed rows) along the way. Preliminary pooled ML MAE: off-grid 2.453° vs. on-grid-control 2.365°, both near the replicate render-noise band [2.168°, 2.283°] — Phase 4 does the formal bootstrap/paired comparison.
- [x] Phase 4 — comparison: grid-distance regression, paired bias, CP mediation, tail metrics. **Image-level primary endpoint is a bounded null** (Theil-Sen 0.011°/unit, 95% CI [-0.054, 0.091], includes zero) — contrasts with Phase 2A's real contour-level effect (0.135°/unit), as predicted by the plan's Background section (image-pipeline noise buries the contour-level effect). Classical-method null slopes are larger than ML's and mostly significant, so ML isn't an outlier. CP-error mediation is strong (r=0.71 pooled) and explains the worst tail case. See Findings for full detail.
- [x] Phase 5 — findings written up + reflection (graduation moved out to Phase 7, rev. 25). ⚠️ The verdict written here is **superseded in part by rev. 24** — the bounded null for a *graded* effect stands, but "users should not expect a measurable accuracy drop" does not. Use the revised verdict at the end of the rev. 24 review section.
- [x] **Post-hoc data review (rev. 24)** — six items found by re-checking the result CSVs rather than the write-up. One changes the headline (a reproducible deployed CPID failure at off-grid reflective roughness, root-caused to a specific branch); the others correct the render-noise band's dilution, the paired analysis's status as evidence, the failure-rate accounting, the "swamped by noise" mechanism, and supply the three-way comparison Phase 4 skipped.
- [ ] Phase 6 — fix the three surfaced bugs (6.1 reflective-branch misfire, 6.2 `iterative_Lfit` sparse-contour crash, 6.3 `prepare_contour`)
- [ ] Phase 7 — graduation: re-run what Phase 6 invalidates, move artefacts into `paper_results/`, update paper outputs, settle the off-grid set's role

**Outstanding work (Phases 6 and 7) has moved to [PART2_REMAINING_WORK.md](PART2_REMAINING_WORK.md).**

## Changelog
- **rev. 26 (2026-08-14)** — Moved the document from `conan_v2/SYNTHETIC_DATA_REANALYSIS.md` into `OpenDrop_OP/synthetic_reanalysis/` alongside the code and data it describes, and split it in two: this Part 1 (completed analysis and findings, Phases 1–5) and `PART2_REMAINING_WORK.md` (Phases 6–7, written to be executable without further design decisions). Added a "How to read this document" preamble warning that the doc is revision-layered and that the rev. 24 revised verdict is the authoritative result — the single biggest trap for anyone reading it cold. Updated the directory tree to match what was actually built.
- **rev. 25 (2026-08-13)** — Restructured the back half of the plan. **Phase 5 is now write-up + reflection only** — no code moves, no paper edits, no fixes — with an explicit instruction not to fix anything during it, so the measurement can't move while the study is being written up. **Graduation moved out to a new Phase 7**, which also absorbs the paper-output regeneration checklist that used to sit under Phase 5, and is explicitly sequenced *after* Phase 6 because 6.1 changes deployed CP placement and would invalidate anything produced before it. Phase 7 adds three things the old checklist lacked: re-running the *existing* on-grid 324 and 171 experimental images against the fixed preprocessing (the paper's current numbers may move), carrying the dataset README/provenance across, and a decision on whether the off-grid set is primary evidence or a supporting check. Also relocated the detailed `iterative_Lfit` mechanism and blast-radius assessment from the old Phase 5 checklist into Phase 6.2 where it belongs, rather than leaving it stranded under a phase that no longer owns it. Phase 5 marked complete.
- **rev. 24 (2026-08-13)** — Post-hoc review of the completed analysis, checking the result CSVs rather than the write-up. **Headline change**: found a reproducible deployed failure at off-grid reflective roughness (angle≈80°, roughness −0.045/−0.015: CPID lands ~21px into the reflection, ~40° error in *every* method including YL), root-caused by live repro to `CPID`'s reflective cascade selecting `preprocessing.py:2371` (correct) at the on-grid roughness −0.03 and `preprocessing.py:2403` (wrong) at neighbouring off-grid values — grid coincidence in the *preprocessing*, which the on-grid evaluation set is structurally unable to detect. This partly reverses rev. 17, whose "CPID avoids this failure mode" conclusion rested on a single test point. Five further corrections: excluding that corner, off-grid MAE (2.069°) is *better* than the on-grid control (2.365°), so the verdict needed qualifying in both directions; 216/324 control images are bit-identical to the originals (only positive roughness is stochastic), so the render-noise band is diluted and the 🔶 decision-point's "10× SD" reasoning is undercut; the paired analysis has `pair_distance > offgrid_grid_distance` in 323/323 pairs and is a failed instrument, not evidence for the null; the off-grid failure rate is 1/324, not the reported 0/323, which was computed after dropping the failure; "swamped by noise" is the wrong mechanism, since the image-level CI upper bound (0.091) lies *below* the contour-level estimate (0.135) — that is attenuation, a stronger claim; and the three-way comparison Phase 4 specified was never run, so it was run here (original 2.399° vs control 2.365° — passes). Added a revised verdict superseding Phase 5's, and **Phase 6** to fix the three surfaced bugs.
- **rev. 23 (2026-08-13)** — Root-caused and scoped the Phase-3 `UnboundLocalError`, in response to follow-up questions (not a new run). Traced it with a live repro to `iterative_Lfit()` (not `find_CP_static_Lfit`, which is dead code — an identical copy-pasted twin only reachable via the never-called `CP_ID_static()`); confirmed the crash is caused by `CPID()`'s *restricted* local contour slice around widest-point/CP-guess indices collapsing to 13 points, not by the full 252-point contour and unrelated to the ML model's own 40-point/50-point window logic. Assessed the blast radius of a fix: empirically the failure never occurs in either existing paper result set (0/324 on-grid synthetic, 0/171 experimental) and occurred exactly once across this study's 1008 images; the buggy branch is a deterministic 100% crash today so a fix can't regress currently-successful predictions; all 3 call sites share the one function; the natural fallback (use already-computed `widest_points` as the final CP) mirrors an existing pattern elsewhere in `CPID()`; and most `CPID()` callers in the video pipelines already degrade gracefully via `try/except`, so a fix mainly recovers currently-skipped frames. Also flagged a separate, unrelated silent-wrong-answer risk noticed while tracing this (unclipped negative slice starts at two of the three `restricted_contour` constructions in `CPID()`). Folded all of this into the existing Phase 5 checklist item rather than leaving it in conversation only. No code changed; user will return to this bug later.
- **rev. 22 (2026-08-13)** — Wrote up Phase 5 findings (write-up only, per explicit user direction -- graduation deferred). Overall verdict: reported synthetic accuracy is not primarily a training-grid coincidence; a real but modest contour-level grid-proximity effect exists but does not survive into image-level/deployed evaluation, so this is the plan's "gap is negligible" branch — existing `paper_results/` figures and the 1.40° MAE figure stand, no regeneration checklist triggered. Drafted (not applied) caveat text for `plot_training_coverage.py`'s caption. Recorded the two real, unfixed bugs surfaced along the way (`prepare_contour`'s IndexError/CP-snap issues, and the newly-found `find_CP_static_Lft` UnboundLocalError in the deployed path) as the concrete residue of this analysis. `synthetic_reanalysis/` left in place, nothing moved or deleted.
- **rev. 21 (2026-08-13)** — Ran Phase 4, the comparative analysis, via a new `compare_grid_vs_offgrid.py`. Primary image-level endpoint (|ML error| vs. grid_distance, Theil-Sen, image-level bootstrap per A.6) is a **bounded null** (95% CI [-0.054, 0.091], includes zero) — the real contour-level effect found in Phase 2A (0.135°/unit) does not survive into image-level, CPID-based evaluation, consistent with the plan's own advance prediction that image-pipeline noise would bury it. All five classical (non-trainable) methods showed larger, mostly-significant grid-distance slopes than ML, so this isn't an ML-specific weakness. CP-localisation error correlates strongly with ML angle error (r=0.71 pooled) and explains the single worst tail case (44.7°, an in-domain point with a 21.8px CP miss) as CPID noise rather than a generalization gap. Paired nearest-on-grid-control matching found no correlation between |Δ| and grid_distance. Phase 4 closed; next is Phase 5 (write-up + graduation decision).
- **rev. 20 (2026-08-13)** — Ran Phase 3, the full image-level evaluation, via a new `evaluate_offgrid.py`. Found and fixed a checkpointing bug: positional (header-less) CSV row appends silently misalign columns whenever a row has fewer keys than the established header (e.g. an early-exit failure dict) -- switched to rewriting the whole accumulated CSV per checkpoint (temp file + atomic replace) so pandas aligns by column name instead of position. Also found a real, reproducible bug in the *deployed* preprocessing pipeline (distinct from the Phase-1 `prepare_contour` bug, which was training-pipeline-only): `find_CP_static_Lft` throws `UnboundLocalError` on a degenerate 13-point contour at the scaler=5.0 high-extrapolation corner (angle=160, bond=0.55, roughness=-0.015) -- flagged for the Phase 5 checklist, not fixed here. Net: 323/324 off-grid, 324/324 on-grid-control, 360/360 replicates evaluated clean; all 7 result CSVs and provenance sidecars written. Preliminary unweighted MAE snapshot (off-grid 2.45° vs. on-grid-control 2.37°, both close to the replicate render-noise band [2.17°, 2.28°]) recorded but explicitly deferred to Phase 4's bootstrap/paired analysis rather than read as a result here. Phase 3 closed; Phase 4 (comparative analysis) is next.
- **rev. 19 (2026-08-13)** — Ran Phase 2B, the image-pipeline smoke test. Built `spotcheck_pipeline.py`, reusing `paper_results/evaluate/evaluate_synthetic.py`'s `evaluate_image`/`parse_synthetic_filename` (worked around a bare-name `config` module collision between this package's `config.py` and `paper_results/config.py` by temporarily evicting `config` from `sys.modules` around that one import, since both files import a same-named local module and Python's module cache doesn't respect per-directory `sys.path` order once a name is already cached). Ran 17 off-grid images (spanning all 6 matrix groups, including both compound extrapolation corners, across 5/9 angles) and 17 matched on-grid-control images through the full pipeline: 0/17 hard failures on either side, no frame-clipping, and `profile_length` distributions comparable between sets (both hit the low-res and auto-downsample thresholds by design, since the subset deliberately includes the extreme-scaler cells). Go decision — proceeding directly to Phase 3 with no generator fixes or matrix trims needed. Phase 2B closed.
- **rev. 18 (2026-08-13)** — Closed the "why ~60deg specifically" question left open in rev. 16: the two-point secant angle between the true curvature point (index 24) and the mislocated snapped CP (index 40) is 56.31deg, close to the model's actual 59.52deg output — plausibly the model is reading the slope of the spurious secant line rather than producing an arbitrary out-of-distribution value. Not a proven mechanism, but a satisfying, consistent closing detail. Phase 2A fully closed; status tracking confirms all items done, one thread (roughness_neg MAE composition) explicitly deferred to Phase 2B/3 rather than left ambiguous.
- **rev. 17 (2026-08-13)** — Decisive check on the outlier: ran the real, deployed `CPID` (not the training-time analytic-CP-snap used everywhere in Phase 2A) on the outlier's rendered image. CPID independently finds `(667, 457)` — pixel-identical to the point visually identified as correct from the window scatter plot — versus the training-time snap's `(669, 460)`, which sits inside the folded/flat region. Rebuilding the 50-point window around CPID's point and re-predicting: **21.08° (correct, GT 20°)**, vs. 59.52° from the training-time-anchored window. Confirms directly that the outlier's cause is training-time CP-labelling inaccuracy, not a model or general rendering-instability limitation. Revises the interpretation of Phase 2A broadly: it measures the training-preprocessing-faithful mapping, which is a real and useful quantity but is not deployment-representative, since deployment always uses CPID. Phase 2B/3 (image-level, CPID-based) should be weighted more heavily than Phase 2A for any claim about real deployed accuracy. Superseded the rev. 14-16 "fix prepare_contour's snap heuristic" checklist item accordingly and reframed the open question as a training-data-quality audit, out of scope without a retraining run.
- **rev. 16 (2026-08-13)** — Two corrections to rev. 15, both from valid objections. (1) Confirmed via direct ODE solve (bypassing rasterization) that Bond number genuinely affects the analytic Young-Laplace shape at angle=20 (~1.4% change in max(X) from Bo=0 to Bo=1) — no bug in the solver; "Bond has zero effect" is a rasterization/quantization artifact (a small analytic difference falling below 1px at dpi=64), not something regeneration/retraining would need to address. (2) Retracted the "avoid flat runs" CP-snap fix framing: shallow and near-vertical monotonic contour runs are both legitimate, angle-dependent geometry for reflective systems, not defects. Comparing the raw points leading into the CP for the bad vs. good case shows the real defect is a **local fold** (X reverses direction, e.g. `...290,289,288,287,288,289,...`) — a topological self-crossing, not a slope. Updated the Phase 5 checklist item to check for local X-non-monotonicity near the candidate CP instead. Also noted explicitly: why the corrupted window specifically outputs ~60deg (rather than some other wrong value) remains unexplained.
- **rev. 15 (2026-08-13)** — Two follow-up checks on the rev. 14 instability. (1) Repeated the bond sweep with positive roughness and with a different negative-roughness magnitude: bond has a real, non-zero effect in both cases, so the earlier "bond has zero effect" observation was specific to the one degenerate point, not general — flagged so it isn't over-generalized elsewhere. (2) Traced the CP-snapping mechanism precisely: the analytic contact point sits 2px above the flat baseline run (where real curvature exists), but no contour pixel exists exactly there, so `prepare_contour`'s nearest-neighbour snap falls back to a curvature-free point on the flat run instead. This is a concrete, well-evidenced candidate fix (prefer the nearest point with genuine local curvature over the literal nearest pixel when the snap distance is large or lands in a flat run) — added to the Phase 5 regeneration checklist rather than left as a vague to-do.
- **rev. 14 (2026-08-13)** — Swept the outlier's neighborhood (scaler, bond, roughness magnitude, angle individually). Found the real mechanism: a `cv2.findContours` connectivity instability that occasionally produces a contour with a long flat run of duplicate Y-values right through the contact-point index (no real curvature info at the window's most important region), flipping sharply between adjacent parameter values (e.g. scaler differing only in the 4th decimal) rather than degrading smoothly — and occurring at both off-grid and at least one in-domain scaler (2.0). This refines, without reversing, rev. 13's correction: the general "reflective fold is expected below ~90°" point still stands, but there's a separate, narrower rendering-instability artifact layered on top of it. Flagged the `roughness_neg` axis MAE (2.74°) as likely a mix of genuine extrapolation difficulty and an unquantified contribution from this instability — needs a systematic per-point flatness check before being quoted precisely.
- **rev. 13 (2026-08-13)** — Corrected rev. 12: the 39.5° outlier is not a rendering/pipeline bug. The window's "non-monotonic, folding" X-coordinates are the expected, correct shape for a reflective surface with contact angle below ~90° (confirmed by the project owner), not corruption — my initial characterization was wrong. The sign-flip test that seemed to "isolate" the issue was also invalid as an ablation: negative vs. positive roughness are different surface-feature types by design (reflective vs. random jitter), not the same perturbation mirrored, so comparing them doesn't demonstrate a bug in either. Retracted the "likely pre-existing bug in create_contours_args6.py" claim and the associated caveat about the roughness_neg axis's MAE possibly reflecting data corruption — that axis's elevated error (2.74°) stands as a genuine, reinforced finding of model difficulty on reflective-surface extrapolation, with this outlier as a concrete example rather than an artifact to discount.
- **rev. 12 (2026-08-13)** — Root-caused the 39.5° outlier flagged in rev. 9/11: the rendered image geometry is correct (initial "looks too round" visual read was a mismeasurement using full-contour width instead of CP-to-CP width); the real bug is a non-monotonic 50-point model window, isolated by sign-flipping roughness alone (negative -> non-monotonic window, 59.5° prediction; positive/zero of the same magnitude -> clean window, ~exact prediction). This is a genuine, pre-existing bug in the reflective-roughness branch of `create_contours_args6.py` (any `roughness < 0`, regardless of magnitude, takes that branch), not something introduced by this analysis, and not yet fixed. Flagged as a likely partial contributor to the roughness_neg axis's elevated MAE — worth checking how many of those sweep points have non-monotonic windows before attributing all of that axis's error to genuine model-generalization difficulty.
- **rev. 11 (2026-08-13)** — Explained the non-monotonic distance-bin pattern from rev. 10: broken down by `ood_tag`, the gd>=1 region shows reflective/negative roughness (MAE 2.74°) and high scaler (1.94°) are disproportionately hard relative to their nominal grid_distance, while Bond number (0.94°) and positive roughness (0.86°) stay mild even further out — the farthest pooled bucket looked better only because it's dominated by (milder) Bond extrapolation points. Noted this should be reconciled with `CLAUDE.md`'s existing Bond-number caution, which turns out to flag the milder of the two problem axes. Flagged small per-axis sample sizes (n=30-146, not yet bootstrapped) as a to-do before this goes in the paper.
- **rev. 10 (2026-08-13)** — Corrected the rev. 9 "material" framing after binning `|error|` by `grid_distance` directly rather than relying on the linear Theil-Sen projection at the farthest sampled point. The slope is real (95% CI excludes zero), but the binned MAE only rises from ~0.73° (near-grid) to ~1.0° (farthest bucket, ~1.4x) and isn't even monotonic (worst MAE is in the middle distance range, not the farthest). Revised the headline claim from "material degradation" to "statistically real but modest, non-monotonic degradation, staying within ~1-2° absolute" — a more defensible characterization.
- **rev. 9 (2026-08-13)** — Ran Phase 2A(ii): off-grid contour-level sweep (5400 samples, 0 hard failures; 18 benign `sampling()` warnings verified as false alarms above the real wraparound threshold). **Primary finding**: pooled with on-grid, `|error|` regressed on `grid_distance` gives Theil-Sen slope 0.135°/unit (95% CI [0.101,0.171], excludes zero), clearing the 0.2° practical bar at the farthest sampled extrapolation points (~0.68-0.72° projected increase). Survives a sensitivity check excluding the angle=180° boundary artifact (slope strengthens slightly to 0.144). This is the headline result of the reanalysis: a real, material grid-proximity effect exists in the pure shape->angle mapping, independent of the image pipeline. Phase 2A complete.
- **rev. 8 (2026-08-13)** — Ran Phase 2A(i): on-grid contour-level sweep (n=324, MAE=0.674°/RMSE=1.103°, matches the 0.65° validation benchmark once the angle=180° boundary artifact — MAE 2.288° at that one angle vs 0.473° elsewhere — is excluded) and the render-noise band via contour-level replicate predictions ([0.589°, 0.643°], SD=0.020°). Hit the 🔶 decision point: finalised the contour-level practical bar at 0.2° (image-level Phase 4 keeps 1.4°) — reasoning and inputs recorded in Significance criteria. Phase 2A(i) and the decision point are both complete; Phase 2A(ii) (off-grid LHS sweep, 5400 samples) is next.
- **rev. 7 (2026-08-13)** — Retired the A.5 dpi=64-vs-dpi=256 GT-fidelity check after implementing and running it: it was confounded by a windowing artifact (the model's fixed-point-count window covers ~4x less physical arc length at dpi=256 than dpi=64, producing meaningless 3-12° mean discrepancies), and more fundamentally, dpi isn't the resolution-mimicking parameter in this generator — `scaler` is, by design, so that roughness stays a fraction of drop size rather than a fixed pixel count. Substituted the deployed pipeline's own resolution-adequacy check (`prepare_experimental.py`'s `profile_length` thresholds: >974 auto-resize, 565-974 conditional, <234 warning) to be applied in Phase 3/4 instead. Phase 1 is now complete.
- **rev. 6 (2026-08-13)** — Root-caused the `prepare_contour` IndexError (scaler ≤ 0.875 at roughness -0.08 puts the whole contour on one side of the apex x-position, confirmed by a scaler sweep) and fixed it at the matrix-design level by moving the low-scaler extrapolation value from 0.875 to 0.95 (Section 3, Section 5, Appendix A.1 updated). Off-grid matrix now renders 324/324 with zero drops.
- **rev. 5 (2026-08-13)** — Phase 1 implemented in `OpenDrop_OP/synthetic_reanalysis/` (config.py, matrix_utils.py, offgrid_matrix.py, contour_level.py, generate_offgrid_synthetic.py, acceptance_gate.py) and run. Acceptance gate passed exactly (744x492, CPs within 0px). Found and fixed a bug in the A.4 frame-geometry assert (rejected 100% of renders, including the acceptance-gate reference, due to the baseline strip's expected edge-to-edge span). Withdrew the 40px resolution floor as a hard exclusion after visual investigation showed it would disproportionately strip the low-angle/high-scaler corner from both datasets — the regime the scaler axis exists to test; replaced with a 2px degenerate-render guard plus `drop_height_px` recorded as a covariate on every row. Found a pre-existing `prepare_contour()` bug at the predicted risky compound corner (cell #12, angles 140-180), affecting 3/324 off-grid cells; documented as an open item, not blocking. See Findings section above for detail.
- **rev. 4 (2026-08-13)** — Marked the 1.4° practical bar as provisional and added an explicit 🔶 decision point for finalising it: after the Phase-1 render-noise band and the Phase-2A **on-grid** sweep, before any off-grid comparison. Split Phase 2A into (i) on-grid and (ii) off-grid halves so the threshold is set from inputs independent of the result, added the decision to status tracking, and flagged the A.5 GT-fidelity threshold as derived from the bar and needing recomputation with it.
- **rev. 3 (2026-08-13)** — Added Appendix A (implementation contract): the 36-cell matrix construction with an explicit trim rule, the grid-distance formula, seeding/filename specs, Phase-1 asserts and the 744×492 acceptance gate, the GT-fidelity and render-noise-band procedures, exact statistical definitions, a Phase-2A reference implementation with its failure modes, and Phase-3 checkpointing/provenance mechanics.
- **rev. 2 (2026-08-13)** — Corrected the Background's claim that training uses analytically exact contours (it renders at dpi=64 and extracts with `extract_edges_CV`; the real deltas are CP source and edge extractor). Added measured baseline numbers and the finding that the experimental set has no ground truth. Held angles fixed for the primary comparison. Split roughness into two regimes and established that it is feature noise, not label noise. Replaced scaler 0.7 with 0.95 (frame limit is 0.75) and made high scaler the primary extrapolation arm. Added the contour-level sweep (Phase 2A) as the decisive test and demoted the image subset to a smoke test. Added pre-specified significance criteria, the grid-distance metric, control replicates, CP mediation analysis, tail metrics, seeding/provenance/checkpointing, and a Phase-5 regeneration checklist. Fixed the misleading `normalize_ds` docstrings in `create_contours_args6.py` and `prepare_experimental.py`.
