# Off-Grid Synthetic Reanalysis — Part 2: Remaining Work

**Date**: 2026-09-13 (rev. 13)
**Developer**: dgshaw
**Status**: 6.0 done (rev. 11) — everything is committed on `fix/cpid-sparse-contour-crash` (5 commits ahead of `origin/develop`, not pushed). 6.4 done. 6.2 done. 6.3 skipped (training_files/ must stay unaltered). 6.1 Step 1 mapped, but a review (rev. 10) found the map cannot yet support the Step 2 decision.
**Next, in order**: **6.1 Session 1** — the Step 1b scripts are written and smoke-tested (rev. 13); run them per the "Session 1 runbook" under 6.1, then read the verdict. Then 6.1 Step 2. See "Working constraints" below.
**Prerequisite reading**: [PART1_ANALYSIS_AND_FINDINGS.md](PART1_ANALYSIS_AND_FINDINGS.md) — at minimum its "How to read this document" preamble and the "Revised verdict" at the end of the rev. 24 post-hoc review.

---

## What this document is

Two phases remain. **Phase 6** fixes three real defects the study surfaced. **Phase 7** folds the work into `paper_results/` and updates the paper's outputs.

Everything below is **prescriptive**: exact file paths, line numbers, reproduction commands, acceptance criteria and stopping conditions. Where a value is stated, use it. Where a check is stated, write it. This is written so it can be executed without re-deriving any of Part 1's reasoning.

**Order is mandatory: Phase 6 before Phase 7.** Fix 6.1 changes deployed contact-point placement on reflective images, so any figure, CSV or paper number produced before it lands would need regenerating.

## Environment

```bash
conda activate opendrop_op
# or use the explicit interpreter, which is what all commands below assume:
PY=/Users/dgshaw/miniconda3/envs/opendrop_op/bin/python
```

All commands run from `OpenDrop_OP/`. The scratch directory for throwaway output is `synthetic_reanalysis/scratch/`.

## Working constraints (rev. 12, 2026-09-13)

Recorded so the scope doesn't drift. This is unpaid, post-PhD portfolio work done in a few concentrated sessions, not an ongoing research effort.

1. **The supervisor's paper draft is written around the existing `paper_results/` figures.** Regenerating figures means rewriting paper sections. The preferred outcome is therefore a **paper-neutral fix**: 6.1's acceptance criterion of zero regressions on the 324 on-grid + 171 experimental images is not just a safety check, it is the goal. If it holds, the existing figures stand and the draft needs no rewrite.
2. **Ground rule 3 below still applies if a fix *does* move a paper number** — adopt and report, don't hide it — but a fix that moves paper numbers is a worse fix, all else equal, and the choice in Step 2 should weight that.
3. **Merely stating the reflective-surface limitation in the paper is not the preferred outcome.** It waters down the result when a real fix likely exists. It is the fallback if (a)/(a′) fail and (b′) is judged too large.
4. **Phase 7 is rescoped** (see its header): 7.1 is a verification that nothing moved, and 7.3's paper-output updates are optional additions, decided after the fix lands.
5. **Budget: two focused sessions** (a few hours each) for 6.1 if (a)/(a′) work. Escalation to (b′) is a deliberate decision taken at the end of session 1, not drifted into — it is realistically 3–4 further sessions because of the video-dataset validation burden.

## Ground rules

1. **Never edit `preprocessing.py` without a regression run.** The existing 324 on-grid synthetic + 171 experimental images are the paper's current results. Section 6.4 defines the regression harness; build it *first*, before any fix.
2. **Record every fix's before/after in this document's changelog**, same convention as Part 1.
3. **If a fix changes a currently-correct result, adopt the improved numbers and regenerate** (decided 2026-08-14). Nothing is published, and figure generation is largely automated, so regenerating is cheap; the tedious part is rewriting paper sections, which is accepted. Still *report* every change — a net-positive change is a change the paper must describe — but do not preserve a known-wrong result to avoid churn.
4. **Do not re-run the Phase 2A contour-level sweep.** It is deliberately faithful to *training* preprocessing and is unaffected by any of these fixes. Only image-level results move.

---

# Phase 6 — Fix the surfaced bugs

Three defects, all pre-existing, none introduced by this study. All were deliberately left unfixed while the study was in flight so the measurement wouldn't move underneath it. That constraint is now lifted.

Ordered by user impact. **6.1 is the only one that silently returns a wrong answer**; 6.2 and 6.3 crash, which is noisy but safe.

| | Defect | Where | Severity |
|---|---|---|---|
| 6.1 | Reflective-surface branch misfires off the roughness grid | `preprocessing.py` ~1825–1949 (was ~2366–2438 before 6.2's dead-code removal) | ⚠️ silent wrong answer, deployed |
| 6.2 | `UnboundLocalError` on sparse contours | `preprocessing.py` `iterative_Lfit()` :1506 | crash, deployed |
| 6.3 | `prepare_contour` IndexError + fold-blind CP snap | `training_files/create_contours_args6.py` | crash + quality, training only |

---

## 6.0 — Repo hygiene: commit the uncommitted work (do this first) 🔴

**Raised in review, rev. 10.** Nothing in this section is analysis — it is protecting work that currently exists only in a working tree. Do it before 6.1.

State at the time of review:

```
branch: fix/cpid-sparse-contour-crash (2 commits, not pushed)
 M opendrop_ml/modules/ML_model/prepare_experimental.py
 M training_files/README.md, create_dataset_parallel.sh, trained_model/README.md
 D training_files/{combine,create_contours_args4,parse_output,train_continue,train_model,zeropad}.py, train.sh
?? training_files/{create_contours_args6,train_optuna,load_optuna,finish_optuna,…}.py, …
?? synthetic_reanalysis/          ← everything: code, 324 images, all result CSVs, both documents
```

### 6.0.1 — `training_files/` correction is uncommitted

This is **PART1's foundational finding** — the discovery that `training_files/` held the wrong (superseded) pipeline, and its replacement with the real one recovered from `files_from_old_computer/_mse_0.4218772351741791/`. The deletions and the recovered files are all uncommitted. If this tree is lost, PART1's premise becomes unreproducible.

**While committing, curate it** (dgshaw, 2026-08-14): several present files were development-time utilities, not part of the pipeline that generated the data or trained the model, and don't need to be in the repo.

Proposed split — **confirm before acting**:

| Keep — the actual pipeline | Purpose |
|---|---|
| `create_contours_args6.py` | data generation |
| `create_dataset_parallel.sh` | the HPC job that ran it |
| `train_optuna.py`, `load_optuna.py`, `finish_optuna.py` | the three training stages; `finish_optuna.py` produced the shipped model |
| `gpu_train.sh` | training job script |
| `create_env.sh` | environment used |
| `hyperparameter_study.db` | the Optuna study from the original run (already tracked) |
| `trained_model/` | training-time diagnostics (already tracked) |
| `README.md` | documents the above |

| Remove — development-time utilities | What it did |
|---|---|
| `check_imports.py` | import smoke-test on the HPC environment |
| `clean_output_file.py` | reformatted raw training stdout logs for plotting |
| `data_check.py` | ad-hoc sanity check of a generated `.pkl` |
| `shuffle_test.py` | verified the `tf.data` interleave/shuffle behaved |
| `examine_db.py` | printed Optuna trials from the `.db` |

**Two judgement calls for dgshaw:**
- `test_evaluation.py` / `test_model.sh` evaluate a trained checkpoint. Arguably part of the workflow rather than a dev utility — keep or remove?
- `examine_db.py` is the only tool for inspecting `hyperparameter_study.db`, which *is* being kept. Removing it leaves the study file with no reader. Mild argument for keeping it despite being a utility.

**Record the curation in `training_files/README.md`.** PART1's provenance argument rests on this directory matching the archival copy at `files_from_old_computer/_mse_0.4218772351741791/training_files/`; after curation it becomes a *deliberate subset*, not a full mirror. Say so explicitly, and note that the archival copy remains the untouched reference — otherwise a future reader will read the missing files as loss rather than choice.

- [x] Curation confirmed with dgshaw and applied (rev. 11) — the four dev utilities dropped; `test_evaluation.py`/`test_model.sh` and `examine_db.py` kept
- [x] `training_files/README.md` updated to describe what was kept, what was removed, and why
- [x] Committed (`28d2e449`), message carrying the provenance reasoning from PART1 rev. 1

### 6.0.2 — `synthetic_reanalysis/` is entirely untracked

All the code, the 324 off-grid images, every result CSV, and both PART1/PART2 documents. **Phase 7.2 plans to delete this directory** once its contents are moved — so it is currently one `rm` away from gone, with no recovery.

- [x] Committed in full (`9ee844f2`) — code, result CSVs, both documents, and `data/`
- [x] `data/` committed directly to git (9.4 MB; `paper_results/data/synthetic/` is already tracked the same way, so there is precedent and no LFS needed)

### 6.0.3 — `prepare_experimental.py` docstring fix is uncommitted

The `normalize_ds` docstring correction from PART1 rev. 2 (window-local, not apex-relative). Small, but it pairs with the identical fix in `create_contours_args6.py` and the two should land together or the cross-reference in each points at nothing.

- [x] Committed (`867ce288`)

---

## 6.1 — `CPID` reflective-surface branch misfires off the grid

### The defect

`opendrop_ml/modules/preprocessing/preprocessing.py`, inside `CPID()`, contains a reflective-vs-high-angle classification cascade at roughly lines 1825–1949 (line numbers as of rev. 13; they were ~2366–2438 before 6.2 removed dead code above it). It is a chain of `if`/`elif` tests against hand-tuned scalar thresholds (`error_margin = 18`, `> 100`, `< 8`, `> 99`, `> 79`, `< 65`), none of which is aware of how strong the reflection is.

For a reflective-baseline drop near 80°, the cascade reaches `preprocessing.py:1866`:

```python
elif 180 - widest1_bottom_angle > 79 or 180 - widest2_bottom_angle > 79:
    # high angle systems that may appear to the code as reflective
    final_CPs = CP_guesses
```

and misclassifies the drop as a high-angle system. `CP_guesses` sit **inside the reflection**, so the contact point lands ~20 px below the true baseline. Every downstream consumer inherits the error — ML *and* all classical fits, YL included.

The correct branch for these drops is `preprocessing.py:1834` (`final_CPs = widest_points`), reached when the cascade prints `"large angles under widest points - low angle reflective"`.

### Why it was believed fixed

The earlier reflective-CP fix was validated at `roughness = -0.03` — the only reflective value in the 324-image evaluation set. **It genuinely works there**: all 12 on-grid images at angle 80 take the correct branch, CP error 1–3 px. The branch boundary simply happens to sit between −0.03 and its neighbours, so no on-grid test could ever expose it.

This is the study's own thesis appearing in the *preprocessing* rather than the model: the existing evaluation grid is structurally unable to see this failure.

### Reproduction

```bash
$PY - <<'EOF'
import matplotlib; matplotlib.use('Agg')
import cv2, io, contextlib, glob
from opendrop_ml.modules.preprocessing.preprocessing import isolate_contour, CPID

cases = [
    ('synthetic_reanalysis/data/synthetic_offgrid/80.0_0.5500_2.8300_-0.0450*', 'FAILS', (273, 439)),
    ('synthetic_reanalysis/data/synthetic_offgrid/80.0_0.5500_2.8300_-0.0150*', 'FAILS', (273, 454)),
    ('synthetic_reanalysis/data/synthetic_offgrid/60.0_0.5500_2.8300_-0.0450*', 'ok',    (273, 439)),
    ('paper_results/data/synthetic/80.0_0.4_1.0_-0.03_*',                       'ok',    ( 93, 447)),
]
for pat, expect, gt in cases:
    path = glob.glob(pat)[0]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, contour, _, _ = isolate_contour(cv2.imread(path), DISPLAY=False, blur=True)
        CPs = CPID(contour, display=True)
    branch = [l for l in buf.getvalue().split('\n') if 'reflect' in l.lower() or 'high angle' in l.lower()]
    print(f'{expect:6s} left_CP={tuple(CPs[0])} GT={gt} branch={branch[:1]}')
EOF
```

Expected output today:

| image | branch taken | CPID left CP | GT | verdict |
|---|---|---|---|---|
| `80.0_..._-0.0450` | `"looks reflective but is high angle"` | (279, **460**) | (273, **439**) | ✗ 21 px |
| `80.0_..._-0.0150` | `"looks reflective but is high angle"` | (274, **460**) | (273, 454) | ✗ 6 px |
| `60.0_..._-0.0450` | cascade not entered | (273, 437) | (273, 439) | ✓ 2 px |
| `80.0_..._-0.03` (on-grid) | `"large angles under widest points…"` | (92, 444) | (93, 447) | ✓ |

Measured impact across the negative-roughness axis at angle 80 (from `results/synthetic_offgrid_results.csv`):

| roughness | −0.080 | −0.045 | **−0.030 (on-grid)** | −0.015 |
|---|---|---|---|---|
| ML error | 0.67° | **21.72°** | **0.18–2.78°** | 3.18° |
| CP error | 1.8 px | **11.9 px** | **1.6–3.0 px** | 5.7 px |

### Step 1 — Map the failure region *before* touching any code

Do not fix from the three known images. Characterise the boundary first.

**Write** `synthetic_reanalysis/phase6_reflective_sweep.py`:

- Render with `draw_drop(angle, bond, scaler, roughness, 5000, dpi=64)` from `training_files/create_contours_args6.py`, seeded per `matrix_utils.cell_seed` exactly as `generate_offgrid_synthetic.py` does — reuse that helper, do not re-implement.
- Sweep: `roughness` −0.005 → −0.080 in steps of 0.005 (16 values) × `angle` 50 → 110 in steps of 5 (13 values) × `scaler` ∈ {1.0, 2.83, 4.0} (3 values) × `bond` = 0.55 fixed. **624 renders.** Bond is fixed because Part 1 established it has near-zero effect in this corner.
- For each: run `isolate_contour` + `CPID` with `display=True`, capturing stdout to identify the branch, then record:

| column | meaning |
|---|---|
| `angle, bond, scaler, roughness` | parameters |
| `gt_cpl_x, gt_cpl_y, gt_cpr_x, gt_cpr_y` | from `draw_drop`'s returned CPs |
| `cpid_cpl_x, …` | CPID's result |
| `cp_error_px` | Euclidean, per side |
| `branch` | the matched print string, or `"none"` |
| `ml_left_angle, ml_right_angle` | full `evaluate_image` prediction |
| `status`, `error` | for crashes |

- Write `synthetic_reanalysis/results/phase6_reflective_sweep.csv`.
- Estimated runtime ~35–40 min based on Phase 3's ~34 min for 648 images. Checkpoint per row using the atomic-rewrite pattern from `evaluate_offgrid.py` (**not** header-less append — see Part 1's A.8 for why).

**Acceptance for step 1**: a printed table of `branch` × `roughness` × `angle` showing where each branch fires and the resulting CP error. You should be able to point at the boundary and say which thresholds move it.

> **🔶 The map may be a paper deliverable in its own right — reconsider once results are in.** If the retune only shifts the boundary rather than eliminating the failure (plausible, given the near-90° reflective discrimination is genuinely hard for static images), then the map itself is the useful output: a characterised statement of where static reflective CP detection fails, with video CP tracking as the known mitigation. That is publishable, honest, and arguably more valuable than a retune that relocates the coincidence to a different roughness value.
>
> **Candidate destination: Supplementary Information.** Decide after Step 2, not now. If it goes in the SI, the sweep should be re-run at presentation quality — finer roughness steps, a heatmap of CP error over (roughness × angle) per scaler, and both branch and CP error shown — so budget for a second, denser run rather than trying to make the diagnostic run double as a figure. Carry the decision into Phase 7.3.

### Step 1a — Corrections to rev. 5's reading of the map (review, rev. 10)

Checked against `results/phase6_reflective_sweep.csv` directly. The sweep itself is sound; three things in its summary need correcting before the fix is chosen from it.

**1. There is a fourth branch, unmentioned in rev. 5.** `"maybe high angle, Lfit - angle…"` — the cascade's `else` path at `preprocessing.py:2416-2429` — fires on **56 cells** and contains **the single worst CP error in the whole sweep (42.9 px)**, with 9 cells over 10 px at angle 75, scalers 2.83 and 4.0. **A retune of `>79` does not touch this path at all.**

**2. Some failures bypass the cascade entirely.** The `none` branch has 4 cells over 10 px (angles 90/95, scalers 1.0/4.0, max 13.0 px).

Of the 99 cells with CP error > 10 px:

| branch | n | retunable via `>79`? |
|---|---|---|
| `looks reflective but is high angle` | 86 | yes |
| `maybe high angle, Lfit` | 9 | **no** |
| `none` (cascade not entered) | 4 | **no** |

So **~13% of the failure region is outside anything a retune of that threshold can reach.** Rev. 5 reads as though the whole failure region is attributable to the one branch.

**3. The scaler characterisation is wrong in a way that matters.** Rev. 5 says the buggy branch fires "for essentially all cells with angle ≥ 85° regardless of scaler." It does not: at scaler 4.0, angles 80 *and* 85 take the **correct** branch 13/16 with CP error 1.9 and 2.6 px. The failure window **shifts** with scaler rather than being one fixed band:

| scaler | failing angle band | mean CP error in band |
|---|---|---|
| 1.0 | 80–90 | 11–20 px |
| 2.83 | 75–90 | 16–20 px |
| 4.0 | 90–95 | 20–21 px |

This is worse news than "scaler 2.83 is the bad one": the discriminant tracks **apparent image scale**, not contact angle, so any single scalar threshold is chasing a moving target. It also means the paper's on-grid case being correct is doubly coincidental — right roughness *and* right scaler.

**4. Data-quality nit.** The `ml_error` column holds error-message *strings*, not numeric ML error (1 non-null: `'index 10 is out of bounds…'`). It collides with the numeric `ml_error` in `phase4_measurements.csv` and silently yields NaN if aggregated. **Rename to `error_message`** in the sweep script. `ml_left_angle`/`ml_right_angle` are populated (617/624), so numeric ML error is recoverable as `|ml_angle − angle|`.

### Step 1b — Instrument the map before choosing a fix (required, rev. 10)

The rev. 5 sweep cannot support Step 2's decision as it stands, for three reasons. All three are fixed by one wider run.

**Gap 1 — the discriminant values aren't recorded.** The CSV stores which branch fired, but not the statistics the branch tested. Without them, every candidate threshold requires a fresh 40-minute sweep. **Record per cell**: `widest1_top_angle`, `widest2_top_angle`, `widest1_bottom_angle`, `widest2_bottom_angle`, plus the `widest_points` and `CP_guesses` coordinates. That converts Step 2 from a guess-and-re-run loop into an offline analysis over a single table.

**Gap 2 — only one class is mapped.** Every cell is reflective (roughness −0.005 … −0.080). The class the `>79` branch exists to *protect* — genuine high-angle, non-reflective drops — is entirely absent, so there is no way to see how much margin a new threshold would leave. The 6.4 harness would catch a gross break, but it is a pass/fail gate 20 minutes wide, not a margin map.

**Gap 3 — the angle range stops at 110°.** Genuine high-angle systems (115–180°) are exactly the cases a more permissive threshold endangers, and none were sampled.

**Extend the sweep to:**

| axis | rev. 5 | Step 1b |
|---|---|---|
| angle | 50–110 step 5 | **50–180 step 5** (27 values) |
| roughness | −0.005…−0.080 step 0.005 | **the same, plus 0.0 and +0.002…+0.010 step 0.002** (21 values) |
| scaler | 1.0, 2.83, 4.0 | unchanged (3) |
| bond | 0.55 | unchanged |

≈1700 cells, roughly 2× the rev. 5 runtime. One run, not one per candidate threshold.

**Analysis to produce** (this *is* Step 2's input):

- [ ] Plot the discriminant (`180 − widest_bottom_angle`) for the two classes — reflective sub-90° vs genuine high-angle — as overlapping distributions, faceted by scaler.
- [ ] State whether **any** single threshold separates them, per scaler and pooled. If not, say so plainly: that is the answer, and it decides between (a) and (b)/(b′) without further iteration.
- [ ] Report the CP-error consequence of the best available threshold, including how many of the 13 non-retunable cells from Step 1a remain.

**Expected outcome, stated in advance** so a null isn't over-read: given that the failure window shifts with scaler (Step 1a item 3) and that the branch is non-monotonic in roughness (rev. 5 point 3), a clean separating threshold is *unlikely* to exist. If it doesn't, the SI-map outcome flagged at Step 1 becomes the primary deliverable and (b)/(b′) become the only paths to an actual fix — with the full multi-dataset regression burden that implies.

### Step 2 — Choose the fix from the map

Three candidates, in increasing order of ambition:

**(a) Retune the `> 79` boundary at `preprocessing.py:2403`.** Cheapest. Moves the boundary; does not remove the class of bug. Only acceptable if the map shows a wide, clean margin between true high-angle drops and reflective sub-90° drops on that statistic.

**(a′) Scale-normalise the threshold (added rev. 12).** Step 1a item 3 shows the failure window *shifts with scaler* — the discriminant tracks apparent image scale, not contact angle — which is exactly why a fixed `79` can't work everywhere. Before abandoning the retune, test replacing the constant with a function of drop size in pixels (e.g. CP-to-CP width, or the `profile_length` already computed in `prepare_experimental.py`). Same one-branch blast radius as (a), same harness, same regression guard. It targets the pattern the map actually showed. Note it still cannot reach the ~13% of failure cells outside the `>79` branch (Step 1a) — shrinking the failure region from a grid coincidence to a small, documented residue is the realistic win.

**(b) Add a symmetry-based discriminator.** A reflection is a vertical mirror of the drop about the baseline; a genuine high-angle drop's lower profile is not. Test the candidate region's symmetry *about the baseline* instead of thresholding an angle: take contour points within ±N rows of the candidate baseline, reflect the lower set about it, and measure agreement with the upper set at matched heights (e.g. mean absolute x-difference, normalised by drop width), **per side**. High agreement ⇒ reflective ⇒ use `widest_points`.

> **Note — this is upper-vs-lower, not left-vs-right.** A tilted/sliding drop with different advancing and receding angles is therefore fine: each side is compared only against its own reflection.
>
> **Weaknesses assessed 2026-08-14:**
> 1. ~~Tilted substrate breaks the mirror axis~~ — **resolved, not a concern.** Tilted systems are tilt-corrected *first* by design, because the contact angle is defined relative to the macroscopic surface tilt. `isolate_contour` runs `tilt_correction` at `preprocessing.py:780`. Horizontal-mirror symmetry is therefore a valid assumption downstream.
> 2. **Real reflections are attenuated, blurred and often partial**, so the traced contour may not follow the reflection fully. Agreement will be far cleaner on synthetic images than on real ones — a bad property for a test whose job is real-world robustness. Validate on the 171 experimental images, not just synthetic.
> 3. **Near 180° a sessile drop approaches a sphere**, which is genuinely near-mirror-symmetric about its own equator. Symmetry does not resolve it.

**(b′) Termination-shape discriminator — worth testing alongside (b).** What terminates the lower extent: a rounded mirrored apex (reflection) or an abrupt horizontal cut where the opaque substrate begins (real drop)? Topological rather than a tuned angle, and unlike (b) it does not degenerate at 180°.

**(c) Restructure the cascade** so reflective detection isn't a chain of `elif` cutoffs at all. Largest change, largest regression risk, out of scope unless the others fail.

### What this branch is actually for — and why a clean fix may not exist

The `"looks reflective but is high angle"` branch exists to separate **contact angles just below 90° on reflective surfaces** from genuine high-angle systems. For a *static* image those two are optically almost identical — this is a genuinely hard discrimination, not a carelessly-tuned threshold. It is hard enough that the **video pipeline solves it by a different route entirely: tracking contact points across frames**, using temporal consistency that a single frame doesn't have.

Two consequences for how this fix is scoped:

1. **The realistic best outcome may be "the boundary moves and the residual limitation is documented," not "fixed."** That is a legitimate result. The current defect isn't that the discrimination is imperfect — it's that the *threshold happens to land correctly at exactly the on-grid roughness value (−0.03) and incorrectly at its neighbours*. Removing that coincidence is the goal; perfect static-image discrimination is not necessarily achievable.
2. **A residual limitation should be stated in the paper as a static-image limitation with a known mitigation** (video CP tracking), which is a much stronger position than leaving it undescribed.

### Preference order (revised 2026-08-14)

**Try (a) first, then (a′).** If a retune of the boundary — fixed or scale-normalised — fixes the failure region without regressions, take it: it is the minimal change to a hard, well-understood discrimination, and it is the only path that keeps the paper draft untouched.

### Session plan (rev. 12)

**Session 1 — Step 1b and the separability read.** Scripts done and smoke-tested ahead of time (rev. 13).

*Session 1 runbook* (from `OpenDrop_OP/synthetic_reanalysis/`, `opendrop_op` env):

```bash
python phase6_reflective_sweep.py          # 1782 cells, ~100 min, resumable; writes results/phase6_reflective_sweep_1b.csv
python phase6_separability.py              # offline analysis -> stdout verdict, figures/phase6_separability_*.png, results/phase6_separability_summary.csv
```

- The sweep is the rev. 5 script extended per Step 1b: full angle range 50–180, roughness −0.080…−0.005 plus 0 and +0.002…+0.010 (22 values, one more than Step 1b's "21"), the four discriminants computed from the contour by mirroring `CPID`'s own code *and* cross-checked against what `CPID` prints (`disc_parse_ok`), `widest_points`/`CP_guesses` coordinates and each one's error vs GT, size covariates for (a′), a complete branch classifier, `ml_error` → `error_message`. It writes a **new** file so the rev. 5 map stays intact. `--smoke` runs the three repro cells plus one genuine 150° drop in ~1 min — use it if anything looks off; **do not read its verdict as a result** (any feature "separates" two points).
- `phase6_separability.py` labels each decision cell by which of `widest_points`/`CP_guesses` was nearer GT (3 px margin), then for the current statistic, its width-normalised form (a′), and seven other quantities `CPID` already has in hand, finds the single threshold with fewest misassignments, per scaler and pooled. Every candidate is deployable — nothing uses `gt_*`. The first thing it reports is how many decision cells have `disc_max` **saturated at exactly 90** (vertical fit under the widest point).
- **Watch for this first.** In the smoke run, `disc_max` was exactly 90.00 for both failing reflective cells *and* the genuine 150° drop. Four cells prove nothing, but if the full sweep shows the same, then `180 − widest_bottom_angle` is uninformative in this regime and **neither (a) nor (a′) can work on that statistic** — the fix becomes a one-branch swap to whichever other candidate the script finds separable, which is the same blast radius as (a) and still under the 6.4 harness.
- Caveat the script states: cells below the `>79` threshold go to `iterative_Lfit`, not `widest_points`, so the `widest_points` error is a proxy for that side. Session 2's re-run uses the real path.
- **Go/no-go at end of session**: the script prints it. Record the verdict, the winning feature and threshold (or the absence of one), and the saturation count in the changelog.

**Session 2 (if go) — implement (a) or (a′) and guard it.**
- Change the one branch at `preprocessing.py:2403`.
- Run the 6.4 harness before and after; require zero diffs on 324 + 171.
- Re-run the sweep; confirm the failure region is gone or materially shrunk.
- Confirm the three reproduction images give CP error ≤ 3 px.
- Commit. Then 7.1 verification (re-run paper results, confirm unchanged) can be a short follow-up.

**If no-go**: stop and decide deliberately between (b′) with its full validation burden, or documenting the limitation with the map as an SI figure. Do not start (b′) in the same session.

**(b)/(b′) only if (a) cannot.** These are global changes to the algorithm's logic, and per dgshaw's constraint they require verification against **all system types**, not just the reflective corner:

- 324 on-grid synthetic + 171 experimental (the 6.4 harness)
- **plus** the video datasets: `paper_results/data/video_analysis/{inverse_phase,isaac,kristina}` — these exercise hydrophobic, low-angle, and reflective systems the static sets don't cover
- plus a full re-run and regression check of every result in `paper_results/results/`

Budget for that accordingly: it is a substantially larger piece of work than (a), and the escalation decision should be made deliberately, not drifted into.

**(c) is off the table** unless (a), (b) and (b′) all fail.

**No pre-commitment beyond that: let the map decide** between (a) and the rest. An earlier draft preferred (b); that preference was withdrawn. Record the decision and its justification in the changelog.

### Step 3 — Implement and regression-guard

- Implement behind the regression harness of 6.4. **Run the harness before and after.**
- Hard requirement: **zero regressions** on the 324 on-grid synthetic and 171 experimental images. They currently take the *correct* branch, so a careless retune breaks results the paper already reports.
- Re-run `phase6_reflective_sweep.py` and confirm the failure region is gone or materially shrunk.
- Re-run the three reproduction images above and confirm CP error drops to the 1–3 px range.

### Acceptance criteria for 6.1

- [x] `results/phase6_reflective_sweep.csv` exists and the failure region is characterised (rev. 5)
- [x] Map reading corrected — fourth branch, non-cascade failures, scaler-shifting failure window (rev. 10, Step 1a)
- [ ] **Step 1b**: sweep extended to both classes and full angle range, discriminant values recorded, `ml_error` column renamed
- [ ] **Step 1b analysis**: separability of the two classes stated per scaler — the direct input to the choice below
- [ ] Fix chosen with recorded justification
- [ ] Regression harness: 0 changes on 324 on-grid + 171 experimental (or every change individually reviewed and accepted in writing)
- [ ] Post-fix sweep shows the failure region gone or materially reduced
- [ ] The three reproduction images give CP error ≤ 3 px

---

## 6.2 — `iterative_Lfit` `UnboundLocalError` on sparse contours

### The defect

`opendrop_ml/modules/preprocessing/preprocessing.py`, `iterative_Lfit()` (line 1506).

**Not `find_CP_static_Lfit` (line 1246)** — that is a near-identical copy-pasted twin, including the same stale `"find_CP_static_Lft"` print text, but it is dead code reachable only via `CP_ID_static()`, which nothing in the codebase calls. This duplication already caused one misdiagnosis during the study; see 6.2 step 3.

**Mechanism** (traced with a live repro, not inferred): `CPID()` restricts the full contour to a small local slice around its widest-point/CP-guess indices (`contour[idx1 - 4*n_pts : idx2 + 4*n_pts]`) before refining. `iterative_Lfit()`'s branch logic handles only `len(contour) >= 110` (full apex-split) or `== 2` (pre-split `[left, right]` pair). Anything in `[3, 109]` falls into an `else` that prints a warning and **never assigns `left`/`right`**, then references them at **`preprocessing.py:1560`** (`contours = [left, right]`) — that is the line the traceback points at.

At the reproducing image the full contour is 252 points but the restricted slice collapses to 13. This is upstream of, and unrelated to, the ML model's own 40-point-minimum / 50-point-window logic — CPID never gets far enough to identify contact points at all.

### Reproduction

```bash
$PY -c "
import matplotlib; matplotlib.use('Agg')
import cv2, glob
from opendrop_ml.modules.preprocessing.preprocessing import isolate_contour, CPID
p = glob.glob('synthetic_reanalysis/data/synthetic_offgrid/160.0_0.5500_5.0000_-0.0150*')[0]
_, c, _, _ = isolate_contour(cv2.imread(p), DISPLAY=False, blur=True)
print('contour length:', len(c))
CPID(c, display=False)   # UnboundLocalError: local variable 'left' referenced before assignment
"
```

Deterministic — failed identically on two independent runs during Phase 3.

### Why a fix is low-risk (assessed in Part 1, rev. 23)

1. The branch is a **100% deterministic crash today**, so nothing currently-successful can regress — a fix can only convert crashes into results.
2. Empirically rare: **zero** occurrences across the 324-image on-grid synthetic set and the 171-image experimental set; exactly **once** across this study's 1008 evaluated images, at a deliberately-extreme scaler=5.0 cell.
3. All 3 call sites route through the same shared `iterative_Lfit()`, so one fix covers all of them.
4. The natural fallback — use the already-computed `widest_points`/`CP_guesses` when the local window is too small to refine — reuses a pattern (`final_CPs = widest_points`) already present in several of `CPID()`'s other branches. Not a new heuristic.
5. Most `CPID()` call sites in the video CA pipelines already wrap it in `try/except`, degrading to a per-frame "cpid_failed" skip, so a fix there **recovers** frames rather than changing already-processed ones — **except `video_two_pass.py:436`, which is not directly wrapped and must be checked before shipping.**

### Work

**Step 1 — fix the crash.** Assign `left`/`right` on the sparse-contour path via the `widest_points` fallback. Add a guard clause with a clear message naming the contour length, rather than falling through an `else`.

**Step 2 — flag reduced confidence.** A recovered result on a genuinely sparse contour is inherently lower-precision. Emit a low-confidence signal mirroring the existing `profile_length < 234` "predictions may be unreliable" warning in `prepare_experimental.py`, rather than presenting it at the same confidence as a well-resolved drop.

**Step 3 — fix the adjacent silent-wrong-answer risk.** Two of the three `restricted_contour` constructions in `CPID()` (~line 2424 and ~2464) don't clip their start index (`CP_guess_index1 - 4*n_pts`, no `max(0, ...)`), unlike the first one which does. A negative start **silently wraps from the end of the array** instead of clipping — distinct from the crash, and potentially affecting normal (non-sparse) images whose CP guesses sit near the start of the contour array. Same failure family as the `sampling()` negative-slice issue found in Phase 2A. Clamp all three.

**Step 4 — resolve the dead twin.** Decide whether to delete `find_CP_static_Lfit()` and `CP_ID_static()`. Confirm nothing calls them first:

```bash
grep -rn "CP_ID_static\|find_CP_static_Lfit" --include=*.py OpenDrop_OP/
```

Dead code duplicating a buggy branch is a trap for whoever fixes this next — it already cost one misdiagnosis here.

**Step 5 — regression test.** Add a test at the reproducing image asserting CPID returns two plausible contact points rather than raising.

### Acceptance criteria for 6.2

- [x] Reproduction command returns contact points instead of raising (rev. 6)
- [x] Low-confidence signal emitted on the sparse path (rev. 6)
- [x] All three `restricted_contour` slice starts clamped (rev. 6)
- [x] `video_two_pass.py:436` checked for unwrapped exposure (rev. 6: confirmed wrapped, no fix needed)
- [x] Dead twin deleted, or an explicit decision recorded not to (rev. 6: deleted)
- [x] Regression test added; regression harness clean (rev. 6)

---

## 6.3 — `prepare_contour` crashes and mis-snaps — SKIPPED (rev. 8)

**Skipped per dgshaw's direction (rev. 8).** `OpenDrop_OP/training_files/` is the record of the approach actually used to train the shipped model and must stay reproducible/unaltered — it is not a general-purpose dev copy to improve. Defects 1 and 2 below were implemented and then fully reverted in rev. 7/8 once this was clarified; see the rev. 8 changelog entry for the reasoning and the revert diff. If either defect needs fixing for a *future* retraining effort, do it in a separate copy (e.g. under a scratch/experimental directory), never in `training_files/` directly. Defect 3 (the audit) was already left open for unrelated reasons (out of scope, retraining not planned).

`training_files/create_contours_args6.py`, `prepare_contour()`. **Training pipeline only** — no effect on current users. Its value is preventing the next dataset regeneration from inheriting the same defects, plus the training-data audit it unblocks.

### Defect 1 — `IndexError` on degenerate geometry

`IndexError: too many indices for array` from `l_drop[:,0]` when the extracted contour lands entirely on one side of the computed apex x-position, leaving `l_drop` or `r_drop` empty. Occurs at scaler ≤ 0.875 with strongly negative roughness; clean at scaler ≥ 0.9 at every angle tested. Currently worked around at the matrix level by using scaler 0.95.

Reproduce with `create_contour(angle=160, Bo=0.55, scaler=0.875, roughness=-0.08)`.

**Fix**: guard the empty-half case with an explicit error naming the parameters, rather than an `IndexError` from a downstream slice.

### Defect 2 — CP snap lands in a folded contour region

The analytic CP sits ~2 px above the baseline. When no contour pixel exists exactly there, the nearest-neighbour snap can fall onto a point inside a local fold where the traced X-coordinate **reverses direction** (`...290, 289, 288, 287, 288, 289...`) — a topological self-crossing, not legitimate geometry.

**The correct check is local non-monotonicity in X near the candidate CP** — *not* flatness and *not* steepness. Both of those are valid angle-dependent features for reflective systems (shallow runs near 0°, near-vertical runs near 90°), and an earlier revision's "avoid flat runs" framing was retracted for exactly that reason. Do not reintroduce it.

**Fix**: make the CP snap reject candidates inside a local X-fold, preferring the nearest candidate that is locally monotonic.

### Defect 3 (analysis, not a code fix) — the training-data question

**First, what the training CP is and isn't** — searched exhaustively 2026-08-14, confirming dgshaw's recollection that the stored CPs are unused:

**The CPs embedded in the stored data-point keys are never read by anything.** The only two places in the codebase that touch key fields 4/5 are `create_contours_args6.py:513` and `:560`, both inside `sampling()`/`sampling0()` — the generator. Every consumer of the training pkls reads field `[0]`, the angle, and nothing else:

| consumer | parses |
|---|---|
| `finish_optuna.py:45` | `float(key.split('_')[0])` — angle |
| `train_optuna.py:45` | angle |
| `load_optuna.py:63-64` | angle |
| `test_evaluation.py:44` | angle |
| `shuffle_test.py:26` | angle |

Nothing in `opendrop_ml/` uses that key format at all. So the CPs are vestigial in the stored data — consistent with the recollection that including them was an attempt to help tune CPID that was subsequently discarded.

**The one thing that survives**: the CP *value* decided where `sampling()` cut the 50-point window at generation time (`contour[index - 40 : index + 10]`, anchored on the CP). Its consequences are baked into the stored windows even though the coordinate itself is never re-read. That, and only that, is what the audit below can speak to.

**Bound**: training augments each sample with a ±5-point window shift (`window_shift_start=-5, window_shift_end=5`), so the model was explicitly trained to tolerate small CP errors. Only misplacements larger than that — like the fold case in rev. 17 — fall outside what augmentation already covers.

> **Descope candidate.** This audit cannot fix anything: a mis-cut window is baked into the shipped model and is only correctable by regenerating and retraining. Its entire value is characterising a limitation for the paper. It is also the most compute-heavy item in Phase 6. **If time or compute is short, drop it** — the two code fixes above are quick and independent of it.

Part 1's rev. 17 established that `prepare_contour`'s analytic snap was measurably *less* accurate than `CPID` at one tested point. That opens a question the study could not answer: **how often would the two disagree across the actual training grid?**

This does **not** require retraining. Sample the real training sweep (angle 20–180 step 0.05, Bo `linspace(0,1,11)`, scaler `geomspace(1,4,7)`, the 12 roughness values — see Part 1's Background), render each sampled point, and compare `prepare_contour`'s snapped CP against `CPID`'s independent detection. A few thousand samples is enough to size it.

Report disagreement in **contour points**, not just pixels, and split at the ±5 augmentation threshold:

- Disagreement ≤ 5 points → absorbed by the training-time window-shift augmentation; not a concern.
- Disagreement > 5 points → outside what augmentation covers; some fraction of training windows showed a different part of the profile than their angle label described.
- If the >5 fraction is negligible → a footnote.
- If it is material → it belongs in the paper as a limitation, and it partially explains any residual model error that this study attributed to difficulty.

Write results to `results/phase6_training_cp_audit.csv`.

### Acceptance criteria for 6.3 — N/A, item skipped (rev. 8)

- [ ] Empty-half case raises a clear, parameterised error — **skipped (rev. 8): would alter `training_files/`, the fixed record of the trained model's approach**
- [ ] CP snap rejects local X-folds (monotonicity test, not flatness) — **skipped (rev. 8), same reason**
- [ ] Training-grid CP-disagreement audit run and sized; result recorded in the changelog — descoped, unrelated to the above (see rev. 7)
- [x] Confirm no change to `synthetic_reanalysis/` outputs — trivially true, nothing in `training_files/` was ultimately changed (rev. 8)

---

## 6.4 — The regression harness (build this first)

Required by 6.1 and 6.2. Build before either.

**Write** `synthetic_reanalysis/phase6_regression.py`:

```
For each of:
  - paper_results/data/synthetic/      (324 images, GT from filename)
  - paper_results/data/experimental/   (171 images, no GT — compare to the stored run)
run evaluate_image() and compare against the stored baselines:
  - paper_results/results/synthetic_results.csv
  - paper_results/results/experimental_results.csv
```

**Comparison rules:**

- Compare per-image, per-side, for **every** method column (`ml_*`, `yl_*`, `tangent_*`, `polynomial_*`, `circular_*`, `ellipse_*`) plus `left_CP_*` / `right_CP_*`.
- Tolerance: contact points must match **exactly** (integers). Angles: flag any change > 0.01°.
- Report as three buckets: **unchanged**, **changed-and-better** (only meaningful for synthetic, which has GT), **changed-and-worse**.
- Exit non-zero if any image regresses.

**Important**: 216 of the 324 on-grid images are *deterministic* renders (only positive roughness draws random numbers — see Part 1's rev. 24 item 3), so those must reproduce bit-identically. That makes this an unusually sharp instrument: any change there is a real code effect, not render noise.

Baseline capture before starting:

```bash
$PY synthetic_reanalysis/phase6_regression.py --capture-baseline
```

### Acceptance criteria for 6.4

- [x] Harness runs clean against unmodified code (0 changes — proves the harness itself is sound) (rev. 5)
- [x] Runtime documented so it can be run repeatedly without surprise (expect ~25–30 min for 495 images) (rev. 5: 19.9 min measured, faster than estimated)

---

## Phase 6 status

- [x] **6.0 — repo hygiene: `training_files/` (curated), `synthetic_reanalysis/`, and the `prepare_experimental.py` fix all committed on `fix/cpid-sparse-contour-crash` (rev. 11)**
- [x] 6.4 — regression harness built and verified clean against unmodified code (rev. 5)
- [ ] 6.1 — sweep mapped (rev. 5) and its reading corrected (rev. 10); **Step 1b instrumentation outstanding**, then fix chosen and justified, regression-guarded, post-fix sweep clean
- [x] 6.2 — sparse-contour path fixed, low-confidence signal added, all three slice starts clamped, `video_two_pass.py:436` checked, dead twin (and 4 further orphans found by an extended sweep) resolved, regression test added, committed on `fix/cpid-sparse-contour-crash` (rev. 6, rev. 9)
- [x] 6.3 — SKIPPED (rev. 8): `training_files/` must stay unaltered as the reproducible training record; see the 6.3 section header and rev. 7/8 changelog entries
- [ ] 6.3 audit — training-grid CP-disagreement audit run and sized **(descopable — characterises a limitation, fixes nothing)**

---

# Phase 7 — Graduation

Fold the surviving work into `paper_results/` and update the paper's outputs. **Runs after Phase 6.**

> **Rescoped rev. 12.** Under the working constraints, 7.1 is a **verification** that the paper's existing numbers are unchanged by the fix — the expected outcome, not a regeneration. 7.2 (moving artefacts) is still worth doing for a tidy public repo. 7.3 (paper-output updates) and 7.4 (framing) are **optional**: they only apply if the off-grid set is added to the paper as a supplementary robustness section, which is a separate decision to take with the supervisor after the fix lands. Nothing in 7.3/7.4 blocks merging the fix.

## 7.1 — Re-run what Phase 6 invalidates

- [ ] Re-run Phase 3 evaluation on the off-grid set and the on-grid control with the fixed preprocessing:
      `$PY synthetic_reanalysis/evaluate_offgrid.py`
- [ ] Re-run Phase 4's comparison: `$PY synthetic_reanalysis/compare_grid_vs_offgrid.py`
- [ ] Check whether the bounded-null primary endpoint survives. It **should** — 6.1 removes a discrete failure, it doesn't change the graded-effect question — but confirm rather than assume, and expect the off-grid MAE to *improve* toward the 2.069° figure Part 1 measured with the reflective corner excluded.
- [ ] Re-run the **existing** on-grid 324 and the 171 experimental images and diff against `paper_results/results/`. 6.1 changes deployed preprocessing, so the paper's current numbers may move even though those images take the correct branch today. This is the 6.4 harness; reuse it. **Per the ground rules, adopt the new numbers** — overwrite `paper_results/results/synthetic_results.csv` and `experimental_results.csv`, and record the before/after deltas in the changelog so the paper rewrite has a reference.
- [ ] **Do not re-run the Phase 2A contour-level sweep** — it is faithful to training preprocessing and is unaffected.

## 7.2 — Move the artefacts

- [ ] `synthetic_reanalysis/data/synthetic_offgrid/` → `paper_results/data/`
- [ ] `synthetic_reanalysis/data/synthetic_ongrid_control/` → `paper_results/data/`, **if kept** — decide whether the control has standalone value or was purely diagnostic
- [ ] `evaluate_offgrid.py`, `compare_grid_vs_offgrid.py`, `contour_level_sweep.py` → `paper_results/evaluate/`
- [ ] Merge path constants into `paper_results/config.py` (`SYNTHETIC_OFFGRID_DATA_DIR`, `SYNTHETIC_ONGRID_CONTROL_DIR`)
- [ ] Register the new steps in `paper_results/run_all.py`
- [ ] Carry the dataset README and provenance sidecars across — the off-grid set was built to paper quality specifically so this is a move, not a rewrite
- [ ] Move both PART1/PART2 documents somewhere durable (suggest `paper_results/` alongside the code they now describe) rather than deleting them with the directory
- [ ] Watch for the `config` module name collision documented in Part 1's rev. 19: both `synthetic_reanalysis/config.py` and `paper_results/config.py` are imported as bare `config`, and Python's module cache doesn't respect per-directory `sys.path` order once a name is cached. Moving the scripts into `paper_results/` should *remove* this problem; verify rather than assume.
- [ ] Delete `OpenDrop_OP/synthetic_reanalysis/`

## 7.3 — Update paper outputs

- [ ] `plot_training_coverage.py` figure + caption — apply the caveat text drafted in Part 1's Phase 5, **revised for the rev. 24 verdict**. The drafted version predates the reflective finding and understates it; it needs rewriting, not pasting.
- [ ] `plot_synthetic.py` figures — add or reframe for the off-grid set
- [ ] `plot_cp_comparison.py` — CP-error findings from Phase 4, plus the reflective-branch result from 6.1
- [ ] Any paper text quoting the 1.40° on-grid ML MAE or the method-ranking table
- [ ] **State the reflective-substrate limitation explicitly.** After 6.1 it is a fixed bug rather than a live limitation, but the paper should still describe what the on-grid set could not have detected — that is the clearest single argument for the off-grid set's existence.
- [ ] `CLAUDE.md`'s "Bond numbers ≤ 1" best-practice line — the Bo 1.2/1.5 cells test it directly, and Bond turned out to be one of the *milder* extrapolation axes relative to reflective roughness and high scaler. Reconcile the guidance with what was measured.
- [ ] Bootstrap the per-axis MAE estimates before quoting them. Part 1 flags these as unbootstrapped point estimates on n=18–146; they are currently not paper-ready.
- [ ] **🔶 Decide whether the 6.1 reflective map goes in the Supplementary Information** (carried from 6.1 Step 1). If yes: re-run the sweep at presentation quality — finer roughness steps, a per-scaler heatmap of CP error over (roughness × angle), branch and CP error both shown — rather than reusing the diagnostic run. The case for including it is strongest if the retune shifts the failure boundary without eliminating it, since the map then documents a real limitation with a known mitigation (video CP tracking).

## 7.4 — Decide the off-grid set's role in the paper

- [ ] Primary generalization evidence, or a supporting robustness check alongside the existing on-grid set?

The rev. 24 verdict pulls both ways: the latter for **accuracy** claims (no graded penalty was found), the former for **coverage** claims (the on-grid set is structurally unable to see the reflective failure). Pick one framing and apply it consistently throughout the paper.

## Phase 7 status

- [ ] 7.1 — re-runs complete, bounded null re-confirmed, paper-number diff reviewed
- [ ] 7.2 — artefacts moved, config/run_all updated, `synthetic_reanalysis/` deleted
- [ ] 7.3 — figures, captions and paper text updated; per-axis estimates bootstrapped
- [ ] 7.4 — framing decided and applied consistently

---

## Known traps

Collected from Part 1 so they aren't rediscovered the hard way.

1. **Header-less CSV append silently misaligns columns.** `df.to_csv(mode="a", header=False)` writes positionally, so a short row (e.g. an early-exit failure dict with 11 keys vs 55+ for a success) shifts every value into the wrong column with no error. Use the accumulate-and-atomically-rewrite pattern already in `evaluate_offgrid.py`.
2. **Negative slice starts wrap, they don't clip.** Bitten twice: `sampling()` in Phase 2A and `CPID()`'s `restricted_contour` in 6.2. Always `max(0, ...)`.
3. **`prepare_contour` mutates its arguments in place.** Never reuse a contour or CP array across calls.
4. **`draw_drop` defaults to `dpi=256`.** Pass 64 explicitly; all existing images are 744×492.
5. **Two-thirds of the on-grid set renders deterministically** (only positive roughness is stochastic). Useful as a sharp regression instrument; misleading if treated as a render-noise sample.
6. **Angle=180° carries a one-sided boundary artifact** independent of everything else. Exclude or report it separately.
7. **Single-point conclusions in Part 1 have twice proven wrong** (revs. 13 and 17). If a finding rests on one image, widen it before acting on it.

## Changelog

- **rev. 13 (2026-09-13)** — Session 1 prepared, not run. Rewrote `phase6_reflective_sweep.py` to the Step 1b spec (new output `phase6_reflective_sweep_1b.csv`; the rev. 5 CSV is untouched) and added `phase6_separability.py`, the offline analysis that produces the Step 2 input. Smoke-tested both end to end on the three 6.1 repro cells plus a genuine 150° non-reflective drop: the discriminants computed by mirroring `CPID` match its printed values 4/4; the repro cells fail and the 150° drop passes as expected. **One early flag**: `disc_max` was saturated at exactly 90 on all three cells that reached the `>79` test, both classes — if that holds across the sweep the statistic is uninformative and (a)/(a′) are ruled out on it; the separability script tests seven alternative deployable discriminants for that case. Corrected the cascade's line numbers throughout 6.1 (now ~1825–1949; `>79` at :1866; correct branch at :1834) after 6.2's dead-code removal shifted them. No `preprocessing.py` change.
- **rev. 12 (2026-09-13)** — Planning only, no code or analysis changed. Added a "Working constraints" section recording the post-PhD scope: paper-neutral fix preferred, limitation-only outcome is the fallback not the target, two-session budget with a deliberate escalation decision. Added option (a′), a scale-normalised threshold, between (a) and (b) — motivated directly by Step 1a's finding that the failure window shifts with scaler. Added a two-session plan under 6.1 with an explicit go/no-go. Rescoped Phase 7: 7.1 is verification, 7.3/7.4 are optional and depend on a supervisor decision about adding the off-grid set to the paper.
- **rev. 11 (2026-09-13)** — 6.0 done. Curated `training_files/` (dropped `check_imports.py`, `clean_output_file.py`, `data_check.py`, `shuffle_test.py`; kept `test_evaluation.py`/`test_model.sh` — a post-training held-out-test scorer, not a restart script as first remembered — and `examine_db.py` as the only reader for the kept `.db`), recorded the curation in its README, and committed it, the `prepare_experimental.py` docstring fix, and all of `synthetic_reanalysis/` including `data/` as three commits. Working tree is clean. No analysis changed. Decision on whether PART1/PART2 ship in the public repo deferred to merge time — PART1 is a working log and probably should not; PART2 could plausibly stay as a methodology/limitations record.
- **rev. 10 (2026-08-14)** — Review pass over revs. 5–9, checking the sweep CSV directly rather than its summary. No code changed. **6.1 Step 2 is not ready to start**; three additions gate it.

  **New 6.0 — repo hygiene, marked do-first.** All of the following exist only in a working tree: the `training_files/` correction (PART1's foundational finding — the superseded pipeline deleted, the recovered real one untracked), the entire `synthetic_reanalysis/` directory (code, 324 images, every result CSV, both documents — and Phase 7.2 plans to *delete* that directory), and the `prepare_experimental.py` docstring fix. Added a curation step per dgshaw: several `training_files/` entries (`check_imports.py`, `clean_output_file.py`, `data_check.py`, `shuffle_test.py`, `examine_db.py`) were development-time utilities rather than part of the pipeline that generated data or trained the model, and needn't be in the repo — proposed keep/remove split recorded, with `test_evaluation.py`/`test_model.sh` and `examine_db.py` (the only reader of the retained `hyperparameter_study.db`) flagged as judgement calls. Noted that curation makes `training_files/` a deliberate subset rather than a mirror of the archival copy, which must be said in its README so future readers don't read absence as loss.

  **New Step 1a — corrections to rev. 5's map reading.** (1) A **fourth branch** exists that rev. 5 never mentions — `"maybe high angle, Lfit"`, 56 cells, containing the sweep's single worst CP error (42.9 px) and 9 cells >10 px at angle 75 — and a `>79` retune does not touch it. (2) Four more >10 px cells come from the `none` branch, bypassing the cascade entirely. So of the 99 failing cells, 86 are retunable and **13 are not** — rev. 5 reads as though all of it is the one branch. (3) The scaler claim is wrong: at scaler 4.0 angles 80 *and* 85 take the **correct** branch (13/16, 1.9–2.6 px); the failure window **shifts** with scaler (1.0 → 80–90, 2.83 → 75–90, 4.0 → 90–95) rather than being one band, meaning the discriminant tracks apparent image scale rather than contact angle. (4) The `ml_error` column holds error-message strings, not numeric error, and collides with the numeric column of the same name in `phase4_measurements.csv`.

  **New Step 1b — instrument the map before choosing a fix.** The rev. 5 sweep can't support the decision: it records which branch fired but not the discriminant values the branch tested (so every candidate threshold costs a fresh 40-minute run), samples only reflective cells (so the class the `>79` branch exists to protect is unmapped and no margin can be assessed), and stops at 110° (excluding the genuine high-angle systems a looser threshold would endanger). Specified one wider run — angle 50–180, roughness −0.080 through +0.010 including 0.0, discriminant values recorded per cell — ≈1700 cells at ~2× the previous runtime, replacing an iterative loop with a single offline analysis. Required output is the two-class separability question answered per scaler. Expected outcome recorded in advance: given the scaler-shifting window and the non-monotonic roughness behaviour, a clean separating threshold is unlikely to exist — in which case the SI-map becomes the primary deliverable and (b)/(b′) become the only real fix paths.

- **rev. 9 (2026-08-14)** — Extended 6.2's dead-code cleanup and committed 6.2 to its own branch. No further `preprocessing.py` behavior change beyond what rev. 6 already described.

  **Dead-code sweep, extended.** Rev. 6 flagged `find_CP_static_hydrophobic()` as orphaned (collateral from deleting its only caller, `CP_ID_static()`) but left it in place as out of 6.2's stated scope. Per dgshaw's request to clean it up, swept the rest of `preprocessing.py` the same way: grepped every top-level `def` name repo-wide against its call sites. Found four more with zero real callers — `linear_fit_ols`, `check_RMSE_gap`, `remove_close_values` (its only other occurrences were inside its own docstring's `>>>` doctest examples, never executed since no doctest runner is configured), and `check_order`. All five verified independently self-contained (no shared helpers that would themselves become newly orphaned) before removal — 255 lines deleted. Regression harness re-run after: 0 changes on the 324 synthetic + 171 experimental paper images.

  **6.2 committed.** Created `fix/cpid-sparse-contour-crash` off `origin/develop` (not off `fix/flake8-pyflakes-cleanup`, the branch HEAD happened to be on — that branch's 34 commits were already merged into `origin/develop` via PR #346, confirmed by checking that local `develop` is a clean, non-diverged ancestor of `origin/develop` and that `fix/flake8-pyflakes-cleanup`'s tip is exactly `origin/develop`'s pre-merge parent — so branching from `develop` avoided carrying unrelated, already-landed history). Two commits: the crash fix itself, then the extended dead-code sweep as a separate commit. Commit messages rewritten to match `develop`'s actual convention (`type(scope): subject` + prose body explaining root cause and verification evidence; no `Co-Authored-By` trailer — none of the last 30 commits on `develop` use one) via `git reset --soft` + recommit rather than an interactive rebase, with unrelated uncommitted working-tree changes (the `prepare_experimental.py` edit, `training_files/` changes, untracked `synthetic_reanalysis/`) stashed and restored around it, verified byte-identical after. Not pushed.

  **Real-dataset verification of 6.2's actual impact (prompted by dgshaw's questions about video coverage).** Checked whether the sparse-contour defect ever fires outside the deliberate off-grid sweep:
  - 324 synthetic + 171 experimental: confirmed via the before/after regression-harness diff (0 changes) — proves the defect fired in neither run.
  - Isaac Pass 2 (794 stored rows) and inverse-phase (419 stored rows): both datasets' stored results (produced by the original, unfixed code) have zero CPID-related error/skip rows, so the defect never fired there either.
  - Isaac Pass 1 (calibration sampling — silently drops failed frames, never recorded in any CSV, the one gap the stored artifacts alone couldn't rule out): re-ran directly against the fixed code, capturing every exception explicitly. 80/80 sampled frames across all three Isaac datasets succeeded, and the sparse-contour fallback warning never printed once — confirms the fallback path isn't even being silently exercised there, not just that nothing crashed.
  - Kristina (55 frames, all of them): re-ran in memory against the fixed code (not touching the stored CSV). The 13 frames marked `skip_reason=cpid_failed` in the stored baseline are **unchanged** post-fix — traced directly to a different, previously undocumented defect: `IndexError: index 10 is out of bounds for axis 0 with size 8` in `check_for_high_angle()` (`widest = edge[10]` with no bounds check), one function upstream of everything 6.2 touched. dgshaw confirmed this is expected: the drop exits the frame near the end of this video, so CPID legitimately has nothing to detect. Not fixed this session — out of 6.2's scope, noted here so it isn't rediscovered as a surprise.

  Net: the 6.2 defect is confirmed absent from every dataset the repo ships results for. It was only ever observed in the deliberate Phase 6 sweep (6/624 cells) and its one documented reproduction image.

- **rev. 8 (2026-08-14)** — Reverted all of rev. 7's `training_files/create_contours_args6.py` changes (both defect 1 and defect 2) and marked 6.3's code fixes **skipped**, per dgshaw's clarification: `OpenDrop_OP/training_files/` exists specifically as the fixed, reproducible record of the approach used to train the shipped model, and must not be altered even to fix real bugs in it — the rev. 7 provenance check confirmed the *archival copy* was untouched, but missed that this working copy itself is meant to be held fixed too, not just a disposable dev artifact. If either defect needs fixing for a future retraining effort, it should be done in a separate copy outside `training_files/`, never in-place. Confirmed via diff that `training_files/create_contours_args6.py` is now identical to the archival original at `files_from_old_computer/_mse_0.4218772351741791/training_files/create_contours_args6.py` except for the pre-existing, already-documented `normalize_ds` docstring fix from PART1 rev. 2. No files touched this rev besides this doc and the revert itself.

- **rev. 7 (2026-08-14)** — *(Superseded by rev. 8 — code changes described below were reverted.)* — Implemented 6.3 defects 1 and 2 (`training_files/create_contours_args6.py`, `prepare_contour()`), started in parallel with 6.2 since it touches an entirely different, training-only file. Defect 3 (the training-grid CP-disagreement audit) deliberately left open per dgshaw's direction: retraining is out of scope this session, the paper is likely closed to new limitations sections, and the audit is the most compute-heavy item in Phase 6 — descoping it is explicitly sanctioned by the spec's own "descope candidate" framing. A cheap sanity check (a few hundred sampled points instead of the full grid) was suggested as a lower-cost way to test the intuition that fixing it wouldn't move results, but not run.

  **Provenance check (raised by dgshaw before proceeding):** does editing this file obscure the record of how the shipped model's training data was actually generated? No — confirmed by diffing the edited working copy against the archival original. Per PART1 rev. 6, `OpenDrop_OP/training_files/create_contours_args6.py` was already a *recovered copy*, not the generation-time file itself; the actual generation-time file, provenance-verified by MD5 match on `saved_model.pb` against the shipped model, is frozen at `files_from_old_computer/_mse_0.4218772351741791/training_files/create_contours_args6.py` — outside `OpenDrop_OP/`, untouched by this or any other change here (mtime still May 2025). Diffing the two confirms the only pre-existing divergence was the `normalize_ds` docstring correction already documented in PART1 rev. 2; this session's edits are the only other delta. The historical record of what actually produced the shipped model survives independently of this file, so 6.3 does not need to be skipped on provenance grounds — it only affects hypothetical future regeneration, which is moot anyway since retraining is out of scope this session.

  **What changed:**
  1. **Defect 1 (empty-half `IndexError`).** `prepare_contour()` now checks `l_drop`/`r_drop` for emptiness right after the apex-side split and raises a `ValueError` naming which side was empty, the computed `xapex`, and the generating parameters (`angle`, `Bo`, `scaler`, `roughness`), instead of failing later with an opaque `IndexError: too many indices for array` at the `max(l_drop[:,0])` transpose step. `prepare_contour()` gained an optional `params` dict for this context; `create_contour()` (the only caller with the angle/Bo/scaler/roughness in scope) now passes it through.
  2. **Defect 2 (CP snap into a folded contour region).** Added `_is_locally_monotonic_x()` (checks a ±2-point window of `drop[:,0]` for a sign change — a local X-reversal indicating a topological self-crossing) and `_snap_cp_to_contour()` (walks candidate contour points in ascending distance order, returning the nearest one that passes the monotonicity check; falls back to the plain-nearest point only if every candidate is folded). Both of the CP-not-in-contour snap sites in `prepare_contour()` now call `_snap_cp_to_contour()` instead of a bare `min(..., key=distance)`. Deliberately a monotonicity test, not flatness/steepness, per the spec's explicit warning not to reintroduce the retracted flat-run framing.

  **Verification:**
  - Reproduction command (`create_contour(angle=160, Bo=0.55, scaler=0.875, roughness=-0.08)`) now raises `ValueError: prepare_contour: extracted contour has no points on the left side of the apex (xapex=372.0) for angle=160, Bo=0.55, scaler=0.875, roughness=-0.08...` instead of `IndexError`.
  - A previously-working case (`angle=90, Bo=0.55, scaler=1.5, roughness=0.0`) still succeeds unchanged.
  - Broad sanity sweep: 5 angles × 3 Bo × 3 scaler × 3 roughness (135 combinations, chosen ad hoc, not the full training grid) — 135/135 completed without any unexpected exception (0 `IndexError`, 0 uncaught errors); the new `ValueError` path wasn't hit in this sweep since none of the combinations land in the empty-half regime.
  - No `opendrop_ml/` or `preprocessing.py` file touched; only `training_files/create_contours_args6.py` changed, so 6.2's regression harness and the paper's `synthetic_reanalysis/` outputs are unaffected by construction.
  - No formal pytest test file added yet — `training_files/` has no existing test suite/convention to extend (unlike `opendrop_ml/`, which uses co-located `*_test.py`); flagged here rather than silently skipped.

  Files touched: `training_files/create_contours_args6.py` only.

- **rev. 6 (2026-08-14)** — Implemented and shipped 6.2 (`iterative_Lfit` sparse-contour crash), the only `preprocessing.py` change this session. 6.1's retune/discriminator decision was deliberately left open (see rev. 5) — 6.2 was independent, lower-risk, and already surfaced by the rev. 5 sweep as 6 crashed cells, so it was done first per dgshaw's direction.

  **What changed in `opendrop_ml/modules/preprocessing/preprocessing.py`:**
  1. **`iterative_Lfit()` sparse-contour branch (Step 1).** The `else` branch for contour lengths outside `{>=110, ==2}` used to print a warning and fall through with `left`/`right` unassigned, crashing at `contours = [left, right]`. It now takes an optional `fallback_points=[left_point, right_point]` kwarg: if supplied, it returns those points directly (skipping refinement, which cannot run reliably on a handful of points anyway — downstream RMSE/peak-finding needs at least `n_pts` points per side and would only crash again deeper in the function) and prints a low-confidence warning; if not supplied, it raises a `ValueError` naming the actual contour length instead of an unrelated `UnboundLocalError`. All three call sites in `CPID()` were updated to pass `fallback_points=widest_points` or `fallback_points=CP_guesses` (whichever candidate points were used to build that call's restricted window) — this reuses the exact fallback pattern (`final_CPs = widest_points`) already present elsewhere in `CPID()`, not a new heuristic, per the spec's framing.
  2. **Low-confidence signal (Step 2).** The fallback path prints `"WARNING: iterative_Lfit contour is sparse (...); using the caller's candidate points directly instead of refining. Predictions may be unreliable."`, matching the wording style and print-only mechanism of the existing `profile_length < 234` warning in `prepare_experimental.py` (that warning is also just a print, not a return flag, so this doesn't introduce a new signalling channel).
  3. **Slice-start clamping (Step 3).** Two of the three `restricted_contour` constructions in `CPID()` lacked `max(0, ...)` on their start index; the third already had it. Added `max(0, ...)` to the other two (the "otherwise use Lfit if unsure" and "maybe high angle" branches). This incidentally fixed a *second*, previously-undocumented crash the rev. 5 sweep surfaced at `(angle=75, scaler=4.0, roughness=-0.015)`: `'int' object is not subscriptable`. Root-caused: an unclamped negative start wrapped from the end of the array (Known trap #2) and happened to produce an exactly-2-point restricted contour, which hit `iterative_Lfit`'s `len(contour) == 2` branch under the wrong assumption (a real 2-point window there is supposed to mean "already a pre-split `[left, right]` pair", not two literal raw points) — sorting a single `[x, y]` pair as if it were a list of many points then crashed on `x[1]` where `x` is a bare float. Clamping the start prevents the wrap that produced the degenerate window in the first place. Verified fixed: all 6 originally-crashed sweep cells (5x the documented `UnboundLocalError`, 1x this one) now return plausible CPs.
  4. **`video_two_pass.py:436` (Step 3 continued).** Checked the call chain: `CPID()` at that line is called from `_rerun_preprocess_downsampled()`, which is only called from `_predict_ml()`, which is only called from `_try_ml()` — and `_try_ml()` wraps that entire call in `try/except Exception: pass`. So it *is* covered, just not by a try/except immediately surrounding line 436 itself. No code change needed there; confirmed by grepping all call sites, not by inspection alone.
  5. **Dead twin (Step 4).** Confirmed via repo-wide grep that nothing calls `find_CP_static_Lfit()` or `CP_ID_static()` outside their own definitions and each other. Deleted both. Collateral: `find_CP_static_hydrophobic()` was only ever called from the now-deleted `CP_ID_static()`, so it is now also orphaned — left in place since it wasn't named in the spec's Step 4 scope, but flagged here so it isn't silently forgotten; a candidate for a future, explicitly-scoped cleanup pass.

  **Verification:**
  - Reproduction command (`160.0_0.5500_5.0000_-0.0150`) now returns `CPs = [[340, 454], [403, 454]]` instead of raising.
  - New test file `opendrop_ml/modules/preprocessing/preprocessing_test.py` (none existed for this module before): 4 tests — sparse contour without fallback raises a clear `ValueError`; sparse contour with fallback returns the fallback points; the `len(contour) == 2` pre-split-pair branch is confirmed unaffected (still not `UnboundLocalError` if it fails for other reasons); and the actual reproducing image (rendered via `draw_drop` with a hardcoded seed rather than a file path, so the test doesn't depend on `synthetic_reanalysis/` surviving Phase 7's planned deletion) round-trips through `CPID()` without raising. All 4 pass. `flake8` clean on both changed files.
  - **Regression harness (6.4) re-run after the change: 0 changed cells across 324 synthetic + 171 experimental images, 19.8 min.** Matches the spec's expectation (zero occurrences of this defect in either paper set) and confirms the fix is a pure crash-recovery with no effect on currently-passing images.

  Files touched: `opendrop_ml/modules/preprocessing/preprocessing.py` (only shipped-code change), `opendrop_ml/modules/preprocessing/preprocessing_test.py` (new).

- **rev. 5 (2026-08-14)** — Built and ran 6.4 and 6.1 Step 1. No code in `preprocessing.py` or elsewhere in shipped code was touched.

  **6.4 — regression harness.** Built `phase6_regression.py`. Baseline is a frozen copy of `paper_results/results/{synthetic,experimental}_results.csv` under `synthetic_reanalysis/results/phase6_baseline_*.csv` (captured via `--capture-baseline`), not the live `paper_results/results/` files, since Phase 7 will later overwrite those per the ground rules and the harness needs a stable reference to stay meaningful after that happens. Compares every `ml_*`/`yl_*`/`tangent_*`/`polynomial_*`/`circular_*`/`ellipse_*` angle column (tolerance 0.01°) and every `left_CP_*`/`right_CP_*` column (exact integer match) per image, checkpointed with the accumulate+atomic-rewrite pattern (reusing the same approach as `evaluate_offgrid.py`, not a header-less append). Run against unmodified code: **324 synthetic + 171 experimental images, 0 changed cells, 19.9 min** (faster than the ~25–30 min estimate). Harness verified sound; ready to gate any `preprocessing.py` change.

  **6.1 Step 1 — reflective-sweep map.** Built and ran `phase6_reflective_sweep.py`: 624 cells (angle 50–110 step 5 × roughness −0.005…−0.080 step 0.005 × scaler ∈ {1.0, 2.83, 4.0}, bond=0.55 fixed), `dpi=64` explicit, seeded via `matrix_utils.cell_seed` exactly as the offgrid generator does. **618 ok, 6 crashed** — all 6 crashes are the *known* 6.2 `UnboundLocalError`/sparse-contour defect (`scaler=4.0`, angle 70–75, assorted negative roughness), an unplanned but useful cross-confirmation that 6.2's failure mode reaches into the same off-grid corner as 6.1, though fixing it is out of this session's scope. Results in `results/phase6_reflective_sweep.csv`.

  **What the map shows:**
  1. The buggy branch (`"looks reflective but is high angle"`) fires for essentially **all** cells with angle ≥ 85° regardless of scaler or roughness, and for angle 80° at most negative-roughness cells too. The correct branch (`"large angles under widest points…"`) only ever fires in a narrow angle window, roughly 70–85° depending on scaler.
  2. **Scaler matters as much as roughness, and the spec didn't anticipate how badly.** At `scaler=1.0` and `scaler=4.0` the correct branch does fire for a real (if narrow) subset of angle-80/85 cells. At **`scaler=2.83` it never fires once** across all 16 roughness values tested at angle 80 or 85 — every single one of those 32 cells takes the wrong branch. This matters because 2.83 is the scaler used in the study's own off-grid reproduction cases (`80.0_0.5500_2.8300_-0.0450` etc.) — the failure isn't just "the on-grid roughness happened to be lucky," it's that the discriminating statistic (the under-widest-point angle) doesn't track true contact angle consistently across image scale at all.
  3. **The branch is not monotonic in roughness at fixed angle/scaler.** At `scaler=1.0, angle=80`, the branch alternates correct/wrong roughly every other 0.005 step in roughness (R, H, R, H, R, H…) rather than switching cleanly at one boundary value. The per-cell random surface jitter moves the measured angle across the `>79` threshold from one seed to the next even at similar nominal roughness — the boundary is noisy, not a clean line.
  4. **CP error tracks exactly where the physical harm is.** Where the wrong branch fires, mean CP error is ~20–24 px for angle 80–90° (max ~42 px), drops to ~8.5 px at 95°, and is down to 1–5 px by 100–110° — because by 100–110° the wrong-looking branch is actually the *correct* classification (those are genuine high-angle systems). So the real failure region is angle ∈ [80°, ~95°] combined with negative roughness, exactly as the spec anticipated, but scaler ≈ 2.83 is a materially worse sub-case than scaler 1.0/4.0.

  **Whether a retune (candidate (a)) looks likely to fix it:** partially, and not cleanly. At scaler 1.0/4.0 there's a genuine (if narrow) margin the boundary could be pushed into, which could plausibly extend correct classification from ~85° up toward 90–95° without disturbing the paper's own on-grid case (scaler 1.0, roughness −0.03, angle 80, currently correct). But at scaler ≈ 2.83 the discriminant never once reaches the correct side of any threshold in the tested range — there is no single `>N` value for the existing statistic that fixes that sub-case, since the problem there isn't threshold placement, it's that the statistic itself doesn't separate the two classes at that scaler. Combined with the non-monotonic roughness behavior at scaler 1.0 (point 3 above), a scalar retune looks likely to **shrink** the failure window at some scalers while leaving scaler≈2.83 essentially untouched — consistent with the spec's own expectation that "the boundary moves, it doesn't necessarily get eliminated." This is a finding to inform the choice, not a decision — the choice between (a) and (b)/(b′) is the user's, per the spec.

  No code changed. `results/phase6_reflective_sweep.csv` and `results/phase6_regression_diff.csv` are the artifacts. `results/phase6_baseline_{synthetic,experimental}.csv` are the frozen regression baselines for future runs.

- **rev. 4 (2026-08-14)** — Flagged the 6.1 reflective-failure map as a possible Supplementary Information deliverable, to be decided after the fix is chosen rather than now. If the retune shifts the boundary without eliminating the failure, the map documents a real static-image limitation with a known mitigation (video CP tracking), which is publishable in its own right. Noted that an SI version should be a second, denser presentation-quality run rather than the diagnostic sweep reused as a figure, and carried the decision into the Phase 7.3 checklist.
- **rev. 3 (2026-08-14)** — Two corrections from dgshaw, both narrowing scope. (1) **6.1 reframed.** Tilt weakness retracted: tilted systems are tilt-corrected first by design, since the contact angle is defined relative to macroscopic surface tilt, so horizontal-mirror symmetry is a valid downstream assumption. More importantly, the `"looks reflective but is high angle"` branch targets contact angles *just below 90° on reflective surfaces*, which are optically almost identical in a static image — genuinely hard, which is why the video pipeline resolves it by tracking CPs across frames instead. So the realistic best outcome is "the boundary moves and the residual limitation is documented," not "fixed"; the actual defect is that the threshold lands correctly at exactly the on-grid roughness (−0.03) and wrongly at its neighbours. Preference order revised to **try the retune (a) first**, with (b)/(b′) treated as global algorithm changes requiring verification across *all system types* — including the three video datasets — plus a full `paper_results/` re-run. (2) **6.3's CP question settled by exhaustive search**: the CPs stored in the data-point keys are read by *nothing* — only `create_contours_args6.py:513`/`:560` (the generator) touch fields 4/5, and all five pkl consumers parse only field [0], the angle. dgshaw's recollection was correct. The only surviving concern is that the CP positioned the 50-point window at generation time, which is baked into the shipped model and unfixable without retraining — so the audit is now explicitly marked **descopable**, since it characterises a limitation rather than fixing anything.
- **rev. 2 (2026-08-14)** — Three clarifications from dgshaw. (1) **Regeneration decided**: nothing is published and figure generation is largely automated, so improved numbers are adopted and the paper rewritten, rather than preserving known-wrong results; ground rule 3 and Phase 7.1 updated. (2) **Symmetry-test recommendation withdrawn.** Clarified it is an upper-vs-lower mirror about the baseline, not left-vs-right — so sliding drops with unequal advancing/receding angles are unaffected — but recorded three real weaknesses raised in review (tilted substrate breaks the mirror axis, which is exactly the sliding-drop setup; real reflections are attenuated/partial so synthetic performance won't transfer; near-180° spheres are genuinely mirror-symmetric and stay degenerate). Added candidate (b′), a termination-shape discriminator that doesn't degenerate at 180°, and removed the pre-commitment so the Step-1 map decides. (3) **Corrected the framing of 6.3's training-CP audit**: the CP is confirmed *not* a training label (`finish_optuna.py:45` uses angle alone) but it *does* anchor the 50-point window via `sampling()`, so a mislocated CP slides the window while the label stays true — which raises the audit's value rather than lowering it. Bounded it against the ±5-point window-shift augmentation, and rewrote the audit's reporting to split at that threshold.
- **rev. 1 (2026-08-14)** — Split out of `SYNTHETIC_DATA_REANALYSIS.md` as the forward-looking half. Expanded Phases 6 and 7 from checklists into an executable specification: added the 6.4 regression harness as a prerequisite for any `preprocessing.py` change, runnable reproduction commands for 6.1 and 6.2, an explicit 624-render sweep spec for mapping the reflective failure boundary before choosing a fix, per-item acceptance criteria, and a "Known traps" section collecting the failure modes Part 1 hit.
