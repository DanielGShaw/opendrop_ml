# Phase 4 -- comparative analysis



Pooled off-grid (n_images=324) + on-grid-control (n_images=324).

## Render-noise null band (context, from Phase 1/3 replicates)

replicate_0: MAE=2.2383
replicate_1: MAE=2.2832
replicate_2: MAE=2.1767
replicate_3: MAE=2.1685
replicate_4: MAE=2.2489
Band: [2.1685, 2.2832], SD=0.0439

## Primary endpoint: image-level |error| vs. grid_distance slope

| Method | OLS slope | Theil-Sen slope | Theil-Sen 95% CI (analytical) |
|---|---|---|---|
| tangent | 0.6424 | 0.2592 | [0.0000, 0.5420] |
| polynomial | 0.6359 | 0.5181 | [0.1460, 0.9261] |
| circular | 1.5237 | 0.7060 | [0.2696, 1.2685] |
| ellipse | 0.4838 | 0.2481 | [0.0657, 0.4562] |
| yl | 0.1715 | 0.1617 | [0.0524, 0.2862] |
| ml | 0.0668 | -0.0040 | [-0.0525, 0.0458] |

ML OLS slope 95% bootstrap CI (image-level, n=10000): [-0.1853, 0.3455]
OLS/Theil-Sen ratio: -16.57 (material disagreement -> Theil-Sen is the primary estimate)
ML Theil-Sen slope 95% bootstrap CI (image-level, n=2000): [-0.0708, 0.0660]

Primary estimate: Theil-Sen slope = -0.0040 deg/unit, 95% CI [-0.0708, 0.0660]
CI excludes zero: False
Predicted |error| increase at max observed grid_distance (5.12): -0.0206 deg vs. image-level practical bar 1.4 deg
Verdict: BOUNDED NULL -- any grid-proximity effect is below 0.0660 deg/unit (95% CI)

## Paired nearest-on-grid-control matching

n pairs: 324
Mean signed difference (bias, off-grid minus matched on-grid): 0.2780 deg
Mean |difference|: 1.3774 deg
Correlation(|Δ|, off-grid grid_distance): r=0.038
Out-of-domain subset (n=108): mean bias -0.1779 deg, mean |Δ| 1.3960 deg

## CP-localisation mediation

offgrid: n=648, mean CP error 2.66 px, corr(CP error px, ML angle error) = 0.643
ongrid_control: n=648, mean CP error 2.77 px, corr(CP error px, ML angle error) = 0.629
pooled: n=1296, corr = 0.634

## Beyond-MAE tail metrics

| set | method | n | MAE | median | p90 | max | failure rate |
|---|---|---|---|---|---|---|---|
| offgrid | tangent | 648 | 7.780 | 4.784 | 15.519 | 114.741 | 0/324 |
| offgrid | polynomial | 648 | 10.238 | 7.509 | 20.041 | 173.646 | 0/324 |
| offgrid | circular | 648 | 14.619 | 7.954 | 42.159 | 63.021 | 0/324 |
| offgrid | ellipse | 648 | 8.002 | 3.325 | 23.972 | 39.104 | 0/324 |
| offgrid | yl | 648 | 3.691 | 1.927 | 11.812 | 23.878 | 0/324 |
| offgrid | ml | 648 | 2.022 | 0.838 | 6.990 | 21.268 | 0/324 |
| ongrid_control | tangent | 648 | 6.672 | 4.511 | 13.305 | 80.990 | 0/324 |
| ongrid_control | polynomial | 648 | 9.192 | 6.076 | 16.230 | 171.469 | 0/324 |
| ongrid_control | circular | 648 | 12.421 | 5.562 | 37.206 | 58.440 | 0/324 |
| ongrid_control | ellipse | 648 | 7.282 | 2.283 | 22.451 | 56.651 | 0/324 |
| ongrid_control | yl | 648 | 3.407 | 1.443 | 9.402 | 22.948 | 0/324 |
| ongrid_control | ml | 648 | 2.365 | 0.873 | 8.095 | 18.662 | 0/324 |

### ML error by in-domain / out-of-domain (off-grid only)
| group | n | MAE | median | max |
|---|---|---|---|---|
| in_domain | 432 | 1.708 | 0.750 | 14.351 |
| out_of_domain | 216 | 2.649 | 0.942 | 21.268 |

### ML error by out-of-domain axis (off-grid only)
| ood_axis | n | MAE | median | max |
|---|---|---|---|---|
| bond | 72 | 1.627 | 0.849 | 11.271 |
| compound_bond_scaler | 18 | 6.350 | 7.423 | 21.268 |
| compound_scaler_roughness | 18 | 1.904 | 0.871 | 9.215 |
| roughness | 36 | 1.871 | 0.829 | 13.304 |
| scaler | 72 | 3.321 | 1.638 | 18.571 |

### ML error by angle bucket
| set | bucket | n | MAE | median |
|---|---|---|---|---|
| offgrid | <90 | 288 | 0.842 | 0.604 |
| offgrid | 90-110 | 72 | 1.415 | 0.854 |
| offgrid | >=110 | 288 | 3.353 | 1.603 |
| ongrid_control | <90 | 288 | 0.768 | 0.601 |
| ongrid_control | 90-110 | 72 | 3.413 | 1.794 |
| ongrid_control | >=110 | 288 | 3.700 | 1.444 |