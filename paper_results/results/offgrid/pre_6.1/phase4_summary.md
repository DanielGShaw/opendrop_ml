# Phase 4 -- comparative analysis



Pooled off-grid (n_images=323) + on-grid-control (n_images=324).

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
| tangent | 0.8010 | 0.3527 | [0.0410, 0.6572] |
| polynomial | 0.7319 | 0.6134 | [0.2189, 1.0314] |
| circular | 1.4978 | 0.6179 | [0.2117, 1.1397] |
| ellipse | 0.4937 | 0.2808 | [0.0909, 0.4891] |
| yl | 0.1855 | 0.1905 | [0.0741, 0.3189] |
| ml | 0.1035 | 0.0112 | [-0.0383, 0.0655] |

ML OLS slope 95% bootstrap CI (image-level, n=10000): [-0.1442, 0.3890]
OLS/Theil-Sen ratio: 9.23 (material disagreement -> Theil-Sen is the primary estimate)
ML Theil-Sen slope 95% bootstrap CI (image-level, n=2000): [-0.0543, 0.0909]

Primary estimate: Theil-Sen slope = 0.0112 deg/unit, 95% CI [-0.0543, 0.0909]
CI excludes zero: False
Predicted |error| increase at max observed grid_distance (5.12): 0.0574 deg vs. image-level practical bar 1.4 deg
Verdict: BOUNDED NULL -- any grid-proximity effect is below 0.0909 deg/unit (95% CI)

## Paired nearest-on-grid-control matching

n pairs: 323
Mean signed difference (bias, off-grid minus matched on-grid): 0.7414 deg
Mean |difference|: 1.8032 deg
Correlation(|Δ|, off-grid grid_distance): r=-0.021
Out-of-domain subset (n=107): mean bias -0.0720 deg, mean |Δ| 1.4124 deg

## CP-localisation mediation

offgrid: n=646, mean CP error 2.88 px, corr(CP error px, ML angle error) = 0.785
ongrid_control: n=648, mean CP error 2.77 px, corr(CP error px, ML angle error) = 0.629
pooled: n=1294, corr = 0.710

## Beyond-MAE tail metrics

| set | method | n | MAE | median | p90 | max | failure rate |
|---|---|---|---|---|---|---|---|
| offgrid | tangent | 646 | 8.438 | 4.930 | 19.718 | 114.741 | 0/323 |
| offgrid | polynomial | 646 | 10.774 | 7.497 | 23.571 | 173.646 | 0/323 |
| offgrid | circular | 646 | 14.511 | 7.522 | 42.179 | 63.021 | 0/323 |
| offgrid | ellipse | 646 | 8.196 | 3.462 | 24.379 | 39.104 | 0/323 |
| offgrid | yl | 646 | 3.935 | 2.030 | 12.984 | 28.256 | 0/323 |
| offgrid | ml | 646 | 2.453 | 0.859 | 7.376 | 44.732 | 0/323 |
| ongrid_control | tangent | 648 | 6.672 | 4.511 | 13.305 | 80.990 | 0/324 |
| ongrid_control | polynomial | 648 | 9.192 | 6.076 | 16.230 | 171.469 | 0/324 |
| ongrid_control | circular | 648 | 12.421 | 5.562 | 37.206 | 58.440 | 0/324 |
| ongrid_control | ellipse | 648 | 7.282 | 2.283 | 22.451 | 56.651 | 0/324 |
| ongrid_control | yl | 648 | 3.407 | 1.443 | 9.402 | 22.948 | 0/324 |
| ongrid_control | ml | 648 | 2.365 | 0.873 | 8.095 | 18.662 | 0/324 |

### ML error by in-domain / out-of-domain (off-grid only)
| group | n | MAE | median | max |
|---|---|---|---|---|
| in_domain | 432 | 2.346 | 0.792 | 44.732 |
| out_of_domain | 214 | 2.668 | 0.963 | 21.268 |

### ML error by out-of-domain axis (off-grid only)
| ood_axis | n | MAE | median | max |
|---|---|---|---|---|
| bond | 72 | 1.792 | 0.859 | 11.271 |
| compound_bond_scaler | 18 | 6.350 | 7.423 | 21.268 |
| compound_scaler_roughness | 18 | 1.904 | 0.871 | 9.215 |
| roughness | 36 | 1.871 | 0.829 | 13.304 |
| scaler | 70 | 3.228 | 1.472 | 18.571 |

### ML error by angle bucket
| set | bucket | n | MAE | median |
|---|---|---|---|---|
| offgrid | <90 | 288 | 1.841 | 0.620 |
| offgrid | 90-110 | 72 | 1.415 | 0.854 |
| offgrid | >=110 | 286 | 3.331 | 1.601 |
| ongrid_control | <90 | 288 | 0.768 | 0.601 |
| ongrid_control | 90-110 | 72 | 3.413 | 1.794 |
| ongrid_control | >=110 | 288 | 3.700 | 1.444 |