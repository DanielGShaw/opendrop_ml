# Phase 2B — image-pipeline smoke test

Smoke test / large-effect detector only (n~17-18 per set) -- see Phase 2A (contour-level, 5400 samples) for the statistically powered primary endpoint. Go/no-go: renderer or preprocessing failures here mean fix the generator or trim the matrix before Phase 3, not chase a formal comparison at this n.

### Off-grid subset (n=17)
- Hard failures (preprocessing/evaluate_image exception): 0/17
- profile_length: min=112, median=482, max=1076 (reference thresholds from prepare_experimental.py: <234 low-res warning, 565-974 conditional downsample, >974 auto-downsample)
    - 3 image(s) below the 234-point low-resolution warning threshold
    - 2 image(s) above the 974-point auto-downsample threshold
- ML angle MAE (avg of L+R vs. gt_angle): 4.48 deg (median 1.83, max 21.23, n=17)

### On-grid control subset (n=17)
- Hard failures (preprocessing/evaluate_image exception): 0/17
- profile_length: min=138, median=556, max=1013 (reference thresholds from prepare_experimental.py: <234 low-res warning, 565-974 conditional downsample, >974 auto-downsample)
    - 5 image(s) below the 234-point low-resolution warning threshold
    - 3 image(s) above the 974-point auto-downsample threshold
- ML angle MAE (avg of L+R vs. gt_angle): 4.53 deg (median 2.32, max 13.86, n=17)
