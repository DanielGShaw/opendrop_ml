"""Compares a freshly-regenerated experimental_results.csv against a saved
reference on every column except the *_time timing columns, which are
expected to vary by machine. Exits non-zero if any non-timing column differs.

Default (strict) mode is the same-machine regression check: every column
must match to within 1e-6. --cross-platform is what smoke-test.yml uses to
compare a Windows/Linux build against the committed macOS reference. It
widens the ML-angle tolerance to cover float32 TensorFlow kernel noise and
reports, but does not fail on, the KNOWN_PLATFORM_SENSITIVE images.

Usage: python diff_experimental_results.py [--cross-platform] <reference.csv> <regenerated.csv>
"""

import sys

import pandas as pd

TOLERANCE = 1e-6

# --cross-platform only. The ML model runs in float32, and TensorFlow's CPU
# kernels differ by platform: observed ML-angle differences are <= ~1e-4 deg.
CROSS_PLATFORM_TOLERANCES = {"ml_left_angle": 1e-3, "ml_right_angle": 1e-3}

# --cross-platform only. Images whose pipeline sits on a knife-edge that
# sub-ULP float differences can tip, as {image_name: reason}. None at present.
KNOWN_PLATFORM_SENSITIVE = {}


def main():
    args = sys.argv[1:]
    cross_platform = "--cross-platform" in args
    ref_path, new_path = [a for a in args if a != "--cross-platform"]
    ref = pd.read_csv(ref_path)
    new = pd.read_csv(new_path)
    tolerances = CROSS_PLATFORM_TOLERANCES if cross_platform else {}
    exempt = (
        ref["image_name"].isin(KNOWN_PLATFORM_SENSITIVE) if cross_platform else False
    )

    if list(ref["image_name"]) != list(new["image_name"]):
        sys.exit(
            "Row set/order differs from the reference -- cannot compare column-wise"
        )

    timing_cols = [c for c in ref.columns if c.endswith("_time")]
    compare_cols = [
        c for c in ref.columns if c not in timing_cols and c != "image_name"
    ]

    mismatches = {}
    exempt_hits = {}
    for col in compare_cols:
        # Coerce both sides to numeric first, rather than trusting the
        # CSV-inferred dtype: a run with failures introduces NaNs that
        # upcast an int64 column to float64 on one side only, and pandas
        # then renders e.g. 1066 vs 1066.0 -- comparing those as strings
        # would flag every row as different even when the value matches.
        r = pd.to_numeric(ref[col], errors="coerce")
        n = pd.to_numeric(new[col], errors="coerce")
        if r.notna().any() or n.notna().any():
            both_nan = r.isna() & n.isna()
            one_nan = r.isna() ^ n.isna()
            diff = (r - n).abs()
            tol = tolerances.get(col, TOLERANCE)
            bad_mask = (~both_nan) & (one_nan | (diff.fillna(float("inf")) > tol))
        else:
            bad_mask = ref[col].astype(str) != new[col].astype(str)
        for name in ref.loc[bad_mask & exempt, "image_name"]:
            exempt_hits.setdefault(name, []).append(col)
        n_bad = int((bad_mask & ~exempt).sum())
        if n_bad:
            mismatches[col] = n_bad

    for name, cols in exempt_hits.items():
        print(
            f"Known platform-sensitive image {name} differs in {len(cols)} column(s) "
            f"(not counted): {KNOWN_PLATFORM_SENSITIVE[name]}"
        )

    if mismatches:
        print("Non-timing columns differ from the reference:")
        for col, n in mismatches.items():
            print(f"  {col}: {n} row(s) differ")
        sys.exit(1)

    print(
        f"OK: all {len(compare_cols)} non-timing columns match the reference "
        f"across {len(ref)} rows"
        + (" (cross-platform tolerances)." if cross_platform else " (within 1e-6).")
    )


if __name__ == "__main__":
    main()
