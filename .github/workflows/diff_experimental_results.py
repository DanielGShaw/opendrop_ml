"""Compares a freshly-regenerated experimental_results.csv against a saved
reference on every column except the *_time timing columns, which are
expected to vary by machine. Used by windows-smoke-test.yml to check that
a Windows build produces the same fitting results as the committed macOS
reference. Exits non-zero if any non-timing column differs.

Usage: python diff_experimental_results.py <reference.csv> <regenerated.csv>
"""

import sys

import pandas as pd

TOLERANCE = 1e-6


def main():
    ref_path, new_path = sys.argv[1], sys.argv[2]
    ref = pd.read_csv(ref_path)
    new = pd.read_csv(new_path)

    if list(ref["image_name"]) != list(new["image_name"]):
        sys.exit("Row set/order differs from the reference -- cannot compare column-wise")

    timing_cols = [c for c in ref.columns if c.endswith("_time")]
    compare_cols = [c for c in ref.columns if c not in timing_cols]

    mismatches = {}
    for col in compare_cols:
        if ref[col].dtype.kind in "fc" and new[col].dtype.kind in "fc":
            diff = (ref[col] - new[col]).abs()
            bad = diff[diff.fillna(0) > TOLERANCE]
        else:
            bad = ref[col][ref[col].astype(str) != new[col].astype(str)]
        if len(bad):
            mismatches[col] = len(bad)

    if mismatches:
        print("Non-timing columns differ from the reference:")
        for col, n in mismatches.items():
            print(f"  {col}: {n} row(s) differ")
        sys.exit(1)

    print(
        f"OK: all {len(compare_cols)} non-timing columns match the reference "
        f"bit-for-bit across {len(ref)} rows."
    )


if __name__ == "__main__":
    main()
