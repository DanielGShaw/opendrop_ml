#!/usr/bin/env python
"""Master script: runs the full paper results pipeline.

Run from OpenDrop_OP/:
    python paper_results/run_all.py
"""

import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPENDROP_DIR = os.path.dirname(SCRIPT_DIR)
PYTHON = sys.executable

# Pipeline steps in order
STEPS = [
    ("Experimental evaluation", "evaluate/evaluate_experimental.py"),
    ("Synthetic evaluation", "evaluate/evaluate_synthetic.py"),
    ("Experimental figures", "plot/plot_experimental.py"),
    ("Training-domain coverage figures", "plot/plot_training_coverage.py"),
    ("Synthetic figures", "plot/plot_synthetic.py"),
    ("Speed figures", "plot/plot_speed.py"),
    ("Sensitivity dataset evaluation", "evaluate/evaluate_sensitivity.py"),
    ("Sensitivity dataset figures", "plot/plot_sensitivity.py"),
    ("Inverse phase evaluation", "evaluate/evaluate_inverse_phase.py"),
    ("Isaac evaluation", "evaluate/evaluate_isaac.py"),
    ("Kristina evaluation", "evaluate/evaluate_kristina.py"),
    ("Video analysis figures", "plot/plot_video_analysis.py"),
    ("CP-comparison composite figures", "plot/plot_cp_comparison.py"),
]


def main():
    print("=" * 60)
    print("Paper Results — Full Pipeline")
    print("=" * 60)
    print()

    total_start = time.time()
    failed = []

    for step_name, script_path in STEPS:
        full_path = os.path.join(SCRIPT_DIR, script_path)
        if not os.path.exists(full_path):
            print(f"SKIP: {step_name} — {script_path} not found")
            print()
            continue

        print(f"--- {step_name} ---")
        t0 = time.time()
        ret = subprocess.run(
            [PYTHON, full_path],
            cwd=OPENDROP_DIR,
        )
        elapsed = time.time() - t0

        if ret.returncode != 0:
            print(f"FAILED: {step_name} (exit code {ret.returncode})")
            failed.append(step_name)
        else:
            print(f"OK: {step_name} ({elapsed / 60:.1f} min)")
        print()

    total_time = time.time() - total_start
    print("=" * 60)
    print(f"Pipeline complete in {total_time / 60:.1f} minutes.")
    if failed:
        print(f"FAILED steps: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All steps succeeded.")


if __name__ == "__main__":
    main()
