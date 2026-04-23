"""Integration test for the video-mode needle-in two-pass pipeline.

Runs the Isaac SC69_1 dataset end-to-end through ``_needle_in_two_pass``
and asserts angles match
``paper_results/results/isaac_SC69_1_results.csv`` within backend MAE.

Not part of the default test run. Execute explicitly:

    pytest opendrop_ml/modules/contact_angle/video_two_pass_needle_in_integration_test.py -m slow

Skipped automatically if the SC69_1 video or reference CSV aren't
present (e.g. in a fresh clone without paper_results data).
"""

import csv
import gc
import os
import platform
import resource
import sys
import time

import numpy as np
import pandas as pd
import pytest


_TRACEMALLOC_ENABLED = os.environ.get("OPENDROP_TRACEMALLOC") == "1"
if _TRACEMALLOC_ENABLED:
    import tracemalloc

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

from opendrop_ml.modules.contact_angle.ca_data_processor import UnifiedCsvWriter
from opendrop_ml.modules.contact_angle.video_two_pass_needle_in import (
    _needle_in_two_pass,
)
from opendrop_ml.modules.core.classes import ExperimentalSetup
from opendrop_ml.utils.enums import FittingMethod


# Reference CSV columns (paper_results/evaluate/evaluate_isaac.py). Isaac
# reports a strict subset of the inverse-phase reference: no polynomial,
# ``circular_*`` (not ``circle_*``), no per-method ``*_analysis_time`` or
# ``*_max_error`` columns.
_COLUMN_PAIRS = {
    FittingMethod.TANGENT_FIT: (
        ("tangent fit left angle", "tangent_left_angle"),
        ("tangent fit right angle", "tangent_right_angle"),
    ),
    FittingMethod.CIRCLE_FIT: (
        ("circle fit left angle", "circular_left_angle"),
        ("circle fit right angle", "circular_right_angle"),
    ),
    FittingMethod.ELLIPSE_FIT: (
        ("ellipse fit left angle", "ellipse_left_angle"),
        ("ellipse fit right angle", "ellipse_right_angle"),
    ),
    FittingMethod.YL_FIT: (
        ("YL fit left angle", "yl_left_angle"),
        ("YL fit right angle", "yl_right_angle"),
    ),
    FittingMethod.ML_MODEL: (
        ("ML model left angle", "ml_left_angle"),
        ("ML model right angle", "ml_right_angle"),
    ),
}

# Per-method MAE thresholds. ML tight per the validation plan
# (SC69_1 upper bound: 0.16° L / 0.01° R). Geometric fits looser because
# any delta comes from float-ordering artifacts.
_METHOD_MAE_THRESHOLD = {
    FittingMethod.TANGENT_FIT: (0.5, 0.5),
    FittingMethod.CIRCLE_FIT: (0.5, 0.5),
    FittingMethod.ELLIPSE_FIT: (0.5, 0.5),
    FittingMethod.YL_FIT: (0.5, 0.5),
    FittingMethod.ML_MODEL: (0.16, 0.01),
}


_RSS_BYTES_PER_UNIT = 1 if platform.system() == "Darwin" else 1024


def _rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_BYTES_PER_UNIT


def _fmt_mb(n_bytes):
    return "{:.1f} MB".format(n_bytes / (1024 * 1024))


class _MemorySampler:
    """Same contract as the auto-pipeline sampler — see
    ``video_two_pass_integration_test.py`` for the rationale on
    tracemalloc gating."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.t0 = time.monotonic()
        self.samples = []
        if _TRACEMALLOC_ENABLED:
            tracemalloc.start()
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "elapsed_s", "phase", "frame", "rss_peak_bytes",
                "py_current_bytes", "py_peak_bytes",
            ])

    def sample(self, phase, frame):
        if _TRACEMALLOC_ENABLED:
            py_cur, py_peak = tracemalloc.get_traced_memory()
        else:
            py_cur, py_peak = 0, 0
        row = (
            round(time.monotonic() - self.t0, 2),
            phase,
            frame,
            _rss_bytes(),
            py_cur,
            py_peak,
        )
        self.samples.append(row)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def summary(self):
        if not self.samples:
            return "no samples"
        rss_peak = max(r[3] for r in self.samples)
        py_peak = max(r[5] for r in self.samples)
        first_rss = self.samples[0][3]
        last_rss = self.samples[-1][3]
        heap_str = (
            "python heap peak={}; ".format(_fmt_mb(py_peak))
            if _TRACEMALLOC_ENABLED else ""
        )
        return (
            "peak RSS={} (started {}, ended {}); "
            "{}{} samples over {:.1f}s"
        ).format(
            _fmt_mb(rss_peak), _fmt_mb(first_rss), _fmt_mb(last_rss),
            heap_str, len(self.samples),
            self.samples[-1][0],
        )

    def stop(self):
        if _TRACEMALLOC_ENABLED:
            tracemalloc.stop()


def _sampling_progress_cb(sampler, every=50):
    def cb(current, total, phase):
        if current == 1 or current == total or current % every == 0:
            sampler.sample(phase, current)

    return cb


def _paper_results_paths():
    paper_results = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "paper_results")
    )
    sys.path.insert(0, paper_results)
    from config import ISAAC_DIR, RESULTS_DIR
    video_path = os.path.join(ISAAC_DIR, "SC69_1", "raw_data", "SC69_1.avi")
    ref_csv = os.path.join(RESULTS_DIR, "isaac_SC69_1_results.csv")
    return video_path, ref_csv


@pytest.mark.slow
def test_sc69_1_matches_reference(tmp_path):
    video_path, reference_csv = _paper_results_paths()
    if not os.path.isfile(video_path):
        pytest.skip("SC69_1 video not present at {}".format(video_path))
    if not os.path.isfile(reference_csv):
        pytest.skip("reference CSV not present at {}".format(reference_csv))

    ref = pd.read_csv(reference_csv)
    # Reference uses 0-based frame indexing; app emits 1-based
    # (``frame_idx + 1``). The app processes every frame of the video;
    # the reference stops at ``end_sec=42`` in the Isaac config (420
    # frames from a 440-frame video).
    ref_frame_count = len(ref)

    setup = ExperimentalSetup()
    setup.is_video = True
    setup.needle_in_drop = True
    setup.video_skip_interval = 10
    setup.import_files = [video_path]
    # Enable only the methods the reference emits (no polynomial).
    for m in setup.analysis_methods_ca:
        setup.analysis_methods_ca[m] = False
    for m in (FittingMethod.TANGENT_FIT, FittingMethod.CIRCLE_FIT,
              FittingMethod.ELLIPSE_FIT, FittingMethod.YL_FIT,
              FittingMethod.ML_MODEL):
        setup.analysis_methods_ca[m] = True

    out_csv = str(tmp_path / "sc69_1_app.csv")
    writer = UnifiedCsvWriter(out_csv, setup)

    mem_csv = str(tmp_path / "memory_samples.csv")
    sampler = _MemorySampler(mem_csv)
    sampler.sample("start", 0)

    try:
        result = _needle_in_two_pass(
            setup, writer,
            progress=_sampling_progress_cb(sampler, every=50),
            cancel=lambda: False,
        )
    finally:
        gc.collect()
        sampler.sample("end", -1)
        print("\n[memory] {}".format(sampler.summary()))
        print("[memory] samples CSV: {}".format(mem_csv))
        sampler.stop()

    assert result["status"] == "completed"
    assert result["pass2_written"] >= ref_frame_count

    app_full = pd.read_csv(out_csv)
    # App writes 1-based frame_number; reference uses 0-based. Map by
    # ``app.frame_number == ref.frame_number + 1`` and keep only rows
    # the reference also produced.
    app = app_full.copy()
    app["ref_frame_number"] = app["frame_number"] - 1
    app = app[app["ref_frame_number"].isin(ref["frame_number"])]
    app = app.set_index("ref_frame_number").loc[ref["frame_number"].values]
    assert len(app) == len(ref), (
        "app coverage over reference frame range: {} vs {}".format(
            len(app), len(ref)
        )
    )

    for method, pairs in _COLUMN_PAIRS.items():
        left_app_col, left_ref_col = pairs[0]
        right_app_col, right_ref_col = pairs[1]
        left_threshold, right_threshold = _METHOD_MAE_THRESHOLD[method]

        mae_left = _column_mae(app[left_app_col], ref[left_ref_col])
        mae_right = _column_mae(app[right_app_col], ref[right_ref_col])

        assert mae_left <= left_threshold, (
            "{} left MAE {:.4f} exceeds {:.4f}".format(
                method.value, mae_left, left_threshold
            )
        )
        assert mae_right <= right_threshold, (
            "{} right MAE {:.4f} exceeds {:.4f}".format(
                method.value, mae_right, right_threshold
            )
        )


def _column_mae(app_series, ref_series):
    """MAE across rows where BOTH app and reference produced a value."""
    a = pd.to_numeric(pd.Series(app_series).reset_index(drop=True),
                      errors="coerce")
    r = pd.to_numeric(pd.Series(ref_series).reset_index(drop=True),
                      errors="coerce")
    mask = a.notna() & r.notna()
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(a[mask].values - r[mask].values)))
