import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

from opendrop_ml.modules.contact_angle import _video_helpers, video_two_pass
from opendrop_ml.modules.core.classes import ExperimentalSetup
from opendrop_ml.utils.config import LEFT_ANGLE, RIGHT_ANGLE
from opendrop_ml.utils.enums import FittingMethod


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

def _make_setup(skip=10, methods=None):
    setup = ExperimentalSetup()
    setup.is_video = True
    setup.video_skip_interval = skip
    setup.import_files = ["/fake/source"]
    for m in setup.analysis_methods_ca:
        setup.analysis_methods_ca[m] = False
    if methods:
        for m in methods:
            setup.analysis_methods_ca[m] = True
    return setup


class FakeWriter:
    def __init__(self):
        self.rows = []

    def write_row(self, *, frame_number, timestamp_s, filepath,
                  error_status, contact_angles):
        self.rows.append({
            "frame_number": frame_number,
            "timestamp_s": timestamp_s,
            "filepath": filepath,
            "error_status": error_status,
            "contact_angles": contact_angles,
        })


def _fake_frames(n, with_fps=False):
    frames = []
    for i in range(n):
        meta = {"fps": 30.0} if with_fps else {"filename": "f{}.png".format(i)}
        frames.append((np.zeros((10, 10, 3), np.uint8), i, meta))
    return frames


def _fake_pass1_return():
    line = np.array([[0.0, 50.0], [100.0, 50.0]])
    cps = np.array([[10.0, 50.0], [20.0, 51.0], [30.0, 49.0]])
    return line, line, cps, cps


def _patch_pipeline(monkeypatch, *, n_frames, analyze_result=None,
                     pass1_return=None):
    """Stand up a mocked _auto_two_pass environment.

    `analyze_result` is the ``(contact_angles, error_status)`` tuple that
    `_analyze_frame` returns each call. `pass1_return` overrides the
    Pass-1 calibration output (set to `None` to let Pass 1 bail as
    cancelled).
    """
    frames = _fake_frames(n_frames)

    # load_frames is called twice (Pass 1 + Pass 2); return a fresh iterator
    # each time so each pass sees the full list.
    def fake_load_frames(*args, **kwargs):
        return iter(list(frames))

    monkeypatch.setattr(video_two_pass, "load_frames", fake_load_frames)
    monkeypatch.setattr(
        video_two_pass, "get_frame_count", lambda path: n_frames
    )

    monkeypatch.setattr(
        video_two_pass, "_pass1_calibrate",
        lambda *a, **kw: (pass1_return if pass1_return is not None
                          else _fake_pass1_return()),
    )

    default = (
        {FittingMethod.TANGENT_FIT: {LEFT_ANGLE: 45.0, RIGHT_ANGLE: 46.0}},
        "",
    )
    monkeypatch.setattr(
        video_two_pass, "_analyze_frame",
        lambda *a, **kw: analyze_result if analyze_result is not None else default,
    )

    # Default: no ML model loading — tests opt in via method toggle.
    monkeypatch.setattr(video_two_pass, "_load_ml_model", lambda: None)
    return frames


# ---------------------------------------------------------------------------
# Pass-2 streaming / progress / cancel
# ---------------------------------------------------------------------------

def test_two_pass_streams_one_row_per_frame(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=5)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()
    progress_calls = []

    def progress(current, total, label):
        progress_calls.append((current, total, label))

    result = video_two_pass._auto_two_pass(
        setup, writer, progress, lambda: False
    )

    assert result["status"] == "completed"
    assert result["phase"] == "pass_2"
    assert result["pass2_written"] == 5
    assert [r["frame_number"] for r in writer.rows] == [1, 2, 3, 4, 5]
    # Pass-2 progress reports monotonic current=total=5 on the final call.
    pass2_progress = [c for c in progress_calls if c[2] == "Pass 2"]
    assert pass2_progress[-1] == (5, 5, "Pass 2")


def test_progress_reports_both_pass_labels(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=3)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])

    # Let the real _pass1_calibrate run so we get real Pass 1 progress.
    monkeypatch.setattr(
        video_two_pass, "_pass1_calibrate",
        lambda *a, **kw: _fake_pass1_return(),
    )

    labels_seen = []

    def progress(current, total, label):
        labels_seen.append(label)

    video_two_pass._auto_two_pass(
        setup, FakeWriter(), progress, lambda: False
    )

    assert "Pass 2" in labels_seen  # Pass 1 is stubbed so only Pass 2 progresses


def test_cancel_mid_pass1_returns_cancelled(monkeypatch):
    _patch_pipeline(
        monkeypatch, n_frames=3, pass1_return=None,
    )

    # Override _pass1_calibrate to simulate cancel → return None.
    monkeypatch.setattr(
        video_two_pass, "_pass1_calibrate", lambda *a, **kw: None
    )

    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()

    result = video_two_pass._auto_two_pass(
        setup, writer, lambda *a: None, lambda: True
    )

    assert result["status"] == "cancelled"
    assert result["phase"] == "pass_1"
    assert result["pass2_written"] == 0
    assert writer.rows == []
    assert result["left_contact_line"] is None


def test_cancel_mid_pass2_keeps_partial_rows(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=10)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()

    # Cancel after 3 Pass-2 frames have been written.
    call_counter = {"n": 0}

    def cancel():
        call_counter["n"] += 1
        # Pass 1 is stubbed → the first cancel() check the real code does is
        # at the top of each Pass 2 iteration. Let 3 frames through then cancel.
        return call_counter["n"] > 3

    result = video_two_pass._auto_two_pass(
        setup, writer, lambda *a: None, cancel
    )

    assert result["status"] == "cancelled"
    assert result["pass2_written"] == 3
    assert [r["frame_number"] for r in writer.rows] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Error / per-method isolation paths
# ---------------------------------------------------------------------------

def test_contour_short_produces_row_with_error_status(monkeypatch):
    _patch_pipeline(
        monkeypatch, n_frames=2,
        analyze_result=({}, "contour_short"),
    )
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()

    video_two_pass._auto_two_pass(
        setup, writer, lambda *a: None, lambda: False
    )

    for row in writer.rows:
        assert row["error_status"] == "contour_short"
        assert row["contact_angles"] == {}


def test_per_fit_failure_isolated(monkeypatch):
    """A per-method exception leaves other methods populated."""
    # 20-point contour so the profile slice comfortably exceeds the
    # profile_too_short threshold.
    contour = np.column_stack([np.arange(20, dtype=float),
                               np.zeros(20, dtype=float)])
    monkeypatch.setattr(
        video_two_pass, "isolate_contour",
        lambda frame, DISPLAY=False, blur=True: (None, contour, None, {}),
    )
    monkeypatch.setattr(
        video_two_pass, "CPID",
        lambda contour, display=False: np.array([[0.0, 0.0], [19.0, 0.0]]),
    )
    monkeypatch.setattr(
        video_two_pass, "find_first_intercept",
        lambda contour_half, line: 2,
    )
    monkeypatch.setattr(
        _video_helpers, "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    def boom(profile, display=False):
        raise RuntimeError("synthetic yl failure")

    monkeypatch.setattr(_video_helpers, "yl_fit", boom)
    monkeypatch.setattr(
        _video_helpers, "circular_fit",
        lambda profile, display=False: ((40.0, 41.0), None, None, None, {}, {}),
    )

    line = np.array([[0.0, 2.0], [10.0, 2.0]])
    angles, error = video_two_pass._analyze_frame(
        frame=np.zeros((10, 10, 3), np.uint8),
        left_line=line,
        right_line=line,
        enabled_methods={
            FittingMethod.TANGENT_FIT,
            FittingMethod.CIRCLE_FIT,
            FittingMethod.YL_FIT,
        },
        ml_model=None,
    )

    assert error == ""
    assert FittingMethod.TANGENT_FIT in angles
    assert FittingMethod.CIRCLE_FIT in angles
    # YL_FIT raised → silently dropped (writer emits NaN for those columns).
    assert FittingMethod.YL_FIT not in angles


# ---------------------------------------------------------------------------
# ML model loading
# ---------------------------------------------------------------------------

def test_ml_model_loaded_once_when_enabled(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=10)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT, FittingMethod.ML_MODEL])

    load_calls = {"n": 0}

    def fake_load():
        load_calls["n"] += 1
        return object()  # opaque sentinel model

    monkeypatch.setattr(video_two_pass, "_load_ml_model", fake_load)

    video_two_pass._auto_two_pass(
        setup, FakeWriter(), lambda *a: None, lambda: False
    )

    assert load_calls["n"] == 1


def test_ml_model_not_loaded_when_disabled(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=5)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])  # no ML_MODEL

    load_calls = {"n": 0}

    def fake_load():
        load_calls["n"] += 1
        return object()

    monkeypatch.setattr(video_two_pass, "_load_ml_model", fake_load)

    video_two_pass._auto_two_pass(
        setup, FakeWriter(), lambda *a: None, lambda: False
    )

    assert load_calls["n"] == 0


# ---------------------------------------------------------------------------
# Timestamp / filepath handling
# ---------------------------------------------------------------------------

def test_timestamp_nan_for_image_directory(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=2)  # default meta has `filename`
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()

    video_two_pass._auto_two_pass(
        setup, writer, lambda *a: None, lambda: False
    )

    for row in writer.rows:
        assert row["timestamp_s"] != row["timestamp_s"]  # NaN self-inequality


def test_timestamp_computed_for_video(monkeypatch):
    # Override load_frames to yield video-style metadata with fps.
    frames = _fake_frames(3, with_fps=True)

    def fake_load(*a, **kw):
        return iter(list(frames))

    monkeypatch.setattr(video_two_pass, "load_frames", fake_load)
    monkeypatch.setattr(video_two_pass, "get_frame_count", lambda p: 3)
    monkeypatch.setattr(
        video_two_pass, "_pass1_calibrate",
        lambda *a, **kw: _fake_pass1_return(),
    )
    monkeypatch.setattr(
        video_two_pass, "_analyze_frame",
        lambda *a, **kw: ({}, ""),
    )
    monkeypatch.setattr(video_two_pass, "_load_ml_model", lambda: None)

    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()

    video_two_pass._auto_two_pass(
        setup, writer, lambda *a: None, lambda: False
    )

    # fps=30, frame_numbers = 0,1,2 from the fake frames; expect 0, 1/30, 2/30.
    assert writer.rows[0]["timestamp_s"] == pytest.approx(0.0)
    assert writer.rows[1]["timestamp_s"] == pytest.approx(1 / 30)
    assert writer.rows[2]["timestamp_s"] == pytest.approx(2 / 30)


def test_filepath_is_image_path_for_directory(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=2)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()

    video_two_pass._auto_two_pass(
        setup, writer, lambda *a: None, lambda: False
    )

    assert writer.rows[0]["filepath"] == os.path.join("/fake/source", "f0.png")
    assert writer.rows[1]["filepath"] == os.path.join("/fake/source", "f1.png")


def test_return_dict_carries_none_needle_x_values(monkeypatch):
    """Auto pipeline emits the needle-x keys as None so Phase I's
    figure generator can branch on presence uniformly."""
    _patch_pipeline(monkeypatch, n_frames=2)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])

    completed = video_two_pass._auto_two_pass(
        setup, FakeWriter(), lambda *a: None, lambda: False
    )
    assert completed["left_needle_x"] is None
    assert completed["right_needle_x"] is None

    # Also verify the cancelled Pass-1 branch emits them.
    monkeypatch.setattr(
        video_two_pass, "_pass1_calibrate", lambda *a, **kw: None
    )
    cancelled = video_two_pass._auto_two_pass(
        setup, FakeWriter(), lambda *a: None, lambda: True
    )
    assert cancelled["left_needle_x"] is None
    assert cancelled["right_needle_x"] is None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
