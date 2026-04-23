import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

from opendrop_ml.modules.contact_angle import (
    _video_helpers,
    video_two_pass_needle_in,
)
from opendrop_ml.modules.core.classes import ExperimentalSetup
from opendrop_ml.utils.config import LEFT_ANGLE, RIGHT_ANGLE
from opendrop_ml.utils.enums import FittingMethod


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_setup(skip=10, methods=None):
    setup = ExperimentalSetup()
    setup.is_video = True
    setup.needle_in_drop = True
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
        frames.append((np.zeros((400, 400, 3), np.uint8), i, meta))
    return frames


def _fake_pass1_return():
    line = np.array([[0.0, 200.0], [400.0, 200.0]])
    cps = np.array([[10.0, 200.0], [20.0, 201.0], [30.0, 199.0]])
    needle_int = np.array([100.0, 50.0])
    crop_for_pass2 = 260  # test-only — asserts pass2 accepts the auto-crop.
    return (line, line, needle_int, needle_int, cps, cps, crop_for_pass2)


def _patch_pipeline(monkeypatch, *, n_frames, analyze_result=None,
                    pass1_return=None):
    """Stand up a mocked _needle_in_two_pass environment.

    `analyze_result` is the ``(contact_angles, error_status)`` tuple
    that ``_analyze_frame_needle_in`` returns each call.
    ``pass1_return`` overrides the Pass-1 output (``None`` = cancelled).
    """
    frames = _fake_frames(n_frames)

    def fake_load_frames(*args, **kwargs):
        return iter(list(frames))

    monkeypatch.setattr(
        video_two_pass_needle_in, "load_frames", fake_load_frames
    )
    monkeypatch.setattr(
        video_two_pass_needle_in, "get_frame_count", lambda path: n_frames
    )
    monkeypatch.setattr(
        video_two_pass_needle_in, "_pass1_calibrate",
        lambda *a, **kw: (pass1_return if pass1_return is not None
                          else _fake_pass1_return()),
    )

    default = (
        {FittingMethod.TANGENT_FIT: {LEFT_ANGLE: 45.0, RIGHT_ANGLE: 46.0}},
        "",
    )
    monkeypatch.setattr(
        video_two_pass_needle_in, "_analyze_frame_needle_in",
        lambda *a, **kw: analyze_result if analyze_result is not None else default,
    )

    monkeypatch.setattr(
        video_two_pass_needle_in, "_load_ml_model", lambda: None
    )
    return frames


# ---------------------------------------------------------------------------
# Cluster-selection helpers
# ---------------------------------------------------------------------------

def test_largest_cluster_picks_highest_count():
    groups = {
        0: np.array([[0, 0], [1, 0]]),
        1: np.array([[10, 10], [11, 10], [12, 10], [13, 10]]),
        2: np.array([[5, 5]]),
    }
    picked = video_two_pass_needle_in._largest_cluster(groups)
    assert len(picked) == 4


def test_largest_cluster_ndarray_passthrough():
    arr = np.array([[1, 2], [3, 4]])
    picked = video_two_pass_needle_in._largest_cluster(arr)
    assert picked is arr or np.array_equal(picked, arr)


def test_rightmost_cluster_picks_max_rightmost_x():
    groups = {
        0: np.array([[0, 0], [5, 0]]),
        1: np.array([[10, 10], [20, 10]]),
        2: np.array([[1, 1], [30, 1]]),
    }
    picked = video_two_pass_needle_in._rightmost_cluster(groups)
    assert picked[-1][0] == 30


def test_leftmost_cluster_picks_min_rightmost_x():
    groups = {
        0: np.array([[0, 0], [5, 0]]),
        1: np.array([[10, 10], [20, 10]]),
        2: np.array([[1, 1], [30, 1]]),
    }
    picked = video_two_pass_needle_in._leftmost_cluster(groups)
    assert max(picked[:, 0]) == 5


def test_derive_auto_crop_median_plus_margin():
    baselines = [
        np.array([[0, 100], [400, 102]]),
        np.array([[0, 110], [400, 108]]),
        np.array([[0, 150], [400, 152]]),  # outlier → median still at 109-ish
    ]
    crop = video_two_pass_needle_in.derive_auto_crop(baselines)
    margin = video_two_pass_needle_in._CROP_MARGIN_BELOW_BASELINE
    # Median of per-baseline mean(y) values is 109 → crop = 109 + margin.
    assert crop == 109 + margin


def test_derive_auto_crop_empty_returns_none():
    assert video_two_pass_needle_in.derive_auto_crop([]) is None
    assert video_two_pass_needle_in.derive_auto_crop([None, None]) is None


def test_create_edge_image_rasterises_and_clips():
    left = np.array([[1, 2], [3, 4], [-1, 5], [100, 100]])  # two out-of-bounds
    right = np.array([[5, 6]])
    img = video_two_pass_needle_in._create_edge_image(left, right, (10, 10))
    assert img.shape == (10, 10)
    assert img[2, 1] == 255
    assert img[4, 3] == 255
    assert img[6, 5] == 255
    # Out-of-bounds points silently dropped → image contains exactly 3 set px.
    assert int(np.sum(img == 255)) == 3


# ---------------------------------------------------------------------------
# compute_pass1_calibration
# ---------------------------------------------------------------------------

def _pass1_result(left_cp, right_cp, left_int, right_int, baseline):
    return {
        "left_CP": np.asarray(left_cp, dtype=float),
        "right_CP": np.asarray(right_cp, dtype=float),
        "left_needle_int": np.asarray(left_int, dtype=float),
        "right_needle_int": np.asarray(right_int, dtype=float),
        "baseline": np.asarray(baseline, dtype=float),
    }


def test_compute_pass1_calibration_happy_path():
    baseline = np.array([[0, 200], [400, 200]])
    results = [
        _pass1_result([10, 200], [300, 201], [50, 40], [350, 41], baseline),
        _pass1_result([12, 200], [302, 201], [51, 40], [349, 41], baseline),
        _pass1_result([11, 200], [301, 201], [50, 40], [350, 41], baseline),
    ]

    (left_line, right_line, left_needle_int, right_needle_int,
     left_cps_arr, right_cps_arr,
     auto_crop) = video_two_pass_needle_in.compute_pass1_calibration(results)

    assert left_cps_arr.shape == (3, 2)
    assert right_cps_arr.shape == (3, 2)
    assert left_line.shape == (2, 2)
    assert right_line.shape == (2, 2)
    assert left_needle_int[0] == pytest.approx(50.0)
    assert right_needle_int[0] == pytest.approx(350.0)
    margin = video_two_pass_needle_in._CROP_MARGIN_BELOW_BASELINE
    assert auto_crop == 200 + margin


def test_compute_pass1_calibration_raises_below_min_successes():
    baseline = np.array([[0, 200], [400, 200]])
    results = [
        _pass1_result([10, 200], [300, 200], [50, 40], [350, 40], baseline),
        _pass1_result([11, 200], [301, 200], [50, 40], [350, 40], baseline),
    ]
    with pytest.raises(RuntimeError, match="Pass 1 calibration failed"):
        video_two_pass_needle_in.compute_pass1_calibration(results)


# ---------------------------------------------------------------------------
# extract_pass2_frame
# ---------------------------------------------------------------------------

def test_extract_pass2_frame_returns_none_on_preprocess_failure(monkeypatch):
    monkeypatch.setattr(
        video_two_pass_needle_in, "_preprocess_pass2",
        lambda frame, crop_value: None,
    )
    line = np.array([[0.0, 15.0], [30.0, 15.0]])
    needle_int = np.array([5.0, 1.0])
    state = video_two_pass_needle_in.extract_pass2_frame(
        frame=np.zeros((100, 100, 3), np.uint8),
        crop_value=400,
        left_line=line,
        right_line=line,
        left_needle_int=needle_int,
        right_needle_int=needle_int,
    )
    assert state is None


def test_extract_pass2_frame_returns_none_on_contour_short(monkeypatch):
    contour = np.column_stack([
        np.arange(30, dtype=float),
        np.linspace(0, 30, 30, dtype=float),
    ])
    monkeypatch.setattr(
        video_two_pass_needle_in, "_preprocess_pass2",
        lambda frame, crop_value: (contour.copy(), contour.copy()),
    )
    monkeypatch.setattr(
        video_two_pass_needle_in, "find_first_intercept",
        lambda contour_, line: -1,
    )
    line = np.array([[0.0, 15.0], [30.0, 15.0]])
    needle_int = np.array([5.0, 1.0])
    state = video_two_pass_needle_in.extract_pass2_frame(
        frame=np.zeros((100, 100, 3), np.uint8),
        crop_value=400,
        left_line=line,
        right_line=line,
        left_needle_int=needle_int,
        right_needle_int=needle_int,
    )
    assert state is None


def test_extract_pass2_frame_returns_populated_state(monkeypatch):
    contour = np.column_stack([
        np.arange(60, dtype=float),
        np.linspace(0, 60, 60, dtype=float),
    ])
    monkeypatch.setattr(
        video_two_pass_needle_in, "_preprocess_pass2",
        lambda frame, crop_value: (contour.copy(), contour.copy()),
    )
    # CP is at index 55 (near bottom of 60-point contour).
    monkeypatch.setattr(
        video_two_pass_needle_in, "find_first_intercept",
        lambda contour_, line: 55,
    )
    # Needle start at index 0.
    monkeypatch.setattr(
        video_two_pass_needle_in, "find_last_intercept",
        lambda contour_, line, max_y=None: 0,
    )

    line = np.array([[0.0, 55.0], [60.0, 55.0]])
    needle_int = np.array([0.0, 0.0])
    state = video_two_pass_needle_in.extract_pass2_frame(
        frame=np.zeros((100, 100, 3), np.uint8),
        crop_value=400,
        left_line=line,
        right_line=line,
        left_needle_int=needle_int,
        right_needle_int=needle_int,
    )
    assert state is not None
    assert set(state.keys()) == {
        "profile", "left_contour", "right_contour",
        "left_cp_idx", "right_cp_idx",
        "left_CP", "right_CP",
        "ml_left_window", "ml_right_window",
    }
    # Profile = left[::-1] (56 pts) stacked with right (56 pts) = 112 rows.
    assert state["profile"].shape == (112, 2)
    assert state["left_cp_idx"] == 55
    # ML window = [CP - _ML_N_DROP : CP + _ML_N_SURFACE], clamped.
    n_drop = video_two_pass_needle_in._ML_N_DROP
    n_surface = video_two_pass_needle_in._ML_N_SURFACE
    expected_window_len = min(60, 55 + n_surface) - max(0, 55 - n_drop)
    assert len(state["ml_left_window"]) == expected_window_len


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

    result = video_two_pass_needle_in._needle_in_two_pass(
        setup, writer, progress, lambda: False
    )

    assert result["status"] == "completed"
    assert result["phase"] == "pass_2"
    assert result["pass2_written"] == 5
    assert [r["frame_number"] for r in writer.rows] == [1, 2, 3, 4, 5]
    pass2_progress = [c for c in progress_calls if c[2] == "Pass 2"]
    assert pass2_progress[-1] == (5, 5, "Pass 2")


def test_return_dict_carries_needle_x_values(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=3)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])

    result = video_two_pass_needle_in._needle_in_two_pass(
        setup, FakeWriter(), lambda *a: None, lambda: False
    )

    assert result["left_needle_x"] == pytest.approx(100.0)
    assert result["right_needle_x"] == pytest.approx(100.0)


def test_cancel_mid_pass1_returns_cancelled(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=3, pass1_return=None)
    monkeypatch.setattr(
        video_two_pass_needle_in, "_pass1_calibrate", lambda *a, **kw: None
    )

    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()

    result = video_two_pass_needle_in._needle_in_two_pass(
        setup, writer, lambda *a: None, lambda: True
    )

    assert result["status"] == "cancelled"
    assert result["phase"] == "pass_1"
    assert result["pass2_written"] == 0
    assert writer.rows == []
    assert result["left_contact_line"] is None
    assert result["left_needle_x"] is None
    assert result["right_needle_x"] is None


def test_cancel_mid_pass2_keeps_partial_rows(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=10)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()

    call_counter = {"n": 0}

    def cancel():
        call_counter["n"] += 1
        return call_counter["n"] > 3

    result = video_two_pass_needle_in._needle_in_two_pass(
        setup, writer, lambda *a: None, cancel
    )

    assert result["status"] == "cancelled"
    assert result["pass2_written"] == 3
    assert [r["frame_number"] for r in writer.rows] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Error / per-method isolation
# ---------------------------------------------------------------------------

def test_contour_short_produces_row_with_error_status(monkeypatch):
    _patch_pipeline(
        monkeypatch, n_frames=2,
        analyze_result=({}, "contour_short"),
    )
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()

    video_two_pass_needle_in._needle_in_two_pass(
        setup, writer, lambda *a: None, lambda: False
    )

    for row in writer.rows:
        assert row["error_status"] == "contour_short"
        assert row["contact_angles"] == {}


def test_per_fit_failure_isolated(monkeypatch):
    """A per-method exception leaves other methods populated. The _try_*
    helpers live in _video_helpers; patching happens there."""
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

    # Build a minimal fake Pass-2 state so _analyze_frame_needle_in can
    # reach the fit-dispatch branch. Monkeypatch the pieces it calls.
    contour = np.column_stack([
        np.arange(30, dtype=float),
        np.linspace(0, 30, 30, dtype=float),
    ])
    monkeypatch.setattr(
        video_two_pass_needle_in, "_preprocess_pass2",
        lambda frame, crop_value: (contour.copy(), contour.copy()),
    )
    monkeypatch.setattr(
        video_two_pass_needle_in, "find_first_intercept",
        lambda contour_, line: len(contour_) - 1,
    )
    monkeypatch.setattr(
        video_two_pass_needle_in, "find_last_intercept",
        lambda contour_, line, max_y=None: 0,
    )

    line = np.array([[0.0, 15.0], [30.0, 15.0]])
    needle_int = np.array([5.0, 1.0])
    angles, error = video_two_pass_needle_in._analyze_frame_needle_in(
        frame=np.zeros((100, 100, 3), np.uint8),
        left_line=line,
        right_line=line,
        left_needle_int=needle_int,
        right_needle_int=needle_int,
        enabled_methods={
            FittingMethod.TANGENT_FIT,
            FittingMethod.CIRCLE_FIT,
            FittingMethod.YL_FIT,
        },
        ml_model=None,
        crop_value=400,
    )

    assert error == ""
    assert FittingMethod.TANGENT_FIT in angles
    assert FittingMethod.CIRCLE_FIT in angles
    assert FittingMethod.YL_FIT not in angles  # raised → silently dropped


# ---------------------------------------------------------------------------
# ML model loading
# ---------------------------------------------------------------------------

def test_ml_model_loaded_once_when_enabled(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=10)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT, FittingMethod.ML_MODEL])

    load_calls = {"n": 0}

    def fake_load():
        load_calls["n"] += 1
        return object()

    monkeypatch.setattr(video_two_pass_needle_in, "_load_ml_model", fake_load)

    video_two_pass_needle_in._needle_in_two_pass(
        setup, FakeWriter(), lambda *a: None, lambda: False
    )

    assert load_calls["n"] == 1


def test_ml_model_not_loaded_when_disabled(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=5)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])

    load_calls = {"n": 0}

    def fake_load():
        load_calls["n"] += 1
        return object()

    monkeypatch.setattr(video_two_pass_needle_in, "_load_ml_model", fake_load)

    video_two_pass_needle_in._needle_in_two_pass(
        setup, FakeWriter(), lambda *a: None, lambda: False
    )

    assert load_calls["n"] == 0


def test_ml_skipped_when_window_too_short():
    """_predict_ml_needle_in returns None when either window != 50 points."""
    short_left = np.zeros((30, 2), dtype=float)
    full_right = np.zeros((50, 2), dtype=float)

    result = video_two_pass_needle_in._predict_ml_needle_in(
        left_contour=short_left,
        right_contour=full_right,
        left_cp_idx=20,
        right_cp_idx=40,
        ml_model=object(),
    )
    assert result is None


# ---------------------------------------------------------------------------
# Timestamp / filepath handling
# ---------------------------------------------------------------------------

def test_timestamp_computed_for_video(monkeypatch):
    frames = _fake_frames(3, with_fps=True)

    def fake_load(*a, **kw):
        return iter(list(frames))

    monkeypatch.setattr(video_two_pass_needle_in, "load_frames", fake_load)
    monkeypatch.setattr(
        video_two_pass_needle_in, "get_frame_count", lambda p: 3
    )
    monkeypatch.setattr(
        video_two_pass_needle_in, "_pass1_calibrate",
        lambda *a, **kw: _fake_pass1_return(),
    )
    monkeypatch.setattr(
        video_two_pass_needle_in, "_analyze_frame_needle_in",
        lambda *a, **kw: ({}, ""),
    )
    monkeypatch.setattr(
        video_two_pass_needle_in, "_load_ml_model", lambda: None
    )

    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()

    video_two_pass_needle_in._needle_in_two_pass(
        setup, writer, lambda *a: None, lambda: False
    )

    assert writer.rows[0]["timestamp_s"] == pytest.approx(0.0)
    assert writer.rows[1]["timestamp_s"] == pytest.approx(1 / 30)
    assert writer.rows[2]["timestamp_s"] == pytest.approx(2 / 30)


def test_filepath_is_image_path_for_directory(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=2)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()

    video_two_pass_needle_in._needle_in_two_pass(
        setup, writer, lambda *a: None, lambda: False
    )

    assert writer.rows[0]["filepath"] == os.path.join("/fake/source", "f0.png")
    assert writer.rows[1]["filepath"] == os.path.join("/fake/source", "f1.png")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
