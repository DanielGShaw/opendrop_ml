"""Unit tests for the single-pass tilted-surface video pipeline.

Mocks per-frame helpers (correct_tilt, isolate_contour, CPID,
touches_border, the _try_* fit wrappers, and the ML loader/predictor)
so the per-frame control flow is exercised without TensorFlow or real
images. End-to-end parity vs the Kristina reference is in
``video_single_pass_tilted_integration_test.py``.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from opendrop_ml.modules.contact_angle import (  # noqa: E402
    _video_helpers,
    video_single_pass_tilted,
)
from opendrop_ml.modules.core.classes import ExperimentalSetup  # noqa: E402
from opendrop_ml.utils.config import LEFT_ANGLE, RIGHT_ANGLE  # noqa: E402
from opendrop_ml.utils.enums import FittingMethod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_setup(methods=None):
    setup = ExperimentalSetup()
    setup.is_video = True
    setup.surface_tilted = True
    setup.needle_in_drop = False
    setup.import_files = ["/fake/source"]
    for m in setup.analysis_methods_ca:
        setup.analysis_methods_ca[m] = False
    if methods:
        for m in methods:
            setup.analysis_methods_ca[m] = True
    return setup


class FakeWriter:
    """Captures write_row kwargs including the new detected_tilt_angle."""

    def __init__(self):
        self.rows = []

    def write_row(
        self,
        *,
        frame_number,
        timestamp_s,
        filepath,
        error_status,
        contact_angles=None,
        detected_tilt_angle=math.nan,
    ):
        self.rows.append(
            {
                "frame_number": frame_number,
                "timestamp_s": timestamp_s,
                "filepath": filepath,
                "error_status": error_status,
                "contact_angles": contact_angles or {},
                "detected_tilt_angle": detected_tilt_angle,
            }
        )


def _fake_frames(n, with_fps=False, shape=(400, 400, 3)):
    frames = []
    for i in range(n):
        meta = {"fps": 30.0} if with_fps else {"filename": "f{}.png".format(i)}
        frames.append((np.zeros(shape, np.uint8), i, meta))
    return frames


def _good_contour(npts=60):
    """A monotonic contour with two interior contact points."""
    return np.column_stack(
        [
            np.arange(npts, dtype=float),
            np.linspace(0, npts, npts, dtype=float),
        ]
    )


def _patch_pipeline(
    monkeypatch,
    *,
    n_frames,
    tilt_angle=60.0,
    cps=None,
    touches=False,
    isolate_raises=False,
    cpid_raises=False,
    with_fps=False,
):
    """Stand up a mocked single-pass-with-tilt environment.

    `cps` defaults to two interior points of the canonical contour.
    """
    contour = _good_contour(60)
    if cps is None:
        cps = np.array([contour[10], contour[50]], dtype=float)

    frames = _fake_frames(n_frames, with_fps=with_fps)

    monkeypatch.setattr(
        video_single_pass_tilted,
        "load_frames",
        lambda *a, **kw: iter(list(frames)),
    )
    monkeypatch.setattr(
        video_single_pass_tilted, "get_frame_count", lambda path: n_frames
    )
    # Capture the fixture's tilt_angle so the lambda can reuse it even
    # though its own `tilt_angle` kwarg shadows the outer binding.
    _fixture_tilt = tilt_angle
    monkeypatch.setattr(
        video_single_pass_tilted,
        "correct_tilt",
        lambda img, tilt_angle=None, crop=True: (
            img,
            tilt_angle if tilt_angle is not None else _fixture_tilt,
        ),
    )

    def fake_isolate(img, DISPLAY=False, blur=True):
        if isolate_raises:
            raise RuntimeError("synthetic isolate failure")
        return img, contour.copy(), None, {}

    monkeypatch.setattr(video_single_pass_tilted, "isolate_contour", fake_isolate)
    monkeypatch.setattr(
        video_single_pass_tilted,
        "touches_border",
        lambda c, shape, margin=2: touches,
    )

    def fake_cpid(c, display=False):
        if cpid_raises:
            raise RuntimeError("synthetic cpid failure")
        return cps.copy()

    monkeypatch.setattr(video_single_pass_tilted, "CPID", fake_cpid)
    monkeypatch.setattr(video_single_pass_tilted, "_load_ml_model", lambda: None)
    return frames, contour, cps


# ---------------------------------------------------------------------------
# _extract_profile_between_cps
# ---------------------------------------------------------------------------


def test_extract_profile_between_cps_happy_path():
    contour = _good_contour(60)
    cps = np.array([contour[10], contour[50]], dtype=float)
    profile, idx_l, idx_r = video_single_pass_tilted._extract_profile_between_cps(
        contour, cps
    )
    assert idx_l == 10
    assert idx_r == 50
    assert profile.shape == (41, 2)
    assert np.array_equal(profile[0], contour[10])
    assert np.array_equal(profile[-1], contour[50])


def test_extract_profile_between_cps_reversed_contour():
    """When CPs come back in right-then-left order, indices swap."""
    contour = _good_contour(60)
    cps = np.array([contour[50], contour[10]], dtype=float)
    profile, idx_l, idx_r = video_single_pass_tilted._extract_profile_between_cps(
        contour, cps
    )
    assert idx_l == 10
    assert idx_r == 50
    assert profile.shape == (41, 2)


def test_extract_profile_between_cps_missing_cp_returns_none():
    """If a CP isn't in the contour, return (None, None, None)."""
    contour = _good_contour(60)
    cps = np.array([[999.0, 999.0], contour[50]], dtype=float)
    profile, idx_l, idx_r = video_single_pass_tilted._extract_profile_between_cps(
        contour, cps
    )
    assert profile is None
    assert idx_l is None
    assert idx_r is None


# ---------------------------------------------------------------------------
# Per-frame happy path
# ---------------------------------------------------------------------------


def test_happy_path_writes_one_row_per_frame_with_angles(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=3)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])

    # Patch the underlying fit so the angles arrive without invoking
    # the real polynomial_fit code path.
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    writer = FakeWriter()
    result = video_single_pass_tilted._single_pass_with_tilt(
        setup, writer, lambda *a: None, lambda: False
    )

    assert len(writer.rows) == 3
    assert [r["frame_number"] for r in writer.rows] == [1, 2, 3]
    for row in writer.rows:
        assert row["error_status"] == ""
        assert row["detected_tilt_angle"] == pytest.approx(60.0)
        ca = row["contact_angles"]
        assert ca[FittingMethod.TANGENT_FIT][LEFT_ANGLE] == pytest.approx(45.0)
        assert ca[FittingMethod.TANGENT_FIT][RIGHT_ANGLE] == pytest.approx(46.0)

    assert result["status"] == "completed"
    assert result["phase"] == "single_pass"
    assert result["frames_processed"] == 3
    assert result["frames_skipped"] == 0


def test_happy_path_progress_callback_called_per_frame(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=4)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    progress_calls = []
    video_single_pass_tilted._single_pass_with_tilt(
        setup,
        FakeWriter(),
        lambda *a: progress_calls.append(a),
        lambda: False,
    )

    assert len(progress_calls) == 4
    assert progress_calls[-1] == (4, 4, "Processing")


# ---------------------------------------------------------------------------
# touches_border skip
# ---------------------------------------------------------------------------


def test_touches_border_skip_writes_drop_off_edge_row(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=2, touches=True, tilt_angle=59.5)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()

    result = video_single_pass_tilted._single_pass_with_tilt(
        setup, writer, lambda *a: None, lambda: False
    )

    assert len(writer.rows) == 2
    for row in writer.rows:
        assert row["error_status"] == "drop_off_edge"
        assert row["contact_angles"] == {}
        # detected_tilt_angle still populated even when the drop is skipped.
        assert row["detected_tilt_angle"] == pytest.approx(59.5)
    assert result["frames_skipped"] == 2
    assert result["frames_processed"] == 0


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_isolate_failure_writes_isolate_failed_row(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=2, isolate_raises=True)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()

    video_single_pass_tilted._single_pass_with_tilt(
        setup, writer, lambda *a: None, lambda: False
    )

    assert len(writer.rows) == 2
    for row in writer.rows:
        assert row["error_status"] == "isolate_failed"
        assert row["contact_angles"] == {}


def test_cpid_failure_writes_cpid_failed_row(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=2, cpid_raises=True)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()

    video_single_pass_tilted._single_pass_with_tilt(
        setup, writer, lambda *a: None, lambda: False
    )

    for row in writer.rows:
        assert row["error_status"] == "cpid_failed"


def test_per_fit_failure_isolated(monkeypatch):
    """One fit raising leaves the others populated."""
    _patch_pipeline(monkeypatch, n_frames=1)
    setup = _make_setup(
        methods=[
            FittingMethod.TANGENT_FIT,
            FittingMethod.CIRCLE_FIT,
            FittingMethod.YL_FIT,
        ]
    )
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )
    monkeypatch.setattr(
        _video_helpers,
        "circular_fit",
        lambda profile, display=False: ((40.0, 41.0), None, None, None, {}, {}),
    )

    def boom(profile, display=False):
        raise RuntimeError("synthetic yl failure")

    monkeypatch.setattr(_video_helpers, "yl_fit", boom)

    writer = FakeWriter()
    video_single_pass_tilted._single_pass_with_tilt(
        setup, writer, lambda *a: None, lambda: False
    )

    angles = writer.rows[0]["contact_angles"]
    assert FittingMethod.TANGENT_FIT in angles
    assert FittingMethod.CIRCLE_FIT in angles
    assert FittingMethod.YL_FIT not in angles  # raised → silently dropped


# ---------------------------------------------------------------------------
# ML model loading + short-window skip
# ---------------------------------------------------------------------------


def test_ml_model_loaded_once_when_enabled(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=5)
    setup = _make_setup(methods=[FittingMethod.ML_MODEL])

    load_calls = {"n": 0}

    def fake_load():
        load_calls["n"] += 1
        return object()

    monkeypatch.setattr(video_single_pass_tilted, "_load_ml_model", fake_load)
    # Stub out the ML predict path so it doesn't hit prepare_for_model /
    # experimental_pred with the canonical fake contour.
    monkeypatch.setattr(
        video_single_pass_tilted,
        "_try_ml_single_pass",
        lambda *a, **kw: None,
    )

    video_single_pass_tilted._single_pass_with_tilt(
        setup, FakeWriter(), lambda *a: None, lambda: False
    )

    assert load_calls["n"] == 1


def test_ml_not_loaded_when_disabled(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=3)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    load_calls = {"n": 0}

    def fake_load():
        load_calls["n"] += 1
        return object()

    monkeypatch.setattr(video_single_pass_tilted, "_load_ml_model", fake_load)

    video_single_pass_tilted._single_pass_with_tilt(
        setup, FakeWriter(), lambda *a: None, lambda: False
    )
    assert load_calls["n"] == 0


def test_ml_failure_does_not_block_other_fits(monkeypatch):
    """If ML preprocessing/prediction raises, the row still writes with
    the surviving (non-ML) angles. Mirrors evaluate_kristina's per-method
    try/except isolation."""
    _patch_pipeline(monkeypatch, n_frames=1)
    setup = _make_setup(
        methods=[
            FittingMethod.TANGENT_FIT,
            FittingMethod.ML_MODEL,
        ]
    )
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )
    monkeypatch.setattr(video_single_pass_tilted, "_load_ml_model", lambda: object())

    def boom(*a, **kw):
        raise RuntimeError("synthetic ML failure")

    monkeypatch.setattr(video_single_pass_tilted, "prepare_for_model", boom)

    writer = FakeWriter()
    video_single_pass_tilted._single_pass_with_tilt(
        setup, writer, lambda *a: None, lambda: False
    )

    angles = writer.rows[0]["contact_angles"]
    assert FittingMethod.TANGENT_FIT in angles
    assert FittingMethod.ML_MODEL not in angles
    assert writer.rows[0]["error_status"] == ""


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def test_cancel_mid_loop_keeps_partial_rows(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=10)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    counter = {"n": 0}

    def cancel():
        counter["n"] += 1
        return counter["n"] > 3

    writer = FakeWriter()
    result = video_single_pass_tilted._single_pass_with_tilt(
        setup, writer, lambda *a: None, cancel
    )

    assert result["status"] == "cancelled"
    assert [r["frame_number"] for r in writer.rows] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Timestamp / filepath
# ---------------------------------------------------------------------------


def test_timestamp_computed_for_video(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=3, with_fps=True)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    writer = FakeWriter()
    video_single_pass_tilted._single_pass_with_tilt(
        setup, writer, lambda *a: None, lambda: False
    )

    assert writer.rows[0]["timestamp_s"] == pytest.approx(0.0)
    assert writer.rows[1]["timestamp_s"] == pytest.approx(1 / 30)
    assert writer.rows[2]["timestamp_s"] == pytest.approx(2 / 30)


def test_filepath_is_image_path_for_directory(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=2)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    writer = FakeWriter()
    video_single_pass_tilted._single_pass_with_tilt(
        setup, writer, lambda *a: None, lambda: False
    )

    assert writer.rows[0]["filepath"] == os.path.join("/fake/source", "f0.png")
    assert writer.rows[1]["filepath"] == os.path.join("/fake/source", "f1.png")


# ---------------------------------------------------------------------------
# Return-dict shape
# ---------------------------------------------------------------------------


def test_return_dict_shape_and_median_tilt(monkeypatch):
    """Return dict carries every documented key + median_tilt_angle is the
    np.median of per-frame detected angles (NaN-filtered)."""
    _patch_pipeline(monkeypatch, n_frames=3, tilt_angle=60.0)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    result = video_single_pass_tilted._single_pass_with_tilt(
        setup, FakeWriter(), lambda *a: None, lambda: False
    )

    expected_keys = {
        "status",
        "phase",
        "frames_processed",
        "frames_skipped",
        "median_tilt_angle",
        "rotated_first_frame",
        "left_contact_line",
        "right_contact_line",
        "pass1_cps",
        "left_needle_x",
        "right_needle_x",
    }
    assert set(result.keys()) == expected_keys
    assert result["median_tilt_angle"] == pytest.approx(60.0)
    # Backward-compat: needle_x always None for non-needle pipelines.
    assert result["left_needle_x"] is None
    assert result["right_needle_x"] is None


def test_return_dict_carries_rotated_first_frame(monkeypatch):
    """rotated_first_frame is a copy of the first successfully corrected
    frame (consumed by the cp_scatter renderer)."""
    _patch_pipeline(monkeypatch, n_frames=2)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    result = video_single_pass_tilted._single_pass_with_tilt(
        setup, FakeWriter(), lambda *a: None, lambda: False
    )

    assert result["rotated_first_frame"] is not None
    assert result["rotated_first_frame"].shape == (400, 400, 3)


def test_return_dict_contact_lines_populated_when_enough_cps(monkeypatch):
    """≥3 frames → fit_contact_line gets called, result is a 2x2 list."""
    _patch_pipeline(monkeypatch, n_frames=4)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    result = video_single_pass_tilted._single_pass_with_tilt(
        setup, FakeWriter(), lambda *a: None, lambda: False
    )

    assert result["left_contact_line"] is not None
    assert result["right_contact_line"] is not None
    assert len(result["left_contact_line"]) == 2  # two endpoints
    assert len(result["right_contact_line"]) == 2


# ---------------------------------------------------------------------------
# Misc / defensive
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Once-per-video tilt (``_USE_PER_VIDEO_TILT``)
# ---------------------------------------------------------------------------


def test_derive_shared_tilt_medians_non_zero_samples(monkeypatch):
    """_derive_shared_tilt filters zeros before taking the median and
    reports the success rate over the full sample list."""
    frames = _fake_frames(4)
    monkeypatch.setattr(
        video_single_pass_tilted,
        "load_frames",
        lambda *a, **kw: iter(list(frames)),
    )
    angles = [60.0, 0.0, 60.5, 59.5]
    angle_iter = iter(angles)
    monkeypatch.setattr(
        video_single_pass_tilted,
        "detect_tilt_angle",
        lambda img: next(angle_iter),
    )

    shared, samples, success_rate = video_single_pass_tilted._derive_shared_tilt(
        "/fake/source",
        skip=10,
        total_samples=4,
        progress=lambda *a: None,
        cancel=lambda: False,
    )

    assert shared == pytest.approx(60.0)  # median of 60.0, 60.5, 59.5
    assert samples == angles
    assert success_rate == pytest.approx(0.75)


def test_per_video_tilt_applied_to_every_frame(monkeypatch):
    """Stage 2 passes the shared median to correct_tilt on every frame
    and populates detected_tilt_angle with it."""
    _patch_pipeline(monkeypatch, n_frames=3, tilt_angle=59.8)
    monkeypatch.setattr(video_single_pass_tilted, "_USE_PER_VIDEO_TILT", True)
    monkeypatch.setattr(
        video_single_pass_tilted,
        "_derive_shared_tilt",
        lambda *a, **kw: (59.8, [60.0, 60.0, 59.5], 1.0),
    )

    correct_tilt_calls = []

    def spy_correct_tilt(img, tilt_angle=None, crop=True):
        correct_tilt_calls.append(tilt_angle)
        return img, tilt_angle if tilt_angle is not None else 0.0

    monkeypatch.setattr(video_single_pass_tilted, "correct_tilt", spy_correct_tilt)

    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    writer = FakeWriter()
    video_single_pass_tilted._single_pass_with_tilt(
        setup,
        writer,
        lambda *a: None,
        lambda: False,
    )

    assert len(correct_tilt_calls) == 3
    assert all(v == 59.8 for v in correct_tilt_calls)
    assert all(row["detected_tilt_angle"] == pytest.approx(59.8) for row in writer.rows)


def test_per_video_fallback_on_zero_success_rate(monkeypatch, capsys):
    """When _derive_shared_tilt reports success_rate==0 (all samples
    returned 0.0), Stage 2 falls back to per-frame auto-detect and
    prints a warning."""
    _patch_pipeline(monkeypatch, n_frames=2, tilt_angle=60.0)
    monkeypatch.setattr(video_single_pass_tilted, "_USE_PER_VIDEO_TILT", True)
    monkeypatch.setattr(
        video_single_pass_tilted,
        "_derive_shared_tilt",
        lambda *a, **kw: (float("nan"), [0.0, 0.0], 0.0),
    )

    tilt_kwargs = []

    def spy_correct_tilt(img, tilt_angle=None, crop=True):
        tilt_kwargs.append(tilt_angle)
        return img, 60.0

    monkeypatch.setattr(video_single_pass_tilted, "correct_tilt", spy_correct_tilt)

    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    video_single_pass_tilted._single_pass_with_tilt(
        setup,
        FakeWriter(),
        lambda *a: None,
        lambda: False,
    )

    assert tilt_kwargs == [None, None]  # per-frame auto-detect path
    assert "per-video tilt detection" in capsys.readouterr().out


def test_default_flag_preserves_per_frame_tilt_behaviour(monkeypatch):
    """With _USE_PER_VIDEO_TILT=False (default), _derive_shared_tilt is
    never called and correct_tilt receives tilt_angle=None (per-frame
    auto-detection)."""
    _patch_pipeline(monkeypatch, n_frames=2, tilt_angle=60.0)
    # Default value is False — assert it explicitly to catch any future
    # promotion of the flag without updating the test.
    assert video_single_pass_tilted._USE_PER_VIDEO_TILT is False

    def fail_helper(*a, **kw):
        raise AssertionError(
            "_derive_shared_tilt called with _USE_PER_VIDEO_TILT=False"
        )

    monkeypatch.setattr(video_single_pass_tilted, "_derive_shared_tilt", fail_helper)

    tilt_kwargs = []

    def spy_correct_tilt(img, tilt_angle=None, crop=True):
        tilt_kwargs.append(tilt_angle)
        return img, 60.0

    monkeypatch.setattr(video_single_pass_tilted, "correct_tilt", spy_correct_tilt)

    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    video_single_pass_tilted._single_pass_with_tilt(
        setup,
        FakeWriter(),
        lambda *a: None,
        lambda: False,
    )

    assert tilt_kwargs == [None, None]


def test_cancel_during_stage1_returns_cancelled_without_stage2(monkeypatch):
    """Cancel asserted before Stage 2 starts → status=cancelled, zero
    CSV rows, no correct_tilt calls."""
    _patch_pipeline(monkeypatch, n_frames=3, tilt_angle=60.0)
    monkeypatch.setattr(video_single_pass_tilted, "_USE_PER_VIDEO_TILT", True)
    monkeypatch.setattr(
        video_single_pass_tilted,
        "_derive_shared_tilt",
        lambda *a, **kw: (float("nan"), [], 0.0),
    )

    correct_tilt_calls = []

    def spy_correct_tilt(img, tilt_angle=None, crop=True):
        correct_tilt_calls.append(tilt_angle)
        return img, 60.0

    monkeypatch.setattr(video_single_pass_tilted, "correct_tilt", spy_correct_tilt)

    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    writer = FakeWriter()
    result = video_single_pass_tilted._single_pass_with_tilt(
        setup,
        writer,
        lambda *a: None,
        lambda: True,
    )

    assert result["status"] == "cancelled"
    assert result["frames_processed"] == 0
    assert len(writer.rows) == 0
    assert correct_tilt_calls == []


def test_effective_tilt_stride_applies_floor():
    """At the default target of 10 samples, the stride tightens for
    videos too short to yield 10 samples at the requested skip, and is
    left alone when the requested skip already meets the floor."""
    fn = video_single_pass_tilted._effective_tilt_stride

    # Kristina: 55 frames, skip=10 → cap=5, effective=5 (11 samples).
    assert fn(total_frames=55, requested_skip=10) == 5
    # 110 frames, skip=10 → cap=11, effective=min(10,11)=10 (11 samples).
    assert fn(total_frames=110, requested_skip=10) == 10
    # Isaac SC69_1: 440 frames, skip=10 → cap=44, effective=10 (unchanged).
    assert fn(total_frames=440, requested_skip=10) == 10
    # Very short: 8 frames, any skip → stride 1 (every frame).
    assert fn(total_frames=8, requested_skip=10) == 1
    # Requested skip already tighter than cap → unchanged.
    assert fn(total_frames=200, requested_skip=5) == 5
    # target_samples override honoured (callers can request a
    # different floor).
    assert fn(total_frames=100, requested_skip=10, target_samples=5) == 10


def test_per_video_tilt_tightens_stride_for_short_video(monkeypatch, capsys):
    """On a 55-frame video with skip=10, _derive_shared_tilt receives
    skip=5 (the floored stride), not the user's requested 10, and an
    INFO line announces the adjustment."""
    _patch_pipeline(monkeypatch, n_frames=55, tilt_angle=60.0)
    monkeypatch.setattr(video_single_pass_tilted, "_USE_PER_VIDEO_TILT", True)

    observed_skips = []

    def fake_derive(source_path, skip, total_samples, progress, cancel):
        observed_skips.append(skip)
        return 60.0, [60.0] * total_samples, 1.0

    monkeypatch.setattr(video_single_pass_tilted, "_derive_shared_tilt", fake_derive)

    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    setup.video_skip_interval = 10  # user's requested stride
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    video_single_pass_tilted._single_pass_with_tilt(
        setup,
        FakeWriter(),
        lambda *a: None,
        lambda: False,
    )

    assert observed_skips == [5]
    out = capsys.readouterr().out
    assert "stride tightened from 10 to 5" in out


def test_per_video_tilt_keeps_requested_stride_when_floor_met(monkeypatch, capsys):
    """On a 200-frame video with skip=10, the requested stride already
    yields 20 samples, so no tightening occurs and no INFO line is emitted."""
    _patch_pipeline(monkeypatch, n_frames=200, tilt_angle=60.0)
    monkeypatch.setattr(video_single_pass_tilted, "_USE_PER_VIDEO_TILT", True)

    observed_skips = []

    def fake_derive(source_path, skip, total_samples, progress, cancel):
        observed_skips.append(skip)
        return 60.0, [60.0] * total_samples, 1.0

    monkeypatch.setattr(video_single_pass_tilted, "_derive_shared_tilt", fake_derive)

    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    setup.video_skip_interval = 10
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    video_single_pass_tilted._single_pass_with_tilt(
        setup,
        FakeWriter(),
        lambda *a: None,
        lambda: False,
    )

    assert observed_skips == [10]
    assert "tightened stride" not in capsys.readouterr().out


def test_derive_shared_tilt_all_zeros_returns_nan(monkeypatch):
    """When every sample returns 0.0 the median is NaN and success
    rate is 0.0 — the caller uses this to trigger the fallback."""
    frames = _fake_frames(3)
    monkeypatch.setattr(
        video_single_pass_tilted,
        "load_frames",
        lambda *a, **kw: iter(list(frames)),
    )
    monkeypatch.setattr(
        video_single_pass_tilted,
        "detect_tilt_angle",
        lambda img: 0.0,
    )

    shared, samples, success_rate = video_single_pass_tilted._derive_shared_tilt(
        "/fake/source",
        skip=10,
        total_samples=3,
        progress=lambda *a: None,
        cancel=lambda: False,
    )

    assert math.isnan(shared)
    assert samples == [0.0, 0.0, 0.0]
    assert success_rate == 0.0


def test_no_frames_raises(monkeypatch):
    monkeypatch.setattr(video_single_pass_tilted, "get_frame_count", lambda p: 0)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    with pytest.raises(ValueError, match="No frames"):
        video_single_pass_tilted._single_pass_with_tilt(
            setup, FakeWriter(), lambda *a: None, lambda: False
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
