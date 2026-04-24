"""Unit tests for the two-pass tilted-surface video pipeline.

Mocks per-frame helpers (correct_tilt, isolate_contour, CPID,
touches_border, the _try_* fit wrappers, _predict_ml, and
_derive_shared_tilt) so the two-pass control flow is exercised
without TensorFlow or real images.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

from opendrop_ml.modules.contact_angle import (  # noqa: E402
    _video_helpers,
    contact_line,
    video_two_pass_tilted,
)
from opendrop_ml.modules.core.classes import ExperimentalSetup  # noqa: E402
from opendrop_ml.utils.config import LEFT_ANGLE, RIGHT_ANGLE  # noqa: E402
from opendrop_ml.utils.enums import FittingMethod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_setup(methods=None, skip=1):
    setup = ExperimentalSetup()
    setup.is_video = True
    setup.surface_tilted = True
    setup.needle_in_drop = False
    setup.import_files = ["/fake/source"]
    setup.video_skip_interval = skip
    for m in setup.analysis_methods_ca:
        setup.analysis_methods_ca[m] = False
    if methods:
        for m in methods:
            setup.analysis_methods_ca[m] = True
    return setup


class FakeWriter:
    """Captures write_row kwargs; matches the UnifiedCsvWriter contract
    including detected_tilt_angle for tilted-mode CSVs."""

    def __init__(self):
        self.rows = []

    def write_row(self, *, frame_number, timestamp_s, filepath,
                  error_status, contact_angles=None,
                  detected_tilt_angle=math.nan):
        self.rows.append({
            "frame_number": frame_number,
            "timestamp_s": timestamp_s,
            "filepath": filepath,
            "error_status": error_status,
            "contact_angles": contact_angles or {},
            "detected_tilt_angle": detected_tilt_angle,
        })


def _fake_frames(n, with_fps=False, shape=(400, 400, 3)):
    """n frames as (img, frame_number, meta) tuples."""
    frames = []
    for i in range(n):
        meta = {"fps": 30.0} if with_fps else {"filename": "f{}.png".format(i)}
        frames.append((np.zeros(shape, np.uint8), i, meta))
    return frames


def _good_contour(npts=60):
    """Monotonic contour with usable interior CP positions."""
    return np.column_stack([
        np.arange(npts, dtype=float),
        np.linspace(0, npts, npts, dtype=float),
    ])


def _patch_pipeline(
    monkeypatch,
    *,
    n_frames,
    shared_tilt=60.0,
    success_rate=1.0,
    contour=None,
    cps=None,
    isolate_raises=False,
    cpid_raises=False,
    cpid_returns_none=False,
    with_fps=False,
):
    """Stand up a mocked two-pass-tilted environment.

    Three load_frames call-sites to satisfy: inside ``_derive_shared_tilt``
    (bypassed because we stub the helper directly), Pass 1 replay, and
    Pass 2 main loop. The stub respects ``frame_step`` by slicing.
    """
    if contour is None:
        contour = _good_contour(60)
    if cps is None:
        cps = np.array([contour[10], contour[50]], dtype=float)

    all_frames = _fake_frames(n_frames, with_fps=with_fps)

    def fake_load_frames(source, frame_step=1):
        return iter([all_frames[i] for i in range(0, n_frames, frame_step)])

    spy = {"derive_shared_tilt_calls": [], "correct_tilt_calls": []}

    def fake_derive(source_path, skip, total_samples, progress, cancel,
                    label="Tilt calibration"):
        spy["derive_shared_tilt_calls"].append({
            "source_path": source_path,
            "skip": skip,
            "total_samples": total_samples,
            "label": label,
        })
        for i, _ in enumerate(range(0, n_frames, skip)):
            if cancel():
                break
            progress(i + 1, total_samples, label)
        return shared_tilt, [shared_tilt], success_rate

    def fake_correct_tilt(img, tilt_angle=None, crop=True):
        spy["correct_tilt_calls"].append(tilt_angle)
        return img, float(tilt_angle) if tilt_angle is not None else 0.0

    def fake_isolate(img, crop=True, DISPLAY=False, blur=True):
        if isolate_raises:
            raise RuntimeError("synthetic isolate failure")
        # crop=False returns 2-tuple (longest_cluster, threshold).
        return contour.copy(), 127

    def fake_cpid(c, display=False):
        if cpid_raises:
            raise RuntimeError("synthetic cpid failure")
        if cpid_returns_none:
            return None
        return cps.copy()

    monkeypatch.setattr(video_two_pass_tilted, "load_frames", fake_load_frames)
    monkeypatch.setattr(
        video_two_pass_tilted, "get_frame_count", lambda path: n_frames
    )
    monkeypatch.setattr(
        video_two_pass_tilted, "_derive_shared_tilt", fake_derive
    )
    monkeypatch.setattr(
        video_two_pass_tilted, "correct_tilt", fake_correct_tilt
    )
    monkeypatch.setattr(
        video_two_pass_tilted, "isolate_contour", fake_isolate
    )
    monkeypatch.setattr(video_two_pass_tilted, "CPID", fake_cpid)
    monkeypatch.setattr(
        video_two_pass_tilted, "_load_ml_model", lambda: None
    )
    # Default find_first_intercept: returns a sensible interior index for
    # the canonical contour. Individual tests can override.
    monkeypatch.setattr(
        video_two_pass_tilted, "find_first_intercept",
        lambda half, line: len(half) // 2,
    )
    return all_frames, contour, cps, spy


# ---------------------------------------------------------------------------
# 1 & 2. Pass-1 tilt collection + shared-tilt flow into Pass 2
# ---------------------------------------------------------------------------

def test_pass1_tilt_collection_uses_effective_stride(monkeypatch):
    """_derive_shared_tilt is invoked with the effective stride, not the
    raw video_skip_interval, when the requested skip would yield fewer
    than _PASS1_TARGET_SAMPLES samples."""
    _, _, _, spy = _patch_pipeline(monkeypatch, n_frames=55)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT], skip=10)
    monkeypatch.setattr(
        _video_helpers, "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    video_two_pass_tilted._auto_two_pass_with_tilt_correction(
        setup, FakeWriter(), lambda *a: None, lambda: False,
    )

    assert len(spy["derive_shared_tilt_calls"]) == 1
    call = spy["derive_shared_tilt_calls"][0]
    # n_frames=55, requested=10 → cap = 55 // 10 = 5 → effective = min(10, 5) = 5
    assert call["skip"] == 5
    assert call["label"] == "Pass 1"


def test_pass2_shared_tilt_written_on_every_row(monkeypatch):
    """Every Pass-2 row has detected_tilt_angle == shared_tilt (the
    median from Pass 1), regardless of per-frame variation."""
    _, _, _, _ = _patch_pipeline(monkeypatch, n_frames=4, shared_tilt=57.3)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers, "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    writer = FakeWriter()
    video_two_pass_tilted._auto_two_pass_with_tilt_correction(
        setup, writer, lambda *a: None, lambda: False,
    )

    assert len(writer.rows) == 4
    for row in writer.rows:
        assert row["detected_tilt_angle"] == pytest.approx(57.3)


# ---------------------------------------------------------------------------
# 3. Pass-1 aborts when every sample returns 0.0
# ---------------------------------------------------------------------------

def test_pass1_aborts_when_all_samples_return_zero(monkeypatch):
    """success_rate=0.0 from _derive_shared_tilt (every sample produced
    0.0) triggers a clear RuntimeError pointing at the fallback paths."""
    _patch_pipeline(monkeypatch, n_frames=6, success_rate=0.0, shared_tilt=float("nan"))
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    with pytest.raises(RuntimeError, match="tilt calibration failed"):
        video_two_pass_tilted._auto_two_pass_with_tilt_correction(
            setup, FakeWriter(), lambda *a: None, lambda: False,
        )


# ---------------------------------------------------------------------------
# 4. Pass-1 CP collection uses the shared tilt for every replayed frame
# ---------------------------------------------------------------------------

def test_pass1_cp_collection_uses_shared_tilt_every_frame(monkeypatch):
    """Pass 1's CP-collection replay calls correct_tilt with the shared
    tilt angle on every sampled frame (never None, never per-frame detected)."""
    _, _, _, spy = _patch_pipeline(
        monkeypatch, n_frames=4, shared_tilt=62.0,
    )
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT], skip=1)
    monkeypatch.setattr(
        _video_helpers, "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    video_two_pass_tilted._auto_two_pass_with_tilt_correction(
        setup, FakeWriter(), lambda *a: None, lambda: False,
    )

    # 4 Pass-1 replay calls + 4 Pass-2 calls = 8; every one with tilt_angle=62.0
    assert len(spy["correct_tilt_calls"]) == 8
    assert all(a == 62.0 for a in spy["correct_tilt_calls"])


# ---------------------------------------------------------------------------
# 5. Pass-1 CPID-failure skip. Pass 1 uses isolate_contour(crop=False),
#    which includes the baseline so touches_border would always fire.
#    Pass-1 skip responsibility lives entirely in CPID sanity.
# ---------------------------------------------------------------------------

def test_pass1_aborts_when_cpid_returns_none_on_all_samples(monkeypatch):
    """When every Pass-1 sample's CPID returns None, no CPs are
    collected and Pass 1 aborts with the min-samples error."""
    _patch_pipeline(monkeypatch, n_frames=6, cpid_returns_none=True)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    with pytest.raises(RuntimeError, match="CP collection failed"):
        video_two_pass_tilted._auto_two_pass_with_tilt_correction(
            setup, FakeWriter(), lambda *a: None, lambda: False,
        )


# ---------------------------------------------------------------------------
# 6. Median contact line + residual IQR (fit_contact_line_with_stats)
# ---------------------------------------------------------------------------

def test_median_contact_line_derivation_iqr_plus_odr():
    """fit_contact_line_with_stats returns a horizontal-ish line + a
    finite, small residual IQR for noisy CPs clustered around y=100."""
    rng = np.random.default_rng(42)
    xs = np.linspace(0, 200, 30)
    ys = 100 + rng.normal(0, 0.5, size=30)
    cps = np.column_stack([xs, ys])

    line, stats = contact_line.fit_contact_line_with_stats(cps)

    assert line.shape == (2, 2)
    assert abs(line[0, 1] - 100) < 1.0
    assert abs(line[1, 1] - 100) < 1.0
    assert stats["n_total"] == 30
    assert stats["n_filtered"] > 0
    assert np.isfinite(stats["residual_iqr"])
    assert 0 < stats["residual_iqr"] < 1.0


# ---------------------------------------------------------------------------
# 7. Pass 2 control flow — correct_tilt → isolate_contour → CPID → find_first_intercept
# ---------------------------------------------------------------------------

def test_pass2_control_flow_calls_expected_helpers(monkeypatch):
    _, _, _, spy = _patch_pipeline(monkeypatch, n_frames=3)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers, "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    call_log = []
    real_isolate = video_two_pass_tilted.isolate_contour
    real_cpid = video_two_pass_tilted.CPID
    real_find = video_two_pass_tilted.find_first_intercept

    def log_isolate(*a, **kw):
        call_log.append("isolate")
        return real_isolate(*a, **kw)

    def log_cpid(*a, **kw):
        call_log.append("cpid")
        return real_cpid(*a, **kw)

    def log_find(*a, **kw):
        call_log.append("find")
        return real_find(*a, **kw)

    monkeypatch.setattr(video_two_pass_tilted, "isolate_contour", log_isolate)
    monkeypatch.setattr(video_two_pass_tilted, "CPID", log_cpid)
    monkeypatch.setattr(
        video_two_pass_tilted, "find_first_intercept", log_find,
    )

    video_two_pass_tilted._auto_two_pass_with_tilt_correction(
        setup, FakeWriter(), lambda *a: None, lambda: False,
    )

    # Pass 1 replay + Pass 2 main: isolate before cpid, cpid before find.
    pass2_slice = call_log[call_log.index("find") - 2:]
    assert pass2_slice[0] == "isolate"
    assert pass2_slice[1] == "cpid"
    assert pass2_slice[2] == "find"


# ---------------------------------------------------------------------------
# 8. Pass 2 shared-tilt application
# ---------------------------------------------------------------------------

def test_pass2_shared_tilt_applied_every_frame(monkeypatch):
    """Pass 2's correct_tilt calls use the shared_tilt from Pass 1 for
    every frame — not per-frame detection, not None."""
    _, _, _, spy = _patch_pipeline(
        monkeypatch, n_frames=3, shared_tilt=55.5,
    )
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers, "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    video_two_pass_tilted._auto_two_pass_with_tilt_correction(
        setup, FakeWriter(), lambda *a: None, lambda: False,
    )

    # 3 Pass-1 replay + 3 Pass-2 = 6 calls, all with tilt_angle=55.5
    assert len(spy["correct_tilt_calls"]) == 6
    assert all(a == 55.5 for a in spy["correct_tilt_calls"])


# ---------------------------------------------------------------------------
# 9. Cancel mid-Pass-1
# ---------------------------------------------------------------------------

def test_cancel_mid_pass1_returns_early(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=10)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])

    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2  # fires during Pass 1 replay

    writer = FakeWriter()
    result = video_two_pass_tilted._auto_two_pass_with_tilt_correction(
        setup, writer, lambda *a: None, cancel,
    )

    assert result["status"] == "cancelled"
    assert result["phase"] == "two_pass_tilted"
    assert writer.rows == []  # No Pass-2 rows written


# ---------------------------------------------------------------------------
# 10. Cancel mid-Pass-2
# ---------------------------------------------------------------------------

def test_cancel_mid_pass2_keeps_partial_rows(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=10)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers, "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    # Let Pass 1 finish (count ≈ 10 cancel calls through replay),
    # then fire cancel after the 3rd Pass-2 frame.
    counter = {"n": 0}
    cancel_after_pass2 = {"n": 0}

    def cancel():
        counter["n"] += 1
        # Once Pass 2 starts writing, count frames and cancel after 3.
        if counter["n"] > 20:
            cancel_after_pass2["n"] += 1
            return cancel_after_pass2["n"] > 3
        return False

    writer = FakeWriter()
    result = video_two_pass_tilted._auto_two_pass_with_tilt_correction(
        setup, writer, lambda *a: None, cancel,
    )
    assert result["status"] == "cancelled"
    assert 1 <= len(writer.rows) <= 10


# ---------------------------------------------------------------------------
# 11. Per-fit-failure isolation
# ---------------------------------------------------------------------------

def test_per_fit_failure_isolated(monkeypatch):
    """One method raising leaves the others populated; row still writes
    with error_status=''."""
    _patch_pipeline(monkeypatch, n_frames=3)
    setup = _make_setup(methods=[
        FittingMethod.TANGENT_FIT,
        FittingMethod.CIRCLE_FIT,
        FittingMethod.YL_FIT,
    ])
    monkeypatch.setattr(
        _video_helpers, "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )
    monkeypatch.setattr(
        _video_helpers, "circular_fit",
        lambda profile, display=False: ((40.0, 41.0), None, None, None, {}, {}),
    )

    def boom(profile, display=False):
        raise RuntimeError("synthetic yl failure")

    monkeypatch.setattr(_video_helpers, "yl_fit", boom)

    writer = FakeWriter()
    video_two_pass_tilted._auto_two_pass_with_tilt_correction(
        setup, writer, lambda *a: None, lambda: False,
    )

    angles = writer.rows[0]["contact_angles"]
    assert FittingMethod.TANGENT_FIT in angles
    assert FittingMethod.CIRCLE_FIT in angles
    assert FittingMethod.YL_FIT not in angles
    assert writer.rows[0]["error_status"] == ""


# ---------------------------------------------------------------------------
# 12. Return-dict shape + median_tilt_angle == shared_tilt
# ---------------------------------------------------------------------------

def test_return_dict_shape_and_median_tilt_equals_shared(monkeypatch):
    _patch_pipeline(monkeypatch, n_frames=3, shared_tilt=58.4)
    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers, "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    result = video_two_pass_tilted._auto_two_pass_with_tilt_correction(
        setup, FakeWriter(), lambda *a: None, lambda: False,
    )

    assert set(result.keys()) == {
        "status", "phase", "frames_processed", "frames_skipped",
        "median_tilt_angle", "rotated_first_frame",
        "left_contact_line", "right_contact_line", "pass1_cps",
        "left_needle_x", "right_needle_x",
    }
    assert result["phase"] == "two_pass_tilted"
    assert result["status"] == "completed"
    assert result["median_tilt_angle"] == pytest.approx(58.4)
    assert result["left_needle_x"] is None
    assert result["right_needle_x"] is None
    assert isinstance(result["left_contact_line"], list)
    assert isinstance(result["right_contact_line"], list)


# ---------------------------------------------------------------------------
# 13. Effective-stride fallback (parametrised)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_frames,requested_skip,expected_stride", [
    (55, 10, 5),    # Kristina: tightened to meet floor.
    (440, 10, 10),  # Long enough: unchanged.
    (12, 10, 1),    # Very short: every frame.
])
def test_effective_stride_fallback(
    monkeypatch, n_frames, requested_skip, expected_stride,
):
    _, _, _, spy = _patch_pipeline(monkeypatch, n_frames=n_frames)
    setup = _make_setup(
        methods=[FittingMethod.TANGENT_FIT], skip=requested_skip,
    )
    monkeypatch.setattr(
        _video_helpers, "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    video_two_pass_tilted._auto_two_pass_with_tilt_correction(
        setup, FakeWriter(), lambda *a: None, lambda: False,
    )

    assert spy["derive_shared_tilt_calls"][0]["skip"] == expected_stride


# ---------------------------------------------------------------------------
# 14. Pass-2 CP-outlier gate (parametrised)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("distance,expect_skip", [
    # Image height in the mocked _fake_frames is 400 px, so the floor
    # evaluates to 0.015 × 400 = 6.0 px for these cases.
    (3.0, False),   # below both 3×IQR (= 3.0) and the 6.0-px floor
    (5.0, False),   # above 3×IQR but still below the 6.0-px floor
    (12.0, True),   # above both 3×IQR and the 6.0-px floor → filtered
])
def test_pass2_cp_outlier_gate(monkeypatch, distance, expect_skip):
    """Pass 2 CP-outlier gate: CPID CP perpendicular distance vs
    max(3× Pass-1 IQR, 1.5% × image_height).

    Setup: flat horizontal contour at y=100, CPs on the contour, and
    the Pass-1 line pinned at y = 100 - distance so the perpendicular
    distance from each CP to the line is exactly ``distance``.
    residual_iqr is pinned to 1.0 so the IQR-scaled threshold (3 px)
    is below the resolution-relative floor — tests the floor-
    dominated regime.
    """
    flat_contour = np.column_stack([
        np.arange(60, dtype=float),
        np.full(60, 100.0),
    ])
    cps = np.array([flat_contour[10], flat_contour[50]], dtype=float)
    _patch_pipeline(monkeypatch, n_frames=3, contour=flat_contour, cps=cps)

    # Pin Pass-1 line below the CPs by `distance`. Perp distance from
    # each CP at y=100 to a horizontal line at y=100-distance is `distance`.
    pinned_line = np.array([
        [0.0, 100.0 - distance],
        [60.0, 100.0 - distance],
    ])
    stats = {"residual_iqr": 1.0, "n_filtered": 10, "n_total": 10}
    monkeypatch.setattr(
        video_two_pass_tilted, "fit_contact_line_with_stats",
        lambda arr: (pinned_line, stats),
    )

    setup = _make_setup(methods=[FittingMethod.TANGENT_FIT])
    monkeypatch.setattr(
        _video_helpers, "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, {}, {}),
    )

    writer = FakeWriter()
    video_two_pass_tilted._auto_two_pass_with_tilt_correction(
        setup, writer, lambda *a: None, lambda: False,
    )

    assert len(writer.rows) == 3  # all 3 frames write a row (passed or skipped)
    row = writer.rows[0]
    if expect_skip:
        assert row["error_status"] == "drop_off_edge"
        assert row["contact_angles"] == {}
    else:
        assert row["error_status"] == ""
        assert FittingMethod.TANGENT_FIT in row["contact_angles"]
