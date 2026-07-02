import csv
import math
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from opendrop_ml.modules.contact_angle import ca_data_processor
from opendrop_ml.modules.contact_angle.ca_data_processor import (
    UnifiedCsvWriter,
    run_video_analysis,
)
from opendrop_ml.modules.core.classes import ExperimentalSetup
from opendrop_ml.utils.config import LEFT_ANGLE, RIGHT_ANGLE
from opendrop_ml.utils.enums import FittingMethod


@pytest.fixture
def setup_with_two_methods():
    setup = ExperimentalSetup()
    for method in setup.analysis_methods_ca:
        setup.analysis_methods_ca[method] = False
    setup.analysis_methods_ca[FittingMethod.TANGENT_FIT] = True
    setup.analysis_methods_ca[FittingMethod.ML_MODEL] = True
    return setup


def _read_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.reader(f))


def test_experimental_setup_has_new_fields():
    """All six video-mode flags default correctly on a fresh setup."""
    setup = ExperimentalSetup()
    assert setup.is_video is False
    assert setup.needle_in_drop is True
    assert setup.surface_tilted is False
    assert setup.tilt_angle_deg == 0.0
    assert setup.manual_preprocessing is False
    assert setup.video_skip_interval == 10
    assert isinstance(setup.tilt_angle_deg, float)
    assert isinstance(setup.video_skip_interval, int)


def test_writer_header_is_unified(tmp_path, setup_with_two_methods):
    """Header = 4 video columns + 2 enabled methods * (left, right)."""
    path = str(tmp_path / "out.csv")
    UnifiedCsvWriter(path, setup_with_two_methods)

    rows = _read_csv(path)
    assert len(rows) == 1
    header = rows[0]
    assert header[:4] == ["frame_number", "timestamp_s", "filepath", "error_status"]
    assert header[4] == "detected_tilt_angle"
    assert header[5:] == [
        "tangent fit left angle",
        "tangent fit right angle",
        "ML model left angle",
        "ML model right angle",
    ]


def test_writer_all_columns_every_row(tmp_path, setup_with_two_methods):
    """Every row has the same column count; missing values serialise as nan."""
    path = str(tmp_path / "out.csv")
    writer = UnifiedCsvWriter(path, setup_with_two_methods)

    # Full population
    writer.write_row(
        frame_number=0,
        timestamp_s=0.0,
        filepath="f0.png",
        error_status="",
        contact_angles={
            FittingMethod.TANGENT_FIT: {LEFT_ANGLE: 45.0, RIGHT_ANGLE: 46.0},
            FittingMethod.ML_MODEL: {LEFT_ANGLE: 44.5, RIGHT_ANGLE: 45.5},
        },
    )
    # Partial — only one method populated
    writer.write_row(
        frame_number=1,
        timestamp_s=0.1,
        filepath="f1.png",
        error_status="",
        contact_angles={
            FittingMethod.TANGENT_FIT: {LEFT_ANGLE: 47.0, RIGHT_ANGLE: None},
        },
    )
    # Empty — fit failure
    writer.write_row(
        frame_number=2,
        timestamp_s=0.2,
        filepath="f2.png",
        error_status="contour_short",
        contact_angles=None,
    )

    rows = _read_csv(path)
    header = rows[0]
    data_rows = rows[1:]

    assert len(data_rows) == 3
    for row in data_rows:
        assert len(row) == len(header)

    # detected_tilt_angle column defaults to NaN for every row (writer called
    # without the new kwarg — existing two-pass pipelines take this path).
    assert data_rows[0][4] == "nan"
    assert data_rows[1][4] == "nan"
    assert data_rows[2][4] == "nan"
    # Full row: all numeric
    assert data_rows[0][5:] == ["45.0", "46.0", "44.5", "45.5"]
    # Partial row: ML_MODEL NaNs, TANGENT right NaN
    assert data_rows[1][5] == "47.0"
    assert data_rows[1][6] == "nan"
    assert data_rows[1][7] == "nan"
    assert data_rows[1][8] == "nan"
    # Empty row: all NaNs; error_status populated
    assert data_rows[2][3] == "contour_short"
    assert data_rows[2][5:] == ["nan", "nan", "nan", "nan"]


def test_writer_append_mode_preserves_header(tmp_path, setup_with_two_methods):
    """A second writer against the same path does not re-emit the header."""
    path = str(tmp_path / "out.csv")

    w1 = UnifiedCsvWriter(path, setup_with_two_methods)
    w1.write_row(0, 0.0, "f0.png", "", {})

    w2 = UnifiedCsvWriter(path, setup_with_two_methods)
    w2.write_row(1, 0.1, "f1.png", "", {})

    rows = _read_csv(path)
    # One header + two data rows (no duplicate header from w2).
    assert len(rows) == 3
    assert rows[0][0] == "frame_number"
    assert rows[1][0] == "0"
    assert rows[2][0] == "1"


def test_run_video_analysis_routes_to_two_pass(monkeypatch):
    """needle_in_drop=False forwards all four args to _auto_two_pass."""
    calls = []

    def spy(user_input_data, writer, progress, cancel):
        calls.append((user_input_data, writer, progress, cancel))
        return "spy-return"

    monkeypatch.setattr(ca_data_processor, "_auto_two_pass", spy)

    setup = ExperimentalSetup()
    # Default is False, but set explicitly to make intent obvious.
    setup.needle_in_drop = False
    sentinel_writer = object()

    def progress(*args):
        return None

    def cancel():
        return False

    result = run_video_analysis(setup, sentinel_writer, progress, cancel)

    assert result == "spy-return"
    assert len(calls) == 1
    assert calls[0] == (setup, sentinel_writer, progress, cancel)


def test_run_video_analysis_routes_to_needle_in(monkeypatch):
    """needle_in_drop=True forwards all four args to _needle_in_two_pass."""
    calls = []

    def spy(user_input_data, writer, progress, cancel):
        calls.append((user_input_data, writer, progress, cancel))
        return "needle-spy-return"

    monkeypatch.setattr(ca_data_processor, "_needle_in_two_pass", spy)

    setup = ExperimentalSetup()
    setup.needle_in_drop = True
    sentinel_writer = object()

    def progress(*args):
        return None

    def cancel():
        return False

    result = run_video_analysis(setup, sentinel_writer, progress, cancel)

    assert result == "needle-spy-return"
    assert len(calls) == 1
    assert calls[0] == (setup, sentinel_writer, progress, cancel)


def test_run_video_analysis_routes_to_single_pass_with_tilt(monkeypatch):
    """(needle_in_drop=False, surface_tilted=True) → _single_pass_with_tilt."""
    calls = []

    def spy(user_input_data, writer, progress, cancel):
        calls.append((user_input_data, writer, progress, cancel))
        return "tilt-spy-return"

    monkeypatch.setattr(ca_data_processor, "_single_pass_with_tilt", spy)

    setup = ExperimentalSetup()
    setup.needle_in_drop = False
    setup.surface_tilted = True
    sentinel_writer = object()
    progress = lambda *a: None
    cancel = lambda: False

    result = run_video_analysis(setup, sentinel_writer, progress, cancel)

    assert result == "tilt-spy-return"
    assert len(calls) == 1
    assert calls[0] == (setup, sentinel_writer, progress, cancel)


def test_run_video_analysis_raises_for_needle_plus_tilt():
    """(needle_in_drop=True, surface_tilted=True) is explicitly unsupported."""
    setup = ExperimentalSetup()
    setup.needle_in_drop = True
    setup.surface_tilted = True

    with pytest.raises(NotImplementedError) as excinfo:
        run_video_analysis(setup, object(), lambda *a: None, lambda: False)
    assert "needle-in-drop" in str(excinfo.value)
    assert "tilted-surface" in str(excinfo.value)


def test_writer_detected_tilt_angle_populates_column(tmp_path, setup_with_two_methods):
    """Passing detected_tilt_angle writes the value into column index 4."""
    path = str(tmp_path / "out.csv")
    writer = UnifiedCsvWriter(path, setup_with_two_methods)

    writer.write_row(
        frame_number=0,
        timestamp_s=0.0,
        filepath="f0.png",
        error_status="",
        contact_angles=None,
        detected_tilt_angle=60.25,
    )

    rows = _read_csv(path)
    assert rows[1][4] == "60.25"


def test_auto_two_pass_delegates_to_video_two_pass(monkeypatch):
    """ca_data_processor._auto_two_pass forwards to video_two_pass._auto_two_pass."""
    from opendrop_ml.modules.contact_angle import video_two_pass

    calls = []

    def spy(user_input_data, writer, progress, cancel):
        calls.append((user_input_data, writer, progress, cancel))
        return "delegate-return"

    monkeypatch.setattr(video_two_pass, "_auto_two_pass", spy)

    sentinel_user = object()
    sentinel_writer = object()
    progress = lambda *a: None
    cancel = lambda: False

    result = ca_data_processor._auto_two_pass(
        sentinel_user, sentinel_writer, progress, cancel
    )
    assert result == "delegate-return"
    assert calls == [(sentinel_user, sentinel_writer, progress, cancel)]


def test_process_data_does_not_mutate_cropped_image(monkeypatch):
    """cropped_image must remain the unrotated canvas captured by set_drop_region;
    nothing in the processing pipeline should overwrite it. Regression guard for
    the tangent-overlay misalignment bug — every fit method's overlay renders on
    cropped_image, so anything that mutates it desyncs the overlay from the coords.
    """
    from opendrop_ml.utils.enums import ThresholdSelect

    original_cropped = object()  # sentinel for the unrotated canvas

    def fake_get_image(raw_experiment, user_input_data, i):
        raw_experiment.image = object()
        raw_experiment.cropped_image = original_cropped

    def fake_set_drop_region(raw_experiment, user_input_data, i):
        return None

    def fake_extract_drop_profile(raw_experiment, user_input_data):
        return None

    def fake_set_surface_line(raw_experiment, user_input_data):
        return None

    def fake_perform_fits(raw_experiment, **kwargs):
        if kwargs.get("yl"):
            raw_experiment.contact_angles[FittingMethod.YL_FIT] = {
                LEFT_ANGLE: 90.0, RIGHT_ANGLE: 90.0,
            }
        else:
            raw_experiment.contact_angles.setdefault(FittingMethod.TANGENT_FIT, {
                LEFT_ANGLE: 45.0, RIGHT_ANGLE: 45.0,
            })

    monkeypatch.setattr(ca_data_processor, "get_image", fake_get_image)
    monkeypatch.setattr(ca_data_processor, "set_drop_region", fake_set_drop_region)
    monkeypatch.setattr(
        ca_data_processor, "extract_drop_profile", fake_extract_drop_profile
    )
    monkeypatch.setattr(ca_data_processor, "set_surface_line", fake_set_surface_line)
    monkeypatch.setattr(ca_data_processor, "perform_fits", fake_perform_fits)

    setup = ExperimentalSetup()
    setup.number_of_frames = 1
    setup.import_files = ["dummy.png"]
    setup.baseline_method = ThresholdSelect.AUTOMATED
    for method in setup.analysis_methods_ca:
        setup.analysis_methods_ca[method] = False
    setup.analysis_methods_ca[FittingMethod.TANGENT_FIT] = True
    setup.analysis_methods_ca[FittingMethod.YL_FIT] = True

    captured_drops = []

    def callback(num, raw_experiment):
        captured_drops.append(raw_experiment)

    processor = ca_data_processor.CaDataProcessor()
    processor.process_data(None, setup, callback)

    assert len(captured_drops) == 1
    assert captured_drops[0].cropped_image is original_cropped


if __name__ == "__main__":
    pytest.main(["-v", __file__])
