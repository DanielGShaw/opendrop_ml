"""Tests for video_state — JSON + PNG sidecar persistence of the
per-run pipeline state."""

import csv
import json
import os

import numpy as np

from opendrop_ml.modules.contact_angle.video_state import (
    load_state,
    regenerate_figures,
    save_state,
)


def _sample_result(include_rotated=True):
    result = {
        "status": "completed",
        "phase": "single_pass_tilted",
        "frames_processed": 20,
        "frames_skipped": 35,
        "median_tilt_angle": 60.0,
        "pass1_cps": {
            "left": [[100.0, 300.0], [101.0, 301.0]],
            "right": [[400.0, 300.0], [401.0, 301.0]],
        },
        "left_contact_line": [[90, 290], [110, 310]],
        "right_contact_line": [[390, 290], [410, 310]],
        "left_needle_x": None,
        "right_needle_x": None,
    }
    if include_rotated:
        result["rotated_first_frame"] = np.full((60, 80, 3), 128, dtype=np.uint8)
    else:
        result["rotated_first_frame"] = None
    return result


def test_save_state_writes_json_and_png_sidecar(tmp_path):
    result = _sample_result(include_rotated=True)
    given_source_path = "/videos/input.mp4"
    json_path = save_state(result, str(tmp_path), "run1", source_path=given_source_path)

    assert os.path.isfile(json_path)
    assert os.path.isfile(str(tmp_path / "run1_state.png"))

    with open(json_path) as f:
        payload = json.load(f)
    assert payload["has_rotated_first_frame"] is True
    # save_state stores os.path.abspath(source_path); compare against that
    # rather than the raw literal so this passes on Windows too, where
    # abspath resolves a leading-slash path onto the current drive and
    # uses backslash separators.
    assert payload["source_path"] == os.path.abspath(given_source_path)
    assert "rotated_first_frame_file" not in payload
    assert payload["median_tilt_angle"] == 60.0
    assert payload["frames_processed"] == 20


def test_save_state_omits_png_when_no_rotated_frame(tmp_path):
    result = _sample_result(include_rotated=False)
    save_state(result, str(tmp_path), "run2", source_path=None)

    assert os.path.isfile(str(tmp_path / "run2_state.json"))
    assert not os.path.isfile(str(tmp_path / "run2_state.png"))

    with open(str(tmp_path / "run2_state.json")) as f:
        payload = json.load(f)
    assert payload["has_rotated_first_frame"] is False
    assert payload["source_path"] is None


def test_save_state_records_absolute_source_path(tmp_path):
    result = _sample_result(include_rotated=False)
    cwd_relative = os.path.join("some", "relative", "path.mp4")
    save_state(result, str(tmp_path), "run_prov", source_path=cwd_relative)

    with open(str(tmp_path / "run_prov_state.json")) as f:
        payload = json.load(f)
    assert os.path.isabs(payload["source_path"])
    assert payload["source_path"].endswith(cwd_relative)


def test_load_state_follows_renamed_bundle(tmp_path):
    """Renaming both JSON and PNG to a new stem must still work because
    load_state derives the sidecar path from the JSON filename rather
    than a stored pointer."""
    result = _sample_result(include_rotated=True)
    given_source_path = "/in.mp4"
    save_state(result, str(tmp_path), "original", source_path=given_source_path)

    os.rename(
        str(tmp_path / "original_state.json"),
        str(tmp_path / "renamed_state.json"),
    )
    os.rename(
        str(tmp_path / "original_state.png"),
        str(tmp_path / "renamed_state.png"),
    )

    loaded = load_state(str(tmp_path / "renamed_state.json"))
    assert isinstance(loaded["rotated_first_frame"], np.ndarray)
    assert loaded["rotated_first_frame"].shape == (60, 80, 3)
    assert loaded["source_path"] == os.path.abspath(given_source_path)


def test_load_state_handles_missing_sidecar_gracefully(tmp_path):
    """If has_rotated_first_frame=True but the PNG is missing (e.g. user
    deleted it), load_state returns rotated_first_frame=None rather than
    raising."""
    result = _sample_result(include_rotated=True)
    json_path = save_state(result, str(tmp_path), "lost")
    os.remove(str(tmp_path / "lost_state.png"))

    loaded = load_state(json_path)
    assert loaded["rotated_first_frame"] is None


def test_roundtrip_preserves_fields(tmp_path):
    result = _sample_result(include_rotated=True)
    json_path = save_state(result, str(tmp_path), "run3")

    loaded = load_state(json_path)
    assert loaded["status"] == "completed"
    assert loaded["frames_processed"] == 20
    assert loaded["median_tilt_angle"] == 60.0
    assert loaded["pass1_cps"]["left"] == [[100.0, 300.0], [101.0, 301.0]]
    assert loaded["left_needle_x"] is None
    assert isinstance(loaded["rotated_first_frame"], np.ndarray)
    assert loaded["rotated_first_frame"].shape == (60, 80, 3)
    assert np.all(loaded["rotated_first_frame"] == 128)


def test_roundtrip_without_rotated_frame(tmp_path):
    result = _sample_result(include_rotated=False)
    json_path = save_state(result, str(tmp_path), "run4")

    loaded = load_state(json_path)
    assert loaded["rotated_first_frame"] is None
    assert loaded["pass1_cps"]["right"] == [[400.0, 300.0], [401.0, 301.0]]


def test_save_state_accepts_numpy_arrays(tmp_path):
    """pass1_cps sometimes arrives as numpy arrays — save_state must coerce."""
    result = _sample_result(include_rotated=False)
    result["pass1_cps"] = {
        "left": np.array([[100.0, 300.0]]),
        "right": np.array([[400.0, 300.0]]),
    }
    result["median_tilt_angle"] = np.float64(59.5)
    json_path = save_state(result, str(tmp_path), "run5")

    loaded = load_state(json_path)
    assert loaded["pass1_cps"]["left"] == [[100.0, 300.0]]
    assert loaded["median_tilt_angle"] == 59.5


def _write_sample_csv(path):
    columns = [
        "frame_number",
        "timestamp_s",
        "filepath",
        "error_status",
        "tangent fit left angle",
        "tangent fit right angle",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for i in range(1, 6):
            w.writerow([i, i * 0.033, "/test", "", 60.0 + i, 60.0 - i])


def test_regenerate_figures_produces_png_outputs(tmp_path):
    result = _sample_result(include_rotated=True)
    json_path = save_state(result, str(tmp_path), "run6")
    _write_sample_csv(str(tmp_path / "run6_results.csv"))

    ts_path, scatter_path = regenerate_figures(json_path)

    assert os.path.isfile(ts_path)
    assert os.path.getsize(ts_path) > 1000
    assert os.path.isfile(scatter_path)
    assert os.path.getsize(scatter_path) > 1000


def test_regenerate_figures_non_tilted_uses_source_path(tmp_path):
    result = _sample_result(include_rotated=False)
    json_path = save_state(result, str(tmp_path), "run7")
    _write_sample_csv(str(tmp_path / "run7_results.csv"))

    ts_path, scatter_path = regenerate_figures(
        json_path, source_path="/nonexistent/path"
    )
    assert os.path.isfile(ts_path)
    assert os.path.isfile(scatter_path)


def test_regenerate_figures_falls_back_to_recorded_source_path(tmp_path):
    """When caller doesn't pass source_path, regenerate_figures uses the
    provenance field written by save_state."""
    result = _sample_result(include_rotated=False)
    json_path = save_state(
        result, str(tmp_path), "run8", source_path="/nonexistent/recorded"
    )
    _write_sample_csv(str(tmp_path / "run8_results.csv"))

    ts_path, scatter_path = regenerate_figures(json_path)
    assert os.path.isfile(ts_path)
    assert os.path.isfile(scatter_path)
