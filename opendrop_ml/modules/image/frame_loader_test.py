import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from opendrop_ml.modules.image.frame_loader import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    get_first_frame,
    get_frame_count,
    load_frames,
    natural_sort_key,
)


def _write_pngs(directory, names):
    for name in names:
        arr = np.full((8, 8, 3), 128, dtype=np.uint8)
        cv2.imwrite(os.path.join(directory, name), arr)


def test_natural_sort_key_orders_embedded_numbers():
    files = ["a10.png", "a2.png", "a1.png"]
    assert sorted(files, key=natural_sort_key) == ["a1.png", "a2.png", "a10.png"]


def test_load_frames_directory_yields_in_natural_order(tmp_path):
    _write_pngs(tmp_path, ["a1.png", "a10.png", "a2.png"])
    yielded = list(load_frames(str(tmp_path)))
    names = [meta["filename"] for _frame, _i, meta in yielded]
    assert names == ["a1.png", "a2.png", "a10.png"]
    # Frame numbers are sequential 0-based indices of the sorted list.
    assert [i for _frame, i, _m in yielded] == [0, 1, 2]


def test_load_frames_directory_respects_frame_step(tmp_path):
    _write_pngs(tmp_path, ["a1.png", "a2.png", "a3.png", "a4.png"])
    yielded = list(load_frames(str(tmp_path), frame_step=2))
    assert [i for _frame, i, _m in yielded] == [0, 2]


def test_load_frames_directory_filters_non_image_files(tmp_path):
    _write_pngs(tmp_path, ["good.png"])
    (tmp_path / "notes.txt").write_text("ignore me")
    yielded = list(load_frames(str(tmp_path)))
    assert len(yielded) == 1
    assert yielded[0][2]["filename"] == "good.png"


def test_load_frames_directory_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="No images found"):
        list(load_frames(str(tmp_path)))


def test_load_frames_nonexistent_path_raises():
    with pytest.raises(FileNotFoundError):
        list(load_frames("/definitely/not/here"))


def test_load_frames_unsupported_file_raises(tmp_path):
    bad = tmp_path / "data.csv"
    bad.write_text("not a video")
    with pytest.raises(ValueError, match="Unsupported file format"):
        list(load_frames(str(bad)))


def test_get_frame_count_directory(tmp_path):
    _write_pngs(tmp_path, ["a.png", "b.png", "c.bmp"])
    assert get_frame_count(str(tmp_path)) == 3


def test_get_first_frame_directory(tmp_path):
    _write_pngs(tmp_path, ["b2.png", "a1.png", "c3.png"])
    frame, frame_number, meta = get_first_frame(str(tmp_path))
    assert frame_number == 0
    assert meta["filename"] == "a1.png"
    assert frame.shape == (8, 8, 3)


def test_get_first_frame_empty_directory_raises(tmp_path):
    with pytest.raises(ValueError, match="No images found"):
        get_first_frame(str(tmp_path))


def test_extension_constants_contain_expected_values():
    assert ".bmp" in IMAGE_EXTENSIONS
    assert ".png" in IMAGE_EXTENSIONS
    assert ".mp4" in VIDEO_EXTENSIONS
    assert ".avi" in VIDEO_EXTENSIONS


if __name__ == "__main__":
    pytest.main(["-v", __file__])
