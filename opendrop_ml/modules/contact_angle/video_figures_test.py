"""Tests for video_figures.py — diagnostic figure generation."""

import csv
import math
import os
import tempfile

import numpy as np
import pytest

from opendrop_ml.modules.contact_angle.video_figures import (
    generate_timeseries_figure,
    generate_cp_scatter_figure,
)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a small unified CSV with two methods and 10 frames."""
    csv_path = str(tmp_path / "test_results.csv")
    columns = [
        "frame_number",
        "timestamp_s",
        "filepath",
        "error_status",
        "tangent fit left angle",
        "tangent fit right angle",
        "circle fit left angle",
        "circle fit right angle",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(1, 11):
            if i == 5:
                writer.writerow([i, float("nan"), "/test", "contour_short",
                                 float("nan"), float("nan"),
                                 float("nan"), float("nan")])
            else:
                writer.writerow([i, i * 0.033, "/test", "",
                                 110.0 + i * 0.5, 108.0 + i * 0.3,
                                 111.0 + i * 0.4, 109.0 + i * 0.2])
    return csv_path


@pytest.fixture
def sample_cps():
    """Create synthetic Pass 1 contact point data."""
    np.random.seed(42)
    left = np.column_stack([
        np.random.normal(100, 3, 20),
        np.random.normal(300, 5, 20),
    ])
    right = np.column_stack([
        np.random.normal(400, 3, 20),
        np.random.normal(300, 5, 20),
    ])
    return {"left": left.tolist(), "right": right.tolist()}


@pytest.fixture
def sample_lines():
    """Create synthetic contact line endpoints."""
    return {
        "left": [[90, 290], [110, 310]],
        "right": [[390, 290], [410, 310]],
    }


class TestGenerateTimeseriesFigure:
    def test_creates_png(self, sample_csv, tmp_path):
        output = str(tmp_path / "timeseries.png")
        generate_timeseries_figure(sample_csv, output)
        assert os.path.isfile(output)
        assert os.path.getsize(output) > 1000

    def test_png_header(self, sample_csv, tmp_path):
        output = str(tmp_path / "timeseries.png")
        generate_timeseries_figure(sample_csv, output)
        with open(output, "rb") as f:
            header = f.read(8)
        assert header[:4] == b"\x89PNG"

    def test_raises_on_missing_frame_number(self, tmp_path):
        csv_path = str(tmp_path / "bad.csv")
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["col_a", "col_b"])
            csv.writer(f).writerow([1, 2])
        with pytest.raises(ValueError, match="frame_number"):
            generate_timeseries_figure(csv_path, str(tmp_path / "out.png"))

    def test_raises_on_no_angle_columns(self, tmp_path):
        csv_path = str(tmp_path / "no_angles.csv")
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["frame_number", "error_status"])
            csv.writer(f).writerow([1, ""])
        with pytest.raises(ValueError, match="No angle columns"):
            generate_timeseries_figure(csv_path, str(tmp_path / "out.png"))


class TestGenerateCpScatterFigure:
    def test_creates_png_with_image_dir(self, sample_cps, sample_lines, tmp_path):
        img_dir = str(tmp_path / "images")
        os.makedirs(img_dir)
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(os.path.join(img_dir, "frame_001.png"), dummy_img)

        output = str(tmp_path / "scatter.png")
        generate_cp_scatter_figure(img_dir, sample_cps, sample_lines, output)
        assert os.path.isfile(output)
        assert os.path.getsize(output) > 1000

    def test_creates_png_without_background(self, sample_cps, sample_lines, tmp_path):
        output = str(tmp_path / "scatter_nobg.png")
        generate_cp_scatter_figure(
            "/nonexistent/path", sample_cps, sample_lines, output
        )
        assert os.path.isfile(output)

    def test_raises_on_none_cps(self, sample_lines, tmp_path):
        with pytest.raises(ValueError, match="No Pass 1"):
            generate_cp_scatter_figure(
                "/tmp", None, sample_lines, str(tmp_path / "out.png")
            )

    def test_raises_on_empty_cps(self, sample_lines, tmp_path):
        empty_cps = {"left": [], "right": []}
        with pytest.raises(ValueError, match="no contact points"):
            generate_cp_scatter_figure(
                "/tmp", empty_cps, sample_lines, str(tmp_path / "out.png")
            )

    def test_renders_with_both_needle_lines(
        self, sample_cps, sample_lines, tmp_path,
    ):
        output = str(tmp_path / "scatter_needle_both.png")
        generate_cp_scatter_figure(
            "/nonexistent/path", sample_cps, sample_lines, output,
            left_needle_x=120.0, right_needle_x=380.0,
        )
        assert os.path.isfile(output)
        assert os.path.getsize(output) > 1000

    def test_renders_with_one_needle_line(
        self, sample_cps, sample_lines, tmp_path,
    ):
        output = str(tmp_path / "scatter_needle_left_only.png")
        generate_cp_scatter_figure(
            "/nonexistent/path", sample_cps, sample_lines, output,
            left_needle_x=120.0, right_needle_x=None,
        )
        assert os.path.isfile(output)

    def test_backward_compat_without_needle_kwargs(
        self, sample_cps, sample_lines, tmp_path,
    ):
        output = str(tmp_path / "scatter_no_needle.png")
        generate_cp_scatter_figure(
            "/nonexistent/path", sample_cps, sample_lines, output,
        )
        assert os.path.isfile(output)
