from opendrop_ml.modules.image.select_regions import (
    set_drop_region,
    set_surface_line,
    user_line,
    draw_rectangle,
    draw_line,
    optimized_path,
    intersection,
)
from opendrop_ml.modules.core.classes import ExperimentalSetup, ExperimentalDrop
from opendrop_ml.modules.fitting.fits import perform_fits
from opendrop_ml.utils.enums import RegionSelect, ThresholdSelect

from itertools import cycle
from unittest.mock import patch, MagicMock
from numpy.testing import assert_array_equal
import numpy as np
import unittest
import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestSelectRegions(unittest.TestCase):

    @patch("opendrop_ml.modules.preprocessing.preprocessing.isolate_contour")
    def test_set_drop_region_auto(self, mock_isolate_contour: MagicMock):
        # Create mock return value
        # Simulate a blank image of size 100x100
        mock_image = np.zeros((100, 100, 3))
        # isolate_contour returns (img_processed, longest_contour, drop_only, info)
        # info['bounds'] is [top, bottom, left, right]
        mock_isolate_contour.return_value = (
            mock_image,
            np.zeros((10, 2)),
            np.zeros((5, 2)),
            {"bounds": [20, 80, 10, 90]},
        )

        # Simulate experimental_drop and experimental_setup objects
        experimental_drop = ExperimentalDrop()
        experimental_drop.image = mock_image
        experimental_setup = ExperimentalSetup()
        experimental_setup.drop_id_method = RegionSelect.AUTOMATED
        experimental_setup.screen_resolution = [1, 1]

        # Call the function to be tested
        set_drop_region(experimental_drop, experimental_setup, index=0)

        # Check if isolate_contour was called correctly
        mock_isolate_contour.assert_called_once_with(mock_image)

        # set_drop_region applies info["bounds"] to crop isolate_contour's
        # full-image return (top=20, bottom=80, left=10, right=90).
        self.assertTrue(
            np.array_equal(experimental_drop.cropped_image, mock_image[20:80, 10:90])
        )
        self.assertEqual(experimental_setup.drop_region, [(10, 20), (90, 80)])

    @patch("opendrop_ml.modules.image.select_regions.user_roi")
    @patch("opendrop_ml.modules.image.select_regions.image_crop")
    def test_set_drop_region_user_selected(
        self, mock_image_crop: MagicMock, mock_user_roi: MagicMock
    ):
        # Create mock return value
        # Simulate a blank image of size 100x100
        mock_image = np.zeros((100, 100, 3))
        # Simulate the user-selected region
        mock_user_roi.return_value = [(10, 20), (90, 80)]

        # Simulate the cropped image as a 3D array
        # Assume the cropped result is still 3D
        mock_cropped_image = np.zeros((60, 80, 3))
        mock_image_crop.return_value = mock_cropped_image

        # Simulate experimental_drop and experimental_setup objects
        experimental_drop = ExperimentalDrop()
        experimental_drop.image = mock_image
        experimental_setup = ExperimentalSetup()
        experimental_setup.drop_id_method = RegionSelect.USER_SELECTED
        experimental_setup.screen_resolution = [1, 1]

        # Call the function to be tested
        set_drop_region(experimental_drop, experimental_setup, index=1)

        # Check if user_roi was called correctly
        mock_user_roi.assert_called_once_with(
            mock_image, "Select drop region for Image 1", 0.008, [0, 0]
        )

        # Verify if image_crop was called correctly
        mock_image_crop.assert_called_once_with(mock_image, [(10, 20), (90, 80)])

        # Check if experimental_drop and experimental_setup handled user_roi's return value correctly
        self.assertEqual(experimental_setup.drop_region, [(10, 20), (90, 80)])
        self.assertEqual(
            experimental_drop.cropped_image.shape, (60, 80, 3)
        )  # Verify the cropped image size


class TestSurfaceLine(unittest.TestCase):

    @patch("opendrop_ml.modules.image.select_regions.preprocess")
    def test_set_surface_line_auto(self, mock_preprocess):
        # preprocess returns (img_processed, CPs_chosen, longest_contour)
        mock_contour = np.array([[0, 0], [1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
        mock_CPs = np.array([[1, 2], [4, 8]])
        mock_preprocess.return_value = (
            np.zeros((100, 100, 3), dtype=np.uint8),
            mock_CPs,
            mock_contour,
        )

        # Simulate experimental_drop and experimental_setup objects
        experimental_drop = MagicMock()
        experimental_drop.image = np.zeros((100, 100, 3), dtype=np.uint8)
        experimental_setup = MagicMock()
        experimental_setup.baseline_method = ThresholdSelect.AUTOMATED

        # Call the function to be tested
        set_surface_line(experimental_drop, experimental_setup)

        # Check if preprocess was called correctly
        mock_preprocess.assert_called_once_with(experimental_drop.image, display=False)

        # Verify drop_contour is the slice between the two CPs
        expected_contour = mock_contour[1:5]  # idx 1 to 4 inclusive
        self.assertTrue(
            np.array_equal(experimental_drop.drop_contour, expected_contour)
        )
        self.assertTrue(np.array_equal(experimental_drop.contact_points, mock_CPs))


class TestDrawRectangle(unittest.TestCase):

    @patch("opendrop_ml.modules.image.select_regions.cv2.rectangle")
    @patch.dict(
        "opendrop_ml.modules.image.select_regions.__dict__",
        {"image_TEMP": np.zeros((100, 100, 3)), "img": np.zeros((100, 100, 3))},
    )
    def test_draw_rectangle(self, mock_rectangle):
        global drawing, ix, iy, fx, fy
        # Initialize local variables
        drawing = False
        ix, iy = 0, 0
        fx, fy = 0, 0

        # Simulate mouse button down event
        draw_rectangle(cv2.EVENT_LBUTTONDOWN, 10, 20, None, None)
        # No rectangle should be drawn when mouse is pressed down
        mock_rectangle.assert_not_called()

        # Simulate mouse move event, cv2.rectangle should be called
        draw_rectangle(cv2.EVENT_MOUSEMOVE, 30, 40, None, None)
        called_args = mock_rectangle.call_args
        assert np.array_equal(
            called_args[0][0], np.zeros((100, 100, 3))
        )  # Manually compare numpy arrays
        assert called_args[0][1] == (10, 20)
        assert called_args[0][2] == (30, 40)
        assert called_args[0][3] == (0, 0, 255)
        assert called_args[0][4] == 2

        # Simulate mouse button release event, cv2.rectangle should be called again
        draw_rectangle(cv2.EVENT_LBUTTONUP, 50, 60, None, None)
        called_args = mock_rectangle.call_args
        assert np.array_equal(
            called_args[0][0], np.zeros((100, 100, 3))
        )  # Manually compare numpy arrays
        assert called_args[0][1] == (10, 20)
        assert called_args[0][2] == (50, 60)
        assert called_args[0][3] == (0, 255, 0)
        assert called_args[0][4] == 2


class TestDrawLine(unittest.TestCase):

    @patch("opendrop_ml.modules.image.select_regions.cv2.line")
    @patch.dict(
        "opendrop_ml.modules.image.select_regions.__dict__",
        {"image_TEMP": np.zeros((100, 100, 3)), "img": np.zeros((100, 100, 3))},
    )
    def test_draw_line(self, mock_line):
        global drawing, ix, iy, fx, fy
        # Initialize local variables
        drawing = False
        ix, iy = 0, 0
        fx, fy = 0, 0

        # Simulate mouse button down event
        draw_line(cv2.EVENT_LBUTTONDOWN, 10, 20, None, None)
        mock_line.assert_not_called()  # No line should be drawn when mouse is pressed down

        # Simulate mouse move event, cv2.line should be called
        draw_line(cv2.EVENT_MOUSEMOVE, 30, 40, None, None)
        args, kwargs = mock_line.call_args
        # Compare first argument
        assert_array_equal(args[0], np.zeros((100, 100, 3)))
        self.assertEqual(args[1], (10, 20))  # Compare start point
        self.assertEqual(args[2], (30, 40))  # Compare end point
        self.assertEqual(args[3], (0, 0, 255))  # Compare color
        self.assertEqual(args[4], 2)  # Compare line thickness

        # Simulate mouse button release event, cv2.line should be called again
        draw_line(cv2.EVENT_LBUTTONUP, 50, 60, None, None)
        args, kwargs = mock_line.call_args
        # Compare first argument
        assert_array_equal(args[0], np.zeros((100, 100, 3)))
        self.assertEqual(args[1], (10, 20))  # Compare start point
        self.assertEqual(args[2], (50, 60))  # Compare end point
        self.assertEqual(args[3], (0, 255, 0))  # Compare color
        self.assertEqual(args[4], 2)  # Compare line thickness


class TestOptimizedPath(unittest.TestCase):

    def test_optimized_path_with_mock_distance(self):
        coords = [(0, 0), (1, 2), (3, 4), (5, 6)]
        start = (0, 0)

        # Use patch to mock the behavior of distance function
        with patch(
            "opendrop_ml.modules.image.select_regions.distance", side_effect=cycle([1])
        ) as mock_distance:
            result = optimized_path(coords, start)

            # Add debug output
            print("Distance call count:", mock_distance.call_count)

            # Expected call count is 6 times
            self.assertEqual(mock_distance.call_count, 6)

    def test_optimized_path_without_start(self):
        coords = [(0, 0), (1, 2), (3, 4), (5, 6)]

        # Verify when start is not provided, it defaults to the first point
        result = optimized_path(coords)
        expected_result = np.array([(0, 0), (1, 2), (3, 4), (5, 6)])
        np.testing.assert_array_equal(result, expected_result)

    def test_optimized_path_with_single_point(self):
        coords = [(0, 0)]

        # Handle the case of a single point, the result should return the input single point
        result = optimized_path(coords)
        expected_result = np.array([(0, 0)])
        np.testing.assert_array_equal(result, expected_result)


if __name__ == "__main__":
    unittest.main()
