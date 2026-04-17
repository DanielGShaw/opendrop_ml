from opendrop_ml.views.acquisition import Acquisition

from unittest.mock import patch, MagicMock
from unittest import mock
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_ctk():
    with mock.patch("customtkinter.CTk"):
        yield


@pytest.fixture
def mock_user_input():
    class MockUserInput:
        image_source = "Local images"
        import_files = []
        frame_interval = 0
        edgefinder = ""
        number_of_frames = 0
        is_video = False

    return MockUserInput()


@pytest.fixture
def component(mock_user_input):
    with patch("opendrop_ml.views.acquisition.Acquisition.__init__", return_value=None):
        component = Acquisition.__new__(Acquisition)
        component.image_source = MagicMock()
        component.frame_interval = MagicMock()
        component.edgefinder = MagicMock()
        component.choose_files_button = MagicMock()
        component.choose_files_label = MagicMock()
        component.initialize_image_display = MagicMock()
        component.load_image = MagicMock()
        component.update_index_entry = MagicMock()
        component.name_label = MagicMock()
        component.current_index = 0
        component.current_image = None
        component.user_input_data = mock_user_input
        component.images_frame = MagicMock()
        component.display_image = MagicMock()
        component.create_empty_image_placeholder = MagicMock()
        component.input_mode = "Static"
        component.video_source = "Video File"
        component.function_type = MagicMock()
        component.video_source_dropdown = MagicMock()
        component.image_handler = MagicMock()
        return component


def test_component_initialization(component):
    assert hasattr(component, "image_source")
    assert hasattr(component, "frame_interval")
    assert hasattr(component, "edgefinder")
    assert hasattr(component, "choose_files_button")


def test_update_frame_interval(component, mock_user_input):
    component.frame_interval.get_value = MagicMock(return_value=5)
    component.update_frame_interval()
    assert mock_user_input.frame_interval == 5


def test_change_image_updates_index(component, mock_user_input):
    mock_user_input.import_files = ["img1", "img2", "img3"]
    mock_user_input.number_of_frames = 3
    component.current_index = 0
    component.load_image = MagicMock()
    component.update_index_entry = MagicMock()
    component.name_label = MagicMock()

    component.change_image(1)
    assert component.current_index == 1
    component.load_image.assert_called_once_with("img2")


def test_update_input_mode_video(component, mock_user_input):
    from opendrop_ml.utils.enums import FunctionType

    component.function_type = FunctionType.CONTACT_ANGLE
    component.update_input_mode("Video")

    assert mock_user_input.is_video is True
    assert component.input_mode == "Video"
    component.video_source_dropdown.label.grid.assert_called()
    component.video_source_dropdown.optionmenu.grid.assert_called()
    component.frame_interval.label.grid_remove.assert_called()
    component.frame_interval.entry.grid_remove.assert_called()
    component.edgefinder.label.grid_remove.assert_called()
    component.edgefinder.optionmenu.grid_remove.assert_called()


def test_update_input_mode_static(component, mock_user_input):
    from opendrop_ml.utils.enums import FunctionType

    component.function_type = FunctionType.CONTACT_ANGLE
    component.update_input_mode("Video")
    component.update_input_mode("Static")

    assert mock_user_input.is_video is False
    assert component.input_mode == "Static"
    component.video_source_dropdown.label.grid_remove.assert_called()
    component.video_source_dropdown.optionmenu.grid_remove.assert_called()
    component.frame_interval.label.grid.assert_called()
    component.frame_interval.entry.grid.assert_called()
    component.edgefinder.label.grid.assert_called()
    component.edgefinder.optionmenu.grid.assert_called()


@patch("opendrop_ml.views.acquisition.get_middle_frame")
@patch("opendrop_ml.views.acquisition.get_frame_count", return_value=100)
@patch("opendrop_ml.views.acquisition.filedialog")
def test_select_video_file_sets_input(
    mock_dialog, mock_frame_count, mock_middle_frame, component, mock_user_input
):
    mock_dialog.askopenfilename.return_value = "/path/to/video.mp4"
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_middle_frame.return_value = (dummy_frame, 50, {"fps": 30.0})
    component._show_video_preview = MagicMock()

    component.input_mode = "Video"
    component.video_source = "Video File"
    component.select_files()

    assert mock_user_input.import_files == ["/path/to/video.mp4"]
    assert mock_user_input.number_of_frames == 100
    assert mock_user_input.is_video is True


@patch("opendrop_ml.views.acquisition.messagebox")
@patch("opendrop_ml.views.acquisition.os.listdir", return_value=["readme.txt"])
@patch("opendrop_ml.views.acquisition.filedialog")
def test_select_image_folder_empty_rejected(
    mock_dialog, mock_listdir, mock_msgbox, component, mock_user_input
):
    mock_dialog.askdirectory.return_value = "/path/to/empty_folder"

    component.input_mode = "Video"
    component.video_source = "Image Folder"
    component.select_files()

    mock_msgbox.showerror.assert_called_once()
    assert mock_user_input.import_files is None or mock_user_input.import_files == []


@patch("opendrop_ml.views.acquisition.get_middle_frame")
@patch("opendrop_ml.views.acquisition.get_frame_count", return_value=50)
@patch("opendrop_ml.views.acquisition.os.listdir", return_value=["img001.png", "img002.png"])
@patch("opendrop_ml.views.acquisition.filedialog")
def test_select_image_folder_valid(
    mock_dialog, mock_listdir, mock_frame_count, mock_middle_frame,
    component, mock_user_input,
):
    mock_dialog.askdirectory.return_value = "/path/to/images"
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_middle_frame.return_value = (dummy_frame, 25, {"filename": "img025.png"})
    component._show_video_preview = MagicMock()

    component.input_mode = "Video"
    component.video_source = "Image Folder"
    component.select_files()

    assert mock_user_input.import_files == ["/path/to/images"]
    assert mock_user_input.number_of_frames == 50
    assert mock_user_input.is_video is True


def test_reset_on_mode_switch(component, mock_user_input):
    from opendrop_ml.utils.enums import FunctionType

    component.function_type = FunctionType.CONTACT_ANGLE
    mock_user_input.import_files = ["/some/video.mp4"]
    mock_user_input.number_of_frames = 100
    mock_user_input.is_video = True
    component.input_mode = "Video"

    component.update_input_mode("Static")

    assert mock_user_input.import_files is None
    assert mock_user_input.number_of_frames == 0
    assert mock_user_input.is_video is False
