from opendrop_ml.views.ca_preparation import CaPreparation

from unittest.mock import patch, MagicMock
import numpy as np
import pytest


@pytest.fixture
def dummy_user_input_data():
    return MagicMock()


@pytest.fixture
def dummy_experimental_drop():
    return MagicMock()


@pytest.fixture
def dummy_parent():
    return MagicMock()


@pytest.fixture
def fake_frame():
    return MagicMock()


def test_create_user_input_fields():
    with patch("opendrop_ml.views.ca_preparation.create_user_inputs_cm") as mock_create:
        mock_create.return_value = MagicMock()
        instance = MagicMock(spec=CaPreparation)
        instance.user_input_data = MagicMock()
        instance.user_input_data.is_video = False
        CaPreparation.create_user_input_fields(
            instance, parent_frame=mock_create.return_value
        )
        mock_create.assert_called_once()


def test_create_analysis_method_fields():
    with patch("opendrop_ml.views.ca_preparation.create_analysis_checklist_cm") as mock_create:
        mock_create.return_value = MagicMock()
        instance = MagicMock(spec=CaPreparation)
        instance.user_input_data = MagicMock()
        CaPreparation.create_analysis_method_fields(
            instance, parent_frame=mock_create.return_value
        )
        mock_create.assert_called_once()


def test_create_fitting_view_fields():
    with patch("opendrop_ml.views.ca_preparation.create_plotting_checklist") as mock_create:
        mock_create.return_value = MagicMock()
        instance = MagicMock(spec=CaPreparation)
        instance.user_input_data = MagicMock()
        CaPreparation.create_fitting_view_fields(
            instance, parent_frame=mock_create.return_value
        )
        mock_create.assert_called_once()


def test_create_user_input_fields_passes_needle_callback_in_video_mode():
    with patch("opendrop_ml.views.ca_preparation.create_user_inputs_cm") as mock_create:
        mock_create.return_value = MagicMock()
        instance = MagicMock(spec=CaPreparation)
        instance.user_input_data = MagicMock()
        instance.user_input_data.is_video = True
        instance._on_needle_toggled = MagicMock()
        CaPreparation.create_user_input_fields(
            instance, parent_frame=mock_create.return_value
        )
        _, kwargs = mock_create.call_args
        assert kwargs.get("on_needle_toggle") is instance._on_needle_toggled


def test_create_user_input_fields_omits_callback_in_static_mode():
    with patch("opendrop_ml.views.ca_preparation.create_user_inputs_cm") as mock_create:
        mock_create.return_value = MagicMock()
        instance = MagicMock(spec=CaPreparation)
        instance.user_input_data = MagicMock()
        instance.user_input_data.is_video = False
        CaPreparation.create_user_input_fields(
            instance, parent_frame=mock_create.return_value
        )
        _, kwargs = mock_create.call_args
        assert kwargs.get("on_needle_toggle") is None


def _preview_instance():
    instance = MagicMock(spec=CaPreparation)
    instance.user_input_data = MagicMock()
    instance.user_input_data.import_files = ["/tmp/fake_video.mp4"]
    instance._preview_container = None
    instance.image_app_frame = MagicMock()
    instance._image_handler = MagicMock()
    instance._image_handler.get_fitting_dimensions.return_value = (600, 400)
    instance._pack_preview_image = MagicMock()
    # Bind the real helper so _render_needle_preview sees tuple returns.
    instance._build_needle_overlay = lambda frame: CaPreparation._build_needle_overlay(
        instance, frame
    )
    return instance


def test_render_needle_preview_with_needle_on_invokes_detection():
    instance = _preview_instance()
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "opendrop_ml.views.ca_preparation.get_middle_frame",
        return_value=(fake_frame, 0, {}),
    ), patch(
        "opendrop_ml.views.ca_preparation.extract_edges_two_contours",
        return_value=(np.zeros((5, 2), dtype=int), np.zeros((5, 2), dtype=int), 128),
    ) as mock_extract, patch(
        "opendrop_ml.views.ca_preparation.detect_vertical_lines",
        return_value=(
            np.array([[10, 0], [10, 99]]),
            np.array([[90, 0], [90, 99]]),
            None,
            None,
        ),
    ) as mock_detect, patch(
        "opendrop_ml.views.ca_preparation.CTkFrame"
    ), patch(
        "opendrop_ml.views.ca_preparation.CTkLabel"
    ), patch(
        "opendrop_ml.views.ca_preparation.set_light_only_color"
    ):
        CaPreparation._render_needle_preview(instance, needle_on=True)

    mock_extract.assert_called_once()
    mock_detect.assert_called_once()


def test_render_needle_preview_with_needle_off_skips_detection():
    instance = _preview_instance()
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "opendrop_ml.views.ca_preparation.get_middle_frame",
        return_value=(fake_frame, 0, {}),
    ), patch(
        "opendrop_ml.views.ca_preparation.extract_edges_two_contours"
    ) as mock_extract, patch(
        "opendrop_ml.views.ca_preparation.detect_vertical_lines"
    ) as mock_detect, patch(
        "opendrop_ml.views.ca_preparation.CTkFrame"
    ), patch(
        "opendrop_ml.views.ca_preparation.CTkLabel"
    ), patch(
        "opendrop_ml.views.ca_preparation.set_light_only_color"
    ):
        CaPreparation._render_needle_preview(instance, needle_on=False)

    mock_extract.assert_not_called()
    mock_detect.assert_not_called()


def test_render_needle_preview_handles_detection_failure():
    instance = _preview_instance()
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "opendrop_ml.views.ca_preparation.get_middle_frame",
        return_value=(fake_frame, 0, {}),
    ), patch(
        "opendrop_ml.views.ca_preparation.extract_edges_two_contours",
        return_value=(np.zeros((5, 2), dtype=int), np.zeros((5, 2), dtype=int), 128),
    ), patch(
        "opendrop_ml.views.ca_preparation.detect_vertical_lines",
        return_value=(None, None, None, None),
    ), patch(
        "opendrop_ml.views.ca_preparation.CTkFrame"
    ), patch(
        "opendrop_ml.views.ca_preparation.CTkLabel"
    ) as mock_label, patch(
        "opendrop_ml.views.ca_preparation.set_light_only_color"
    ):
        CaPreparation._render_needle_preview(instance, needle_on=True)

    caption_texts = [
        call.kwargs.get("text", "")
        for call in mock_label.call_args_list
    ]
    assert any("Needle not auto-detected" in t for t in caption_texts)


def test_render_needle_preview_handles_extract_exception():
    instance = _preview_instance()
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "opendrop_ml.views.ca_preparation.get_middle_frame",
        return_value=(fake_frame, 0, {}),
    ), patch(
        "opendrop_ml.views.ca_preparation.extract_edges_two_contours",
        side_effect=ValueError("no contours"),
    ), patch(
        "opendrop_ml.views.ca_preparation.detect_vertical_lines"
    ) as mock_detect, patch(
        "opendrop_ml.views.ca_preparation.CTkFrame"
    ), patch(
        "opendrop_ml.views.ca_preparation.CTkLabel"
    ) as mock_label, patch(
        "opendrop_ml.views.ca_preparation.set_light_only_color"
    ):
        CaPreparation._render_needle_preview(instance, needle_on=True)

    mock_detect.assert_not_called()
    caption_texts = [
        call.kwargs.get("text", "")
        for call in mock_label.call_args_list
    ]
    assert any("Needle not auto-detected" in t for t in caption_texts)
    instance._pack_preview_image.assert_called_once()


def test_needle_toggle_callback_triggers_preview_rerender():
    instance = MagicMock(spec=CaPreparation)
    CaPreparation._on_needle_toggled(instance, True)
    instance._render_needle_preview.assert_called_once_with(True)
