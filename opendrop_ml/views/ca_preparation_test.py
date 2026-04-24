from opendrop_ml.views.ca_preparation import CaPreparation

from unittest.mock import patch, MagicMock
import cv2
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
    with patch(
        "opendrop_ml.views.ca_preparation.create_analysis_checklist_cm"
    ) as mock_create:
        mock_create.return_value = MagicMock()
        instance = MagicMock(spec=CaPreparation)
        instance.user_input_data = MagicMock()
        CaPreparation.create_analysis_method_fields(
            instance, parent_frame=mock_create.return_value
        )
        mock_create.assert_called_once()


def test_create_fitting_view_fields():
    with patch(
        "opendrop_ml.views.ca_preparation.create_plotting_checklist"
    ) as mock_create:
        mock_create.return_value = MagicMock()
        instance = MagicMock(spec=CaPreparation)
        instance.user_input_data = MagicMock()
        CaPreparation.create_fitting_view_fields(
            instance, parent_frame=mock_create.return_value
        )
        mock_create.assert_called_once()


def test_create_user_input_fields_passes_toggle_callbacks_in_video_mode():
    with patch("opendrop_ml.views.ca_preparation.create_user_inputs_cm") as mock_create:
        mock_create.return_value = MagicMock()
        instance = MagicMock(spec=CaPreparation)
        instance.user_input_data = MagicMock()
        instance.user_input_data.is_video = True
        instance._on_needle_toggled = MagicMock()
        instance._on_tilt_toggled = MagicMock()
        CaPreparation.create_user_input_fields(
            instance, parent_frame=mock_create.return_value
        )
        _, kwargs = mock_create.call_args
        assert kwargs.get("on_needle_toggle") is instance._on_needle_toggled
        assert kwargs.get("on_surface_tilted") is instance._on_tilt_toggled


def test_create_user_input_fields_omits_callbacks_in_static_mode():
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
        assert kwargs.get("on_surface_tilted") is None


def _preview_instance(needle_on=False, tilt_on=False):
    instance = MagicMock(spec=CaPreparation)
    instance.user_input_data = MagicMock()
    instance.user_input_data.import_files = ["/tmp/fake_video.mp4"]
    instance.user_input_data.is_video = True
    instance.user_input_data.needle_in_drop = needle_on
    instance.user_input_data.surface_tilted = tilt_on
    instance._preview_container = None
    instance.image_app_frame = MagicMock()
    instance._image_handler = MagicMock()
    instance._image_handler.get_fitting_dimensions.return_value = (600, 400)
    instance._pack_preview_image = MagicMock()
    # Bind the real helpers so _render_video_preview sees tuple returns.
    instance._build_needle_overlay = lambda frame: CaPreparation._build_needle_overlay(
        instance, frame
    )
    instance._build_tilt_preview = lambda frame: CaPreparation._build_tilt_preview(
        instance, frame
    )
    return instance


def test_render_video_preview_with_needle_on_invokes_detection():
    instance = _preview_instance(needle_on=True)
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
        CaPreparation._render_video_preview(instance)

    mock_extract.assert_called_once()
    mock_detect.assert_called_once()


def test_render_video_preview_with_both_off_skips_detection():
    instance = _preview_instance(needle_on=False, tilt_on=False)
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "opendrop_ml.views.ca_preparation.get_middle_frame",
        return_value=(fake_frame, 0, {}),
    ), patch(
        "opendrop_ml.views.ca_preparation.extract_edges_two_contours"
    ) as mock_extract, patch(
        "opendrop_ml.views.ca_preparation.detect_vertical_lines"
    ) as mock_detect, patch(
        "opendrop_ml.views.ca_preparation.correct_tilt"
    ) as mock_tilt, patch(
        "opendrop_ml.views.ca_preparation.CTkFrame"
    ), patch(
        "opendrop_ml.views.ca_preparation.CTkLabel"
    ), patch(
        "opendrop_ml.views.ca_preparation.set_light_only_color"
    ):
        CaPreparation._render_video_preview(instance)

    mock_extract.assert_not_called()
    mock_detect.assert_not_called()
    mock_tilt.assert_not_called()


def test_render_video_preview_handles_needle_detection_failure():
    instance = _preview_instance(needle_on=True)
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
        CaPreparation._render_video_preview(instance)

    caption_texts = [call.kwargs.get("text", "") for call in mock_label.call_args_list]
    assert any("Needle not auto-detected" in t for t in caption_texts)


def test_render_video_preview_handles_extract_exception():
    instance = _preview_instance(needle_on=True)
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
        CaPreparation._render_video_preview(instance)

    mock_detect.assert_not_called()
    caption_texts = [call.kwargs.get("text", "") for call in mock_label.call_args_list]
    assert any("Needle not auto-detected" in t for t in caption_texts)
    instance._pack_preview_image.assert_called_once()


def test_render_video_preview_with_tilt_on_shows_rotated_frame():
    instance = _preview_instance(needle_on=False, tilt_on=True)
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    rotated_frame = np.ones((100, 100, 3), dtype=np.uint8)

    with patch(
        "opendrop_ml.views.ca_preparation.get_middle_frame",
        return_value=(fake_frame, 0, {}),
    ), patch(
        "opendrop_ml.views.ca_preparation.correct_tilt",
        return_value=(rotated_frame, 60.0),
    ) as mock_tilt, patch(
        "opendrop_ml.views.ca_preparation.CTkFrame"
    ), patch(
        "opendrop_ml.views.ca_preparation.CTkLabel"
    ) as mock_label, patch(
        "opendrop_ml.views.ca_preparation.set_light_only_color"
    ):
        CaPreparation._render_video_preview(instance)

    mock_tilt.assert_called_once()
    _, kwargs = mock_tilt.call_args
    assert kwargs.get("crop") is True
    instance._pack_preview_image.assert_called_once()
    packed_frame = instance._pack_preview_image.call_args[0][1]
    assert packed_frame is rotated_frame
    caption_texts = [call.kwargs.get("text", "") for call in mock_label.call_args_list]
    assert any("Auto-detected tilt: 60.0" in t for t in caption_texts)


def test_render_video_preview_with_tilt_detection_zero_shows_warning():
    instance = _preview_instance(needle_on=False, tilt_on=True)
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "opendrop_ml.views.ca_preparation.get_middle_frame",
        return_value=(fake_frame, 0, {}),
    ), patch(
        "opendrop_ml.views.ca_preparation.correct_tilt",
        return_value=(fake_frame, 0.0),
    ), patch(
        "opendrop_ml.views.ca_preparation.CTkFrame"
    ), patch(
        "opendrop_ml.views.ca_preparation.CTkLabel"
    ) as mock_label, patch(
        "opendrop_ml.views.ca_preparation.set_light_only_color"
    ):
        CaPreparation._render_video_preview(instance)

    packed_frame = instance._pack_preview_image.call_args[0][1]
    assert packed_frame is fake_frame
    caption_texts = [call.kwargs.get("text", "") for call in mock_label.call_args_list]
    assert any("Tilt not auto-detected" in t for t in caption_texts)


def test_render_video_preview_handles_correct_tilt_exception():
    instance = _preview_instance(needle_on=False, tilt_on=True)
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "opendrop_ml.views.ca_preparation.get_middle_frame",
        return_value=(fake_frame, 0, {}),
    ), patch(
        "opendrop_ml.views.ca_preparation.correct_tilt",
        side_effect=cv2.error("bad input"),
    ), patch(
        "opendrop_ml.views.ca_preparation.CTkFrame"
    ), patch(
        "opendrop_ml.views.ca_preparation.CTkLabel"
    ) as mock_label, patch(
        "opendrop_ml.views.ca_preparation.set_light_only_color"
    ):
        CaPreparation._render_video_preview(instance)

    packed_frame = instance._pack_preview_image.call_args[0][1]
    assert packed_frame is fake_frame
    caption_texts = [call.kwargs.get("text", "") for call in mock_label.call_args_list]
    assert any("Tilt not auto-detected" in t for t in caption_texts)


def test_render_video_preview_combined_flags_shows_block_warning():
    instance = _preview_instance(needle_on=True, tilt_on=True)
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "opendrop_ml.views.ca_preparation.get_middle_frame",
        return_value=(fake_frame, 0, {}),
    ), patch("opendrop_ml.views.ca_preparation.correct_tilt") as mock_tilt, patch(
        "opendrop_ml.views.ca_preparation.extract_edges_two_contours"
    ) as mock_extract, patch(
        "opendrop_ml.views.ca_preparation.CTkFrame"
    ), patch(
        "opendrop_ml.views.ca_preparation.CTkLabel"
    ) as mock_label, patch(
        "opendrop_ml.views.ca_preparation.set_light_only_color"
    ):
        CaPreparation._render_video_preview(instance)

    mock_tilt.assert_not_called()
    mock_extract.assert_not_called()
    packed_frame = instance._pack_preview_image.call_args[0][1]
    assert packed_frame is fake_frame
    caption_texts = [call.kwargs.get("text", "") for call in mock_label.call_args_list]
    assert any("Combined needle-in-drop + tilted-surface" in t for t in caption_texts)


def test_refresh_next_button_state_disables_when_combined():
    instance = MagicMock(spec=CaPreparation)
    instance.user_input_data = MagicMock()
    instance.user_input_data.is_video = True
    instance.user_input_data.needle_in_drop = True
    instance.user_input_data.surface_tilted = True
    next_button = MagicMock()
    instance.master = MagicMock()
    instance.master.next_button = next_button

    CaPreparation._refresh_next_button_state(instance)
    next_button.configure.assert_called_with(state="disabled")

    instance.user_input_data.surface_tilted = False
    CaPreparation._refresh_next_button_state(instance)
    next_button.configure.assert_called_with(state="normal")


def test_toggle_callback_triggers_preview_rerender_and_next_refresh():
    instance = MagicMock(spec=CaPreparation)
    CaPreparation._on_needle_toggled(instance, True)
    instance._render_video_preview.assert_called_once_with()
    instance._refresh_next_button_state.assert_called_once_with()

    instance._render_video_preview.reset_mock()
    instance._refresh_next_button_state.reset_mock()
    CaPreparation._on_tilt_toggled(instance, True)
    instance._render_video_preview.assert_called_once_with()
    instance._refresh_next_button_state.assert_called_once_with()


def test_render_video_preview_tears_down_previous_container():
    instance = _preview_instance(needle_on=False, tilt_on=False)
    previous_container = MagicMock()
    instance._preview_container = previous_container
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch(
        "opendrop_ml.views.ca_preparation.get_middle_frame",
        return_value=(fake_frame, 0, {}),
    ), patch("opendrop_ml.views.ca_preparation.CTkFrame"), patch(
        "opendrop_ml.views.ca_preparation.CTkLabel"
    ), patch(
        "opendrop_ml.views.ca_preparation.set_light_only_color"
    ):
        CaPreparation._render_video_preview(instance)

    previous_container.destroy.assert_called_once()
