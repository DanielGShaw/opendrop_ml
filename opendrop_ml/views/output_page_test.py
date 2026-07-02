from opendrop_ml.views.output_page import OutputPage, _build_inspection_hint

from unittest import mock
import pytest


class DummyUserInput:
    def __init__(self):
        self.output_directory = None
        self.filename = None
        self.needle_in_drop = False
        self.surface_tilted = False


@pytest.fixture
def output_page():
    dummy_input = DummyUserInput()

    # Completely bypass __init__
    page = OutputPage.__new__(OutputPage)

    # Inject only the attributes we need for testing
    page.user_input_data = dummy_input
    page.location_entry = mock.Mock()
    page.filename_var = mock.Mock()
    page.plot_summary_label = mock.Mock()
    page.check_vars = [mock.Mock(get=mock.Mock(
        return_value="on")) for _ in range(3)]

    return page, dummy_input


def test_browse_location_manual(output_page):
    page, dummy_input = output_page

    page.location_entry.get.return_value = "C:/Users/Test"
    dummy_input.output_directory = "C:/Users/Test"

    assert page.location_entry.get() == "C:/Users/Test"
    assert dummy_input.output_directory == "C:/Users/Test"


def test_on_filename_change(output_page):
    page, dummy_input = output_page

    page.filename_var.get.return_value = "result_file"
    page.on_filename_change()
    assert dummy_input.filename == "result_file"


def test_inspection_hint_default():
    hint = _build_inspection_hint(DummyUserInput())
    assert "preprocessing" in hint
    assert "needle detection" not in hint
    assert "tilt correction" not in hint


def test_inspection_hint_needle_in_drop():
    ui = DummyUserInput()
    ui.needle_in_drop = True
    hint = _build_inspection_hint(ui)
    assert "needle detection" in hint
    assert "tilt correction" not in hint


def test_inspection_hint_surface_tilted():
    ui = DummyUserInput()
    ui.surface_tilted = True
    hint = _build_inspection_hint(ui)
    assert "tilt correction" in hint
    assert "needle detection" not in hint


def test_inspection_hint_needle_precedence_when_both_flags_set():
    """Defence-in-depth: the router blocks (True, True), but if the UI
    state ever reaches output_page with both flags, needle wins — matches
    the router's first-match branching."""
    ui = DummyUserInput()
    ui.needle_in_drop = True
    ui.surface_tilted = True
    hint = _build_inspection_hint(ui)
    assert "needle detection" in hint
    assert "tilt correction" not in hint
