import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from opendrop_ml.modules.contact_angle import _video_helpers
from opendrop_ml.utils.config import LEFT_ANGLE, RIGHT_ANGLE
from opendrop_ml.utils.enums import FittingMethod


def test_try_polynomial_captures_errors(monkeypatch):
    """The video wrapper must not discard the RMSE errors dict that
    polynomial_fit() already computes (regression: it used to be thrown
    away via `angles, *_ = polynomial_fit(...)`)."""
    errors = {"RMSE left": 0.5, "RMSE right": 0.6}
    monkeypatch.setattr(
        _video_helpers,
        "polynomial_fit",
        lambda profile, **kw: ((45.0, 46.0), None, None, errors, {}),
    )

    out = {}
    _video_helpers._try_polynomial(out, FittingMethod.TANGENT_FIT, profile=None, degree=1)

    assert out[FittingMethod.TANGENT_FIT][LEFT_ANGLE] == 45.0
    assert out[FittingMethod.TANGENT_FIT][RIGHT_ANGLE] == 46.0
    assert out[FittingMethod.TANGENT_FIT]["errors"] == errors


def test_try_circular_captures_errors(monkeypatch):
    errors = {"RMSE": 0.3}
    monkeypatch.setattr(
        _video_helpers,
        "circular_fit",
        lambda profile, display=False: ((40.0, 41.0), None, None, None, errors, {}),
    )

    out = {}
    _video_helpers._try_circular(out, profile=None)

    assert out[FittingMethod.CIRCLE_FIT][LEFT_ANGLE] == 40.0
    assert out[FittingMethod.CIRCLE_FIT][RIGHT_ANGLE] == 41.0
    assert out[FittingMethod.CIRCLE_FIT]["errors"] == errors


def test_try_ellipse_captures_errors(monkeypatch):
    errors = {"RMSE": 0.4}
    monkeypatch.setattr(
        _video_helpers,
        "ellipse_fit",
        lambda profile, display=False: (
            (35.0, 36.0),
            None,
            None,
            None,
            None,
            errors,
            {},
        ),
    )

    out = {}
    _video_helpers._try_ellipse(out, profile=None)

    assert out[FittingMethod.ELLIPSE_FIT][LEFT_ANGLE] == 35.0
    assert out[FittingMethod.ELLIPSE_FIT][RIGHT_ANGLE] == 36.0
    assert out[FittingMethod.ELLIPSE_FIT]["errors"] == errors


def test_try_ellipse_no_angles_is_skipped(monkeypatch):
    """ellipse_fit()'s failure path returns an empty CA list — the wrapper
    must bail out before storing anything, same as before this change."""
    monkeypatch.setattr(
        _video_helpers,
        "ellipse_fit",
        lambda profile, display=False: (
            [],
            [],
            None,
            None,
            None,
            "no line-ellipse intersection",
            {},
        ),
    )

    out = {}
    _video_helpers._try_ellipse(out, profile=None)

    assert FittingMethod.ELLIPSE_FIT not in out


def test_try_yl_captures_bond_number_and_errors(monkeypatch):
    bond_number = 0.42
    errors = {"RMSE": 0.12}
    monkeypatch.setattr(
        _video_helpers,
        "yl_fit",
        lambda profile, display=False: (
            (100.0, 101.0),
            bond_number,
            None,
            None,
            None,
            None,
            errors,
            None,
            {},
        ),
    )

    out = {}
    _video_helpers._try_yl(out, profile=None)

    assert out[FittingMethod.YL_FIT][LEFT_ANGLE] == 100.0
    assert out[FittingMethod.YL_FIT][RIGHT_ANGLE] == 101.0
    assert out[FittingMethod.YL_FIT]["bond number"] == bond_number
    assert out[FittingMethod.YL_FIT]["errors"] == errors


@pytest.mark.parametrize(
    "fn_name, wrapper_name, wrapper_kwargs",
    [
        ("polynomial_fit", "_try_polynomial", {"method": FittingMethod.TANGENT_FIT, "degree": 1}),
        ("circular_fit", "_try_circular", {}),
        ("ellipse_fit", "_try_ellipse", {}),
        ("yl_fit", "_try_yl", {}),
    ],
)
def test_try_wrappers_isolate_failures(monkeypatch, fn_name, wrapper_name, wrapper_kwargs):
    """A raising fit function leaves the method key absent, not partially set."""

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(_video_helpers, fn_name, boom)

    out = {}
    getattr(_video_helpers, wrapper_name)(out, profile=None, **wrapper_kwargs)

    assert out == {}


if __name__ == "__main__":
    pytest.main()
