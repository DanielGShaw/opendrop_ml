"""Shared helpers for video-mode contact-angle pipelines.

Small, stateless utilities used by both the auto (`video_two_pass`) and
needle-in (`video_two_pass_needle_in`) pipelines:

  * Per-method try/except wrappers that isolate fit failures so one
    algorithm crashing leaves the rest populated.
  * One-shot TensorFlow loader (ML model).
  * Frame-metadata helpers for the unified CSV row.

Package-private (leading underscore); not exported from ``__init__``.
"""

import os
import warnings
from typing import Dict

import numpy as np

from opendrop_ml.modules.fitting.BA_fit import yl_fit
from opendrop_ml.modules.fitting.circular_fit import circular_fit
from opendrop_ml.modules.fitting.ellipse_fit import ellipse_fit
from opendrop_ml.modules.fitting.polynomial_fit import polynomial_fit
from opendrop_ml.utils.config import LEFT_ANGLE, RIGHT_ANGLE
from opendrop_ml.utils.enums import FittingMethod
from opendrop_ml.utils.os import resource_path


N_TANGENT_POINTS = 10


# ---------------------------------------------------------------------------
# Per-method fit wrappers — each isolates its own failure so unrelated
# methods still populate the row. Failures manifest as NaN columns in the
# unified CSV (the writer fills NaN where a method key is missing).
# ---------------------------------------------------------------------------


def _try_polynomial(out: Dict, method: FittingMethod, profile, degree: int) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            angles, *_ = polynomial_fit(
                profile,
                num_points=N_TANGENT_POINTS,
                polynomial_degree=degree,
                display=False,
            )
        out[method] = {
            LEFT_ANGLE: float(angles[0]),
            RIGHT_ANGLE: float(angles[1]),
        }
    except Exception:
        pass


def _try_circular(out: Dict, profile) -> None:
    try:
        angles, *_ = circular_fit(profile, display=False)
        out[FittingMethod.CIRCLE_FIT] = {
            LEFT_ANGLE: float(angles[0]),
            RIGHT_ANGLE: float(angles[1]),
        }
    except Exception:
        pass


def _try_ellipse(out: Dict, profile) -> None:
    try:
        angles, *_ = ellipse_fit(profile, display=False)
        if not angles:
            return
        out[FittingMethod.ELLIPSE_FIT] = {
            LEFT_ANGLE: float(angles[0]),
            RIGHT_ANGLE: float(angles[1]),
        }
    except Exception:
        pass


def _try_yl(out: Dict, profile) -> None:
    try:
        result = yl_fit(profile, display=False)
        angles = result[0]
        out[FittingMethod.YL_FIT] = {
            LEFT_ANGLE: float(angles[0]),
            RIGHT_ANGLE: float(angles[1]),
        }
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ML model loader — lazy TF import; called once per run when ML_MODEL
# is enabled.
# ---------------------------------------------------------------------------


def _load_ml_model():
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    model_path = resource_path("modules/ML_model/")
    return tf.keras.models.load_model(model_path)


# ---------------------------------------------------------------------------
# Frame-metadata helpers — populate the unified CSV's filepath and
# timestamp columns. Directories have per-frame filenames but no fps;
# videos have fps but a single source path.
# ---------------------------------------------------------------------------


def _frame_filepath(source_path: str, meta: Dict) -> str:
    if "filename" in meta:
        return os.path.join(source_path, meta["filename"])
    return source_path


def _frame_timestamp(frame_number: int, meta: Dict) -> float:
    fps = meta.get("fps")
    if fps is None or fps <= 0:
        return float("nan")
    return frame_number / fps
