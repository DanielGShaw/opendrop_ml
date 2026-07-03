from opendrop_ml.modules.core.classes import (
    ExperimentalDrop,
    ExperimentalSetup,
    DropData,
)
from opendrop_ml.modules.image.read_image import get_image
from opendrop_ml.modules.image.select_regions import (
    set_drop_region,
    set_surface_line,
    # run_set_surface_line
)
from opendrop_ml.modules.contact_angle.extract_profile import extract_drop_profile
from opendrop_ml.modules.fitting.fits import perform_fits
from opendrop_ml.utils.enums import FittingMethod, ThresholdSelect
from opendrop_ml.utils.config import LEFT_ANGLE, RIGHT_ANGLE, CONTACT_POINTS
from opendrop_ml.utils.os import resource_path

from typing import Callable, Dict, List, Optional
import csv
import math
import os
import copy


# Per-method fit-quality columns appended after left/right angle. These
# surface the RMSE (and, for YL, the Bond number) that perform_fits()
# already computes into contact_angles[method]["errors"]/["bond number"]
# but that both CSV writers used to discard.
_TANGENT_POLY_EXTRA_COLUMNS = ["left RMSE", "right RMSE"]
_CIRCLE_ELLIPSE_EXTRA_COLUMNS = ["fit RMSE"]
_YL_EXTRA_COLUMNS = ["Bond number", "fit RMSE"]


def _extra_columns_for_method(method: FittingMethod) -> List[str]:
    """Column suffixes appended after left/right angle for a fitting method."""
    if method in (FittingMethod.TANGENT_FIT, FittingMethod.POLYNOMIAL_FIT):
        return _TANGENT_POLY_EXTRA_COLUMNS
    if method in (FittingMethod.CIRCLE_FIT, FittingMethod.ELLIPSE_FIT):
        return _CIRCLE_ELLIPSE_EXTRA_COLUMNS
    if method == FittingMethod.YL_FIT:
        return _YL_EXTRA_COLUMNS
    return []


def _extra_values_for_method(method: FittingMethod, method_data: Dict) -> List:
    """Extract extra column values for a method, in the same order as
    _extra_columns_for_method. Missing values are None (callers serialise
    as needed)."""
    errors = method_data.get("errors") or {}
    if method in (FittingMethod.TANGENT_FIT, FittingMethod.POLYNOMIAL_FIT):
        return [errors.get("RMSE left"), errors.get("RMSE right")]
    if method in (FittingMethod.CIRCLE_FIT, FittingMethod.ELLIPSE_FIT):
        return [errors.get("RMSE")]
    if method == FittingMethod.YL_FIT:
        return [method_data.get("bond number"), errors.get("RMSE")]
    return []


class UnifiedCsvWriter:
    """Streams per-frame CA results to a CSV with a fixed schema.

    Columns are a strict superset of the static-CA CSV: four video columns
    (frame_number, timestamp_s, filepath, error_status) followed by left/
    right angle columns (plus each method's fit-quality columns — RMSE, and
    Bond number for YL) for every enabled fitting method. Missing values
    serialise as NaN so downstream tooling sees a stable schema regardless
    of which pipeline produced the row.
    """

    _VIDEO_COLUMNS = (
        "frame_number",
        "timestamp_s",
        "filepath",
        "error_status",
        "detected_tilt_angle",
    )

    def __init__(self, path: str, user_input_data: ExperimentalSetup):
        self.path = path
        self._methods = [
            m for m, enabled in user_input_data.analysis_methods_ca.items() if enabled
        ]
        method_columns = []
        for method in self._methods:
            method_columns.append(f"{method.value} left angle")
            method_columns.append(f"{method.value} right angle")
            for suffix in _extra_columns_for_method(method):
                method_columns.append(f"{method.value} {suffix}")
        self.columns = list(self._VIDEO_COLUMNS) + method_columns

        header_needed = not (
            os.path.exists(self.path) and os.path.getsize(self.path) > 0
        )
        if header_needed:
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(self.columns)

    def write_row(
        self,
        frame_number: int,
        timestamp_s: float,
        filepath: str,
        error_status: str,
        contact_angles: Optional[Dict] = None,
        detected_tilt_angle: float = math.nan,
    ) -> None:
        contact_angles = contact_angles or {}
        row = [
            frame_number,
            timestamp_s,
            filepath,
            error_status,
            detected_tilt_angle,
        ]
        for method in self._methods:
            method_data = contact_angles.get(method) or {}
            left = method_data.get(LEFT_ANGLE)
            right = method_data.get(RIGHT_ANGLE)
            row.append(math.nan if left is None else left)
            row.append(math.nan if right is None else right)
            for value in _extra_values_for_method(method, method_data):
                row.append(math.nan if value is None else value)

        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
            f.flush()


def run_video_analysis(
    user_input_data: ExperimentalSetup,
    writer: UnifiedCsvWriter,
    progress: Callable[[int, int, str], None],
    cancel: Callable[[], bool],
):
    """Dispatch video-mode CA analysis.

    Branches on ``(needle_in_drop, surface_tilted)``:
    - ``(False, False)`` → standard sessile two-pass pipeline.
    - ``(True,  False)`` → needle-in-drop (Isaac-style) two-pass pipeline.
    - ``(False, True)``  → single-pass with auto tilt detection.
    - ``(True,  True)``  → unsupported (raises ``NotImplementedError``).
    """
    key = (user_input_data.needle_in_drop, user_input_data.surface_tilted)
    if key == (False, False):
        return _auto_two_pass(user_input_data, writer, progress, cancel)
    if key == (True, False):
        return _needle_in_two_pass(user_input_data, writer, progress, cancel)
    if key == (False, True):
        return _single_pass_with_tilt(user_input_data, writer, progress, cancel)
    raise NotImplementedError(
        "Combined needle-in-drop and tilted-surface analysis is not "
        "yet supported or validated."
    )


def _auto_two_pass(
    user_input_data: ExperimentalSetup,
    writer: UnifiedCsvWriter,
    progress: Callable[[int, int, str], None],
    cancel: Callable[[], bool],
):
    # Lazy import to keep this module import-time cheap and avoid pulling
    # the full two-pass pipeline (and its heavy dependencies) into callers
    # that only need UnifiedCsvWriter.
    from opendrop_ml.modules.contact_angle.video_two_pass import (
        _auto_two_pass as _impl,
    )

    return _impl(user_input_data, writer, progress, cancel)


def _needle_in_two_pass(
    user_input_data: ExperimentalSetup,
    writer: UnifiedCsvWriter,
    progress: Callable[[int, int, str], None],
    cancel: Callable[[], bool],
):
    # Lazy-import stub — keeps module-level imports cheap so the
    # default ``test_all.py`` run stays fast.
    from opendrop_ml.modules.contact_angle.video_two_pass_needle_in import (
        _needle_in_two_pass as _impl,
    )

    return _impl(user_input_data, writer, progress, cancel)


def _single_pass_with_tilt(
    user_input_data: ExperimentalSetup,
    writer: UnifiedCsvWriter,
    progress: Callable[[int, int, str], None],
    cancel: Callable[[], bool],
):
    # Lazy-import stub. Two-pass tilted variant is gated by
    # ``_USE_TWO_PASS_TILTED`` in ``video_single_pass_tilted``. Keep
    # module-level imports cheap so the default ``test_all.py`` run
    # stays fast.
    from opendrop_ml.modules.contact_angle.video_single_pass_tilted import (
        _USE_TWO_PASS_TILTED,
        _single_pass_with_tilt as _impl,
    )

    if _USE_TWO_PASS_TILTED:
        from opendrop_ml.modules.contact_angle.video_two_pass_tilted import (
            _auto_two_pass_with_tilt_correction,
        )

        return _auto_two_pass_with_tilt_correction(
            user_input_data,
            writer,
            progress,
            cancel,
        )
    return _impl(user_input_data, writer, progress, cancel)


class CaDataProcessor:
    def process_data(
        self,
        fitted_drop_data: DropData,
        user_input_data: ExperimentalSetup,
        callback: Callable,
    ) -> None:

        analysis_methods: Dict[FittingMethod, bool] = dict(
            user_input_data.analysis_methods_ca
        )
        n_frames: int = user_input_data.number_of_frames

        self.results = []

        for i in range(n_frames):
            input_file = user_input_data.import_files[i]
            try:
                raw_experiment = ExperimentalDrop()

                # Load image into the experimental drop object
                get_image(raw_experiment, user_input_data, i)
                set_drop_region(raw_experiment, user_input_data, i + 1)
                extract_drop_profile(raw_experiment, user_input_data)

                # fits performed here if baseline_method is User-selected
                set_surface_line(raw_experiment, user_input_data)

                # these methods don't need tilt correction
                if user_input_data.baseline_method == ThresholdSelect.AUTOMATED:
                    if (
                        analysis_methods[FittingMethod.TANGENT_FIT]
                        or analysis_methods[FittingMethod.POLYNOMIAL_FIT]
                        or analysis_methods[FittingMethod.CIRCLE_FIT]
                        or analysis_methods[FittingMethod.ELLIPSE_FIT]
                    ):
                        perform_fits(
                            raw_experiment,
                            tangent=analysis_methods[FittingMethod.TANGENT_FIT],
                            polynomial=analysis_methods[FittingMethod.POLYNOMIAL_FIT],
                            circle=analysis_methods[FittingMethod.CIRCLE_FIT],
                            ellipse=analysis_methods[FittingMethod.ELLIPSE_FIT],
                        )

                # YL and ML run after the up-front extract_drop_profile + set_surface_line
                # at lines 193-196 have already populated drop_contour from preprocess(.image).
                # ML uses raw_experiment.image directly; YL uses drop_contour. Neither needs
                # cropped_image, so no tilt correction is required at this stage — preprocess
                # already does its own internal tilt correction inside isolate_contour.
                if (
                    analysis_methods[FittingMethod.ML_MODEL]
                    or analysis_methods[FittingMethod.YL_FIT]
                ):
                    if analysis_methods[FittingMethod.YL_FIT]:
                        perform_fits(
                            raw_experiment, yl=analysis_methods[FittingMethod.YL_FIT]
                        )
                    if analysis_methods[FittingMethod.ML_MODEL]:

                        from opendrop_ml.modules.ML_model.prepare_experimental import (
                            prepare_for_model,
                            experimental_pred,
                        )
                        import tensorflow as tf

                        tf.compat.v1.logging.set_verbosity(
                            tf.compat.v1.logging.ERROR
                        )  # to minimise tf warnings
                        model_path = resource_path("modules/ML_model/")
                        model = tf.keras.models.load_model(model_path)

                        try:
                            pred_ds, ml_contact_points = prepare_for_model(
                                raw_experiment.image, return_cps=True
                            )
                        except ValueError as e:
                            print(f"ML preprocessing failed: {e}")
                            raw_experiment.contact_angles[FittingMethod.ML_MODEL] = {}
                            raw_experiment.contact_angles[FittingMethod.ML_MODEL][
                                LEFT_ANGLE
                            ] = None
                            raw_experiment.contact_angles[FittingMethod.ML_MODEL][
                                RIGHT_ANGLE
                            ] = None
                            raw_experiment.contact_angles[FittingMethod.ML_MODEL][
                                "timings"
                            ] = {}
                        else:
                            ML_predictions, timings = experimental_pred(pred_ds, model)
                            raw_experiment.contact_angles[FittingMethod.ML_MODEL] = {}
                            raw_experiment.contact_angles[FittingMethod.ML_MODEL][
                                LEFT_ANGLE
                            ] = ML_predictions[0, 0]
                            raw_experiment.contact_angles[FittingMethod.ML_MODEL][
                                RIGHT_ANGLE
                            ] = ML_predictions[1, 0]
                            raw_experiment.contact_angles[FittingMethod.ML_MODEL][
                                CONTACT_POINTS
                            ] = list(ml_contact_points)
                            raw_experiment.contact_angles[FittingMethod.ML_MODEL][
                                "timings"
                            ] = timings

                self.results.append(copy.deepcopy(raw_experiment.contact_angles))

                if callback:
                    callback(i + 1, raw_experiment)
            except Exception as e:
                raise RuntimeError(
                    f"CA analysis failed at frame {i+1}/{n_frames} for file '{input_file}'"
                ) from e

    def save_result(self, user_input_data: ExperimentalSetup, output_file_path: str):
        methods = [
            m for m, enabled in user_input_data.analysis_methods_ca.items() if enabled
        ]
        columns = ["Filename", "Time (s)"]
        for method in methods:
            columns.append(f"{method.value} left angle")
            columns.append(f"{method.value} right angle")
            for suffix in _extra_columns_for_method(method):
                columns.append(f"{method.value} {suffix}")

        with open(output_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for index, contact_angles in enumerate(self.results):
                filepath = user_input_data.import_files[index]
                if isinstance(filepath, tuple):
                    filepath = filepath[0]
                row = [str(filepath), user_input_data.frame_interval * index]
                for method in methods:
                    method_data = contact_angles.get(method) or {}
                    left = method_data.get(LEFT_ANGLE)
                    right = method_data.get(RIGHT_ANGLE)
                    row.append(math.nan if left is None else left)
                    row.append(math.nan if right is None else right)
                    for value in _extra_values_for_method(method, method_data):
                        row.append(math.nan if value is None else value)
                writer.writerow(row)
