from opendrop_ml.modules.core.classes import ExperimentalDrop, ExperimentalSetup, DropData
from opendrop_ml.modules.image.read_image import get_image
from opendrop_ml.modules.image.select_regions import (
    set_drop_region,
    set_surface_line,
    correct_tilt,
    # run_set_surface_line
)
from opendrop_ml.modules.contact_angle.extract_profile import extract_drop_profile
from opendrop_ml.modules.fitting.fits import perform_fits
from opendrop_ml.utils.enums import FittingMethod, ThresholdSelect
from opendrop_ml.utils.config import LEFT_ANGLE, RIGHT_ANGLE
from opendrop_ml.utils.os import resource_path

from typing import Callable, Dict, Optional
import csv
import math
import os
import numpy as np
import copy


class UnifiedCsvWriter:
    """Streams per-frame CA results to a CSV with a fixed schema.

    Columns are a strict superset of the static-CA CSV: four video columns
    (frame_number, timestamp_s, filepath, error_status) followed by left/
    right angle columns for every enabled fitting method. Missing values
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
            m
            for m, enabled in user_input_data.analysis_methods_ca.items()
            if enabled
        ]
        method_columns = []
        for method in self._methods:
            method_columns.append(f"{method.value} left angle")
            method_columns.append(f"{method.value} right angle")
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
            user_input_data, writer, progress, cancel,
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

                # YL fit and ML model need tilt correction
                if (
                    analysis_methods[FittingMethod.ML_MODEL]
                    or analysis_methods[FittingMethod.YL_FIT]
                ):
                    correct_tilt(raw_experiment, user_input_data)
                    extract_drop_profile(raw_experiment, user_input_data)
                    if user_input_data.baseline_method == ThresholdSelect.AUTOMATED:
                        set_surface_line(raw_experiment, user_input_data)

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
                            pred_ds = prepare_for_model(raw_experiment.image)
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
        for index, contact_angles in enumerate(self.results):
            out = []
            filepath = user_input_data.import_files[index]
            if isinstance(filepath, tuple):
                filepath = filepath[0]
            out.append(str(filepath))
            out.append(user_input_data.frame_interval * index)
            header = []
            header.append("Filename,")
            header.append("Time (s),")
            for key1 in contact_angles.keys():
                for key2 in contact_angles[key1].keys():
                    if "angle" in key2:
                        header.append(key1 + " " + key2 + ",")
                        out.append(str(contact_angles[key1][key2]))
            string = ""
            for heading in header:
                string = string + heading
            string = string[:-1]
            array = np.array(out)
            array = array.reshape(1, array.shape[0])

            with open(output_file_path, "a") as f:
                if index == 0:
                    f.write(string + "\n")
                # for val in array[0]:
                for val in out:
                    f.write(str(val) + ",")
                f.write("\n")
