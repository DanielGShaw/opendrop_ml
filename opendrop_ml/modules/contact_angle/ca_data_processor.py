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

from typing import Callable, Dict
import numpy as np
import timeit
import copy

# import os


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
                print(f"\nProcessing frame {i+1} of {n_frames}...")
                print(f"\nProcessing {input_file}")
                time_start = timeit.default_timer()
                raw_experiment = ExperimentalDrop()

                # save image in here...
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
                        print("Performing YL fit...")
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

                print("Extracted outputs:")
                for key1 in raw_experiment.contact_angles.keys():
                    for key2 in raw_experiment.contact_angles[key1].keys():
                        print(key1 + " " + key2 + ": ")
                        print("    ", raw_experiment.contact_angles[key1][key2])
                        print()

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
