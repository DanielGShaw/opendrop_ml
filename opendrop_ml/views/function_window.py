from opendrop_ml.modules.contact_angle.ca_data_processor import CaDataProcessor
from opendrop_ml.modules.ift.ift_data_processor import IftDataProcessor
from opendrop_ml.modules.core.classes import ExperimentalSetup, ExperimentalDrop, DropData
from opendrop_ml.utils.os import resource_path, is_windows
from opendrop_ml.views.helper.validation import (
    validate_user_input_data_ift,
    validate_user_input_data_cm,
    validate_frame_interval,
)
from opendrop_ml.views.helper.style import get_color, set_light_only_color
from opendrop_ml.views.helper.theme import LIGHT_MODE
from opendrop_ml.views.navigation import create_navigation
from opendrop_ml.views.acquisition import Acquisition
from opendrop_ml.views.ift_preparation import IftPreparation
from opendrop_ml.views.ift_analysis import IftAnalysis
from opendrop_ml.views.ca_preparation import CaPreparation
from opendrop_ml.views.ca_analysis import CaAnalysis
from opendrop_ml.views.main_window import MainWindow
from opendrop_ml.views.output_page import OutputPage
from opendrop_ml.utils.enums import FunctionType, Stage, Move, RegionSelect

from customtkinter import CTkFrame, CTkButton, CTkToplevel, get_appearance_mode
from tkinter import messagebox, PhotoImage
from typing import List, Callable
from datetime import datetime
import os
import queue
import shutil
import threading
import traceback


def call_user_input(
    function_type: FunctionType, fitted_drop_data: DropData, main_window: MainWindow
):
    FunctionWindow(function_type, fitted_drop_data, main_window)


class FunctionWindow(CTkToplevel):
    main_window: MainWindow
    ca_processor: CaDataProcessor
    ift_processor: IftDataProcessor
    acquisition_frame: Acquisition
    ca_preparation_frame: CaPreparation
    ca_analysis_frame: CaAnalysis
    ift_preparation_frame: IftPreparation
    ift_analysis_frame: IftAnalysis
    output_frame: OutputPage
    after_ids: List[int]
    button_frame: CTkFrame
    back_button: CTkButton
    next_button: CTkButton
    save_button: CTkButton
    stages: List[Stage]
    current_stage: Stage
    next_stage: Callable[[], None]
    prev_stage: Callable[[], None]

    def __init__(
        self,
        function_type: FunctionType,
        fitted_drop_data: DropData,
        main_window: MainWindow,
    ):
        super().__init__()  # Call the parent class constructor
        self.title(function_type.value)
        self.geometry("1000x750")
        self.minsize(1000, 750)

        # main window
        self.main_window = main_window
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # after callback
        self.after_ids = []
        self._is_closing = False
        if get_appearance_mode() == LIGHT_MODE:
            self.FG_COLOR: str = get_color("background")
        else:
            self.FG_COLOR = self.cget("fg_color")

        set_light_only_color(self, "background")

        self.ca_processor = CaDataProcessor()
        self.ift_processor = IftDataProcessor()

        user_input_data: ExperimentalSetup = ExperimentalSetup()
        user_input_data.from_yaml(resource_path("user_config.yaml"))
        experimental_drop: ExperimentalDrop = ExperimentalDrop()

        user_input_data.screen_resolution = [
            self.winfo_screenwidth(),
            self.winfo_screenheight(),
        ]

        self.widgets(
            function_type, user_input_data, experimental_drop, fitted_drop_data
        )

        self.stages = list(Stage)
        self.current_stage = Stage.ACQUISITION

        self.mainloop()  # Start the main loop

    def widgets(
        self,
        function_type: FunctionType,
        user_input_data: ExperimentalSetup,
        experimental_drop: ExperimentalDrop,
        fitted_drop_data: DropData,
    ):
        # Create the navigation bar (progress bar style)
        self.next_stage, self.prev_stage = create_navigation(self)

        # Initialise frame for first stage
        self.acquisition_frame = Acquisition(
            self, user_input_data, function_type, fg_color=self.FG_COLOR
        )
        self.acquisition_frame.pack(fill="both", expand=True)

        # Frame for navigation buttons
        self.button_frame = CTkFrame(self)
        set_light_only_color(self.button_frame, "outerframe")
        self.button_frame.pack(side="bottom", fill="x", pady=10)

        # Add navigation buttons to the button frame
        self.back_button = CTkButton(
            self.button_frame,
            text="Back",
            command=lambda: self.back(function_type, user_input_data),
        )

        self.next_button = CTkButton(
            self.button_frame,
            text="Next",
            command=lambda: self.next(
                function_type, user_input_data, experimental_drop, fitted_drop_data
            ),
        )
        self.next_button.pack(side="right", padx=10, pady=10)

        # Add save button for OutputPage (initially hidden)
        self.save_button = CTkButton(
            self.button_frame,
            text="Save",
            command=lambda: self.save_output(function_type, user_input_data),
        )

    def back(self, function_type: FunctionType, user_input_data: ExperimentalSetup):
        self.update_stage(Move.Back.value)
        # Go back to the previous screen
        if self.current_stage == Stage.ACQUISITION:

            self.back_button.pack_forget()
            self.acquisition_frame.pack(fill="both", expand=True)

            if function_type == FunctionType.INTERFACIAL_TENSION:
                self.ift_preparation_frame.pack_forget()
            else:
                self.ca_preparation_frame.pack_forget()

        elif self.current_stage == Stage.PREPARATION:
            if function_type == FunctionType.INTERFACIAL_TENSION:
                self.ift_preparation_frame.pack(fill="both", expand=True)
                self.ift_analysis_frame.pack_forget()
            else:
                self.ca_preparation_frame.pack(fill="both", expand=True)
                self.ca_analysis_frame.pack_forget()

        elif self.current_stage == Stage.ANALYSIS:
            if function_type == FunctionType.INTERFACIAL_TENSION:
                self.ift_analysis_frame.pack(fill="both", expand=True)
            else:
                self.ca_analysis_frame.pack(fill="both", expand=True)

            self.output_frame.pack_forget()

            # Show the next button and hide the save button when going back
            self.next_button.pack(side="right", padx=10, pady=10)
            self.save_button.pack_forget()

    # def next(self, function_type, user_input_data, experimental_drop, fitted_drop_data):
    #     try:
    #         self.update_stage(Move.Next.value)
    #         # Handle the "Next" button functionality
    #         if self.current_stage == Stage.PREPARATION:

    #             # First check if the user has imported files
    #             if not self.check_import(user_input_data):
    #                 self.update_stage(Move.Back.value)
    #                 messagebox.showinfo(
    #                     "No Selection", "Please select at least one file.", parent=self
    #                 )
    #                 return

    #             # Then check if the frame interval is valid
    #             # if function_type == FunctionType.INTERFACIAL_TENSION:
    #             if not validate_frame_interval(user_input_data):
    #                 self.update_stage(Move.Back.value)
    #                 messagebox.showinfo(
    #                     "Missing", "Frame Interval is required.", parent=self
    #                 )
    #                 return

    #             self.back_button.pack(side="left", padx=10, pady=10)
    #             # self.ift_processor.processPreparation(user_input_data)
    #             # user have selected at least one file
    #             self.acquisition_frame.pack_forget()
    #             # Initialise Preparation frame
    #             if function_type == FunctionType.INTERFACIAL_TENSION:
    #                 self.ift_preparation_frame = IftPreparation(
    #                     self,
    #                     user_input_data,
    #                     experimental_drop,
    #                     self.ift_processor,
    #                     fg_color=self.FG_COLOR,
    #                 )
    #                 self.ift_preparation_frame.pack(fill="both", expand=True)

    #             else:
    #                 self.ca_preparation_frame = CaPreparation(
    #                     self, user_input_data, experimental_drop, fg_color=self.FG_COLOR
    #                 )
    #                 self.ca_preparation_frame.pack(fill="both", expand=True)

    #         elif self.current_stage == Stage.ANALYSIS:
    #             # Validate user input data
    #             if function_type == FunctionType.INTERFACIAL_TENSION:
    #                 validation_messages = validate_user_input_data_ift(user_input_data)

    #             elif function_type == FunctionType.CONTACT_ANGLE:
    #                 validation_messages = validate_user_input_data_cm(
    #                     user_input_data, experimental_drop
    #                 )

    #             if validation_messages:
    #                 self.update_stage(Move.Back.value)
    #                 all_messages = "\n".join(validation_messages)
    #                 # Show a single pop-up message with all validation messages
    #                 messagebox.showinfo("Missing: \n", all_messages, parent=self)
    #             else:
    #                 if function_type == FunctionType.INTERFACIAL_TENSION:
    #                     self.ift_preparation_frame.pack_forget()
    #                     self.ift_analysis_frame = IftAnalysis(
    #                         self,
    #                         user_input_data,
    #                         self.ift_processor,
    #                         fg_color=self.FG_COLOR,
    #                     )
    #                     self.ift_analysis_frame.pack(fill="both", expand=True)
    #                     self.withdraw()
    #                     self.ift_processor.process_data(user_input_data)
    #                     self.deiconify()
    #                 else:
    #                     self.ca_preparation_frame.pack_forget()
    #                     self.ca_analysis_frame = CaAnalysis(
    #                         self, user_input_data, fg_color=self.FG_COLOR
    #                     )
    #                     self.ca_analysis_frame.pack(fill="both", expand=True)

    #                     # analysis the given input data and send the output to the ca_analysis_frame for display
    #                     try:
    #                         self.withdraw()
    #                         self.ca_processor.process_data(
    #                             fitted_drop_data,
    #                             user_input_data,
    #                             callback=self.ca_analysis_frame.receive_output,
    #                         )
    #                     except Exception as e:
    #                         print(f"[Error] Unexpected exception: {e}")
    #                         error_msg = str(e)

    #                         if "array must not contain infs or NaNs" in error_msg or "too many indices for array: array is 1-dimensional, but 2 were indexed" in error_msg:
    #                             messagebox.showerror(
    #                             "Invalid Image",
    #                             f"This image is not for {function_type.value} analysis.\nPlease go back and select another image or application.",
    #                             parent=self,
    #                         )
    #                             self.on_closing()
    #                             return
    #                         elif "list index out of range" in error_msg or "'NoneType' object is not iterable" in error_msg:
    #                             messagebox.showerror(
    #                             "Invalid Region",
    #                             f"The selected region is invalid — no usable droplet was detected or the boundary is incomplete.",
    #                             parent=self,
    #                             )
    #                             self.update_stage(Move.Back.value)
    #                             self.ca_analysis_frame.pack_forget()
    #                             self.ca_preparation_frame.pack(fill="both", expand=True)
    #                             return
    #                         else:
    #                             messagebox.showerror(
    #                                 "Error",
    #                                 f"Error: \n{e}\n\n"
    #                                 "Please try again.",
    #                                 parent=self
    #                             )
    #                             self.on_closing()
    #                             return

    #                     finally:
    #                         self.deiconify()

    #         elif self.current_stage == Stage.OUTPUT:
    #             if function_type == FunctionType.INTERFACIAL_TENSION:
    #                 self.ift_analysis_frame.pack_forget()
    #             else:
    #                 self.ca_analysis_frame.pack_forget()

    #             # Initialise Output frame
    #             self.output_frame = OutputPage(
    #                 self, user_input_data, fg_color=self.FG_COLOR
    #             )

    #             # Show the OutputPage
    #             self.output_frame.pack(fill="both", expand=True)

    #             # Hide the next button and show the save button
    #             self.next_button.pack_forget()
    #             self.save_button.pack(side="right", padx=10, pady=10)
    #     except Exception as e:
    #         print(f"[Error] Unexpected exception: {e}")
    #         error_msg = str(e)

    #         if "cannot unpack non-iterable NoneType" in error_msg or "array must not contain infs or NaNs" in error_msg or "too many indices for array: array is 1-dimensional, but 2 were indexed" in error_msg:
    #             messagebox.showerror(
    #             "Invalid Image",
    #             f"This image is not for {function_type.value} analysis.\nPlease go back and select another image or application.",
    #             parent=self,
    #         )
    #             self.on_closing()
    #             return
    #         elif "Parameter estimatation failed for this data set" in error_msg or "ERKStepEvolve()" in error_msg:
    #             self.update_stage(Move.Back.value)
    #             messagebox.showerror(
    #             "Invalid Region",
    #             f"The selected region is invalid — no usable droplet was detected or the boundary is incomplete.",
    #             parent=self,
    #             )
    #             return
    #         else:
    #             messagebox.showerror(
    #                 "Error",
    #                 f"Error: \n{e}\n\n"
    #                 "Please try again.",
    #                 parent=self
    #             )
    #             self.on_closing()
    #             return
    def next(self, function_type, user_input_data, experimental_drop, fitted_drop_data):
        try:
            self.update_stage(Move.Next.value)

            if self.current_stage == Stage.PREPARATION:
                self._handle_preparation_stage(
                    function_type, user_input_data, experimental_drop)

            elif self.current_stage == Stage.ANALYSIS:

                self._handle_analysis_stage(
                    function_type, user_input_data, experimental_drop, fitted_drop_data)

            elif self.current_stage == Stage.OUTPUT:
                self._handle_output_stage(function_type, user_input_data)

        except Exception as e:
            error_msg = str(e)
            print(error_msg)
            if "cannot unpack non-iterable NoneType object" in error_msg:
                self._show_error(
                    "Invalid Image",
                    "This image is not suitable for Interfacial Tension analysis.\nPlease go back and select another image.",
                )
                self.on_closing()
                return
            self._show_error(
                "Error",
                f"Unexpected Error:\n{e}",
            )
            self.on_closing()

    def _handle_preparation_stage(self, function_type, user_input_data, experimental_drop):
        if not self.check_import(user_input_data):
            self.update_stage(Move.Back.value)
            self._show_info("No Selection", "Please select at least one file.")
            return

        if not validate_frame_interval(user_input_data):
            self.update_stage(Move.Back.value)
            self._show_info("Missing", "Frame Interval is required.")
            return

        self.back_button.pack(side="left", padx=10, pady=10)
        self.acquisition_frame.pack_forget()

        if function_type == FunctionType.INTERFACIAL_TENSION:
            self.ift_preparation_frame = IftPreparation(
                self, user_input_data, experimental_drop, self.ift_processor, fg_color=self.FG_COLOR
            )
            self.ift_preparation_frame.pack(fill="both", expand=True)
        else:
            self.ca_preparation_frame = CaPreparation(
                self, user_input_data, experimental_drop, fg_color=self.FG_COLOR
            )
            self.ca_preparation_frame.pack(fill="both", expand=True)

    def _handle_analysis_stage(self, function_type, user_input_data, experimental_drop, fitted_drop_data):
        if function_type == FunctionType.INTERFACIAL_TENSION:
            messages = validate_user_input_data_ift(user_input_data)
        else:
            messages = validate_user_input_data_cm(
                user_input_data, experimental_drop)

        if messages:
            self.update_stage(Move.Back.value)
            self._show_info("Missing", "\n".join(messages))
            return

        if function_type == FunctionType.CONTACT_ANGLE:
            user_input_data.save_analysis_methods(resource_path("user_config.yaml"))

        if function_type == FunctionType.INTERFACIAL_TENSION:
            self._run_ift_analysis(user_input_data)
        elif user_input_data.is_video:
            self._run_video_analysis(user_input_data)
        else:
            self._run_ca_analysis(user_input_data, fitted_drop_data)

    def _resize_for_analysis(self, function_type, user_input_data):
        """Resize window width to fit the analysis results table."""
        if function_type == FunctionType.CONTACT_ANGLE:
            n_methods = sum(
                1 for v in user_input_data.analysis_methods_ca.values() if v
            )
            # Index column (~80px) + method columns (~110px each) + viz panel (~450px)
            needed_width = 530 + n_methods * 110
        else:
            # IFT has 6 fixed columns; 1100px accommodates them comfortably
            needed_width = 1100

        screen_w = self.winfo_screenwidth()
        width = max(1000, min(needed_width, screen_w - 50))
        height = self.winfo_height()

        self.minsize(width, 750)
        self.geometry(f"{width}x{height}")

    def _run_ift_analysis(self, user_input_data):
        self._resize_for_analysis(FunctionType.INTERFACIAL_TENSION, user_input_data)
        self.ift_preparation_frame.pack_forget()
        self.ift_analysis_frame = IftAnalysis(
            self, user_input_data, self.ift_processor, fg_color=self.FG_COLOR
        )
        self.ift_analysis_frame.pack(fill="both", expand=True)

        try:
            self.withdraw()
            self.ift_processor.process_data(user_input_data)
        except Exception as e:
            error_msg = str(e)
            print(f"[Error] IFT processing failed: {error_msg}")
            log_path = self._log_analysis_exception(
                "IFT",
                e,
                user_input_data,
                {"n_frames": user_input_data.number_of_frames},
            )

            if "Parameter estimation failed" in error_msg or "ERKStepEvolve()" in error_msg:
                self.update_stage(Move.Back.value)
                self.ift_analysis_frame.pack_forget()
                self.ift_preparation_frame.pack(fill="both", expand=True)
                self._show_error(
                    "Invalid Region",
                    "No usable droplet was detected or the boundary is incomplete.\n"
                    f"Debug log saved to:\n{log_path}",
                )
            else:
                self._show_error(
                    "Error",
                    f"Error: \n{e}\n\nDebug log saved to:\n{log_path}",
                )
                self.on_closing()
        finally:
            self._safe_deiconify()

    def _run_ca_analysis(self, user_input_data, fitted_drop_data):
        self._resize_for_analysis(FunctionType.CONTACT_ANGLE, user_input_data)
        self.ca_preparation_frame.pack_forget()
        self.ca_analysis_frame = CaAnalysis(
            self, user_input_data, fg_color=self.FG_COLOR)
        self.ca_analysis_frame.pack(fill="both", expand=True)

        try:
            self.withdraw()
            self.ca_processor.process_data(
                fitted_drop_data,
                user_input_data,
                callback=self.ca_analysis_frame.receive_output,
            )
        except Exception as e:
            error_msg = str(e)
            print(error_msg)
            log_path = self._log_analysis_exception(
                "CONTACT_ANGLE",
                e,
                user_input_data,
                {"n_frames": user_input_data.number_of_frames},
            )
            if any(keyword in error_msg for keyword in [
               "array must not contain infs or NaNs",
               ]):
                self._show_error(
                    "Invalid Image",
                    "This image is not for Contact Angle analysis.\n"
                    "Please go back and select another image or application.\n"
                    f"Debug log saved to:\n{log_path}",
                )
                self.on_closing()
            elif any(keyword in error_msg for keyword in [
                "list index out of range",
                "'NoneType' object is not iterable",
                "float division by zero",
                "too many indices",
                "do not appear to intersect",
            ]):
                self._show_error(
                    "Invalid Region",
                    "No usable droplet detected or the boundary is incomplete.\n"
                    f"Debug log saved to:\n{log_path}",
                )
                self.update_stage(Move.Back.value)
                self.ca_analysis_frame.pack_forget()
                self.ca_preparation_frame.pack(fill="both", expand=True)
            else:
                self._show_error(
                    "Error",
                    f"Error: \n{e}\n\nDebug log saved to:\n{log_path}",
                )
                self.on_closing()
        finally:
            self._safe_deiconify()

    # ------------------------------------------------------------------
    # Video-mode CA analysis
    # ------------------------------------------------------------------

    def _run_video_analysis(self, user_input_data):
        self.ca_preparation_frame.pack_forget()
        self.ca_analysis_frame = CaAnalysis(
            self, user_input_data, fg_color=self.FG_COLOR
        )
        self.ca_analysis_frame.pack(fill="both", expand=True)

        source_path = user_input_data.import_files[0]
        if os.path.isdir(source_path):
            source_stem = os.path.basename(source_path.rstrip(os.sep))
        else:
            source_stem = os.path.splitext(os.path.basename(source_path))[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._video_tmp_dir = os.path.join(
            os.path.expanduser("~"), "OpenDrop", "tmp",
            "{}_{}".format(timestamp, source_stem),
        )
        os.makedirs(self._video_tmp_dir, exist_ok=True)

        csv_path = os.path.join(
            self._video_tmp_dir, "{}_results.csv".format(source_stem)
        )

        from opendrop_ml.modules.contact_angle.ca_data_processor import (
            UnifiedCsvWriter,
            run_video_analysis,
        )

        writer = UnifiedCsvWriter(csv_path, user_input_data)

        self._video_queue = queue.Queue()
        self._video_cancel = threading.Event()
        self._video_source_stem = source_stem

        self.ca_analysis_frame.set_cancel_event(self._video_cancel)

        self.back_button.configure(state="disabled")
        self.next_button.configure(state="disabled")

        def worker():
            try:
                result = run_video_analysis(
                    user_input_data,
                    writer,
                    lambda cur, tot, lbl: self._video_queue.put(
                        ("progress", cur, tot, lbl)
                    ),
                    lambda: self._video_cancel.is_set(),
                )
                self._video_queue.put(("done", result))
            except Exception as e:
                self._video_queue.put(("error", str(e)))

        self._video_thread = threading.Thread(target=worker, daemon=True)
        self._video_thread.start()

        self._poll_video_progress(user_input_data)

    def _poll_video_progress(self, user_input_data):
        if self._is_closing:
            return

        try:
            while True:
                try:
                    msg = self._video_queue.get_nowait()
                except queue.Empty:
                    break

                if msg[0] == "progress":
                    _, current, total, label = msg
                    if self._window_exists():
                        self.ca_analysis_frame.update_video_progress(
                            current, total, label
                        )

                elif msg[0] == "done":
                    self._on_video_complete(user_input_data, msg[1])
                    return

                elif msg[0] == "error":
                    self._on_video_error(msg[1], user_input_data)
                    return
        except Exception as e:
            print("[Warning] video poll error: {}".format(e))

        self._video_poll_id = self.register_after(
            100,
            lambda: self._poll_video_progress(user_input_data),
        )

    def _on_video_complete(self, user_input_data, result):
        status = result.get("status", "unknown")

        if self._window_exists():
            self.ca_analysis_frame.on_video_complete(status)

        if status == "completed":
            self._generate_video_figures(user_input_data, result)
            if self._window_exists():
                self.ca_analysis_frame.on_figures_complete()

        user_input_data._video_tmp_dir = self._video_tmp_dir
        user_input_data._video_status = status

        self.back_button.configure(state="normal")
        self.next_button.configure(state="normal")

    def _generate_video_figures(self, user_input_data, result):
        from opendrop_ml.modules.contact_angle.video_figures import (
            generate_timeseries_figure,
            generate_cp_scatter_figure,
        )

        stem = self._video_source_stem
        csv_path = os.path.join(
            self._video_tmp_dir, "{}_results.csv".format(stem)
        )

        try:
            ts_path = os.path.join(
                self._video_tmp_dir, "{}_timeseries.png".format(stem)
            )
            generate_timeseries_figure(csv_path, ts_path)
        except Exception as e:
            print("[Warning] Timeseries figure generation failed: {}".format(e))

        try:
            scatter_path = os.path.join(
                self._video_tmp_dir, "{}_cp_scatter.png".format(stem)
            )
            source_path = user_input_data.import_files[0]
            generate_cp_scatter_figure(
                source_path=source_path,
                pass1_cps=result.get("pass1_cps"),
                contact_lines={
                    "left": result.get("left_contact_line"),
                    "right": result.get("right_contact_line"),
                },
                output_path=scatter_path,
            )
        except Exception as e:
            print("[Warning] CP scatter figure generation failed: {}".format(e))

    def _on_video_error(self, error_msg, user_input_data):
        log_path = self._log_analysis_exception(
            "VIDEO_CA",
            RuntimeError(error_msg),
            user_input_data,
            {"tmp_dir": getattr(self, "_video_tmp_dir", None)},
        )
        if self._window_exists():
            self.ca_analysis_frame.on_video_complete(
                "error: {}".format(error_msg)
            )
        self.back_button.configure(state="normal")
        self.next_button.configure(state="normal")
        self._show_error(
            "Video Analysis Error",
            "Video analysis failed:\n{}\n\nDebug log saved to:\n{}".format(
                error_msg, log_path
            ),
        )

    def _save_video_output(self, user_input_data):
        # Sync location from the entry widget (user may have typed directly)
        if hasattr(self, "output_frame") and hasattr(self.output_frame, "location_entry"):
            entered_dir = self.output_frame.location_entry.get()
            if entered_dir:
                user_input_data.output_directory = entered_dir

        if not user_input_data.output_directory:
            user_input_data.output_directory = os.path.join(
                os.path.expanduser("~"), "OpenDrop", "outputs"
            )
        if user_input_data.output_directory.startswith("~"):
            user_input_data.output_directory = os.path.expanduser(
                user_input_data.output_directory
            )

        os.makedirs(user_input_data.output_directory, exist_ok=True)

        tmp_dir = getattr(user_input_data, "_video_tmp_dir", None)
        if not tmp_dir or not os.path.isdir(tmp_dir):
            self._show_error("Save Error", "No video analysis results to save.")
            return

        custom_name = user_input_data.filename
        source_stem = getattr(self, "_video_source_stem", "")

        moved_files = []
        for filename in os.listdir(tmp_dir):
            src = os.path.join(tmp_dir, filename)
            if not os.path.isfile(src):
                continue

            if custom_name:
                dst_name = filename.replace(source_stem, custom_name, 1)
            else:
                dst_name = filename

            dst = os.path.join(user_input_data.output_directory, dst_name)
            shutil.move(src, dst)
            moved_files.append(dst)

        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

        files_str = "\n".join(
            "  {}".format(os.path.basename(f)) for f in moved_files
        )
        self._show_info(
            "Save Successful",
            "Results saved to:\n{}\n\nFiles:\n{}".format(
                user_input_data.output_directory, files_str
            ),
        )
        self.on_closing()

    def _handle_output_stage(self, function_type, user_input_data):
        if function_type == FunctionType.INTERFACIAL_TENSION:
            self.ift_analysis_frame.pack_forget()
        else:
            self.ca_analysis_frame.pack_forget()

        self.output_frame = OutputPage(
            self, user_input_data, fg_color=self.FG_COLOR)
        self.output_frame.pack(fill="both", expand=True)

        self.next_button.pack_forget()
        self.save_button.pack(side="right", padx=10, pady=10)

    def save_output(
        self, function_type: FunctionType, user_input_data: ExperimentalSetup
    ):
        if user_input_data.is_video:
            self._save_video_output(user_input_data)
            return

        if not user_input_data.output_directory:
            user_input_data.output_directory = os.path.join(
                os.path.join(os.path.expanduser("~")), "OpenDrop", "outputs"
            )

        if user_input_data.output_directory.startswith("~"):
            user_input_data.output_directory = os.path.expanduser(
                user_input_data.output_directory
            )

        # Prepare output path
        os.makedirs(user_input_data.output_directory, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename: str = ""
        if user_input_data.filename:
            filename = f"{user_input_data.filename}_{timestamp}.csv"
        else:
            function_type_formatted = function_type.value.replace(
                " ", "_"
            )  # Interfacial Tension -> Interfacial_Tension
            if (
                user_input_data.drop_id_method == RegionSelect.USER_SELECTED
                or user_input_data.needle_region_method == RegionSelect.USER_SELECTED
            ):
                filename = f"Manual_{function_type_formatted}_{timestamp}.csv"
            else:
                filename = f"Automated_{function_type_formatted}_{timestamp}.csv"

        output_file = os.path.join(user_input_data.output_directory, filename)
        if function_type == FunctionType.INTERFACIAL_TENSION:
            self.ift_processor.save_result(user_input_data, output_file)
        else:
            self.ca_processor.save_result(user_input_data, output_file)

        self._show_info(
            "Save Successful",
            f"Results have been saved to:\n{output_file}",
        )
        self.on_closing()

    def update_stage(self, direction: int):
        self.current_stage = self.stages[
            (self.stages.index(self.current_stage) + direction) % len(self.stages)
        ]
        if direction == Move.Next.value:
            self.next_stage()
        elif direction == Move.Back.value:
            self.prev_stage()

    def check_import(self, user_input_data: ExperimentalSetup) -> bool:
        if user_input_data.number_of_frames is None or user_input_data.number_of_frames <= 0:
            return False
        if user_input_data.import_files is None or len(user_input_data.import_files) == 0:
            return False

        if user_input_data.is_video:
            return len(user_input_data.import_files) == 1
        else:
            return len(user_input_data.import_files) == user_input_data.number_of_frames

    def register_after(self, delay_ms, callback: Callable):
        after_id = self.after(delay_ms, callback)
        self.after_ids.append(after_id)
        return after_id

    def _window_exists(self) -> bool:
        try:
            return bool(self.winfo_exists())
        except Exception:
            return False

    def _safe_deiconify(self):
        if not self._window_exists() or getattr(self, "_is_closing", False):
            return
        try:
            self.deiconify()
        except Exception as e:
            print("[Warning] deiconify() failed:", e)

    def _show_info(self, title: str, message: str):
        kwargs = {"parent": self} if self._window_exists() else {}
        try:
            messagebox.showinfo(title, message, **kwargs)
        except Exception as e:
            print("[Warning] showinfo() failed:", e)

    def _show_error(self, title: str, message: str):
        kwargs = {"parent": self} if self._window_exists() else {}
        try:
            messagebox.showerror(title, message, **kwargs)
        except Exception as e:
            print("[Warning] showerror() failed:", e)

    def _log_analysis_exception(
        self,
        analysis_type: str,
        error: Exception,
        user_input_data: ExperimentalSetup,
        extra_context: dict = None,
    ) -> str:
        logs_dir = os.path.join(os.path.expanduser("~"), "OpenDrop", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(
            logs_dir, f"analysis_debug_{datetime.now().strftime('%Y%m%d')}.log"
        )
        context = {
            "analysis_type": analysis_type,
            "frame_count": getattr(user_input_data, "number_of_frames", None),
            "input_files": getattr(user_input_data, "import_files", None),
            "baseline_method": str(getattr(user_input_data, "baseline_method", None)),
            "analysis_methods_ca": str(
                getattr(user_input_data, "analysis_methods_ca", None)
            ),
            "timestamp": datetime.now().isoformat(),
        }
        if extra_context:
            context.update(extra_context)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 90 + "\n")
            f.write(f"Error: {type(error).__name__}: {error}\n")
            for key, value in context.items():
                f.write(f"{key}: {value}\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
            f.write("\n")

        return log_path

    def on_closing(self):
        if getattr(self, "_is_closing", False):
            return
        self._is_closing = True

        try:
            if not self._window_exists():
                return

            # Cancel video analysis thread if running
            if getattr(self, "_video_cancel", None) is not None:
                self._video_cancel.set()

            # Step 1: cancel all  after()
            if hasattr(self, 'after_ids'):
                for after_id in self.after_ids:
                    try:
                        self.after_cancel(after_id)
                    except Exception as e:
                        print("[Warning] after_cancel failed:", e)

            # Step 2: safely destory all widget
            for widget in self.winfo_children():
                try:
                    widget.destroy()
                except Exception as e:
                    print("[Warning] widget destroy failed:", e)

            # Step 3: hide current window
            if self._window_exists():
                self.withdraw()

            # Step 4: exit the loop and destory the window
            try:
                self.quit()     # Stop mainloop if running
            except Exception as e:
                print("[Warning] quit() failed:", e)

            try:
                self.destroy()  # Fully destroy this window
            except Exception as e:
                print("[Warning] destroy() failed:", e)

            # Step 5: main window if existed
            if hasattr(self, 'main_window') and self.main_window.winfo_exists():
                self.main_window.deiconify()

        except Exception as e:
            print("[Fatal] on_closing encountered an error:", e)
            import sys
            sys.exit(1)
