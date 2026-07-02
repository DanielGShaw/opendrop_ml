# views/output_page.py

from opendrop_ml.modules.core.classes import ExperimentalSetup
from opendrop_ml.views.helper.style import set_light_only_color

from tkinter import filedialog
import customtkinter as ctk
import os
import tkinter as tk
from opendrop_ml.utils.tooltip_util import create_tooltip


def _build_inspection_hint(user_input_data) -> str:
    """Return the cp_scatter inspection-hint text for the active pipeline.

    The three branches mirror the router in ``ca_data_processor``: needle-
    in-drop takes precedence, then tilted surface, then the auto default.
    The ``(True, True)`` combination is blocked at the router so first-
    match ordering is safe.
    """
    if getattr(user_input_data, "needle_in_drop", False):
        return (
            "Tip: Inspect the *_cp_scatter.png file to verify\n"
            "needle detection and contact-point placement before\n"
            "trusting the CSV results."
        )
    if getattr(user_input_data, "surface_tilted", False):
        return (
            "Tip: Inspect the *_cp_scatter.png file to verify\n"
            "tilt correction and contact-point placement before\n"
            "trusting the CSV results."
        )
    return (
        "Tip: Inspect the *_cp_scatter.png file to verify\n"
        "preprocessing before trusting the CSV results."
    )


class OutputPage(ctk.CTkFrame):
    def __init__(self, parent, user_input_data: ExperimentalSetup, **kwargs):
        super().__init__(parent, **kwargs)
        self.user_input_data = user_input_data

        # allow the middle column to expand when resized
        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # -----------------------
        # Output Location Section
        # -----------------------
        output_frame = ctk.CTkFrame(self)
        set_light_only_color(output_frame, "outerframe")
        output_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        output_frame.grid_columnconfigure(1, weight=1)

        # Section title
        ctk.CTkLabel(
            output_frame,
            text="Output Location",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=4, padx=10, pady=(10, 5), sticky="w")

        # Location row
        ctk.CTkLabel(output_frame, text="Location:", anchor="w").grid(
            row=1, column=0, sticky="w", padx=10, pady=5
        )

        self.location_entry = ctk.CTkEntry(output_frame, width=300)
        self.location_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.location_entry._entry.insert(0, user_input_data.output_directory)

        ctk.CTkButton(output_frame, text="Browse", command=self.browse_location).grid(
            row=1, column=3, padx=10, pady=5
        )

        # Filename row
        ctk.CTkLabel(output_frame, text="Filename:", anchor="w").grid(
            row=2, column=0, sticky="w", padx=10, pady=(5, 10)
        )
        self.filename_var = ctk.StringVar(value=user_input_data.filename)
        self.filename_var.trace_add("write", self.on_filename_change)
        self.filename_entry = ctk.CTkEntry(
            output_frame, width=300, textvariable=self.filename_var
        )
        self.filename_entry.grid(row=2, column=1, padx=10, pady=(5, 10), sticky="ew")

        # "❓" help icon next to filename
        help_label = ctk.CTkLabel(
            output_frame,
            text="❓",
            font=ctk.CTkFont(size=16, weight="bold"),
            cursor="question_arrow",
        )
        help_label.grid(row=2, column=2, padx=5, pady=(5, 10), sticky="w")

        # Combined tooltip: naming convention + default folder notice
        tooltip_text = (
            "Output files are named according to the following convention:\n"
            "  <filename>_YYYYMMDD_HHMMSS.csv\n\n"
            "Where:\n"
            "  • <filename> is your specified filename, if provided.\n"
            "  • Otherwise, <filename> is constructed as <prefix>_<function_type>.\n"
            "    - <prefix> is 'Manual' if any region selection is user-selected,\n"
            "      or 'Automated' if all regions are automated.\n"
            "    - <function_type> is the analysis type (e.g., Interfacial_Tension),\n"
            "      with spaces replaced by underscores.\n"
            "  • YYYYMMDD represents the extraction date (year, month, day).\n"
            "  • HHMMSS represents the extraction time (hour, minute, second).\n\n"
            "Examples:\n"
            "  MyFile_20250524_192455.csv\n"
            "  Manual_Interfacial_Tension_20250524_192455.csv\n"
            "  Automated_Contact_Angle_20250524_192455.csv\n\n"
            "This naming convention ensures unique filenames for each run.\n"
            f"If no output location is specified, files will be saved to the '{user_input_data.output_directory}' folder by default."
        )
        create_tooltip(help_label, tooltip_text, position="below-left-cursor")

        # -----------------------
        # Video Analysis Info
        # -----------------------
        if user_input_data.is_video:
            self._add_video_info_section()

    def _add_video_info_section(self):
        info_frame = ctk.CTkFrame(self)
        set_light_only_color(info_frame, "outerframe")
        info_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        info_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            info_frame,
            text="Video Analysis Results",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")

        status = getattr(self.user_input_data, "_video_status", "unknown")
        if status == "completed":
            status_text = "Analysis completed successfully."
        elif status == "cancelled":
            status_text = "Analysis was cancelled. Partial CSV may be available."
        else:
            status_text = "Analysis status: {}".format(status)

        ctk.CTkLabel(info_frame, text=status_text, anchor="w").grid(
            row=1, column=0, padx=10, pady=2, sticky="w"
        )

        tmp_dir = getattr(self.user_input_data, "_video_tmp_dir", None)
        if tmp_dir and os.path.isdir(tmp_dir):
            files = sorted(
                f
                for f in os.listdir(tmp_dir)
                if os.path.isfile(os.path.join(tmp_dir, f))
            )
            if files:
                files_text = "Files to save:\n  " + "\n  ".join(files)
            else:
                files_text = "No files generated."
            ctk.CTkLabel(info_frame, text=files_text, anchor="w", justify="left").grid(
                row=2, column=0, padx=10, pady=2, sticky="w"
            )

        hint = _build_inspection_hint(self.user_input_data)
        ctk.CTkLabel(
            info_frame,
            text=hint,
            anchor="w",
            justify="left",
            text_color="gray",
        ).grid(row=3, column=0, padx=10, pady=(5, 10), sticky="w")

    def browse_location(self):
        default_dir = self.user_input_data.output_directory or os.getcwd()
        directory = filedialog.askdirectory(initialdir=default_dir)

        if directory:
            self.location_entry.delete(0, tk.END)
            self.location_entry.insert(0, directory)
            self.user_input_data.output_directory = directory

    def on_filename_change(self, *args):
        """Store the filename in the user_input_data when changed."""
        self.user_input_data.filename = self.filename_var.get()
