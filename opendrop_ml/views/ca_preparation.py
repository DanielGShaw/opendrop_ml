from opendrop_ml.views.component.preparation import (
    create_user_inputs_cm,
    create_plotting_checklist,
    create_analysis_checklist_cm,
)
from opendrop_ml.views.component.imageProcessing import ImageApp
from opendrop_ml.views.helper.style import set_light_only_color
from opendrop_ml.modules.image.frame_loader import get_middle_frame
from opendrop_ml.modules.image.needle_masking import (
    extract_edges_two_contours,
    detect_vertical_lines,
)
from opendrop_ml.modules.image.tilt_correction import correct_tilt
from opendrop_ml.utils.image_handler import ImageHandler

from customtkinter import CTkFrame, CTkLabel, CTkImage
from PIL import Image
import cv2
import numpy as np

PREVIEW_WIDTH = 600
PREVIEW_HEIGHT = 400
NEEDLE_LINE_COLOR_BGR = (0, 0, 255)
NEEDLE_LINE_ALPHA = 0.5
NEEDLE_LINE_THICKNESS = 2
CAPTION_SUCCESS = "Red: auto-detected needle lines"
CAPTION_FAILURE = "\u26a0 Needle not auto-detected. Run may use wrong contact points."
CAPTION_TILT_SUCCESS = "Auto-detected tilt: {angle:.1f}\u00b0"
CAPTION_TILT_FAILURE = (
    "\u26a0 Tilt not auto-detected. Run may produce misaligned angles."
)
CAPTION_COMBINED_BLOCK = (
    "\u26a0 Combined needle-in-drop + tilted-surface not yet tested. "
    "Disable one toggle to proceed."
)
WARNING_COLOR = "#c00"


class CaPreparation(CTkFrame):
    def __init__(self, parent, user_input_data, experimental_drop, **kwargs):
        super().__init__(parent, **kwargs)
        self.application = "CA"
        self.user_input_data = user_input_data
        self.experimental_drop = experimental_drop
        self._image_handler = ImageHandler()
        self._preview_container = None
        # Configure the grid to allow expansion for both columns
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)  # Left column for input fields
        self.grid_columnconfigure(1, weight=1)  # Right column for ImageApp

        # Create the frame for organizing input fields on the left
        self.input_fields_frame = CTkFrame(self)
        set_light_only_color(self.input_fields_frame, "outerframe")
        self.input_fields_frame.grid(
            row=0, column=0, sticky="nsew", padx=15, pady=(10, 0)
        )  # Left side for input fields

        # Ensure that the parent frame (input_fields_frame) resizes properly
        self.input_fields_frame.grid_rowconfigure(0, weight=1)  # Ensure row 0 expands
        self.input_fields_frame.grid_columnconfigure(
            0, weight=1
        )  # Ensure column 0 expands

        # Create a frame for the right side image processing
        self.image_app_frame = CTkFrame(self)
        set_light_only_color(self.image_app_frame, "background")
        self.image_app_frame.grid(
            row=0, column=1, sticky="nsew", padx=15, pady=(10, 0)
        )  # Right side for ImageApp

        if user_input_data.is_video:
            self._render_video_preview()
            self._refresh_next_button_state()
        else:
            # Instantiate the ImageApp on the right
            self.image_app = ImageApp(
                self.image_app_frame,
                self.user_input_data,
                self.experimental_drop,
                self.application,
            )
            # Pack the image app to fill the frame
            self.image_app.pack(fill="both", expand=True)

        # Create user input fields and analysis fields on the left
        self.create_user_input_fields(self.input_fields_frame)
        self.create_analysis_method_fields(self.input_fields_frame)
        self.create_fitting_view_fields(self.input_fields_frame)

    def create_user_input_fields(self, parent_frame):
        """Create and pack user input fields into the specified parent frame."""
        is_video = self.user_input_data.is_video
        needle_cb = self._on_needle_toggled if is_video else None
        tilt_cb = self._on_tilt_toggled if is_video else None
        user_input_frame = create_user_inputs_cm(
            self,
            parent_frame,
            self.user_input_data,
            on_needle_toggle=needle_cb,
            on_surface_tilted=tilt_cb,
        )
        user_input_frame.grid(row=0, column=0, sticky="nsew", pady=(10, 0))  # Use row 0

    def create_analysis_method_fields(self, parent_frame):
        """Create and pack analysis method fields into the specified parent frame."""
        analysis_frame = create_analysis_checklist_cm(
            self, parent_frame, self.user_input_data
        )
        analysis_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

    def create_fitting_view_fields(self, parent_frame):
        """Create and pack Statisitcal Output fields into the specified parent frame."""
        fitting_view_frame = create_plotting_checklist(
            self, parent_frame, self.user_input_data
        )
        fitting_view_frame.grid(row=2, column=0, sticky="nsew", pady=10)  # Use row 1

    def _on_needle_toggled(self, needle_on: bool):
        self._render_video_preview()
        self._refresh_next_button_state()

    def _on_tilt_toggled(self, tilt_on: bool):
        self._render_video_preview()
        self._refresh_next_button_state()

    def _render_video_preview(self):
        if self._preview_container is not None:
            self._preview_container.destroy()
            self._preview_container = None

        container = CTkFrame(self.image_app_frame)
        set_light_only_color(container, "background")
        container.pack(fill="both", expand=True)
        self._preview_container = container

        source_path = None
        if self.user_input_data.import_files:
            source_path = self.user_input_data.import_files[0]

        if not source_path:
            CTkLabel(
                container,
                text="No video source selected.",
                font=("Roboto", 13),
                justify="center",
            ).pack(expand=True)
            return

        try:
            frame_bgr, _frame_number, _metadata = get_middle_frame(source_path)
        except (IOError, ValueError, FileNotFoundError) as exc:
            CTkLabel(
                container,
                text="Preview unavailable: {}".format(exc),
                font=("Roboto", 13),
                justify="center",
                text_color=WARNING_COLOR,
            ).pack(expand=True)
            return

        needle_on = bool(self.user_input_data.needle_in_drop)
        tilt_on = bool(self.user_input_data.surface_tilted)

        caption_text = ""
        caption_color = None
        display_bgr = frame_bgr

        if needle_on and tilt_on:
            caption_text = CAPTION_COMBINED_BLOCK
            caption_color = WARNING_COLOR
        elif needle_on:
            display_bgr, caption_text, caption_color = self._build_needle_overlay(
                frame_bgr
            )
        elif tilt_on:
            display_bgr, caption_text, caption_color = self._build_tilt_preview(
                frame_bgr
            )

        self._pack_preview_image(container, display_bgr)

        if caption_text:
            CTkLabel(
                container,
                text=caption_text,
                font=("Roboto", 12),
                justify="center",
                text_color=caption_color if caption_color else None,
                wraplength=PREVIEW_WIDTH,
            ).pack(pady=(0, 10))

    def _build_tilt_preview(self, frame_bgr):
        """Return (display_bgr, caption, color). Failures drop to caption."""
        try:
            corrected, tilt_angle = correct_tilt(frame_bgr, crop=True)
        except (ValueError, cv2.error):
            return frame_bgr, CAPTION_TILT_FAILURE, WARNING_COLOR

        if tilt_angle == 0.0:
            return frame_bgr, CAPTION_TILT_FAILURE, WARNING_COLOR

        caption = CAPTION_TILT_SUCCESS.format(angle=tilt_angle)
        return corrected, caption, None

    def _refresh_next_button_state(self):
        if not self.user_input_data.is_video:
            return
        next_button = getattr(self.master, "next_button", None)
        if next_button is None:
            return
        combined = bool(self.user_input_data.needle_in_drop) and bool(
            self.user_input_data.surface_tilted
        )
        next_button.configure(state="disabled" if combined else "normal")

    def _build_needle_overlay(self, frame_bgr):
        """Return (display_bgr, caption, color). Failures drop to caption."""
        try:
            left_contour, right_contour, _ = extract_edges_two_contours(
                frame_bgr, blur=False
            )
        except (ValueError, cv2.error):
            return frame_bgr, CAPTION_FAILURE, WARNING_COLOR

        h, w = frame_bgr.shape[:2]
        edge_img = np.zeros((h, w), dtype=np.uint8)
        for contour in (left_contour, right_contour):
            for pt in contour:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    edge_img[y, x] = 255

        left_ext, right_ext, _, _ = detect_vertical_lines(edge_img)
        if left_ext is None or right_ext is None:
            return frame_bgr, CAPTION_FAILURE, WARNING_COLOR

        overlay = np.zeros_like(frame_bgr)
        for line in (left_ext, right_ext):
            p1 = (int(line[0][0]), int(line[0][1]))
            p2 = (int(line[1][0]), int(line[1][1]))
            cv2.line(
                overlay,
                p1,
                p2,
                NEEDLE_LINE_COLOR_BGR,
                thickness=NEEDLE_LINE_THICKNESS,
            )

        mask = (overlay != 0).any(axis=2)
        display = frame_bgr.copy()
        blended = cv2.addWeighted(
            frame_bgr, 1.0 - NEEDLE_LINE_ALPHA, overlay, NEEDLE_LINE_ALPHA, 0
        )
        display[mask] = blended[mask]

        return display, CAPTION_SUCCESS, None

    def _pack_preview_image(self, container, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        width, height = pil_image.size
        new_width, new_height = self._image_handler.get_fitting_dimensions(
            width,
            height,
            max_width=PREVIEW_WIDTH,
            max_height=PREVIEW_HEIGHT,
        )
        tk_image = CTkImage(pil_image, size=(new_width, new_height))
        image_label = CTkLabel(
            container,
            text="",
            fg_color="lightgrey",
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT,
        )
        image_label.pack(padx=10, pady=10)
        image_label.configure(image=tk_image)
        image_label.image = tk_image
