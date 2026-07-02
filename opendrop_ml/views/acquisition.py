from opendrop_ml.modules.core.classes import ExperimentalSetup
from opendrop_ml.utils.image_handler import ImageHandler
from opendrop_ml.utils.enums import FunctionType
from opendrop_ml.utils.config import (
    PATH_TO_SCRIPT,
    IMAGE_TYPE,
    VIDEO_TYPE,
    FILE_SOURCE_OPTIONS_CA,
    INPUT_MODE_OPTIONS,
    VIDEO_SOURCE_OPTIONS,
)
from opendrop_ml.utils.tooltip_util import create_tooltip
from opendrop_ml.modules.image.frame_loader import (
    get_middle_frame,
    get_frame_count,
    IMAGE_EXTENSIONS,
)
from opendrop_ml.views.component.option_menu import OptionMenu
from opendrop_ml.views.component.float_entry import FloatEntry
from opendrop_ml.views.helper.style import set_light_only_color

from customtkinter import CTkFrame, CTkLabel, CTkButton, CTkEntry, CTkImage
from tkinter import filedialog, messagebox
from PIL import Image
import cv2
import os


IMAGE_FRAME_WIDTH = 600
IMAGE_FRAME_HEIGHT = 400


class Acquisition(CTkFrame):
    def __init__(
        self,
        parent,
        user_input_data: ExperimentalSetup,
        function_type: FunctionType,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self.user_input_data = user_input_data
        self.image_handler = ImageHandler()
        self.function_type = function_type
        self.input_mode = "Static"
        self.video_source = "Video File"

        # Configure the grid to allow expansion for both columns
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)  # Left column for input fields
        # Right column for image display
        self.grid_columnconfigure(1, weight=1)

        # Create the frame for organizing input fields on the left
        self.input_fields_frame = CTkFrame(self)
        set_light_only_color(self.input_fields_frame, "outerframe")
        self.input_fields_frame.grid(
            row=0, column=0, sticky="nsew", padx=15, pady=(10, 0)
        )

        # Ensure that the input fields frame resizes properly
        self.input_fields_frame.grid_rowconfigure(0, weight=1)
        self.input_fields_frame.grid_columnconfigure(0, weight=1)

        # Create a frame for image acquisition options inside input_fields_frame
        self.image_acquisition_frame = CTkFrame(self.input_fields_frame)
        set_light_only_color(self.image_acquisition_frame, "entry")
        self.image_acquisition_frame.grid(
            row=0, column=0, sticky="new", padx=15, pady=10
        )
        self.image_acquisition_frame.grid_columnconfigure(
            1, weight=1
        )  # Second column (values) can expand

        # Row 0: Image source
        self.image_source = OptionMenu(
            self,
            self.image_acquisition_frame,
            "Image source:",
            ["Local images"],
            self.update_image_source,
            rw=0,
        )
        self.image_source.optionmenu.grid_configure(sticky="ew")
        self.image_acquisition_frame.grid_rowconfigure(0, weight=0)

        # Row 1-2: Input mode + video source (CA only)
        if function_type == FunctionType.CONTACT_ANGLE:
            self.input_mode_dropdown = OptionMenu(
                self,
                self.image_acquisition_frame,
                "Input mode:",
                INPUT_MODE_OPTIONS,
                self.update_input_mode,
                rw=1,
            )
            self.input_mode_dropdown.optionmenu.grid_configure(sticky="ew")
            self.image_acquisition_frame.grid_rowconfigure(1, weight=0)

            create_tooltip(
                self.input_mode_dropdown.label,
                "Static: each image is an independent measurement.\n"
                "Video: a video file or image folder is treated as "
                "one system over time.",
                position="below-align-left",
            )

            self.video_source_dropdown = OptionMenu(
                self,
                self.image_acquisition_frame,
                "Video source:",
                VIDEO_SOURCE_OPTIONS,
                self.update_video_source,
                rw=2,
            )
            self.video_source_dropdown.optionmenu.grid_configure(sticky="ew")
            self.image_acquisition_frame.grid_rowconfigure(2, weight=0)

            # Initially hidden — only visible in video mode
            self.video_source_dropdown.label.grid_remove()
            self.video_source_dropdown.optionmenu.grid_remove()

        # Row 3: Choose files button
        self.setup_choose_files_frame(self.image_acquisition_frame)

        # Row 6: Frame interval
        default_value = getattr(self.user_input_data, "frame_interval", 1)

        self.frame_interval = FloatEntry(
            self,
            self.image_acquisition_frame,
            "Frame interval (s)*:",
            self.update_frame_interval,
            rw=6,
            cl=0,
            default_value=default_value,
        )
        self.image_acquisition_frame.grid_rowconfigure(6, weight=0)

        # Create right side frame for image display with vertical centering
        self.images_frame = CTkFrame(self)
        set_light_only_color(self.images_frame, "outerframe")
        self.images_frame.grid(row=0, column=1, sticky="nsew", padx=15, pady=(10, 0))

        # Configure right frame to center content
        self.images_frame.grid_rowconfigure(0, weight=1)
        self.images_frame.grid_columnconfigure(0, weight=1)

        # Create empty placeholder
        self.create_empty_image_placeholder()

    def create_empty_image_placeholder(self):
        # Create a container for vertical centering
        container_frame = CTkFrame(self.images_frame)
        set_light_only_color(container_frame, "outerframe")
        container_frame.grid(row=0, column=0, sticky="nsew")

        # Configure container for vertical centering
        container_frame.grid_rowconfigure(0, weight=1)  # Top space
        container_frame.grid_rowconfigure(1, weight=0)  # Content area
        container_frame.grid_rowconfigure(2, weight=1)  # Bottom space
        container_frame.grid_columnconfigure(0, weight=1)

        # Create internal frame to display placeholder
        placeholder_frame = CTkFrame(container_frame)
        set_light_only_color(placeholder_frame, "innerframe")
        placeholder_frame.grid(row=1, column=0, padx=15, pady=15)

        # Empty image label as placeholder
        placeholder_label = CTkLabel(
            placeholder_frame,
            text="No image selected",
            fg_color="lightgrey",
            text_color="black",
            width=IMAGE_FRAME_WIDTH,
            height=IMAGE_FRAME_HEIGHT,
        )
        placeholder_label.pack(padx=10, pady=10)

    def update_image_source(self, selection: str):
        print(selection)
        if selection == FILE_SOURCE_OPTIONS_CA[0]:
            # local images
            self.choose_files_button.configure(state="normal")
        else:
            self.choose_files_button.configure(state="disabled")

        self.user_input_data.image_source = selection

    def update_frame_interval(self, *args):
        self.user_input_data.frame_interval = self.frame_interval.get_value()

    def update_input_mode(self, selection: str):
        self.input_mode = selection

        if selection == "Video":
            self.user_input_data.is_video = True

            self.video_source_dropdown.label.grid()
            self.video_source_dropdown.optionmenu.grid()

            self.choose_files_label.configure(text="Video input *:")
            self.choose_files_button.configure(text="Choose")

            self.frame_interval.label.grid_remove()
            self.frame_interval.entry.grid_remove()

        else:  # "Static"
            self.user_input_data.is_video = False

            self.video_source_dropdown.label.grid_remove()
            self.video_source_dropdown.optionmenu.grid_remove()

            self.choose_files_label.configure(text="Image files *:")
            self.choose_files_button.configure(text="Choose File(s)")

            self.frame_interval.label.grid()
            self.frame_interval.entry.grid()

        self._reset_selection()

    def update_video_source(self, selection: str):
        self.video_source = selection
        self._reset_selection()

    def _reset_selection(self):
        self.user_input_data.import_files = None
        self.user_input_data.number_of_frames = 0

        if self.input_mode == "Static":
            self.choose_files_button.configure(text="Choose File(s)")
        else:
            self.choose_files_button.configure(text="Choose")

        for widget in self.images_frame.winfo_children():
            widget.destroy()
        self.create_empty_image_placeholder()

    def setup_choose_files_frame(self, frame):
        self.choose_files_label = CTkLabel(
            frame, text="Image files *:", width=150, anchor="w"
        )
        self.choose_files_label.grid(
            row=3, column=0, sticky="w", padx=(5, 5), pady=(5, 5)
        )

        self.choose_files_button = CTkButton(
            frame,
            text="Choose File(s)",
            command=self.select_files,
            width=150,
        )
        self.choose_files_button.grid(
            row=3, column=1, sticky="ew", padx=(5, 5), pady=(5, 5)
        )

    def select_files(self):
        for widget in self.images_frame.winfo_children():
            widget.destroy()

        if self.input_mode == "Static":
            self._select_static_files()
        else:
            if self.video_source == "Video File":
                self._select_video_file()
            else:
                self._select_image_folder()

    def _select_static_files(self):
        self.user_input_data.import_files = filedialog.askopenfilenames(
            title="Select Files", filetypes=IMAGE_TYPE, initialdir=PATH_TO_SCRIPT
        )

        num_files = len(self.user_input_data.import_files)
        self.user_input_data.number_of_frames = num_files

        if num_files > 0:
            self.choose_files_button.configure(text=f"{num_files} File(s) Selected")
            self.current_index = 0
            self.initialize_image_display()
        else:
            self.choose_files_button.configure(text="Choose File(s)")
            messagebox.showinfo("No Selection", "No files were selected.")
            self.create_empty_image_placeholder()

    def _select_video_file(self):
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=VIDEO_TYPE,
            initialdir=PATH_TO_SCRIPT,
        )

        if not video_path:
            messagebox.showinfo("No Selection", "No video file was selected.")
            self.create_empty_image_placeholder()
            return

        ext = os.path.splitext(video_path)[1].lower()
        if ext not in (".mp4", ".avi", ".mov"):
            messagebox.showerror(
                "Invalid File",
                "Unsupported video format '{}'.\n"
                "Supported formats: .mp4, .avi, .mov".format(ext),
            )
            self.create_empty_image_placeholder()
            return

        self._set_video_input(video_path)

    def _select_image_folder(self):
        folder_path = filedialog.askdirectory(
            title="Select Image Folder",
            initialdir=PATH_TO_SCRIPT,
        )

        if not folder_path:
            messagebox.showinfo("No Selection", "No folder was selected.")
            self.create_empty_image_placeholder()
            return

        image_files = [
            f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

        if not image_files:
            messagebox.showerror(
                "No Images Found",
                "The selected folder contains no image files.\n"
                "Supported formats: {}\n\n"
                "Selected: {}".format(", ".join(IMAGE_EXTENSIONS), folder_path),
            )
            self.create_empty_image_placeholder()
            return

        self._set_video_input(folder_path)

    def _set_video_input(self, source_path):
        self.user_input_data.import_files = [source_path]
        self.user_input_data.is_video = True

        try:
            frame_count = get_frame_count(source_path)
        except (IOError, ValueError) as e:
            messagebox.showerror("Error", "Could not read source:\n{}".format(e))
            self.user_input_data.import_files = None
            self.user_input_data.number_of_frames = 0
            self.create_empty_image_placeholder()
            return

        self.user_input_data.number_of_frames = frame_count

        basename = os.path.basename(source_path)
        self.choose_files_button.configure(
            text="{} ({} frames)".format(basename, frame_count)
        )

        self._show_video_preview(source_path)

    def _show_video_preview(self, source_path):
        try:
            frame_bgr, frame_number, metadata = get_middle_frame(source_path)
        except (IOError, ValueError) as e:
            messagebox.showerror(
                "Preview Error", "Could not load preview:\n{}".format(e)
            )
            self.create_empty_image_placeholder()
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        container_frame = CTkFrame(self.images_frame)
        set_light_only_color(container_frame, "outerframe")
        container_frame.grid(row=0, column=0, sticky="nsew")

        container_frame.grid_rowconfigure(0, weight=1)
        container_frame.grid_rowconfigure(1, weight=0)
        container_frame.grid_rowconfigure(2, weight=1)
        container_frame.grid_columnconfigure(0, weight=1)

        display_frame = CTkFrame(container_frame)
        set_light_only_color(display_frame, "innerframe")
        display_frame.grid(row=1, column=0, padx=15, pady=15)

        image_label = CTkLabel(
            display_frame,
            text="",
            fg_color="lightgrey",
            width=IMAGE_FRAME_WIDTH,
            height=IMAGE_FRAME_HEIGHT,
        )
        image_label.pack(padx=10, pady=10)

        width, height = pil_image.size
        new_width, new_height = self.image_handler.get_fitting_dimensions(
            width,
            height,
            max_width=IMAGE_FRAME_WIDTH,
            max_height=IMAGE_FRAME_HEIGHT,
        )
        tk_image = CTkImage(pil_image, size=(new_width, new_height))
        image_label.configure(image=tk_image)
        image_label.image = tk_image

        if "filename" in metadata:
            info_text = "Preview: {} (frame {})".format(
                metadata["filename"], frame_number + 1
            )
        else:
            info_text = "Preview: frame {}".format(frame_number + 1)

        source_basename = os.path.basename(source_path)
        info_label = CTkLabel(
            display_frame,
            text="{}\n{}".format(source_basename, info_text),
            font=("Arial", 11),
        )
        info_label.pack(pady=5)

    def initialize_image_display(self):
        # Create a container for vertical centering
        container_frame = CTkFrame(self.images_frame)
        set_light_only_color(container_frame, "outerframe")
        container_frame.grid(row=0, column=0, sticky="nsew")

        # Configure container for vertical centering
        container_frame.grid_rowconfigure(0, weight=1)  # Top space
        container_frame.grid_rowconfigure(1, weight=0)  # Content area
        container_frame.grid_rowconfigure(2, weight=1)  # Bottom space
        container_frame.grid_columnconfigure(0, weight=1)

        # Create internal frame to display image and controls
        display_frame = CTkFrame(container_frame)
        set_light_only_color(display_frame, "innerframe")
        display_frame.grid(row=1, column=0, padx=15, pady=15)

        # Image label

        self.image_label = CTkLabel(
            display_frame,
            text="",
            fg_color="lightgrey",
            width=IMAGE_FRAME_WIDTH,
            height=IMAGE_FRAME_HEIGHT,
        )
        self.image_label.pack(padx=10, pady=10)

        # File name label
        file_name = os.path.basename(
            self.user_input_data.import_files[self.current_index]
        )
        self.name_label = CTkLabel(display_frame, text=file_name)
        self.name_label.pack(pady=5)

        # Navigation controls frame
        self.image_navigation_frame = CTkFrame(display_frame)
        set_light_only_color(self.image_navigation_frame, "entry")
        self.image_navigation_frame.pack(pady=10)

        # Navigation controls
        self.prev_button = CTkButton(
            self.image_navigation_frame,
            text="<",
            command=lambda: self.change_image(-1),
            width=30,
        )
        self.prev_button.grid(row=0, column=0, padx=5, pady=5)

        self.index_entry = CTkEntry(self.image_navigation_frame, width=40)
        self.index_entry.grid(row=0, column=1, padx=5, pady=5)
        self.index_entry.bind("<Return>", lambda event: self.update_index_from_entry())
        self.index_entry.insert(0, str(self.current_index + 1))

        self.navigation_label = CTkLabel(
            self.image_navigation_frame,
            text=f" of {self.user_input_data.number_of_frames}",
            font=("Arial", 12),
        )
        self.navigation_label.grid(row=0, column=2, padx=5, pady=5)

        self.next_button = CTkButton(
            self.image_navigation_frame,
            text=">",
            command=lambda: self.change_image(1),
            width=30,
        )
        self.next_button.grid(row=0, column=3, padx=5, pady=5)

        # Load image
        self.load_image(self.user_input_data.import_files[self.current_index])

    def load_image(self, selected_image):
        """Load and display the selected image."""
        try:
            self.current_image = Image.open(selected_image)
            self.display_image()

        except FileNotFoundError:
            print(f"Error: The image file {selected_image} was not found.")
            self.current_image = None

    def display_image(self):
        """Display the currently loaded image."""
        width, height = self.current_image.size
        new_width, new_height = self.image_handler.get_fitting_dimensions(
            width, height, max_width=IMAGE_FRAME_WIDTH, max_height=IMAGE_FRAME_HEIGHT
        )
        self.tk_image = CTkImage(self.current_image, size=(new_width, new_height))

        self.image_label.image = self.tk_image
        self.update_idletasks()
        self.image_label.configure(image=self.tk_image)

    def change_image(self, direction):
        """Change the currently displayed image based on the direction."""
        if self.user_input_data.import_files:
            self.current_index = (
                self.current_index + direction
            ) % self.user_input_data.number_of_frames
            # Load the new image
            self.load_image(self.user_input_data.import_files[self.current_index])
            self.update_index_entry()  # Update the entry with the current index
            file_name = os.path.basename(
                self.user_input_data.import_files[self.current_index]
            )
            self.name_label.configure(text=file_name)

    def update_index_from_entry(self):
        """Update current index based on user input in the entry."""
        try:
            new_index = int(self.index_entry.get()) - 1  # Convert to zero-based index
            if 0 <= new_index < self.user_input_data.number_of_frames:
                self.current_index = new_index
                # Load the new image
                self.load_image(self.user_input_data.import_files[self.current_index])
            else:
                print("Index out of range.")
        except ValueError:
            print("Invalid input. Please enter a number.")

        self.update_index_entry()  # Update the entry display

    def update_index_entry(self):
        """Update the index entry to reflect the current index."""
        self.index_entry.delete(0, "end")  # Clear the current entry
        # Insert the new index (1-based)
        self.index_entry.insert(0, str(self.current_index + 1))
