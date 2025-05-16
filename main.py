import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from tkinter import filedialog, Canvas
from PIL import Image, ImageTk
import os
from image_io import *
from operations_basic import *
from operations_color import *


class ImageProcessingApp(ttk.Window):
    def __init__(self, themename="flatly"):
        super().__init__(themename=themename)
        self.title("Image Processing Tool")
        self.geometry("1000x1200")
        self.resizable(False, False)

        # Variables
        self.image_type = ttk.StringVar(value="Grayscale")
        self.image_path = ttk.StringVar()
        self.selected_basic_op = ttk.StringVar()
        self.selected_color_op = ttk.StringVar()
        self.selected_color = ttk.StringVar(value="Red")
        self.swap_from_color = ttk.StringVar(value="Red")
        self.swap_to_color = ttk.StringVar(value="Blue")
        self.histogram_visible = ttk.IntVar(value=1)
        self.eliminate_red = ttk.IntVar()
        self.eliminate_green = ttk.IntVar()
        self.eliminate_blue = ttk.IntVar()
        self.blur_method = ttk.StringVar(value="Average")

        self.original_image = None
        self.filtered_image = None
        self.img_pil = None
        self.img_original_pil = None
        self.current_theme = themename
        self.second_image_path = None
        self.history_stack = []
        self.average_imgs = []

        self.create_widgets()

    def create_widgets(self):
        self.create_theme_toggle()
        self.create_upload_section()
        self.create_display_section()
        self.create_basic_operations_section()
        self.create_advanced_options_section()
        self.create_color_operations_section()

    def create_theme_toggle(self):
        ttk.Button(
            self,
            text="Toggle Dark Mode",
            command=self.toggle_theme,
            bootstyle="warning-outline",
        ).place(x=830, y=10)

    def toggle_theme(self):
        if self.current_theme in ["flatly", "minty", "vapor"]:
            self.style.theme_use("superhero")
            self.current_theme = "superhero"
        else:
            self.style.theme_use("flatly")
            self.current_theme = "flatly"

        for widget in self.winfo_children():
            try:
                widget.configure(bootstyle="")
            except Exception:
                pass

        for frame in self.winfo_children():
            for child in frame.winfo_children():
                try:
                    child.configure(bootstyle="")
                except Exception:
                    pass

    def create_upload_section(self):
        frame = ttk.LabelFrame(self, text="Upload Image", padding=10)
        frame.place(x=20, y=50, width=960, height=100)

        ttk.Radiobutton(frame, text="RGB", variable=self.image_type, value="RGB").grid(
            row=0, column=0, padx=5
        )
        ttk.Radiobutton(
            frame, text="Grayscale", variable=self.image_type, value="Grayscale"
        ).grid(row=0, column=1, padx=5)

        ttk.Button(
            frame,
            text="Upload Image",
            command=lambda: upload_image(self),
            bootstyle=PRIMARY,
        ).grid(row=0, column=2, padx=15)

        ttk.Entry(frame, textvariable=self.image_path, width=70, state="readonly").grid(
            row=0, column=3, padx=10
        )

    def create_display_section(self):
        frame = ttk.LabelFrame(self, text="Image Display", padding=10)
        frame.place(x=20, y=160, width=960, height=300)

        self.canvas_original = Canvas(frame, bg="white", width=220, height=220)
        self.canvas_hist_original = Canvas(frame, bg="white", width=220, height=220)
        self.canvas_filtered = Canvas(frame, bg="white", width=220, height=220)
        self.canvas_hist_filtered = Canvas(frame, bg="white", width=220, height=220)

        self.canvas_original.grid(row=0, column=0, padx=10, pady=5)
        self.canvas_hist_original.grid(row=0, column=1, padx=10, pady=5)
        self.canvas_filtered.grid(row=0, column=2, padx=10, pady=5)
        self.canvas_hist_filtered.grid(row=0, column=3, padx=10, pady=5)

        ttk.Label(frame, text="Original Image").grid(row=1, column=0)
        ttk.Label(frame, text="Histogram of Original").grid(row=1, column=1)
        ttk.Label(frame, text="Filtered Image").grid(row=1, column=2)
        ttk.Label(frame, text="Histogram of Filtered").grid(row=1, column=3)

    def create_advanced_options_section(self):
        frame = ttk.LabelFrame(self, text="Advanced Options", padding=10)
        frame.place(x=510, y=470, width=470, height=180)

        # Allow columns to expand equally
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            frame,
            text="Show Histogram",
            variable=self.histogram_visible,
            bootstyle="success-round-toggle",
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        ttk.Button(
            frame,
            text="Save Filtered Image",
            command=lambda: save_filtered_image(self),
            bootstyle=SUCCESS,
        ).grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        ttk.Button(
            frame,
            text="Save Histogram",
            command=lambda: save_histogram_image(self),
            bootstyle=INFO,
        ).grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        ttk.Button(
            frame,
            text="Reset to Original",
            command=lambda: reset_image(self),
            bootstyle=DANGER,
        ).grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.undo_btn = ttk.Button(
            frame,
            text="Undo Last Operation",
            command=self.undo_last_operation,
            bootstyle="warning-outline",
        )
        self.undo_btn.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

    def create_basic_operations_section(self):
        frame = ttk.LabelFrame(self, text="Basic Operations", padding=10)
        frame.place(x=20, y=470, width=470, height=600)

        ttk.Label(frame, text="Choose Operation:").grid(row=0, column=0, padx=5)
        operations = [
            "Point Operation",
            "Brightness",
            "Contrast",
            "Inverse",
            "Blur",
            "Edge Detection",
            "Noise",
            "Segmentation",
        ]
        op_menu = ttk.Combobox(
            frame,
            values=operations,
            textvariable=self.selected_basic_op,
            state="readonly",
        )
        op_menu.grid(row=0, column=1, padx=10)
        op_menu.bind("<<ComboboxSelected>>", basic_operation_selected)

        # Point Operation
        self.point_op_label = ttk.Label(frame, text="Operation Type:")
        self.point_op_label.grid(row=27, column=0, padx=5, pady=5, sticky="e")
        self.point_op_label.grid_remove()

        self.point_op_type = ttk.StringVar(value="Addition")
        self.point_op_menu = ttk.Combobox(
            frame,
            values=["Addition", "Subtraction", "Division", "Complement"],
            textvariable=self.point_op_type,
            state="readonly",
        )
        self.point_op_menu.grid(row=27, column=1, padx=10, pady=5, sticky="w")
        self.point_op_menu.grid_remove()

        self.point_op_factor_label = ttk.Label(frame, text="Factor:")
        self.point_op_factor_label.grid(row=28, column=0, padx=5, pady=5, sticky="e")
        self.point_op_factor_label.grid_remove()

        self.point_op_factor_entry = ttk.Entry(frame)
        self.point_op_factor_entry.insert(0, "50")
        self.point_op_factor_entry.grid(row=28, column=1, padx=10, pady=5, sticky="w")
        self.point_op_factor_entry.grid_remove()

        self.point_op_apply_btn = ttk.Button(
            frame,
            text="Apply Operation",
            command=lambda: apply_point_operation(
                app=self, operation=self.point_op_type.get()
            ),
            bootstyle=SUCCESS,
        )
        self.point_op_apply_btn.grid(row=29, column=0, columnspan=2, pady=5)
        self.point_op_apply_btn.grid_remove()

        # Brightness
        self.brightness_slider = tk.Scale(
            frame, from_=-10, to=10, resolution=1, orient="horizontal", length=300
        )
        self.brightness_slider.grid(row=1, column=0, columnspan=2, pady=5)
        self.brightness_slider.grid_remove()

        self.apply_brightness_btn = ttk.Button(
            frame,
            text="Apply Brightness",
            command=lambda: apply_brightness(self),
            bootstyle=SUCCESS,
        )
        self.apply_brightness_btn.grid(row=2, column=0, columnspan=2, pady=5)
        self.apply_brightness_btn.grid_remove()

        # Contrast
        self.contrast_method_label = ttk.Label(frame, text="Contrast Method:")
        self.contrast_method_menu = ttk.Combobox(
            frame,
            values=["Gamma Correction", "Equalization", "Stretching", "Matching"],
            state="readonly",
        )
        self.contrast_method_menu.grid(row=4, column=1, padx=10)
        self.contrast_method_menu.grid_remove()
        self.contrast_method_menu.bind(
            "<<ComboboxSelected>>", self.on_contrast_method_selected
        )

        self.contrast_factor_label = ttk.Label(frame, text="Factor:")
        self.contrast_factor_label.grid(row=5, column=0, padx=5)
        self.contrast_factor_label.grid_remove()

        self.contrast_factor_entry = ttk.Entry(frame)
        self.contrast_factor_entry.grid(row=5, column=1, padx=10)
        self.contrast_factor_entry.grid_remove()

        self.upload_second_img_btn = ttk.Button(
            frame,
            text="Upload 2nd Image",
            command=lambda: upload_second_image(self),
            bootstyle=INFO,
        )
        self.upload_second_img_btn.grid(row=6, column=0, columnspan=2, pady=5)
        self.upload_second_img_btn.grid_remove()

        self.apply_contrast_btn = ttk.Button(
            frame,
            text="Apply Contrast",
            command=lambda: apply_contrast(self),
            bootstyle=SUCCESS,
        )
        self.apply_contrast_btn.grid(row=7, column=0, columnspan=2, pady=5)
        self.apply_contrast_btn.grid_remove()

        # Inverse
        self.partial_inverse_check = ttk.Checkbutton(
            frame, text="Partial Inverse", bootstyle="info-round-toggle"
        )
        self.partial_inverse_check.grid(row=8, column=0, columnspan=2, pady=5)
        self.partial_inverse_check.grid_remove()

        self.inverse_threshold_label = ttk.Label(frame, text="Threshold:")
        self.inverse_threshold_label.grid(row=9, column=0, padx=5)
        self.inverse_threshold_label.grid_remove()

        self.inverse_threshold_entry = ttk.Entry(frame)
        self.inverse_threshold_entry.grid(row=9, column=1, padx=10)
        self.inverse_threshold_entry.grid_remove()

        self.apply_inverse_btn = ttk.Button(
            frame,
            text="Apply Inverse",
            command=lambda: apply_inverse(self),
            bootstyle=SUCCESS,
        )
        self.apply_inverse_btn.grid(row=10, column=0, columnspan=2, pady=5)
        self.apply_inverse_btn.grid_remove()

        # Blur Dropdown (with Average, Median, Gaussian)
        self.blur_type_menu = ttk.Combobox(
            frame,
            values=["Average", "Median", "Gaussian"],
            textvariable=self.blur_method,
            state="readonly",
        )

        self.blur_type_menu.grid(row=11, column=0, columnspan=2, pady=5)
        self.blur_type_menu.grid_remove()

        self.apply_blur_btn = ttk.Button(
            frame,
            text="Apply Blur",
            command=lambda: apply_blur(self),
            bootstyle=SUCCESS,
        )
        self.apply_blur_btn.grid(row=12, column=0, columnspan=2, pady=5)
        self.apply_blur_btn.grid_remove()

        # Kernel Size Entry for Blur/Noise (Symmetric layout)
        self.kernel_size_label = ttk.Label(frame, text="Kernel Size (W x H):")
        self.kernel_size_label.grid(row=13, column=0, padx=5, pady=5, sticky="e")
        self.kernel_size_label.grid_remove()

        self.kernel_size_entry_w = ttk.Entry(frame, width=5)
        self.kernel_size_entry_h = ttk.Entry(frame, width=5)
        self.kernel_size_entry_w.insert(0, "3")
        self.kernel_size_entry_h.insert(0, "3")
        self.kernel_size_entry_w.grid(row=13, column=1, padx=(0, 2), pady=5, sticky="w")
        self.kernel_size_entry_h.grid(
            row=13, column=2, padx=(2, 10), pady=5, sticky="w"
        )
        self.kernel_size_entry_w.grid_remove()
        self.kernel_size_entry_h.grid_remove()

        # Edge Detection

        self.sobel_mag_btn = ttk.Button(
            frame,
            text="Apply Sobel Magnitude",
            command=lambda: apply_sobel_magnitude(self),
            bootstyle=SUCCESS,
        )
        self.sobel_mag_btn.grid(row=25, column=0, columnspan=2, pady=5)
        self.sobel_mag_btn.grid_remove()
        self.laplacian_btn = ttk.Button(
            frame,
            text="Apply Laplacian Filter",
            command=lambda: apply_laplacian(self),
            bootstyle=SUCCESS,
        )
        self.laplacian_btn.grid(row=26, column=0, columnspan=2, pady=5)
        self.laplacian_btn.grid_remove()

        # Noise
        self.noise_method_label = ttk.Label(frame, text="Noise Removal Method:")
        self.noise_method_label.grid(row=14, column=0, padx=5, pady=5, sticky="e")
        self.noise_method_label.grid_remove()

        self.noise_method_menu = ttk.Combobox(
            frame,
            values=[
                "Min",
                "Max",
                "Most Frequent",
                "Median",
                "Outlier",
                "Average",
                "Image averaging",
            ],
            state="readonly",
        )
        self.noise_method_menu.grid(row=14, column=1, padx=5, pady=5, sticky="w")
        self.noise_method_menu.grid_remove()
        # Add: Outlier threshold entry (hidden by default)
        self.outlier_threshold_label = ttk.Label(frame, text="Outlier Threshold:")
        self.outlier_threshold_label.grid(row=16, column=0, padx=5, pady=5, sticky="e")
        self.outlier_threshold_label.grid_remove()

        self.outlier_threshold_entry = ttk.Entry(frame, width=10)
        self.outlier_threshold_entry.grid(row=16, column=1, padx=10, pady=5, sticky="w")
        self.outlier_threshold_entry.grid_remove()

        self.apply_noise_btn = ttk.Button(
            frame,
            text="Apply Noise Removal",
            command=lambda: apply_noise(self),
            bootstyle=SUCCESS,
        )
        self.apply_noise_btn.grid(row=17, column=0, columnspan=2, pady=5)
        self.apply_noise_btn.grid_remove()

        self.noise_method_menu.bind(
            "<<ComboboxSelected>>", self.on_noise_method_selected
        )

        self.avg_upload_btn = ttk.Button(
            frame,
            text="Add Image to Average",
            command=lambda: add_image_to_average(self),
            bootstyle=INFO,
        )
        self.avg_upload_btn.grid(row=17, column=0, columnspan=2, pady=5)
        self.avg_upload_btn.grid_remove()

        self.apply_image_avg_btn = ttk.Button(
            frame,
            text="Apply Averaging",
            command=lambda: apply_image_averaging(self),
            bootstyle=SUCCESS,
        )
        self.apply_image_avg_btn.grid(row=18, column=0, columnspan=2, pady=5)
        self.apply_image_avg_btn.grid_remove()

        # Threshold slider + apply button
        self.threshold_method = ttk.StringVar(value="Basic global thresholding")
        self.threshold_method_menu = ttk.Combobox(
            frame,
            values=[
                "Basic global thresholding",
                "Automatic thresholding",
                "Adaptive thresholding",
            ],
            textvariable=self.threshold_method,
            state="readonly",
        )
        self.threshold_method_menu.grid(row=19, column=0, columnspan=2, pady=5)
        self.threshold_method_menu.grid_remove()
        self.apply_threshold_btn = ttk.Button(
            frame,
            text="Apply Thresholding",
            command=lambda: apply_threshold(self),
            bootstyle=SUCCESS,
        )
        self.apply_threshold_btn.grid(row=21, column=0, columnspan=2, pady=5)
        self.apply_threshold_btn.grid_remove()
        self.threshold_value_label = ttk.Label(frame, text="Threshold Value:")
        self.threshold_value_label.grid(row=20, column=0, padx=5)
        self.threshold_value_label.grid_remove()

        self.threshold_value_entry = ttk.Entry(frame)
        self.threshold_value_entry.insert(0, "128")  # default value
        self.threshold_value_entry.grid(row=20, column=1, padx=10)
        self.threshold_value_entry.grid_remove()

    def create_color_operations_section(self):
        frame = ttk.LabelFrame(self, text="Color Operations", padding=10)
        frame.place(x=20, y=760, width=960, height=350)

        ttk.Label(frame, text="Color Operation:").grid(row=0, column=0, padx=5)
        color_ops = ["Change Color Intensity", "Swap Two Colors", "Eliminate Color"]
        color_menu = ttk.Combobox(
            frame,
            values=color_ops,
            textvariable=self.selected_color_op,
            state="readonly",
        )
        color_menu.grid(row=0, column=1, padx=10)
        color_menu.bind("<<ComboboxSelected>>", color_operation_selected)

        self.color_frame = ttk.LabelFrame(frame, text="Select Color", padding=10)
        self.color_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.swap_frame = ttk.LabelFrame(frame, text="Swap Colors", padding=10)
        self.swap_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.eliminate_frame = ttk.LabelFrame(
            frame, text="Eliminate Colors", padding=10
        )
        self.eliminate_frame.grid(row=3, column=0, columnspan=2, pady=10)

        self.color_radios = {
            "Red": ttk.Radiobutton(
                self.color_frame, text="Red", variable=self.selected_color, value="Red"
            ),
            "Green": ttk.Radiobutton(
                self.color_frame,
                text="Green",
                variable=self.selected_color,
                value="Green",
            ),
            "Blue": ttk.Radiobutton(
                self.color_frame,
                text="Blue",
                variable=self.selected_color,
                value="Blue",
            ),
        }

        col = 0
        for _, radio in self.color_radios.items():
            radio.grid(row=0, column=col, padx=10)
            col += 1

        self.color_slider = tk.Scale(
            frame, from_=-10, to=10, resolution=1, orient="horizontal", length=300
        )
        self.color_slider.grid(row=4, column=0, columnspan=2, pady=10)

        ttk.Label(self.swap_frame, text="From:").grid(row=0, column=0, padx=5)
        self.swap_from_menu = ttk.Combobox(
            self.swap_frame,
            values=["Red", "Green", "Blue"],
            textvariable=self.swap_from_color,
            state="readonly",
        )
        self.swap_from_menu.grid(row=0, column=1, padx=5)

        ttk.Label(self.swap_frame, text="To:").grid(row=0, column=2, padx=5)
        self.swap_to_menu = ttk.Combobox(
            self.swap_frame,
            values=["Red", "Green", "Blue"],
            textvariable=self.swap_to_color,
            state="readonly",
        )
        self.swap_to_menu.grid(row=0, column=3, padx=5)

        self.eliminate_red_cb = ttk.Checkbutton(
            self.eliminate_frame, text="Red", variable=self.eliminate_red
        )
        self.eliminate_green_cb = ttk.Checkbutton(
            self.eliminate_frame, text="Green", variable=self.eliminate_green
        )
        self.eliminate_blue_cb = ttk.Checkbutton(
            self.eliminate_frame, text="Blue", variable=self.eliminate_blue
        )

        self.eliminate_red_cb.grid(row=0, column=0, padx=10)
        self.eliminate_green_cb.grid(row=0, column=1, padx=10)
        self.eliminate_blue_cb.grid(row=0, column=2, padx=10)

        self.apply_color_op_btn = ttk.Button(
            frame,
            text="Apply Color Operation",
            command=lambda: apply_color_operation(self),
            bootstyle=SUCCESS,
        )
        self.apply_color_op_btn.grid(row=5, column=0, columnspan=2, pady=10)

        self.color_frame.grid_remove()
        self.color_slider.grid_remove()
        self.swap_frame.grid_remove()
        self.eliminate_frame.grid_remove()
        self.apply_color_op_btn.grid_remove()

    def undo_last_operation(self):
        if self.history_stack:
            last_img = self.history_stack.pop()
            self.img_pil = last_img.copy()
            display_filtered(self, last_img)
            Messagebox.ok("Undo", "Last operation undone.")
        else:
            Messagebox.ok("Undo", "No operations to undo.")

    def on_noise_method_selected(self, event=None):
        method = self.noise_method_menu.get()
        if method == "Outlier":
            self.outlier_threshold_label.grid()
            self.outlier_threshold_entry.grid()
        else:
            self.outlier_threshold_label.grid_remove()
            self.outlier_threshold_entry.grid_remove()

    def on_contrast_method_selected(self, event=None):
        method = self.contrast_method_menu.get()
        if method == "Gamma Correction":
            self.contrast_factor_label.grid(row=5, column=0, padx=5)
            self.contrast_factor_entry.grid(row=5, column=1, padx=10)
        else:
            self.contrast_factor_label.grid_remove()
            self.contrast_factor_entry.grid_remove()
        if method == "Matching":
            self.upload_second_img_btn.grid(row=6, column=0, columnspan=2, pady=5)
        else:
            self.upload_second_img_btn.grid_remove()
        # Always show the Apply Contrast button when a method is selected
        self.apply_contrast_btn.grid(row=7, column=0, columnspan=2, pady=5)


# --- Run the app ---
if __name__ == "__main__":
    app = ImageProcessingApp()

    def on_closing():
        print("Thank you for using the Image Processing Tool!")
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()
