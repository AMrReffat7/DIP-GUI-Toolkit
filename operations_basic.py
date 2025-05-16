# operations_basic.py

from ttkbootstrap.dialogs import Messagebox
from PIL import Image
import numpy as np
from collections import Counter
from image_io import display_filtered
import pandas as pd
import cv2 as cv
from tkinter import messagebox


def apply_brightness(app):
    if not app.img_original_pil:
        return
    brightness_val = app.brightness_slider.get()
    np_img = np.array(
        app.img_original_pil.convert(
            "L" if app.image_type.get() == "Grayscale" else "RGB"
        )
    )
    factor = 1 + (brightness_val / 10)

    adjusted = np.clip(np_img * factor, 0, 255).astype(np.uint8)

    img_out = Image.fromarray(adjusted)
    app.img_pil = img_out  # update filtered image reference
    display_filtered(app, img_out)


def apply_contrast(app):
    if not app.img_pil:
        return

    method = app.contrast_method_menu.get()
    try:
        factor = float(app.contrast_factor_entry.get())
    except:
        factor = 1.0

    img = app.img_pil

    # Determine if image is RGB or L
    is_rgb = img.mode == "RGB"
    if is_rgb:
        channels = img.split()
        processed_channels = []
        for ch in channels:
            processed_channels.append(
                _apply_contrast_to_channel(ch, method, factor, app)
            )
        img_out = Image.merge("RGB", processed_channels)
        display_filtered(app, img_out)
    else:
        img_out = _apply_contrast_to_channel(img, method, factor, app)
        display_filtered(app, img_out)


def _apply_contrast_to_channel(img, method, factor, app):
    # img is a single channel (L)
    if method == "Gamma Correction":
        gamma = factor if factor > 0 else 1.0
        return img.point(lambda p: int(255 * ((p / 255) ** (1 / gamma))))
    elif method == "Equalization":
        histogram = img.histogram()
        lut = []
        total_pixels = sum(histogram)
        cumulative = 0
        for h in histogram:
            cumulative += h
            lut.append(round(255 * cumulative / total_pixels))
        return img.point(lut)
    elif method == "Stretching":
        np_img = np.array(img)
        min_val = np.min(np_img)
        max_val = np.max(np_img)
        if max_val - min_val == 0:
            stretched = np_img
        else:
            stretched = ((np_img - min_val) / (max_val - min_val) * 255).astype(
                np.uint8
            )
        return Image.fromarray(stretched)
    elif method == "Matching":
        if not app.second_image_path:
            Messagebox.ok("Error", "Please upload a second image for matching.")
            return img
        target_img = Image.open(app.second_image_path).convert("L").resize(img.size)
        source_array = np.array(img)
        target_array = np.array(target_img)
        source_hist, _ = np.histogram(source_array.flatten(), 256, [0, 256])
        target_hist, _ = np.histogram(target_array.flatten(), 256, [0, 256])
        source_cdf = np.cumsum(source_hist) / np.sum(source_hist)
        target_cdf = np.cumsum(target_hist) / np.sum(target_hist)
        mapping = np.zeros(256, dtype=np.uint8)
        for src_pixel in range(256):
            closest = np.abs(target_cdf - source_cdf[src_pixel]).argmin()
            mapping[src_pixel] = closest
        matched_array = mapping[source_array]
        return Image.fromarray(matched_array)
    else:
        Messagebox.ok("Error", "Please select a contrast method.")
        return img


def apply_inverse(app):
    if not app.img_pil:
        return
    img = app.img_pil.convert("RGB")
    pixels = img.load()
    is_partial = app.partial_inverse_check.instate(["selected"])
    try:
        threshold = int(app.inverse_threshold_entry.get())
    except:
        threshold = 125
    width, height = img.size
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            avg = (r + g + b) // 3
            if not is_partial or avg > threshold:
                pixels[x, y] = (255 - r, 255 - g, 255 - b)
    display_filtered(app, img)


def basic_operation_selected(event=None):
    app = event.widget.master.master  # Gets root window

    op = app.selected_basic_op.get()

    widgets = [
        app.point_op_label,
        app.point_op_menu,
        app.point_op_factor_label,
        app.point_op_factor_entry,
        app.point_op_apply_btn,
        app.brightness_slider,
        app.apply_brightness_btn,
        app.contrast_method_label,
        app.contrast_method_menu,
        app.contrast_factor_label,
        app.contrast_factor_entry,
        app.upload_second_img_btn,
        app.apply_contrast_btn,
        app.partial_inverse_check,
        app.inverse_threshold_label,
        app.inverse_threshold_entry,
        app.apply_inverse_btn,
        app.apply_blur_btn,
        app.noise_method_label,
        app.noise_method_menu,
        app.apply_noise_btn,
        app.blur_type_menu,
        app.apply_blur_btn,
        app.threshold_method_menu,
        app.threshold_value_label,
        app.threshold_value_entry,
        app.outlier_threshold_label,
        app.outlier_threshold_entry,
        app.apply_threshold_btn,
        app.sobel_mag_btn,
        app.avg_upload_btn,
        app.apply_image_avg_btn,
        app.laplacian_btn,
        app.kernel_size_label,
        app.kernel_size_entry_h,
        app.kernel_size_entry_w,
    ]

    for w in widgets:
        w.grid_remove()

    if op == "Point Operation":
        app.point_op_label.grid(row=27, column=0, padx=5, pady=5, sticky="e")
        app.point_op_menu.grid(row=27, column=1, padx=10, pady=5, sticky="w")

        def toggle_point_factor(*args):
            if app.point_op_type.get() == "Complement":
                app.point_op_factor_label.grid_remove()
                app.point_op_factor_entry.grid_remove()
            else:
                app.point_op_factor_label.grid(
                    row=28, column=0, padx=5, pady=5, sticky="e"
                )
                app.point_op_factor_entry.grid(
                    row=28, column=1, padx=10, pady=5, sticky="w"
                )

        app.point_op_menu.bind("<<ComboboxSelected>>", toggle_point_factor)
        toggle_point_factor()

        app.point_op_apply_btn.grid(row=29, column=0, columnspan=2, pady=5)

    elif op == "Brightness":
        app.brightness_slider.grid(row=1, column=0, columnspan=2, pady=5)
        app.apply_brightness_btn.grid(row=2, column=0, columnspan=2, pady=5)

    elif op == "Contrast":
        app.contrast_method_label.grid(row=4, column=0, padx=5)
        app.contrast_method_menu.grid(row=4, column=1, padx=10)

    elif op == "Inverse":
        app.partial_inverse_check.grid(row=8, column=0, columnspan=2, pady=5)
        app.inverse_threshold_label.grid(row=9, column=0, padx=5)
        app.inverse_threshold_entry.grid(row=9, column=1, padx=10)
        app.apply_inverse_btn.grid(row=10, column=0, columnspan=2, pady=5)

    elif op == "Blur":
        app.blur_type_menu.grid(row=11, column=1, columnspan=2, pady=5)
        app.kernel_size_label.grid(row=13, column=0, padx=5)
        app.kernel_size_entry_w.grid(row=13, column=1, padx=5)
        app.kernel_size_entry_h.grid(row=13, column=2, padx=5)
        app.apply_blur_btn.grid(row=14, column=0, columnspan=2, pady=5)

    elif op == "Edge Detection":
        app.sobel_mag_btn.grid(row=25, column=0, columnspan=2, pady=5)
        app.laplacian_btn.grid(row=26, column=0, columnspan=2, pady=5)

    elif op == "Noise":
        app.noise_method_label.grid(row=14, column=0, padx=5, pady=5)
        app.noise_method_menu.grid(row=14, column=1, padx=5, pady=5)
        app.apply_noise_btn.grid(row=17, column=0, columnspan=2, pady=5)
        app.kernel_size_label.grid(row=15, column=0, padx=5)
        app.kernel_size_entry_w.grid(row=15, column=1, padx=5, pady=5)
        app.kernel_size_entry_h.grid(row=15, column=2, padx=5, pady=5)

        def update_noise_widgets(*args):
            if app.noise_method_menu.get() == "Image averaging":
                app.avg_upload_btn.grid()
                app.apply_image_avg_btn.grid()

                app.outlier_threshold_label.grid_remove()
                app.outlier_threshold_entry.grid_remove()
                app.apply_noise_btn.grid_remove()  # Hide normal blur button

            elif app.noise_method_menu.get() == "Outlier":
                app.outlier_threshold_label.grid()
                app.outlier_threshold_entry.grid()

                app.apply_noise_btn.grid(row=17, column=0, columnspan=2, pady=5)

                app.avg_upload_btn.grid_remove()
                app.apply_image_avg_btn.grid_remove()

            else:
                app.outlier_threshold_label.grid_remove()
                app.outlier_threshold_entry.grid_remove()

                app.avg_upload_btn.grid_remove()
                app.apply_image_avg_btn.grid_remove()

                app.noise_method_label.grid(row=14, column=0, padx=5, pady=5)
                app.noise_method_menu.grid(row=14, column=1, padx=5, pady=5)
                app.apply_noise_btn.grid(row=16, column=0, columnspan=2, pady=5)

        app.noise_method_menu.bind("<<ComboboxSelected>>", update_noise_widgets)
        update_noise_widgets()

    elif op == "Segmentation":
        app.threshold_method_menu.grid(row=19, column=0, columnspan=2, pady=5)
        app.apply_threshold_btn.grid(row=21, column=0, columnspan=2, pady=5)

        def toggle_input(*args):
            if app.threshold_method.get() == "Basic global thresholding":
                app.threshold_value_label.grid(row=20, column=0)
                app.threshold_value_entry.grid(row=20, column=1)
            else:
                app.threshold_value_label.grid_remove()
                app.threshold_value_entry.grid_remove()

        app.threshold_method_menu.bind("<<ComboboxSelected>>", toggle_input)
        toggle_input()


def contrast_method_selected(event=None):
    app = event.widget.master.master
    app.apply_contrast_btn.grid(row=7, column=0, columnspan=2, pady=5)

    method = app.contrast_method_menu.get()
    if method == "Gamma Correction":
        app.contrast_factor_label.grid(row=5, column=0, padx=5)
        app.contrast_factor_entry.grid(row=5, column=1, padx=10)
    else:
        app.contrast_factor_label.grid_remove()
        app.contrast_factor_entry.grid_remove()
    if method == "Matching":
        app.upload_second_img_btn.grid(row=6, column=0, columnspan=2, pady=5)
    else:
        app.upload_second_img_btn.grid_remove()


def conv(src, kernel):
    if ((src.shape[0] - kernel.shape[0]) < 0) or ((src.shape[1] - kernel.shape[1]) < 0):
        print("Kernel shape error\n")
        return None

    output_image = np.zeros(
        (src.shape[0] - kernel.shape[0] + 1, src.shape[1] - kernel.shape[1] + 1)
    )
    for index, pixel in np.ndenumerate(src):
        if ((src.shape[0] - index[0]) < kernel.shape[0]) or (
            (src.shape[1] - index[1]) < kernel.shape[1]
        ):
            continue
        else:
            row_end = index[0] + kernel.shape[0] - 1
            col_end = index[1] + kernel.shape[1] - 1
            core_pixel = np.multiply(
                src[index[0] : row_end + 1, index[1] : col_end + 1], kernel
            ).sum()
            output_image[index] = core_pixel

    return output_image


def non_linear_filter(src, method, size=(3, 3), th=0.4):
    if ((src.shape[0] - size[0]) < 0) or ((src.shape[1] - size[1]) < 0):
        print("shape error\n")
        return None

    output_image = np.zeros((src.shape[0] - size[0] + 1, src.shape[1] - size[1] + 1))
    for index, pixel in np.ndenumerate(src):
        if ((src.shape[0] - index[0]) < size[0]) or (
            (src.shape[1] - index[1]) < size[1]
        ):
            continue
        else:
            kernel = src[index[0] : index[0] + size[0], index[1] : index[1] + size[1]]
            core_pixel = 0
            if str.lower(method) == "min":
                core_pixel = np.min(kernel)
            elif str.lower(method) == "max":
                core_pixel = np.max(kernel)
            elif str.lower(method) == "median":
                core_pixel = np.median(kernel)
            elif str.lower(method) == "most frequent":
                core_pixel = pd.value_counts(kernel.flatten()).index[0]
            elif str.lower(method) == "outlier":
                neighbors_mean = (np.sum(kernel) - src[index]) / (
                    (size[0] * size[1]) - 1
                )
                if abs(src[index] - neighbors_mean) > th:
                    core_pixel = neighbors_mean
                else:
                    core_pixel = src[index]
            output_image[index] = core_pixel

    return output_image


def apply_blur(app):
    if not app.img_pil:
        return

    img = app.img_pil
    method = app.blur_method.get().lower()
    try:
        k_w = int(app.kernel_size_entry_w.get())
        k_h = int(app.kernel_size_entry_h.get())
    except:
        k_w = k_h = 3

    if img.mode == "RGB":
        channels = img.split()
        blurred_channels = []
        for ch in channels:
            np_img = np.array(ch)
            if method == "average":
                kernel = np.ones((k_w, k_h)) / (k_w * k_h)
                result = conv(np_img, kernel)
            elif method == "median":
                result = non_linear_filter(np_img, "median", size=(k_w, k_h))
            elif method == "gaussian":
                kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
                result = conv(np_img, kernel)
            else:
                Messagebox.ok("Error", "Please select a valid blur method.")
                return
            if result is not None:
                filtered = Image.fromarray(np.uint8(np.clip(result, 0, 255)))
                blurred_channels.append(filtered)
            else:
                blurred_channels.append(ch)
        img_out = Image.merge("RGB", blurred_channels)
        display_filtered(app, img_out)
    else:
        np_img = np.array(img.convert("L"))
        if method == "average":
            kernel = np.ones((k_w, k_h)) / (k_w * k_h)
            result = conv(np_img, kernel)
        elif method == "median":
            result = non_linear_filter(np_img, "median", size=(k_w, k_h))
        elif method == "gaussian":
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
            result = conv(np_img, kernel)
        else:
            Messagebox.ok("Error", "Please select a valid blur method.")
            return
        if result is not None:
            filtered = Image.fromarray(np.uint8(np.clip(result, 0, 255)))
            display_filtered(app, filtered)


def apply_noise(app):
    if not app.img_pil:
        return

    method = app.noise_method_menu.get().lower()
    img = app.img_pil
    try:
        k_w = int(app.kernel_size_entry_w.get())
        k_h = int(app.kernel_size_entry_h.get())
    except:
        k_w = k_h = 3

    if img.mode == "RGB":
        channels = img.split()
        denoised_channels = []
        for ch in channels:
            np_img = np.array(ch)
            if method == "average":
                kernel = np.ones((k_w, k_h)) / (k_w * k_h)
                denoised = conv(np_img, kernel)
            elif method == "outlier":
                try:
                    th = float(app.outlier_threshold_entry.get())
                except:
                    th = 20
                denoised = non_linear_filter(np_img, method, size=(k_w, k_h), th=th)
            else:
                denoised = non_linear_filter(np_img, method, size=(k_w, k_h))
            if denoised is not None:
                img_filtered = Image.fromarray(np.uint8(denoised))
                denoised_channels.append(img_filtered)
            else:
                denoised_channels.append(ch)
        img_out = Image.merge("RGB", denoised_channels)
        display_filtered(app, img_out)
    else:
        np_img = np.array(img.convert("L"))
        if method == "average":
            kernel = np.ones((k_w, k_h)) / (k_w * k_h)
            denoised = conv(np_img, kernel)
        elif method == "outlier":
            try:
                th = float(app.outlier_threshold_entry.get())
            except:
                th = 20
            denoised = non_linear_filter(np_img, method, size=(k_w, k_h), th=th)
        else:
            denoised = non_linear_filter(np_img, method, size=(k_w, k_h))
        if denoised is not None:
            img_filtered = Image.fromarray(np.uint8(denoised))
            display_filtered(app, img_filtered)


def apply_sobel_magnitude(app):
    if not app.img_pil:
        return
    img = app.img_pil.convert("L")
    np_img = np.array(img)
    x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    x_filtered = conv(np_img, x_kernel)
    y_filtered = conv(np_img, y_kernel)
    magnitude = np.sqrt(np.square(x_filtered) + np.square(y_filtered))
    result = np.clip(magnitude, 0, 255).astype(np.uint8)

    img_out = Image.fromarray(result)
    display_filtered(app, img_out)


def apply_laplacian(app):
    if not app.img_pil:
        return

    img = app.img_pil.convert("L")
    np_img = np.array(img)

    # Laplacian kernel (you can switch to 8-neighbor version if needed)
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    result = conv(np_img, laplacian_kernel)

    if result is not None:
        result = np.clip(result, 0, 255).astype(np.uint8)
        img_out = Image.fromarray(result)
        display_filtered(app, img_out)


def apply_gaussian_blur(app):
    if not app.img_pil:
        return

    img = app.img_pil.convert("L")
    np_img = np.array(img)

    gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0

    blurred = conv(np_img, gaussian_kernel)

    if blurred is not None:

        img_blur = Image.fromarray(np.uint8(np.clip(blurred, 0, 255)))
        display_filtered(app, img_blur)


def basic_global_thresholding(img_np, thr):
    return (img_np > thr).astype(np.uint8) * 255


def automatic_thresholding(img_np):
    T = img_np.mean()
    while True:
        g1 = img_np[img_np <= T]
        g2 = img_np[img_np > T]
        new_T = 0.5 * (g1.mean() + g2.mean())
        if abs(T - new_T) < 1:
            break
        T = new_T
    return (img_np > T).astype(np.uint8) * 255


def adaptive_thresholding(img_np, block_size=15, c=5):
    pad = block_size // 2
    padded = np.pad(img_np, pad, mode="reflect")
    result = np.zeros_like(img_np)

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            block = padded[i : i + block_size, j : j + block_size]
            thresh = block.mean() - c
            result[i, j] = 255 if img_np[i, j] > thresh else 0

    return result.astype(np.uint8)


def apply_threshold(app):
    if not app.img_pil:
        return

    method = app.threshold_method.get()
    img_np = np.array(app.img_pil.convert("L"))

    if method == "Basic global thresholding":
        try:
            thr = int(app.threshold_value_entry.get())
        except:
            thr = 128
        result = (img_np > thr).astype(np.uint8) * 255  # ← this was missing

    elif method == "Automatic thresholding":
        result = automatic_thresholding(img_np)

    elif method == "Adaptive thresholding":
        result = adaptive_thresholding(img_np, block_size=15, c=5)

    else:
        from tkinter import messagebox

        messagebox.showerror("Error", f"Unsupported threshold method: {method}")
        return

    img_out = Image.fromarray(result)
    display_filtered(app, img_out)


def apply_point_operation(app, operation):
    if not app.img_pil:
        return

    try:
        factor = float(app.point_op_factor_entry.get())
    except:
        factor = 50  # default fallback

    img = app.img_pil.convert("L" if app.image_type.get() == "Grayscale" else "RGB")
    np_img = np.array(img).astype(np.float32)

    if operation == "Addition":
        result = np_img + factor
    elif operation == "Subtraction":
        result = np_img - factor
    elif operation == "Division":
        np_img = np.where(np_img == 0, 1, np_img)
        result = factor / np_img
    elif operation == "Complement":
        result = 255 - np_img
    else:
        return

    result = np.clip(result, 0, 255).astype(np.uint8)
    img_out = Image.fromarray(result)
    display_filtered(app, img_out)
