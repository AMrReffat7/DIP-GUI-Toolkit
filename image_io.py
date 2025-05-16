# image_io.py

from tkinter import filedialog
from ttkbootstrap.dialogs import Messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import os

TEMP_HISTOGRAM_PATH = "temp_hist.png"


def upload_image(app):
    filetypes = (("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*"))
    file = filedialog.askopenfilename(title="Open Image", filetypes=filetypes)
    if not file:
        Messagebox.ok("No File Selected", "Please select an image to upload.")
        return
    if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        Messagebox.ok("Invalid File", "Please select a valid image file.")
        return
    app.image_path.set(file)
    load_image(app, file)


def upload_second_image(app):
    filetypes = (("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*"))
    file = filedialog.askopenfilename(title="Open Second Image", filetypes=filetypes)
    if file:
        app.second_image_path = file
        print(f"Second image loaded: {file}")


def load_image(app, filepath):
    try:
        img = Image.open(filepath)

        # Convert to RGB or Grayscale
        if app.image_type.get() == "Grayscale":
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        img.thumbnail((220, 220))
        app.img_pil = img
        app.img_original_pil = img.copy()
        app.original_image = ImageTk.PhotoImage(img)
        app.canvas_original.create_image(110, 110, image=app.original_image)
        show_histogram(app, img, app.canvas_hist_original)
    except Exception as e:
        Messagebox.ok("Error", f"Failed to load image.\n{str(e)}")


def show_histogram(app, img, canvas_widget):
    if app.histogram_visible.get() == 0:
        canvas_widget.delete("all")
        return

    plt.figure(figsize=(2, 2))
    plt.hist(img.convert("L").getdata(), bins=256, range=(0, 256), color="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(TEMP_HISTOGRAM_PATH)
    plt.close()

    hist_img = Image.open(TEMP_HISTOGRAM_PATH)
    hist_img = hist_img.resize((220, 220))
    hist_tk = ImageTk.PhotoImage(hist_img)
    canvas_widget.create_image(110, 110, image=hist_tk)
    canvas_widget.image = hist_tk

    if os.path.exists(TEMP_HISTOGRAM_PATH):
        os.remove(TEMP_HISTOGRAM_PATH)


def reset_image(app):
    if app.img_original_pil:
        app.img_pil = app.img_original_pil.copy()
        display_filtered(app, app.img_pil)
        Messagebox.ok("Reset", "Image has been reset to original.")


def save_filtered_image(app):
    if not hasattr(app, "filtered_image") or app.filtered_image is None:
        Messagebox.ok("Error", "No filtered image to save!")
        return

    filepath = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("All files", "*.*"),
        ],
    )
    if filepath:
        app.img_pil.save(filepath)
        Messagebox.ok("Success", "Image saved successfully!")


def save_histogram_image(app):
    if not app.img_pil:
        Messagebox.ok("Error", "No image loaded to generate histogram!")
        return

    filepath = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("All files", "*.*"),
        ],
    )
    if filepath:
        img_to_use = app.img_pil.convert("L")
        plt.figure(figsize=(4, 4))
        plt.hist(img_to_use.getdata(), bins=256, range=(0, 256), color="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        Messagebox.ok("Success", "Histogram saved successfully!")


def display_filtered(app, img_result):
    if app.img_pil:
        app.history_stack.append(app.img_pil.copy())

    app.img_pil = img_result.copy()
    app.filtered_image = ImageTk.PhotoImage(img_result)
    app.canvas_filtered.create_image(110, 110, image=app.filtered_image)

    show_histogram(app, img_result, app.canvas_hist_filtered)


def add_image_to_average(app):
    filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp")]
    files = filedialog.askopenfilenames(title="Select Images", filetypes=filetypes)

    if not hasattr(app, "average_imgs"):
        app.average_imgs = []

    if files:
        app.average_imgs.extend(files)
        Messagebox.ok("Images Added", f"{len(files)} image(s) added for averaging.")


def apply_image_averaging(app):
    if not app.average_imgs:
        Messagebox.ok("Error", "No images selected for averaging.")
        return

    try:
        images = []
        sizes = []

        for path in app.average_imgs:
            with Image.open(path).convert("L") as img:
                img_arr = np.array(img.copy(), dtype=np.float32)
                images.append(img_arr)
                sizes.append(img.size)

        # Ensure all sizes match
        if len(set(sizes)) != 1:
            Messagebox.ok("Error", "All images must be the same size.")
            return

        # Compute average
        avg_img = np.mean(np.stack(images), axis=0).astype(np.uint8)
        result_img = Image.fromarray(avg_img)

        app.average_imgs.clear()
        display_filtered(app, result_img)
        Messagebox.ok("Success", "Image averaging completed.")

    except Exception as e:
        Messagebox.ok("Error", f"Failed to average images.\n{str(e)}")
