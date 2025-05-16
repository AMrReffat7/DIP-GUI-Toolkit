# operations_color.py
import numpy as np
import cv2 as cv
from PIL import Image
from image_io import display_filtered


def apply_color_operation(app):
    if not app.img_original_pil:
        return

    selected_op = app.selected_color_op.get()

    np_img = np.array(
        app.img_original_pil.convert("L" if app.image_type.get() == "Grayscale" else "RGB")
    )
    
    r, g, b = cv.split(np_img)

    if selected_op == "Change Color Intensity":
        color = app.selected_color.get()
        amount = app.color_slider.get()
        factor = 1 + (amount / 10)

        if color == "Red":
            r = np.clip(r * factor, 0, 255).astype(np.uint8)
        
        elif color == "Green":
            g = np.clip(g * factor, 0, 255).astype(np.uint8)
        
        elif color == "Blue":
            b = np.clip(b * factor, 0, 255).astype(np.uint8)
        

    elif selected_op == "Swap Two Colors":
        from_c = app.swap_from_color.get()
        to_c = app.swap_to_color.get()
        rgb = {"Red": r, "Green": g, "Blue": b}
        rgb[from_c], rgb[to_c] = rgb[to_c], rgb[from_c]
        r, g, b = rgb["Red"], rgb["Green"], rgb["Blue"]  # update variables

    elif selected_op == "Eliminate Color":
        if app.eliminate_red.get():
            r = np.zeros_like(r)
        if app.eliminate_green.get():
            g = np.zeros_like(g)
        if app.eliminate_blue.get():
            b = np.zeros_like(b)

    filtered = cv.merge([r, g, b])
    img_out = Image.fromarray(filtered)
    app.img_pil = img_out 
    display_filtered(app, img_out)


def color_operation_selected(event=None):
    app = event.widget.master.master  # Grabs root window (main app)

    selected_op = app.selected_color_op.get()

    # Hide all color frames first
    app.color_frame.grid_remove()
    app.color_slider.grid_remove()
    app.swap_frame.grid_remove()
    app.eliminate_frame.grid_remove()
    app.apply_color_op_btn.grid_remove()

    if selected_op == "Change Color Intensity":
        app.color_frame.grid()
        app.color_slider.grid()
        app.apply_color_op_btn.grid()

    elif selected_op == "Swap Two Colors":
        app.swap_frame.grid()
        app.apply_color_op_btn.grid()

    elif selected_op == "Eliminate Color":
        app.eliminate_frame.grid()
        app.apply_color_op_btn.grid()
