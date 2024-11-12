import cv2
from src.ui.window import DisplayWindow
from src.ui.layout import HStack, VStack
window_title = "Image Segmentation - q to quit"


def setup_display(display_size):
    """Set up full display layout and render out the image."""

    global ui_elems, uictrl, header_msgbar
    global record_prompt_btn, saved_masks_btn, track_video_btn
    print(record_prompt_btn)

    # Set up full display layout
    horizontal = HStack(record_prompt_btn, saved_masks_btn, track_video_btn)
    image_layout = VStack(header_msgbar, ui_elems.layout, horizontal).set_debug_name("DisplayLayout")

    # Render out an image with a target size
    display_image = image_layout.render(h=display_size, w=display_size)
    min_display = image_layout._rdr.limits.min_h

    return display_image, min_display, image_layout


def setup_window(ui_elems, image_layout):
    """Set up an opencv window with the necessary controls."""

    # Set up the opencv window
    window = DisplayWindow(window_title, display_fps=60)
    window = window.attach_mouse_callbacks(image_layout)
    window.move(200, 50)

    # Change tools on arrow keys and keypress for clearing prompts
    uictrl.attach_arrowkey_callbacks(window)
    window.attach_keypress_callback("c", ui_elems.tools.clear.click)

    return window
