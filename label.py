import os.path as osp
import cv2
import torch
import numpy as np
from pathlib import Path

from src.video_data import VideoData
from src.ui.window import DisplayWindow
from src.ui.layout import HStack, VStack
from src.ui.buttons import ToggleButton, ImmediateButton
from src.ui.text import TitledTextBlock, ValueBlock
from src.ui.static import StaticMessageBar
from src.v2_sam.make_sam_v2 import make_samv2_from_original_state_dict as make_sam


from src.helpers.shared_ui_layout import PromptUIControl, PromptUI, ReusableBaseImage
from src.helpers.video_frame_select_ui import run_video_frame_select_ui
from src.helpers.contours import get_contours_from_mask
from src.helpers.history_keeper import HistoryKeeper
from src.helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from src.helpers.misc import PeriodicVRAMReport, get_default_device_string, make_device_config

#.......................................................................................................................
# Initialize SAM 2 and the video data

model_path = "tm/sam2_small.pt"
video_path = "videos/plastic_bag.mp4"
storage_path = "datasets/plastic_bag"


# Load the resources for SAM 2
device = get_default_device_string()
config_dict, model = make_sam(model_path)
model.to(**make_device_config(device, False))

# Check whether the storage directory was created
Path(storage_path).mkdir(parents=True, exist_ok=True)
video = VideoData(n_samples=20, storage_path=storage_path, video_path=video_path)


#-----------------------------------------------------------------------------------------------------------------------
# Setup the user interface

# Set up shared UI elements & control logic
ui_elems = PromptUI(video.current_image(), torch.zeros(1, 4, 256, 256))
uictrl = PromptUIControl(ui_elems)

# Set up message bars to communicate data info & controls
model_name = osp.basename(model_path)
header_msgbar = StaticMessageBar(model_name, f"64x64 tokens", device, space_equally=True)

# Controls for adding/finishing
num_prompts_textblock = TitledTextBlock("Stored Memories").set_text(0)
record_prompt_btn = ImmediateButton("Record Prompt", color=(135, 115, 35))
track_video_btn = ImmediateButton("Track Video", color=(80, 140, 20))


# Set up full display layout
horizontal =  HStack(record_prompt_btn, num_prompts_textblock, track_video_btn)
imgseg_layout = VStack(header_msgbar, ui_elems.layout, horizontal).set_debug_name("DisplayLayout")


# Render out an image with a target size, to figure out which side we should limit when rendering
display_image = imgseg_layout.render(h=900, w=900)
render_side = "h" if display_image.shape[1] > display_image.shape[0] else "w"
render_limit_dict = {render_side: 900}
min_display_size_px = imgseg_layout._rdr.limits.min_h if render_side == "h" else imgseg_layout._rdr.limits.min_w


#---------------------------------------------------------------------------------------------------------------------------
# Image Segmentation
window_title = "Image Segmentation - q to quit"


# Image segmentation loop
while not video.is_finished():

    # Initialize the image and the mask
    mask = torch.zeros(1, 4, 256, 256)
    idx, image = video.current_image()

    # Set up shared UI elements & control logic
    ui_elems = PromptUI(image, mask)
    uictrl = PromptUIControl(ui_elems)

    # Set up display
    cv2.destroyAllWindows()
    window = DisplayWindow(window_title, display_fps=60).attach_mouse_callbacks(imgseg_layout)
    window.move(200, 50)

    # Change tools/masks on arrow keys
    uictrl.attach_arrowkey_callbacks(window)

    # Keypress for clearing prompts
    window.attach_keypress_callback("c", ui_elems.tools.clear.click)

    # For clarity, some additional keypress codes
    KEY_ZOOM_IN = ord("=")
    KEY_ZOOM_OUT = ord("-")

    # Set up helper for managing display data
    base_img_maker = ReusableBaseImage(image)
    label_is_finished = False

    while not label_is_finished:
        # Read prompt input data & selected mask
        need_prompt_encode, prompts = uictrl.read_prompts()
        is_mask_changed, mselect_idx, selected_mask_btn = ui_elems.masks_constraint.read()


        # Record existing prompts into a single memory
        if record_prompt_btn.read():
            # Wipe out prompts, to indicate storage
            ui_elems.tools.clear.click()
            ui_elems.tools_constraint.change_to(ui_elems.tools.hover)

        # Close image segmentation to begin tracking
        if track_video_btn.read():
            label_is_finished = True
            cv2.destroyAllWindows()

        # Only run the model when an input affecting the output has changed!
        if need_prompt_encode:
            pass

        # Update mask previews & selected mask for outlines
        need_mask_update = any((need_prompt_encode, is_mask_changed))
        if need_mask_update:
            mask_preds = torch.zeros(1, 4, 256, 256)
            uictrl.update_mask_previews(mask_preds)

        # Process contour data
        final_mask_uint8 = np.zeros((1024, 1024), dtype=np.uint8)
        _, mask_contours_norm = get_contours_from_mask(final_mask_uint8, normalize=True)

        # Re-generate display image at required display size
        # -> Not strictly needed, but can avoid constant re-sizing of base image (helpful for large images)
        display_hw = ui_elems.image.get_render_hw()
        disp_img = base_img_maker.regenerate(display_hw)

        # Update the main display image in the UI
        uictrl.update_main_display_image(disp_img, final_mask_uint8, mask_contours_norm)

        # Render final output
        display_image = imgseg_layout.render(**render_limit_dict)
        req_break, keypress = window.show(display_image)
        if req_break:
            break

        # Scale display size up when pressing +/- keys
        if keypress == KEY_ZOOM_IN:
            display_size_px = min(display_size_px + 50, 10000)
            render_limit_dict = {render_side: display_size_px}

        if keypress == KEY_ZOOM_OUT:
            display_size_px = max(display_size_px - 50, min_display_size_px)
            render_limit_dict = {render_side: display_size_px}
