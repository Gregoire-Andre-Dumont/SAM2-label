import os.path as osp
import cv2
import torch

from src.image_data import ImageData
from src.ui.window import DisplayWindow
from src.ui.layout import HStack, VStack
from src.ui.static import StaticMessageBar
from src.ui.buttons import ImmediateButton
from src.ui.text import TitledTextBlock

from src.v2_sam.make_sam_v2 import make_samv2_from_original_state_dict as make_sam
from src.helpers.shared_ui_layout import PromptUIControl, PromptUI, ReusableBaseImage
from src.helpers.contours import get_contours_from_mask
from src.helpers.misc import get_default_device_string, make_device_config

#.......................................................................................................................
# Initialize SAM 2 and image data

model_path = "tm/sam_hiera_base.pt"
storage_path = "datasets/C1H0_CAM01"
display_size = 700

# Display the start of loading the model weights
print("", "Loading model weights...", sep="\n", flush=True)

# Load the resources for SAM 2
device = get_default_device_string()
config_dict, model = make_sam(model_path)

# Convert the model to mixed precision and initialize the image data
model.to(**make_device_config(device, False))
video = ImageData(storage_path=storage_path)


#-----------------------------------------------------------------------------------------------------------------------
# Set up the basic ui components

# Controls for adding/finishing
saved_masks_btn = TitledTextBlock("Saved Masks").set_text(0)
record_prompt_btn = ImmediateButton("Record Prompt", color=(135, 115, 35))
track_video_btn = ImmediateButton("Next Image", color=(80, 140, 20))

# Set up message bars to communicate data info & controls
model_name = osp.basename(model_path)
header_msgbar = StaticMessageBar(model_name, device, space_equally=True)
horizontal = HStack(record_prompt_btn, saved_masks_btn, track_video_btn)


#-----------------------------------------------------------------------------------------------------------------------
# Initialize the ui for the current image

while not video.is_finished():
    mask_preds = torch.zeros(1, 4, 256, 256)
    full_image = video.current_image()

    # Set up shared UI elements & control logic
    ui_elems = PromptUI(full_image, mask_preds)
    uictrl = PromptUIControl(ui_elems)

    # Set up the full display layout and render an image
    image_layout = VStack(header_msgbar, ui_elems.layout, horizontal)
    image_layout = image_layout.set_debug_name("DisplayLayout")
    display_image = image_layout.render(h=display_size, w=display_size)

    # Set up the opencv window with the title and initial position
    window = DisplayWindow("Image Segmentation - q to quit", display_fps=60)
    window = window.attach_mouse_callbacks(image_layout)
    window.move(200, 50)

    # Change tools on arrow keys and keypress for clearing prompts
    uictrl.attach_arrowkey_callbacks(window)
    window.attach_keypress_callback("c", ui_elems.tools.clear.click)

    # Set up shared image encoder settings to be cached
    image_encoder_dict = {"max_side_length": 1024, "use_square_sizing": False}
    encoded_img, _, initial_hw = model.encode_image(full_image, **image_encoder_dict)

    # Set up helper for managing display data
    base_img_maker = ReusableBaseImage(full_image)
    label_is_finished = False

    #-------------------------------------------------------------------------------------------------------------------
    # Perform Segment Anything on the current image

    # Read prompt input data & selected mask
    while not label_is_finished:
        need_prompt_encode, prompts = uictrl.read_prompts()
        is_mask_changed, mselect_idx, selected_mask_btn = ui_elems.masks_constraint.read()

        # Record existing prompts into a single memory
        if record_prompt_btn.read():
            ui_elems.tools.clear.click()
            ui_elems.tools_constraint.change_to(ui_elems.tools.hover)

        # Close image segmentation to begin tracking
        if track_video_btn.read():
            label_is_finished = True
            cv2.destroyAllWindows()

            # Display the number of saved masks
            n_masks = video.saved_masks()
            n_images = video.saved_images()
            saved_masks_btn.set_text(f"{n_masks}/{n_images}")


        # Only run the model when an input affecting the output has changed!
        if need_prompt_encode:
            encoded_prompts = model.encode_prompts(*prompts)
            mask_preds, iou_preds = model.generate_masks(
                encoded_img, encoded_prompts, mask_hint=None, blank_promptless_output=True)


        # Update mask previews & selected mask for outlines
        need_mask_update = any((need_prompt_encode, is_mask_changed))
        if need_mask_update:
            selected_mask_uint8 = uictrl.create_hires_mask_uint8(mask_preds, mselect_idx, initial_hw)
            uictrl.update_mask_previews(mask_preds)

        # Process contour data
        final_mask_uint8 = selected_mask_uint8
        _, mask_contours_norm = get_contours_from_mask(final_mask_uint8, normalize=True)

        # Re-generate display image at required display size
        # -> Not strictly needed, but can avoid constant re-sizing of base image (helpful for large images)
        display_hw = ui_elems.image.get_render_hw()
        disp_img = base_img_maker.regenerate(display_hw)

        # Update the main display image in the UI
        uictrl.update_main_display_image(disp_img, final_mask_uint8, mask_contours_norm)

        # Render final output
        display_image = image_layout.render(h=display_size)
        req_break, keypress = window.show(display_image)

    video.save_mask(final_mask_uint8)

