import os.path as osp

from src.ui.video import LoopingVideoReader, LoopingVideoPlaybackSlider, ValueChangeTracker
from src.ui.window import DisplayWindow
from src.ui.layout import HStack, VStack
from src.ui.buttons import ToggleButton, ImmediateButton
from src.ui.text import TitledTextBlock, ValueBlock
from src.ui.static import StaticMessageBar


from src.helpers.video_frame_select_ui import run_video_frame_select_ui
from src.helpers.contours import get_contours_from_mask
from src.helpers.video_data_storage import SAM2VideoObjectResults

from src.helpers.history_keeper import HistoryKeeper
from src.helpers.loading import ask_for_path_if_missing, ask_for_model_path_if_missing
from src.helpers.misc import PeriodicVRAMReport, get_default_device_string, make_device_config

#.......................................................................................................................
# Setup the default arguments
device = get_default_device_string()
device_config_dict = make_device_config(device, False)

# Create history to re-use selected inputs
root_path = osp.dirname(osp.dirname(__file__))
history = HistoryKeeper(root_path)
_, history_imgpath = history.read("image_path")
_, history_vidpath = history.read("video_path")
_, history_modelpath = history.read("model_path")

# Get pathing to resources, if not provided already
image_path = ask_for_path_if_missing("image", history_imgpath)
video_path = ask_for_path_if_missing( "video", history_vidpath)
model_path = ask_for_model_path_if_missing(root_path, history_modelpath)

# Store history for use on reload
history.store(image_path=image_path, video_path=video_path, model_path=model_path)

