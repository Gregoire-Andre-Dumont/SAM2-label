"""Module that manages video frames adn their corresponding segmentation masks."""
import cv2
import os
import random

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

@dataclass
class VideoData:
    """Class to manage video frames and the segmentation masks."""

    # Number of images and masks
    n_samples : int | None = None

    # Path of the images and masks
    image_paths: list[Path] | None = None
    mask_paths: list[Path] | None = None

    # Paths to the dataset and the video
    storage_path: str | None = None
    video_path: str | None = None


    def __post_init__(self):
        """Extract and store random video frames if necessary."""

        # Create the storage of the frames and the masks
        image_path = Path(self.storage_path) / "images"
        masks_path = Path(self.storage_path) / "masks"

        # Check whether the folder are already created
        image_path.mkdir(parents=True, exist_ok=True)
        masks_path.mkdir(parents=True, exist_ok=True)


        # Check whether the images are already loaded
        if not os.listdir(image_path):

            # Load the video and initialize the frames
            video = cv2.VideoCapture(self.video_path)
            total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = random.sample(range(total), self.n_samples)

            # Extract and save the video frames
            for idx, frame_id in enumerate(indices):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = video.read()
                cv2.imwrite(image_path / f"{idx:05d}.jpg", frame)

        self.image_paths = [image_path / x for x in os.listdir(image_path)]
        self.mask_paths = [masks_path / x for x in os.listdir(masks_path)]


    def current_image(self) -> tuple[int, npt.NDArray[np.uint8]]:
        """Extract the current image to perform segmentation."""

        current_id = len(self.mask_paths)
        return current_id, cv2.imread(self.image_paths[current_id])

    def is_finished(self) -> bool:
        """Check whether all the images in the dataset are labeled."""
        return len(self.mask_paths) == len(self.image_paths)








