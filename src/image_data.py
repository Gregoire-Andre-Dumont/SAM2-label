"""Module that manages the image and their corresponding segmentation masks."""
from tqdm import tqdm
import cv2
import os
import random

import numpy as np
import numpy.typing as npt
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ImageData:
    """Class to manage the images and the segmentation masks."""

    storage_path: str | None = None
    video_path: str | None = None

    def saved_masks(self) -> int:
        """Returns the number of segmentation masks created"""

        masks = Path(self.storage_path) / 'masks'
        return len(os.listdir(masks))

    def saved_images(self) -> int:
        """Returns the number of images created"""

        images = Path(self.storage_path) / 'images'
        return len(os.listdir(images))


    def current_image(self) -> npt.NDArray[np.uint8]:
        """Extract the current image to perform segmentation."""

        # Extract the paths to the images
        directory = Path(self.storage_path) / 'images'
        image_names = sorted(os.listdir(directory))

        # Initialize the path to the current image
        current_idx = self.saved_masks()
        image_path = directory / image_names[current_idx]

        # Extract and return the image using opencv
        return cv2.imread(image_path)


    def is_finished(self) -> bool:
        """Check whether all the images in the dataset are labeled."""

        image_path = Path(self.storage_path) / 'images'
        current_idx = self.saved_masks()
        return current_idx == len(os.listdir(image_path))


    def save_mask(self, mask: npt.NDArray[np.uint8]):
        """Save the segmentation mask of the current image."""

        # Extract the paths and names of the images
        directory = Path(self.storage_path) / 'images'
        image_names = sorted(os.listdir(directory))

        # Extract the name of the corresponding image
        current_idx = self.saved_masks()
        image_name = image_names[current_idx]

        # Save the mask with the same name
        mask_path = Path(self.storage_path) / "masks"
        cv2.imwrite(mask_path/image_name, mask)









