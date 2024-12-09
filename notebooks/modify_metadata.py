import os
import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path



def update_labels(directory: str, csv_path: str):
    """Modifies the labels of the frames without plastic."""

    directory = Path(directory) / 'masks'

    # Load the metadata and keep track of the indices
    metadata = pd.read_csv(csv_path)
    images_to_remove = []

    # Iterate through the rows and check each mask
    for index, row in metadata.iterrows():
        image_path = directory / row['image_name']
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Check if the mask is empty (contains only zeros)
        if mask is not None and np.all(mask == 0):
            images_to_remove.append(index)
            metadata.at[index, 'class'] = 'none'

    return  metadata, images_to_remove


def remove_images(directory: str, metadata: pd.DataFrame, images_to_remove: list):
    """Perform under-sampling on the frames without plastics."""

    directory = Path(directory)

    # Check how many images with and without plastics
    no_plastic_count = len(images_to_remove)
    plastic_count = len(metadata) - no_plastic_count

    if no_plastic_count > plastic_count:
        n_samples = no_plastic_count - plastic_count
        indices = random.sample(images_to_remove, n_samples)

        # Remove the selected empty images and their masks
        for idx in indices:
            image_name = metadata.at[idx, 'image_name']
            image_path = directory / 'images' / image_name
            mask_path = directory / 'masks' / image_name

            # Remove the image and the mask
            os.remove(image_path)
            os.remove(mask_path)

        # Remove the corresponding rows from the CSV
        metadata = metadata.drop(indices)

    # Save the updated CSV to the output path
    metadata.to_csv(directory/'metadata.csv', index=False)


# Define paths
directory_path = "../datasets/training/masks"
csv_file_path = "../datasets/training/metadata.csv"

# Update the labels
metadata, images_to_remove = update_labels(directory_path, csv_file_path)
remove_images(directory_path, metadata, images_to_remove)