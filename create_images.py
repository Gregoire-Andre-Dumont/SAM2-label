from pathlib import Path
from tqdm import tqdm
import cv2
import os
import random

def load_images(n_samples: int, video_path: str, storage_path: str):
    """Extract and save randomly chosen video frames."""

    # Create the storage of the frames and the masks
    storage_path = Path(storage_path)
    image_path = storage_path / "images"
    masks_path = storage_path / "masks"

    image_path.mkdir(parents=True, exist_ok=True)
    masks_path.mkdir(parents=True, exist_ok=True)
    storage_path.mkdir(parents=True, exist_ok=True)

    # Load the video and initialize the frames
    video = cv2.VideoCapture(video_path)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = random.sample(range(total), n_samples)

    # Check whether the images are already loaded
    if not os.listdir(image_path):
        for idx, frame_id in tqdm(enumerate(indices), total=n_samples):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

            ret, frame = video.read()
            cv2.imwrite(image_path / f"{frame_id}.jpg", frame)

if __name__ == "__main__":
    n_samples = 20
    video_path = 'videos/plastic_bag.mp4'
    storage_path = 'datasets/plastic_bag'
    load_images(n_samples, video_path, storage_path)